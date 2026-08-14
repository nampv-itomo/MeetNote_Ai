"""
app.py
------
Web app self-hosted để ghi biên bản họp từ file audio, chạy 100% local:
  - Qwen3-ASR 1.7B / 0.6B (transcribe_qwen3.py / transcribe_qwen3_fast.py)
    -> chuyển giọng nói thành văn bản
  - llama.cpp server (summarize.py) -> tóm tắt thành biên bản họp
  - ffmpeg (audio_utils.py)  -> cắt file audio dài thành từng đoạn nhỏ

Chạy:
    python app.py
Sau đó mở trình duyệt: http://localhost:5001
"""

import os
import shutil
import threading
import uuid

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from audio_utils import cleanup_files, split_audio
from load_env import QWEN3_ASR_MODEL_17B, QWEN3_ASR_MODEL_06B, LLAMA_BASE_URL, LLAMA_MODEL, UPLOAD_FOLDER, TEMP_SEGMENTS, RESULTS_FOLDER
from summarize import summarize_transcript
from transcribe_qwen3 import transcribe_segments as transcribe_qwen3
from transcribe_qwen3_fast import transcribe_segments as transcribe_qwen3_fast
from transcribe_gemma import transcribe_segments as transcribe_gemma
from tts import list_voices, synthesize_job, get_job_status, get_job_output

ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4", "ogg", "flac", "webm"}
SEGMENT_MINUTES = 10
WHISPER_LANGUAGE = "vi"
TEMP_FOLDER = os.path.join(TEMP_SEGMENTS)
RESULTS_FOLDER = RESULTS_FOLDER
OLLAMA_MODEL = LLAMA_MODEL

for folder in (UPLOAD_FOLDER, TEMP_FOLDER, RESULTS_FOLDER):
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # giới hạn 2GB / file

# Lưu trạng thái các job trong bộ nhớ. Với nhu cầu production nhiều người dùng
# đồng thời, nên thay bằng Redis/SQLite thay vì dict trong RAM.
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def update_job(job_id: str, **kwargs) -> None:
    with jobs_lock:
        jobs[job_id].update(kwargs)


def process_audio_file(file_path: str, job_id: str, transcribe_engine: str = "whisper") -> None:
    """Pipeline chạy nền: cắt audio -> transcribe từng đoạn -> tóm tắt."""
    segment_dir = os.path.join(TEMP_FOLDER, job_id)
    segment_paths = []

    try:
        update_job(job_id, status="splitting", message="Đang chia nhỏ file audio...")
        segment_paths = split_audio(file_path, segment_dir, segment_minutes=SEGMENT_MINUTES)

        total = len(segment_paths)
        engine_label = "Qwen3-ASR 1.7B" if transcribe_engine == "qwen3" else (
            "Qwen3-ASR 0.6B" if transcribe_engine == "qwen3-fast" else "Gemma 4 E2B"
        )
        update_job(job_id, status="transcribing",
                   message=f"Đang chuyển giọng nói thành văn bản bằng {engine_label} (0/{total})...")

        def on_progress(current, total_segments):
            update_job(
                job_id,
                message=f"Đang chuyển giọng nói thành văn bản bằng {engine_label} ({current}/{total_segments})...",
            )

        if transcribe_engine == "gemma":
            transcript = transcribe_gemma(
                segment_paths, language=WHISPER_LANGUAGE, progress_callback=on_progress
            )
        elif transcribe_engine == "qwen3-fast":
            transcript = transcribe_qwen3_fast(
                segment_paths, language=WHISPER_LANGUAGE, progress_callback=on_progress
            )
        elif transcribe_engine == "qwen3":
            transcript = transcribe_qwen3(
                segment_paths, language=WHISPER_LANGUAGE, progress_callback=on_progress
            )
        else:
            transcript = transcribe_gemma(
                segment_paths, language=WHISPER_LANGUAGE, progress_callback=on_progress
            )

        update_job(job_id, status="summarizing", message="Đang tóm tắt thành biên bản họp...")
        with jobs_lock:
            context = jobs[job_id].get("context", "")
        minutes = summarize_transcript(transcript, context=context, model=OLLAMA_MODEL)

        # Lưu kết quả ra file để tiện tải về / xem lại sau
        result_path = os.path.join(RESULTS_FOLDER, f"{job_id}.txt")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write("=== BIÊN BẢN HỌP ===\n\n")
            f.write(minutes)
            f.write("\n\n=== TRANSCRIPT ĐẦY ĐỦ ===\n\n")
            f.write(transcript)

        update_job(
            job_id,
            status="completed",
            message="Hoàn tất!",
            transcript=transcript,
            minutes=minutes,
        )

    except Exception as e:
        update_job(job_id, status="error", message=f"Lỗi: {e}")

    finally:
        # Dọn dẹp file tạm dù thành công hay lỗi
        cleanup_files(segment_paths)
        shutil.rmtree(segment_dir, ignore_errors=True)
        try:
            os.remove(file_path)
        except OSError:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/audio-to-text")
def audio_to_text():
    return render_template("audio_to_text.html")


@app.route("/text-to-audio")
def text_to_audio():
    return render_template("text_to_audio.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy file trong request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Định dạng file không được hỗ trợ. "
                                  f"Hỗ trợ: {', '.join(sorted(ALLOWED_EXTENSIONS))}"}), 400

    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    stored_name = f"{job_id}_{filename}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    file.save(file_path)

    # Lấy engine transcribe và context từ form data
    transcribe_engine = request.form.get("engine", "qwen3")
    if transcribe_engine not in ("qwen3", "qwen3-fast", "gemma"):
        transcribe_engine = "qwen3"
    meet_context = request.form.get("context", "")

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "message": "Đang chờ xử lý...",
            "filename": filename,
            "context": meet_context,
        }

    thread = threading.Thread(target=process_audio_file, args=(file_path, job_id, transcribe_engine))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id, "status": "started"})


@app.route("/status/<job_id>")
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        return jsonify({"error": "Không tìm thấy job"}), 404

    # Không cần trả transcript/minutes đầy đủ trong lúc polling để tiết kiệm băng thông
    response = {"status": job["status"], "message": job["message"]}
    if job["status"] == "completed":
        response["transcript"] = job["transcript"]
        response["minutes"] = job["minutes"]

    return jsonify(response)


# ─── TTS Routes ──────────────────────────────────────────────────

@app.route("/api/tts/voices")
def tts_voices():
    """Trả về danh sách giọng nói có sẵn."""
    try:
        voices = list_voices()
        return jsonify({"voices": voices})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts/synthesize", methods=["POST"])
def tts_synthesize():
    """Bắt đầu job synthesize, trả về job_id để frontend polling.

    Body JSON:
      {
        "text": "Xin chào...",
        "voice": "Xuân Vĩnh",
        "style": "tu_nhien"
      }
    """
    data = request.get_json()
    if not data or not data.get("text"):
        return jsonify({"error": "Thiếu nội dung text"}), 400

    text = data["text"]
    voice = data.get("voice", "Xuân Vĩnh")
    style = data.get("style", "tu_nhien")

    try:
        job_id = synthesize_job(text, voice=voice, style=style)
        return jsonify({"job_id": job_id, "status": "started"})
    except Exception as e:
        return jsonify({"error": f"Lỗi bắt đầu job: {e}"}), 500


@app.route("/api/tts/status/<job_id>")
def tts_status(job_id):
    """Trả về progress của job synthesize."""
    status = get_job_status(job_id)
    if "error" in status:
        return jsonify(status), 404
    return jsonify(status)


@app.route("/api/tts/output/<job_id>")
def tts_output(job_id):
    """Trả về đường dẫn file audio khi job hoàn tất."""
    result = get_job_output(job_id)
    if "error" in result:
        return jsonify(result), 404
    return jsonify({
        "output_path": result["output_path"],
        "filename": result["filename"],
    })


@app.route("/api/tts/download/<job_id>")
def tts_download(job_id):
    """Trả về file audio để download."""
    result = get_job_output(job_id)
    if "error" in result:
        return jsonify(result), 404
    return send_file(
        result["output_path"],
        mimetype="audio/wav",
        as_attachment=True,
        download_name="meetnote_tts.wav",
    )


if __name__ == "__main__":
    print("Meeting Note Taker đang chạy tại http://localhost:5000")
    app.run(host="0.0.0.0", port=5001, debug=True)
