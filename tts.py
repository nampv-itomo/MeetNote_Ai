from load_env import *
"""
tts.py
-----
Text-to-Speech bằng VieNeu-TTS v3 Turbo (chạy CPU/ONNX).

Sử dụng SDK `vieneu`:
  pip install vieneu

Model tự động tải lần đầu (~100MB, lưu cache ~/.cache/vieneu/).
Chạy hoàn toàn offline sau đó.

Job tracking: mỗi lần synthesize chạy trong background thread,
job_id được lưu vào global dict để frontend polling progress real-time.
"""

import os
import time
import uuid
import threading

import numpy as np

from vieneu import Vieneu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_OUTPUT_FOLDER = os.path.join(BASE_DIR, "tts_output")
os.makedirs(TTS_OUTPUT_FOLDER, exist_ok=True)

# Khởi tạo một lần (singleton)
_tts_instance: Vieneu | None = None

# Job tracking dict: job_id -> {status, message, progress, output_path, error}
_tts_jobs: dict[str, dict] = {}
_tts_lock = threading.Lock()


def get_tts(precision: str = "fp32") -> Vieneu:
    """Lazy-load VieNeu instance (singleton)."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = Vieneu(precision=precision)
    return _tts_instance


def list_voices() -> list[dict]:
    """Trả về danh sách giọng nói có sẵn.

    Returns:
        [{"id": "Trúc Ly", "name": "Nữ (miền Nam)"}, ...]
    """
    tts = get_tts()
    voices = tts.list_preset_voices()
    return [{"id": vid, "name": label} for label, vid in voices]


def _synthesis_worker(text: str, voice: str, style: str, job_id: str):
    """Background thread để synthesize text → audio với progress tracking real-time.

    Sử dụng infer_stream() — generator trả về từng chunk audio,
    mỗi chunk đến là update progress bar.

    Progress pipeline:
      0%  → Đang tải model TTS...
      5%  → Đang chuẩn bị text...
      10% → Bắt đầu tạo giọng nói (chunk 0)...
      ... → Đang tạo giọng nói (chunk N/M)...
      ~100% → Đang nối các đoạn audio...
      100% → ✅ Hoàn tất!
    """
    def set_progress(progress: int, message: str):
        with _tts_lock:
            if job_id in _tts_jobs:
                _tts_jobs[job_id]["progress"] = progress
                _tts_jobs[job_id]["message"] = message

    try:
        # Phase 1: Load model (0-5%)
        set_progress(0, "⏳ Đang tải model TTS...")
        time.sleep(0.2)
        tts = get_tts()

        # Phase 2: Bắt đầu stream (5%)
        set_progress(5, "🎙️ Đang chuẩn bị text...")
        time.sleep(0.1)

        # Phase 3: Stream chunks (5-95%)
        stream = tts.infer_stream(
            text=text,
            voice=voice,
            style=style,
            max_chars=300,
            apply_watermark=False,
        )

        all_chunks = []
        chunk_idx = 0

        for chunk in stream:
            all_chunks.append(chunk)
            chunk_idx += 1
            # Progress: 5% + (chunk_idx / estimated_total) * 90%
            # Ước tính tổng chunk từ độ dài text (300 ký tự/chunk)
            estimated_total = max(1, len(text.strip()) // 300)
            progress = min(95, int(5 + (chunk_idx / estimated_total) * 90))
            set_progress(progress, f"🎙️ Đang tạo giọng nói ({chunk_idx})...")

        # Phase 4: Nối audio (95-98%)
        if len(all_chunks) > 1:
            set_progress(96, "🔗 Đang nối các đoạn audio...")
            combined = np.concatenate(all_chunks)
        else:
            combined = all_chunks[0]

        # Phase 5: Lưu file (98-100%)
        set_progress(98, "💾 Đang lưu file audio...")
        output_filename = f"tts_{uuid.uuid4().hex[:12]}.wav"
        output_path = os.path.join(TTS_OUTPUT_FOLDER, output_filename)
        tts.save(combined, output_path)

        set_progress(100, "✅ Hoàn tất!")

        with _tts_lock:
            if job_id in _tts_jobs:
                _tts_jobs[job_id]["status"] = "completed"
                _tts_jobs[job_id]["output_path"] = output_path

    except Exception as e:
        set_progress(0, f"❌ Lỗi: {e}")
        with _tts_lock:
            if job_id in _tts_jobs:
                _tts_jobs[job_id]["status"] = "error"
                _tts_jobs[job_id]["error"] = str(e)

    # Cleanup job sau 30s
    time.sleep(30)
    with _tts_lock:
        _tts_jobs.pop(job_id, None)


def synthesize_job(text: str, voice: str = "Xuân Vĩnh", style: str = "tu_nhien") -> str:
    """Bắt đầu synthesize trong background thread, trả về job_id.

    Frontend polling job status qua /api/tts/status/<job_id>.
    """
    job_id = str(uuid.uuid4())[:12]
    with _tts_lock:
        _tts_jobs[job_id] = {
            "status": "queued",
            "message": "Đang chờ xử lý...",
            "progress": 0,
            "output_path": None,
            "error": None,
        }
    thread = threading.Thread(
        target=_synthesis_worker,
        args=(text, voice, style, job_id),
    )
    thread.daemon = True
    thread.start()
    return job_id


def get_job_status(job_id: str) -> dict:
    """Kiểm tra status của job synthesize."""
    with _tts_lock:
        job = _tts_jobs.get(job_id)
    if job is None:
        return {"error": "Không tìm thấy job"}
    return {
        "status": job["status"],
        "message": job["message"],
        "progress": job["progress"],
    }


def get_job_output(job_id: str) -> dict:
    """Lấy output path của job (chỉ gọi khi status == completed)."""
    with _tts_lock:
        job = _tts_jobs.get(job_id)
    if job is None:
        return {"error": "Không tìm thấy job"}
    if job["status"] != "completed":
        return {"error": "Job chưa hoàn tất"}
    return {
        "output_path": job["output_path"],
        "filename": os.path.basename(job["output_path"]),
    }
