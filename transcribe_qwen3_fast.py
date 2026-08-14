"""
transcribe_qwen3_fast.py
------------------------
Dùng Qwen3-ASR-0.6B (CrispASR + Vulkan) để chuyển audio thành văn bản tiếng Việt.

Model nhẹ hơn (602MB), tốc độ nhanh hơn 1.7B, độ chính xác vẫn tốt cho tiếng Việt.
Chạy trên AMD GPU (Vulkan).

Model path: /run/media/viper/Data/models/qwen3-asr-0.6b/qwen3-asr-0.6b-q4_k.gguf
Wrapper: ~/bin/qwen3-asr-stt <audio_file> vi
"""

import os
import subprocess
import tempfile


# ============================================================
# CẤU HÌNH
# ============================================================
QWEN3_WRAPPER = "/home/viper/bin/qwen3-asr-stt"
from load_env import QWEN3_ASR_MODEL_06B
QWEN3_MODEL = QWEN3_ASR_MODEL_06B
# Legacy path (fallback)
QWEN3_MODEL_FALLBACK = "/run/media/viper/Data/models/qwen3-asr-0.6b/qwen3-asr-0.6b-q4_k.gguf"
LANG = "vi"
TIMEOUT = 300  # giây


# ============================================================
# CONVERT AUDIO -> WAV 16kHz MONO (ffmpeg)
# ============================================================
def convert_to_wav(audio_path: str) -> str:
    """
    Convert audio file to WAV 16kHz mono — định dạng Qwen3-ASR yêu cầu.
    Trả về đường dẫn file WAV mới tạo (tạm thời).
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ar", "16000",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                wav_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed: {result.stderr[:500]}"
            )
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            raise RuntimeError("ffmpeg output is empty")
        return wav_path
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg không được tìm thấy. Cài đặt: sudo apt install ffmpeg"
        )
    except Exception as e:
        try:
            os.remove(wav_path)
        except OSError:
            pass
        raise


# ============================================================
# TRANSCRIBE 1 FILE AUDIO
# ============================================================
def transcribe_file(audio_path: str) -> str:
    """
    Chuyển 1 file audio thành văn bản bằng Qwen3-ASR-0.6B.
    Tự động convert M4A/Mp3 -> WAV nếu cần.
    """
    if not os.path.exists(QWEN3_MODEL):
        raise RuntimeError(f"Không tìm thấy model Qwen3-ASR-0.6B tại {QWEN3_MODEL}")

    if not os.path.exists(QWEN3_WRAPPER):
        raise RuntimeError(f"Không tìm thấy wrapper tại {QWEN3_WRAPPER}")

    need_convert = False
    wav_temp = None

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in (".wav",):
        need_convert = True
        wav_temp = convert_to_wav(audio_path)
        transcribe_path = wav_temp
    else:
        transcribe_path = audio_path

    try:
        result = subprocess.run(
            [QWEN3_WRAPPER, transcribe_path, LANG],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Qwen3-ASR failed (code {result.returncode}): {result.stderr[:500]}")
        text = result.stdout.strip()
        # CrispASR v0.8+ output "." for empty audio — treat as no speech
        if text == ".":
            text = ""
        return text
    finally:
        if wav_temp:
            try:
                os.remove(wav_temp)
            except OSError:
                pass


# ============================================================
# TRANSCRIBE NHIỀU ĐOẠN VÀ GHÉP LẠI
# ============================================================
def transcribe_segments(segment_paths: list[str], language: str | None = None,
                        progress_callback=None) -> str:
    """
    Transcribe nhiều đoạn audio liên tiếp bằng Qwen3-ASR-0.6B và ghép thành văn bản.

    Args:
        segment_paths: danh sách đường dẫn các đoạn audio (đúng thứ tự).
        language: mã ngôn ngữ.
        model: tên model Ollama.
        progress_callback: hàm callback(current_index, total) để báo tiến độ.

    Returns:
        Văn bản transcript đầy đủ. Trả về chuỗi rỗng nếu không có đoạn nào transcribe được.
    """
    full_text_parts = []
    total = len(segment_paths)
    any_success = False

    for idx, path in enumerate(segment_paths, start=1):
        try:
            text = transcribe_file(path)
            if text:
                full_text_parts.append(text)
                any_success = True
            else:
                full_text_parts.append("(Không có nội dung speech trong đoạn này)")
        except Exception as e:
            full_text_parts.append(f"[Lỗi transcribe đoạn {idx}: {e}]")

        if progress_callback:
            progress_callback(idx, total)

    if not any_success:
        return ""
    return "\n".join(full_text_parts)
