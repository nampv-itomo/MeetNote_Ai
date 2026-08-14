"""
audio_utils.py
--------------
Các hàm xử lý audio: cắt file dài thành nhiều đoạn nhỏ để model AI
xử lý hiệu quả hơn và tránh tràn bộ nhớ với file quá lớn.

Sử dụng ffmpeg (không phụ thuộc pydub) để tương thích Python 3.13+.
"""

import math
import os
import subprocess
import tempfile


def split_audio(audio_path: str, temp_dir: str, segment_minutes: int = 10) -> list[str]:
    """
    Cắt file audio thành nhiều đoạn nhỏ bằng ffmpeg.

    Args:
        audio_path: đường dẫn file audio gốc.
        temp_dir: thư mục để lưu các đoạn đã cắt.
        segment_minutes: độ dài mỗi đoạn (phút).

    Returns:
        Danh sách đường dẫn tới các file đoạn nhỏ, theo đúng thứ tự thời gian.
    """
    os.makedirs(temp_dir, exist_ok=True)

    # Lấy duration của file audio bằng ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr[:200]}")
        duration_sec = float(result.stdout.strip())
    except FileNotFoundError:
        raise RuntimeError("ffprobe không được tìm thấy. Cài đặt: sudo apt install ffmpeg")

    total_ms = int(duration_sec * 1000)
    segment_ms = segment_minutes * 60 * 1000
    num_segments = math.ceil(total_ms / segment_ms) if total_ms > 0 else 1

    segment_paths = []
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    for i in range(num_segments):
        start = i * segment_ms / 1000  # convert to seconds for ffmpeg
        if i == num_segments - 1:
            # Last segment: take the rest
            end = None
        else:
            end = segment_ms / 1000

        chunk_path = os.path.join(temp_dir, f"{base_name}_part{i:03d}.wav")

        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-ss", str(start),
        ]
        if end:
            cmd.extend(["-t", str(end)])
        cmd.extend([
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            chunk_path,
        ])

        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)
        segment_paths.append(chunk_path)

    return segment_paths


def cleanup_files(paths: list[str]) -> None:
    """Xoá các file tạm sau khi xử lý xong."""
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def cleanup_dir(dir_path: str) -> None:
    """Xoá toàn bộ thư mục tạm."""
    if os.path.exists(dir_path):
        for f in os.listdir(dir_path):
            try:
                os.remove(os.path.join(dir_path, f))
            except OSError:
                pass
        try:
            os.rmdir(dir_path)
        except OSError:
            pass
