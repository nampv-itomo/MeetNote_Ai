"""
zimage.py
---------
Text-to-Image bằng Z-Image-Turbo GGUF qua stable-diffusion.cpp (sd-cli, Vulkan).

Model:
  - DiT:  z-image-turbo-Q4_K_M.gguf  (5 GB)
  - VAE:  ae.safetensors              (335 MB)
  - LLM:  Qwen3-4B-Instruct-2507-Q4_K_M.gguf  (2.5 GB)

Job tracking: background thread, job_id -> {status, message, progress, output_path, error}
Tương tự tts.py.
"""

import os
import subprocess
import threading
import time
import uuid

from load_env import (
    ZIMAGE_SD_CLI,
    ZIMAGE_DIT,
    ZIMAGE_DIT_QUALITY,
    ZIMAGE_VAE,
    ZIMAGE_LLM,
    ZIMAGE_OUTPUT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIMAGE_OUTPUT_FOLDER = os.path.join(BASE_DIR, "zimage_output")
os.makedirs(ZIMAGE_OUTPUT_FOLDER, exist_ok=True)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _set(job_id: str, **kwargs):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def generate_job(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 8,
    cfg_scale: float = 1.0,
    seed: int = -1,
    quality: str = "light",
) -> str:
    """Bắt đầu job generate ảnh, trả về job_id để frontend polling.

    quality: 'light' (Q4_K_M) | 'medium' (Q5_K_M) | 'high' (Q8_0)
    """
    # Chuẩn hoá + fallback về light nếu chất lượng chưa có file
    if quality not in ZIMAGE_DIT_QUALITY:
        quality = "light"
    dit_path = ZIMAGE_DIT_QUALITY[quality]
    if not os.path.exists(dit_path):
        quality = "light"
        dit_path = ZIMAGE_DIT_QUALITY["light"]

    job_id = f"zimg_{uuid.uuid4().hex[:10]}"
    with _lock:
        _jobs[job_id] = {
            "status": "queued",
            "message": "Đang chờ xử lý...",
            "progress": 0,
            "output_path": None,
            "error": None,
            "prompt": prompt,
            "quality": quality,
        }
    t = threading.Thread(
        target=_worker,
        args=(job_id, prompt, width, height, steps, cfg_scale, seed, dit_path),
        daemon=True,
    )
    t.start()
    return job_id


def _worker(job_id: str, prompt: str, width: int, height: int, steps: int, cfg: float, seed: int, dit_path: str):
    quality_name = os.path.basename(dit_path).replace("z-image-turbo-", "").replace(".gguf", "")
    output_filename = f"zimage_{uuid.uuid4().hex[:10]}.png"
    output_path = os.path.join(ZIMAGE_OUTPUT_FOLDER, output_filename)

    _set(job_id, status="running", message=f"🎨 Đang tải model ({quality_name} + VAE + LLM)...", progress=5)

    cmd = [
        ZIMAGE_SD_CLI,
        "--diffusion-model", dit_path,
        "--vae", ZIMAGE_VAE,
        "--llm", ZIMAGE_LLM,
        "-p", prompt,
        "--cfg-scale", str(cfg),
        "--steps", str(steps),
        "-H", str(height),
        "-W", str(width),
        "--output", output_path,
        "--offload-to-cpu",
        "--backend", "vulkan",
    ]
    if seed >= 0:
        cmd += ["--seed", str(seed)]

    _set(job_id, message="⏳ Đang tạo ảnh (8 bước)...", progress=20)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            err = proc.stderr.strip()[-500:]
            _set(job_id, status="error", message=f"❌ Lỗi sd-cli: {err}", error=err)
            return

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            _set(job_id, status="error", message="❌ Không tìm thấy file output", error="no output file")
            return

        _set(
            job_id,
            status="completed",
            message="✅ Hoàn tất!",
            progress=100,
            output_path=output_path,
        )

    except subprocess.TimeoutExpired:
        _set(job_id, status="error", message="❌ Timeout (>300s)", error="timeout")
    except Exception as e:
        _set(job_id, status="error", message=f"❌ Lỗi: {e}", error=str(e))

    # Giữ job trong bộ nhớ để frontend có thể download bất cứ lúc nào
    # (trước đây xóa sau 60s → bấm download muộn bị 404).
    # Giới hạn: khi vượt quá 200 job thì xóa job cũ đã xong.
    with _lock:
        if len(_jobs) > 200:
            done = [jid for jid, j in _jobs.items()
                    if j["status"] in ("completed", "error")]
            for jid in done[: len(done) - 100]:
                _jobs.pop(jid, None)


def get_job_status(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return {"error": "Không tìm thấy job"}
    return {
        "status": job["status"],
        "message": job["message"],
        "progress": job["progress"],
    }


def get_job_output(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return {"error": "Không tìm thấy job"}
    if job["status"] != "completed":
        return {"error": "Job chưa hoàn tất"}
    return {
        "output_path": job["output_path"],
        "filename": os.path.basename(job["output_path"]),
        "quality": job.get("quality", "light"),
    }


def list_qualities() -> list:
    """Trả về danh sách mức lượng tử có file DiT tồn tại trên đĩa.

    Mỗi entry: {"key", "label", "file", "available"}
    """
    labels = {
        "light": "Nhẹ (Q4_K_M)",
        "medium": "Vừa (Q5_K_M)",
        "high": "Cao nhất (Q8_0)",
    }
    out = []
    for key in ("light", "medium", "high"):
        path = ZIMAGE_DIT_QUALITY.get(key)
        out.append({
            "key": key,
            "label": labels[key],
            "file": os.path.basename(path) if path else None,
            "available": bool(path) and os.path.exists(path),
        })
    return out
