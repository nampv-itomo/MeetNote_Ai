"""
zimage.py
---------
Text-to-Image qua stable-diffusion.cpp (sd-server, Vulkan).

Model:
  - DiT:  z-image-turbo-Q4_K_M.gguf  (5 GB)
  - VAE:  ae.safetensors              (335 MB)
  - LLM:  Qwen3-4B-Instruct-2507-Q4_K_M.gguf  (2.5 GB)

Job tracking: background thread, job_id -> {status, message, progress, output_path, error}
Tương tự tts.py.
"""

import atexit
import base64
import os
import subprocess
import threading
import time
import uuid

import requests

from load_env import (
    ZIMAGE_SD_SERVER,
    ZIMAGE_SERVER_PORT,
    ZIMAGE_SERVER_START_TIMEOUT,
    ZIMAGE_DIT_QUALITY,
    ZIMAGE_VAE,
    ZIMAGE_LLM,
    ZIMAGE_OUTPUT,
    QWEN_IMG_DIT,
    QWEN_IMG_VAE,
    QWEN_IMG_LLM,
)

# Hai engine text-to-image. "zimage" = mặc định, "qwenimg" = Qwen-Image 2.1.
MODELS = {
    "zimage": {
        "label": "Z-Image-Turbo 6B",
        "vae": ZIMAGE_VAE,
        "llm": ZIMAGE_LLM,
        "default_steps": 8,
        "default_cfg": 1.0,
        "is_turbo": True,
    },
    "qwenimg": {
        "label": "Qwen-Image 2.1 (Unsloth GGUF)",
        "vae": QWEN_IMG_VAE,
        "llm": QWEN_IMG_LLM,
        "default_steps": 20,
        "default_cfg": 6.0,
        "is_turbo": False,
    },
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ZIMAGE_OUTPUT_FOLDER = os.path.join(BASE_DIR, "zimage_output")
os.makedirs(ZIMAGE_OUTPUT_FOLDER, exist_ok=True)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_generation_lock = threading.Lock()
_server_proc = None
_loaded_model_key = None
_unloading = False


def _model_key(model: str, dit_path: str) -> tuple:
    spec = MODELS[model]
    return model, dit_path, spec["vae"], spec["llm"]


def _stop_process(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _clear_server_state(proc=None) -> None:
    global _server_proc, _loaded_model_key
    with _lock:
        if proc is None or _server_proc is proc:
            _server_proc = None
            _loaded_model_key = None


def _ensure_server(model: str, dit_path: str) -> None:
    """Start the local inference server lazily and reuse its loaded model."""
    global _server_proc, _loaded_model_key
    key = _model_key(model, dit_path)
    with _lock:
        proc = _server_proc
        loaded_key = _loaded_model_key
        if proc is not None and proc.poll() is None and loaded_key == key:
            return

    if proc is not None:
        _clear_server_state(proc)
        _stop_process(proc)

    spec = MODELS[model]
    cmd = [
        ZIMAGE_SD_SERVER,
        "--diffusion-model", dit_path,
        "--vae", spec["vae"],
        "--llm", spec["llm"],
        "--offload-to-cpu",
        "--backend", "vulkan",
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(ZIMAGE_SERVER_PORT),
    ]
    if not spec["is_turbo"]:
        cmd += ["--sampling-method", "euler"]

    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
    with _lock:
        shutting_down = _unloading
        if not shutting_down:
            _server_proc = proc
            _loaded_model_key = key
    if shutting_down:
        _stop_process(proc)
        raise RuntimeError("MeetNote đang tắt; không thể nạp model mới")

    endpoint = f"http://127.0.0.1:{ZIMAGE_SERVER_PORT}/sdapi/v1/options"
    deadline = time.monotonic() + ZIMAGE_SERVER_START_TIMEOUT
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("sd-server dừng trước khi sẵn sàng; kiểm tra log ứng dụng")
            try:
                response = requests.get(endpoint, timeout=2)
                response.raise_for_status()
                return
            except requests.RequestException:
                time.sleep(0.5)
        raise TimeoutError(f"sd-server không sẵn sàng sau {ZIMAGE_SERVER_START_TIMEOUT}s")
    except Exception:
        _clear_server_state(proc)
        _stop_process(proc)
        raise


def get_runtime_status() -> dict:
    """Trạng thái an toàn để hiển thị trên giao diện, không lộ đường dẫn model."""
    global _server_proc, _loaded_model_key
    with _lock:
        if _server_proc is not None and _server_proc.poll() is not None:
            _server_proc = None
            _loaded_model_key = None
        busy = any(job["status"] in ("queued", "running") for job in _jobs.values())
        model = _loaded_model_key[0] if _loaded_model_key else None
        return {
            "is_loaded": _server_proc is not None,
            "loaded_model": model,
            "model_label": MODELS[model]["label"] if model else None,
            "busy": busy,
            "unloading": _unloading,
        }


def unload_model() -> dict:
    """Unload only this module's server; never interrupt queued/running jobs."""
    global _unloading, _server_proc, _loaded_model_key
    with _lock:
        if _unloading:
            return {"status": "busy", "busy": True, "message": "Đang giải phóng model..."}
        if any(job["status"] in ("queued", "running") for job in _jobs.values()):
            return {
                "status": "busy", "busy": True,
                "message": "Đang tạo ảnh; hãy chờ job hoàn tất rồi thử lại.",
            }
        proc = _server_proc
        if proc is None or proc.poll() is not None:
            _server_proc = None
            _loaded_model_key = None
            return {"status": "already_unloaded", "busy": False}
        _unloading = True

    try:
        with _generation_lock:
            _clear_server_state(proc)
            _stop_process(proc)
        return {"status": "unloaded", "busy": False}
    finally:
        with _lock:
            _unloading = False


def shutdown_server() -> None:
    """Stop the owned inference server when Flask exits or receives SIGTERM."""
    global _unloading, _server_proc, _loaded_model_key
    with _lock:
        proc = _server_proc
        _server_proc = None
        _loaded_model_key = None
        _unloading = True
    try:
        _stop_process(proc)
    finally:
        with _lock:
            _unloading = False


atexit.register(shutdown_server)


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
    model: str = "zimage",
) -> str:
    """Bắt đầu job generate ảnh, trả về job_id để frontend polling.

    model: 'zimage' | 'qwenimg'
    quality: chỉ áp cho zimage — 'light' (Q4_K_M) | 'medium' (Q5_K_M) | 'high' (Q8_0)
    """
    model = model if model in MODELS else "zimage"
    spec = MODELS[model]

    # Chọn DiT theo model + mức lượng tử
    if model == "zimage":
        if quality not in ZIMAGE_DIT_QUALITY:
            quality = "light"
        dit_path = ZIMAGE_DIT_QUALITY[quality]
        if not os.path.exists(dit_path):
            quality = "light"
            dit_path = ZIMAGE_DIT_QUALITY["light"]
    else:
        dit_path = QWEN_IMG_DIT
        quality = ""

    if not os.path.exists(dit_path):
        return "__missing:" + dit_path

    job_id = f"zimg_{uuid.uuid4().hex[:10]}"
    with _lock:
        if _unloading:
            return "__busy"
        _jobs[job_id] = {
            "status": "queued",
            "message": "Đang chờ xử lý...",
            "progress": 0,
            "output_path": None,
            "error": None,
            "prompt": prompt,
            "quality": quality,
            "model": model,
        }
    t = threading.Thread(
        target=_worker,
        args=(job_id, prompt, width, height, steps, cfg_scale, seed, dit_path, model),
        daemon=True,
    )
    t.start()
    return job_id


def _worker(job_id: str, prompt: str, width: int, height: int, steps: int, cfg: float, seed: int, dit_path: str, model: str):
    spec = MODELS[model]
    output_filename = f"{model}_{uuid.uuid4().hex[:10]}.png"
    output_path = os.path.join(ZIMAGE_OUTPUT_FOLDER, output_filename)

    _set(job_id, status="running", message=f"🎨 Đang nạp model ({spec['label']} + VAE + text encoder)...", progress=5)
    try:
        with _generation_lock:
            with _lock:
                if _unloading:
                    raise RuntimeError("Model đang được giải phóng; vui lòng gửi lại job sau.")
            _ensure_server(model, dit_path)
            _set(job_id, message=f"⏳ Đang tạo ảnh ({steps} bước)...", progress=20)

            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg,
                "batch_size": 1,
            }
            if model == "qwenimg":
                payload["sampler_name"] = "euler"
            if seed >= 0:
                payload["seed"] = seed

            response = requests.post(
                f"http://127.0.0.1:{ZIMAGE_SERVER_PORT}/sdapi/v1/txt2img",
                json=payload,
                timeout=(10, None),
            )
            response.raise_for_status()
            images = response.json().get("images") or []
            if not images or not isinstance(images[0], str):
                raise RuntimeError("sd-server không trả về ảnh")
            encoded_image = images[0].split(",", 1)[-1]
            image_bytes = base64.b64decode(encoded_image, validate=True)
            if not image_bytes:
                raise RuntimeError("sd-server trả về ảnh rỗng")
            with open(output_path, "wb") as image_file:
                image_file.write(image_bytes)

        _set(
            job_id,
            status="completed",
            message="✅ Hoàn tất!",
            progress=100,
            output_path=output_path,
        )

    except Exception as e:
        error = str(e)[-500:]
        _set(job_id, status="error", message=f"❌ Lỗi sd-server: {error}", error=error)

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


def list_models() -> list:
    """Trả về danh sách engine text-to-image sẵn sàng trên đĩa.

    Mỗi entry: {"key", "label", "available", "reason"}
    """
    out = []
    for key, spec in MODELS.items():
        if key == "zimage":
            avail = any(
                bool(p) and os.path.exists(p) for p in ZIMAGE_DIT_QUALITY.values()
            )
        else:
            avail = os.path.exists(QWEN_IMG_DIT) and os.path.exists(spec["vae"]) and os.path.exists(spec["llm"])
        reason = "" if avail else "chưa tải đủ file"
        out.append({"key": key, "label": spec["label"], "available": avail, "reason": reason})
    return out
