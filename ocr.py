"""
ocr.py
------
OCR (Image -> Text) bằng Qwen2.5-VL, ON-DEMAND — server chỉ load khi có
job, và tự shutdown sau khi idle (giải phóng VRAM).

Hỗ trợ 2 model (Qwen2.5-VL):
  - "7b" : 7B-Instruct (chính xác cao)  -> port 8081, alias "ocr-7b"
  - "3b" : 3B-Instruct (nhanh, nhẹ)     -> port 8082, alias "ocr-3b"

Cơ chế on-demand:
  - ocr_job() -> worker check server; nếu chưa chạy thì spawn llama-server
    bằng subprocess (start_new_session=True deot tách khỏi Hermes session),
    chờ health OK rồi gửi ảnh.
  - Một watchdog thread định kỳ kill server nào idle > OCR_IDLE_TIMEOUT s.
  - Mỗi server được spawn tối đa 1 lần (theo port), tái sử dụng cho batch.

Gửi ảnh (base64 data-URL) tới /chat/completions content dạng:
  [{"type": "image_url", ...}, {"type": "text", ...}]

Không có dữ liệu nào gửi ra ngoài máy.
"""

import base64
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid

import requests

from load_env import (
    OCR_BIN,
    OCR_7B_BASE_URL, OCR_7B_MODEL, OCR_7B_GGUF, OCR_7B_MMPROJ,
    OCR_3B_BASE_URL, OCR_3B_MODEL, OCR_3B_GGUF, OCR_3B_MMPROJ,
    OCR_IDLE_TIMEOUT, OCR_AUTO_START,
    OCR_OUTPUT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OCR_UPLOAD_FOLDER = os.path.join(BASE_DIR, "ocr_upload")
os.makedirs(OCR_UPLOAD_FOLDER, exist_ok=True)
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_MODEL = "7b"

# Cấu hình từng model (port, path, mmproj, alias)
_MODEL_CONF = {
    "7b": {
        "base_url": OCR_7B_BASE_URL,
        "model": OCR_7B_MODEL,
        "label": "Qwen2.5-VL 7B (chính xác)",
        "gguf": OCR_7B_GGUF,
        "mmproj": OCR_7B_MMPROJ,
        "port": 8081,
        "extra_args": [],  # mmproj F16, không cần image-min-tokens
    },
    "3b": {
        "base_url": OCR_3B_BASE_URL,
        "model": OCR_3B_MODEL,
        "label": "Qwen2.5-VL 3B (nhanh)",
        "gguf": OCR_3B_GGUF,
        "mmproj": OCR_3B_MMPROJ,
        "port": 8082,
        "extra_args": ["--image-min-tokens", "1024"],
    },
}
OCR_MODELS = {k: (v["base_url"], v["model"], v["label"]) for k, v in _MODEL_CONF.items()}

# Job tracking: job_id -> {status, message, progress, text, image_path, error, model}
_ocr_jobs: dict[str, dict] = {}
_ocr_lock = threading.Lock()

# Process quản lý: port -> {"proc": Popen, "last_used": float, "stopped": bool}
_servers: dict[str, dict] = {}
_servers_lock = threading.Lock()


# ───────────── On-demand server lifecycle ─────────────

def _port_healthy(base_url: str, timeout: float = 1.5) -> bool:
    """Check server sẵn sàng qua endpoint health (port -> host)."""
    port = base_url.split(":")[-1].split("/")[0]
    host = base_url.split(":")[1].lstrip("/") if base_url.startswith("http") else "localhost"
    url = f"http://{host}:{port}/health"
    try:
        return requests.get(url, timeout=timeout).json().get("status") == "ok"
    except Exception:
        return False


def _server_cmd(key: str) -> list[str]:
    c = _MODEL_CONF[key]
    cmd = [
        OCR_BIN,
        "-m", c["gguf"],
        "--mmproj", c["mmproj"],
        "--host", "0.0.0.0",
        "--port", str(c["port"]),
        "-ngl", "999",
        "-np", "1",
        "--ctx-size", "32768",
        "--flash-attn", "on",
        "--jinja",
        "-a", c["model"],
    ] + c["extra_args"]
    return cmd


def ensure_server(key: str, wait: float = 120.0) -> None:
    """Đảm bảo server OCR của model key đang chạy. Spawn nếu chưa.

    - Reuse server đang chạy trên port (kể cả spawn bởi phiên trước).
    - Spawn mới bằng subprocess, start_new_session=True để tách khỏi
      Hermes session (tránh bị kill khi cleanup/compact).
    """
    c = _MODEL_CONF[key]

    # 1. Đã có server sống trên port (trong dict hoặc ngoài dict)? -> reuse
    with _servers_lock:
        ent = _servers.get(str(c["port"]))
        if ent and not ent["stopped"] and ent["proc"] and ent["proc"].poll() is None:
            ent["last_used"] = time.time()
            return

    if _port_healthy(c["base_url"]):
        # Server do phiên/process khác spawn đang chạy — gắn lại, đừng spawn trùng
        with _servers_lock:
            _servers[str(c["port"])] = {"proc": None, "last_used": time.time(), "stopped": False}
        return

    with _servers_lock:
        _servers[str(c["port"])] = {"proc": None, "last_used": time.time(), "stopped": False}

    cmd = _server_cmd(key)
    log_path = os.path.join(LOG_DIR, f"ocr-{key}.log")
    with open(log_path, "ab") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,   # tách khỏi process group của Hermes
            cwd=os.path.dirname(OCR_BIN),
        )

    with _servers_lock:
        ent = _servers.get(str(c["port"]))
        if ent:
            ent["proc"] = proc
            ent["last_used"] = time.time()
            ent["stopped"] = False

    # Chờ health
    deadline = time.time() + wait
    proc_alive = proc.poll() is None
    while time.time() < deadline and proc_alive:
        if _port_healthy(c["base_url"]):
            return
        time.sleep(1)
        proc_alive = proc.poll() is None

    # Load không xong → hủy
    if proc.poll() is None:
        _kill_proc(proc)
    raise RuntimeError(
        f"Không thể khởi động OCR server ({key}) trong {int(wait)}s. "
        f"Xem log: {log_path}"
    )


def _server_in_use(port: int) -> bool:
    """Có process nào đang nghe port này không (server cũ chưa thoát hẳn)."""
    with _servers_lock:
        ent = _servers.get(str(port))
        if ent and ent["proc"] is not None and ent["proc"].poll() is None:
            return True
    return False


def _kill_proc(proc: subprocess.Popen) -> None:
    """Gửi SIGTERM rồi SIGKILL nếu cần — giải phóng GPU/VRAM."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def stop_server(key: str) -> None:
    """Kill server của model key (giải phóng VRAM)."""
    c = _MODEL_CONF[key]
    with _servers_lock:
        ent = _servers.get(str(c["port"]))
        if ent and not ent["stopped"]:
            ent["stopped"] = True
            proc = ent["proc"]
        else:
            return
    if proc:
        _kill_proc(proc)


def _watchdog_loop(interval: float = 10.0):
    """Kill server idle > OCR_IDLE_TIMEOUT giây."""
    while True:
        time.sleep(interval)
        now = time.time()
        with _servers_lock:
            entries = [(p, e) for p, e in _servers.items() if not e["stopped"]]
        for port, ent in entries:
            if ent["proc"] and ent["proc"].poll() is None and (now - ent["last_used"]) > OCR_IDLE_TIMEOUT:
                port_int = int(port)
                key = next((k for k, v in _MODEL_CONF.items() if v["port"] == port_int), None)
                print(f"[ocr] idle timeout — freeing {key or port}", flush=True)
                _kill_proc(ent["proc"])
                with _servers_lock:
                    if port in _servers:
                        _servers[port]["stopped"] = True
                        _servers[port]["proc"] = None


# ───────────── Core OCR ─────────────

def _set(job_id: str, **kwargs):
    with _ocr_lock:
        if job_id in _ocr_jobs:
            _ocr_jobs[job_id].update(kwargs)


def _image_to_data_url(image_path: str) -> str:
    """Đọc file ảnh -> base64 data URL (mimetype theo extension)."""
    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".bmp": "image/bmp", ".gif": "image/gif",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }.get(ext, "image/png")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _ocr_prompt(language_hint: str = "") -> str:
    hint = ""
    if language_hint:
        hint = f" Văn bản trong ảnh chủ yếu là tiếng {language_hint}."
    return (
        "Hãy trích xuất TOÀN BỘ văn bản có trong hình ảnh này.\n"
        "Yêu cầu:\n"
        "- Giữ nguyên bố cục, thứ tự và ngắt dòng như bản gốc.\n"
        "- Bảng biểu: giữ dạng bảng (dùng khoảng cách / | để tách cột).\n"
        "- Không thêm lời giải thích, không dịch, không sửa nội dung.\n"
        "- Chỉ trả về văn bản đã trích xuất."
        + hint
        + "\nNếu hình ảnh không có chữ nào, trả về: (Không có text trong ảnh)"
    )


def _resolve_model(model_key: str):
    if model_key not in OCR_MODELS:
        model_key = DEFAULT_MODEL
    base_url, model_name, _ = OCR_MODELS[model_key]
    return base_url, model_name


def _call_vllm(image_path: str, prompt: str, model_key: str = DEFAULT_MODEL,
               temperature: float = 0.1, max_tokens: int = 8192,
               timeout: int = 300) -> str:
    """Đảm bảo server chạy rồi gửi chat completion kèm ảnh."""
    if OCR_AUTO_START:
        ensure_server(model_key)

    base_url, model_name = _resolve_model(model_key)
    url = f"{base_url}/chat/completions"
    data_url = _image_to_data_url(image_path)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        port = base_url.split(":")[-1].split("/")[0]
        raise RuntimeError(
            f"Không kết nối được tới OCR server ({model_key}) trên port {port}."
        ) from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            "OCR server phản hồi quá lâu (timeout). Ảnh có thể quá lớn."
        ) from e

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return "(Không có phản hồi từ model)"
    return choices[0].get("message", {}).get("content", "").strip()


def _worker(job_id: str, image_path: str, language_hint: str, model_key: str):
    """Background thread: OCR 1 ảnh với progress tracking."""
    _set(job_id, status="running", message="🔍 Đang phân tích ảnh...", progress=20)
    try:
        port = _MODEL_CONF[model_key]["port"]
        # Cập nhật thời điểm server bắt đầu được dùng (tránh watchdog giết giữa chừng)
        with _servers_lock:
            ent = _servers.get(str(port))
            if ent:
                ent["last_used"] = time.time()

        prompt = _ocr_prompt(language_hint)
        start = time.time()

        def _bump():
            elapsed = time.time() - start
            pct = min(90, 20 + int(elapsed * 4))
            _set(job_id, progress=pct)

        text = _call_vllm(image_path, prompt, model_key=model_key)
        _bump()
        _set(job_id, status="completed", message="✅ Hoàn tất!",
             progress=100, text=text)

        # Cập nhật last_used sau khi dùng — watchdog sẽ clear khi idle
        with _servers_lock:
            ent = _servers.get(str(port))
            if ent:
                ent["last_used"] = time.time()

    except Exception as e:
        _set(job_id, status="error", message=f"❌ Lỗi: {e}", error=str(e))

    with _ocr_lock:
        if len(_ocr_jobs) > 200:
            done = [jid for jid, j in _ocr_jobs.items()
                    if j["status"] in ("completed", "error")]
            for jid in done[: len(done) - 100]:
                _ocr_jobs.pop(jid, None)


def ocr_job(image_path: str, language_hint: str = "Việt",
            model_key: str = DEFAULT_MODEL) -> str:
    """Bắt đầu job OCR (on-demand), trả về job_id để frontend polling."""
    if model_key not in OCR_MODELS:
        model_key = DEFAULT_MODEL
    job_id = f"ocr_{uuid.uuid4().hex[:10]}"
    with _ocr_lock:
        _ocr_jobs[job_id] = {
            "status": "queued",
            "message": "Đang chờ xử lý...",
            "progress": 0,
            "text": None,
            "image_path": image_path,
            "error": None,
            "model": model_key,
        }
    thread = threading.Thread(
        target=_worker, args=(job_id, image_path, language_hint, model_key),
        daemon=True,
    )
    thread.start()
    return job_id


def get_job_status(job_id: str) -> dict:
    with _ocr_lock:
        job = _ocr_jobs.get(job_id)
    if job is None:
        return {"error": "Không tìm thấy job"}
    return {
        "status": job["status"],
        "message": job["message"],
        "progress": job["progress"],
    }


def get_job_output(job_id: str) -> dict:
    with _ocr_lock:
        job = _ocr_jobs.get(job_id)
    if job is None:
        return {"error": "Không tìm thấy job"}
    if job["status"] != "completed":
        return {"error": "Job chưa hoàn tất"}
    return {
        "text": job["text"],
        "image_path": job["image_path"],
    }


# ───────────── Khởi động watchdog khi import ─────────────
if OCR_AUTO_START:
    _watchdog = threading.Thread(target=_watchdog_loop, daemon=True)
    _watchdog.start()