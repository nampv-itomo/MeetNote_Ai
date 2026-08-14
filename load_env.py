"""
load_env.py
-----------
Tải các biến cấu hình từ file .env (nếu có).
Các module khác import từ đây để có được cấu hình trung tâm.
"""

import os
from pathlib import Path


def load_env_file(env_path: str = ".env") -> dict:
    """Load các biến từ file .env."""
    config = {}
    env_file = Path(env_path)
    if not env_file.exists():
        return config
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config


# Load cấu hình
_env_config = load_env_file(str(Path(__file__).parent / ".env"))


def get(key: str, default: str = "") -> str:
    """Lấy giá trị từ .env hoặc fallback."""
    return _env_config.get(key, default)


# --- Model Paths ---
QWEN3_ASR_MODEL_17B = get("QWEN3_ASR_MODEL_17B", "/run/media/viper/Data/models/qwen3-asr-0.6b/qwen3-asr-1.7b-q4_k.gguf")
QWEN3_ASR_MODEL_06B = get("QWEN3_ASR_MODEL_06B", "/run/media/viper/Data/models/qwen3-asr-0.6b/qwen3-asr-0.6b-q4_k.gguf")
CRISP_ASR_BIN = get("CRISP_ASR_BIN", "/home/viper/bin/qwen3-asr")

# --- LLM / Summarize ---
LLAMA_BASE_URL = get("LLAMA_BASE_URL", "http://localhost:8080/v1")
LLAMA_MODEL = get("LLAMA_MODEL", "llama-model")

# --- TTS ---
TTS_MODEL_PATH = get("TTS_MODEL_PATH", "/run/media/viper/Data/models")

# --- Paths ---
BASE_DIR = str(Path(__file__).parent)
UPLOAD_FOLDER = get("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
TEMP_SEGMENTS = get("TEMP_SEGMENTS", os.path.join(BASE_DIR, "temp_segments"))
RESULTS_FOLDER = get("RESULTS_FOLDER", os.path.join(BASE_DIR, "results"))
TTS_OUTPUT = get("TTS_OUTPUT", os.path.join(BASE_DIR, "tts_output"))
