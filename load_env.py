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


BASE_DIR = str(Path(__file__).parent)

# --- Model Paths ---
QWEN3_ASR_MODEL_17B = get("QWEN3_ASR_MODEL_17B", "/run/media/viper/Data/models/qwen3-asr-0.6b/qwen3-asr-1.7b-q4_k.gguf")
QWEN3_ASR_MODEL_06B = get("QWEN3_ASR_MODEL_06B", "/run/media/viper/Data/models/qwen3-asr-0.6b/qwen3-asr-0.6b-q4_k.gguf")
CRISP_ASR_BIN = get("CRISP_ASR_BIN", "/home/viper/bin/qwen3-asr")

# --- LLM / Summarize ---
LLAMA_BASE_URL = get("LLAMA_BASE_URL", "http://localhost:8080/v1")
LLAMA_MODEL = get("LLAMA_MODEL", "llama-model")

# --- TTS ---
TTS_MODEL_PATH = get("TTS_MODEL_PATH", "/run/media/viper/Data/models")

# --- Z-Image (text-to-image) ---
ZIMAGE_SD_CLI = get("ZIMAGE_SD_CLI", "/home/viper/Work/stable-diffusion.cpp/build-vulkan/bin/sd-cli")
ZIMAGE_SD_SERVER = get("ZIMAGE_SD_SERVER", "/home/viper/Work/stable-diffusion.cpp/build-vulkan/bin/sd-server")
ZIMAGE_SERVER_PORT = int(get("ZIMAGE_SERVER_PORT", "1234"))
ZIMAGE_SERVER_START_TIMEOUT = int(get("ZIMAGE_SERVER_START_TIMEOUT", "300"))
ZIMAGE_DIT_DIR = get("ZIMAGE_DIT_DIR", "/run/media/viper/Data/models/z-image-turbo/dit")
# 3 mức lượng tử: nhẹ / vừa / cao nhất
ZIMAGE_DIT_QUALITY = {
    "light":   get("ZIMAGE_DIT_LIGHT",   os.path.join(ZIMAGE_DIT_DIR, "z-image-turbo-Q4_K_M.gguf")),
    "medium":  get("ZIMAGE_DIT_MEDIUM",  os.path.join(ZIMAGE_DIT_DIR, "z-image-turbo-Q5_K_M.gguf")),
    "high":    get("ZIMAGE_DIT_HIGH",    os.path.join(ZIMAGE_DIT_DIR, "z-image-turbo-Q8_0.gguf")),
}
ZIMAGE_DIT = get("ZIMAGE_DIT", ZIMAGE_DIT_QUALITY["light"])  # backward compat
ZIMAGE_VAE = get("ZIMAGE_VAE", "/run/media/viper/Data/models/z-image-turbo/vae/split_files/vae/ae.safetensors")
ZIMAGE_LLM = get("ZIMAGE_LLM", "/run/media/viper/Data/models/z-image-turbo/llm/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
ZIMAGE_OUTPUT = get("ZIMAGE_OUTPUT", os.path.join(BASE_DIR, "zimage_output"))

# --- Qwen-Image 2.1 (text-to-image, model thứ 2) ---
QWEN_IMG_DIR = get("QWEN_IMG_DIR", "/run/media/viper/Data/models/qwen-image-2.1")
QWEN_IMG_DIT = get("QWEN_IMG_DIT", os.path.join(QWEN_IMG_DIR, "qwen-image-2.1-Q4_K_M.gguf"))
QWEN_IMG_LLM = get("QWEN_IMG_LLM", os.path.join(QWEN_IMG_DIR, "text_encoders", "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf"))
QWEN_IMG_VAE = get("QWEN_IMG_VAE", os.path.join(QWEN_IMG_DIR, "vae", "qwen_image_2.1_vae_bf16.safetensors"))

# --- OCR (image-to-text, Qwen2.5-VL) ---
# On-demand: server tự start khi có job, tự kill sau khi idle OCR_IDLE_TIMEOUT giây
OCR_BIN = get("OCR_BIN", "/home/viper/Work/llama.mtp/llama.cpp/build-vulkan/bin/llama-server")
OCR_7B_BASE_URL = get("OCR_7B_BASE_URL", "http://localhost:8081/v1")
OCR_7B_MODEL = get("OCR_7B_MODEL", "ocr-7b")
OCR_7B_GGUF = get("OCR_7B_GGUF", "/run/media/viper/Data/models/qwen2.5-vl-7b/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
OCR_7B_MMPROJ = get("OCR_7B_MMPROJ", "/run/media/viper/Data/models/qwen2.5-vl-7b/mmproj-F16.gguf")
OCR_3B_BASE_URL = get("OCR_3B_BASE_URL", "http://localhost:8082/v1")
OCR_3B_MODEL = get("OCR_3B_MODEL", "ocr-3b")
OCR_3B_GGUF = get("OCR_3B_GGUF", "/run/media/viper/Data/models/qwen2.5-vl-3b/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf")
OCR_3B_MMPROJ = get("OCR_3B_MMPROJ", "/run/media/viper/Data/models/qwen2.5-vl-3b/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf")
# Thời gian server rảnh trước khi bị kill (giây)
OCR_IDLE_TIMEOUT = int(get("OCR_IDLE_TIMEOUT", "60"))
# If server đã bị kill bởi bên ngoài, có tự spawn lại không
OCR_AUTO_START = get("OCR_AUTO_START", "1") == "1"
# Backward compat
OCR_BASE_URL = OCR_7B_BASE_URL
OCR_MODEL = OCR_7B_MODEL
OCR_OUTPUT = get("OCR_OUTPUT", os.path.join(BASE_DIR, "ocr_upload"))

# --- Paths ---
UPLOAD_FOLDER = get("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads"))
TEMP_SEGMENTS = get("TEMP_SEGMENTS", os.path.join(BASE_DIR, "temp_segments"))
RESULTS_FOLDER = get("RESULTS_FOLDER", os.path.join(BASE_DIR, "results"))
TTS_OUTPUT = get("TTS_OUTPUT", os.path.join(BASE_DIR, "tts_output"))
