#!/usr/bin/env bash
# =============================================================================
# start_all.sh - Khởi động tất cả dịch vụ MeetNote AI trên evo
#   - 3 llama-server (Qwen3.8-27B summarize, Qwen2.5-VL-7B OCR, Qwen2.5-VL-3B OCR)
#   - Flask app (port 5001)
#
# Dùng `setsid` để mỗi process tách khỏi phiên gọi -> không bị kill khi
# Hermes cleanup / context compact. Chạy:  bash start_all.sh
# Stop:     bash stop_all.sh
# =============================================================================
set -u

BIN=/home/viper/Work/llama.mtp/llama.cpp/build-vulkan/bin/llama-server
MODELS=/run/media/viper/Data/models
LOG=/tmp/meetnote-logs
mkdir -p "$LOG"

start() { # start <name> <port> <cmd...>
  local name="$1" port="$2"; shift 2
  if curl -s -m 1 "http://localhost:${port}/health" 2>/dev/null | grep -q ok; then
    echo "[SKIP] $name> dang chay tren port $port"
    return
  fi
  setsid "$@" >"$LOG/$name.log" 2>&1 &
  echo "[START] $name> port $port (pid $!)"
}

start summarize 8080 \
  "$BIN" -m "$MODELS/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf" \
  --reasoning on --chat-template-kwargs '{"preserve_thinking":true,"reasoning_effort":"low"}' \
  --ctx-size 131072 -ngl 999 -np 1 --spec-type draft-mtp --spec-draft-n-max 2 \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --batch-size 2048 --ubatch-size 512 --parallel 1 --cache-reuse 256 --no-mmap \
  --threads 16 --jinja --host 0.0.0.0 --port 8080 -a llama-model

# Lưu ý: OCR (Qwen2.5-VL 7B/3B) KHÔNG khởi động ở đây — chạy on-demand
# qua ocr.py (tự spawn khi có ảnh, tự kill sau idle OCR_IDLE_TIMEOUT giây).
# Xem logs/ocr-<key>.log trong thư mục dự án.

# ---- Flask ----
if curl -s -m 1 -o /dev/null "http://localhost:5001/" 2>/dev/null; then
  echo "[SKIP] flask> dang chay tren port 5001"
else
  setsid bash -c 'cd /home/viper/Work/ME/MeetNote_Ai && exec venv/bin/python app.py' >"$LOG/flask.log" 2>&1 &
  echo "[START] flask> port 5001 (pid $!)"
fi

echo ""
echo "Cho model load (27B can vai phut). Kiem tra nhanh:"
echo "  bash check_srv.sh"