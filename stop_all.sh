#!/usr/bin/env bash
# =============================================================================
# stop_all.sh - Tắt tất cả dịch vụ MeetNote AI trên evo
#   - llama-server summarize Qwen3.8-27B (port 8080)
#   - OCR server Qwen2.5-VL 7B/3B (port 8081/8082, nếu đang chạy)
#   - Flask app (port 5001)
#
# Chạy: bash stop_all.sh  (cũng được gọi từ nút "Tắt hệ thống" trên web UI)
# Start lại: bash start_all.sh
# =============================================================================
set -u

# Chờ 2 giây đầu tiên: nếu script được gọi từ nút tắt trên web UI,
# Flask cần thời gian trả xong response HTTP trước khi bị kill.
sleep 2

stop_port() { # stop_port <port> <tên>
  local port="$1" name="$2"
  if fuser -k "${port}/tcp" >/dev/null 2>&1; then
    echo "[STOP] $name> port $port"
  else
    echo "[SKIP] $name> khong dang chay (port $port)"
  fi
}

# Tắt model trước, Flask cuối (Flask là nơi gọi script này)
stop_port 8081 ocr-7B
stop_port 8082 ocr-3B
stop_port 8080 summarize-27B
stop_port 5001 flask

echo ""
echo "Da tat het dich vu. Khoi dong lai: bash start_all.sh"
