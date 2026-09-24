#!/usr/bin/env bash
# check_srv.sh - Xem trạng thái các dịch vụ MeetNote AI
for spec in "8080 summarize-27B" "8081 ocr-7B" "8082 ocr-3B" "5001 flask"; do
  P="${spec%% *}"; N="${spec##* }"
  if curl -s -m 2 "http://localhost:${P}/health" 2>/dev/null | grep -q ok; then
    echo "[:)] $N (port $P): UP"
  elif curl -s -m 2 -o /dev/null "http://localhost:${P}/" 2>/dev/null; then
    echo "[:)] $N (port $P): UP"
  else
    echo "[:(] $N (port $P): DOWN"
  fi
done
echo ""
echo "RAM: $(free -h | awk '/Mem:/{print $3" / "$2}')"