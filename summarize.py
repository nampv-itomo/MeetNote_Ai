"""
summarize.py
------------
Gọi llama.cpp server (chạy local trên evo, mặc định cổng 8080) để tóm tắt transcript
thành biên bản họp có cấu trúc. Sử dụng OpenAI-compatible API.

Model: Qwen3.6-35B-A3B-UD-Q4_K_M.gguf (MTP, Vulkan trên AMD 8060S)

Không có dữ liệu nào được gửi ra ngoài máy.
"""

import requests


from load_env import LLAMA_BASE_URL, LLAMA_MODEL
# LLAMA_BASE_URL = "http://localhost:8080/v1"
# DEFAULT_MODEL = "llama-model"  # tên model trên llama.cpp server


MEETING_MINUTES_PROMPT = """Bạn là trợ lý ghi biên bản cuộc họp. Dựa vào bản ghi transcript dưới đây và bối cảnh cuộc họp (nếu có),
hãy viết biên bản họp chuyên nghiệp, súc tích, gồm các phần:

1. Thông tin cuộc họp (ngày/giờ, địa điểm, người tham gia — nếu transcript/bối cảnh đề cập)
2. Tóm tắt chung (2-3 câu)
3. Các nội dung chính đã thảo luận (gạch đầu dòng)
4. Quyết định đã chốt (nếu có)
5. Việc cần làm tiếp theo / Action items (ai làm gì, deadline nếu có)

Bối cảnh cuộc họp (tùy chọn):
---
{context}
---

Transcript:
---
{transcript}
---

Chỉ trả về biên bản họp, không thêm lời dẫn hay giải thích khác."""


def query_llama(messages: list[dict], model: str = LLAMA_MODEL,
                temperature: float = 0.7, max_tokens: int = 4096,
                timeout: int = 600) -> str:
    """
    Gửi chat completion tới llama.cpp server và lấy kết quả.

    Args:
        messages: danh sách messages theo OpenAI format [{"role": "user", "content": "..."}].
        model: tên model (mặc định "llama-model").
        temperature: nhiệt độ sinh (mặc định 0.7).
        max_tokens: số token tối đa (mặc định 4096).
        timeout: thời gian chờ tối đa (giây).

    Returns:
        Văn bản phản hồi từ model.
    """
    url = f"{LLAMA_BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            "Không kết nối được tới llama.cpp server. "
            "Hãy chắc chắn server đang chạy:\n"
            "  llama-server -m <model.gguf> --host 0.0.0.0 --port 8080 -a llama-model"
        ) from e
    except requests.exceptions.Timeout as e:
        raise RuntimeError(
            "llama.cpp phản hồi quá lâu (timeout). "
            "Transcript có thể quá dài, hãy thử giảm đoạn hoặc tăng timeout."
        ) from e

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        return "(Không có phản hồi từ model)"
    return choices[0].get("message", {}).get("content", "").strip()


def summarize_transcript(transcript: str, context: str = "", model: str = LLAMA_MODEL) -> str:
    """Tạo biên bản họp từ transcript đầy đủ.

    Args:
        transcript: nội dung transcript.
        context: bối cảnh cuộc họp (công ty, ngày giờ, người tham gia, chủ đề...).
        model: tên model trên llama.cpp server.
    """
    if not transcript.strip():
        return "(Không có nội dung transcript để tóm tắt.)"

    # Kiểm tra transcript chỉ chứa thông báo "không có speech"
    parts = [p.strip() for p in transcript.split("\n") if p.strip()]
    all_silent = all("(không có nội dung speech" in p.lower() or 
                     "không có nội dung speech" in p.lower() 
                     for p in parts)
    if all_silent and len(parts) > 0:
        return "(Không có nội dung transcript để tóm tắt.)"

    prompt = MEETING_MINUTES_PROMPT.format(
        transcript=transcript,
        context=context or "(Không có bối cảnh)"
    )
    messages = [{"role": "user", "content": prompt}]
    return query_llama(messages, model=model)
