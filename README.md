# MiAI_MeetNote_Ai

AI Meeting Note Taker — tự động chuyển audio cuộc họp thành biên bản họp có cấu trúc.

**Chạy 100% local, không cần internet, riêng tư tuyệt đối.**

## Pipeline

1. **Upload** audio (mp3, wav, m4a, mp4, ogg, flac, webm)
2. **Split** audio thành các đoạn 10 phút bằng ffmpeg
3. **Transcribe** từng đoạn bằng:
   - 🎙️ **Qwen3-ASR 1.7B** (AMD Vulkan, chính xác cao tiếng Việt) — mặc định
   - 💎 **Gemma 4 E2B** (qua Ollama, multimodal audio)
4. **Summarize** transcript thành biên bản họp bằng llama.cpp (Qwen3.6-35B-A3B)

## Yêu cầu

- Python 3.13+
- ffmpeg (cắt audio + convert format)
- Ollama (tùy chọn, để summarize)
- Qwen3-ASR binary + model (cho engine qwen3)

## Cài đặt

```bash
cd MiAI_MeetNote_Ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Chạy

```bash
python app.py
```

Mở trình duyệt: http://localhost:5001

## Cấu trúc

| File | Mô tả |
|------|-------|
| `app.py` | Flask web app, orchestrator pipeline |
| `transcribe_qwen3.py` | Qwen3-ASR-1.7B transcribe (mặc định) |
| `transcribe_gemma.py` | Gemma 4 E2B qua Ollama |
| `summarize.py` | llama.cpp (OpenAI API) → biên bản họp |
| `audio_utils.py` | ffmpeg-based audio splitting |
| `templates/index.html` | Frontend UI |

#MìAI
Fanpage: http://page.miai.vn
Group: https://group.miai.vn
Website: http://miai.vn
