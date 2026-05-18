# 🎙 VoxScript — Audio Transcription & Translation

A full-stack web application for transcribing and translating large audio/video files with **word-level timestamp subtitles**, powered by OpenAI Whisper.

---

## ✨ Features

- 🎯 **Transcription** — Transcribe audio in 50+ languages
- 🌐 **Translation** — Translate transcripts to English, Hindi, German, Japanese, French, and more
- ⏱ **Word-level timestamps** — Karaoke-style subtitle precision
- 📄 **Multiple export formats** — SRT, VTT, Word SRT, JSON, TXT
- 🎬 **Large file support** — Audio & video (MP4, MKV, AVI, MOV, etc.)
- 🔧 **Model selection** — Tiny to Large-v3 (speed vs accuracy tradeoff)
- 🌍 **50+ languages** — Including Hindi, Tamil, Bengali, Urdu, and more

---

## 🚀 Quick Start

### Option 1: Direct Python (Recommended for dev)

```bash
# 1. Install system dependencies
sudo apt-get install ffmpeg   # Ubuntu/Debian
# or: brew install ffmpeg     # macOS

# 2. Install Python packages
cd backend
pip install -r requirements.txt

# 3. Start server (serves both API + frontend)
python main.py
```

Open: http://localhost:8000

---

### Option 2: Docker Compose (Recommended for production)

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000  
- Backend API: http://localhost:8000  
- API Docs: http://localhost:8000/docs

---

## 📁 Project Structure

```
audiotranscribe/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile
│   ├── uploads/             # Temp storage for uploaded files
│   └── outputs/             # Generated SRT/VTT/JSON files
├── frontend/
│   └── index.html           # Full SPA frontend
├── docker-compose.yml
├── nginx.conf
└── start.sh
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload file and start job |
| `GET` | `/api/job/{id}` | Poll job status + results |
| `GET` | `/api/job/{id}/download/{fmt}` | Download output (srt/vtt/word_srt/json/txt) |
| `DELETE` | `/api/job/{id}` | Delete job and files |
| `GET` | `/api/languages` | List supported languages |
| `GET` | `/api/models` | List available Whisper models |
| `GET` | `/api/health` | Health check |

---

## ⚙️ Upload Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | File | required | Audio/video file |
| `source_language` | string | `auto` | Source language code (e.g. `hi`, `en`, `auto`) |
| `target_language` | string | `en` | Target language (used with translate task) |
| `task` | string | `transcribe` | `transcribe` or `translate` |
| `model_size` | string | `base` | `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` |

---

## 💡 Tips

- **Large files**: Use `small` or `medium` model for a good speed/accuracy balance
- **Hindi/Indic languages**: Set source language explicitly for better accuracy
- **Same-language text**: Use `transcribe` with `auto` or the exact source language
- **Non-English translation**: Uses `deep-translator` after Whisper creates the transcript
- **Noisy audio**: Use `medium` or `large-v3` model
- **Quick drafts**: Use `tiny` or `base` for fast processing
- **Production**: Use Docker Compose with proper volume mounts for Whisper model cache

---

## 🖥 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8+ GB |
| Disk | 5 GB | 20 GB |
| GPU (optional) | — | NVIDIA 8GB+ VRAM for large model |
| Python | 3.9+ | 3.11 |
| ffmpeg | required | — |

---

## 📦 Dependencies

- **FastAPI** — Backend framework
- **OpenAI Whisper** — ASR & translation engine
- **PyTorch** — ML backend
- **ffmpeg** — Audio/video processing
- **uvicorn** — ASGI server

---

## 🔒 Privacy

Your audio files are processed locally on your server. English translation uses Whisper locally. Non-English translation uses `deep-translator`, so transcript text may be sent to the translation provider configured by that package.
