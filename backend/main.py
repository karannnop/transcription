import os
import uuid
import json
import time
import asyncio
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="AudioTranscribe API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (use Redis/DB in production)
jobs: Dict[str, Dict[str, Any]] = {}

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Supported languages
LANGUAGES = {
    "auto": "Auto Detect",
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "ur": "Urdu",
    "pa": "Punjabi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "th": "Thai",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "sk": "Slovak",
    "ro": "Romanian",
    "hu": "Hungarian",
    "uk": "Ukrainian",
    "id": "Indonesian",
    "ms": "Malay",
    "fa": "Persian",
    "he": "Hebrew",
    "sw": "Swahili",
}

def update_job(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        jobs[job_id]["updated_at"] = time.time()

def extract_audio(input_path: Path, output_path: Path) -> bool:
    """Extract audio from video file using ffmpeg"""
    try:
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(output_path), "-y", "-loglevel", "error"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        return result.returncode == 0
    except Exception:
        return False

def get_audio_duration(file_path: Path) -> float:
    """Get audio duration using ffprobe"""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "json", str(file_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except Exception:
        return 0.0

def format_srt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp HH:MM:SS,mmm"""
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def format_vtt_time(seconds: float) -> str:
    """Format seconds to WebVTT timestamp HH:MM:SS.mmm"""
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = int(seconds) // 60 % 60
    h = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def segments_to_srt(segments: list) -> str:
    """Convert whisper segments to SRT format"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        lines.append(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n")
    return "\n".join(lines)

def segments_to_vtt(segments: list) -> str:
    """Convert whisper segments to WebVTT format"""
    lines = ["WEBVTT\n"]
    for seg in segments:
        start = format_vtt_time(seg["start"])
        end = format_vtt_time(seg["end"])
        lines.append(f"{start} --> {end}\n{seg['text'].strip()}\n")
    return "\n".join(lines)

def word_timestamps_to_srt(words: list) -> str:
    """Convert word-level timestamps to SRT"""
    lines = []
    # Group words into subtitle chunks (max 10 words or 5 seconds)
    chunks = []
    current_chunk = []
    chunk_start = None

    for word in words:
        if not current_chunk:
            chunk_start = word.get("start", 0)
        current_chunk.append(word)
        duration = word.get("end", 0) - chunk_start
        if len(current_chunk) >= 8 or duration >= 4:
            chunks.append({
                "start": chunk_start,
                "end": word.get("end", chunk_start + 1),
                "words": current_chunk[:]
            })
            current_chunk = []
            chunk_start = None

    if current_chunk:
        chunks.append({
            "start": chunk_start,
            "end": current_chunk[-1].get("end", chunk_start + 1),
            "words": current_chunk
        })

    for i, chunk in enumerate(chunks, 1):
        start = format_srt_time(chunk["start"])
        end = format_srt_time(chunk["end"])
        text = " ".join(w.get("word", "").strip() for w in chunk["words"])
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    return "\n".join(lines)

async def run_transcription(
    job_id: str,
    audio_path: Path,
    source_lang: str,
    target_lang: str,
    task: str,  # "transcribe" or "translate"
    model_size: str
):
    """Run whisper transcription in background"""
    try:
        update_job(job_id, status="loading_model", progress=5,
                   message=f"Loading Whisper {model_size} model...")

        try:
            import whisper
        except ImportError:
            update_job(job_id, status="error",
                       message="Whisper not installed. Run: pip install openai-whisper")
            return

        update_job(job_id, status="loading_model", progress=15,
                   message=f"Loading model into memory...")

        model = whisper.load_model(model_size)

        update_job(job_id, status="transcribing", progress=30,
                   message="Starting transcription...")

        # Build whisper options
        decode_options = {
            "task": task,
            "word_timestamps": True,
            "verbose": False,
            "fp16": False,
        }

        if source_lang != "auto":
            decode_options["language"] = source_lang

        update_job(job_id, status="transcribing", progress=40,
                   message="Transcribing audio (this may take a while for large files)...")

        # Run transcription
        result = model.transcribe(str(audio_path), **decode_options)

        update_job(job_id, status="processing", progress=80,
                   message="Processing results...")

        # Extract segments and words
        segments = []
        all_words = []

        for seg in result.get("segments", []):
            segments.append({
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            })
            # Word-level timestamps
            for word in seg.get("words", []):
                all_words.append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                    "probability": word.get("probability", 1.0)
                })

        # Generate subtitle files
        srt_content = segments_to_srt(segments)
        vtt_content = segments_to_vtt(segments)
        word_srt_content = word_timestamps_to_srt(all_words) if all_words else srt_content

        # Save files
        srt_path = OUTPUT_DIR / f"{job_id}.srt"
        vtt_path = OUTPUT_DIR / f"{job_id}.vtt"
        word_srt_path = OUTPUT_DIR / f"{job_id}_words.srt"
        json_path = OUTPUT_DIR / f"{job_id}.json"

        srt_path.write_text(srt_content, encoding="utf-8")
        vtt_path.write_text(vtt_content, encoding="utf-8")
        word_srt_path.write_text(word_srt_content, encoding="utf-8")

        full_result = {
            "job_id": job_id,
            "detected_language": result.get("language", source_lang),
            "task": task,
            "full_text": result.get("text", "").strip(),
            "segments": segments,
            "words": all_words,
            "duration": segments[-1]["end"] if segments else 0,
        }
        json_path.write_text(json.dumps(full_result, ensure_ascii=False, indent=2), encoding="utf-8")

        update_job(
            job_id,
            status="completed",
            progress=100,
            message="Transcription complete!",
            result=full_result,
            files={
                "srt": str(srt_path),
                "vtt": str(vtt_path),
                "word_srt": str(word_srt_path),
                "json": str(json_path),
            }
        )

    except Exception as e:
        update_job(job_id, status="error", message=str(e), progress=0)
    finally:
        # Cleanup audio file
        try:
            if audio_path.exists():
                audio_path.unlink()
        except Exception:
            pass


@app.get("/api/languages")
async def get_languages():
    return {"languages": LANGUAGES}

@app.get("/api/models")
async def get_models():
    return {
        "models": [
            {"id": "tiny", "name": "Tiny (Fast, ~1GB VRAM, lower accuracy)"},
            {"id": "base", "name": "Base (Balanced, ~1GB VRAM)"},
            {"id": "small", "name": "Small (Good accuracy, ~2GB VRAM)"},
            {"id": "medium", "name": "Medium (High accuracy, ~5GB VRAM)"},
            {"id": "large-v3", "name": "Large v3 (Best accuracy, ~10GB VRAM)"},
            {"id": "large-v2", "name": "Large v2 (~10GB VRAM)"},
        ]
    }

@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_language: str = Form("auto"),
    target_language: str = Form("en"),
    task: str = Form("transcribe"),
    model_size: str = Form("base"),
):
    # Validate file type
    allowed_extensions = {
        ".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".webm",
        ".mkv", ".avi", ".mov", ".wmv", ".aac", ".opus", ".wma"
    }
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {file_ext}")

    # Validate task
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, "Task must be 'transcribe' or 'translate'")

    # Save uploaded file
    job_id = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / f"{job_id}{file_ext}"

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(400, "Uploaded file is empty")

        with open(upload_path, "wb") as f:
            f.write(content)

        file_size_mb = len(content) / (1024 * 1024)

    except Exception as e:
        raise HTTPException(500, f"Failed to save file: {e}")

    # Initialize job
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued...",
        "filename": file.filename,
        "file_size_mb": round(file_size_mb, 2),
        "source_language": source_language,
        "target_language": target_language,
        "task": task,
        "model_size": model_size,
        "created_at": time.time(),
        "updated_at": time.time(),
        "result": None,
        "files": None,
    }

    # Extract audio if video file
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".webm"}
    audio_path = upload_path

    if file_ext in video_extensions:
        audio_path = UPLOAD_DIR / f"{job_id}.wav"
        update_job(job_id, status="extracting_audio", progress=10,
                   message="Extracting audio from video...")
        success = extract_audio(upload_path, audio_path)
        if not success:
            # Fallback: try to use the original file
            audio_path = upload_path
        else:
            upload_path.unlink()  # Remove original video

    # Start background transcription
    background_tasks.add_task(
        run_transcription,
        job_id, audio_path, source_language, target_language, task, model_size
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "File uploaded successfully. Processing started.",
        "filename": file.filename,
        "file_size_mb": round(file_size_mb, 2),
    }

@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id].copy()
    # Don't return file paths to client
    job.pop("files", None)
    return job

@app.get("/api/job/{job_id}/download/{format}")
async def download_file(job_id: str, format: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, "Job not completed yet")

    files = job.get("files", {})
    format_map = {
        "srt": ("srt", "application/x-subrip"),
        "vtt": ("vtt", "text/vtt"),
        "word_srt": ("word_srt", "application/x-subrip"),
        "json": ("json", "application/json"),
        "txt": None,  # Generate on-the-fly
    }

    if format not in format_map:
        raise HTTPException(400, f"Unknown format: {format}")

    if format == "txt":
        text = job["result"]["full_text"]
        filename = f"{Path(job['filename']).stem}_transcript.txt"
        return StreamingResponse(
            iter([text]),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )

    key, media_type = format_map[format]
    file_path = Path(files.get(key, ""))

    if not file_path.exists():
        raise HTTPException(404, "Output file not found")

    stem = Path(job["filename"]).stem
    ext = "srt" if "srt" in format else format
    download_name = f"{stem}_{format}.{ext}"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=download_name
    )

@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    # Clean up output files
    if job.get("files"):
        for path_str in job["files"].values():
            p = Path(path_str)
            if p.exists():
                p.unlink()

    del jobs[job_id]
    return {"message": "Job deleted"}

@app.get("/api/health")
async def health():
    try:
        import whisper
        whisper_available = True
        whisper_version = getattr(whisper, "__version__", "unknown")
    except ImportError:
        whisper_available = False
        whisper_version = None

    ffmpeg_available = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True
    ).returncode == 0

    return {
        "status": "ok",
        "whisper_available": whisper_available,
        "whisper_version": whisper_version,
        "ffmpeg_available": ffmpeg_available,
    }

# Mount frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
