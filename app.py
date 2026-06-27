"""
FastAPI backend for MOMentum.
API docs available at:
    http://localhost:8000/docs
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessor import preprocess, cleanup_extracted_audio, VIDEO_FORMATS, SUPPORTED_FORMATS
from transcriber import transcribe_from_profile, save_transcript
from summarizer import summarize_from_transcript
from reviewer import review, ReviewResult
from formatter import format_mom_from_review
from exporter import export
from embedder import embed_from_pipeline, list_meetings
from search import index_meeting, search_and_display

# Read deployment mode
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "full")

app = FastAPI(
    title="MOMentum API",
    description="Generate Minutes of Meeting from audio or video recordings.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "output"
UPLOAD_DIR = "uploads_temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALL_SUPPORTED = SUPPORTED_FORMATS | VIDEO_FORMATS
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 500))


#CRON JOB
# Optimization: pre-load models at startup so first user request is instant.
# In lite mode (deployed) — only Whisper tiny is preloaded to stay within 512MB RAM.
# In full mode (local) — only Whisper base is preloaded; distilbart loads on first request.
# distilbart and MiniLM load lazily on first use and stay cached in memory after.

@app.on_event("startup")
async def startup_event():
    print(f"\n  MOMentum — Starting up (mode: {DEPLOYMENT_MODE.upper()}) ...")
    try:
        import whisper
        model_size = "tiny" if DEPLOYMENT_MODE == "lite" else "base"
        print(f"  Pre-loading Whisper {model_size} model...")
        whisper.load_model(model_size)
        print(f"  Whisper {model_size} ready.")
    except Exception as e:
        print(f"  Warning: Whisper pre-load failed ({e})")
    print("  MOMentum is live!\n")


@app.get("/api/health", tags=["System"])
async def health_check():
    meetings = list_meetings()
    return {
        "status": "ok",
        "version": "1.0.0",
        "deployment_mode": DEPLOYMENT_MODE,
        "meetings_indexed": len(meetings),
        "supported_formats": sorted(list(ALL_SUPPORTED)),
    }


@app.post("/api/upload", tags=["Pipeline"])
async def upload_and_process(
    file: UploadFile = File(...),
    format: str = Form(default="md"),
    title: Optional[str] = Form(default=None),
    save_transcript_flag: bool = Form(default=False),
):
    start_time = time.time()

    if format not in ["txt", "md", "pdf"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{format}'. Use: txt, md, pdf"
        )

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALL_SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. "
                   f"Supported: {', '.join(sorted(ALL_SUPPORTED))}"
        )

    tmp_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(tmp_path, "wb") as f_out:
            content = await file.read()
            size_mb = len(content) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large ({size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB"
                )
            f_out.write(content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")

    try:
        profile = preprocess(tmp_path, output_dir=OUTPUT_DIR)
        transcript = transcribe_from_profile(profile)

        if transcript.word_count == 0:
            raise HTTPException(
                status_code=422,
                detail="No speech detected in the audio. "
                       "Please check the file contains clear speech."
            )

        if save_transcript_flag:
            transcript_path = os.path.join(
                OUTPUT_DIR,
                Path(file.filename).stem + "_transcript.txt"
            )
            save_transcript(transcript, transcript_path)

        summary = summarize_from_transcript(transcript)

        try:
            review_result = review(summary, transcript.full_text)
        except Exception:
            review_result = ReviewResult(
                original_summary=summary,
                refined_summary=summary,
                passes_taken=0,
                final_score=0,
                accepted=False,
            )

        mom = format_mom_from_review(
            profile=profile,
            transcript=transcript,
            review_result=review_result,
            title=title,
        )
        mom_path = export(mom, format=format, output_dir=OUTPUT_DIR)

        try:
            embedding = embed_from_pipeline(
                transcript=transcript,
                profile=profile,
                mom_path=mom_path,
            )
            index_meeting(embedding)
        except Exception:
            pass

        cleanup_extracted_audio(profile)

        elapsed = round(time.time() - start_time, 1)
        mom_filename = Path(mom_path).name

        return JSONResponse({
            "status": "success",
            "filename": mom_filename,
            "duration": profile.audio_file.duration_formatted,
            "language": profile.language_name,
            "whisper_model_used": profile.recommended_model,
            "word_count": transcript.word_count,
            "review_passes": review_result.passes_taken,
            "review_score": review_result.final_score,
            "processing_time_seconds": elapsed,
            "download_url": f"/api/download/{mom_filename}",
            "preview": {
                "overview": mom.overview,
                "action_items": mom.action_items,
                "decisions": mom.decisions,
                "next_steps": mom.next_steps,
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/api/download/{filename}", tags=["Files"])
async def download_mom(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    ext = Path(filename).suffix.lower()
    media_types = {
        ".md":  "text/markdown",
        ".txt": "text/plain",
        ".pdf": "application/pdf",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(path=filepath, filename=filename, media_type=media_type)


@app.get("/api/search", tags=["Search"])
async def semantic_search(q: str, top_k: int = 3):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        from search import search
        results = search(q.strip(), top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    return {
        "query": q,
        "total_results": len(results),
        "results": [
            {
                "meeting_id": r.meeting_id,
                "date": r.date,
                "audio_file": r.audio_file,
                "duration": r.duration,
                "score": round(r.score, 3),
                "transcript_preview": r.transcript_preview,
                "download_url": f"/api/download/{Path(r.mom_path).name}"
                                if r.mom_path and os.path.exists(r.mom_path) else None,
            }
            for r in results
        ]
    }


@app.get("/api/meetings", tags=["Meetings"])
async def get_meetings():
    meetings = list_meetings()
    return {
        "total": len(meetings),
        "meetings": [
            {
                **m,
                "download_url": f"/api/download/{Path(m['mom_path']).name}"
                                if m.get("mom_path") and os.path.exists(m["mom_path"])
                                else None,
            }
            for m in meetings
        ]
    }


frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)