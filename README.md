# MOMentum 
> Upload your meeting recording or video. Get a professional Minutes of Meeting document instantly.

MOMentum is a full-stack AI web application that takes any meeting audio or video file, intelligently preprocesses it, transcribes it, reviews and refines the output, and delivers a clean structured Minutes of Meeting document. Entirely free, no API keys, runs on your own server.

---

## Live Demo
> Coming soon ...

---

## Table of Contents

- [How It Works](#how-it-works)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Output Format](#output-format)
- [Deployment](#deployment)
- [License](#license)

---

## How It Works

```
1. User visits the website
2. Uploads a meeting audio or video file
3. If video : audio track is extracted automatically
4. Pre-processing agent analyzes audio quality, detects language, and selects the best Whisper model
5. Whisper transcribes the audio locally on the server
6. Summarizer extracts summary, action items, and decisions
7. Review agent checks the MOM for gaps and refines iteratively
8. Final MOM is returned as a downloadable document
9. Transcript is indexed into FAISS for future semantic search
10. User can later query : "What did we decide about X last month?"
```

No account needed. No API keys. No cost per request.
All AI models run directly on the server.

---

## Features

- **Video & Audio Support** : Accepts MP4, MKV, MOV, AVI, WEBM, MP3, WAV, M4A, FLAC, OGG
- **Pre-processing Agent** : Analyzes audio quality, detects language, auto-selects Whisper model
- **Local Transcription** : OpenAI Whisper running on the server, no paid API
- **Auto Summarization** : Concise meeting summary via HuggingFace Transformers
- **Action Item Extraction** : Tasks, owners, and deadlines pulled automatically
- **Review Agent** : Self checks the MOM for completeness, refines gaps iteratively
- **Semantic Search** : FAISS-powered search over all past meeting transcripts
- **Structured MOM Output** : Date, attendees, agenda, discussion, decisions, action items, next steps
- **Multiple Export Formats** : Download as `.txt`, `.md`, or `.pdf`
- **Web Interface** : Clean drag-and-drop UI, no installation needed
- **REST API** : FastAPI backend, usable independently by developers

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                 User's Browser                   │
│                                                  │
│   ┌──────────────────────────────────────────┐   │
│   │         Frontend (HTML/CSS/JS)           │   │
│   │   • Drag & drop audio/video upload       │   │
│   │   • Real-time progress updates           │   │
│   │   • MOM preview & download               │   │
│   │   • Semantic search interface            │   │
│   └──────────────┬───────────────────────────┘   │
└──────────────────│───────────────────────────────┘
                   │ HTTP multipart/form-data
                   ▼
┌──────────────────────────────────────────────────┐
│              FastAPI Backend Server              │
│                                                  │
│  ┌───────────────────────────────────────────┐   │
│  │  STAGE 1 — Pre-processing Agent           │   │
│  │  • Video → extract audio (moviepy)        │   │
│  │  • Validate & load audio                  │   │
│  │  • Analyze quality, detect language       │   │
│  │  • select Whisper model                   │   │
│  └──────────────────┬────────────────────────┘   │
│                     ▼                            │
│  ┌───────────────────────────────────────────┐   │
│  │  STAGE 2 — Transcription                  │   │
│  │  • Whisper (auto-selected model)          │   │
│  │  • Raw transcript + timestamps            │   │
│  └──────────────────┬────────────────────────┘   │
│                     ▼                            │
│  ┌───────────────────────────────────────────┐   │
│  │  STAGE 3 — Summarization                  │   │
│  │  • Meeting summary (distilbart)           │   │
│  │  • Action items extraction                │   │
│  │  • Decisions extraction                   │   │
│  └──────────────────┬────────────────────────┘   │
│                     ▼                            │
│  ┌───────────────────────────────────────────┐   │
│  │  STAGE 4 — Review Agent                   │   │
│  │  • Check action items completeness        │   │
│  │  • Check decisions clearly stated         │   │
│  │  • Refine & loop (max 3 passes)           │   │
│  └──────────────────┬────────────────────────┘   │
│                     ▼                            │
│  ┌───────────────────────────────────────────┐   │
│  │  STAGE 5 — Format & Export                │   │
│  │  • Structure into final MOM               │   │
│  │  • Export as txt / md / pdf               │   │
│  └──────────────────┬────────────────────────┘   │
│                     ▼                            │
│  ┌───────────────────────────────────────────┐   │
│  │  STAGE 6 — Memory & Search                │   │
│  │  • Embed transcript (MiniLM ~80MB)        │   │
│  │  • Index into FAISS vector store          │   │
│  │  • Expose semantic search API             │   │
│  └───────────────────────────────────────────┘   │
│                                                  │
│  GET /api/search?q=budget+decision               │
└──────────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│          Deployed on Render / Railway            │
└──────────────────────────────────────────────────┘
```

---

## Project Structure

```
MOMentum/
├── src/
│   ├── __init__.py
│   ├── audio.py            # Audio loading, validation, metadata extraction
│   ├── preprocessor.py     # Pre-processing agent — video→audio, quality, model selection
│   ├── transcriber.py      # Whisper transcription
│   ├── summarizer.py       # Summary, action items, decisions extraction
│   ├── reviewer.py         # Review agent — quality check + refinement loop
│   ├── formatter.py        # MOM document structure assembly
│   ├── exporter.py         # Export to txt / md / pdf (more file formats to come according to meeting discussions)
│   ├── embedder.py         # Transcript → vector embeddings (MiniLM)
│   ├── search.py           # FAISS semantic search over past meetings
│   └── utils.py            # Shared helpers and utilities
├── frontend/
│   ├── index.html          # Main web UI
│   ├── style.css           # Styles
│   └── app.js              # Upload logic, progress bar, search UI
├── tests/
│   └── __init__.py
├── samples/                # Test audio/video files (gitignored)
├── output/                 # Generated MOM files (gitignored)
├── vector_store/           # FAISS index files (gitignored)
├── app.py                  # FastAPI application & all routes
├── momentum.py             # CLI entry point
├── requirements.txt
├── setup.py
├── .env
├── .gitignore
└── README.md
```

---

## Tech Stack

### Backend & AI Pipeline
| Purpose | Library |
|---|---|
| REST API | `fastapi`, `uvicorn` |
| Video → audio extraction | `moviepy` |
| Audio loading & validation | `torchaudio`, `soundfile` |
| Speech-to-text | `openai-whisper` (local) |
| Summarization | `transformers` — distilbart-cnn-12-6 |
| Embeddings for search | `sentence-transformers` — all-MiniLM-L6-v2 |
| Vector store | `faiss-cpu` |
| PDF export | `fpdf2` |
| ML backend | `torch`, `torchaudio` |

### Frontend
| Purpose | Tech |
|---|---|
| UI | HTML5, CSS3, Vanilla JS |
| File upload | Fetch API, multipart/form-data |
| Search interface | Fetch API + dynamic rendering |

### Model sizes at runtime
| Model | Size | Purpose |
|---|---|---|
| Whisper tiny | ~75MB | Fast — short/clear audio |
| Whisper base | ~140MB | Default — recommended |
| Whisper small | ~460MB | Better — noisy/accented audio |
| distilbart-cnn-12-6 | ~300MB | Summarization |
| all-MiniLM-L6-v2 | ~80MB | Semantic search embeddings |

> Total memory footprint: ~1GB models + processing. Comfortable on 8GB RAM.

---

## Setup & Installation

### Prerequisites
- Python 3.11.9
- Git
- ~3GB free disk space (model downloads on first run)
- FFmpeg — required by moviepy for video processing

### Install FFmpeg (Windows)
```bash
# Using winget
winget install ffmpeg

# Or download from https://ffmpeg.org/download.html
# and add to PATH manually
```

Verify:
```bash
ffmpeg -version
```

### Steps

```bash
# 1. Clone
git clone https://github.com/HaitaanshDixit/MOMentum.git
cd MOMentum

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install setuptools
pip install openai-whisper --no-build-isolation
pip install -r requirements.txt

# 4. NLTK data
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger

# 5. Configure environment
cp .env.example .env
```

### Environment variables (`.env`)
```env
WHISPER_MODEL=base        # tiny | base | small
OUTPUT_DIR=output
SAMPLES_DIR=samples
MAX_FILE_SIZE_MB=500
```

---

## Usage

### Web Interface
Visit the live URL, drag and drop your audio or video file, select export format, download your MOM.

### CLI
```bash
# Audio file
python momentum.py --input meeting.mp3 --format md

# Video file
python momentum.py --input meeting.mp4 --format pdf

# Semantic search
python momentum.py --search "What was decided about the budget?"
```

### API
```bash
uvicorn app:app --reload
# Swagger docs at http://localhost:8000/docs
```

---

## API Reference

### `POST /api/upload`
Upload audio or video, receive MOM document.

**Request**
```
Content-Type: multipart/form-data
file    : <file>      # mp4, mkv, mov, avi, webm, mp3, wav, m4a, flac, ogg
format  : "md"        # txt | md | pdf  (default: md)
```

**Response**
```json
{
  "status": "success",
  "meeting_id": "mtg_20250610_001",
  "filename": "MOM_2025-06-10.md",
  "duration": "45:32",
  "language": "en",
  "whisper_model_used": "base",
  "review_passes": 2,
  "download_url": "/api/download/MOM_2025-06-10.md",
  "preview": "## Meeting Summary\n..."
}
```

### `GET /api/search?q={query}`
Semantic search over all past meeting transcripts.

**Response**
```json
{
  "query": "budget decision",
  "results": [
    {
      "meeting_id": "mtg_20250605_001",
      "date": "2025-06-05",
      "score": 0.91,
      "excerpt": "The team decided to increase Q3 budget by 15%...",
      "download_url": "/api/download/MOM_2025-06-05.md"
    }
  ]
}
```

### `GET /api/health`
```json
{ "status": "ok", "whisper_model": "base", "meetings_indexed": 12 }
```

---

## Output Format

```
========================================
         MINUTES OF MEETING
========================================
Date            : 10 June 2025
Duration        : 45:32
Language        : English
Generated by    : MOMentum

AGENDA
------
- Project status update
- Q3 planning and budget discussion
- Resource allocation review

DISCUSSION SUMMARY
------------------
The team reviewed Q3 milestones and sprint velocity.
One raised concerns about resource allocation across
two parallel workstreams...

ACTION ITEMS
------------
Send updated project timeline       (by Friday)
Schedule follow-up with design team (by Monday)
Review and approve budget proposal  (by EOD)

DECISIONS MADE
--------------
- Q3 deadline extended by two weeks
- Budget increased by 15% for Q3
- Design review moved to biweekly cadence

NEXT STEPS
----------
Follow-up meeting to be scheduled on x date.
========================================
Generated by MOMentum
https://github.com/HaitaanshDixit/MOMentum
```

---

## Deployment

The app deploys as a single FastAPI service on **Render** or **Railway** free tier.

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
```

Full deployment guide coming with the final release.

---

## License

MIT License — free to use, modify, and distribute.
