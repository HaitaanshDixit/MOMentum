import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"   # ~80MB, fast, accurate for semantic search
VECTOR_STORE_DIR = "vector_store"  # where embeddings and metadata are saved
METADATA_FILE = "meetings_metadata.json"


@dataclass
class MeetingEmbedding:
    """Represents an embedded meeting with its metadata."""
    meeting_id: str
    date: str
    audio_file: str
    duration: str
    transcript_preview: str       # first 200 chars of transcript
    mom_path: str                 # path to the exported MOM file
    embedding: np.ndarray         # vector embedding of the transcript
    word_count: int

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage (excluding embedding array)."""
        return {
            "meeting_id": self.meeting_id,
            "date": self.date,
            "audio_file": self.audio_file,
            "duration": self.duration,
            "transcript_preview": self.transcript_preview,
            "mom_path": self.mom_path,
            "word_count": self.word_count,
        }


_model_cache = None

def _load_model() -> SentenceTransformer:
    """
    Load the sentence transformer model.
    Downloads ~80MB on first run, cached locally after that.
    Uses a module-level cache to avoid reloading within the same session.
    """
    global _model_cache
    if _model_cache is None:
        print(f"  Loading embedding model ({MODEL_NAME}) ...")
        print(f"  (First run downloads ~80MB — cached after that)")
        _model_cache = SentenceTransformer(MODEL_NAME)
        print(f"  Embedding model loaded.")
    return _model_cache


def embed_transcript(
    transcript_text: str,
    meeting_id: str,
    audio_file: str,
    duration: str,
    mom_path: str,
    word_count: int = 0,
) -> MeetingEmbedding:
    
    model = _load_model()

    sentences = [s.strip() for s in transcript_text.split(".") if len(s.strip()) > 10]

    if not sentences:
        sentences = [transcript_text[:500]]

    print(f"  Embedding transcript ({len(sentences)} sentences) ...")
    sentence_embeddings = model.encode(sentences, show_progress_bar=False)
    mean_embedding = np.mean(sentence_embeddings, axis=0).astype(np.float32)

    preview = transcript_text[:200].replace("\n", " ").strip()
    if len(transcript_text) > 200:
        preview += "..."

    return MeetingEmbedding(
        meeting_id=meeting_id,
        date=datetime.now().strftime("%d %B %Y"),
        audio_file=audio_file,
        duration=duration,
        transcript_preview=preview,
        mom_path=mom_path,
        embedding=mean_embedding,
        word_count=word_count or len(transcript_text.split()),
    )


def embed_from_pipeline(
    transcript,
    profile,
    mom_path: str,
    meeting_id: str = None,
) -> MeetingEmbedding:

    if not meeting_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meeting_id = f"mtg_{timestamp}"

    return embed_transcript(
        transcript_text=transcript.full_text,
        meeting_id=meeting_id,
        audio_file=profile.audio_file.file_name,
        duration=profile.audio_file.duration_formatted,
        mom_path=mom_path,
        word_count=transcript.word_count,
    )


def save_embedding(
    embedding: MeetingEmbedding,
    store_dir: str = VECTOR_STORE_DIR,
) -> str:
    
    os.makedirs(store_dir, exist_ok=True)

    # Save embedding vector
    vec_path = os.path.join(store_dir, f"{embedding.meeting_id}.npy")
    np.save(vec_path, embedding.embedding)

    # Update metadata file
    meta_path = os.path.join(store_dir, METADATA_FILE)
    metadata = _load_metadata(store_dir)
    metadata[embedding.meeting_id] = embedding.to_dict()

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Embedding saved: {vec_path}")
    return vec_path


def load_all_embeddings(
    store_dir: str = VECTOR_STORE_DIR,
) -> list[MeetingEmbedding]:
    
    if not os.path.exists(store_dir):
        return []

    metadata = _load_metadata(store_dir)
    embeddings = []

    for meeting_id, meta in metadata.items():
        vec_path = os.path.join(store_dir, f"{meeting_id}.npy")
        if not os.path.exists(vec_path):
            continue
        try:
            vector = np.load(vec_path)
            embeddings.append(MeetingEmbedding(
                meeting_id=meeting_id,
                date=meta.get("date", ""),
                audio_file=meta.get("audio_file", ""),
                duration=meta.get("duration", ""),
                transcript_preview=meta.get("transcript_preview", ""),
                mom_path=meta.get("mom_path", ""),
                embedding=vector,
                word_count=meta.get("word_count", 0),
            ))
        except Exception as e:
            print(f"  Warning: Could not load embedding for {meeting_id}: {e}")

    return embeddings


def _load_metadata(store_dir: str) -> dict:
    """Load metadata JSON file, return empty dict if not found."""
    meta_path = os.path.join(store_dir, METADATA_FILE)
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def list_meetings(store_dir: str = VECTOR_STORE_DIR) -> list[dict]:
    metadata = _load_metadata(store_dir)
    meetings = list(metadata.values())
    meetings.sort(key=lambda x: x.get("date", ""), reverse=True)
    return meetings


if __name__ == "__main__":
    import sys

    meetings = list_meetings()
    if not meetings:
        print("No meetings indexed yet.")
        print("Run momentum.py to process a meeting and index it.")
        sys.exit(0)

    print(f"\n{'='*44}")
    print(f"  Indexed Meetings ({len(meetings)} total)")
    print(f"{'='*44}")
    for m in meetings:
        print(f"  [{m['date']}] {m['audio_file']} — {m['duration']}")
        print(f"    {m['transcript_preview'][:80]}...")
        print()