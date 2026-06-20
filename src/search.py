import os
from dataclasses import dataclass, field

import faiss
import numpy as np

from embedder import (
    MeetingEmbedding,
    _load_model,
    load_all_embeddings,
    VECTOR_STORE_DIR,
)


@dataclass
class SearchResult:
    """A single search result from FAISS semantic search."""
    meeting_id: str
    date: str
    audio_file: str
    duration: str
    score: float              # similarity score 0-1 (higher = more relevant)
    transcript_preview: str
    mom_path: str
    word_count: int

    def __str__(self):
        return (
            f"\n  [{self.date}] {self.audio_file} — {self.duration}\n"
            f"  Score    : {self.score:.2f}\n"
            f"  Preview  : {self.transcript_preview[:120]}...\n"
            f"  MOM file : {self.mom_path}"
        )

def _build_faiss_index(
    embeddings: list[MeetingEmbedding],
) -> tuple[faiss.Index, list[MeetingEmbedding]]:
    
    if not embeddings:
        raise ValueError("No embeddings to index — process some meetings first.")

    dim = embeddings[0].embedding.shape[0]

    # Normalize vectors for cosine similarity
    vectors = np.array([e.embedding for e in embeddings], dtype=np.float32)
    faiss.normalize_L2(vectors)

    # Flat index — exact search, best for small-medium collections
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    return index, embeddings


def search(
    query: str,
    top_k: int = 3,
    store_dir: str = VECTOR_STORE_DIR,
    min_score: float = 0.3,
) -> list[SearchResult]:
    
    # Load all embeddings from disk
    embeddings = load_all_embeddings(store_dir)

    if not embeddings:
        raise ValueError(
            "No meetings indexed yet.\n"
            "Run momentum.py to process meetings and build the search index."
        )

    print(f"\n{'='*44}")
    print(f"  MOMentum Semantic Search")
    print(f"{'='*44}")
    print(f"  Query    : {query}")
    print(f"  Indexed  : {len(embeddings)} meeting(s)")

    index, ordered_embeddings = _build_faiss_index(embeddings)

    model = _load_model()
    query_vector = model.encode([query], show_progress_bar=False)
    query_vector = np.array(query_vector, dtype=np.float32)
    faiss.normalize_L2(query_vector)

    k = min(top_k, len(embeddings))
    scores, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or score < min_score:
            continue
        emb = ordered_embeddings[idx]
        results.append(SearchResult(
            meeting_id=emb.meeting_id,
            date=emb.date,
            audio_file=emb.audio_file,
            duration=emb.duration,
            score=float(score),
            transcript_preview=emb.transcript_preview,
            mom_path=emb.mom_path,
            word_count=emb.word_count,
        ))

    print(f"  Results  : {len(results)} match(es) found\n")
    return results


def search_and_display(
    query: str,
    top_k: int = 3,
    store_dir: str = VECTOR_STORE_DIR,
) -> list[SearchResult]:
    
    try:
        results = search(query, top_k=top_k, store_dir=store_dir)
    except ValueError as e:
        print(f"\n  {e}")
        return []

    if not results:
        print(f"  No relevant meetings found for: '{query}'")
        print(f"  Try a different query or process more meetings.")
        return []

    sep = "=" * 44
    print(f"{sep}")
    print(f"  Search Results for: '{query}'")
    print(f"{sep}")

    for i, result in enumerate(results, 1):
        print(f"\n  Result {i} — Score: {result.score:.2f}")
        print(f"  Date     : {result.date}")
        print(f"  File     : {result.audio_file}")
        print(f"  Duration : {result.duration}")
        print(f"  Preview  : {result.transcript_preview[:150]}")
        if result.mom_path and os.path.exists(result.mom_path):
            print(f"  MOM      : {result.mom_path}")

    print(f"\n{sep}")
    return results

def index_meeting(
    embedding: MeetingEmbedding,
    store_dir: str = VECTOR_STORE_DIR,
) -> None:
    
    from embedder import save_embedding
    save_embedding(embedding, store_dir)
    print(f"  Meeting indexed: {embedding.meeting_id}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/search.py <query>")
        print('Example: python src/search.py "What was decided about the budget?"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    try:
        results = search_and_display(query)
        if not results:
            sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)