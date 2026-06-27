"""
Summarization module for MOMentum.
Takes raw transcript text and extracts: Meeting summary, Action items, Decisions made, Next steps

Deployment modes:
- DEPLOYMENT_MODE=full  → uses distilbart AI model for overview (local, needs ~600MB RAM)
- DEPLOYMENT_MODE=lite  → uses extractive summarization only (deployed, lightweight)

Set DEPLOYMENT_MODE=lite in Render environment variables.
"""

import os
import re
from dataclasses import dataclass, field

DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "full")

if DEPLOYMENT_MODE == "full":
    from transformers import pipeline

ACTION_KEYWORDS = [
    "will", "shall", "must", "need to", "needs to", "have to", "has to",
    "going to", "should", "action", "task", "todo", "to do", "to-do",
    "follow up", "follow-up", "assign", "assigned", "responsible",
    "deadline", "by monday", "by tuesday", "by wednesday", "by thursday",
    "by friday", "by eod", "by end of", "next week", "asap",
    "please", "make sure", "ensure", "complete", "finish", "send",
    "schedule", "arrange", "prepare", "review", "submit", "update",
    "create", "build", "implement", "fix", "resolve", "check",
]

DECISION_KEYWORDS = [
    "decided", "agreed", "confirmed", "approved", "rejected", "concluded",
    "resolution", "resolved", "finalized", "finalised", "accepted",
    "voted", "consensus", "agreed upon", "agreed to", "will proceed",
    "moving forward", "going ahead", "we will", "team will",
    "it was decided", "it was agreed", "decision was", "decision is",
]

NEXT_STEPS_KEYWORDS = [
    "next meeting", "next time", "next session", "follow up", "follow-up",
    "next steps", "going forward", "moving forward", "after this",
    "subsequently", "thereafter", "upcoming", "future", "plan to",
    "schedule next", "reconvene", "circle back",
]


@dataclass
class MeetingSummary:
    overview: str                         
    action_items: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    word_count_original: int = 0
    word_count_summary: int = 0

    def __str__(self):
        sep = "=" * 40

        def fmt_list(items):
            if not items:
                return "  None identified"
            return "\n".join(f"  • {item}" for item in items)

        return (
            f"\n{sep}\n"
            f"  Meeting Summary\n"
            f"{sep}\n"
            f"  OVERVIEW\n"
            f"  {self.overview}\n\n"
            f"  ACTION ITEMS\n"
            f"{fmt_list(self.action_items)}\n\n"
            f"  DECISIONS MADE\n"
            f"{fmt_list(self.decisions)}\n\n"
            f"  NEXT STEPS\n"
            f"{fmt_list(self.next_steps)}\n"
            f"{sep}\n"
            f"  Original: {self.word_count_original} words → "
            f"Summary: {self.word_count_summary} words\n"
            f"{sep}"
        )


def _load_summarizer():
    print("  Loading summarization model (distilbart-cnn-12-6) ...")
    print("  (First run downloads ~300MB — cached after that)")
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        tokenizer="sshleifer/distilbart-cnn-12-6",
    )
    print("  Summarization model loaded.")
    return summarizer


def _chunk_text(text: str, max_words: int = 900) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


def _summarize_text(text: str, summarizer) -> str:
    word_count = len(text.split())

    if word_count <= 900:
        result = summarizer(
            text,
            max_length=min(180, max(50, len(text.split()) // 2)),
            min_length=30,
            do_sample=False,
        )
        return result[0]["summary_text"].strip()

    print(f"  Long transcript ({word_count} words) — processing in chunks ...")
    chunks = _chunk_text(text, max_words=900)
    chunk_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"  Summarizing chunk {i + 1}/{len(chunks)} ...")
        result = summarizer(
            chunk,
            max_length=150,
            min_length=40,
            do_sample=False,
        )
        chunk_summaries.append(result[0]["summary_text"].strip())

    if len(chunk_summaries) > 1:
        combined = " ".join(chunk_summaries)
        if len(combined.split()) > 900:
            combined = " ".join(combined.split()[:900])
        result = summarizer(
            combined,
            max_length=200,
            min_length=80,
            do_sample=False,
        )
        return result[0]["summary_text"].strip()

    return chunk_summaries[0]


def _extractive_summary(text: str, num_sentences: int = 5) -> str:
    """
    Pick the most information-dense sentences from the transcript.
    Used in lite/deployment mode — no AI model needed.
    """
    info_keywords = [
        "project", "team", "meeting", "discuss", "update", "plan",
        "budget", "deadline", "decision", "agree", "confirm", "review",
        "status", "progress", "issue", "problem", "solution", "result",
        "will", "should", "must", "need", "action", "next", "follow",
    ]

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 5]

    if not sentences:
        return text[:300] + "..."

    scored = []
    for s in sentences:
        score = sum(1 for kw in info_keywords if kw in s.lower())
        scored.append((score, s))

    scored.sort(reverse=True)
    top = [s for _, s in scored[:num_sentences]]

    original_order = [s for s in sentences if s in top]
    return " ".join(original_order) if original_order else " ".join(top)


def _extract_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _contains_keyword(sentence: str, keywords: list[str]) -> bool:
    sentence_lower = sentence.lower()
    return any(kw in sentence_lower for kw in keywords)


def _clean_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence.endswith((".","!","?")):
        sentence += "."
    return sentence[0].upper() + sentence[1:]


def _extract_action_items(sentences: list[str]) -> list[str]:
    action_items = []
    seen = set()
    for sentence in sentences:
        if _contains_keyword(sentence, ACTION_KEYWORDS):
            cleaned = _clean_sentence(sentence)
            if cleaned not in seen and len(cleaned.split()) >= 4:
                seen.add(cleaned)
                action_items.append(cleaned)
    return action_items[:10]


def _extract_decisions(sentences: list[str]) -> list[str]:
    decisions = []
    seen = set()
    for sentence in sentences:
        if _contains_keyword(sentence, DECISION_KEYWORDS):
            cleaned = _clean_sentence(sentence)
            if cleaned not in seen and len(cleaned.split()) >= 4:
                seen.add(cleaned)
                decisions.append(cleaned)
    return decisions[:8]


def _extract_next_steps(sentences: list[str]) -> list[str]:
    next_steps = []
    seen = set()
    for sentence in sentences:
        if _contains_keyword(sentence, NEXT_STEPS_KEYWORDS):
            cleaned = _clean_sentence(sentence)
            if cleaned not in seen and len(cleaned.split()) >= 4:
                seen.add(cleaned)
                next_steps.append(cleaned)
    return next_steps[:6]


def summarize(transcript_text: str) -> MeetingSummary:
    
    if not transcript_text or not transcript_text.strip():
        raise ValueError("Transcript text is empty — nothing to summarize.")

    word_count_original = len(transcript_text.split())
    print(f"\n{'='*40}")
    print(f"  MOMentum Summarizer")
    print(f"{'='*40}")
    print(f"  Transcript: {word_count_original} words")
    print(f"  Mode: {DEPLOYMENT_MODE.upper()}")

    if DEPLOYMENT_MODE == "lite":
        print("  Generating extractive overview (lite mode) ...")
        overview = _extractive_summary(transcript_text)
    else:
        summarizer = _load_summarizer()
        print("  Generating meeting overview ...")
        try:
            overview = _summarize_text(transcript_text, summarizer)
        except Exception as e:
            raise RuntimeError(f"Summarization failed: {e}")

    word_count_summary = len(overview.split())
    print(f"  Overview generated — {word_count_summary} words.")

    print("  Extracting action items, decisions, and next steps ...")
    sentences = _extract_sentences(transcript_text)

    action_items = _extract_action_items(sentences)
    decisions = _extract_decisions(sentences)
    next_steps = _extract_next_steps(sentences)

    print(f"  Found: {len(action_items)} action items, "
          f"{len(decisions)} decisions, "
          f"{len(next_steps)} next steps.")

    summary = MeetingSummary(
        overview=overview,
        action_items=action_items,
        decisions=decisions,
        next_steps=next_steps,
        word_count_original=word_count_original,
        word_count_summary=word_count_summary,
    )

    print(f"\n  Summarization complete:{summary}")
    return summary


def summarize_from_transcript(transcript) -> MeetingSummary:
    return summarize(transcript.full_text)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/summarizer.py <path_to_transcript.txt>")
        print("       Pass a plain text transcript file to summarize.")
        sys.exit(1)

    transcript_file = sys.argv[1]

    try:
        with open(transcript_file, "r", encoding="utf-8") as f:
            text = f.read()

        if "FULL TEXT" in text:
            text = text.split("FULL TEXT")[-1].strip()
            text = text.replace("=" * 40, "").strip()

        result = summarize(text)
        print(result)

    except FileNotFoundError:
        print(f"Error: File not found — '{transcript_file}'")
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)