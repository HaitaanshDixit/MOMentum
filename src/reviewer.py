"""
Review Agent for MOMentum. Takes a draft MeetingSummary, runs quality checks, identifies issues, refines the output iteratively, and returns an improved MeetingSummary.
(Loops up to MAX_REVIEW_PASSES times until quality passes or limit is reached)
"""

import re
from dataclasses import dataclass, field
from summarizer import MeetingSummary

MAX_REVIEW_PASSES = 3
MIN_OVERVIEW_WORDS = 30
MIN_ACTION_ITEMS = 1
MIN_DECISIONS = 1

@dataclass
class QualityReport:
    """Result of a single quality check pass."""
    pass_number: int
    issues_found: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    passed: bool = False
    score: int = 0          # 0-100
    max_score: int = 100

    def __str__(self):
        status = "✓ PASSED" if self.passed else "✗ NEEDS REFINEMENT"
        issues = "\n    ".join(self.issues_found) if self.issues_found else "None"
        suggestions = "\n    ".join(self.suggestions) if self.suggestions else "None"
        return (
            f"\n  Pass {self.pass_number} Quality Report — {status}\n"
            f"  Score     : {self.score}/{self.max_score}\n"
            f"  Issues    : {issues}\n"
            f"  Suggestions: {suggestions}"
        )


@dataclass
class ReviewResult:
    """Final result returned by the review agent."""
    original_summary: MeetingSummary
    refined_summary: MeetingSummary
    passes_taken: int
    quality_reports: list[QualityReport] = field(default_factory=list)
    final_score: int = 0
    accepted: bool = False

    def __str__(self):
        sep = "=" * 40
        reports = "\n".join(str(r) for r in self.quality_reports)
        return (
            f"\n{sep}\n"
            f"  Review Agent Report\n"
            f"{sep}\n"
            f"  Passes taken  : {self.passes_taken}/{MAX_REVIEW_PASSES}\n"
            f"  Final score   : {self.final_score}/100\n"
            f"  Accepted      : {'Yes' if self.accepted else 'No — max passes reached'}\n"
            f"{sep}\n"
            f"  Quality Reports:\n"
            f"{reports}\n"
            f"{sep}"
        )


def _check_overview(summary: MeetingSummary, report: QualityReport) -> int:
    
    score = 0

    if not summary.overview or not summary.overview.strip():
        report.issues_found.append("Overview is empty.")
        report.suggestions.append("Generate a meeting overview from the transcript.")
        return 0

    word_count = len(summary.overview.split())

    if word_count < MIN_OVERVIEW_WORDS:
        report.issues_found.append(
            f"Overview too short ({word_count} words — minimum {MIN_OVERVIEW_WORDS})."
        )
        report.suggestions.append("Expand the overview with more detail from the transcript.")
        score += 10
    else:
        score += 30

    filler_phrases = [
        "in this meeting", "the meeting discussed", "it was mentioned",
        "various topics", "several things", "a number of"
    ]
    overview_lower = summary.overview.lower()
    fillers_found = [p for p in filler_phrases if p in overview_lower]
    if fillers_found:
        report.issues_found.append(f"Overview contains vague filler phrases: {fillers_found}")
        report.suggestions.append("Replace filler phrases with specific details.")
        score -= 5

    return max(0, score)


def _check_action_items(summary: MeetingSummary, report: QualityReport) -> int:

    score = 0

    if not summary.action_items:
        report.issues_found.append("No action items found.")
        report.suggestions.append(
            "Look for sentences with: will, should, must, need to, "
            "schedule, send, review, prepare, follow up."
        )
        return 0

    score += 20

    vague_keywords = ["something", "things", "stuff", "it", "this", "that"]
    vague_items = []
    for item in summary.action_items:
        if any(vk in item.lower().split() for vk in vague_keywords):
            vague_items.append(item[:50])

    if vague_items:
        report.issues_found.append(f"Vague action items detected: {vague_items}")
        report.suggestions.append("Make action items more specific with clear owners and tasks.")
        score += 5
    else:
        score += 10

    return score


def _check_decisions(summary: MeetingSummary, report: QualityReport) -> int:
    if not summary.decisions:
        report.issues_found.append("No decisions found.")
        report.suggestions.append(
            "Look for sentences with: decided, agreed, confirmed, "
            "approved, resolved, concluded, finalized."
        )
        return 0

    return 25


def _check_next_steps(summary: MeetingSummary, report: QualityReport) -> int:
    if not summary.next_steps:
        report.issues_found.append("No next steps found.")
        report.suggestions.append(
            "Look for forward-looking sentences with: next meeting, "
            "follow up, going forward, upcoming, plan to."
        )
        return 0

    return 15


def _run_quality_check(summary: MeetingSummary, pass_number: int) -> QualityReport:
    """
    Args:
        summary     : MeetingSummary to check
        pass_number : Current pass number (1, 2, 3)

    Returns:
        QualityReport with issues, suggestions, and score
    """
    report = QualityReport(pass_number=pass_number)

    # Run all checks and accumulate score
    score = 0
    score += _check_overview(summary, report)
    score += _check_action_items(summary, report)
    score += _check_decisions(summary, report)
    score += _check_next_steps(summary, report)

    report.score = min(100, score)
    report.passed = len(report.issues_found) == 0 and report.score >= 70

    return report


def _refine_action_items(
    summary: MeetingSummary,
    transcript_text: str,
    issues: list[str]
) -> list[str]:
    """
    Attempt to find more action items from transcript if none were found
    or existing ones are vague.
    Uses broader keyword matching on retry.
    """
    if summary.action_items and "No action items" not in " ".join(issues):
        return summary.action_items

    # Broader search on retry
    broader_keywords = [
        "will", "need", "should", "must", "going to", "plan",
        "want to", "try to", "hope to", "expect to", "intend to",
        "supposed to", "required to", "asked to", "told to",
    ]

    sentences = re.split(r'(?<=[.!?])\s+', transcript_text.strip())
    found = []
    seen = set()

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(kw in sentence_lower for kw in broader_keywords):
            cleaned = sentence.strip()
            if cleaned and cleaned not in seen and len(cleaned.split()) >= 4:
                seen.add(cleaned)
                found.append(cleaned)

    return found[:10] if found else summary.action_items


def _refine_decisions(
    summary: MeetingSummary,
    transcript_text: str,
    issues: list[str]
) -> list[str]:
    """
    Attempt to find more decisions from transcript if none were found.
    Uses broader keyword matching on retry.
    """
    if summary.decisions and "No decisions" not in " ".join(issues):
        return summary.decisions

    broader_keywords = [
        "decided", "agreed", "confirmed", "approved", "rejected",
        "will proceed", "going ahead", "moving forward", "we will",
        "the team will", "everyone agreed", "all agreed",
        "unanimous", "confirmed that", "it was", "has been",
    ]

    sentences = re.split(r'(?<=[.!?])\s+', transcript_text.strip())
    found = []
    seen = set()

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(kw in sentence_lower for kw in broader_keywords):
            cleaned = sentence.strip()
            if cleaned and cleaned not in seen and len(cleaned.split()) >= 4:
                seen.add(cleaned)
                found.append(cleaned)

    return found[:8] if found else summary.decisions


def _refine_next_steps(
    summary: MeetingSummary,
    transcript_text: str,
    issues: list[str]
) -> list[str]:
    """
    Attempt to find more next steps from transcript if none were found.
    """
    if summary.next_steps and "No next steps" not in " ".join(issues):
        return summary.next_steps

    broader_keywords = [
        "next", "follow", "upcoming", "future", "later", "after",
        "subsequently", "schedule", "plan", "reconvene", "meet again",
        "touch base", "check in", "circle back", "catch up",
    ]

    sentences = re.split(r'(?<=[.!?])\s+', transcript_text.strip())
    found = []
    seen = set()

    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(kw in sentence_lower for kw in broader_keywords):
            cleaned = sentence.strip()
            if cleaned and cleaned not in seen and len(cleaned.split()) >= 4:
                seen.add(cleaned)
                found.append(cleaned)

    return found[:6] if found else summary.next_steps


def _refine_overview(summary: MeetingSummary, transcript_text: str) -> str:
    """
    If overview is too short or empty, fall back to extracting
    the most informative sentences from the transcript directly.
    """
    if (summary.overview and
            len(summary.overview.split()) >= MIN_OVERVIEW_WORDS):
        return summary.overview

    # Extractive fallback — pick sentences with highest information density
    sentences = re.split(r'(?<=[.!?])\s+', transcript_text.strip())

    # Score sentences by length and keyword richness
    info_keywords = [
        "project", "team", "meeting", "discuss", "update", "plan",
        "budget", "deadline", "decision", "agree", "confirm", "review",
        "status", "progress", "issue", "problem", "solution", "result",
    ]

    scored = []
    for s in sentences:
        if len(s.split()) < 5:
            continue
        score = sum(1 for kw in info_keywords if kw in s.lower())
        scored.append((score, s))

    scored.sort(reverse=True)
    top_sentences = [s for _, s in scored[:5]]

    if top_sentences:
        return " ".join(top_sentences)

    # Last resort — return first 50 words of transcript
    return " ".join(transcript_text.split()[:50]) + "..."


def _apply_refinements(
    summary: MeetingSummary,
    report: QualityReport,
    transcript_text: str,
) -> MeetingSummary:
    """
    Apply targeted refinements based on issues found in quality report.

    Args:
        summary         : Current MeetingSummary
        report          : QualityReport from current pass
        transcript_text : Original transcript to re-mine for content

    Returns:
        MeetingSummary: Refined summary
    """
    issues = report.issues_found

    refined_overview = _refine_overview(summary, transcript_text)
    refined_action_items = _refine_action_items(summary, transcript_text, issues)
    refined_decisions = _refine_decisions(summary, transcript_text, issues)
    refined_next_steps = _refine_next_steps(summary, transcript_text, issues)

    return MeetingSummary(
        overview=refined_overview,
        action_items=refined_action_items,
        decisions=refined_decisions,
        next_steps=refined_next_steps,
        word_count_original=summary.word_count_original,
        word_count_summary=len(refined_overview.split()),
    )



def review(
    summary: MeetingSummary,
    transcript_text: str,
) -> ReviewResult:
    """
    Run the review agent on a draft MeetingSummary.
    Checks quality, refines iteratively, and returns the best version.

    This is the agentic loop:
        observe (quality check) → reason (identify issues)
        → act (refine) → repeat until passed or max passes reached

    Args:
        summary         : Draft MeetingSummary from summarizer.py
        transcript_text : Original transcript text for re-mining

    Returns:
        ReviewResult: Contains refined summary, quality reports, and pass count
    """
    print(f"\n{'='*40}")
    print(f"  MOMentum Review Agent")
    print(f"{'='*40}")
    print(f"  Starting quality review (max {MAX_REVIEW_PASSES} passes) ...")

    current_summary = summary
    quality_reports = []
    passes_taken = 0

    for pass_num in range(1, MAX_REVIEW_PASSES + 1):
        passes_taken = pass_num
        print(f"\n  --- Pass {pass_num}/{MAX_REVIEW_PASSES} ---")

        # Observe — run quality checks
        report = _run_quality_check(current_summary, pass_num)
        quality_reports.append(report)

        print(f"  Score: {report.score}/100")
        if report.issues_found:
            print(f"  Issues: {', '.join(report.issues_found)}")

        if report.passed:
            print(f"  Quality check passed on pass {pass_num}!")
            break

        # Act — apply refinements if not on last pass
        if pass_num < MAX_REVIEW_PASSES:
            print(f"  Applying refinements ...")
            current_summary = _apply_refinements(
                current_summary, report, transcript_text
            )
        else:
            print(f"  Max passes reached — accepting best version.")

    final_report = quality_reports[-1]

    result = ReviewResult(
        original_summary=summary,
        refined_summary=current_summary,
        passes_taken=passes_taken,
        quality_reports=quality_reports,
        final_score=final_report.score,
        accepted=final_report.passed,
    )

    print(result)
    return result


def review_from_transcript(summary: MeetingSummary, transcript) -> ReviewResult:
    """
    Convenience wrapper — takes a TranscriptResult directly.

    Args:
        summary    : MeetingSummary from summarizer.py
        transcript : TranscriptResult from transcriber.py

    Returns:
        ReviewResult
    """
    return review(summary, transcript.full_text)


if __name__ == "__main__":
    import sys
    from summarizer import summarize

    if len(sys.argv) < 2:
        print("Usage: python src/reviewer.py <path_to_transcript.txt>")
        sys.exit(1)

    transcript_file = sys.argv[1]

    try:
        with open(transcript_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Extract FULL TEXT section if it's a saved transcript file
        if "FULL TEXT" in text:
            text = text.split("FULL TEXT")[-1].strip()
            text = text.replace("=" * 40, "").strip()

        print("  Running summarizer first ...")
        draft_summary = summarize(text)

        print("\n  Running review agent ...")
        result = review(draft_summary, text)

        print("\n  Final refined summary:")
        print(result.refined_summary)

    except FileNotFoundError:
        print(f"Error: File not found — '{transcript_file}'")
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)