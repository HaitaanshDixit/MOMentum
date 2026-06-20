import argparse
import os
import sys
import time
from pathlib import Path

# Add src/ to path so all modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessor import preprocess, cleanup_extracted_audio
from transcriber import transcribe_from_profile, save_transcript
from summarizer import summarize_from_transcript
from reviewer import review
from formatter import format_mom_from_review
from exporter import export

from embedder import embed_from_pipeline
from search import index_meeting


BANNER = """
╔══════════════════════════════════════════╗
║           MOMentum  🎙️                   ║
║   Minutes of Meeting Generator           ║
║   github.com/HaitaanshDixit/MOMentum     ║
╚══════════════════════════════════════════╝
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="momentum",
        description="Generate Minutes of Meeting from audio or video recordings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python momentum.py --input meeting.mp3
  python momentum.py --input meeting.mp4 --format pdf
  python momentum.py --input meeting.mp3 --format md --output ./output
  python momentum.py --input meeting.mp3 --title "Q3 Planning Meeting"
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to audio (.mp3 .wav .m4a .flac .ogg) or video (.mp4 .mkv .mov .avi .webm) file"
    )

    parser.add_argument(
        "--format", "-f",
        choices=["txt", "md", "pdf"],
        default="md",
        help="Output format: txt, md, or pdf (default: md)"
    )

    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for generated files (default: ./output)"
    )

    parser.add_argument(
        "--title", "-t",
        default=None,
        help="Meeting title (auto-generated from date if not provided)"
    )

    parser.add_argument(
        "--model", "-m",
        choices=["tiny", "base", "small", "medium"],
        default=None,
        help="Whisper model to use — overrides auto-selection (tiny/base/small/medium)"
    )

    parser.add_argument(
        "--save-transcript",
        action="store_true",
        help="Also save the raw transcript to the output directory"
    )

    parser.add_argument(
        "--all-formats",
        action="store_true",
        help="Export MOM in all formats: txt, md, and pdf"
    )

    return parser


def run_pipeline(args) -> dict:
   
    start_time = time.time()
    exported_files = {}

    print(BANNER)

    print("[ Stage 1/6 ]  Pre-processing ...\n")
    try:
        profile = preprocess(args.input, output_dir=args.output)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n  Error in pre-processing: {e}")
        sys.exit(1)

    # Override model if specified by user
    if args.model:
        print(f"\n  Model override: '{args.model}' (was '{profile.recommended_model}')")
        profile.recommended_model = args.model

    print(f"\n[ Stage 2/6 ]  Transcribing audio ...\n")
    try:
        transcript = transcribe_from_profile(profile)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n  Error in transcription: {e}")
        cleanup_extracted_audio(profile)
        sys.exit(1)

    if args.save_transcript:
        transcript_path = os.path.join(
            args.output,
            Path(args.input).stem + "_transcript.txt"
        )
        save_transcript(transcript, transcript_path)

    if transcript.word_count == 0:
        print("\n  Warning: Transcript is empty — no speech detected in audio.")
        print("  Please check your audio file contains clear speech.")
        cleanup_extracted_audio(profile)
        sys.exit(1)

    print(f"\n[ Stage 3/6 ]  Summarizing ...\n")
    try:
        summary = summarize_from_transcript(transcript)
    except (ValueError, RuntimeError) as e:
        print(f"\n  Error in summarization: {e}")
        cleanup_extracted_audio(profile)
        sys.exit(1)

    print(f"\n[ Stage 4/6 ]  Review agent running ...\n")
    try:
        review_result = review(summary, transcript.full_text)
    except Exception as e:
        print(f"\n  Warning: Review agent failed ({e}) — using unreviewed summary.")
        from reviewer import ReviewResult
        review_result = ReviewResult(
            original_summary=summary,
            refined_summary=summary,
            passes_taken=0,
            final_score=0,
            accepted=False,
        )

    print(f"\n[ Stage 5/6 ]  Formatting and exporting ...\n")
    try:
        mom = format_mom_from_review(
            profile=profile,
            transcript=transcript,
            review_result=review_result,
            title=args.title,
        )

        if args.all_formats:
            for fmt in ["txt", "md", "pdf"]:
                path = export(mom, format=fmt, output_dir=args.output)
                exported_files[fmt] = path
        else:
            path = export(mom, format=args.format, output_dir=args.output)
            exported_files[args.format] = path

    except Exception as e:
        print(f"\n  Error in export: {e}")
        cleanup_extracted_audio(profile)
        sys.exit(1)

    print(f"\n[ Stage 6/6 ]  Indexing for semantic search ...\n")
    try:
        mom_path = list(exported_files.values())[0]
        embedding = embed_from_pipeline(
            transcript=transcript,
            profile=profile,
            mom_path=mom_path,
        )
        index_meeting(embedding)
        print(f"  Meeting indexed successfully.")
    except Exception as e:
        print(f"  Warning: Indexing failed ({e}) — search won't include this meeting.")

    cleanup_extracted_audio(profile)

    elapsed = round(time.time() - start_time, 1)
    sep = "=" * 44

    print(f"\n{sep}")
    print(f"  MOMentum — Pipeline Complete ✓")
    print(f"{sep}")
    print(f"  Audio file   : {Path(args.input).name}")
    print(f"  Duration     : {profile.audio_file.duration_formatted}")
    print(f"  Transcript   : {transcript.word_count} words")
    print(f"  Review score : {review_result.final_score}/100")
    print(f"  Time taken   : {elapsed}s")
    print(f"{sep}")
    print(f"  Exported files:")
    for fmt, path in exported_files.items():
        print(f"    [{fmt.upper()}] {path}")
    print(f"{sep}\n")

    return exported_files


#CLI
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)