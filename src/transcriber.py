"""
Transcription module for MOMentum. Uses OpenAI Whisper (local) to transcribe audio files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torchaudio
import whisper

@dataclass
class TranscriptSegment:
    """A single timestamped segment of the transcript."""
    start: float    
    end: float        
    text: str

    def __str__(self):
        return f"[{_format_time(self.start)} → {_format_time(self.end)}] {self.text.strip()}"


@dataclass
class TranscriptResult:
    audio_path: str
    model_used: str
    language: str
    duration_seconds: float
    full_text: str
    segments: list[TranscriptSegment] = field(default_factory=list)
    word_count: int = 0

    def __str__(self):
        sep = "=" * 40
        segments_preview = "\n".join(str(s) for s in self.segments[:5])
        more = (f"\n  ... and {len(self.segments) - 5} more segments"
                if len(self.segments) > 5 else "")
        return (
            f"\n{sep}\n"
            f"  Transcription Result\n"
            f"{sep}\n"
            f"  Audio    : {Path(self.audio_path).name}\n"
            f"  Model    : {self.model_used}\n"
            f"  Language : {self.language}\n"
            f"  Duration : {_format_time(self.duration_seconds)}\n"
            f"  Words    : {self.word_count}\n"
            f"  Segments : {len(self.segments)}\n"
            f"{sep}\n"
            f"  Preview (first 5 segments):\n"
            f"{segments_preview}{more}\n"
            f"{sep}"
        )


def _format_time(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _load_whisper_model(model_name: str) -> whisper.Whisper:
    """Load Whisper model — downloads on first use, cached after that."""
    print(f"  Loading Whisper model: '{model_name}' ...")
    print(f"  (First run downloads ~140MB for 'base' — cached after that)")
    model = whisper.load_model(model_name)
    print(f"  Whisper '{model_name}' loaded successfully.")
    return model


def _build_segments(raw_segments: list[dict]) -> list[TranscriptSegment]:

    segments = []
    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append(TranscriptSegment(
            start=seg["start"],
            end=seg["end"],
            text=text,
        ))
    return segments


def transcribe(audio_path: str, model_name: str = "base",
               language: str = None) -> TranscriptResult:
    
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: '{audio_path}'")

    model = _load_whisper_model(model_name)

    try:
        info = torchaudio.info(str(path))
        duration_seconds = info.num_frames / info.sample_rate
    except Exception:
        duration_seconds = 0.0

    options = {
        "verbose": False,
        "task": "transcribe",
        "fp16": torch.cuda.is_available(),  # fp16 only on GPU, CPU uses fp32
    }
    if language:
        options["language"] = language

    print(f"\n  Starting transcription ...")
    print(f"  File     : {path.name}")
    print(f"  Model    : {model_name}")
    print(f"  Language : {language or 'auto-detect'}")
    print(f"  Device   : {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  (Roughly 1x–3x audio duration on CPU — please wait)\n")

    try:
        result = model.transcribe(str(path), **options)
    except Exception as e:
        raise RuntimeError(
            f"Transcription failed for '{path.name}'.\n"
            f"Error: {e}"
        )

    full_text = result.get("text", "").strip()
    raw_segments = result.get("segments", [])
    detected_language = result.get("language", language or "en")

    segments = _build_segments(raw_segments)
    word_count = len(full_text.split())

    transcript = TranscriptResult(
        audio_path=str(path.resolve()),
        model_used=model_name,
        language=detected_language,
        duration_seconds=round(duration_seconds, 2),
        full_text=full_text,
        segments=segments,
        word_count=word_count,
    )

    print(f"  Transcription complete — {word_count} words, {len(segments)} segments.")
    return transcript


def transcribe_from_profile(profile) -> TranscriptResult:
    
    audio_path = profile.extracted_audio_path or profile.audio_file.file_path

    return transcribe(
        audio_path=audio_path,
        model_name=profile.recommended_model,
        language=profile.language_code,
    )


def save_transcript(transcript: TranscriptResult, output_path: str) -> str:
    
    os.makedirs(Path(output_path).parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("TRANSCRIPT\n")
        f.write(f"{'='*40}\n")
        f.write(f"Audio    : {Path(transcript.audio_path).name}\n")
        f.write(f"Language : {transcript.language}\n")
        f.write(f"Duration : {_format_time(transcript.duration_seconds)}\n")
        f.write(f"Words    : {transcript.word_count}\n")
        f.write(f"{'='*40}\n\n")

        for seg in transcript.segments:
            f.write(f"{str(seg)}\n")

        f.write(f"\n{'='*40}\n")
        f.write("FULL TEXT\n")
        f.write(f"{'='*40}\n\n")
        f.write(transcript.full_text)

    print(f"  Transcript saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/transcriber.py <path_to_audio> [model]")
        print("       model: tiny | base | small | medium  (default: base)")
        sys.exit(1)

    audio_file = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "base"

    try:
        result = transcribe(audio_file, model_name=model)
        print(result)

        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, Path(audio_file).stem + "_transcript.txt")
        save_transcript(result, out_file)

    except (FileNotFoundError, RuntimeError) as e:
        print(f"\nError: {e}")
        sys.exit(1)