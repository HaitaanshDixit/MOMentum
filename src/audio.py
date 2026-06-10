"""
audio.py
Audio ingestion and validation module for MOMentum.
Handles loading, validating, and extracting metadata from meeting audio files.
Video files must be preprocessed via preprocessor.extract_audio() before passing here.
"""

import os
from dataclasses import dataclass
from pathlib import Path
import soundfile as sf
import torchaudio


SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
VIDEO_FORMATS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
ALL_SUPPORTED = SUPPORTED_FORMATS | VIDEO_FORMATS


@dataclass
class AudioFile:
    """Represents a loaded and validated audio file with its metadata."""
    file_path: str
    file_name: str
    format: str
    duration_seconds: float
    duration_formatted: str
    sample_rate: int
    channels: int
    file_size_mb: float

    def __str__(self):
        return (
            f"\n{'='*40}\n"
            f"  File       : {self.file_name}\n"
            f"  Format     : {self.format.upper()}\n"
            f"  Duration   : {self.duration_formatted}\n"
            f"  Sample Rate: {self.sample_rate} Hz\n"
            f"  Channels   : {self.channels} "
            f"({'Stereo' if self.channels == 2 else 'Mono'})\n"
            f"  Size       : {self.file_size_mb:.2f} MB\n"
            f"{'='*40}"
        )


def _format_duration(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _get_file_size_mb(file_path: str) -> float:
    """Return file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)


def _validate_file(file_path: str) -> Path:
    """
    Validate that the file exists and is a supported audio format.
    Returns a Path object if valid, raises an error otherwise.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: '{file_path}'\n"
            f"Please check the file path and try again."
        )

    if not path.is_file():
        raise ValueError(
            f"'{file_path}' is not a file. Please provide a valid file path."
        )

    suffix = path.suffix.lower()

    # Video files are supported by the pipeline but must go through
    # preprocessor.extract_audio() before reaching this module
    if suffix in VIDEO_FORMATS:
        raise ValueError(
            f"'{path.name}' is a video file.\n"
            f"Please use preprocessor.extract_audio() first, "
            f"then pass the extracted audio to load_audio()."
        )

    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: '{suffix}'\n"
            f"Supported audio : {', '.join(sorted(SUPPORTED_FORMATS))}\n"
            f"Supported video : {', '.join(sorted(VIDEO_FORMATS))} — extract audio first"
        )

    return path


def load_audio(file_path: str) -> AudioFile:
    """
    Load and validate an audio file, returning an AudioFile dataclass.
    For video files, use preprocessor.extract_audio() first.

    Args:
        file_path: Path to the audio file (.mp3, .wav, .flac, .m4a, .ogg)

    Returns:
        AudioFile: Dataclass containing file metadata

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file is a video or unsupported format
        RuntimeError: If the file cannot be read or decoded
    """
    path = _validate_file(file_path)

    try:
        # torchaudio handles mp3, m4a, wav, flac reliably
        info = torchaudio.info(str(path))
        sample_rate = info.sample_rate
        channels = info.num_channels
        num_frames = info.num_frames
        duration_seconds = num_frames / sample_rate

    except Exception as e:
        # Fallback to soundfile for wav/flac if torchaudio fails
        try:
            with sf.SoundFile(str(path)) as f:
                sample_rate = f.samplerate
                channels = f.channels
                duration_seconds = len(f) / sample_rate
        except Exception:
            raise RuntimeError(
                f"Could not read audio file: '{file_path}'\n"
                f"The file may be corrupted or in an unsupported codec.\n"
                f"Original error: {e}"
            )

    audio_file = AudioFile(
        file_path=str(path.resolve()),
        file_name=path.name,
        format=path.suffix.lower().lstrip("."),
        duration_seconds=round(duration_seconds, 2),
        duration_formatted=_format_duration(duration_seconds),
        sample_rate=sample_rate,
        channels=channels,
        file_size_mb=round(_get_file_size_mb(str(path)), 2),
    )

    return audio_file


if __name__ == "__main__":
    # Quick test — run with: python src/audio.py <path_to_audio>
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/audio.py <path_to_audio_file>")
        sys.exit(1)

    try:
        audio = load_audio(sys.argv[1])
        print(f"Audio loaded successfully:{audio}")
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")
        sys.exit(1)