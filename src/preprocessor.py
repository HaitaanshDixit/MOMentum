"""
preprocessor.py  - Rule based agent I guess
Pre-processing agent for MOMentum. (agentic part of the pipeline) which Handles video to audio extraction, audio quality analysis, language detection, speaker count estimation, and Whisper model selection.
"""

import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torchaudio
from moviepy import VideoFileClip
from audio import load_audio, AudioFile, VIDEO_FORMATS, SUPPORTED_FORMATS

WHISPER_MODELS = ["tiny", "base", "small", "medium"]

SUPPORTED_LANGUAGES = {
    "en": "English", "hi": "Hindi", "fr": "French",
    "de": "German", "es": "Spanish", "zh": "Chinese",
    "ar": "Arabic", "pt": "Portuguese", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "it": "Italian",
}

@dataclass
class AudioProfile:
    original_file: str
    audio_file: AudioFile
    is_video: bool
    extracted_audio_path: str | None  
    language_code: str
    language_name: str
    estimated_speakers: int
    audio_quality: str  # "clean", "moderate", "noisy"
    recommended_model: str # whisper model to use
    rms_level: float  # average signal level
    notes: list  # any warnings or observations

    def __str__(self):
        extracted = f"\n  Extracted Audio : {self.extracted_audio_path}" if self.is_video else ""
        notes_str = "\n  ".join(self.notes) if self.notes else "None"
        return (
            f"\n{'='*40}\n"
            f"  Source File     : {self.original_file}\n"
            f"  Input Type      : {'Video' if self.is_video else 'Audio'}"
            f"{extracted}\n"
            f"  Language        : {self.language_name} ({self.language_code})\n"
            f"  Est. Speakers   : {self.estimated_speakers}\n"
            f"  Audio Quality   : {self.audio_quality.capitalize()}\n"
            f"  RMS Level       : {self.rms_level:.4f}\n"
            f"  Whisper Model   : {self.recommended_model}\n"
            f"  Notes           : {notes_str}\n"
            f"{'='*40}"
        )

# for vids
def extract_audio(video_path: str, output_dir: str = None) -> str:
    path = Path(video_path)

    if path.suffix.lower() not in VIDEO_FORMATS:
        raise ValueError(
            f"'{path.name}' is not a supported video file.\n"
            f"Supported video formats: {', '.join(sorted(VIDEO_FORMATS))}"
        )

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: '{video_path}'")

    # Output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, path.stem + "_extracted.wav")
    else:
        tmp_dir = tempfile.mkdtemp()
        out_path = os.path.join(tmp_dir, path.stem + "_extracted.wav")

    try:
        print(f"  Extracting audio from video: {path.name} ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clip = VideoFileClip(str(path))
            if clip.audio is None:
                raise RuntimeError(
                    f"No audio track found in '{path.name}'.\n"
                    f"The video file may be silent or corrupted."
                )
            clip.audio.write_audiofile(out_path, logger=None)
            clip.close()
        print(f"  Audio extracted to: {out_path}")
        return out_path

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract audio from '{path.name}'.\n"
            f"Error: {e}"
        )


def _analyze_quality(audio_path: str) -> tuple[float, str]:
    # RMS energy used
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Use first 60 seconds max for speed
        max_frames = sample_rate * 60
        waveform = waveform[:, :max_frames]

        rms = float(waveform.pow(2).mean().sqrt())

        if rms > 0.05:
            quality = "clean"
        elif rms > 0.01:
            quality = "moderate"
        else:
            quality = "noisy"

        return round(rms, 6), quality

    except Exception:
        return 0.0, "unknown"


def _estimate_speakers(audio_path: str, duration_seconds: float) -> int:
   # energy-based silence detection. (rough estimate for 1-6 speakers)
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Analyse first 4 minutes max
        max_frames = sample_rate * 240
        waveform = waveform[:, :max_frames]

        window = sample_rate
        energies = []
        for i in range(0, waveform.shape[1] - window, window):
            chunk = waveform[:, i:i + window]
            energy = float(chunk.pow(2).mean())
            energies.append(energy)

        if not energies:
            return 1

        energies = np.array(energies)
        mean_energy = np.mean(energies)

        # Count transitions between high and low energy regions
        threshold = mean_energy * 0.3
        in_speech = energies > threshold
        transitions = int(np.sum(np.diff(in_speech.astype(int)) != 0))

        # Map transitions to rough speaker count
        if transitions < 5:
            return 1
        elif transitions < 15:
            return 2
        elif transitions < 30:
            return 3
        elif transitions < 50:
            return 4
        else:
            return min(6, transitions // 15)

    except Exception:
        return 2


def _detect_language(audio_path: str) -> tuple[str, str]:
    
    try:
        import whisper
        print("  Detecting language (using whisper tiny) ...")
        model = whisper.load_model("tiny")

        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        max_frames = sr * 30
        waveform = waveform[:, :max_frames]

        tmp_path = audio_path + "_lang_sample.wav"
        torchaudio.save(tmp_path, waveform, sr)

        audio = whisper.load_audio(tmp_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        lang_code = max(probs, key=probs.get)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())
        print(f"  Language detected: {lang_name} ({lang_code})")
        return lang_code, lang_name

    except Exception as e:
        print(f"  Language detection failed ({e}), defaulting to English.")
        return "en", "English"



def _select_model(
    quality: str,
    language_code: str,
    duration_seconds: float,
    estimated_speakers: int
) -> tuple[str, list]:

    notes = []
    model = "base"  # default

    if quality == "noisy":
        model = "small"
        notes.append("Noisy audio detected — upgraded to 'small' model for better accuracy.")

    if language_code != "en":
        if model == "base":
            model = "small"
        notes.append(f"Non-English audio ({language_code}) — using '{model}' for better multilingual accuracy.")

    if duration_seconds > 3600:
        if model == "small":
            model = "base"
            notes.append("Long recording (>1hr) — reverted to 'base' to manage memory.")
        else:
            notes.append("Long recording detected — processing may take several minutes.")

    if estimated_speakers >= 4:
        notes.append(f"Multiple speakers detected (~{estimated_speakers}) — speaker labels will be approximate.")

    if quality == "clean" and duration_seconds < 600 and language_code == "en":
        model = "base"
        notes.append("Clean short audio — 'base' model is sufficient.")

    return model, notes


def preprocess(file_path: str, output_dir: str = None) -> AudioProfile:
    
    path = Path(file_path)
    is_video = path.suffix.lower() in VIDEO_FORMATS
    extracted_audio_path = None

    print(f"\n{'='*40}")
    print(f"  MOMentum Pre-processing Agent")
    print(f"{'='*40}")
    print(f"  Input: {path.name}")

    if is_video:
        extracted_audio_path = extract_audio(file_path, output_dir)
        audio_path_for_analysis = extracted_audio_path
    else:
        audio_path_for_analysis = file_path

    print("  Loading audio metadata ...")
    audio_file = load_audio(audio_path_for_analysis)
    print(f"  Duration: {audio_file.duration_formatted} | "
          f"Sample Rate: {audio_file.sample_rate} Hz | "
          f"Channels: {audio_file.channels}")

    print("  Analyzing audio quality ...")
    rms_level, quality = _analyze_quality(audio_path_for_analysis)
    print(f"  Quality: {quality.capitalize()} (RMS: {rms_level:.4f})")

    lang_code, lang_name = _detect_language(audio_path_for_analysis)

    print("  Estimating speaker count ...")
    estimated_speakers = _estimate_speakers(
        audio_path_for_analysis,
        audio_file.duration_seconds
    )
    print(f"  Estimated speakers: {estimated_speakers}")

    recommended_model, notes = _select_model(
        quality, lang_code,
        audio_file.duration_seconds,
        estimated_speakers
    )
    print(f"  Recommended Whisper model: {recommended_model}")

    profile = AudioProfile(
        original_file=str(path.resolve()),
        audio_file=audio_file,
        is_video=is_video,
        extracted_audio_path=extracted_audio_path,
        language_code=lang_code,
        language_name=lang_name,
        estimated_speakers=estimated_speakers,
        audio_quality=quality,
        recommended_model=recommended_model,
        rms_level=rms_level,
        notes=notes,
    )

    print(f"\n  Pre-processing complete:{profile}")
    return profile


def cleanup_extracted_audio(profile: AudioProfile) -> None:
    """
    Delete the temporary extracted audio file after pipeline completion.
    Only deletes if the audio was extracted from a video — never deletes original files.
    """
    if profile.is_video and profile.extracted_audio_path:
        path = Path(profile.extracted_audio_path)
        if path.exists():
            path.unlink()
            print(f"  Cleaned up temp file: {path.name}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/preprocessor.py <path_to_audio_or_video>")
        sys.exit(1)

    try:
        profile = preprocess(sys.argv[1])
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\nError: {e}")
        sys.exit(1)