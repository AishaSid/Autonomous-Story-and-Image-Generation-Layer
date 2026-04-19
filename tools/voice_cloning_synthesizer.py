from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path


def _write_placeholder_wav(output_path: Path, duration_seconds: float = 1.2) -> None:
    sample_rate = 22050
    frequency = 220.0
    samples = int(sample_rate * duration_seconds)
    frames = array("h")

    for i in range(samples):
        # Low-amplitude tone placeholder to represent synthesized speech output.
        value = int(1600 * math.sin(2 * math.pi * frequency * (i / sample_rate)))
        frames.append(value)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames.tobytes())


def voice_cloning_synthesizer(
    scene_id: str,
    dialogue_beats: list[str],
    output_path: str,
) -> str:
    """Mock MCP tool for voice cloning/TTS in CPU-only environments."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    transcript_preview = " ".join(dialogue_beats)[:140]

    try:
        # Optional CPU fallback if pyttsx3 is available in the environment.
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.save_to_file(transcript_preview or f"Scene {scene_id}", str(destination))
        engine.runAndWait()
    except Exception:
        # API-style placeholder fallback: emit a valid wav container.
        _write_placeholder_wav(destination)

    return str(destination)
