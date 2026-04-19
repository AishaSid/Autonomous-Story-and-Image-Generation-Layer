from __future__ import annotations

from pathlib import Path
from typing import Any

import pyttsx3

try:
    import pythoncom
except Exception:  # pragma: no cover - only relevant on Windows hosts.
    pythoncom = None


def _dialogue_entry_to_line(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()

    if isinstance(entry, dict):
        for key in ("line", "text", "dialogue", "utterance", "content"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _dialogue_to_text(dialogue_beats: list[Any], scene_id: str) -> str:
    lines = [_dialogue_entry_to_line(entry) for entry in dialogue_beats]
    normalized = [line for line in lines if line]
    if not normalized:
        return f"Scene {scene_id} dialogue."
    return " ".join(normalized)


def voice_cloning_synthesizer(
    scene_id: str,
    dialogue_beats: list[Any],
    output_path: str,
) -> str:
    """CPU-friendly speech synthesis using pyttsx3 from manifest dialogue lines."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    speech_text = _dialogue_to_text(dialogue_beats=dialogue_beats, scene_id=scene_id)

    if pythoncom is not None:
        pythoncom.CoInitialize()

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 0.95)
        engine.save_to_file(speech_text, str(destination))
        engine.runAndWait()
    finally:
        if pythoncom is not None:
            pythoncom.CoUninitialize()

    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"TTS generation failed for scene {scene_id}")

    return str(destination)
