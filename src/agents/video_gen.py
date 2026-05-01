from __future__ import annotations

import math
import time
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.agents.common import invoke_mcp_tool_via_protocol

try:
    from tools.enhanced_video_generator import (
        generate_scene_video_from_images,
        mux_audio_to_video,
    )
except ImportError:
    generate_scene_video_from_images = None
    mux_audio_to_video = None


def _scene_palette(scene_id: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    seed = sum(ord(ch) for ch in scene_id)
    color_a = ((seed * 29) % 180 + 30, (seed * 53) % 180 + 30, (seed * 71) % 180 + 30)
    color_b = ((seed * 41) % 180 + 30, (seed * 67) % 180 + 30, (seed * 89) % 180 + 30)
    return color_a, color_b


def _build_scene_frame(
    scene_id: str,
    summary: str,
    visual_cues: list[str],
    width: int,
    height: int,
    progress: float,
) -> np.ndarray:
    color_a, color_b = _scene_palette(scene_id)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Animated vertical gradient, driven by scene progress.
    for row in range(height):
        alpha = row / max(height - 1, 1)
        r = int((1 - alpha) * color_a[0] + alpha * color_b[0])
        g = int((1 - alpha) * color_a[1] + alpha * color_b[1])
        b = int((1 - alpha) * color_a[2] + alpha * color_b[2])
        frame[row, :, :] = (b, g, r)

    camera_shift = int(24 * math.sin(progress * 2.0 * math.pi))
    panel_left = max(30 + camera_shift, 12)
    panel_right = min(width - 30 + camera_shift, width - 12)
    cv2.rectangle(frame, (panel_left, 36), (panel_right, height - 36), (18, 18, 18), 2)

    current_cue = visual_cues[min(int(progress * max(len(visual_cues), 1)), max(len(visual_cues) - 1, 0))] if visual_cues else "Interview scene."
    summary_text = (summary or "Scene context unavailable.")[:92]
    cue_text = current_cue[:92]
    cv2.putText(frame, f"Scene {scene_id}", (52, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Summary: {summary_text}", (52, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Cue: {cue_text}", (52, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def _dialogue_entry_to_line(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ("line", "text", "dialogue", "utterance", "content"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _estimate_duration_from_dialogue(dialogue_beats: list[Any]) -> float:
    lines = [_dialogue_entry_to_line(entry) for entry in dialogue_beats]
    transcript = " ".join([line for line in lines if line])
    word_count = len([word for word in transcript.split() if word.strip()])
    # Approximate natural speech at ~150 words per minute.
    estimated_seconds = (word_count / 150.0) * 60.0
    return max(estimated_seconds, 1.2)


def _duration_from_audio(audio_path: str, fallback_seconds: float) -> float:
    audio_file = Path(audio_path)
    if not audio_file.exists():
        # Voice and video branches run concurrently; wait briefly for the corresponding wav.
        for _ in range(120):
            if audio_file.exists():
                break
            time.sleep(0.1)

    if not audio_file.exists():
        return fallback_seconds

    for _ in range(120):
        try:
            with wave.open(str(audio_file), "rb") as wav_file:
                frame_count = wav_file.getnframes()
                frame_rate = wav_file.getframerate() or 22050
                return max(frame_count / float(frame_rate), 0.5)
        except Exception:
            time.sleep(0.1)

    return fallback_seconds


def generate_scene_video(
    scene_id: str,
    output_path: str,
    summary: str,
    visual_cues: list[str],
    reference_image_paths: list[str],
    character_profile: dict,
    image_assets_dir: str,
    dialogue_beats: list[Any],
    audio_path: str,
    fps: int = 24,
) -> tuple[str, str]:
    """
    Generate a scene MP4 video using Pexels stock footage.
    """
    width, height = 640, 360
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"[video_generation] Generating video for scene {scene_id}")
    
    # Get duration from audio
    estimated_duration = _estimate_duration_from_dialogue(dialogue_beats)
    actual_duration = _duration_from_audio(audio_path, estimated_duration)
    print(f"[video_generation] Target duration: {actual_duration:.1f} seconds")

    # Use Pexels stock footage for proper video
    print(f"[video_generation] Querying stock footage from Pexels")
    tool_result = invoke_mcp_tool_via_protocol(
        "query_stock_footage",
        {
            "character_traits": [
                character_profile.get("name", scene_id),
                character_profile.get("appearance_description", ""),
                *visual_cues,
            ],
            "scene_summary": summary,
            "visual_cues": visual_cues,
            "output_path": str(destination),
            "target_duration_seconds": max(int(round(actual_duration)), 1),
        },
    )
    
    video_path = Path(str(tool_result.get("video_path") or "")).expanduser()
    if not video_path.exists() and tool_result.get("video_url"):
        raise RuntimeError(
            "query_stock_footage returned a video_url but did not provide a local video_path; "
            "download must happen inside the MCP tool."
        )

    if not video_path.exists():
        raise RuntimeError("query_stock_footage did not return a usable mp4 path.")

    if video_path.suffix.lower() != ".mp4":
        raise RuntimeError(f"query_stock_footage returned non-mp4 asset: {video_path}")

    if not _validate_mp4(video_path):
        raise RuntimeError(f"Downloaded video is not playable or too small: {video_path}")

    if video_path.resolve() != destination.resolve():
        destination.write_bytes(video_path.read_bytes())

    if not _validate_mp4(destination):
        raise RuntimeError(f"Final output is not a valid playable mp4: {destination}")

    # Mux audio into video using MoviePy
    if audio_path and Path(audio_path).exists():
        print(f"[video_generation] Adding audio to video")
        final_output = str(destination).replace('.mp4', '_with_audio.mp4')
        try:
            from moviepy import VideoFileClip, AudioFileClip
            video = VideoFileClip(str(destination))
            audio = AudioFileClip(str(audio_path))
            video = video.with_audio(audio)
            video.write_videofile(final_output, codec='libx264', audio_codec='aac')
            video.close()
            audio.close()
            if Path(final_output).exists():
                destination = Path(final_output)
                print(f"✓ Audio added: {destination}")
        except Exception as e:
            print(f"[WARN] Audio mux failed: {e}")

    print(f"[video_generation] Stock footage: {destination.as_posix()}")
    return str(destination), str(destination)



def _validate_mp4(path: Path, min_size_bytes: int = 50 * 1024) -> bool:
    if not path.exists() or path.stat().st_size < min_size_bytes:
        return False
    if path.suffix.lower() != ".mp4":
        return False

    capture = cv2.VideoCapture(str(path))
    try:
        return capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
    finally:
        capture.release()
