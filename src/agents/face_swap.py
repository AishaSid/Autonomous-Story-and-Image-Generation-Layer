from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2


def _extract_emotion_tag(visual_cues: list[str]) -> str:
    cue_text = " ".join([cue.lower() for cue in visual_cues])

    if "conflict" in cue_text or "tense" in cue_text:
        return "tense"
    if "emotional" in cue_text or "close-up" in cue_text:
        return "emotive"
    if "establishing" in cue_text or "wide" in cue_text:
        return "calm"
    return "neutral"


def _validate_identity(scene_task: dict[str, Any], character_db_path: str) -> tuple[bool, str, list[str]]:
    character_profile = scene_task.get("asset_context", {}).get("character_profile", {})
    expected_name = str(character_profile.get("name", "")).strip()
    expected_traits = [str(item).strip().lower() for item in character_profile.get("personality_traits", [])]

    db_file = Path(character_db_path)
    if not db_file.exists():
        return False, expected_name, expected_traits

    try:
        payload = json.loads(db_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, expected_name, expected_traits

    characters = payload.get("characters", []) if isinstance(payload, dict) else []
    if not isinstance(characters, list):
        return False, expected_name, expected_traits

    for character in characters:
        if not isinstance(character, dict):
            continue

        name = str(character.get("name", "")).strip()
        traits = [str(item).strip().lower() for item in character.get("personality_traits", [])]

        if expected_name and name != expected_name:
            continue

        if expected_traits and not set(expected_traits).issubset(set(traits)):
            continue

        return True, expected_name or name, expected_traits

    return False, expected_name, expected_traits


def face_swap_validate_and_map(
    scene_id: str,
    input_video_path: str,
    output_path: str,
    scene_task: dict[str, Any],
    character_db_path: str,
) -> tuple[str, bool, str, str]:
    """Validate character identity using character_db and burn overlay labels on mapped video."""
    source = Path(input_video_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    is_valid, character_name, _ = _validate_identity(scene_task, character_db_path)
    if not is_valid:
        return str(destination), False, character_name, "invalid"

    visual_cues = scene_task.get("parallel_branches", {}).get("video", {}).get("inputs", {}).get("visual_cues", [])
    emotion_tag = _extract_emotion_tag([str(cue) for cue in visual_cues])

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return str(destination), False, character_name, emotion_tag

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        return str(destination), False, character_name, emotion_tag

    overlay_name = character_name or "Unknown"
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        cv2.putText(
            frame,
            f"Character: {overlay_name}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Emotion: {emotion_tag}",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (120, 255, 180),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Scene: {scene_id} | Frame: {frame_index}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()

    return str(destination), True, overlay_name, emotion_tag
