from __future__ import annotations

import json
from pathlib import Path


def identity_validator(
    scene_id: str,
    video_path: str,
    character_db_path: str,
    expected_character: str,
    expected_image_path: str,
) -> bool:
    """Validate identity context against the character DB before face mapping."""
    if not scene_id or not scene_id.startswith("scene_"):
        return False

    video_file = Path(video_path)
    if not video_file.exists():
        return False

    if expected_image_path and not Path(expected_image_path).exists():
        return False

    character_db_file = Path(character_db_path)
    if not character_db_file.exists():
        return False

    try:
        payload = json.loads(character_db_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    characters = payload.get("characters", []) if isinstance(payload, dict) else []
    if not isinstance(characters, list):
        return False

    if expected_character:
        return any(
            isinstance(character, dict) and str(character.get("name", "")) == expected_character
            for character in characters
        )

    return len(characters) > 0
