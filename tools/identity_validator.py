from __future__ import annotations

from pathlib import Path


def identity_validator(scene_id: str, video_path: str) -> bool:
    """Lightweight identity validation placeholder before face mapping."""
    if not scene_id or not scene_id.startswith("scene_"):
        return False

    video_file = Path(video_path)
    return video_file.exists()
