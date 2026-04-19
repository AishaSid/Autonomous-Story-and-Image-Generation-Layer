from __future__ import annotations

import shutil
from pathlib import Path


def face_swapper(scene_id: str, input_video_path: str, output_path: str) -> str:
    """Mock face swapper that preserves a valid video stream for downstream fusion."""
    _ = scene_id
    source = Path(input_video_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(f"Input video for face swap not found: {source}")

    shutil.copyfile(source, destination)
    return str(destination)
