from __future__ import annotations

from pathlib import Path


def face_swapper(scene_id: str, input_video_path: str, output_path: str) -> str:
    """Mock face swapper that creates a mapped-video placeholder artifact."""
    source = Path(input_video_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.exists():
        source_content = source.read_text(encoding="utf-8", errors="ignore")
    else:
        source_content = "placeholder_source_missing"

    destination.write_text(
        "\n".join(
            [
                f"scene_id: {scene_id}",
                "identity_validation: passed",
                "face_swap_status: mapped",
                f"source_video: {source}",
                f"source_preview: {source_content[:120]}",
            ]
        ),
        encoding="utf-8",
    )
    return str(destination)
