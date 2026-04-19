from __future__ import annotations

from pathlib import Path


def query_stock_footage(
    scene_id: str,
    summary: str,
    visual_cues: list[str],
    output_path: str,
    reference_image_paths: list[str] | None = None,
    character_profile: dict | None = None,
) -> str:
    """Mock MCP tool for video generation with API-like placeholder output."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    reference_image_paths = reference_image_paths or []
    character_profile = character_profile or {}

    content = "\n".join(
        [
            f"scene_id: {scene_id}",
            f"summary: {summary}",
            f"visual_cues: {', '.join(visual_cues)}",
            f"reference_image_paths: {', '.join(reference_image_paths)}",
            f"character_profile: {character_profile}",
            "status: placeholder_video_asset",
        ]
    )
    destination.write_text(content, encoding="utf-8")
    return str(destination)
