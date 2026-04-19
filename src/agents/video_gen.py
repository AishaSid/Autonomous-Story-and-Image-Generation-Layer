from __future__ import annotations

from pathlib import Path

import cv2


def _resolve_reference_image(
    reference_image_paths: list[str],
    character_profile: dict,
    image_assets_dir: str,
) -> Path:
    candidates: list[Path] = []

    for image_path in reference_image_paths:
        if image_path:
            candidates.append(Path(image_path))

    character_name = str(character_profile.get("name", "")).strip()
    assets_root = Path(image_assets_dir) if image_assets_dir else None
    if character_name and assets_root is not None:
        candidates.append(assets_root / f"{character_name}.jpg")
        candidates.append(assets_root / f"{character_name}.png")
        candidates.append(assets_root / f"{character_name}.jpeg")

    if assets_root is not None and assets_root.exists():
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            candidates.extend(sorted(assets_root.glob(pattern)))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            frame = cv2.imread(str(candidate))
            if frame is not None:
                return candidate

    raise FileNotFoundError("No valid character image found for video generation")


def _fit_to_frame(image, width: int, height: int):
    source_h, source_w = image.shape[:2]
    scale = max(width / source_w, height / source_h)
    resized_w = max(int(source_w * scale), 1)
    resized_h = max(int(source_h * scale), 1)

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    x_start = max((resized_w - width) // 2, 0)
    y_start = max((resized_h - height) // 2, 0)
    return resized[y_start : y_start + height, x_start : x_start + width]


def generate_scene_video(
    scene_id: str,
    output_path: str,
    reference_image_paths: list[str],
    character_profile: dict,
    image_assets_dir: str,
    fps: int = 24,
    duration_seconds: float = 1.2,
) -> tuple[str, str]:
    """Generate a valid MP4 where the selected character image is the frame background."""
    source_image_path = _resolve_reference_image(
        reference_image_paths=reference_image_paths,
        character_profile=character_profile,
        image_assets_dir=image_assets_dir,
    )

    source_image = cv2.imread(str(source_image_path))
    if source_image is None:
        raise RuntimeError(f"Could not decode image: {source_image_path}")

    width, height = 640, 360
    base_frame = _fit_to_frame(source_image, width=width, height=height)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {destination}")

    frame_count = max(int(duration_seconds * fps), 1)
    for _ in range(frame_count):
        writer.write(base_frame)

    writer.release()
    return str(destination), str(source_image_path)
