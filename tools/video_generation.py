from __future__ import annotations

import json
import math
import os
import time
import wave
from pathlib import Path
from typing import Any

import cv2
import requests

try:
    import fal_client
except ImportError:
    fal_client = None


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
        # Voice and video branches run concurrently; wait briefly for the corresponding audio file.
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


def _get_env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return None


def _get_pexels_api_key() -> str | None:
    return _get_env_value("PEXELS_API_KEY", "pexels_api_key", "PexelsApiKey")


def _log_api_call(endpoint: str, payload: dict[str, Any], response: dict[str, Any] | None = None) -> None:
    """Log API request/response details without sensitive headers or keys."""
    try:
        payload_json = json.dumps(payload, ensure_ascii=True)
    except Exception:
        payload_json = str(payload)

    print(f"[video_generation] endpoint called: {endpoint}")
    print(f"[video_generation] payload sent: {payload_json}")

    if response is None:
        return

    response_type = "unknown"
    if isinstance(response, dict):
        if response.get("videos"):
            response_type = "video"
        elif response.get("photos"):
            response_type = "image"

    job_id = response.get("job_id") if isinstance(response, dict) else None
    status = response.get("status") if isinstance(response, dict) else None
    print(
        "[video_generation] response type: "
        f"{response_type}; job_id={job_id if job_id is not None else 'n/a'}; "
        f"status={status if status is not None else 'n/a'}"
    )


def _wait_for_video_job_if_needed(
    endpoint: str,
    response_data: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: int = 90,
    poll_interval_seconds: int = 2,
) -> dict[str, Any]:
    """
    Handle async job-style responses when present.
    Pexels search is usually synchronous; this is a safe fallback for async-style APIs.
    """
    status = str(response_data.get("status", "")).lower()
    job_id = response_data.get("job_id")
    if not status and not job_id:
        return response_data

    if status in {"completed", "succeeded", "done", "success"}:
        return response_data

    status_url = response_data.get("status_url") or response_data.get("job_url") or endpoint
    start = time.time()
    while time.time() - start < timeout_seconds:
        poll_response = requests.get(status_url, headers=headers, timeout=30)
        if poll_response.status_code != 200:
            raise RuntimeError(
                f"Video job polling failed ({poll_response.status_code}): {poll_response.text[:500]}"
            )
        poll_data = poll_response.json()
        poll_status = str(poll_data.get("status", "")).lower()
        _log_api_call(status_url, {"job_id": job_id, "poll": True}, poll_data)
        if poll_status in {"completed", "succeeded", "done", "success"}:
            return poll_data
        if poll_status in {"failed", "error", "cancelled"}:
            raise RuntimeError(f"Video job failed with status '{poll_status}': {poll_data}")
        time.sleep(poll_interval_seconds)

    raise RuntimeError("Timed out waiting for asynchronous video job completion.")


def _build_pexels_video_params(query: str, target_duration_seconds: int) -> dict[str, Any]:
    """
    Build Pexels videos/search params.
    Note: Pexels is a stock search API, not a generative render API.
    Unsupported generative fields like frames/render_mode/output_format are intentionally not sent.
    """
    minimum = max(target_duration_seconds - 1, 1)
    maximum = target_duration_seconds + 2
    return {
        "query": query,
        "per_page": 5,
        "page": 1,
        "min_duration": minimum,
        "max_duration": maximum,
    }


def _select_pexels_video_asset(video_item: dict[str, Any]) -> str | None:
    """Select a real video asset URL (mp4 link), never thumbnails/images."""
    video_files = video_item.get("video_files") or []
    if not video_files:
        return None

    def _score(asset: dict[str, Any]) -> tuple[int, int, int]:
        width = int(asset.get("width") or 0)
        height = int(asset.get("height") or 0)
        file_type = str(asset.get("file_type") or "").lower()
        is_mp4 = 1 if "mp4" in file_type else 0
        # Prefer mp4 and highest resolution.
        return (is_mp4, width * height, int(asset.get("fps") or 0))

    sorted_assets = sorted(video_files, key=_score, reverse=True)
    for asset in sorted_assets:
        link = asset.get("link")
        if isinstance(link, str) and link.strip():
            return link.strip()
    return None


def _download_pexels_video(
    query: str,
    destination_video_path: str,
    target_duration_seconds: int = 5,
) -> str | None:
    api_key = _get_pexels_api_key()
    if not api_key:
        return None

    headers = {"Authorization": api_key}
    endpoint = "https://api.pexels.com/videos/search"
    params = _build_pexels_video_params(query=query, target_duration_seconds=target_duration_seconds)
    _log_api_call(endpoint, params)

    response = requests.get(
        endpoint,
        headers=headers,
        params=params,
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Pexels video search failed ({response.status_code}): {response.text}")

    data = response.json()
    _log_api_call(endpoint, params, data)
    data = _wait_for_video_job_if_needed(endpoint=endpoint, response_data=data, headers=headers)

    videos = data.get("videos", [])
    if not videos:
        return None

    selected = videos[0]
    video_url = _select_pexels_video_asset(selected)
    if not video_url:
        return None

    destination = Path(destination_video_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    video_data = requests.get(video_url, timeout=60)
    if video_data.status_code != 200:
        raise RuntimeError(
            f"Failed to download Pexels video asset ({video_data.status_code}): {video_data.text[:500]}"
        )

    destination.write_bytes(video_data.content)
    print(f"✓ Downloaded Pexels video for '{query}' to {destination}")
    return str(destination)


def _generate_with_pexels(
    scene_id: str,
    output_path: str,
    reference_image_path: str,
    character_profile: dict[str, Any] | None = None,
    dialogue_beats: list[Any] | None = None,
    audio_path: str = "",
) -> tuple[str, str] | None:
    query = scene_id
    if character_profile:
        query = character_profile.get("appearance_description", query) or character_profile.get("name", query)
    if dialogue_beats:
        first_line = _dialogue_entry_to_line(dialogue_beats[0])
        if first_line:
            query = f"{query} {first_line}"

    target_duration = 5
    if audio_path and Path(audio_path).exists():
        target_duration = int(round(_duration_from_audio(audio_path=audio_path, fallback_seconds=5.0)))

    pexels_video_path = Path(output_path).with_name(f"{Path(output_path).stem}_pexels.mp4")
    downloaded_video = _download_pexels_video(
        query=query,
        destination_video_path=str(pexels_video_path),
        target_duration_seconds=max(target_duration, 1),
    )
    if not downloaded_video:
        return None

    return str(downloaded_video), downloaded_video


def _build_video_prompt(
    summary: str,
    visual_cues: list[str],
    character_profile: dict[str, Any] | None = None,
) -> str:
    """
    Build a comprehensive prompt for video generation from scene data.
    """
    cues_text = " ".join(visual_cues) if visual_cues else "Interview scene"
    
    character_desc = ""
    if character_profile:
        name = character_profile.get("name", "Character")
        appearance = character_profile.get("appearance_description", "")
        clothing = character_profile.get("clothing", "")
        character_desc = f"{name}: {appearance}. Wearing: {clothing}."
    
    prompt = f"""Generate a professional 5-second video clip for this scene:
    
Scene Summary: {summary}

Visual Direction: {cues_text}

{character_desc if character_desc else ""}

Requirements:
- 5 seconds duration
- 1920x1080 resolution
- Cinematic quality
- Professional lighting
- Natural character movement
- Interview/dialogue context
- Clear facial expressions
- Professional color grading"""
    
    return prompt


def generate_scene_video(
    scene_id: str,
    output_path: str,
    reference_image_paths: list[str] | None = None,
    character_profile: dict[str, Any] | None = None,
    image_assets_dir: str = "",
    dialogue_beats: list[Any] | None = None,
    audio_path: str = "",
    use_fal_api: bool = True,
) -> tuple[str, str]:
    """
    Generate video using fal.ai (luma-dream-machine or kling-video).
    
    Args:
        scene_id: Scene identifier
        output_path: Output video file path
        reference_image_paths: List of reference image paths
        character_profile: Character profile dict
        image_assets_dir: Directory containing character images
        dialogue_beats: Dialogue lines (for prompt enrichment)
        audio_path: Path to audio file (for duration sync)
        use_fal_api: Whether to use fal.ai API or fallback
    
    Returns:
        Tuple of (video_path, reference_image_path)
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine reference image
    source_image_path = None
    if reference_image_paths:
        for ref_path in reference_image_paths:
            if ref_path and Path(ref_path).exists():
                source_image_path = ref_path
                break
    
    if not source_image_path and image_assets_dir:
        assets_dir = Path(image_assets_dir)
        if assets_dir.exists():
            for img_file in assets_dir.glob("*.png"):
                if cv2.imread(str(img_file)) is not None:
                    source_image_path = str(img_file)
                    break
    
    # Fallback: create placeholder if no image found
    if not source_image_path:
        source_image_path = _create_fallback_image(image_assets_dir)
    
    # Try fal.ai API if enabled and configured
    if use_fal_api:
        try:
            result = _generate_with_fal_api(
                scene_id=scene_id,
                output_path=str(destination),
                reference_image_path=source_image_path,
                character_profile=character_profile,
                dialogue_beats=dialogue_beats or [],
            )
            if result:
                return result
        except Exception as e:
            print(f"⚠ fal.ai video generation failed for {scene_id}: {e}. Falling back to local generation.")
    
    # Fallback: try Pexels-based video generation if configured
    try:
        pexels_result = _generate_with_pexels(
            scene_id=scene_id,
            output_path=str(destination),
            reference_image_path=source_image_path,
            character_profile=character_profile,
            dialogue_beats=dialogue_beats or [],
            audio_path=audio_path,
        )
        if pexels_result:
            return pexels_result
    except Exception as e:
        print(f"⚠ Pexels fallback failed: {e}. Falling back to local generation.")

    # Local image-based video fallback
    return _generate_local_video(
        scene_id=scene_id,
        output_path=str(destination),
        reference_image_path=source_image_path,
        audio_path=audio_path,
        dialogue_beats=dialogue_beats or [],
    )


def _generate_with_fal_api(
    scene_id: str,
    output_path: str,
    reference_image_path: str,
    character_profile: dict[str, Any] | None = None,
    dialogue_beats: list[Any] | None = None,
) -> tuple[str, str] | None:
    """Generate video using fal.ai (luma-dream-machine or kling-video)."""
    if fal_client is None:
        raise RuntimeError("fal_client package not installed. Please install with: pip install fal-client")
    
    api_key = _get_env_value("FAL_KEY", "FAL_AI_KEY", "fal_key", "fal_ai_key")
    if not api_key:
        raise RuntimeError("fal.ai API key not set. Set FAL_KEY or FAL_AI_KEY.")
    
    # Set fal.ai credentials
    fal_client.api_key = api_key
    
    # Build prompt
    summary = f"Scene {scene_id}"
    visual_cues = []
    if dialogue_beats:
        summary = f"{scene_id}: Interview dialogue scene with multiple speakers"
    
    prompt = _build_video_prompt(
        summary=summary,
        visual_cues=visual_cues,
        character_profile=character_profile,
    )
    
    try:
        # Try luma-dream-machine first
        result = fal_client.run(
            "fal-ai/luma-dream-machine",
            arguments={
                "prompt": prompt,
                "duration": 5,  # 5-second video
                "keyframe": reference_image_path,
            },
        )
        
        video_url = result.get("video", {}).get("url")
        if not video_url:
            return None
        
        # Download video
        import requests
        response = requests.get(video_url, timeout=30)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✓ Generated video for {scene_id} using fal.ai ({len(response.content)} bytes)")
            return str(output_path), reference_image_path
        
    except Exception as e:
        print(f"⚠ fal.ai/luma-dream-machine failed: {e}")
        # Try kling-video as fallback
        try:
            result = fal_client.run(
                "fal-ai/kling-video",
                arguments={
                    "prompt": prompt,
                    "duration": 5,
                    "keyframe": reference_image_path,
                },
            )
            video_url = result.get("video", {}).get("url")
            if video_url:
                import requests
                response = requests.get(video_url, timeout=30)
                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    print(f"✓ Generated video for {scene_id} using kling-video ({len(response.content)} bytes)")
                    return str(output_path), reference_image_path
        except Exception as e2:
            print(f"⚠ fal.ai/kling-video also failed: {e2}")
            return None


def _generate_local_video(
    scene_id: str,
    output_path: str,
    reference_image_path: str,
    audio_path: str = "",
    dialogue_beats: list[Any] | None = None,
    fps: int = 24,
) -> tuple[str, str]:
    """Fallback: Generate video locally using reference image."""
    source_image = cv2.imread(str(reference_image_path))
    if source_image is None:
        raise RuntimeError(f"Could not decode image: {reference_image_path}")

    width, height = 640, 360
    # Fit image to frame
    source_h, source_w = source_image.shape[:2]
    scale = max(width / source_w, height / source_h)
    resized_w = max(int(source_w * scale), 1)
    resized_h = max(int(source_h * scale), 1)
    resized = cv2.resize(source_image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    x_start = max((resized_w - width) // 2, 0)
    y_start = max((resized_h - height) // 2, 0)
    base_frame = resized[y_start : y_start + height, x_start : x_start + width]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    fallback_duration = _estimate_duration_from_dialogue(dialogue_beats or [])
    duration_seconds = _duration_from_audio(audio_path=audio_path, fallback_seconds=fallback_duration)
    frame_count = max(int(math.ceil(duration_seconds * fps)), 1)
    for _ in range(frame_count):
        writer.write(base_frame)

    writer.release()
    print(f"✓ Generated local video for {scene_id} ({frame_count} frames @ {fps}fps)")
    return str(output_path), str(reference_image_path)


def _create_fallback_image(image_assets_dir: str) -> str:
    """Create a simple fallback image if no reference found."""
    assets_dir = Path(image_assets_dir) if image_assets_dir else Path("image_assets")
    assets_dir.mkdir(parents=True, exist_ok=True)
    
    fallback_path = assets_dir / "fallback.png"
    if fallback_path.exists():
        return str(fallback_path)
    
    # Create a simple solid-color image
    import numpy as np
    fallback_image = np.zeros((360, 640, 3), dtype=np.uint8)
    fallback_image[:] = (100, 150, 200)  # Blue-ish background
    cv2.imwrite(str(fallback_path), fallback_image)
    return str(fallback_path)
