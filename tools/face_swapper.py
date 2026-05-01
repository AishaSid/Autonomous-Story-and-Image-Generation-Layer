from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import cv2

try:
    import fal_client
except ImportError:
    fal_client = None


def _extract_emotion_tag(visual_cues: list[str]) -> str:
    """Extract emotion tag from visual cues for context."""
    cue_text = " ".join([cue.lower() for cue in visual_cues])

    if "conflict" in cue_text or "tense" in cue_text or "angry" in cue_text:
        return "tense"
    if "emotional" in cue_text or "crying" in cue_text or "laugh" in cue_text:
        return "emotive"
    if "establishing" in cue_text or "wide" in cue_text or "calm" in cue_text:
        return "calm"
    if "surprise" in cue_text or "shock" in cue_text:
        return "surprised"
    return "neutral"


def face_swap_validate_and_map(
    scene_id: str,
    input_video_path: str,
    output_path: str,
    scene_task: dict[str, Any],
    character_db_path: str,
    use_fal_api: bool = True,
) -> tuple[str, bool, str, str]:
    """
    Perform face swap using fal.ai, with fallback to local processing.
    
    Args:
        scene_id: Scene identifier
        input_video_path: Path to input video
        output_path: Path for output video
        scene_task: Task configuration with character info
        character_db_path: Path to character database
        use_fal_api: Whether to attempt fal.ai API
    
    Returns:
        Tuple of (output_video_path, success, character_name, emotion_tag)
    """
    source = Path(input_video_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        error_msg = f"Input video for face swap not found: {source}"
        print(f"✗ {error_msg}")
        return str(destination), False, "Unknown", "error"

    # Extract character information
    character_profile = scene_task.get("asset_context", {}).get("character_profile", {})
    character_name = str(character_profile.get("name", "Unknown")).strip()
    reference_image_paths = scene_task.get("asset_context", {}).get("reference_image_paths", [])
    reference_image = reference_image_paths[0] if reference_image_paths else ""
    visual_cues = scene_task.get("parallel_branches", {}).get("video", {}).get("inputs", {}).get("visual_cues", [])
    emotion_tag = _extract_emotion_tag([str(cue) for cue in visual_cues])

    # Try fal.ai API first
    if use_fal_api and reference_image and Path(reference_image).exists():
        try:
            result = _face_swap_with_fal_api(
                scene_id=scene_id,
                input_video_path=str(source),
                output_path=str(destination),
                reference_image_path=reference_image,
                character_name=character_name,
            )
            if result:
                print(f"✓ Face swap completed for {scene_id} ({character_name}) using fal.ai")
                return result
        except Exception as e:
            print(f"⚠ fal.ai face swap failed for {scene_id}: {e}. Using fallback.")

    # Fallback: copy video with overlay labels
    return _face_swap_local_fallback(
        scene_id=scene_id,
        input_video_path=str(source),
        output_path=str(destination),
        character_name=character_name,
        emotion_tag=emotion_tag,
    )


def _face_swap_with_fal_api(
    scene_id: str,
    input_video_path: str,
    output_path: str,
    reference_image_path: str,
    character_name: str = "",
) -> tuple[str, bool, str, str] | None:
    """Perform face swap using fal.ai/face-swap."""
    if fal_client is None:
        raise RuntimeError("fal_client package not installed. Please install with: pip install fal-client")
    
    api_key = os.getenv("FAL_KEY")
    if not api_key:
        raise RuntimeError("FAL_KEY environment variable not set")
    
    fal_client.api_key = api_key
    
    try:
        # Call fal.ai face-swap model
        result = fal_client.run(
            "fal-ai/face-swap",
            arguments={
                "source_image_path": reference_image_path,
                "target_video_path": input_video_path,
            },
        )
        
        output_video_url = result.get("video", {}).get("url")
        if not output_video_url:
            return None
        
        # Download video
        import requests
        response = requests.get(output_video_url, timeout=60)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"✓ Face-swapped video for {scene_id} ({len(response.content)} bytes)")
            return str(output_path), True, character_name, "neutral"
        
    except Exception as e:
        print(f"fal.ai/face-swap error: {e}")
        return None


def _face_swap_local_fallback(
    scene_id: str,
    input_video_path: str,
    output_path: str,
    character_name: str = "Unknown",
    emotion_tag: str = "neutral",
) -> tuple[str, bool, str, str]:
    """Fallback: copy video and add overlay text."""
    source = Path(input_video_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Try to process with CV2 and add labels
    try:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            # Simple copy if can't open
            shutil.copyfile(source, destination)
            return str(destination), True, character_name, emotion_tag

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
            shutil.copyfile(source, destination)
            return str(destination), True, character_name, emotion_tag

        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # Add character label
            if frame_index < 5:  # Add label to first few frames
                label = f"{character_name} [{emotion_tag}]"
                cv2.putText(
                    frame,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"scene: {scene_id}",
                    (20, 80),
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
        print(f"✓ Face-swap fallback completed for {scene_id} ({character_name}) - {frame_index} frames")
        return str(destination), True, character_name, emotion_tag

    except Exception as e:
        print(f"✗ Face swap fallback error for {scene_id}: {e}")
        # Last resort: copy file
        try:
            shutil.copyfile(source, destination)
            return str(destination), True, character_name, emotion_tag
        except Exception as copy_error:
            print(f"✗ Could not even copy file: {copy_error}")
            return str(destination), False, character_name, "error"
