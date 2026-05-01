from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
except ImportError:
    FaceAnalysis = None
    get_model = None


INSWAPPER_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSIGHTFACE_MODEL_DIR = Path(f"{PROJECT_ROOT.drive}\\insightface_models")
INSIGHTFACE_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _extract_emotion_tag(visual_cues: list[str]) -> str:
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


def _get_env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return None


def _download_inswapper_model() -> Path:
    model_path = INSIGHTFACE_MODEL_DIR / "inswapper_128.onnx"
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    print(f"[face_swap] downloading inswapper model to {model_path}")
    response = requests.get(INSWAPPER_URL, timeout=120, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download inswapper model ({response.status_code}): {response.text[:500]}")

    with model_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                handle.write(chunk)

    if model_path.stat().st_size < 50 * 1024 * 1024:
        raise RuntimeError("Downloaded inswapper model looks too small to be valid.")

    return model_path


def _largest_face(faces: list[Any]) -> Any | None:
    if not faces:
        return None
    return max(faces, key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))


def _load_cpu_face_analysis() -> FaceAnalysis:
    if FaceAnalysis is None:
        raise RuntimeError("insightface is not installed. Install with: pip install insightface onnxruntime")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _load_swapper():
    if get_model is None:
        raise RuntimeError("insightface model_zoo is not available.")

    model_path = _download_inswapper_model()
    try:
        return get_model(str(model_path), providers=["CPUExecutionProvider"])
    except TypeError:
        # Older insightface versions may not accept providers on get_model.
        return get_model(str(model_path))


def _ensure_valid_mp4(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    capture = cv2.VideoCapture(str(path))
    try:
        return capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
    finally:
        capture.release()


def _copy_with_overlay_labels(
    source: Path,
    destination: Path,
    scene_id: str,
    character_name: str,
    emotion_tag: str,
) -> tuple[str, bool, str, str]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        shutil.copyfile(source, destination)
        return str(destination), _ensure_valid_mp4(destination), character_name, emotion_tag

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        shutil.copyfile(source, destination)
        return str(destination), _ensure_valid_mp4(destination), character_name, emotion_tag

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.putText(frame, f"Character: {character_name}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Emotion: {emotion_tag}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 255, 180), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Scene: {scene_id} | Frame: {frame_index}", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()
    print(f"[WARN] face swap fallback overlay completed for {scene_id} ({frame_index} frames)")
    return str(destination), _ensure_valid_mp4(destination), character_name, emotion_tag


def _swap_faces_with_insightface(
    scene_id: str,
    input_video_path: str,
    output_path: str,
    reference_image_path: str,
    character_name: str,
) -> tuple[str, bool, str, str]:
    source = Path(input_video_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    face_app = _load_cpu_face_analysis()
    swapper = _load_swapper()

    source_image = cv2.imread(str(reference_image_path))
    if source_image is None:
        raise RuntimeError(f"Could not decode reference image: {reference_image_path}")

    source_faces = face_app.get(source_image)
    source_face = _largest_face(source_faces)
    if source_face is None:
        raise RuntimeError(f"No face found in reference image: {reference_image_path}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or source_image.shape[1] or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or source_image.shape[0] or 360)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output video writer: {destination}")

    frame_index = 0
    swapped_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            target_faces = face_app.get(frame)
            if target_faces:
                target_faces = sorted(
                    target_faces,
                    key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
                    reverse=True,
                )
                for target_face in target_faces:
                    try:
                        frame = swapper.get(frame, target_face, source_face, paste_back=True)
                        swapped_frames += 1
                    except Exception:
                        continue

            cv2.putText(
                frame,
                f"Scene: {scene_id} | Character: {character_name}",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if not _ensure_valid_mp4(destination):
        raise RuntimeError(f"InsightFace face swap produced an invalid mp4: {destination}")

    print(f"[OK] face swap completed for {scene_id} using InsightFace CPU; frames={frame_index}; swaps={swapped_frames}")
    return str(destination), True, character_name, "neutral"


def face_swap_validate_and_map(
    scene_id: str,
    input_video_path: str,
    output_path: str,
    scene_task: dict[str, Any],
    character_db_path: str,
    use_cpu_only: bool = True,
) -> tuple[str, bool, str, str]:
    """CPU-first face swap using InsightFace inswapper_128.onnx, with overlay fallback."""
    source = Path(input_video_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        error_msg = f"Input video for face swap not found: {source}"
        print(f"[ERROR] {error_msg}")
        return str(destination), False, "Unknown", "error"

    character_profile = scene_task.get("asset_context", {}).get("character_profile", {})
    character_name = str(character_profile.get("name", "Unknown")).strip() or "Unknown"
    reference_image_paths = scene_task.get("asset_context", {}).get("reference_image_paths", [])
    reference_image = reference_image_paths[0] if reference_image_paths else ""
    visual_cues = scene_task.get("parallel_branches", {}).get("video", {}).get("inputs", {}).get("visual_cues", [])
    emotion_tag = _extract_emotion_tag([str(cue) for cue in visual_cues])

    is_valid, validated_character, _ = _validate_identity(scene_task, character_db_path)
    if not is_valid:
        print(f"[WARN] identity validation failed for {scene_id}; continuing with best-effort swap")
        validated_character = character_name

    if use_cpu_only and reference_image and Path(reference_image).exists():
        try:
            return _swap_faces_with_insightface(
                scene_id=scene_id,
                input_video_path=str(source),
                output_path=str(destination),
                reference_image_path=reference_image,
                character_name=validated_character,
            )
        except Exception as e:
            print(f"[WARN] InsightFace CPU swap failed for {scene_id}: {e}. Using fallback overlay.")

    return _copy_with_overlay_labels(
        source=source,
        destination=destination,
        scene_id=scene_id,
        character_name=validated_character,
        emotion_tag=emotion_tag,
    )
