from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE2_OUTPUTS_DIR = PROJECT_ROOT / "phase2_outputs"
PHASE2_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return None


def _log_video_fetch(endpoint: str, payload: dict[str, Any], response: dict[str, Any] | None = None) -> None:
    print(f"[query_stock_footage] endpoint called: {endpoint}")
    print(f"[query_stock_footage] payload sent: {json.dumps(payload, ensure_ascii=True)}")
    if response is None:
        return

    response_type = "video" if response.get("videos") else "unknown"
    print(
        "[query_stock_footage] response type: "
        f"{response_type}; status={response.get('status', 'n/a')}; job_id={response.get('job_id', 'n/a')}"
    )


def _build_search_query(character_traits: list | None, scene_summary: str = "", visual_cues: list[str] | None = None) -> str:
    parts: list[str] = []
    for trait in character_traits or []:
        if isinstance(trait, dict):
            for key in ("trait", "name", "label", "keyword"):
                value = trait.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
        else:
            text = str(trait).strip()
            if text:
                parts.append(text)
    if scene_summary.strip():
        parts.append(scene_summary.strip())
    if visual_cues:
        parts.extend([cue.strip() for cue in visual_cues if str(cue).strip()])
    query = " ".join(parts).strip()
    return query or "cinematic interview"


def _build_video_search_params(query: str, target_duration_seconds: int) -> dict[str, Any]:
    return {
        "query": query,
        "per_page": 15,
        "page": 1,
        "min_duration": max(target_duration_seconds - 1, 1),
        "max_duration": max(target_duration_seconds + 3, 4),
    }


def _pick_mp4_asset(video_item: dict[str, Any]) -> dict[str, Any] | None:
    video_files = video_item.get("video_files") or []
    candidates = []
    for asset in video_files:
        link = str(asset.get("link") or "").strip()
        file_type = str(asset.get("file_type") or "").lower()
        width = int(asset.get("width") or 0)
        height = int(asset.get("height") or 0)
        if not link:
            continue
        if "mp4" not in file_type and not link.lower().endswith(".mp4"):
            continue
        candidates.append((width * height, int(asset.get("fps") or 0), asset))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0] * 10 + item[1], reverse=True)
    return candidates[0][2]


def _validate_video_file(path: Path, min_size_bytes: int = 50 * 1024) -> bool:
    if not path.exists() or path.stat().st_size < min_size_bytes:
        return False

    if path.suffix.lower() != ".mp4":
        return False

    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        ok = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
        capture.release()
        return ok
    except Exception:
        return False


def _download_mp4(url: str, destination: Path, timeout: int = 120) -> None:
    response = requests.get(url, timeout=timeout, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"Video download failed ({response.status_code}): {response.text[:300]}")

    content_type = response.headers.get("Content-Type", "").lower()
    if "video" not in content_type and not url.lower().endswith(".mp4"):
        raise RuntimeError(f"Unexpected content type for video asset: {content_type or 'unknown'}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                handle.write(chunk)


def _retry_next_asset(video_results: list[dict[str, Any]], destination: Path) -> dict[str, Any]:
    for index, result in enumerate(video_results, start=1):
        asset = _pick_mp4_asset(result)
        if not asset:
            continue

        video_url = str(asset.get("link") or "").strip()
        if not video_url:
            continue

        print(
            f"[query_stock_footage] trying result={index} mp4_url={video_url.split('?')[0]} "
            f"width={asset.get('width', 'n/a')} height={asset.get('height', 'n/a')} fps={asset.get('fps', 'n/a')}"
        )
        _download_mp4(video_url, destination)
        if _validate_video_file(destination):
            return {
                "status": "success",
                "video_path": str(destination),
                "video_url": video_url,
                "source_type": "video",
                "selected_result_index": index,
                "asset": {
                    "width": asset.get("width"),
                    "height": asset.get("height"),
                    "fps": asset.get("fps"),
                    "file_type": asset.get("file_type"),
                },
            }

    raise RuntimeError("No playable .mp4 asset found in Pexels results.")


def query_stock_footage(
    character_traits: list | None = None,
    scene_summary: str = "",
    visual_cues: list[str] | None = None,
    output_path: str = "",
    target_duration_seconds: int = 5,
) -> Dict[str, Any]:
    normalized = {str(trait).strip().lower() for trait in (character_traits or []) if str(trait).strip()}
    if {"dark", "brooding", "mysterious"} & normalized:
        style = "noir documentary"
    elif {"energetic", "playful", "optimistic"} & normalized:
        style = "bright handheld commercial"
    else:
        style = "cinematic neutral"

    query = _build_search_query(character_traits, scene_summary=scene_summary, visual_cues=visual_cues)
    if isinstance(output_path, str) and output_path.endswith(".mp4"):
        destination = Path(output_path)
    else:
        safe_name = f"pexels_{int(time.time())}.mp4"
        destination = PHASE2_OUTPUTS_DIR / "raw_scenes" / safe_name
    destination.parent.mkdir(parents=True, exist_ok=True)

    api_key = _get_env_value("PEXELS_API_KEY", "pexels_api_key", "PexelsApiKey")
    if not api_key:
        raise RuntimeError("PEXELS_API_KEY is not configured.")

    endpoint = "https://api.pexels.com/videos/search"
    payload = _build_video_search_params(query=query, target_duration_seconds=target_duration_seconds)
    headers = {"Authorization": api_key}
    _log_video_fetch(endpoint, payload)

    response = requests.get(endpoint, headers=headers, params=payload, timeout=60)
    _log_video_fetch(endpoint, payload, response.json() if response.ok else {"status": response.status_code})
    if response.status_code != 200:
        raise RuntimeError(f"Pexels videos/search failed ({response.status_code}): {response.text[:500]}")

    data = response.json()
    videos = data.get("videos", [])
    print(
        f"[query_stock_footage] key fields: videos={len(videos)} first_has_video_files={bool(videos and videos[0].get('video_files'))}"
    )
    if not videos:
        raise RuntimeError("Pexels returned no video results.")

    result = _retry_next_asset(videos, destination)
    if not _validate_video_file(Path(result["video_path"])):
        raise RuntimeError("Downloaded Pexels asset was not a valid playable MP4.")

    return {
        "reference_style": style,
        "status": "success",
        **result,
    }
