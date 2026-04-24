from __future__ import annotations

import base64
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ASSETS_DIR = PROJECT_ROOT / "outputs" / "image_assets"
IMAGE_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
NEGATIVE_VISUAL_TERMS = (
    "mountain landscape, architecture-only scene, empty building exterior, skyline with no people, "
    "forest scenery, abstract background, no human, faceless subject"
)


def _looks_like_image(binary_data: bytes) -> bool:
    if not binary_data:
        return False
    return (
        binary_data.startswith(b"\x89PNG\r\n\x1a\n")
        or binary_data.startswith(b"\xff\xd8\xff")
        or binary_data.startswith(b"GIF87a")
        or binary_data.startswith(b"GIF89a")
        or binary_data[:4] == b"RIFF" and binary_data[8:12] == b"WEBP"
    )


def _write_safe_placeholder(image_path: Path, seed: Optional[int] = None) -> None:
    """Writes a guaranteed valid PNG placeholder when all providers fail."""
    seed_value = int(seed) if seed is not None else random.randint(1, 2**31 - 1)
    rng = random.Random(seed_value)
    color = (rng.randint(40, 180), rng.randint(40, 180), rng.randint(40, 180))
    img = Image.new("RGB", (1024, 576), color=color)
    img.save(image_path, format="PNG")


def generate_character_image(
    refined_prompt: str,
    character_name: str,
    filename: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    return generate_character_image_with_seed(refined_prompt, character_name, filename=filename, seed=seed)


def generate_character_image_with_seed(
    refined_prompt: str,
    character_name: str,
    filename: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    stability_api_key = os.environ.get("STABILITY_API_KEY")
    safe_filename = Path(filename or f"{character_name}.jpg").name
    if not Path(safe_filename).suffix:
        safe_filename = f"{safe_filename}.png"
    image_path = IMAGE_ASSETS_DIR / safe_filename

    if stability_api_key:
        headers = {
            "Authorization": f"Bearer {stability_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "text_prompts": [
                {"text": refined_prompt, "weight": 1},
                {"text": NEGATIVE_VISUAL_TERMS, "weight": -1},
            ],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "steps": 30,
            "samples": 1,
            "seed": int(seed) if seed is not None else random.randint(1, 2**31 - 1),
        }

        response = requests.post(STABILITY_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            response_data = response.json()
            image_data = response_data["artifacts"][0]["base64"]
            image_path.write_bytes(base64.b64decode(image_data))
            return {
                "image_path": str(image_path),
                "character_name": character_name,
                "prompt_used": refined_prompt,
                "seed": int(seed) if seed is not None else None,
            }

    fallback_seed = quote_plus(
        str(seed if seed is not None else Path(safe_filename).stem or refined_prompt[:40] or character_name)
    )
    constrained_prompt = (
        f"{refined_prompt}. Primary subject is a person on screen, interview frame, visible face, speaking."
    )
    candidate_urls = [
        (
            "https://image.pollinations.ai/prompt/"
            f"{quote_plus(constrained_prompt)}?width=1024&height=576&seed={fallback_seed}&nologo=true&model=flux"
        ),
        (
            "https://image.pollinations.ai/prompt/"
            f"{quote_plus(constrained_prompt)}?width=1024&height=576&seed={fallback_seed}&nologo=true"
        ),
    ]

    for url in candidate_urls:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=20) as response:
                content_type = str(response.headers.get("Content-Type", "")).lower()
                body = response.read()
                if "image" not in content_type and not _looks_like_image(body):
                    continue
                if not _looks_like_image(body):
                    continue
                image_path.write_bytes(body)
            return {
                "image_path": str(image_path),
                "character_name": character_name,
                "prompt_used": refined_prompt,
                "seed": int(seed) if seed is not None else None,
            }
        except (URLError, TimeoutError, OSError):
            continue

    _write_safe_placeholder(image_path, seed=seed)
    return {
        "image_path": str(image_path),
        "character_name": character_name,
        "prompt_used": refined_prompt,
        "seed": int(seed) if seed is not None else None,
    }
