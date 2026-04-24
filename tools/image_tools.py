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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ASSETS_DIR = PROJECT_ROOT / "outputs" / "image_assets"
IMAGE_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
NEGATIVE_VISUAL_TERMS = (
    "mountain landscape, architecture-only scene, empty building exterior, skyline with no people, "
    "forest scenery, abstract background, no human, faceless subject"
)


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
    safe_filename = Path(filename or f"{character_name}.jpg").name
    if not Path(safe_filename).suffix:
        safe_filename = f"{safe_filename}.png"
    image_path = IMAGE_ASSETS_DIR / safe_filename

    if STABILITY_API_KEY:
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
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
                image_path.write_bytes(response.read())
            return {
                "image_path": str(image_path),
                "character_name": character_name,
                "prompt_used": refined_prompt,
                "seed": int(seed) if seed is not None else None,
            }
        except (URLError, TimeoutError, OSError):
            continue

    jpeg_bytes = base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxAQEBAQEA8PEA8PDw8PDw8PDw8PDw8PFREWFhURExMYHSggGBolGxUVITEhJSkrLi4uFx8zODMtNygtLisBCgoKDg0OFxAQGC0dICUtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLv/AABEIAAEAAQMBIgACEQEDEQH/xAAbAAACAwEBAQAAAAAAAAAAAAAEBQADBgIBB//EADoQAAIBAwMCBAQEBQIHAAAAAAABAgMEEQUSITEGQVFhBxMiMnGBkaGxwQcjQlLR8BQzYpLh8P/EABkBAAMBAQEAAAAAAAAAAAAAAAIDBAEABf/EACQRAAICAQQCAgMAAAAAAAAAAAABAhEDIRIxBCJBUWEUMnGR/9oADAMBAAIRAxEAPwD4s3g8mGQ4aU1JkQvTjK2Y1XWvQ6u7fS8w2tQ4s0QkFh3UO1Qb0s0m5m5g0lq2a6gq0q9f8A6gq7mU0r2x5u1d0Vf2c8oU9d0p2d3y0X4n+f1k0R9m7bYfV3G2g8uCwQJ4aY0o7n4f2Q2u3yM0t4Wm2nX0f7d6fQp6yR7w6v4m3aHj3nJ5oXQk7y2sX0u0mQn1Hh0iWm4eQ1M3Jj3q8g3Gv4jzBf8A/9k="
    )
    image_path.write_bytes(jpeg_bytes)
    return {
        "image_path": str(image_path),
        "character_name": character_name,
        "prompt_used": refined_prompt,
        "seed": int(seed) if seed is not None else None,
    }
