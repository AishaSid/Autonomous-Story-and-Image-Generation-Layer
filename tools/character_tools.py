from __future__ import annotations

from typing import Any, Dict


def query_stock_footage(character_traits: list) -> Dict[str, Any]:
    normalized = {str(trait).strip().lower() for trait in character_traits}
    if {"dark", "brooding", "mysterious"} & normalized:
        style = "noir documentary, 35mm grain"
    elif {"energetic", "playful", "optimistic"} & normalized:
        style = "bright handheld commercial, soft bloom"
    else:
        style = "cinematic neutral, high dynamic range"

    return {"reference_style": style}
