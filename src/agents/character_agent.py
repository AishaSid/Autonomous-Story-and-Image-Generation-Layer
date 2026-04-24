from __future__ import annotations

from typing import Dict, List

from state import State

from .common import invoke_mcp_tool_via_protocol


def character_node(state: State) -> Dict[str, List[Dict[str, object]]]:
    prompt = state.get("user_prompt", "")
    scenes = state.get("script", {}).get("scenes", []) if isinstance(state.get("script"), dict) else []

    inferred_traits = ["optimistic", "curious"]
    if scenes:
        inferred_traits.append("story_driven")

    style_result = invoke_mcp_tool_via_protocol("query_stock_footage", {"character_traits": inferred_traits})

    characters = [
        {
            "name": "Lead",
            "personality_traits": inferred_traits,
            "appearance_description": f"Derived from prompt: {prompt[:60]}",
            "reference_style": style_result["reference_style"],
        }
    ]
    return {"characters": characters, "status": "characters_created"}
