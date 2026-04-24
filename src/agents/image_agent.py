from __future__ import annotations

from typing import Dict, List

from state import State

from .common import PROJECT_ROOT, character_index, invoke_mcp_tool_via_protocol, load_json_file, resolve_current_scene


def image_node(state: State) -> Dict[str, List[Dict[str, object]]]:
    manifest = load_json_file(PROJECT_ROOT / "scene_manifest.json")
    character_db = load_json_file(PROJECT_ROOT / "character_db.json")
    character_lookup = character_index(character_db)
    current_scene = resolve_current_scene(state, manifest)

    scene_id = str(current_scene.get("scene_id", "scene_001"))
    scene_summary = str(current_scene.get("summary", "")).strip()
    visual_cues = current_scene.get("visual_cues", []) if isinstance(current_scene, dict) else []
    action_text = scene_summary or (
        str(visual_cues[0]).strip() if isinstance(visual_cues, list) and visual_cues else "the current scene action"
    )

    asset_context = current_scene.get("asset_context", {}) if isinstance(current_scene, dict) else {}
    character_names = asset_context.get("character_names", []) if isinstance(asset_context, dict) else []
    if not character_names:
        character_names = [str(character.get("name", "Character")) for character in state.get("characters", []) if character.get("name")]

    images: List[Dict[str, object]] = []

    for character_name in character_names:
        character_name = str(character_name).strip()
        if not character_name:
            continue

        character_key = character_name.lower()
        character_entry = character_lookup.get(character_key, {})
        appearance = str(character_entry.get("appearance_description", "")).strip() or character_name
        reference_style = str(character_entry.get("reference_style", "")).strip() or "cinematic portrait"
        refined_prompt = f"{appearance} performing {action_text} in {reference_style}"

        image_result = invoke_mcp_tool_via_protocol(
            "generate_character_image",
            {"refined_prompt": refined_prompt, "character_name": character_name},
        )
        images.append(
            {
                "scene_id": scene_id,
                "character": character_name,
                "appearance_description": appearance,
                "reference_style": reference_style,
                "refined_prompt": refined_prompt,
                "path": image_result["image_path"],
            }
        )

    return {"images": images, "status": "images_generated"}
