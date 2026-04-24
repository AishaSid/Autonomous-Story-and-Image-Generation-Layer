from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from state import State

from .common import PROJECT_ROOT, ensure_outputs_dir, invoke_mcp_tool_via_protocol, outputs_path


def memory_commit_node(state: State) -> Dict[str, str]:
    script = state.get("script", {})
    scenes = script.get("scenes", []) if isinstance(script, dict) else []
    first_scene = scenes[0] if scenes else {}
    image_paths = [str(image.get("path", "")) for image in state.get("images", []) if image.get("path")]
    scene_image_map = {}
    for image in state.get("images", []):
        scene_id = str(image.get("scene_id", ""))
        if scene_id and image.get("path"):
            scene_image_map.setdefault(scene_id, []).append(str(image.get("path", "")))

    invoke_mcp_tool_via_protocol(
        "commit_memory",
        {
            "data": {
                "scene_id": str(first_scene.get("scene_id", "scene_001")),
                "content": str(first_scene.get("summary", state.get("user_prompt", ""))),
                "metadata": {"status": state.get("status", "")},
            },
            "collection_name": "script_history",
        },
    )

    for character in state.get("characters", []):
        character_name = str(character.get("name", "Unknown"))
        reference_image_path = next(
            (str(image.get("path", "")) for image in state.get("images", []) if image.get("character") == character_name),
            "",
        )
        invoke_mcp_tool_via_protocol(
            "commit_memory",
            {
                "data": {
                    "name": character_name,
                    "age": str(character.get("age", "")),
                    "personality_traits": [str(t) for t in character.get("personality_traits", [])],
                    "appearance_description": str(character.get("appearance_description", "")),
                    "clothing": str(character.get("clothing", "")),
                    "hair_texture": str(character.get("hair_texture", "")),
                    "eye_color": str(character.get("eye_color", "")),
                    "signature_item": str(character.get("signature_item", "")),
                    "base_visual_style": str(character.get("base_visual_style", character.get("reference_style", ""))),
                    "reference_style": str(character.get("base_visual_style", character.get("reference_style", ""))),
                    "reference_image_path": reference_image_path,
                    "metadata": {"source": "graph_pipeline"},
                },
                "collection_name": "character_metadata",
            },
        )

    for image in state.get("images", []):
        invoke_mcp_tool_via_protocol(
            "commit_memory",
            {
                "data": {
                    "document": str(image.get("path", "")),
                    "metadata": {
                        "character": str(image.get("character", "")),
                        "character_names": [str(name) for name in image.get("character_names", [])],
                        "scene_id": str(image.get("scene_id", "")),
                        "frame_id": str(image.get("frame_id", "")),
                        "visual_cue": str(image.get("visual_cue", "")),
                        "reference_style": str(image.get("reference_style", "")),
                    },
                },
                "collection_name": "image_references",
            },
        )

    manifest_payload = dict(state.get("script", {}))
    if isinstance(manifest_payload, dict):
        manifest_payload["scenes"] = [
            {
                **scene,
                "reference_image_paths": image_paths,
                "frame_image_paths": scene_image_map.get(str(scene.get("scene_id", "")), []),
                "asset_context": {
                    "character_names": [str(character.get("name", "Unknown")) for character in state.get("characters", [])],
                    "image_paths": image_paths,
                    "frame_image_paths": scene_image_map.get(str(scene.get("scene_id", "")), []),
                },
            }
            for scene in scenes
        ]

    outputs_dir = ensure_outputs_dir()
    (outputs_dir / "scene_manifest.json").write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    (outputs_dir / "character_db.json").write_text(
        json.dumps(
            {
                "characters": [
                    {
                        **character,
                        "age": str(character.get("age", "")),
                        "reference_image_path": next(
                            (
                                str(image.get("path", ""))
                                for image in state.get("images", [])
                                if image.get("character") == character.get("name")
                            ),
                            "",
                        ),
                    }
                    for character in state.get("characters", [])
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {"status": "memory_committed"}
