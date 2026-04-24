from __future__ import annotations

from typing import Any, Dict, List

from .models import ScriptScene


def refine_visual_cues(scenes: List[ScriptScene], character_names: List[str]) -> List[Dict[str, Any]]:
    normalized_names = [name.strip() for name in character_names if name.strip()]
    canonical_names = ", ".join(normalized_names)
    refined_scenes: List[Dict[str, Any]] = []

    for index, scene in enumerate(scenes, start=1):
        scene_payload = scene.model_dump()
        cues = [str(cue).strip() for cue in scene_payload.get("visual_cues", []) if str(cue).strip()]

        if normalized_names and not any(any(name in cue for name in normalized_names) for cue in cues):
            cues.insert(0, f"{canonical_names} in a cinematic shot with clear character focus.")

        if not cues:
            cues = [
                f"{canonical_names} in a cinematic establishing shot." if canonical_names else "Cinematic establishing shot."
            ]

        scene_payload["scene_id"] = scene_payload.get("scene_id") or f"scene_{index:03d}"
        scene_payload["visual_cues"] = cues
        refined_scenes.append(scene_payload)

    return refined_scenes


def refine_script_scenes(scenes: List[ScriptScene], character_names: List[str]) -> List[Dict[str, Any]]:
    return refine_visual_cues(scenes, character_names)
