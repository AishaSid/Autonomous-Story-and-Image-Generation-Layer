from __future__ import annotations

from typing import Any, Dict, List

from state import State

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


def _character_aliases(name: str) -> List[str]:
    cleaned = name.strip().lower()
    parts = [part for part in cleaned.replace("-", " ").split() if part]
    aliases = {cleaned}
    if parts:
        aliases.add(parts[-1])
    if "explorer" in cleaned:
        aliases.add("explorer")
    if "archivist" in cleaned:
        aliases.add("archivist")
    return [alias for alias in aliases if alias]


def _select_primary_character(cue: str, character_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    cue_text = cue.lower()
    for profile in character_profiles:
        name = str(profile.get("name", "")).strip()
        if not name:
            continue
        if any(alias in cue_text for alias in _character_aliases(name)):
            return profile

    if character_profiles:
        return character_profiles[0]

    raise ValueError("visual_refiner_node requires at least one character profile")


def _extract_action(cue: str, primary_name: str) -> str:
    action = cue.strip().rstrip(".")
    lowered = action.lower()
    for alias in sorted(_character_aliases(primary_name), key=len, reverse=True):
        for prefix in (f"the {alias} ", f"{alias} "):
            if lowered.startswith(prefix):
                action = action[len(prefix) :].strip()
                lowered = action.lower()
                break
    return action if action else cue.strip().rstrip(".")


def _validate_character_details(profile: Dict[str, Any]) -> None:
    required_fields = ["appearance_description", "clothing", "signature_item"]
    missing = [field for field in required_fields if not str(profile.get(field, "")).strip()]
    if missing:
        name = str(profile.get("name", "Unknown"))
        raise ValueError(
            f"Missing required character details for '{name}': {', '.join(missing)}"
        )


def _build_refined_prompt(primary: Dict[str, Any], cue: str, scene_summary: str) -> str:
    name = str(primary.get("name", "")).strip()
    appearance = str(primary.get("appearance_description", "")).strip()
    clothing = str(primary.get("clothing", "")).strip()
    signature_item = str(primary.get("signature_item", "")).strip()
    action = _extract_action(cue, name)
    environment = scene_summary.strip() or "story-rich environment"
    physical_details = f"{appearance}, carrying {signature_item}" if signature_item else appearance
    return (
        f"Cinematic 35mm film, {name} with {physical_details} wearing {clothing}, "
        f"{action}, {environment}."
    )


def visual_refiner_node(state: State) -> Dict[str, Any]:
    script = state.get("script", {})
    if not isinstance(script, dict):
        return {"status": "visual_refinement_skipped", "script": script}

    state_characters = state.get("characters", []) if isinstance(state.get("characters"), list) else []
    script_characters = script.get("characters", []) if isinstance(script.get("characters"), list) else []

    character_profiles = [character for character in state_characters if isinstance(character, dict)] or [
        character for character in script_characters if isinstance(character, dict)
    ]
    character_names = [
        str(character.get("name", "")).strip()
        for character in character_profiles
        if isinstance(character, dict) and character.get("name")
    ]

    if not character_names:
        raise ValueError("visual_refiner_node could not find character names in state['characters'] or script['characters']")

    refined_scenes: List[Dict[str, Any]] = []
    scenes = script.get("scenes", []) if isinstance(script.get("scenes"), list) else []

    for scene_index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue

        scene_payload = dict(scene)
        cues = [str(cue).strip() for cue in scene_payload.get("visual_cues", []) if str(cue).strip()]
        if character_names and not any(any(name in cue for name in character_names) for cue in cues):
            cues.insert(0, f"{', '.join(character_names)} in a cinematic frame with consistent identity and clear blocking.")

        if not cues:
            cues = ["Cinematic frame with clear blocking."]

        scene_payload["scene_id"] = scene_payload.get("scene_id") or f"scene_{scene_index:03d}"
        scene_payload["visual_cues"] = cues
        scene_summary = str(scene_payload.get("summary", "")).strip()
        frame_prompts: List[Dict[str, str]] = []
        for frame_index, cue in enumerate(cues, start=1):
            primary_profile = _select_primary_character(cue, character_profiles)
            _validate_character_details(primary_profile)
            primary_name = str(primary_profile.get("name", "")).strip()
            refined_prompt = _build_refined_prompt(primary_profile, cue, scene_summary)
            frame_prompts.append(
                {
                    "frame_id": f"{scene_payload['scene_id']}_frame_{frame_index:02d}",
                    "primary_character": primary_name,
                    "visual_cue": cue,
                    "refined_prompt": refined_prompt,
                }
            )

        scene_payload["frame_prompts"] = frame_prompts
        refined_scenes.append(scene_payload)

    script["scenes"] = refined_scenes
    return {"script": script, "status": "visuals_refined"}
