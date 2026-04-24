from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple

from state import State

from .common import PROJECT_ROOT, character_index, invoke_mcp_tool_via_protocol, load_json_file, outputs_path


def _seed_from_name(name: str) -> int:
    digest = hashlib.sha256(name.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _character_aliases(name: str) -> List[str]:
    cleaned = name.strip().lower()
    parts = [part for part in cleaned.replace("-", " ").split() if part]
    aliases = {cleaned}

    if parts:
        aliases.add(parts[-1])
    if len(parts) >= 2:
        aliases.add(" ".join(parts[-2:]))

    if "explorer" in cleaned:
        aliases.add("explorer")
    if "archivist" in cleaned:
        aliases.add("archivist")

    return [alias for alias in aliases if alias]


def _cue_mentions_character(cue: str, character_name: str) -> bool:
    cue_text = cue.lower()
    return any(alias in cue_text for alias in _character_aliases(character_name))


def _select_prompt_characters(character_profiles: List[Dict[str, object]], cue: str) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    if not character_profiles:
        empty = {"name": "Lead"}
        return empty, []

    mentioned = [
        profile
        for profile in character_profiles
        if _cue_mentions_character(cue, str(profile.get("name", "")))
    ]

    if not mentioned:
        return character_profiles[0], []

    primary = mentioned[0]
    supporting = [
        profile
        for profile in mentioned[1:]
        if str(profile.get("name", "")).strip().lower() != str(primary.get("name", "")).strip().lower()
    ]
    return primary, supporting


def _to_participle(action_text: str) -> str:
    words = action_text.split()
    if not words:
        return action_text

    verb_map = {
        "steps": "stepping",
        "step": "stepping",
        "walks": "walking",
        "walk": "walking",
        "stands": "standing",
        "stand": "standing",
        "gazes": "gazing",
        "gaze": "gazing",
        "consults": "consulting",
        "consult": "consulting",
        "examines": "examining",
        "examine": "examining",
        "readies": "readying",
        "ready": "readying",
        "hovers": "hovering",
        "hover": "hovering",
        "widens": "widening",
        "widen": "widening",
        "enters": "entering",
        "enter": "entering",
    }
    first = words[0].lower()
    if first in verb_map:
        words[0] = verb_map[first]
    return " ".join(words)


def _extract_action_from_cue(cue: str, primary_name: str) -> str:
    action = cue.strip().rstrip(".")
    primary_aliases = sorted(_character_aliases(primary_name), key=len, reverse=True)

    for alias in primary_aliases:
        for prefix in (f"the {alias} ", f"{alias} "):
            if action.lower().startswith(prefix):
                action = action[len(prefix) :].strip()
                break

    action = _to_participle(action)
    return action or "holding a tense pause"


def _build_character_description(profile: Dict[str, object]) -> str:
    appearance = str(profile.get("appearance_description", "")).strip()
    clothing = str(profile.get("clothing", "")).strip()
    hair = str(profile.get("hair_texture", "")).strip()
    eyes = str(profile.get("eye_color", "")).strip()
    signature = str(profile.get("signature_item", "")).strip()

    bits = [bit for bit in [appearance, clothing, hair] if bit]
    if not bits:
        name = str(profile.get("name", "Unknown")).strip() or "Unknown"
        raise ValueError(f"Missing physical description fields for character '{name}'")
    description = ", ".join(bits)
    if eyes:
        description += f", {eyes} eyes"
    if signature:
        description += f", marked by {signature}"
    return description


def _build_supporting_description(profiles: List[Dict[str, object]]) -> str:
    names = [str(profile.get("name", "")).strip() for profile in profiles if str(profile.get("name", "")).strip()]
    if not names:
        return ""

    if len(names) == 1:
        return f"with {names[0]} in the frame"

    return "with " + ", ".join(names[:-1]) + f" and {names[-1]} in the frame"


def _build_super_prompt(
    primary_profile: Dict[str, object],
    supporting_profiles: List[Dict[str, object]],
    cue: str,
    scene: Dict[str, object],
) -> str:
    style = str(primary_profile.get("base_visual_style", primary_profile.get("reference_style", ""))).strip() or "Cinematic 35mm film shot"
    camera_style = style if "shot" in style.lower() else f"{style} shot"

    primary_description = _build_character_description(primary_profile)
    action = _extract_action_from_cue(cue, str(primary_profile.get("name", "")))
    support = _build_supporting_description(supporting_profiles)
    if support:
        action = f"{action}, {support}"

    background = str(scene.get("summary", "")).strip() or str(scene.get("title", "")).strip() or "story-rich environment"
    lighting = "misty morning light"
    cue_lower = cue.lower()
    if any(token in cue_lower for token in ["night", "dark", "moon"]):
        lighting = "dramatic low-key night lighting"
    elif any(token in cue_lower for token in ["sunset", "dusk", "golden"]):
        lighting = "warm golden-hour light"
    elif any(token in cue_lower for token in ["engine", "plaza", "archive", "interior"]):
        lighting = "focused cinematic rim lighting"

    return f"{camera_style}, {primary_description} performing {action}, {background}, {lighting}."


def _load_scene_list(state: State) -> List[Dict[str, object]]:
    script = state.get("script", {})
    if isinstance(script, dict) and isinstance(script.get("scenes"), list) and script.get("scenes"):
        return [scene for scene in script.get("scenes", []) if isinstance(scene, dict)]

    manifest = load_json_file(outputs_path("scene_manifest.json"))
    if isinstance(manifest, dict) and isinstance(manifest.get("scenes"), list) and manifest.get("scenes"):
        return [scene for scene in manifest.get("scenes", []) if isinstance(scene, dict)]

    return []


def image_node(state: State) -> Dict[str, List[Dict[str, object]]]:
    script = state.get("script", {})
    scenes = _load_scene_list(state)
    characters = script.get("characters", []) if isinstance(script, dict) else []

    if not characters:
        characters = state.get("characters", [])

    character_db = load_json_file(outputs_path("character_db.json"))
    if not character_db:
        character_db = load_json_file(PROJECT_ROOT / "character_db.json")
    character_lookup = character_index(character_db)
    character_names = [str(character.get("name", "Character")) for character in characters if isinstance(character, dict) and character.get("name")]

    images: List[Dict[str, object]] = []

    for scene_index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, dict):
            continue

        scene_id = str(scene.get("scene_id", f"{scene_index:03d}"))
        scene_id_token = scene_id.removeprefix("scene_")
        frame_entries = scene.get("frame_prompts", []) if isinstance(scene.get("frame_prompts"), list) else []
        frame_entries = [entry for entry in frame_entries if isinstance(entry, dict)]
        if not frame_entries:
            raise ValueError(
                f"image_node requires structured frame_prompts from visual_refiner for scene '{scene_id}'"
            )

        scene_character_names = [
            str(entry.get("primary_character", "")).strip()
            for entry in frame_entries
            if str(entry.get("primary_character", "")).strip()
        ]
        scene_character_names = list(dict.fromkeys(scene_character_names))
        if not scene_character_names:
            scene_character_names = character_names
        if not scene_character_names:
            raise ValueError(f"image_node could not resolve primary character names for scene '{scene_id}'")

        scene_image_paths: List[str] = []
        character_seeds = {character_name: _seed_from_name(character_name) for character_name in scene_character_names}
        for frame_index, frame_entry in enumerate(frame_entries, start=1):
            primary_character_name = str(frame_entry.get("primary_character", "")).strip()
            cue = str(frame_entry.get("visual_cue", "")).strip()
            refined_prompt = str(frame_entry.get("refined_prompt", "")).strip()

            if not primary_character_name:
                raise ValueError(f"Missing primary_character for scene '{scene_id}' frame {frame_index}")
            if not refined_prompt:
                raise ValueError(f"Missing refined_prompt for scene '{scene_id}' frame {frame_index}")

            if primary_character_name.lower() not in character_lookup:
                raise ValueError(
                    f"Primary character '{primary_character_name}' in scene '{scene_id}' frame {frame_index} "
                    "was not found in character_db"
                )

            resolved_character_names = scene_character_names
            seed = character_seeds.get(primary_character_name, _seed_from_name(primary_character_name))

            image_result = invoke_mcp_tool_via_protocol(
                "generate_character_image",
                {
                    "refined_prompt": refined_prompt,
                    "character_name": primary_character_name,
                    "filename": f"scene_{scene_id_token}_frame_{frame_index:02d}.png",
                    "seed": seed,
                },
            )

            image_path = image_result["image_path"]
            scene_image_paths.append(image_path)
            images.append(
                {
                    "scene_id": scene_id,
                    "frame_id": f"{scene_id}_frame_{frame_index:02d}",
                    "character_names": resolved_character_names,
                    "character_seeds": character_seeds,
                    "primary_character": primary_character_name,
                    "character_seed": seed,
                    "super_prompt": refined_prompt,
                    "visual_cue": cue,
                    "refined_prompt": refined_prompt,
                    "path": image_path,
                }
            )

        scene["frame_image_paths"] = scene_image_paths

    return {"images": images, "status": "images_generated"}
