from __future__ import annotations

from typing import Any, Dict, List
from state import State

def _validate_character_details(profile: Dict[str, Any]):
    """Ensure basic fields exist to prevent ValueError crashes."""
    if not profile.get("name"):
        profile["name"] = "The character"
    if not profile.get("appearance_description"):
        profile["appearance_description"] = "vividly detailed"
def _build_refined_prompt(character_profile: Dict[str, Any], cue: str, summary: str) -> str:
    """Builds a frame prompt that prioritizes visible humans and speaking action."""
    name = character_profile.get("name", "The character")
    appearance = character_profile.get("appearance_description", "striking features")
    clothing = character_profile.get("clothing", "appropriate clothing")
    item = character_profile.get("signature_item", "")
    style = character_profile.get("base_visual_style", "Cinematic 35mm film")
    
    item_str = f" holding {item}" if item else ""
    
    safe_cue = cue.split('.')[0] if cue else "a reporter speaking during an interview"
    safe_summary = summary.split(".")[0].strip() if summary else "short interview moment"

    return (
        f"{style}. Human subject in frame: {name}. "
        f"Shot type: medium close-up interview frame, chest-up composition, eye-level camera. "
        f"Action: {name} is speaking naturally to camera, visible mouth movement, expressive face. "
        f"Physical appearance: {appearance}. "
        f"Attire: {clothing}{item_str}. "
        f"Scene cue: {safe_cue}. "
        f"Story context: {safe_summary}. "
        "Environment supports interview context but remains secondary to the person. "
        "Realistic human anatomy, natural skin texture, detailed face, clear eyes, cinematic lighting. "
        "Must include at least one person in foreground. "
        "Avoid empty landscape or architecture-only composition. "
        "(No character blending, no duplicate faces.) --ar 16:9"
    )

def _select_primary_character(cue: str, character_profiles: List[Dict[str, Any]], frame_index: int) -> Dict[str, Any]:
    """Finds which character the visual cue is focusing on."""
    cue_lower = cue.lower()
    for char in character_profiles:
        if char.get("name", "").lower() in cue_lower:
            return char

    if len(character_profiles) >= 2:
        first = character_profiles[0]
        second = character_profiles[1]
        if any(token in cue_lower for token in ["female", "woman", "reporter", "interviewer", "she", "her"]):
            return second
        if any(token in cue_lower for token in ["male", "man", "interviewee", "he", "his"]):
            return first

    if character_profiles:
        return character_profiles[(frame_index - 1) % len(character_profiles)]
    return {}


def _ensure_transition_cues(cues: List[str], character_profiles: List[Dict[str, Any]]) -> List[str]:
    cleaned = [str(cue).strip() for cue in cues if str(cue).strip()]
    if len(cleaned) >= 4:
        return cleaned

    names = [str(char.get("name", "Character")).strip() for char in character_profiles if char.get("name")]
    male = names[0] if names else "Male interviewee"
    female = names[1] if len(names) > 1 else "Female interviewer"

    defaults = [
        f"{female} asks the first question in a close-up interview shot.",
        f"{male} answers while speaking directly in a close-up interview shot.",
        f"{male} continues speaking as {female} listens and takes notes.",
        f"{male} and {female} appear together in a two-shot showing interview transition.",
    ]

    for item in defaults:
        if len(cleaned) >= 4:
            break
        cleaned.append(item)
    return cleaned


def _select_scene_cues(cues: List[str], max_cues: int = 2) -> List[str]:
    """Selects a small set of cues with character + interaction coverage."""
    if len(cues) <= max_cues:
        return cues

    cues_lower = [cue.lower() for cue in cues]
    first_idx = 0
    interaction_idx = -1
    for idx, cue in enumerate(cues_lower):
        if "two-shot" in cue or "speaking beat" in cue or "speaking" in cue:
            interaction_idx = idx
            break

    selected_indices = [first_idx]
    if interaction_idx >= 0 and interaction_idx != first_idx:
        selected_indices.append(interaction_idx)
    elif len(cues) > 1:
        selected_indices.append(1)

    selected = [cues[idx] for idx in selected_indices][:max_cues]
    return selected

def refine_visual_cues(scenes: List[Any], character_names: List[str]) -> List[Dict[str, Any]]:
    """Helper used by scriptwriter to ensure scenes have base cues."""
    normalized_names = [name.strip() for name in character_names if name.strip()]
    canonical_names = ", ".join(normalized_names)
    refined_scenes: List[Dict[str, Any]] = []

    for index, scene in enumerate(scenes, start=1):
        scene_payload = scene.model_dump() if hasattr(scene, 'model_dump') else dict(scene)
        cues = [str(cue).strip() for cue in scene_payload.get("visual_cues", []) if str(cue).strip()]

        if not cues:
            cues = [f"{canonical_names} in a cinematic establishing shot." if canonical_names else "Cinematic establishing shot."]

        scene_payload["scene_id"] = scene_payload.get("scene_id") or f"scene_{index:03d}"
        scene_payload["visual_cues"] = cues
        refined_scenes.append(scene_payload)

    return refined_scenes

def visual_refiner_node(state: State) -> Dict[str, Any]:
    print("--- REFINING VISUAL PROMPTS ---")
    script = state.get("script", {})
    scenes = script.get("scenes", [])
    character_profiles = state.get("characters", [])
    
    if not character_profiles:
        return {"status": "refiner_failed_no_characters"}

    refined_scenes = []
    for scene_index, scene in enumerate(scenes, start=1):
        scene_summary = scene.get("summary", "")
        cues = scene.get("visual_cues", ["Cinematic frame with clear blocking."])
        scene["scene_id"] = scene.get("scene_id") or f"scene_{scene_index:03d}"
        cues = _ensure_transition_cues(cues, character_profiles)
        cues = _select_scene_cues(cues, max_cues=2)
        
        frame_prompts = []
        for frame_index, cue in enumerate(cues, start=1):
            primary = _select_primary_character(cue, character_profiles, frame_index)
            _validate_character_details(primary)
            
            refined_prompt = _build_refined_prompt(primary, cue, scene_summary)
            
            frame_prompts.append({
                "frame_id": f"{scene['scene_id']}_frame_{frame_index:02d}",
                "primary_character": primary.get("name"),
                "visual_cue": cue,
                "refined_prompt": refined_prompt
            })
        
        scene["frame_prompts"] = frame_prompts
        refined_scenes.append(scene)
        
    return {"script": {"scenes": refined_scenes}, "status": "visuals_refined"}