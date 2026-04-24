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

def _select_primary_character(cue: str, character_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Finds which character the visual cue is focusing on."""
    cue_lower = cue.lower()
    for char in character_profiles:
        if char.get("name", "").lower() in cue_lower:
            return char
    return character_profiles[0] if character_profiles else {}

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
        
        frame_prompts = []
        for frame_index, cue in enumerate(cues, start=1):
            primary = _select_primary_character(cue, character_profiles)
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