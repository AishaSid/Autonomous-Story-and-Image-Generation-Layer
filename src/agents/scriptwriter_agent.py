from __future__ import annotations

import json
from typing import Any, Dict, List
from pathlib import Path

from state import State

from .common import build_llm_client, ensure_outputs_dir, load_character_names, resolve_llm_client
from .models import ScriptLLMResponse
from .visual_refiner_agent import refine_visual_cues


def _build_script_prompt(prompt: str, num_scenes: int, character_names: List[str]) -> str:
    character_line = ", ".join(character_names) if character_names else "the established characters"
    return (
        "You are a fast screenplay writer using Groq. Return strictly structured JSON with both characters and scenes. "
        "Every scene must include visual_cues that explicitly mention the exact character names provided. "
        "Do not paraphrase or rename character identities.\n\n"
        f"User prompt: {prompt}\n"
        f"Scene count: {num_scenes}\n"
        f"Character names to preserve exactly: {character_line}\n\n"
        "Rules:\n"
        "1) Include a top-level characters array with exact names, personality traits, appearance_description, and reference_style.\n"
        "2) Reuse the same character names consistently across all scenes.\n"
        "3) Each scene must contain scene_id, title, summary, dialogue_beats, visual_cues.\n"
        "4) visual_cues must be cinematic, specific, and include at least one exact character name.\n"
        "5) Keep dialogue_beats concise and scene-specific.\n"
        "6) Keep the overall story coherent across scenes.\n"
        "7) Output only content that fits the schema."
    )


def scriptwriter_node(state: State, llm_client: Any = None) -> Dict[str, Any]:
    prompt = state.get("user_prompt", "")
    num_scenes = int(state.get("num_scenes", 2))
    memory_character_names = load_character_names()

    llm = llm_client or resolve_llm_client(state, default_provider="groq")
    if llm is None:
        llm = build_llm_client(
            provider="groq",
            model=str(state.get("llm_model") or "llama-3.1-8b-instant"),
            temperature=float(state.get("llm_temperature", 0.3)),
        )

    structured_llm = llm.with_structured_output(ScriptLLMResponse)
    response = structured_llm.invoke(_build_script_prompt(prompt, num_scenes, memory_character_names))

    scenes = response.scenes[:num_scenes]
    characters = [character.model_dump() for character in response.characters]
    character_names = [character["name"] for character in characters if character.get("name")]

    if not characters:
        characters = [
            {
                "name": name,
                "personality_traits": [],
                "appearance_description": "",
                "reference_style": "cinematic neutral",
            }
            for name in memory_character_names
        ]
        character_names = [character["name"] for character in characters]

    if not characters:
        characters = [
            {
                "name": "Lead",
                "personality_traits": ["curious", "driven"],
                "appearance_description": "Explorer with practical travel gear and a focused expression.",
                "reference_style": "cinematic adventure, warm highlights",
            }
        ]
        character_names = ["Lead"]

    script = {
        "source": "groq_structured_output",
        "generated_at": "",
        "prompt": prompt,
        "characters": characters,
        "scenes": refine_visual_cues(scenes, character_names),
    }

    outputs_dir = ensure_outputs_dir()
    scene_manifest_payload = dict(script)
    (outputs_dir / "scene_manifest.json").write_text(json.dumps(scene_manifest_payload, indent=2), encoding="utf-8")
    (outputs_dir / "character_db.json").write_text(
        json.dumps({"characters": characters}, indent=2),
        encoding="utf-8",
    )

    return {"script": script, "status": "script_generated"}
