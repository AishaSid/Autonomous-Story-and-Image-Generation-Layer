from __future__ import annotations

from typing import Any, Dict, List

from state import State

from .common import build_llm_client, load_character_names, resolve_llm_client
from .models import ScriptLLMResponse
from .visual_refiner_agent import refine_visual_cues


def _build_script_prompt(prompt: str, num_scenes: int, character_names: List[str]) -> str:
    character_line = ", ".join(character_names) if character_names else "the established characters"
    return (
        "You are a fast screenplay writer using Groq. Return strictly structured scene JSON. "
        "Every scene must include visual_cues that explicitly mention the exact character names provided. "
        "Do not paraphrase or rename character identities.\n\n"
        f"User prompt: {prompt}\n"
        f"Scene count: {num_scenes}\n"
        f"Character names to preserve exactly: {character_line}\n\n"
        "Rules:\n"
        "1) Each scene must contain scene_id, title, summary, dialogue_beats, visual_cues.\n"
        "2) visual_cues must be cinematic, specific, and include at least one exact character name.\n"
        "3) Keep dialogue_beats concise and scene-specific.\n"
        "4) Keep the overall story coherent across scenes.\n"
        "5) Output only content that fits the schema."
    )


def scriptwriter_node(state: State, llm_client: Any = None) -> Dict[str, Any]:
    prompt = state.get("user_prompt", "")
    num_scenes = int(state.get("num_scenes", 2))
    character_names = load_character_names()

    llm = llm_client or resolve_llm_client(state, default_provider="groq")
    if llm is None:
        llm = build_llm_client(
            provider="groq",
            model=str(state.get("llm_model") or "llama-3.1-8b-instant"),
            temperature=float(state.get("llm_temperature", 0.3)),
        )

    structured_llm = llm.with_structured_output(ScriptLLMResponse)
    response = structured_llm.invoke(_build_script_prompt(prompt, num_scenes, character_names))

    scenes = response.scenes[:num_scenes]
    script = {
        "source": "groq_structured_output",
        "generated_at": "",
        "prompt": prompt,
        "scenes": refine_visual_cues(scenes, character_names),
    }

    return {"script": script, "status": "script_generated"}
