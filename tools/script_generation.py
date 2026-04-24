from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ScriptCharacter(BaseModel):
    name: str = Field(..., description="Exact character name")
    personality_traits: List[str] = Field(default_factory=list)
    appearance_description: str = Field(..., description="Visual identity and appearance details")
    reference_style: str = Field(..., description="Style guidance for image generation")


class ScriptScene(BaseModel):
    scene_id: str
    title: str
    summary: str
    dialogue_beats: List[str]
    visual_cues: List[str]


class ScriptLLMResponse(BaseModel):
    characters: List[ScriptCharacter] = Field(default_factory=list)
    scenes: List[ScriptScene] = Field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_script_prompt(user_prompt: str, num_scenes: int) -> str:
    return (
        "You are a screenplay writing assistant. Produce exactly the requested number of scenes "
        "in a coherent cinematic narrative.\n\n"
        f"User prompt: {user_prompt}\n"
        f"Required scene count: {num_scenes}\n\n"
        "Hard constraints:\n"
        "1) Keep character identity and behavior consistent across scenes. "
        "If a character is established as reckless in Scene 1, they must remain reckless in Scene 2.\n"
        "2) Each scene must include: scene_id, title, summary, dialogue_beats, visual_cues.\n"
        "3) dialogue_beats must be concise and labeled dialogue-like beats.\n"
        "4) visual_cues must contain shot-level cinematic directions.\n"
        "5) Do not add fields outside the schema.\n"
    )


def _build_llm_client(provider: Literal["openai", "google", "groq"], model: str, temperature: float = 0.3) -> Any:
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    from langchain_groq import ChatGroq

    return ChatGroq(model=model or "llama-3.1-8b-instant", temperature=temperature)


def generate_script_segment(
    prompt: str,
    num_scenes: int,
    llm_provider: Optional[Literal["openai", "google", "groq"]] = None,
    llm_model: Optional[str] = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    runtime_client = None
    if llm_provider and llm_model:
        runtime_client = _build_llm_client(llm_provider, llm_model, temperature)

    if runtime_client is not None:
        structured_llm = runtime_client.with_structured_output(ScriptLLMResponse)
        llm_result = structured_llm.invoke(_build_script_prompt(prompt, num_scenes))
        scenes = llm_result.scenes

        if len(scenes) < num_scenes:
            raise ValueError("LLM returned fewer scenes than requested")

        normalized_scenes = [
            ScriptScene(
                scene_id=scene.scene_id or f"scene_{index:03d}",
                title=scene.title,
                summary=scene.summary,
                dialogue_beats=scene.dialogue_beats,
                visual_cues=scene.visual_cues,
            ).model_dump()
            for index, scene in enumerate(scenes[:num_scenes], start=1)
        ]

        return {
            "source": "llm_mcp_tool",
            "generated_at": _utc_now_iso(),
            "prompt": prompt,
            "scenes": normalized_scenes,
        }

    scenes: List[Dict[str, Any]] = []
    for index in range(1, num_scenes + 1):
        scenes.append(
            {
                "scene_id": f"scene_{index:03d}",
                "title": f"Scene {index}: Story Beat",
                "summary": f"A mock progression of the prompt: {prompt}",
                "dialogue_beats": [
                    "Character A establishes intent.",
                    "Character B introduces conflict.",
                    "Both characters align on a next action.",
                ],
                "visual_cues": [
                    "Wide establishing shot",
                    "Medium over-the-shoulder dialogue",
                    "Close-up emotional reaction",
                ],
            }
        )

    return {
        "source": "mock_mcp_tool",
        "generated_at": _utc_now_iso(),
        "prompt": prompt,
        "scenes": scenes,
    }
