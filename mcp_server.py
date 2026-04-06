from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from memory import MemoryLayer

mcp = FastMCP("writer-room-mcp-server")


class ScriptScene(BaseModel):
    scene_id: str
    title: str
    summary: str
    dialogue_beats: List[str]
    visual_cues: List[str]


class ScriptLLMResponse(BaseModel):
    scenes: List[ScriptScene]


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
        "If Character A is established as reckless in Scene 1, they must remain reckless in Scene 2.\n"
        "2) Each scene must include: scene_id, title, summary, dialogue_beats, visual_cues.\n"
        "3) dialogue_beats must be concise and labeled dialogue-like beats.\n"
        "4) visual_cues must contain shot-level cinematic directions.\n"
        "5) Do not add fields outside the schema.\n"
    )


def build_llm_client(provider: Literal["openai", "google"], model: str, temperature: float = 0.3) -> Any:
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise ValueError("provider must be 'openai' or 'google'")


@mcp.tool()
def generate_script_segment(
    prompt: str,
    num_scenes: int,
    llm_provider: Optional[Literal["openai", "google"]] = None,
    llm_model: Optional[str] = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Generate a structured screenplay using an LLM with strict Pydantic schema enforcement."""

    runtime_client = None
    if llm_provider and llm_model:
        runtime_client = build_llm_client(llm_provider, llm_model, temperature)

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


@mcp.tool()
def commit_memory(data: Dict[str, Any], collection_name: Literal["script_history", "character_metadata", "image_references"]) -> Dict[str, Any]:
    """Commit data to the persistent memory layer collections."""

    memory = MemoryLayer()

    if collection_name == "script_history":
        record_id = memory.add_script_segment(
            scene_id=str(data.get("scene_id", "scene_mock")),
            content=str(data.get("content", "")),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else None,
            record_id=str(data.get("record_id")) if data.get("record_id") else None,
        )
    elif collection_name == "character_metadata":
        record_id = memory.add_character(
            name=str(data.get("name", "Unknown")),
            personality_traits=[str(item) for item in data.get("personality_traits", [])],
            appearance_description=str(data.get("appearance_description", "")),
            reference_style=str(data.get("reference_style", "")),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else None,
            record_id=str(data.get("record_id")) if data.get("record_id") else None,
        )
    else:
        record_id = str(data.get("record_id") or uuid4())
        memory.image_references.add(
            ids=[record_id],
            documents=[str(data.get("document", data.get("prompt", "")))],
            metadatas=[
                {
                    "created_at": _utc_now_iso(),
                    **(data.get("metadata") if isinstance(data.get("metadata"), dict) else {}),
                }
            ],
        )

    return {
        "status": "success",
        "collection_name": collection_name,
        "record_id": record_id,
    }


@mcp.tool()
def query_stock_footage(character_traits: List[str]) -> Dict[str, Any]:
    """Return a reference style string based on character traits."""

    normalized = {trait.strip().lower() for trait in character_traits}
    if {"dark", "brooding", "mysterious"} & normalized:
        style = "noir documentary, 35mm grain"
    elif {"energetic", "playful", "optimistic"} & normalized:
        style = "bright handheld commercial, soft bloom"
    else:
        style = "cinematic neutral, high dynamic range"

    return {"reference_style": style}


@mcp.tool()
def generate_image(prompt: str) -> Dict[str, Any]:
    """Mock local ComfyUI/Stable Diffusion generation and return image path."""

    file_id = uuid4().hex
    output_dir = Path("image_assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"generated_{file_id}.png"

    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xff\xff?\x00\x05\xfe\x02\xfe"
        b"\xa7\x9d\xa4\x91\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    image_path.write_bytes(png_bytes)
    _ = prompt

    return {
        "status": "queued_local_mock",
        "image_path": str(image_path),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
