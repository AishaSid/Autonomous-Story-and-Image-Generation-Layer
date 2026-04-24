from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # This looks for the .env file and loads the key automatically

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from state import State

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json_file(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_character_names() -> List[str]:
    payload = load_json_file(PROJECT_ROOT / "character_db.json")
    characters = payload.get("characters", []) if isinstance(payload, dict) else []
    names: List[str] = []

    for character in characters:
        if isinstance(character, dict) and character.get("name"):
            name = str(character["name"]).strip()
            if name:
                names.append(name)

    return names


def build_llm_client(provider: str, model: str, temperature: float = 0.2) -> Any:
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model or "llama-3.1-8b-instant", temperature=temperature)

    raise ValueError("provider must be 'openai', 'google', or 'groq'")


def resolve_llm_client(state: State, llm_client: Any = None, default_provider: str = "groq") -> Any:
    if llm_client is not None:
        return llm_client

    provider = state.get("llm_provider")
    model = state.get("llm_model")
    temperature = float(state.get("llm_temperature", 0.3))

    if isinstance(provider, str) and isinstance(model, str) and provider in {"openai", "google", "groq"}:
        return build_llm_client(provider=provider, model=model, temperature=temperature)

    if default_provider == "groq":
        return build_llm_client(provider="groq", model=str(model or "llama-3.1-8b-instant"), temperature=temperature)

    if isinstance(model, str):
        return build_llm_client(provider=default_provider, model=model, temperature=temperature)

    return None


def get_server_params() -> StdioServerParameters:
    server_path = PROJECT_ROOT / "mcp_server.py"
    return StdioServerParameters(command=sys.executable, args=[str(server_path)])


def extract_call_result_payload(call_result: Any) -> Dict[str, Any]:
    def _normalize(payload: Dict[str, Any]) -> Dict[str, Any]:
        nested = payload.get("result")
        if isinstance(nested, dict):
            return nested
        return payload

    structured = getattr(call_result, "structuredContent", None)
    if isinstance(structured, dict):
        return _normalize(structured)

    content = getattr(call_result, "content", None)
    if isinstance(content, list):
        for item in content:
            item_text = getattr(item, "text", None)
            if isinstance(item_text, str):
                try:
                    parsed = json.loads(item_text)
                    if isinstance(parsed, dict):
                        return _normalize(parsed)
                except json.JSONDecodeError:
                    continue

            if isinstance(item, dict) and isinstance(item.get("text"), str):
                try:
                    parsed = json.loads(item["text"])
                    if isinstance(parsed, dict):
                        return _normalize(parsed)
                except json.JSONDecodeError:
                    continue

    raise RuntimeError("MCP tool response did not contain structured JSON output")


async def _list_mcp_tools_async() -> List[str]:
    server_params = get_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return sorted([tool.name for tool in tools_result.tools])


def list_mcp_tools() -> List[str]:
    return asyncio.run(_list_mcp_tools_async())


async def _invoke_mcp_tool_async(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    server_params = get_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            call_result = await session.call_tool(tool_name, payload)
            return extract_call_result_payload(call_result)


def invoke_mcp_tool_via_protocol(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return asyncio.run(_invoke_mcp_tool_async(tool_name, payload))


def resolve_current_scene(state: State, manifest: Dict[str, Any]) -> Dict[str, Any]:
    script = state.get("script", {})
    script_scenes = script.get("scenes", []) if isinstance(script, dict) else []
    manifest_scenes = manifest.get("scenes", []) if isinstance(manifest, dict) else []

    if not manifest_scenes:
        return {}

    if isinstance(script_scenes, list) and script_scenes:
        script_scene_ids = {
            str(scene.get("scene_id", ""))
            for scene in script_scenes
            if isinstance(scene, dict) and scene.get("scene_id")
        }
        for scene in manifest_scenes:
            if isinstance(scene, dict) and str(scene.get("scene_id", "")) in script_scene_ids:
                return scene

    return manifest_scenes[0] if isinstance(manifest_scenes[0], dict) else {}


def character_index(character_db: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    characters = character_db.get("characters", []) if isinstance(character_db, dict) else []
    indexed: Dict[str, Dict[str, Any]] = {}

    for character in characters:
        if isinstance(character, dict) and character.get("name"):
            indexed[str(character["name"]).strip().lower()] = character

    return indexed
