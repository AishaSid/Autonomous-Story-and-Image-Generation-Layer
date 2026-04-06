from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from state import State


def _build_llm_client(provider: str, model: str, temperature: float = 0.2) -> Any:
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise ValueError("provider must be 'openai' or 'google'")


def _resolve_llm_client(state: State, llm_client: Any = None) -> Any:
    """Resolve explicit client first, otherwise build from optional state config."""

    if llm_client is not None:
        return llm_client

    provider = state.get("llm_provider")
    model = state.get("llm_model")
    temperature = float(state.get("llm_temperature", 0.3))

    if isinstance(provider, str) and isinstance(model, str) and provider in {"openai", "google"}:
        return _build_llm_client(provider=provider, model=model, temperature=temperature)

    return None


def _get_server_params() -> StdioServerParameters:
    """Configure stdio launch parameters for the external MCP server process."""

    server_path = Path(__file__).with_name("mcp_server.py")
    return StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )


def _extract_call_result_payload(call_result: Any) -> Dict[str, Any]:
    """Normalize MCP call_tool result into a plain JSON-serializable dictionary."""

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
    server_params = _get_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return sorted([tool.name for tool in tools_result.tools])


def list_mcp_tools() -> List[str]:
    """List available tools by dynamically connecting to the MCP server over stdio."""

    return asyncio.run(_list_mcp_tools_async())


async def _invoke_mcp_tool_async(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    server_params = _get_server_params()
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            call_result = await session.call_tool(tool_name, payload)
            return _extract_call_result_payload(call_result)


def invoke_mcp_tool_via_protocol(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke an MCP tool via stdio protocol instead of local direct imports."""

    return asyncio.run(_invoke_mcp_tool_async(tool_name, payload))


def scriptwriter_node(state: State, llm_client: Any = None) -> Dict[str, Any]:
    """Generate script JSON via MCP tool invocation with structured LLM output."""

    prompt = state.get("user_prompt", "")
    num_scenes = int(state.get("num_scenes", 2))
    _ = _resolve_llm_client(state, llm_client=llm_client)

    provider: Optional[str] = state.get("llm_provider") if isinstance(state.get("llm_provider"), str) else None
    model: Optional[str] = state.get("llm_model") if isinstance(state.get("llm_model"), str) else None

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "num_scenes": num_scenes,
        "llm_provider": provider if provider in {"openai", "google"} else None,
        "llm_model": model,
        "temperature": float(state.get("llm_temperature", 0.3)),
    }

    script = invoke_mcp_tool_via_protocol("generate_script_segment", payload)
    return {"script": script, "status": "script_generated"}


def _extract_scene_headers(lines: List[str]) -> List[str]:
    header_pattern = re.compile(r"^(INT\.|EXT\.)", re.IGNORECASE)
    return [line for line in lines if header_pattern.match(line)]


def _extract_dialogue_labels(lines: List[str]) -> List[str]:
    dialogue_pattern = re.compile(r"^[A-Z][A-Z0-9 ]{1,30}:\s+.+")
    return [line for line in lines if dialogue_pattern.match(line)]


def _manual_script_to_json(script_text: str) -> Dict[str, Any]:
    """Convert validated manual script text into standardized scene JSON."""

    lines = [line.rstrip() for line in script_text.splitlines() if line.strip()]
    scenes: List[Dict[str, Any]] = []
    current_scene: Optional[Dict[str, Any]] = None

    header_pattern = re.compile(r"^(INT\.|EXT\.)", re.IGNORECASE)
    dialogue_pattern = re.compile(r"^([A-Z][A-Z0-9 ]{1,30}):\s+(.+)")

    for line in lines:
        if header_pattern.match(line):
            if current_scene:
                scenes.append(current_scene)
            current_scene = {
                "scene_id": f"scene_{len(scenes) + 1:03d}",
                "title": line,
                "summary": "",
                "dialogue_beats": [],
                "visual_cues": [],
            }
            continue

        if current_scene is None:
            continue

        dialogue_match = dialogue_pattern.match(line)
        if dialogue_match:
            speaker = dialogue_match.group(1).strip()
            utterance = dialogue_match.group(2).strip()
            current_scene["dialogue_beats"].append(f"{speaker}: {utterance}")
            continue

        current_scene["visual_cues"].append(line)

    if current_scene:
        scenes.append(current_scene)

    for scene in scenes:
        summary_source = scene["visual_cues"][0] if scene["visual_cues"] else "Action beat pending"
        scene["summary"] = str(summary_source)

    return {
        "source": "manual_validated",
        "generated_at": "",
        "prompt": "manual_script",
        "scenes": scenes,
    }


def _build_validation_feedback(script_text: str, issues: List[str], llm_client: Any = None) -> str:
    """Build actionable validator feedback using an LLM when available."""

    if llm_client is None:
        provider = "openai"
        model = "gpt-4o-mini"
        try:
            llm_client = _build_llm_client(provider=provider, model=model, temperature=0.2)
        except Exception:
            llm_client = None

    issue_text = "\n".join([f"- {issue}" for issue in issues])
    if llm_client is None:
        return (
            "Manual script validation failed. Actionable fixes:\n"
            f"{issue_text}\n"
            "Expected format example: INT. LAB - NIGHT | ALEX: Dialogue line | Action description line"
        )

    prompt = (
        "You are a screenplay validator assistant. The script failed structural checks. "
        "Return concise, actionable corrections with concrete line-level guidance.\n\n"
        f"Detected issues:\n{issue_text}\n\n"
        "Script:\n"
        f"{script_text}\n\n"
        "Output as plain text bullet points only."
    )
    response = llm_client.invoke(prompt)
    return str(getattr(response, "content", response)).strip()


def validator_node(state: State, llm_client: Any = None) -> Dict[str, Any]:
    """Validate manual scripts and return actionable correction feedback when invalid."""

    script_text = state.get("user_prompt", "").strip()
    if not script_text:
        feedback = "Manual script is empty. Provide at least one scene header, dialogue, and action line."
        return {
            "status": "validation_failed",
            "validation_feedback": feedback,
        }

    lines = [line.strip() for line in script_text.splitlines() if line.strip()]
    scene_headers = _extract_scene_headers(lines)
    dialogue_lines = _extract_dialogue_labels(lines)
    action_lines = [line for line in lines if line not in scene_headers and line not in dialogue_lines]

    issues: List[str] = []
    if not scene_headers:
        issues.append("Missing scene headers. Add headers starting with INT. or EXT. (example: INT. OFFICE - DAY).")
    if not dialogue_lines:
        issues.append("Missing labeled dialogue. Use SPEAKER: dialogue format (example: ALEX: We have to move now.).")
    if not action_lines:
        issues.append("Missing action descriptions. Add one or more descriptive action lines under each scene.")

    if issues:
        feedback = _build_validation_feedback(script_text, issues, llm_client=llm_client)
        return {
            "status": "validation_failed",
            "validation_feedback": feedback,
        }

    script = _manual_script_to_json(script_text)
    return {
        "status": "validated_manual_input",
        "validation_feedback": "Validation passed.",
        "script": script,
    }
