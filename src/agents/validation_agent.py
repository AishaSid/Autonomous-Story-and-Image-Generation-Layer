from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from state import State

from .common import build_llm_client


def _extract_scene_headers(lines: List[str]) -> List[str]:
    header_pattern = re.compile(r"^(INT\.|EXT\.)", re.IGNORECASE)
    return [line for line in lines if header_pattern.match(line)]


def _extract_dialogue_labels(lines: List[str]) -> List[str]:
    dialogue_pattern = re.compile(r"^[A-Z][A-Z0-9 ]{1,30}:\s+.+")
    return [line for line in lines if dialogue_pattern.match(line)]


def _manual_script_to_json(script_text: str) -> Dict[str, Any]:
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
    if llm_client is None:
        try:
            llm_client = build_llm_client(provider="openai", model="gpt-4o-mini", temperature=0.2)
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
    script_text = state.get("user_prompt", "").strip()
    if not script_text:
        return {
            "status": "validation_failed",
            "validation_feedback": "Manual script is empty. Provide at least one scene header, dialogue, and action line.",
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
        return {
            "status": "validation_failed",
            "validation_feedback": _build_validation_feedback(script_text, issues, llm_client=llm_client),
        }

    return {
        "status": "validated_manual_input",
        "validation_feedback": "Validation passed.",
        "script": _manual_script_to_json(script_text),
    }
