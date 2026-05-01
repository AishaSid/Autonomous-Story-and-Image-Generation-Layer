from __future__ import annotations

from typing import Any, Dict, List, Literal, TypedDict


class State(TypedDict, total=False):
    user_prompt: str
    input_mode: Literal["manual", "auto"]
    status: str
    script: Dict[str, Any]
    character_db: Dict[str, Any]
    image_paths: List[str]
    image_prompts: List[Dict[str, Any]]
    tool_invocations: List[Dict[str, Any]]
    num_scenes: int
    llm_model: str


def initial_state(user_prompt: str, input_mode: Literal["manual", "auto"]) -> State:
    return {
        "user_prompt": user_prompt,
        "input_mode": input_mode,
        "status": "processing",
        "script": {},
        "character_db": {},
        "image_paths": [],
        "image_prompts": [],
        "tool_invocations": [],
    }
