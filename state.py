from typing import Any, Dict, List, Literal, TypedDict


class State(TypedDict):
    input_mode: Literal["manual", "auto"]
    script: Dict[str, Any]
    characters: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    audios: List[Dict[str, Any]]
    status: str
    user_prompt: str


def initial_state(user_prompt: str, input_mode: Literal["manual", "auto"]) -> State:
    return {
        "input_mode": input_mode,
        "script": {},
        "characters": [],
        "images": [],
        "audios": [],
        "status": "processing",
        "user_prompt": user_prompt,
    }
