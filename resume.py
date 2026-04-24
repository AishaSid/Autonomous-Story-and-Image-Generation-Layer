from __future__ import annotations

import json
import sys
from pathlib import Path

from main import build_graph
from src.agents.common import outputs_path, thread_state_path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python resume.py <thread_id>")
        return 1

    thread_id = sys.argv[1].strip()
    if not thread_id:
        print("thread_id cannot be empty")
        return 1

    graph = build_graph(interrupt_before_character=False)
    config = {"configurable": {"thread_id": thread_id}}

    try:
        final_state = graph.invoke(None, config=config)
    except Exception as exc:
        paused_state_path = thread_state_path(thread_id)
        if not paused_state_path.exists():
            print(f"Unable to resume thread '{thread_id}': {exc}")
            return 1

        paused_state = json.loads(paused_state_path.read_text(encoding="utf-8"))

        scene_manifest = _load_json(outputs_path("scene_manifest.json"))
        character_db = _load_json(outputs_path("character_db.json"))
        if isinstance(scene_manifest, dict):
            paused_state["script"] = scene_manifest
        if isinstance(character_db, dict) and isinstance(character_db.get("characters"), list):
            paused_state["characters"] = character_db.get("characters", [])

        final_state = graph.invoke(paused_state, config=config)

    paused_state_path = thread_state_path(thread_id)
    paused_state_path.write_text(json.dumps(final_state, indent=2), encoding="utf-8")

    print(json.dumps(final_state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
