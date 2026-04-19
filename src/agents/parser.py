from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError


class SceneModel(BaseModel):
    scene_id: str
    title: str
    summary: str
    dialogue_beats: list[str] = Field(default_factory=list)
    visual_cues: list[str] = Field(default_factory=list)


class SceneManifestModel(BaseModel):
    source: str
    generated_at: str
    prompt: str
    scenes: list[SceneModel]


class ParserState(TypedDict, total=False):
    manifest_path: str
    manifest: dict[str, Any]
    tasks: list[dict[str, Any]]
    shared_memory: dict[str, Any]
    checkpoints: list[str]
    errors: list[str]


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def commit_memory(
    state: ParserState,
    step: str,
    payload: dict[str, Any],
    checkpoint_dir: str = "state/checkpoints",
) -> ParserState:
    """Persist parser state snapshots so interrupted runs can be resumed."""
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    memory = state.setdefault("shared_memory", {})
    memory[step] = payload

    snapshot = {
        "timestamp": _utc_timestamp(),
        "step": step,
        "state": state,
    }
    checkpoint_file = checkpoint_root / f"{snapshot['timestamp']}_{step}.json"
    checkpoint_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    checkpoints = state.setdefault("checkpoints", [])
    checkpoints.append(str(checkpoint_file))
    return state


def load_latest_checkpoint(checkpoint_dir: str = "state/checkpoints") -> ParserState | None:
    checkpoint_root = Path(checkpoint_dir)
    if not checkpoint_root.exists():
        return None

    checkpoint_files = sorted(checkpoint_root.glob("*.json"))
    if not checkpoint_files:
        return None

    latest = checkpoint_files[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return payload.get("state")


def get_task_graph(manifest: SceneManifestModel) -> list[dict[str, Any]]:
    """MCP tool contract: decompose each scene into parallel audio/video branches."""
    task_graph: list[dict[str, Any]] = []

    for scene in manifest.scenes:
        task_graph.append(
            {
                "scene_id": scene.scene_id,
                "parallel_branches": {
                    "audio": {
                        "agent": "voice_synthesis_agent",
                        "inputs": {
                            "dialogue_beats": scene.dialogue_beats,
                        },
                        "output": f"output/{scene.scene_id}.wav",
                    },
                    "video": {
                        "agent": "video_generation_agent",
                        "inputs": {
                            "summary": scene.summary,
                            "visual_cues": scene.visual_cues,
                        },
                        "output": f"raw_scenes/{scene.scene_id}.mp4",
                    },
                },
                "post_processors": [
                    {
                        "agent": "face_swap_agent",
                        "depends_on": ["video"],
                    },
                    {
                        "agent": "lip_sync_agent",
                        "depends_on": ["audio", "video"],
                    },
                ],
            }
        )

    return task_graph


def validate_manifest_schema(manifest_path: str) -> tuple[bool, str]:
    try:
        raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        SceneManifestModel.model_validate(raw)
    except FileNotFoundError:
        return False, f"Manifest not found: {manifest_path}"
    except json.JSONDecodeError as exc:
        return False, f"Manifest is not valid JSON: {exc}"
    except ValidationError as exc:
        return False, f"Manifest schema mismatch: {exc}"

    return True, "Manifest schema is valid for Phase 2 scene parsing."


def _scene_parser_node_factory(checkpoint_dir: str):
    def _scene_parser_node(state: ParserState) -> ParserState:
        manifest_path = state["manifest_path"]
        manifest_raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest = SceneManifestModel.model_validate(manifest_raw)

        state["manifest"] = manifest.model_dump()
        state = commit_memory(
            state,
            "scene_manifest_loaded",
            {
                "manifest_path": manifest_path,
                "scene_count": len(manifest.scenes),
            },
            checkpoint_dir=checkpoint_dir,
        )
        return state

    return _scene_parser_node


def _task_graph_node_factory(checkpoint_dir: str):
    def _task_graph_node(state: ParserState) -> ParserState:
        manifest = SceneManifestModel.model_validate(state["manifest"])
        tasks = get_task_graph(manifest)
        state["tasks"] = tasks

        state = commit_memory(
            state,
            "task_graph_generated",
            {
                "tasks": len(tasks),
                "scene_ids": [task["scene_id"] for task in tasks],
            },
            checkpoint_dir=checkpoint_dir,
        )
        return state

    return _task_graph_node


def _finalize_node_factory(checkpoint_dir: str):
    def _finalize_node(state: ParserState) -> ParserState:
        task_count = len(state.get("tasks", []))
        state = commit_memory(
            state,
            "parser_complete",
            {
                "task_count": task_count,
                "status": "ready_for_parallel_execution",
            },
            checkpoint_dir=checkpoint_dir,
        )
        return state

    return _finalize_node


def build_parser_graph(checkpoint_dir: str = "state/checkpoints"):
    graph = StateGraph(ParserState)
    graph.add_node("scene_parser_node", _scene_parser_node_factory(checkpoint_dir))
    graph.add_node("task_graph_node", _task_graph_node_factory(checkpoint_dir))
    graph.add_node("finalize_node", _finalize_node_factory(checkpoint_dir))

    graph.add_edge(START, "scene_parser_node")
    graph.add_edge("scene_parser_node", "task_graph_node")
    graph.add_edge("task_graph_node", "finalize_node")
    graph.add_edge("finalize_node", END)

    return graph.compile()


def run_scene_parser(
    manifest_path: str,
    checkpoint_dir: str = "state/checkpoints",
) -> ParserState:
    app = build_parser_graph(checkpoint_dir=checkpoint_dir)
    initial_state: ParserState = {
        "manifest_path": manifest_path,
        "shared_memory": {},
        "checkpoints": [],
        "errors": [],
    }
    return app.invoke(initial_state)


def resume_scene_parser(
    manifest_path: str,
    checkpoint_dir: str = "state/checkpoints",
) -> ParserState:
    app = build_parser_graph(checkpoint_dir=checkpoint_dir)
    recovered = load_latest_checkpoint(checkpoint_dir=checkpoint_dir)

    if recovered is None:
        return run_scene_parser(manifest_path=manifest_path, checkpoint_dir=checkpoint_dir)

    recovered["manifest_path"] = manifest_path
    return app.invoke(recovered)


if __name__ == "__main__":
    default_manifest = "phase1_inputs/scene_manifest.json"
    is_valid, message = validate_manifest_schema(default_manifest)
    print(message)

    if is_valid:
        final_state = run_scene_parser(default_manifest)
        print(f"Task graph generated for {len(final_state.get('tasks', []))} scenes.")
