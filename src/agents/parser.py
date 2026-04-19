from __future__ import annotations

import json
import operator
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

import cv2
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field, ValidationError

try:
    from src.agents.video_gen import generate_scene_video
    from tools.face_swapper import face_swapper
    from tools.identity_validator import identity_validator
    from tools.lip_sync_aligner import lip_sync_aligner
    from tools.voice_cloning_synthesizer import voice_cloning_synthesizer
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.agents.video_gen import generate_scene_video
    from tools.face_swapper import face_swapper
    from tools.identity_validator import identity_validator
    from tools.lip_sync_aligner import lip_sync_aligner
    from tools.voice_cloning_synthesizer import voice_cloning_synthesizer


class SceneModel(BaseModel):
    scene_id: str
    title: str
    summary: str
    dialogue_beats: list[Any] = Field(default_factory=list)
    visual_cues: list[str] = Field(default_factory=list)


class SceneManifestModel(BaseModel):
    source: str
    generated_at: str
    prompt: str
    scenes: list[SceneModel]


class CharacterModel(BaseModel):
    name: str
    personality_traits: list[str] = Field(default_factory=list)
    appearance_description: str = ""
    reference_style: str = ""


class CharacterDBModel(BaseModel):
    characters: list[CharacterModel] = Field(default_factory=list)


class ParserState(TypedDict, total=False):
    manifest_path: str
    phase1_root: str
    character_db_path: str
    image_assets_dir: str
    manifest: dict[str, Any]
    character_db: dict[str, Any]
    image_assets: list[str]
    tasks: list[dict[str, Any]]
    voice_outputs: Annotated[list[dict[str, Any]], operator.add]
    video_outputs: Annotated[list[dict[str, Any]], operator.add]
    face_swap_outputs: Annotated[list[dict[str, Any]], operator.add]
    fused_outputs: Annotated[list[dict[str, Any]], operator.add]
    shared_memory: dict[str, Any]
    checkpoints: list[str]
    errors: Annotated[list[str], operator.add]


class BranchInputState(TypedDict):
    scene_task: dict[str, Any]


def _phase1_root_for_manifest(manifest_path: str) -> Path:
    return Path(manifest_path).resolve().parent


def _character_db_path_for_manifest(manifest_path: str) -> Path:
    return _phase1_root_for_manifest(manifest_path) / "character_db.json"


def _image_assets_dir_for_manifest(manifest_path: str) -> Path:
    return _phase1_root_for_manifest(manifest_path) / "image_assets"


def _load_character_db(character_db_path: Path) -> CharacterDBModel:
    raw = json.loads(character_db_path.read_text(encoding="utf-8"))
    return CharacterDBModel.model_validate(raw)


def _discover_image_assets(image_assets_dir: Path) -> list[str]:
    if not image_assets_dir.exists():
        return []
    image_paths = list(image_assets_dir.glob("*.png"))
    image_paths.extend(list(image_assets_dir.glob("*.jpg")))
    image_paths.extend(list(image_assets_dir.glob("*.jpeg")))

    valid_assets: list[str] = []
    for image_path in sorted(image_paths):
        if cv2.imread(str(image_path)) is not None:
            valid_assets.append(str(image_path))
    return valid_assets


def _select_character_profile(character_db: CharacterDBModel, scene_index: int) -> dict[str, Any]:
    if not character_db.characters:
        return {}

    character = character_db.characters[scene_index % len(character_db.characters)]
    return character.model_dump()


def _map_visual_cue_assets(
    scene_index: int,
    visual_cues: list[str],
    image_assets: list[str],
) -> list[dict[str, str]]:
    if not visual_cues or not image_assets:
        return []

    mapped_assets: list[dict[str, str]] = []
    for cue_index, visual_cue in enumerate(visual_cues):
        asset_index = (scene_index * len(visual_cues) + cue_index) % len(image_assets)
        mapped_assets.append(
            {
                "visual_cue": visual_cue,
                "image_path": image_assets[asset_index],
            }
        )
    return mapped_assets


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


def get_task_graph(
    manifest: SceneManifestModel,
    character_db: CharacterDBModel,
    image_assets: list[str],
    phase1_root: str,
    character_db_path: str,
    image_assets_dir: str,
) -> list[dict[str, Any]]:
    """MCP tool contract: decompose each scene into parallel audio/video branches."""
    task_graph: list[dict[str, Any]] = []

    for scene_index, scene in enumerate(manifest.scenes):
        visual_cue_assets = _map_visual_cue_assets(scene_index, scene.visual_cues, image_assets)
        character_profile = _select_character_profile(character_db, scene_index)
        task_graph.append(
            {
                "scene_id": scene.scene_id,
                "scene_index": scene_index,
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
                            "visual_cue_assets": visual_cue_assets,
                            "reference_image_paths": [
                                asset["image_path"] for asset in visual_cue_assets
                            ],
                            "character_profile": character_profile,
                        },
                        "output": f"raw_scenes/{scene.scene_id}.mp4",
                    },
                },
                "asset_context": {
                    "phase1_root": phase1_root,
                    "character_db_path": character_db_path,
                    "image_assets_dir": image_assets_dir,
                    "character_profile": character_profile,
                    "visual_cue_assets": visual_cue_assets,
                    "reference_image_paths": [asset["image_path"] for asset in visual_cue_assets],
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


def validate_character_db_schema(character_db_path: str) -> tuple[bool, str]:
    try:
        raw = json.loads(Path(character_db_path).read_text(encoding="utf-8"))
        CharacterDBModel.model_validate(raw)
    except FileNotFoundError:
        return False, f"Character database not found: {character_db_path}"
    except json.JSONDecodeError as exc:
        return False, f"Character database is not valid JSON: {exc}"
    except ValidationError as exc:
        return False, f"Character database schema mismatch: {exc}"

    return True, "Character database schema is valid for Phase 2 scene parsing."


def _scene_parser_node_factory(checkpoint_dir: str):
    def _scene_parser_node(state: ParserState) -> ParserState:
        manifest_path = state["manifest_path"]
        phase1_root = _phase1_root_for_manifest(manifest_path)
        character_db_path = phase1_root / "character_db.json"
        image_assets_dir = phase1_root / "image_assets"
        manifest_raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        manifest = SceneManifestModel.model_validate(manifest_raw)
        character_db = _load_character_db(character_db_path)
        image_assets = _discover_image_assets(image_assets_dir)

        state["manifest"] = manifest.model_dump()
        state["character_db"] = character_db.model_dump()
        state["phase1_root"] = str(phase1_root)
        state["character_db_path"] = str(character_db_path)
        state["image_assets_dir"] = str(image_assets_dir)
        state["image_assets"] = image_assets
        state = commit_memory(
            state,
            "scene_manifest_loaded",
            {
                "manifest_path": manifest_path,
                "scene_count": len(manifest.scenes),
                "character_count": len(character_db.characters),
                "image_asset_count": len(image_assets),
                "character_db_path": str(character_db_path),
                "image_assets_dir": str(image_assets_dir),
            },
            checkpoint_dir=checkpoint_dir,
        )
        return state

    return _scene_parser_node


def _task_graph_node_factory(checkpoint_dir: str):
    def _task_graph_node(state: ParserState) -> ParserState:
        manifest = SceneManifestModel.model_validate(state["manifest"])
        character_db = CharacterDBModel.model_validate(state.get("character_db", {"characters": []}))
        image_assets = state.get("image_assets", [])
        tasks = get_task_graph(
            manifest,
            character_db,
            image_assets,
            state.get("phase1_root", ""),
            state.get("character_db_path", ""),
            state.get("image_assets_dir", ""),
        )
        state["tasks"] = tasks

        state = commit_memory(
            state,
            "task_graph_generated",
            {
                "tasks": len(tasks),
                "scene_ids": [task["scene_id"] for task in tasks],
                "image_assets_bound": len(image_assets),
            },
            checkpoint_dir=checkpoint_dir,
        )
        return state

    return _task_graph_node


def _dispatch_parallel_branches(state: ParserState) -> list[Send]:
    sends: list[Send] = []
    for task in state.get("tasks", []):
        sends.append(Send("voice_synth_node", {"scene_task": task}))
        sends.append(Send("video_gen_node", {"scene_task": task}))

    if not sends:
        sends.append(Send("finalize_node", {}))

    return sends


def _voice_synth_node(state: BranchInputState) -> dict[str, Any]:
    task = state["scene_task"]
    scene_id = task["scene_id"]
    dialogue_beats = task["parallel_branches"]["audio"]["inputs"]["dialogue_beats"]

    synthesized_audio = voice_cloning_synthesizer(
        scene_id=scene_id,
        dialogue_beats=dialogue_beats,
        output_path=task["parallel_branches"]["audio"]["output"],
    )

    return {
        "voice_outputs": [
            {
                "scene_id": scene_id,
                "audio_path": synthesized_audio,
                "status": "completed",
            }
        ]
    }


def _video_gen_node(state: BranchInputState) -> dict[str, Any]:
    task = state["scene_task"]
    scene_id = task["scene_id"]
    reference_image_paths = task["parallel_branches"]["video"]["inputs"].get("reference_image_paths", [])
    character_profile = task["parallel_branches"]["video"]["inputs"].get("character_profile", {})
    image_assets_dir = task.get("asset_context", {}).get("image_assets_dir", "")
    dialogue_beats = task["parallel_branches"]["audio"]["inputs"].get("dialogue_beats", [])
    audio_path = task["parallel_branches"]["audio"]["output"]

    generated_video, source_image_path = generate_scene_video(
        scene_id=scene_id,
        output_path=task["parallel_branches"]["video"]["output"],
        reference_image_paths=reference_image_paths,
        character_profile=character_profile,
        image_assets_dir=image_assets_dir,
        dialogue_beats=dialogue_beats,
        audio_path=audio_path,
    )

    return {
        "video_outputs": [
            {
                "scene_id": scene_id,
                "video_path": generated_video,
                "source_image_path": source_image_path,
                "status": "completed",
            }
        ]
    }


def _face_swap_node_factory(checkpoint_dir: str):
    def _face_swap_node(state: ParserState) -> dict[str, Any]:
        tasks = state.get("tasks", [])
        task_count = len(tasks)
        voice_count = len(state.get("voice_outputs", []))
        video_count = len(state.get("video_outputs", []))

        memory = state.setdefault("shared_memory", {})
        if memory.get("face_swap_complete"):
            return {}

        if voice_count < task_count or video_count < task_count:
            return {}

        video_by_scene = {item["scene_id"]: item["video_path"] for item in state.get("video_outputs", [])}
        face_swap_outputs: list[dict[str, Any]] = []
        errors: list[str] = []

        for task in tasks:
            scene_id = task["scene_id"]
            video_path = video_by_scene.get(scene_id)
            character_profile = task.get("asset_context", {}).get("character_profile", {})
            expected_character = str(character_profile.get("name", ""))
            reference_image_paths = task.get("asset_context", {}).get("reference_image_paths", [])
            expected_image_path = reference_image_paths[0] if reference_image_paths else ""
            character_db_path = str(task.get("asset_context", {}).get("character_db_path", ""))
            if not video_path:
                errors.append(f"Missing video output for {scene_id}.")
                continue

            identity_ok = identity_validator(
                scene_id=scene_id,
                video_path=video_path,
                character_db_path=character_db_path,
                expected_character=expected_character,
                expected_image_path=expected_image_path,
            )
            if not identity_ok:
                errors.append(f"Identity validation failed for {scene_id}.")
                continue

            mapped_video_path = face_swapper(
                scene_id=scene_id,
                input_video_path=video_path,
                output_path=f"output/face_swapped/{scene_id}.mp4",
            )
            face_swap_outputs.append(
                {
                    "scene_id": scene_id,
                    "video_path": mapped_video_path,
                    "identity_validated": True,
                    "expected_character": expected_character,
                    "reference_image_path": expected_image_path,
                    "status": "completed",
                }
            )

        memory["face_swap_complete"] = True
        snapshot_state: ParserState = {
            **state,
            "shared_memory": dict(memory),
            "checkpoints": list(state.get("checkpoints", [])),
            "errors": list(state.get("errors", [])) + errors,
        }
        snapshot_state = commit_memory(
            snapshot_state,
            "face_swap_complete",
            {
                "mapped_scenes": len(face_swap_outputs),
                "errors": len(errors),
            },
            checkpoint_dir=checkpoint_dir,
        )

        return {
            "face_swap_outputs": face_swap_outputs,
            "errors": errors,
            "shared_memory": snapshot_state["shared_memory"],
            "checkpoints": snapshot_state["checkpoints"],
        }

    return _face_swap_node


def _lip_sync_node_factory(checkpoint_dir: str):
    def _lip_sync_node(state: ParserState) -> dict[str, Any]:
        tasks = state.get("tasks", [])
        task_count = len(tasks)
        face_count = len(state.get("face_swap_outputs", []))

        memory = state.setdefault("shared_memory", {})
        if memory.get("lip_sync_complete"):
            return {}

        if not memory.get("face_swap_complete"):
            return {}

        if face_count < task_count:
            return {}

        audio_by_scene = {item["scene_id"]: item["audio_path"] for item in state.get("voice_outputs", [])}
        face_by_scene = {item["scene_id"]: item["video_path"] for item in state.get("face_swap_outputs", [])}

        fused_outputs: list[dict[str, Any]] = []
        errors: list[str] = []

        for task in tasks:
            scene_id = task["scene_id"]
            audio_path = audio_by_scene.get(scene_id)
            video_path = face_by_scene.get(scene_id)

            if not audio_path or not video_path:
                errors.append(f"Fusion inputs missing for {scene_id}.")
                continue

            fused_path = lip_sync_aligner(
                scene_id=scene_id,
                audio_path=audio_path,
                video_path=video_path,
                output_path=f"output/raw_scenes/{scene_id}.mp4",
            )
            fused_outputs.append(
                {
                    "scene_id": scene_id,
                    "output_path": fused_path,
                    "status": "completed",
                }
            )

        memory["lip_sync_complete"] = True
        snapshot_state: ParserState = {
            **state,
            "shared_memory": dict(memory),
            "checkpoints": list(state.get("checkpoints", [])),
            "errors": list(state.get("errors", [])) + errors,
        }
        snapshot_state = commit_memory(
            snapshot_state,
            "lip_sync_complete",
            {
                "fused_scenes": len(fused_outputs),
                "errors": len(errors),
            },
            checkpoint_dir=checkpoint_dir,
        )

        return {
            "fused_outputs": fused_outputs,
            "errors": errors,
            "shared_memory": snapshot_state["shared_memory"],
            "checkpoints": snapshot_state["checkpoints"],
        }

    return _lip_sync_node


def _finalize_node_factory(checkpoint_dir: str):
    def _finalize_node(state: ParserState) -> dict[str, Any]:
        tasks = state.get("tasks", [])
        task_count = len(tasks)
        fused_count = len(state.get("fused_outputs", []))

        memory = state.setdefault("shared_memory", {})
        if memory.get("parallel_orchestration_complete"):
            return {}

        if task_count == 0:
            memory["parallel_orchestration_complete"] = True
            snapshot_state: ParserState = {
                **state,
                "shared_memory": dict(memory),
                "checkpoints": list(state.get("checkpoints", [])),
            }
            snapshot_state = commit_memory(
                snapshot_state,
                "parser_complete",
                {
                    "task_count": 0,
                    "status": "no_tasks",
                },
                checkpoint_dir=checkpoint_dir,
            )
            return {
                "shared_memory": snapshot_state["shared_memory"],
                "checkpoints": snapshot_state["checkpoints"],
            }

        if not memory.get("lip_sync_complete"):
            return {}

        if fused_count < task_count:
            return {}

        memory["parallel_orchestration_complete"] = True
        snapshot_state = {
            **state,
            "shared_memory": dict(memory),
            "checkpoints": list(state.get("checkpoints", [])),
        }
        snapshot_state = commit_memory(
            snapshot_state,
            "parser_complete",
            {
                "task_count": task_count,
                "voice_count": len(state.get("voice_outputs", [])),
                "video_count": len(state.get("video_outputs", [])),
                "face_swap_count": len(state.get("face_swap_outputs", [])),
                "fused_count": fused_count,
                "status": "fusion_completed",
            },
            checkpoint_dir=checkpoint_dir,
        )
        return {
            "shared_memory": snapshot_state["shared_memory"],
            "checkpoints": snapshot_state["checkpoints"],
        }

    return _finalize_node


def build_parser_graph(checkpoint_dir: str = "state/checkpoints"):
    graph = StateGraph(ParserState)
    graph.add_node("scene_parser_node", _scene_parser_node_factory(checkpoint_dir))
    graph.add_node("task_graph_node", _task_graph_node_factory(checkpoint_dir))
    graph.add_node("voice_synth_node", _voice_synth_node)
    graph.add_node("video_gen_node", _video_gen_node)
    graph.add_node("face_swap_node", _face_swap_node_factory(checkpoint_dir))
    graph.add_node("lip_sync_node", _lip_sync_node_factory(checkpoint_dir))
    graph.add_node("finalize_node", _finalize_node_factory(checkpoint_dir))

    graph.add_edge(START, "scene_parser_node")
    graph.add_edge("scene_parser_node", "task_graph_node")
    graph.add_conditional_edges(
        "task_graph_node",
        _dispatch_parallel_branches,
        ["voice_synth_node", "video_gen_node", "finalize_node"],
    )
    graph.add_edge("voice_synth_node", "face_swap_node")
    graph.add_edge("video_gen_node", "face_swap_node")
    graph.add_edge("face_swap_node", "lip_sync_node")
    graph.add_edge("lip_sync_node", "finalize_node")
    graph.add_edge("finalize_node", END)

    return graph.compile()


def run_scene_parser(
    manifest_path: str,
    checkpoint_dir: str = "state/checkpoints",
) -> ParserState:
    app = build_parser_graph(checkpoint_dir=checkpoint_dir)
    initial_state: ParserState = {
        "manifest_path": manifest_path,
        "phase1_root": str(_phase1_root_for_manifest(manifest_path)),
        "character_db_path": str(_character_db_path_for_manifest(manifest_path)),
        "image_assets_dir": str(_image_assets_dir_for_manifest(manifest_path)),
        "image_assets": [],
        "voice_outputs": [],
        "video_outputs": [],
        "face_swap_outputs": [],
        "fused_outputs": [],
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
    recovered["phase1_root"] = str(_phase1_root_for_manifest(manifest_path))
    recovered["character_db_path"] = str(_character_db_path_for_manifest(manifest_path))
    recovered["image_assets_dir"] = str(_image_assets_dir_for_manifest(manifest_path))
    recovered.setdefault("voice_outputs", [])
    recovered.setdefault("video_outputs", [])
    recovered.setdefault("face_swap_outputs", [])
    recovered.setdefault("fused_outputs", [])
    recovered.setdefault("image_assets", [])
    recovered.setdefault("errors", [])
    return app.invoke(recovered)


if __name__ == "__main__":
    default_manifest = "phase1_inputs/scene_manifest.json"
    is_valid, message = validate_manifest_schema(default_manifest)
    print(message)

    if is_valid:
        final_state = run_scene_parser(default_manifest)
        print(f"Task graph generated for {len(final_state.get('tasks', []))} scenes.")
        print(f"Voice jobs: {len(final_state.get('voice_outputs', []))}")
        print(f"Video jobs: {len(final_state.get('video_outputs', []))}")
        print(f"Face swap jobs: {len(final_state.get('face_swap_outputs', []))}")
        print(f"Fusion outputs: {len(final_state.get('fused_outputs', []))}")
