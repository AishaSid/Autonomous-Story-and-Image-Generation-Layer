from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents import invoke_mcp_tool_via_protocol, list_mcp_tools, scriptwriter_node, validator_node
from state import *


def _mcp_invoke(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve and invoke tools dynamically from the MCP registry."""

    return invoke_mcp_tool_via_protocol(tool_name=tool_name, payload=payload)


def mode_selector_node(state: State) -> Dict[str, str]:
    """Select processing mode and mark routing decision."""

    mode = state.get("input_mode", "manual")
    if mode not in {"manual", "auto"}:
        mode = "manual"
    return {"input_mode": mode, "status": f"mode_selected:{mode}"}


def hitl_node(state: State) -> Dict[str, str]:
    """Human-in-the-loop checkpoint node; execution resumes here after approval."""

    _ = state
    return {"status": "approved_by_human"}


def character_node(state: State) -> Dict[str, List[Dict[str, object]]]:
    """Create mock character entities from script context."""

    prompt = state.get("user_prompt", "")
    scenes = state.get("script", {}).get("scenes", []) if isinstance(state.get("script"), dict) else []

    inferred_traits = ["optimistic", "curious"]
    if scenes:
        inferred_traits.append("story_driven")

    style_result = _mcp_invoke(
        "query_stock_footage",
        {"character_traits": inferred_traits},
    )

    characters = [
        {
            "name": "Lead",
            "personality_traits": inferred_traits,
            "appearance_description": f"Derived from prompt: {prompt[:60]}",
            "reference_style": style_result["reference_style"],
        }
    ]
    return {"characters": characters, "status": "characters_created"}


def image_node(state: State) -> Dict[str, List[Dict[str, object]]]:
    """Generate mock image references from characters and script context."""

    prompt = state.get("user_prompt", "")
    images: List[Dict[str, object]] = []

    for character in state.get("characters", []):
        character_name = str(character.get("name", "Character"))
        traits = character.get("personality_traits", [])
        style_result = _mcp_invoke(
            "query_stock_footage",
            {"character_traits": [str(t) for t in traits]},
        )
        image_result = _mcp_invoke(
            "generate_image",
            {
                "prompt": f"{prompt} | {character_name} | {style_result['reference_style']}",
                "filename": f"{character_name}.jpg",
            },
        )
        images.append(
            {
                "character": character_name,
                "reference_style": style_result["reference_style"],
                "path": image_result["image_path"],
            }
        )

    return {"images": images, "status": "images_generated"}


def memory_commit_node(state: State) -> Dict[str, str]:
    """Persist generated artifacts into the memory layer via MCP-style tools."""

    script = state.get("script", {})
    scenes = script.get("scenes", []) if isinstance(script, dict) else []
    first_scene = scenes[0] if scenes else {}
    image_paths = [str(image.get("path", "")) for image in state.get("images", []) if image.get("path")]

    _mcp_invoke(
        "commit_memory",
        {
            "data": {
                "scene_id": str(first_scene.get("scene_id", "scene_001")),
                "content": str(first_scene.get("summary", state.get("user_prompt", ""))),
                "metadata": {"status": state.get("status", "")},
            },
            "collection_name": "script_history",
        },
    )

    for character in state.get("characters", []):
        character_name = str(character.get("name", "Unknown"))
        reference_image_path = next(
            (
                str(image.get("path", ""))
                for image in state.get("images", [])
                if image.get("character") == character_name
            ),
            "",
        )
        _mcp_invoke(
            "commit_memory",
            {
                "data": {
                    "name": character_name,
                    "personality_traits": [str(t) for t in character.get("personality_traits", [])],
                    "appearance_description": str(character.get("appearance_description", "")),
                    "reference_style": str(character.get("reference_style", "")),
                    "reference_image_path": reference_image_path,
                    "metadata": {"source": "graph_pipeline"},
                },
                "collection_name": "character_metadata",
            },
        )

    for image in state.get("images", []):
        _mcp_invoke(
            "commit_memory",
            {
                "data": {
                    "document": str(image.get("path", "")),
                    "metadata": {
                        "character": str(image.get("character", "")),
                        "reference_style": str(image.get("reference_style", "")),
                    },
                },
                "collection_name": "image_references",
            },
        )

    # Deliverable exports for downstream pipeline compatibility.
    manifest_payload = dict(state.get("script", {}))
    if isinstance(manifest_payload, dict):
        manifest_payload["scenes"] = [
            {
                **scene,
                "reference_image_paths": image_paths,
                "asset_context": {
                    "character_names": [str(character.get("name", "Unknown")) for character in state.get("characters", [])],
                    "image_paths": image_paths,
                },
            }
            for scene in scenes
        ]

    Path("scene_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )
    Path("character_db.json").write_text(
        json.dumps(
            {
                "characters": [
                    {
                        **character,
                        "reference_image_path": next(
                            (
                                str(image.get("path", ""))
                                for image in state.get("images", [])
                                if image.get("character") == character.get("name")
                            ),
                            "",
                        ),
                    }
                    for character in state.get("characters", [])
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {"status": "memory_committed"}


def _route_from_mode_selector(state: State) -> Literal["validator_node", "scriptwriter_node"]:
    mode = state.get("input_mode", "manual")
    return "validator_node" if mode == "manual" else "scriptwriter_node"


def build_graph():
    builder = StateGraph(State)

    builder.add_node("mode_selector_node", mode_selector_node)
    builder.add_node("validator_node", validator_node)
    builder.add_node("scriptwriter_node", scriptwriter_node)
    builder.add_node("hitl_node", hitl_node)
    builder.add_node("character_node", character_node)
    builder.add_node("image_node", image_node)
    builder.add_node("memory_commit_node", memory_commit_node)

    builder.add_edge(START, "mode_selector_node")
    builder.add_conditional_edges(
        "mode_selector_node",
        _route_from_mode_selector,
        {
            "validator_node": "validator_node",
            "scriptwriter_node": "scriptwriter_node",
        },
    )
    builder.add_edge("validator_node", "hitl_node")
    builder.add_edge("scriptwriter_node", "hitl_node")
    builder.add_edge("hitl_node", "character_node")
    builder.add_edge("character_node", "image_node")
    builder.add_edge("image_node", "memory_commit_node")
    builder.add_edge("memory_commit_node", END)

    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl_node"],
    )
    return graph


def save_graph_visualization(graph, output_path: str = "graph_visualization.png") -> None:
    """Save graph visualization to an image file."""

    png_data = graph.get_graph().draw_mermaid_png()
    with open(output_path, "wb") as file_obj:
        file_obj.write(png_data)


if __name__ == "__main__":
    graph = build_graph()
    save_graph_visualization(graph, output_path="graph_visualization.png")

    available_tools = list_mcp_tools()
    print("Discovered MCP tools:", available_tools)

    test_state = initial_state(
        user_prompt="An explorer enters an ancient floating city at sunrise.",
        input_mode="auto",
    )
    base_thread_id = "demo_auto_run_001"
    edit_counter = 0
    config = {"configurable": {"thread_id": base_thread_id}}
    interrupted_state = graph.invoke(test_state, config=config)

    while True:
        print("\nPaused before hitl_node. Awaiting approval...")
        current_script = interrupted_state.get("script", {})
        print("Current generated script:")
        print(json.dumps(current_script, indent=2))

        if interrupted_state.get("validation_feedback"):
            print("Validation feedback:")
            print(interrupted_state.get("validation_feedback"))

        user_choice = input("Do you approve this script? (y/n/edit): ").strip().lower()

        if user_choice == "y":
            break
        if user_choice == "n":
            raise RuntimeError("Execution halted: script not approved at HITL checkpoint")
        if user_choice == "edit":
            amendment_prompt = input("Enter amendment prompt: ").strip()
            if not amendment_prompt:
                print("No amendment provided. Returning to approval prompt.")
                continue

            amended_state = dict(interrupted_state)
            amended_state["user_prompt"] = (
                f"{interrupted_state.get('user_prompt', '')}\n"
                f"User amendment: {amendment_prompt}"
            ).strip()
            amended_state["status"] = "amendment_requested"

            edit_counter += 1
            config = {"configurable": {"thread_id": f"{base_thread_id}_edit_{edit_counter}"}}
            interrupted_state = graph.invoke(amended_state, config=config)
            continue

        print("Invalid input. Please enter y, n, or edit.")

    approved_final_state = graph.invoke(None, config=config)
    print("Execution resumed after approval.")
    print(approved_final_state)
