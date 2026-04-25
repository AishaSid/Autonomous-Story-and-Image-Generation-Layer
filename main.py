from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal
from dotenv import load_dotenv
load_dotenv()  # This looks for the .env file and loads the key automatically
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents import character_node, image_node, list_mcp_tools, memory_commit_node, scriptwriter_node, validator_node, visual_refiner_node
from src.agents.common import ensure_outputs_dir, thread_state_path
from state import State, initial_state


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


def _route_from_mode_selector(state: State) -> Literal["validator_node", "scriptwriter_node"]:
    mode = state.get("input_mode", "manual")
    return "validator_node" if mode == "manual" else "scriptwriter_node"


def _route_from_scriptwriter(state: State) -> Literal["hitl_node", "character_node"]:
    mode = state.get("input_mode", "manual")
    return "hitl_node" if mode == "manual" else "character_node"


def build_graph(interrupt_before_character: bool = True):
    builder = StateGraph(State)

    builder.add_node("mode_selector_node", mode_selector_node)
    builder.add_node("validator_node", validator_node)
    builder.add_node("scriptwriter_node", scriptwriter_node)
    builder.add_node("visual_refiner_node", visual_refiner_node)
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
    builder.add_conditional_edges(
        "scriptwriter_node",
        _route_from_scriptwriter,
        {
            "hitl_node": "hitl_node",
            "character_node": "character_node",
        },
    )
    builder.add_edge("hitl_node", "character_node")
    builder.add_edge("character_node", "visual_refiner_node")
    builder.add_edge("visual_refiner_node", "image_node")
    builder.add_edge("image_node", "memory_commit_node")
    builder.add_edge("memory_commit_node", END)

    checkpointer = InMemorySaver()
    interrupt_nodes = ["character_node"] if interrupt_before_character else []
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_nodes)
    return graph
if __name__ == "__main__":
    graph = build_graph(interrupt_before_character=True)

    available_tools = list_mcp_tools()
    print("Discovered MCP tools:", available_tools)

    test_state = initial_state(
        user_prompt=(
            "At first light, a storm-worn explorer and a meticulous archivist enter a floating city of fractured bridges "
            "and glass archives, uncovering a dormant celestial engine beneath the central plaza."
        ),
        input_mode="auto",
    )
    test_state["num_scenes"] = 3
    test_state["llm_model"] = "llama-3.3-70b-versatile"
    base_thread_id = "demo_auto_run_001"
    config = {"configurable": {"thread_id": base_thread_id}}
    paused_state = graph.invoke(test_state, config=config)

    paused_state_path = thread_state_path(base_thread_id)
    paused_state_path.write_text(json.dumps(paused_state, indent=2), encoding="utf-8")

    print("\nPipeline paused before character/image generation.")
    print(f"Thread ID: {base_thread_id}")
    print(f"Edit files in phase1_outputs/, then run: python resume.py {base_thread_id}")

