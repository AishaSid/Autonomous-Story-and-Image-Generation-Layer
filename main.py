from __future__ import annotations

import json
from typing import Any, Dict, Literal
from dotenv import load_dotenv
load_dotenv()  # This looks for the .env file and loads the key automatically
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents import character_node, image_node, list_mcp_tools, memory_commit_node, scriptwriter_node, validator_node
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
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["hitl_node"])
    return graph
if __name__ == "__main__":
    graph = build_graph()

    available_tools = list_mcp_tools()
    print("Discovered MCP tools:", available_tools)

    test_state = initial_state(
        user_prompt="An explorer enters an ancient floating city at sunrise.",
        input_mode="auto",
    )
    base_thread_id = "demo_auto_run_001"
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
            amended_state["user_prompt"] = f"{amended_state.get('user_prompt', '')}\n\nAmendment: {amendment_prompt}"
            interrupted_state = graph.invoke(amended_state, config=config)
            continue

        print("Unrecognized choice. Please enter y, n, or edit.")

    print("\nGraph execution completed successfully.")

