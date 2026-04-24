from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from main import build_graph
from src.agents.common import outputs_path
from state import initial_state


def _collect_images(script: Dict[str, Any], images: List[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for image in images:
        image_path = str(image.get("path", "")).strip()
        if image_path:
            paths.append(image_path)

    if paths:
        return list(dict.fromkeys(paths))

    scenes = script.get("scenes", []) if isinstance(script, dict) else []
    for scene in scenes:
        for frame_path in scene.get("frame_image_paths", []) or []:
            frame_path_str = str(frame_path).strip()
            if frame_path_str:
                paths.append(frame_path_str)
    return list(dict.fromkeys(paths))


def _render_json(title: str, payload: Dict[str, Any]) -> None:
    st.subheader(title)
    st.code(json.dumps(payload, indent=2), language="json")


def main() -> None:
    st.set_page_config(page_title="Autonomous Story & Image Generator", layout="wide")
    st.title("Autonomous Story and Image Generation UI")
    st.write(
        "Enter a scene prompt (example: `a man giving interview to a female`) to generate script, "
        "characters, and frame-wise images."
    )

    user_prompt = st.text_input(
        "Scene Prompt",
        value="a man giving interview to a female",
        help="Describe a short 5-6 second scene.",
    )
    num_scenes = st.slider("Number of scenes", min_value=1, max_value=3, value=1)
    run_clicked = st.button("Generate Scene Package", type="primary")

    if not run_clicked:
        return

    with st.spinner("Running multi-agent pipeline..."):
        graph = build_graph(interrupt_before_character=False)
        state = initial_state(user_prompt=user_prompt, input_mode="auto")
        state["num_scenes"] = int(num_scenes)
        state["llm_provider"] = "groq"
        state["llm_model"] = "llama-3.3-70b-versatile"
        state["llm_temperature"] = 0.3
        final_state = graph.invoke(state, config={"configurable": {"thread_id": "ui_run"}})

    script = final_state.get("script", {}) if isinstance(final_state, dict) else {}
    characters = final_state.get("characters", []) if isinstance(final_state, dict) else []
    images = final_state.get("images", []) if isinstance(final_state, dict) else []

    st.success("Pipeline completed.")

    col1, col2 = st.columns(2)
    with col1:
        _render_json("Generated Script", script)
    with col2:
        _render_json("Character Database", {"characters": characters})

    st.subheader("Frame-wise Images")
    image_paths = _collect_images(script, images)
    if not image_paths:
        st.warning("No images were generated.")
    else:
        for image_path in image_paths:
            path_obj = Path(image_path)
            if path_obj.exists():
                st.image(str(path_obj), caption=path_obj.name, use_container_width=True)
            else:
                st.caption(f"Missing file: {image_path}")

    st.subheader("Saved Outputs")
    st.write(f"- Scene Manifest: `{outputs_path('scene_manifest.json')}`")
    st.write(f"- Character DB: `{outputs_path('character_db.json')}`")


if __name__ == "__main__":
    main()
