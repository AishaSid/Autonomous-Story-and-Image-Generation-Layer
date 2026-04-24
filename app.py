from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from PIL import Image, UnidentifiedImageError

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


def _is_valid_image(path_obj: Path) -> bool:
    try:
        with Image.open(path_obj) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def _parse_uploaded_script(uploaded_file) -> tuple[dict[str, Any], str]:
    """Parse uploaded script (JSON or TXT) and return (script_dict, format_str)."""
    try:
        file_content = uploaded_file.read().decode("utf-8")
        
        # Try JSON first
        if uploaded_file.name.endswith(".json"):
            script = json.loads(file_content)
            return script, "json"
        
        # Otherwise treat as screenplay text
        return {"raw_screenplay": file_content}, "screenplay"
    except json.JSONDecodeError:
        return {"raw_screenplay": file_content}, "screenplay"
    except Exception as e:
        return {}, f"error: {str(e)}"


def main() -> None:
    st.set_page_config(page_title="Character Reference Generator", layout="wide")
    st.title("Autonomous Character Reference Generator")
    st.markdown(
        """
        **Phase 1: Character Identity Store**
        
        Transform human intent into structured scripts, character identities, and reference portraits.
        """
    )

    # Tab selection: Prompt-based or Script Upload
    tab_prompt, tab_upload = st.tabs(["Generate from Prompt", "Upload Script"])

    with tab_prompt:
        st.subheader("Auto-Generate from Prompt")
        user_prompt = st.text_input(
            "Scene Prompt",
            value="a man giving interview to a female",
            help="Describe a scene to define character roles and interactions.",
            key="prompt_input"
        )
        run_prompt = st.button("Generate Character References", type="primary", key="btn_prompt")
        
        if run_prompt and user_prompt.strip():
            _run_pipeline(user_prompt=user_prompt, input_mode="auto")
        elif run_prompt:
            st.error("Please enter a scene prompt.")

    with tab_upload:
        st.subheader("Manual Script Upload")
        uploaded_file = st.file_uploader(
            "Upload screenplay (JSON or TXT)",
            type=["json", "txt"],
            help="Upload a screenplay in JSON format or screenplay text format (INT./EXT. headers)"
        )
        
        if uploaded_file:
            script_content, format_type = _parse_uploaded_script(uploaded_file)
            
            if "error" in format_type:
                st.error(f"Failed to parse file: {format_type}")
            else:
                st.info(f"✓ Loaded {format_type} format")
                
                # Show preview
                with st.expander("Preview Uploaded Script", expanded=False):
                    st.code(json.dumps(script_content, indent=2), language="json")
                
                run_upload = st.button("Process Uploaded Script", type="primary", key="btn_upload")
                
                if run_upload:
                    # Convert script content to prompt string for manual mode
                    script_prompt = json.dumps(script_content) if isinstance(script_content, dict) else str(script_content)
                    _run_pipeline(user_prompt=script_prompt, input_mode="manual")


def _run_pipeline(user_prompt: str, input_mode: str) -> None:
    """Execute the multi-agent pipeline."""
    with st.spinner("Running multi-agent pipeline..."):
        try:
            graph = build_graph(interrupt_before_character=False)
            state = initial_state(user_prompt=user_prompt, input_mode=input_mode)
            state["num_scenes"] = 2
            state["max_total_frames"] = 4
            state["reuse_character_memory"] = False
            state["llm_provider"] = "groq"
            state["llm_model"] = "llama-3.3-70b-versatile"
            state["llm_temperature"] = 0.3
            final_state = graph.invoke(state, config={"configurable": {"thread_id": "ui_run"}})
        except Exception as e:
            st.error(f"Pipeline execution failed: {str(e)}")
            return

    # Extract results
    script = final_state.get("script", {}) if isinstance(final_state, dict) else {}
    characters = final_state.get("characters", []) if isinstance(final_state, dict) else []
    images = final_state.get("images", []) if isinstance(final_state, dict) else []

    st.success("Pipeline completed.")

    # Display results
    col1, col2 = st.columns([1, 1])
    with col1:
        _render_json("Character Database (Metadata)", {"characters": characters})
    with col2:
        _render_json("Scene Context", script)

    st.divider()
    st.subheader("Character Reference Images")
    st.write("Identity-defining portraits for character consistency and reference.")
    
    image_paths = _collect_images(script, images)
    if not image_paths:
        st.warning("No character reference images were generated.")
    else:
        # Group images by character
        character_images = {}
        for char in characters:
            char_name = char.get("name", "Unknown")
            char_ref = char.get("reference_image_path", "")
            if char_ref:
                character_images[char_name] = char_ref
        
        if character_images:
            for char_name, image_path in character_images.items():
                path_obj = Path(image_path)
                if path_obj.exists():
                    if _is_valid_image(path_obj):
                        st.image(
                            str(path_obj), 
                            caption=f"{char_name} — Character Reference", 
                            width="stretch",
                            use_container_width=True
                        )
                    else:
                        st.warning(f"Invalid image: {path_obj.name}")
                else:
                    st.warning(f"Image not found: {path_obj.name}")
        else:
            st.info("No character reference images linked in metadata.")

    st.divider()
    st.subheader("Saved Outputs")
    st.write(f"- **Scene Manifest:** `{outputs_path('scene_manifest.json')}`")
    st.write(f"- **Character DB:** `{outputs_path('character_db.json')}`")


if __name__ == "__main__":
    main()
