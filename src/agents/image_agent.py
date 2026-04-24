from __future__ import annotations

import hashlib
from typing import Any, Dict
from state import State

from .common import invoke_mcp_tool_via_protocol

def _seed_from_name(name: str) -> int:
    """Generates a consistent random seed integer based on a character's name."""
    digest = hashlib.sha256(name.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _seed_for_frame(name: str, frame_id: str, visual_cue: str) -> int:
    """Generates deterministic but unique seeds per frame to avoid duplicate outputs."""
    token = f"{name.strip().lower()}|{frame_id.strip().lower()}|{visual_cue.strip().lower()}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

def image_node(state: State) -> Dict[str, Any]:
    print("--- GENERATING IMAGES ---")
    script = state.get("script", {})
    scenes = script.get("scenes", [])
    
    images = []
    
    for scene in scenes:
        scene_id = scene.get("scene_id", "scene_000")
        frame_prompts = scene.get("frame_prompts", [])
        scene_image_paths = []
        
        for frame in frame_prompts:
            frame_id = frame.get("frame_id")
            primary_character = frame.get("primary_character", "Character")
            refined_prompt = frame.get("refined_prompt", "")
            visual_cue = frame.get("visual_cue", "")
            
            seed = _seed_for_frame(primary_character, str(frame_id or ""), str(visual_cue or ""))
            filename = f"{frame_id}.png"
            
            print(f"Generating image for {frame_id} ({primary_character})...")
            
            try:
                # Calls your MCP Tool registered in mcp_server.py
                image_result = invoke_mcp_tool_via_protocol(
                    "generate_character_image",
                    {
                        "refined_prompt": refined_prompt,
                        "character_name": primary_character,
                        "filename": filename,
                        "seed": seed,
                    },
                )
                image_path = image_result.get("image_path", "")
            except Exception as e:
                print(f"Error generating image: {e}")
                image_path = ""
            
            if image_path:
                scene_image_paths.append(image_path)
                images.append({
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "primary_character": primary_character,
                    "character": primary_character,
                    "character_seed": seed,
                    "visual_cue": visual_cue,
                    "refined_prompt": refined_prompt,
                    "path": image_path,
                })
                
        # Link generated images back to the scene metadata
        scene["frame_image_paths"] = scene_image_paths

    return {"images": images, "script": script, "status": "images_generated"}