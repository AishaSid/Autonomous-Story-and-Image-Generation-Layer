"""
Unified Pipeline: Phase-1 (Image/Scene Generation) → Phase-2 (Video Generation)

This orchestrates the complete pipeline:
1. Phase-1: Accepts user prompt → generates script → creates character images → outputs scene manifest JSON
2. Phase-2: Consumes scene manifest JSON → generates voices → creates videos → outputs MP4s
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Literal
from dotenv import load_dotenv

load_dotenv()

# Phase-1 imports
from main import build_graph as build_phase1_graph
from state import State, initial_state

# Phase-2 imports  
from src.agents.parser import run_scene_parser, resume_scene_parser, validate_manifest_schema


def run_full_pipeline(
    user_prompt: str,
    input_mode: Literal["manual", "auto"] = "auto",
    skip_phase1: bool = False,
    resume_phase2: bool = False,
    checkpoint_dir: str = "state/checkpoints",
) -> Dict[str, Any]:
    """
    Execute the complete pipeline from user prompt to final videos.
    
    Args:
        user_prompt: Initial user input describing the story/scenario
        input_mode: "manual" for human-in-the-loop, "auto" for autonomous
        skip_phase1: If True, skip phase-1 and use existing manifest in phase1_inputs/
        resume_phase2: If True, resume from last checkpoint in phase-2
        checkpoint_dir: Directory for phase-2 checkpoints
    
    Returns:
        Dictionary with both phase-1 and phase-2 outputs
    """
    
    phase1_manifest_path = "phase1_inputs/scene_manifest.json"
    results = {
        "phase1": None,
        "phase2": None,
        "success": False,
        "errors": [],
    }
    
    # ===== PHASE-1: Scene Generation =====
    if not skip_phase1:
        print("\n" + "="*60)
        print("PHASE 1: Scene and Character Generation")
        print("="*60)
        
        try:
            # Build and run phase-1 pipeline
            phase1_graph = build_phase1_graph(interrupt_before_character=False)
            
            # Prepare initial state
            phase1_state: State = {
                **initial_state,
                "user_input": user_prompt,
                "input_mode": input_mode,
            }
            
            # Execute phase-1
            print(f"\nRunning Phase-1 with mode: {input_mode}")
            print(f"User prompt: {user_prompt}\n")
            
            final_state = phase1_graph.invoke(
                phase1_state,
                config={
                    "recursion_limit": 100,
                }
            )
            
            results["phase1"] = {
                "status": "completed",
                "script": final_state.get("script"),
                "character_db": final_state.get("character_db"),
                "images": final_state.get("image_paths", []),
                "manifest_path": phase1_manifest_path,
            }
            
            print(f"✓ Phase-1 Complete!")
            print(f"  - Characters defined: {len(final_state.get('character_db', {}))}")
            print(f"  - Scenes generated: {len(final_state.get('script', {}).get('scenes', []))}")
            print(f"  - Images created: {len(final_state.get('image_paths', []))}")
            
        except Exception as e:
            error_msg = f"Phase-1 failed: {str(e)}"
            print(f"✗ {error_msg}")
            results["errors"].append(error_msg)
            return results
    else:
        print("\n" + "="*60)
        print("PHASE 1: SKIPPED (using existing manifest)")
        print("="*60)
    
    # ===== PHASE-2: Video Generation =====
    print("\n" + "="*60)
    print("PHASE 2: Video Generation")
    print("="*60)
    
    # Validate manifest exists
    manifest_path = Path(phase1_manifest_path)
    if not manifest_path.exists():
        error_msg = f"Phase-1 manifest not found at {phase1_manifest_path}"
        print(f"✗ {error_msg}")
        results["errors"].append(error_msg)
        return results
    
    try:
        # Validate schema
        is_valid, message = validate_manifest_schema(str(manifest_path))
        print(f"Manifest validation: {message}")
        
        if not is_valid:
            raise ValueError(f"Invalid manifest: {message}")
        
        # Run phase-2
        print(f"\nRunning Phase-2...")
        
        if resume_phase2:
            print("Attempting to resume from checkpoint...")
            final_state = resume_scene_parser(
                manifest_path=str(manifest_path),
                checkpoint_dir=checkpoint_dir,
            )
        else:
            final_state = run_scene_parser(
                manifest_path=str(manifest_path),
                checkpoint_dir=checkpoint_dir,
            )
        
        results["phase2"] = {
            "status": "completed",
            "voice_outputs": final_state.get("voice_outputs", []),
            "video_outputs": final_state.get("video_outputs", []),
            "face_swap_outputs": final_state.get("face_swap_outputs", []),
            "fused_outputs": final_state.get("fused_outputs", []),
            "errors": final_state.get("errors", []),
        }
        
        print(f"✓ Phase-2 Complete!")
        print(f"  - Voice files: {len(final_state.get('voice_outputs', []))}")
        print(f"  - Videos created: {len(final_state.get('video_outputs', []))}")
        print(f"  - Face swaps: {len(final_state.get('face_swap_outputs', []))}")
        print(f"  - Fused outputs: {len(final_state.get('fused_outputs', []))}")
        
        if final_state.get("errors"):
            print(f"  - Errors: {len(final_state.get('errors', []))}")
            for error in final_state.get("errors", [])[:3]:  # Show first 3 errors
                print(f"    • {error}")
        
        results["success"] = True
        
    except Exception as e:
        error_msg = f"Phase-2 failed: {str(e)}"
        print(f"✗ {error_msg}")
        results["errors"].append(error_msg)
    
    # ===== Summary =====
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Overall Status: {'✓ SUCCESS' if results['success'] else '✗ FAILED'}")
    
    if results["phase1"]:
        print(f"\nPhase-1 Output:")
        print(f"  Manifest: {results['phase1'].get('manifest_path')}")
    
    if results["phase2"]:
        print(f"\nPhase-2 Output:")
        print(f"  Final videos: outputs/")
    
    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"  • {error}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the unified image→video pipeline")
    parser.add_argument("prompt", nargs="?", default="Generate a mysterious story about a hidden temple", 
                       help="User prompt for story generation")
    parser.add_argument("--mode", choices=["manual", "auto"], default="auto",
                       help="Pipeline mode: manual (interactive) or auto (autonomous)")
    parser.add_argument("--skip-phase1", action="store_true",
                       help="Skip phase-1, use existing manifest")
    parser.add_argument("--resume-phase2", action="store_true",
                       help="Resume phase-2 from last checkpoint")
    parser.add_argument("--checkpoint-dir", default="state/checkpoints",
                       help="Directory for checkpoints")
    
    args = parser.parse_args()
    
    results = run_full_pipeline(
        user_prompt=args.prompt,
        input_mode=args.mode,
        skip_phase1=args.skip_phase1,
        resume_phase2=args.resume_phase2,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)
