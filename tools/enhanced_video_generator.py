"""
Enhanced Video Generation Module
Generates high-quality videos from character images with proper composition and transitions.
"""

from __future__ import annotations

import math
import os
import tempfile
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from moviepy.editor import (
        ImageClip, 
        CompositeVideoClip, 
        concatenate_videoclips,
        AudioFileClip,
        TextClip,
        CompositeAudioClip,
        ColorClip
    )
    from moviepy.audio.AudioClip import composite_audio_clips
except ImportError:
    # Fallback for if moviepy isn't available
    ImageClip = None
    CompositeVideoClip = None
    concatenate_videoclips = None
    AudioFileClip = None
    TextClip = None


def _duration_from_audio(audio_path: str, fallback_seconds: float = 8.0) -> float:
    """Extract duration from audio file in seconds."""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        return fallback_seconds

    try:
        with wave.open(str(audio_file), "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate() or 22050
            duration = max(frame_count / float(frame_rate), 1.0)
            # Clamp to 8-10 seconds
            return min(max(duration, 8.0), 10.0)
    except Exception:
        return fallback_seconds


def _validate_image(path: Path) -> bool:
    """Check if image file is valid."""
    if not path.exists():
        return False
    try:
        img = cv2.imread(str(path))
        return img is not None and img.shape[0] > 0 and img.shape[1] > 0
    except Exception:
        return False


def _get_image_dimensions(path: Path) -> tuple[int, int] | None:
    """Get image width and height."""
    try:
        img = cv2.imread(str(path))
        if img is not None:
            return (img.shape[1], img.shape[0])  # width, height
    except Exception:
        pass
    return None


def _create_transition_frame(
    from_img: np.ndarray, 
    to_img: np.ndarray, 
    alpha: float
) -> np.ndarray:
    """Create a transition frame between two images using cross-fade."""
    # Ensure same dimensions
    h, w = from_img.shape[:2]
    to_img_resized = cv2.resize(to_img, (w, h))
    
    # Cross-fade blend
    transition = cv2.addWeighted(from_img, 1.0 - alpha, to_img_resized, alpha, 0)
    return transition


def _generate_frame_sequence(
    image_paths: list[str],
    scene_id: str,
    dialogue_beats: list[str],
    total_duration: float,
    fps: int = 24,
) -> list[np.ndarray]:
    """
    Generate a sequence of video frames from image paths.
    Distributes images across the timeline with transitions.
    """
    frames = []
    
    # Validate and load images
    valid_images = []
    for img_path in image_paths:
        img_path_obj = Path(img_path)
        if _validate_image(img_path_obj):
            valid_images.append(cv2.imread(str(img_path_obj)))
    
    if not valid_images:
        # Generate placeholder frames if no valid images
        height, width = 360, 640
        valid_images = [np.zeros((height, width, 3), dtype=np.uint8)]
    
    # Get frame dimensions from first image
    height, width = valid_images[0].shape[:2]
    total_frames = int(total_duration * fps)
    
    if not valid_images:
        # Return blank frames as fallback
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        return [blank] * total_frames
    
    # Distribute images across timeline
    frames_per_image = total_frames / max(len(valid_images), 1)
    transition_frames = max(int(0.15 * frames_per_image), 3)  # 15% of frame time for transition
    
    current_frame = 0
    
    for img_idx, img in enumerate(valid_images):
        # Ensure image is correct dimensions
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        # Add dialogue text to current image
        display_img = img.copy()
        if img_idx < len(dialogue_beats):
            text = dialogue_beats[img_idx][:60]  # Limit text length
            cv2.putText(
                display_img,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        
        # Calculate frame range for this image
        start_frame = current_frame
        end_frame = int((img_idx + 1) * frames_per_image) if img_idx < len(valid_images) - 1 else total_frames
        
        # Add frames for this image
        image_frame_count = end_frame - start_frame
        
        if image_frame_count > transition_frames * 2:
            # Add hold frames + transition
            hold_frames = image_frame_count - transition_frames
            frames.extend([display_img] * hold_frames)
            
            # Add transition to next image if available
            if img_idx < len(valid_images) - 1:
                next_img = valid_images[img_idx + 1]
                if next_img.shape[:2] != (height, width):
                    next_img = cv2.resize(next_img, (width, height))
                
                for t in range(transition_frames):
                    alpha = t / transition_frames
                    trans_frame = _create_transition_frame(display_img, next_img, alpha)
                    frames.append(trans_frame)
        else:
            # Just hold the image
            frames.extend([display_img] * image_frame_count)
        
        current_frame = end_frame
    
    # Pad to exact frame count if needed
    if len(frames) < total_frames:
        last_frame = frames[-1] if frames else np.zeros((height, width, 3), dtype=np.uint8)
        frames.extend([last_frame] * (total_frames - len(frames)))
    elif len(frames) > total_frames:
        frames = frames[:total_frames]
    
    return frames


def _write_frames_to_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 24,
) -> bool:
    """Write frame list to MP4 video file."""
    if not frames:
        return False
    
    height, width = frames[0].shape[:2]
    
    try:
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height),
        )
        
        if not writer.isOpened():
            return False
        
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
        
        writer.release()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write video: {e}")
        return False


def generate_scene_video_from_images(
    scene_id: str,
    image_paths: list[str],
    dialogue_beats: list[str],
    visual_cues: list[str],
    audio_path: str,
    output_path: str,
    summary: str = "",
    fps: int = 24,
) -> tuple[str, bool]:
    """
    Generate a complete scene video from character images.
    
    Args:
        scene_id: Scene identifier
        image_paths: List of character image paths
        dialogue_beats: List of dialogue lines
        visual_cues: List of visual descriptions
        audio_path: Path to audio file (.wav)
        output_path: Where to save the output video
        summary: Scene summary
        fps: Frames per second for video
    
    Returns:
        Tuple of (output_video_path, success)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get duration from audio
    duration = _duration_from_audio(audio_path, fallback_seconds=8.0)
    print(f"[enhanced_video_gen] Generating {duration:.1f}s video for {scene_id}")
    
    # Generate frame sequence
    frames = _generate_frame_sequence(
        image_paths=image_paths,
        scene_id=scene_id,
        dialogue_beats=dialogue_beats,
        total_duration=duration,
        fps=fps,
    )
    
    print(f"[enhanced_video_gen] Created {len(frames)} frames for {scene_id}")
    
    # Write frames to video
    success = _write_frames_to_video(frames, str(output_file), fps=fps)
    
    if success:
        print(f"✓ Video generated: {output_file} ({len(frames)} frames, {duration:.1f}s)")
        return str(output_file), True
    else:
        print(f"✗ Failed to generate video for {scene_id}")
        return str(output_file), False


def _validate_mp4(path: Path, min_size_bytes: int = 50 * 1024) -> bool:
    """Validate that an MP4 file exists and is playable."""
    if not path.exists() or path.stat().st_size < min_size_bytes:
        return False
    if path.suffix.lower() != ".mp4":
        return False
    
    try:
        capture = cv2.VideoCapture(str(path))
        ok = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
        capture.release()
        return ok
    except Exception:
        return False


def mux_audio_to_video(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> bool:
    """
    Mux audio into video file using ffmpeg.
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file (.wav)
        output_path: Where to save the muxed output
    
    Returns:
        True if successful
    """
    import subprocess
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(output_file),
            ],
            check=True,
            capture_output=True,
        )
        print(f"✓ Audio muxed: {output_file}")
        return True
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Please install ffmpeg.")
        return False
    except Exception as e:
        print(f"[ERROR] ffmpeg failed: {e}")
        return False
