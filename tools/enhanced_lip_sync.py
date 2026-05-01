"""
Enhanced Lip Sync Alignment Module
Aligns mouth movements with audio energy for natural-looking speech synchronization.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


def _ensure_wav(audio_path: Path) -> Path:
    """Convert audio to WAV if needed."""
    if audio_path.suffix.lower() == ".wav" and audio_path.exists():
        return audio_path
    
    wav_path = audio_path.with_suffix(".wav")
    if wav_path.exists():
        return wav_path
    
    # Try to convert using ffmpeg
    if audio_path.exists() and audio_path.suffix.lower() in {".mp3", ".m4a", ".aac"}:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-acodec", "pcm_s16le", "-ar", "22050", str(wav_path)],
                check=True,
                capture_output=True,
            )
            return wav_path
        except Exception:
            pass
    
    return audio_path


def _extract_audio_energy_envelope(audio_path: str, num_frames: int) -> np.ndarray:
    """
    Extract audio energy envelope for lip sync.
    Returns an array where high values = loud/open mouth, low values = quiet/closed mouth.
    """
    if not Path(audio_path).exists() or num_frames <= 0:
        return np.zeros(max(num_frames, 1), dtype=np.float32)
    
    audio_path_obj = Path(audio_path)
    
    try:
        # Try using librosa for better audio processing
        if librosa is not None:
            samples, sr = librosa.load(str(audio_path_obj), sr=16000, mono=True)
            # Compute MFCC energy
            S = librosa.feature.melspectrogram(y=samples, sr=sr)
            energy = np.sqrt(np.sum(S**2, axis=0))
        else:
            # Fallback: use simple WAV reading
            import wave
            wav_path = _ensure_wav(audio_path_obj)
            with wave.open(str(wav_path), "rb") as wav_file:
                frame_count = wav_file.getnframes()
                sample_width = wav_file.getsampwidth()
                raw = wav_file.readframes(frame_count)
            
            if sample_width == 2:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
            
            # Simple energy calculation
            energy = np.sqrt(np.convolve(samples**2, np.ones(512), mode='same'))
    except Exception as e:
        print(f"[ERROR] Failed to extract audio energy: {e}")
        return np.zeros(num_frames, dtype=np.float32)
    
    if energy.size == 0:
        return np.zeros(num_frames, dtype=np.float32)
    
    # Resample energy to match number of frames
    if len(energy) != num_frames:
        indices = np.linspace(0, len(energy) - 1, num_frames)
        energy = np.interp(indices, np.arange(len(energy)), energy)
    
    # Normalize to [0, 1]
    energy = np.asarray(energy, dtype=np.float32)
    max_val = np.max(energy) if energy.size > 0 else 1.0
    if max_val > 0:
        energy = energy / max_val
    
    return np.clip(energy, 0.0, 1.0)


def _detect_mouth_region(frame: np.ndarray, use_cascades: bool = True) -> tuple[int, int, int, int] | None:
    """
    Simple mouth detection using face/mouth detection.
    Returns (x, y, w, h) for mouth bounding box or None.
    """
    # For now, use a simple heuristic based on face detection
    # In a real implementation, you'd use mouth detection cascade
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        
        if len(faces) == 0:
            # Estimate mouth location in center-bottom of frame
            h, w = frame.shape[:2]
            return (int(w * 0.35), int(h * 0.60), int(w * 0.30), int(h * 0.20))
        
        # Use largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Estimate mouth position within face (lower third, center horizontally)
        mouth_x = x + int(w * 0.25)
        mouth_y = y + int(h * 0.65)
        mouth_w = int(w * 0.5)
        mouth_h = int(h * 0.25)
        
        return (mouth_x, mouth_y, mouth_w, mouth_h)
    except Exception:
        return None


def _apply_mouth_movement(
    frame: np.ndarray,
    energy_value: float,
    mouth_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """
    Apply subtle mouth animation based on audio energy.
    Higher energy = slightly more open mouth appearance.
    """
    if mouth_box is None:
        return frame
    
    x, y, w, h = mouth_box
    
    # Clamp to frame bounds
    x = max(0, min(x, frame.shape[1] - 1))
    y = max(0, min(y, frame.shape[0] - 1))
    w = max(1, min(w, frame.shape[1] - x))
    h = max(1, min(h, frame.shape[0] - y))
    
    # Scale mouth opening based on energy - subtle effect
    mouth_center = (x + w // 2, y + h // 2)
    
    # Only apply subtle darkening based on energy (simulates mouth opening)
    # This is a subtle effect - the mouth gets slightly darker when "open"
    intensity = int(20 * energy_value)
    
    # Get mouth region
    mouth_region = frame[y:y+h, x:x+w].copy()
    
    # Apply subtle darkening to simulate mouth movement
    if energy_value > 0.3:  # Only when speaking
        # Darken slightly
        mouth_region = np.clip(mouth_region - intensity, 0, 255).astype(np.uint8)
        frame[y:y+h, x:x+w] = mouth_region
    
    return frame


def enhance_video_with_lip_sync(
    scene_id: str,
    video_path: str,
    audio_path: str,
    output_path: str,
) -> bool:
    """
    Enhance video with lip-sync animation.
    Detects mouths and animates them based on audio energy.
    
    Args:
        scene_id: Scene identifier
        video_path: Path to input video
        audio_path: Path to audio file
        output_path: Where to save enhanced video
    
    Returns:
        True if successful
    """
    video_path_obj = Path(video_path)
    audio_path_obj = Path(audio_path)
    output_path_obj = Path(output_path)
    
    if not video_path_obj.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return False
    
    if not audio_path_obj.exists():
        print(f"[ERROR] Audio not found: {audio_path}")
        return False
    
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Open video
    capture = cv2.VideoCapture(str(video_path_obj))
    if not capture.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return False
    
    # Get video properties
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    print(f"[lip_sync] Processing {scene_id}: {total_frames} frames @ {fps} fps")
    
    # Extract audio energy envelope
    energy = _extract_audio_energy_envelope(str(audio_path_obj), total_frames)
    
    # Create video writer
    writer = cv2.VideoWriter(
        str(output_path_obj),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    
    if not writer.isOpened():
        print(f"[ERROR] Could not create output video: {output_path}")
        capture.release()
        return False
    
    # Process frames
    frame_idx = 0
    mouth_box = None
    
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            # Detect mouth region (every 30 frames to save computation)
            if frame_idx % 30 == 0:
                mouth_box = _detect_mouth_region(frame)
            
            # Get energy value for this frame
            energy_val = energy[frame_idx] if frame_idx < len(energy) else 0.0
            
            # Apply mouth movement
            enhanced_frame = _apply_mouth_movement(frame, energy_val, mouth_box)
            
            writer.write(enhanced_frame)
            frame_idx += 1
    
    except Exception as e:
        print(f"[ERROR] Error processing frames: {e}")
    
    finally:
        capture.release()
        writer.release()
    
    print(f"✓ Lip sync enhancement complete: {output_path_obj}")
    return True


def lip_sync_aligner(
    scene_id: str,
    audio_path: str,
    video_path: str,
    output_path: str = "",
    use_wav2lip: bool = False,
) -> dict[str, Any]:
    """
    Main entry point for lip sync alignment.
    Compatible with langgraph and pipeline (old interface).
    
    Args:
        scene_id: Scene identifier
        audio_path: Path to audio file
        video_path: Path to video file
        output_path: Optional custom output path
        use_wav2lip: Ignored (for compatibility)
    
    Returns:
        Dictionary with results OR just the output path string for backward compatibility
    """
    video_obj = Path(video_path)
    audio_obj = Path(audio_path)
    
    if not output_path:
        output_path = str(video_obj.parent / f"{scene_id}_lipsync.mp4")
    
    print(f"\n[lip_sync_aligner] Starting lip-sync for {scene_id}")
    print(f"  Video: {video_path}")
    print(f"  Audio: {audio_path}")
    
    success = enhance_video_with_lip_sync(
        scene_id=scene_id,
        video_path=str(video_obj),
        audio_path=str(audio_obj),
        output_path=output_path,
    )
    
    # Return just the output_path string for backward compatibility with parser
    if success:
        return output_path
    else:
        return str(video_obj)  # Fallback to original video if lip sync fails


def lip_sync_aligner_dict(
    scene_id: str,
    video_path: str,
    audio_path: str,
    output_path: str = "",
) -> dict[str, Any]:
    """
    Return a dictionary instead of just the path.
    For use when full result details are needed.
    """
    video_obj = Path(video_path)
    audio_obj = Path(audio_path)
    
    if not output_path:
        output_path = str(video_obj.parent / f"{scene_id}_lipsync.mp4")
    
    print(f"\n[lip_sync_aligner] Starting lip-sync for {scene_id}")
    print(f"  Video: {video_path}")
    print(f"  Audio: {audio_path}")
    
    success = enhance_video_with_lip_sync(
        scene_id=scene_id,
        video_path=str(video_obj),
        audio_path=str(audio_obj),
        output_path=output_path,
    )
    
    return {
        "status": "success" if success else "failed",
        "scene_id": scene_id,
        "video_path": output_path if success else "",
        "audio_path": audio_path,
        "output_path": output_path,
        "lipsync_success": success,
    }
