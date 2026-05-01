from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return None


def _ensure_wav(audio_path: Path) -> Path:
    if audio_path.suffix.lower() == ".wav" and audio_path.exists():
        return audio_path

    wav_path = audio_path.with_suffix(".wav")
    if wav_path.exists():
        return wav_path

    if audio_path.suffix.lower() in {".mp3", ".m4a", ".aac"} and audio_path.exists():
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


def _audio_energy_envelope(audio_path: Path, frame_total: int) -> np.ndarray:
    if not audio_path.exists() or frame_total <= 0:
        return np.zeros(max(frame_total, 1), dtype=np.float32)

    try:
        if librosa is not None:
            samples, _sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        else:
            with wave.open(str(audio_path), "rb") as wav_file:
                frame_count = wav_file.getnframes()
                sample_width = wav_file.getsampwidth()
                raw = wav_file.readframes(frame_count)
            if sample_width == 2:
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 255.0
    except Exception:
        return np.zeros(max(frame_total, 1), dtype=np.float32)

    if samples.size == 0:
        return np.zeros(max(frame_total, 1), dtype=np.float32)

    chunks = np.array_split(samples, frame_total)
    rms_values = np.array([float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)
    max_value = float(rms_values.max()) if rms_values.size else 0.0
    if max_value <= 0:
        return np.zeros_like(rms_values)
    return np.clip(rms_values / max_value, 0.0, 1.0)


def _validate_mp4(path: Path, min_size_bytes: int = 50 * 1024) -> bool:
    if not path.exists() or path.stat().st_size < min_size_bytes:
        return False
    if path.suffix.lower() != ".mp4":
        return False

    capture = cv2.VideoCapture(str(path))
    try:
        return capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
    finally:
        capture.release()


def _mux_audio_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as ex:
        raise RuntimeError("ffmpeg is required for audio/video muxing.") from ex


def _apply_mouth_motion(frame: np.ndarray, amplitude: float, face_box: tuple[int, int, int, int] | None) -> np.ndarray:
    if face_box is None:
        return frame

    x, y, w, h = face_box
    if w <= 0 or h <= 0:
        return frame

    overlay = frame.copy()
    mouth_center = (int(x + (w * 0.5)), int(y + (h * 0.72)))
    mouth_width = max(6, int(w * 0.18))
    mouth_height = max(4, int(h * (0.02 + 0.12 * amplitude)))

    cv2.ellipse(overlay, mouth_center, (mouth_width, mouth_height), 0, 0, 360, (20, 20, 20), -1)
    cv2.ellipse(overlay, mouth_center, (max(2, mouth_width // 2), max(2, mouth_height // 2)), 0, 0, 360, (50, 10, 10), -1)
    return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)


def _amplitude_sync_video(scene_id: str, audio_path: Path, video_path: Path, output_path: Path) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_total <= 0:
        frame_total = max(int(math.ceil(5 * fps)), 1)

    energy = _audio_energy_envelope(audio_path, frame_total)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    temp_video = output_path.with_suffix(".sync.mp4")
    writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open output writer: {temp_video}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            face_box = None
            if len(faces) > 0:
                face_box = max(faces, key=lambda box: box[2] * box[3])
                amplitude = float(energy[min(frame_index, len(energy) - 1)]) if energy.size else 0.0
                frame = _apply_mouth_motion(frame, amplitude, tuple(int(v) for v in face_box))
            else:
                amplitude = float(energy[min(frame_index, len(energy) - 1)]) if energy.size else 0.0
                cv2.putText(
                    frame,
                    f"Lip-sync amplitude: {amplitude:.2f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    if not _validate_mp4(temp_video):
        raise RuntimeError(f"Amplitude lip-sync produced invalid video: {temp_video}")

    print(f"[OK] amplitude lip-sync completed for {scene_id}; frames={frame_index}")
    return temp_video


def _run_wav2lip_if_available(scene_id: str, audio_path: Path, video_path: Path, output_path: Path) -> bool:
    wav2lip_repo = Path(_get_env_value("WAV2LIP_REPO") or (PROJECT_ROOT / "Wav2Lip"))
    checkpoint_path = Path(_get_env_value("WAV2LIP_CHECKPOINT") or (wav2lip_repo / "checkpoints" / "wav2lip_gan.pth"))
    inference_py = wav2lip_repo / "inference.py"

    if not inference_py.exists() or not checkpoint_path.exists():
        return False

    command = [
        sys.executable,
        str(inference_py),
        "--checkpoint_path",
        str(checkpoint_path),
        "--face",
        str(video_path),
        "--audio",
        str(audio_path),
        "--outfile",
        str(output_path),
    ]

    print(f"[lip_sync] running Wav2Lip CPU path for {scene_id}")
    try:
        completed = subprocess.run(
            command,
            cwd=str(wav2lip_repo),
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if completed.stdout:
            print(completed.stdout.strip())
        if completed.stderr:
            print(completed.stderr.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as ex:
        print(f"[WARN] Wav2Lip failed for {scene_id}: {ex}")
        return False

    return _validate_mp4(output_path)


def lip_sync_aligner(
    scene_id: str,
    audio_path: str,
    video_path: str,
    output_path: str,
    use_wav2lip: bool = True,
) -> str:
    """
    CPU-only lip sync pipeline.
    Tries Wav2Lip if a local repo/checkpoint is available; otherwise falls back to
    amplitude-driven mouth motion and finally muxes the original audio with ffmpeg.
    """
    audio_file = Path(audio_path)
    video_file = Path(video_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")

    wav_audio = _ensure_wav(audio_file)
    synced_video = destination.with_suffix(".sync.mp4")

    used_wav2lip = False
    if use_wav2lip:
        try:
            used_wav2lip = _run_wav2lip_if_available(scene_id, wav_audio, video_file, synced_video)
        except Exception as e:
            print(f"[WARN] Wav2Lip unavailable for {scene_id}: {e}")
            used_wav2lip = False

    if not used_wav2lip:
        synced_video = _amplitude_sync_video(scene_id, wav_audio, video_file, destination)

    _mux_audio_video(synced_video, wav_audio, destination)

    if not _validate_mp4(destination):
        raise RuntimeError(f"Final lip-synced mp4 is invalid: {destination}")

    if synced_video.exists() and synced_video != destination:
        try:
            synced_video.unlink()
        except OSError:
            pass

    print(f"[OK] lip-sync merge complete for {scene_id}; output={destination}")
    return str(destination)
