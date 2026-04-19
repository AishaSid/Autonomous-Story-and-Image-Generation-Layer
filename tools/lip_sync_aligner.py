from __future__ import annotations

import math
import shutil
import wave
from pathlib import Path

import cv2
import numpy as np


def lip_sync_aligner(scene_id: str, audio_path: str, video_path: str, output_path: str) -> str:
    """Create a valid silent MP4 and keep the WAV beside it for the fusion layer."""
    audio_file = Path(audio_path)
    video_file = Path(video_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    sibling_wav = destination.with_suffix(".wav")
    if audio_file.exists():
        shutil.copyfile(audio_file, sibling_wav)

    duration_seconds = 1.0
    if audio_file.exists():
        try:
            with wave.open(str(audio_file), "rb") as wav_file:
                frame_count = wav_file.getnframes()
                frame_rate = wav_file.getframerate() or 22050
                duration_seconds = max(frame_count / float(frame_rate), 1.0 / 5.0)
        except Exception:
            duration_seconds = 1.0

    fps = 24
    frame_total = max(int(math.ceil(duration_seconds * fps)), 1)
    width, height = 640, 360
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {destination}")

    base_frame = np.zeros((height, width, 3), dtype=np.uint8)
    base_frame[:] = (24, 24, 24)

    cv2.putText(
        base_frame,
        f"{scene_id} fusion layer",
        (24, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        base_frame,
        "temporal alignment placeholder",
        (24, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        base_frame,
        f"audio: {audio_file.name}",
        (24, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        base_frame,
        f"source video: {video_file.name}",
        (24, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Keep the output a valid video container even when the source branch is placeholder data.
    for frame_index in range(frame_total):
        frame = base_frame.copy()
        progress = frame_index / max(frame_total - 1, 1)
        bar_width = int((width - 48) * progress)
        cv2.rectangle(frame, (24, 250), (24 + bar_width, 270), (110, 200, 120), -1)
        cv2.rectangle(frame, (24, 250), (width - 24, 270), (60, 60, 60), 2)
        cv2.putText(
            frame,
            f"frame {frame_index + 1}/{frame_total}",
            (24, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()

    return str(destination)
