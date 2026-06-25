from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from services.video_export import find_ffmpeg_exe


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _normalise_orientation_degrees(value) -> int:
    try:
        degrees = int(round(float(value))) % 360
    except (TypeError, ValueError):
        return 0
    return degrees if degrees in {90, 180, 270} else 0


@lru_cache(maxsize=128)
def _opencv_orientation_meta(video_path: str) -> int:
    orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    if orientation_prop is None:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return 0
        return _normalise_orientation_degrees(cap.get(orientation_prop))
    finally:
        cap.release()


def _apply_orientation_meta(frame, orientation_degrees: int):
    orientation = _normalise_orientation_degrees(orientation_degrees)
    if orientation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if orientation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _read_with_pyav(video_path: str | os.PathLike, frame_idx: int, fps: float | None = None):
    if _truthy_env("MOUSESCOPE_DISABLE_PYAV_FRAME_READER"):
        return None
    try:
        import av
    except Exception:
        return None

    container = None
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        source_fps = float(fps or stream.average_rate or 30.0)
        if source_fps <= 0:
            source_fps = 30.0
        target_time = max(0, int(frame_idx)) / source_fps
        time_base = float(stream.time_base or 0.0)
        if time_base > 0:
            seek_offset = int(target_time / time_base)
            container.seek(seek_offset, stream=stream, any_frame=False, backward=True)

        tolerance = 0.5 / source_fps
        selected = None
        for frame in container.decode(stream):
            selected = frame
            if frame.pts is None or time_base <= 0:
                break
            frame_time = float(frame.pts) * time_base
            if frame_time >= target_time - tolerance:
                break
        if selected is None:
            return None
        frame = selected.to_ndarray(format="bgr24")
        return _apply_orientation_meta(frame, _opencv_orientation_meta(str(video_path)))
    except Exception:
        return None
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


def _read_with_ffmpeg(video_path: str | os.PathLike, frame_idx: int, fps: float | None = None):
    if _truthy_env("MOUSESCOPE_DISABLE_FFMPEG_FRAME_READER"):
        return None
    ffmpeg_exe = find_ffmpeg_exe()
    if not ffmpeg_exe:
        return None

    source_fps = float(fps or 30.0)
    if source_fps <= 0:
        source_fps = 30.0
    timestamp = max(0, int(frame_idx)) / source_fps
    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "3",
        "-",
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            timeout=15,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None

    data = np.frombuffer(proc.stdout, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _read_with_opencv(video_path: str | os.PathLike, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


def read_video_frame(video_path: str | os.PathLike, frame_idx: int, fps: float | None = None):
    path = Path(video_path)
    if not path.exists():
        return None

    frame = _read_with_pyav(path, frame_idx, fps=fps)
    if frame is not None:
        return frame

    frame = _read_with_ffmpeg(path, frame_idx, fps=fps)
    if frame is not None:
        return frame

    return _read_with_opencv(path, frame_idx)
