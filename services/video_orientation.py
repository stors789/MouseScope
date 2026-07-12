from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2

from services.video_export import find_ffmpeg_exe


RIGHT_ANGLE_ROTATIONS = {0, 90, 180, 270}


def normalise_orientation_degrees(value: Any, default: int = 0) -> int:
    """Return a supported clockwise display rotation, or ``default``.

    Video containers express rotation inconsistently: stream ``rotate`` tags
    are clockwise, while ffprobe display-matrix side data is counterclockwise.
    The rest of the application uses one convention: clockwise degrees applied
    to the coded frame.
    """
    try:
        degrees = int(round(float(value))) % 360
    except (TypeError, ValueError):
        return default
    return degrees if degrees in RIGHT_ANGLE_ROTATIONS else default


def orientation_override_degrees(value: Any) -> int | None:
    """Parse an optional user override. ``auto``/blank means no override."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "auto", "metadata", "detected"}:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if raw != raw or int(round(raw)) % 360 not in RIGHT_ANGLE_ROTATIONS:
        return None
    return normalise_orientation_degrees(raw)


def apply_orientation(frame, orientation_degrees: int):
    orientation = normalise_orientation_degrees(orientation_degrees)
    if orientation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if orientation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def oriented_dimensions(width: int | float, height: int | float, orientation_degrees: int) -> tuple[int, int]:
    try:
        source_width = max(0, int(round(float(width))))
        source_height = max(0, int(round(float(height))))
    except (TypeError, ValueError):
        return 0, 0
    if normalise_orientation_degrees(orientation_degrees) in {90, 270}:
        return source_height, source_width
    return source_width, source_height


def _subprocess_creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _ffprobe_executable() -> str | None:
    """Use the ffmpeg-adjacent ffprobe when it exists, then PATH."""
    ffmpeg_exe = find_ffmpeg_exe()
    candidates: list[Path] = []
    if ffmpeg_exe:
        ffmpeg_path = Path(ffmpeg_exe)
        candidates.extend((
            ffmpeg_path.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe"),
            ffmpeg_path.with_name("ffprobe"),
            ffmpeg_path.with_name("ffprobe.exe"),
        ))
    env_path = os.environ.get("MOUSESCOPE_FFPROBE_EXE")
    if env_path:
        candidates.insert(0, Path(env_path))
    which_path = shutil.which("ffprobe")
    if which_path:
        candidates.append(Path(which_path))
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
        candidates.extend((root / "ffprobe.exe", root / "ffprobe", root / "bin" / "ffprobe.exe"))
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        root = Path(meipass)
        candidates.extend((root / "ffprobe.exe", root / "ffprobe", root / "bin" / "ffprobe.exe"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _orientation_from_ffprobe_stream(stream: dict[str, Any]) -> int | None:
    side_data = stream.get("side_data_list") or []
    for item in side_data:
        if not isinstance(item, dict) or item.get("rotation") is None:
            continue
        try:
            # ffprobe display matrix values are counterclockwise.
            return normalise_orientation_degrees(-float(item["rotation"]))
        except (TypeError, ValueError):
            continue

    tags = stream.get("tags") or {}
    if isinstance(tags, dict) and tags.get("rotate") is not None:
        try:
            # The legacy rotate tag is clockwise.
            return normalise_orientation_degrees(tags["rotate"])
        except (TypeError, ValueError):
            pass
    return None


@lru_cache(maxsize=256)
def probe_video_orientation(video_path: str | os.PathLike) -> tuple[int, str]:
    """Return clockwise rotation and its source without decoding a frame."""
    path = Path(video_path)
    if not path.is_file():
        return 0, "none"

    ffprobe = _ffprobe_executable()
    if ffprobe:
        command = [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate:stream_side_data=rotation",
            "-of", "json",
            str(path),
        ]
        try:
            proc = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_subprocess_creationflags(),
                timeout=10,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout:
                data = json.loads(proc.stdout)
                streams = data.get("streams") or []
                if streams:
                    rotation = _orientation_from_ffprobe_stream(streams[0] or {})
                    if rotation is not None:
                        return rotation, "ffprobe"
        except Exception:
            pass

    orientation_prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    if orientation_prop is not None:
        cap = cv2.VideoCapture(str(path))
        try:
            if cap.isOpened():
                return normalise_orientation_degrees(cap.get(orientation_prop)), "opencv"
        finally:
            cap.release()
    return 0, "none"


def resolve_video_orientation(video_path: str | os.PathLike, orientation_override: Any = None) -> tuple[int, str]:
    override = orientation_override_degrees(orientation_override)
    if override is not None:
        return override, "manual"
    return probe_video_orientation(video_path)


class OrientedVideoCapture:
    """A VideoCapture facade that always returns one explicit orientation.

    OpenCV's automatic orientation setting changes across versions/backends.
    Disabling it and rotating decoded frames ourselves keeps calibration,
    analysis, background extraction, and annotated output in one coordinate
    system.
    """

    orientation_applied = True

    def __init__(self, video_path: str | os.PathLike, orientation_override: Any = None) -> None:
        self.video_path = str(video_path)
        self._cap = cv2.VideoCapture(self.video_path)
        orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
        if orientation_auto is not None and self._cap.isOpened():
            try:
                self._cap.set(orientation_auto, 0)
            except Exception:
                pass
        self.orientation_degrees, self.orientation_source = resolve_video_orientation(
            self.video_path,
            orientation_override=orientation_override,
        )

    def isOpened(self) -> bool:
        return bool(self._cap.isOpened())

    def read(self):
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return ok, frame
        return ok, apply_orientation(frame, self.orientation_degrees)

    def get(self, prop_id: int) -> float:
        if prop_id in {cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT}:
            raw_width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            raw_height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            width, height = oriented_dimensions(raw_width, raw_height, self.orientation_degrees)
            return float(width if prop_id == cv2.CAP_PROP_FRAME_WIDTH else height)
        return self._cap.get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        return bool(self._cap.set(prop_id, value))

    def release(self) -> None:
        self._cap.release()

    def __getattr__(self, name: str):
        return getattr(self._cap, name)


def open_video_capture(video_path: str | os.PathLike, orientation_override: Any = None) -> OrientedVideoCapture:
    return OrientedVideoCapture(video_path, orientation_override=orientation_override)
