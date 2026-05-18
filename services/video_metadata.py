from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2

from services.video_export import find_ffmpeg_exe


logger = logging.getLogger(__name__)
DEFAULT_VIDEO_FPS = 30.0
MIN_REASONABLE_FPS = 0.001
MAX_REASONABLE_FPS = 1000.0


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _is_reasonable_fps(value: float | None) -> bool:
    return value is not None and MIN_REASONABLE_FPS <= float(value) <= MAX_REASONABLE_FPS


def _parse_fraction(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A" or text == "0/0":
        return None
    if "/" not in text:
        parsed = _safe_float(text)
        return parsed if _is_reasonable_fps(parsed) else None
    numerator_text, denominator_text = text.split("/", 1)
    numerator = _safe_float(numerator_text)
    denominator = _safe_float(denominator_text)
    if numerator is None or denominator in (None, 0.0):
        return None
    parsed = numerator / denominator
    return parsed if _is_reasonable_fps(parsed) else None


def _parse_duration_hms(text: str) -> float | None:
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    hours = _safe_float(match.group(1), 0.0) or 0.0
    minutes = _safe_float(match.group(2), 0.0) or 0.0
    seconds = _safe_float(match.group(3), 0.0) or 0.0
    duration = hours * 3600.0 + minutes * 60.0 + seconds
    return duration if duration > 0.0 else None


def _subprocess_creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _run_probe_command(command: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_subprocess_creationflags(),
            timeout=timeout,
            check=False,
        )
    except Exception:
        logger.debug("Video probe command failed: %s", command, exc_info=True)
        return None


def _unique_existing_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            unique.append(path)
    return unique


def _candidate_ffprobe_paths() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("MOUSESCOPE_FFPROBE_EXE")
    if env_path:
        candidates.append(Path(env_path))

    ffmpeg_exe = find_ffmpeg_exe()
    if ffmpeg_exe:
        ffmpeg_path = Path(ffmpeg_exe)
        candidates.append(ffmpeg_path.with_name("ffprobe.exe" if os.name == "nt" else "ffprobe"))
        candidates.append(ffmpeg_path.with_name("ffprobe"))
        candidates.append(ffmpeg_path.with_name("ffprobe.exe"))

    which_path = shutil.which("ffprobe")
    if which_path:
        candidates.append(Path(which_path))

    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))
    for root in roots:
        candidates.extend(
            [
                root / "ffprobe.exe",
                root / "ffprobe",
                root / "bin" / "ffprobe.exe",
                root / "bin" / "ffprobe",
            ]
        )

    return _unique_existing_paths(candidates)


def find_ffprobe_exe() -> str | None:
    for candidate in _candidate_ffprobe_paths():
        return str(candidate)
    return None


def _capture_video_props(video_path: str | os.PathLike, cap: cv2.VideoCapture | None = None) -> dict[str, Any]:
    local_cap = None
    active_cap = cap
    if active_cap is None:
        path = Path(str(video_path or ""))
        if not path.is_file():
            return {
                "opencv_fps": None,
                "width": 0,
                "height": 0,
                "total_frames": 0,
            }
        local_cap = cv2.VideoCapture(str(path))
        active_cap = local_cap

    try:
        if active_cap is None or not active_cap.isOpened():
            return {
                "opencv_fps": None,
                "width": 0,
                "height": 0,
                "total_frames": 0,
            }
        return {
            "opencv_fps": _safe_float(active_cap.get(cv2.CAP_PROP_FPS)),
            "width": _safe_int(active_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": _safe_int(active_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": _safe_int(active_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        if local_cap is not None:
            local_cap.release()


def _probe_with_ffprobe(video_path: str | os.PathLike) -> dict[str, Any]:
    ffprobe_exe = find_ffprobe_exe()
    if not ffprobe_exe:
        return {}

    command = [
        ffprobe_exe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = _run_probe_command(command)
    if proc is None or proc.returncode != 0 or not proc.stdout:
        return {}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}

    streams = data.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = data.get("format") or {}
    duration_s = _safe_float(stream.get("duration"))
    if not duration_s or duration_s <= 0.0:
        duration_s = _safe_float(fmt.get("duration"))

    return {
        "width": _safe_int(stream.get("width")),
        "height": _safe_int(stream.get("height")),
        "duration_seconds": duration_s if duration_s and duration_s > 0.0 else None,
        "nb_frames": _safe_int(stream.get("nb_frames")),
        "avg_frame_rate": _parse_fraction(stream.get("avg_frame_rate")),
        "nominal_fps": _parse_fraction(stream.get("r_frame_rate")),
    }


def _probe_with_ffmpeg_banner(video_path: str | os.PathLike) -> dict[str, Any]:
    ffmpeg_exe = find_ffmpeg_exe()
    if not ffmpeg_exe:
        return {}

    proc = _run_probe_command([ffmpeg_exe, "-hide_banner", "-i", str(video_path)])
    if proc is None:
        return {}
    output = f"{proc.stderr}\n{proc.stdout}"
    duration_s = _parse_duration_hms(output)
    fps_match = re.search(r"Stream .* Video: .*?,\s*([0-9]+(?:\.[0-9]+)?)\s*fps", output)
    nominal_fps = _safe_float(fps_match.group(1)) if fps_match else None
    return {
        "duration_seconds": duration_s,
        "nominal_fps": nominal_fps if _is_reasonable_fps(nominal_fps) else None,
    }


def _choose_detected_fps(
    *,
    opencv_fps: float | None,
    avg_frame_rate: float | None,
    nominal_fps: float | None,
    frame_count: int,
    duration_s: float | None,
) -> tuple[float, str]:
    duration_fps = None
    if frame_count > 0 and duration_s and duration_s > 0.0:
        duration_fps = frame_count / duration_s
        if not _is_reasonable_fps(duration_fps):
            duration_fps = None

    if duration_fps is not None and avg_frame_rate is not None:
        relative_delta = abs(duration_fps - avg_frame_rate) / max(duration_fps, avg_frame_rate, 1e-6)
        if relative_delta <= 0.02:
            return float(avg_frame_rate), "ffprobe_avg_frame_rate"
        return float(duration_fps), "frame_count_over_duration"

    if duration_fps is not None:
        return float(duration_fps), "frame_count_over_duration"
    if _is_reasonable_fps(avg_frame_rate):
        return float(avg_frame_rate), "ffprobe_avg_frame_rate"
    if _is_reasonable_fps(opencv_fps):
        return float(opencv_fps), "opencv"
    if _is_reasonable_fps(nominal_fps):
        return float(nominal_fps), "nominal"
    return DEFAULT_VIDEO_FPS, "default"


def probe_video_info(
    video_path: str | os.PathLike,
    fps_override: float | str | None = None,
    *,
    cap: cv2.VideoCapture | None = None,
) -> dict[str, Any]:
    props = _capture_video_props(video_path, cap=cap)
    path = Path(str(video_path or ""))
    if path.is_file():
        ffprobe_info = _probe_with_ffprobe(path)
        ffmpeg_info = {} if ffprobe_info.get("duration_seconds") else _probe_with_ffmpeg_banner(path)
    else:
        ffprobe_info = {}
        ffmpeg_info = {}

    duration_s = ffprobe_info.get("duration_seconds") or ffmpeg_info.get("duration_seconds")
    ffprobe_frames = _safe_int(ffprobe_info.get("nb_frames"))
    opencv_frames = _safe_int(props.get("total_frames"))
    total_frames = ffprobe_frames or opencv_frames

    width = _safe_int(props.get("width")) or _safe_int(ffprobe_info.get("width"))
    height = _safe_int(props.get("height")) or _safe_int(ffprobe_info.get("height"))
    opencv_fps = props.get("opencv_fps")
    nominal_fps = ffprobe_info.get("nominal_fps") or ffmpeg_info.get("nominal_fps")
    detected_fps, fps_source = _choose_detected_fps(
        opencv_fps=opencv_fps,
        avg_frame_rate=ffprobe_info.get("avg_frame_rate"),
        nominal_fps=nominal_fps,
        frame_count=total_frames,
        duration_s=duration_s,
    )

    override_fps = _safe_float(fps_override)
    fps = float(override_fps) if _is_reasonable_fps(override_fps) else float(detected_fps)
    return {
        "fps": fps,
        "detected_fps": float(detected_fps),
        "width": int(width),
        "height": int(height),
        "total_frames": int(total_frames),
        "duration_seconds": float(duration_s) if duration_s else None,
        "nominal_fps": float(nominal_fps) if _is_reasonable_fps(nominal_fps) else None,
        "opencv_fps": float(opencv_fps) if _is_reasonable_fps(opencv_fps) else None,
        "fps_source": fps_source,
    }
