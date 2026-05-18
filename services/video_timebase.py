from __future__ import annotations

import os
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from services.video_metadata import DEFAULT_VIDEO_FPS, find_ffprobe_exe


logger = logging.getLogger(__name__)


def _subprocess_creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _safe_float(value, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _run_text(command: list[str], timeout: float = 60.0) -> str | None:
    try:
        proc = subprocess.run(
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
        logger.debug("Timebase probe command failed: %s", command, exc_info=True)
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _parse_pts_lines(output: str | None) -> list[float]:
    if not output:
        return []
    values: list[float] = []
    for line in output.splitlines():
        text = line.strip().split(",", 1)[0]
        if not text or text == "N/A":
            continue
        parsed = _safe_float(text)
        if parsed is not None:
            values.append(float(parsed))
    return values


def _extract_pts_times(video_path: str | os.PathLike, timeout: float = 60.0) -> np.ndarray:
    ffprobe_exe = find_ffprobe_exe()
    if not ffprobe_exe:
        return np.asarray([], dtype=np.float64)

    path = str(video_path)
    packet_output = _run_text(
        [
            ffprobe_exe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
            path,
        ],
        timeout=timeout,
    )
    values = _parse_pts_lines(packet_output)

    if not values:
        frame_output = _run_text(
            [
                ffprobe_exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "csv=p=0",
                path,
            ],
            timeout=timeout,
        )
        values = _parse_pts_lines(frame_output)

    if not values:
        return np.asarray([], dtype=np.float64)

    times = np.asarray(sorted(values), dtype=np.float64)
    finite = np.isfinite(times)
    times = times[finite]
    if times.size == 0:
        return times
    times = times - float(times[0])
    return times


def _median_positive_dt(times: np.ndarray, fallback_fps: float) -> float:
    fallback = 1.0 / max(float(fallback_fps or DEFAULT_VIDEO_FPS), 1e-6)
    if times.size <= 1:
        return fallback
    diffs = np.diff(times)
    positive = diffs[diffs > 1e-9]
    if positive.size == 0:
        return fallback
    return float(np.median(positive))


def _repair_and_size_times(times: np.ndarray, total_frames: int, fps: float) -> np.ndarray:
    total_frames = int(max(0, total_frames or 0))
    fallback_dt = _median_positive_dt(times, fps)
    if times.size == 0:
        if total_frames <= 0:
            return np.asarray([], dtype=np.float64)
        return np.arange(total_frames, dtype=np.float64) * fallback_dt

    repaired = np.asarray(times, dtype=np.float64).copy()
    for idx in range(1, repaired.size):
        if not np.isfinite(repaired[idx]) or repaired[idx] <= repaired[idx - 1]:
            repaired[idx] = repaired[idx - 1] + fallback_dt

    if total_frames <= 0:
        return repaired
    if repaired.size > total_frames:
        return repaired[:total_frames]
    if repaired.size < total_frames:
        missing = total_frames - repaired.size
        tail = repaired[-1] + np.arange(1, missing + 1, dtype=np.float64) * fallback_dt
        repaired = np.concatenate([repaired, tail])
    return repaired


@dataclass(frozen=True)
class VideoTimebase:
    times: np.ndarray
    fallback_fps: float = DEFAULT_VIDEO_FPS
    source: str = "uniform"
    pts_frame_count: int = 0

    @property
    def frame_count(self) -> int:
        return int(self.times.size)

    @property
    def nominal_dt(self) -> float:
        return 1.0 / max(float(self.fallback_fps or DEFAULT_VIDEO_FPS), 1e-6)

    @property
    def median_dt(self) -> float:
        return _median_positive_dt(self.times, self.fallback_fps)

    def time_at(self, frame_index: int) -> float:
        idx = int(frame_index)
        if 0 <= idx < self.times.size:
            return float(self.times[idx])
        return max(0, idx) * self.nominal_dt

    def duration_between(
        self,
        previous_frame_index: int | None,
        frame_index: int,
        *,
        sample_fps: float | None = None,
    ) -> float:
        fallback = 1.0 / max(float(sample_fps or self.fallback_fps or DEFAULT_VIDEO_FPS), 1e-6)
        if previous_frame_index is None:
            return fallback
        dt = self.time_at(int(frame_index)) - self.time_at(int(previous_frame_index))
        if not np.isfinite(dt) or dt <= 0.0:
            return fallback
        return float(dt)

    def frame_at_or_after(self, seconds: float | int | None) -> int:
        if seconds is None:
            return 0
        if self.times.size == 0:
            return max(0, int(round(float(seconds) * self.fallback_fps)))
        idx = int(np.searchsorted(self.times, float(seconds), side="left"))
        return max(0, min(idx, self.times.size - 1))

    def frame_at_or_before(self, seconds: float | int | None) -> int:
        if self.times.size == 0:
            if seconds is None:
                return 0
            return max(0, int(round(float(seconds) * self.fallback_fps)))
        if seconds is None:
            return self.times.size - 1
        idx = int(np.searchsorted(self.times, float(seconds), side="right")) - 1
        return max(0, min(idx, self.times.size - 1))

    def summary(self) -> dict[str, float | int | str]:
        if self.times.size <= 1:
            return {
                "source": self.source,
                "frame_count": int(self.times.size),
                "pts_frame_count": int(self.pts_frame_count),
                "median_dt": self.nominal_dt,
                "max_dt": self.nominal_dt,
                "long_gap_gt_50ms": 0,
            }
        diffs = np.diff(self.times)
        positive = diffs[diffs > 0]
        median_dt = float(np.median(positive)) if positive.size else self.nominal_dt
        return {
            "source": self.source,
            "frame_count": int(self.times.size),
            "pts_frame_count": int(self.pts_frame_count),
            "median_dt": median_dt,
            "max_dt": float(np.max(positive)) if positive.size else median_dt,
            "long_gap_gt_50ms": int(np.sum(positive > 0.050)),
        }


def build_video_timebase(
    video_path: str | os.PathLike,
    *,
    fps: float | None,
    total_frames: int | None,
    prefer_pts: bool = True,
    logger: Callable[[str], None] | None = None,
) -> VideoTimebase:
    fallback_fps = float(fps or DEFAULT_VIDEO_FPS)
    total = int(max(0, total_frames or 0))
    pts = _extract_pts_times(video_path) if prefer_pts else np.asarray([], dtype=np.float64)
    source = "pts" if pts.size > 0 else "uniform"
    if logger is not None and prefer_pts and pts.size == 0:
        try:
            logger("Video PTS timebase unavailable; falling back to uniform fps timing.")
        except Exception:
            pass
    times = _repair_and_size_times(pts, total, fallback_fps)
    return VideoTimebase(
        times=times,
        fallback_fps=fallback_fps,
        source=source,
        pts_frame_count=int(pts.size),
    )
