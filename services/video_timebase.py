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
MIN_PTS_FRAME_COVERAGE = 0.5
MIN_PTS_DURATION_RATIO = 0.5
MAX_PTS_DURATION_RATIO = 2.0


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


def _normalise_pts_values(values: list[float]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.float64)

    times = np.asarray(sorted(values), dtype=np.float64)
    finite = np.isfinite(times)
    times = times[finite]
    if times.size == 0:
        return times
    times = times - float(times[0])
    return times


def _ratio_outside(value: float, expected: float) -> bool:
    if not np.isfinite(value) or not np.isfinite(expected) or expected <= 0.0:
        return False
    ratio = float(value) / float(expected)
    return ratio < MIN_PTS_DURATION_RATIO or ratio > MAX_PTS_DURATION_RATIO


def _pts_inconsistency_reason(times: np.ndarray, total_frames: int | None, fps: float | None) -> str | None:
    frame_count = int(max(0, total_frames or 0))
    if times.size <= 1:
        if frame_count > max(int(times.size), 1):
            return f"only {times.size}/{frame_count} timestamps were available"
        return None

    fallback_fps = float(fps or DEFAULT_VIDEO_FPS)
    if fallback_fps <= 0.0:
        return None

    if frame_count > 1:
        coverage = float(times.size) / float(frame_count)
        if coverage < MIN_PTS_FRAME_COVERAGE:
            return f"only {times.size}/{frame_count} timestamps were available"

    observed_span = float(times[-1])
    expected_observed_span = float(times.size - 1) / fallback_fps
    if _ratio_outside(observed_span, expected_observed_span):
        return (
            f"timestamp span {observed_span:.3f}s disagrees with "
            f"{times.size} frames at {fallback_fps:.3f}fps"
        )

    if frame_count > 1:
        median_dt = _median_positive_dt(times, fallback_fps)
        if times.size < frame_count:
            padded_span = observed_span + float(frame_count - times.size) * median_dt
        else:
            padded_span = float(times[frame_count - 1])
        expected_total_span = float(frame_count - 1) / fallback_fps
        if _ratio_outside(padded_span, expected_total_span):
            return (
                f"padded timestamp span {padded_span:.3f}s disagrees with "
                f"{frame_count} frames at {fallback_fps:.3f}fps"
            )

    return None


def _log_pts_rejection(logger_func: Callable[[str], None] | None, label: str, reason: str) -> None:
    if logger_func is None:
        return
    try:
        logger_func(f"Video {label} PTS ignored: {reason}.")
    except Exception:
        pass


def _extract_pts_candidate(
    command: list[str],
    *,
    label: str,
    timeout: float,
    total_frames: int | None,
    fps: float | None,
    logger: Callable[[str], None] | None,
) -> np.ndarray:
    values = _parse_pts_lines(_run_text(command, timeout=timeout))
    times = _normalise_pts_values(values)
    reason = _pts_inconsistency_reason(times, total_frames, fps)
    if reason:
        _log_pts_rejection(logger, label, reason)
        return np.asarray([], dtype=np.float64)
    return times


def _extract_pts_times(
    video_path: str | os.PathLike,
    timeout: float = 60.0,
    *,
    total_frames: int | None = None,
    fps: float | None = None,
    logger: Callable[[str], None] | None = None,
) -> np.ndarray:
    ffprobe_exe = find_ffprobe_exe()
    if not ffprobe_exe:
        return np.asarray([], dtype=np.float64)

    path = str(video_path)
    packet_command = [
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
    ]
    frame_command = [
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
    ]

    times = _extract_pts_candidate(
        packet_command,
        label="packet",
        timeout=timeout,
        total_frames=total_frames,
        fps=fps,
        logger=logger,
    )
    if times.size > 0:
        return times

    times = _extract_pts_candidate(
        frame_command,
        label="frame",
        timeout=timeout,
        total_frames=total_frames,
        fps=fps,
        logger=logger,
    )
    if times.size > 0:
        return times

    return np.asarray([], dtype=np.float64)


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

    def frame_before(self, seconds: float | int | None) -> int:
        """Return the last frame strictly before an exclusive end time."""
        if self.times.size == 0:
            if seconds is None:
                return 0
            return max(0, int(np.ceil(float(seconds) * self.fallback_fps)) - 1)
        if seconds is None:
            return self.times.size - 1
        idx = int(np.searchsorted(self.times, float(seconds), side="left")) - 1
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
        long_gap_count = 0 if self.source == "uniform" else int(np.sum(positive > 0.050))
        return {
            "source": self.source,
            "frame_count": int(self.times.size),
            "pts_frame_count": int(self.pts_frame_count),
            "median_dt": median_dt,
            "max_dt": float(np.max(positive)) if positive.size else median_dt,
            "long_gap_gt_50ms": long_gap_count,
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
    pts = (
        _extract_pts_times(video_path, total_frames=total, fps=fallback_fps, logger=logger)
        if prefer_pts
        else np.asarray([], dtype=np.float64)
    )
    source = "pts" if pts.size > 0 else "uniform"
    if logger is not None and prefer_pts and pts.size == 0:
        try:
            logger("Video PTS timebase unavailable or inconsistent; falling back to uniform fps timing.")
        except Exception:
            pass
    times = _repair_and_size_times(pts, total, fallback_fps)
    return VideoTimebase(
        times=times,
        fallback_fps=fallback_fps,
        source=source,
        pts_frame_count=int(pts.size),
    )
