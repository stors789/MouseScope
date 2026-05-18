from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _run_json(command: list[str]) -> dict:
    output = subprocess.check_output(command, text=True, encoding="utf-8", errors="ignore")
    return json.loads(output)


def _packet_pts(video_path: Path) -> list[float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts_time",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    output = subprocess.check_output(command, text=True, encoding="utf-8", errors="ignore")
    values: list[float] = []
    for line in output.splitlines():
        text = line.strip().split(",")[0]
        if not text or text == "N/A":
            continue
        try:
            values.append(float(text))
        except ValueError:
            continue
    # OpenCV returns frames in presentation order. Sorting packet PTS gives the
    # presentation timeline even if the container stores packets out of order.
    return sorted(values)


def extract_pts_dataframe(video_path: str | Path) -> pd.DataFrame:
    video_path = Path(video_path)
    meta = _run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames,duration,time_base",
            "-of",
            "json",
            str(video_path),
        ]
    )
    stream = (meta.get("streams") or [{}])[0]
    pts = np.asarray(_packet_pts(video_path), dtype=np.float64)
    if pts.size == 0:
        raise RuntimeError(f"ffprobe did not return PTS values for {video_path}")

    dt = np.diff(pts)
    if dt.size:
        tail = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 1.0 / 30.0
        dt = np.append(dt, tail)
    else:
        dt = np.asarray([1.0 / 30.0], dtype=np.float64)

    df = pd.DataFrame(
        {
            "frame": np.arange(len(pts), dtype=np.int64),
            "pts_time": pts,
            "pts_dt": dt,
            "video_path": str(video_path),
            "r_frame_rate": stream.get("r_frame_rate"),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "duration": stream.get("duration"),
            "nb_frames": stream.get("nb_frames"),
            "time_base": stream.get("time_base"),
        }
    )
    return df


def load_or_extract_pts(video_path: str | Path, cache_dir: str | Path | None = None) -> pd.DataFrame:
    video_path = Path(video_path)
    if cache_dir is None:
        return extract_pts_dataframe(video_path)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem.replace(":", "-")
    cache_path = cache_dir / f"{stem}_pts.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    df = extract_pts_dataframe(video_path)
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return df


def frame_times_for(video_path: str | Path, cache_dir: str | Path | None = None) -> np.ndarray:
    df = load_or_extract_pts(video_path, cache_dir)
    return df["pts_time"].to_numpy(dtype=np.float64)


def frame_dts_from_times(times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    if len(times) <= 1:
        return np.asarray([1.0 / 30.0] * len(times), dtype=np.float64)
    dts = np.diff(times)
    positive = dts[dts > 0]
    fallback = float(np.median(positive)) if len(positive) else 1.0 / 30.0
    dts = np.where(dts > 0, dts, fallback)
    return np.append(dts, dts[-1])


def summarize_pts(df: pd.DataFrame) -> dict[str, float | int]:
    pts = df["pts_time"].to_numpy(dtype=np.float64)
    dts = frame_dts_from_times(pts)
    median = float(np.median(dts)) if len(dts) else 0.0
    return {
        "frames": int(len(pts)),
        "dt_mean": float(np.mean(dts)) if len(dts) else 0.0,
        "dt_median": median,
        "dt_min": float(np.min(dts)) if len(dts) else 0.0,
        "dt_max": float(np.max(dts)) if len(dts) else 0.0,
        "jitter_gt_1ms": int(np.sum(np.abs(dts - median) > 0.001)) if len(dts) else 0,
        "jitter_gt_5ms": int(np.sum(np.abs(dts - median) > 0.005)) if len(dts) else 0,
        "long_gap_gt_50ms": int(np.sum(dts > 0.050)) if len(dts) else 0,
    }
