from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from file_utils import ascii_slug  # noqa: E402
from services.analysis_sessions import VisionAnalysisSession, configure_analysis_runtime  # noqa: E402
from tools.tst_offline_optimizer import (  # noqa: E402
    FRAME_DT_FALLBACK,
    HU_WIN_END,
    HU_WIN_START,
    VIDEO_MOUSE_MAP,
    confusion_for,
    load_human_intervals,
    metrics_from_confusion,
    result_dirs,
)


CSV_ENCODING = "utf-8-sig"
CHINA_TZ = timezone(timedelta(hours=8))


def china_now() -> datetime:
    return datetime.now(CHINA_TZ)


def safe_display_stem(original_name: str, fallback: str = "video") -> str:
    stem = Path(str(original_name or "")).stem.strip()
    return stem or fallback


def batch_video_subdir(original_name: str, batch_index: int, batch_count: int, fallback: str = "video") -> str:
    total = max(int(batch_count or 1), 1)
    digits = max(2, len(str(total)))
    return f"{int(batch_index) + 1:0{digits}d}_{safe_display_stem(original_name, fallback=fallback)}"


def prepare_output_paths(
    session_id: str,
    original_name: str,
    suffix: str,
    result_folder: str | Path | None = None,
    fallback: str = "video",
    batch_index: int | None = None,
    batch_count: int = 1,
) -> dict[str, Any]:
    final_dir = Path(result_folder or ROOT / "data" / "results")
    if int(batch_count or 1) > 1 and batch_index is not None:
        final_dir = final_dir / batch_video_subdir(original_name, batch_index, batch_count, fallback=fallback)
    final_dir.mkdir(parents=True, exist_ok=True)

    work_dir = ROOT / "data" / "results" / "_work" / str(session_id)
    work_dir.mkdir(parents=True, exist_ok=True)

    display_stem = safe_display_stem(original_name, fallback=fallback)
    ascii_stem = ascii_slug(display_stem, fallback=fallback, max_length=80)
    return {
        "final_dir": final_dir,
        "work_dir": work_dir,
        "display_stem": display_stem,
        "work_base": str(work_dir / f"{ascii_stem}_{suffix}"),
        "final_base": str(final_dir / f"{display_stem}_{suffix}"),
    }


def export_result_file(work_path: str | Path, final_path: str | Path) -> str | None:
    work = Path(work_path)
    final = Path(final_path)
    if not work.exists():
        return None
    final.parent.mkdir(parents=True, exist_ok=True)
    if work.resolve() != final.resolve():
        final.write_bytes(work.read_bytes())
    return str(final)


def write_dataframe_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> bool:
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding=CSV_ENCODING)
    return Path(path).exists()


def bi_label(chinese: str, english: str) -> str:
    return f"{chinese} ({english})"


def report_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    try:
        if isinstance(value, float):
            return f"{value:.3f}{suffix}"
    except Exception:
        pass
    return f"{value}{suffix}"


def report_time_mode_label(value: Any) -> str:
    return str(value or "global")


def report_phase_label(value: Any) -> str:
    return str(value or "")


def write_report_metric_lines(handle, summary: dict[str, Any], specs: list[tuple[str, str, str, str]]) -> None:
    for cn, en, key, suffix in specs:
        handle.write(f"  {bi_label(cn, en)}: {report_value(summary.get(key), suffix)}\n")


def configure_runtime(result_dir: Path) -> None:
    configure_analysis_runtime(
        RESULT_DIR=result_dir,
        HAS_PANDAS=True,
        china_now=china_now,
        _prepare_output_paths=prepare_output_paths,
        _export_result_file=export_result_file,
        _write_dataframe_csv=write_dataframe_csv,
        _bi_label=bi_label,
        _report_value=report_value,
        _report_time_mode_label=report_time_mode_label,
        _report_phase_label=report_phase_label,
        _write_report_metric_lines=write_report_metric_lines,
        _analysis_status_observer=None,
    )


def latest_session_state(results_root: Path) -> Path:
    candidates = sorted(
        (results_root / "_work").glob("*/session_state.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("experiment_type") == "tst" and payload.get("video_paths"):
            return path
    raise FileNotFoundError(f"No TST session_state.json found under {results_root / '_work'}")


def load_session_state(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_type") != "tst":
        raise ValueError(f"{path} is not a TST session state.")
    if not payload.get("video_paths") or not payload.get("video_infos"):
        raise ValueError(f"{path} does not contain video_paths/video_infos.")
    return payload


def build_rerun_config(state: dict[str, Any], output_dir: Path, *, generate_video: bool, parallel: bool, workers: int) -> dict[str, Any]:
    config = copy.deepcopy(state.get("draft_payload") or {})
    config["output_dir"] = str(output_dir)
    config["generate_video"] = bool(generate_video)
    config["parallel_videos"] = bool(parallel)
    config["batch_workers"] = max(1, int(workers or 1))
    if state.get("start_seconds") is not None:
        config["start_sec"] = state.get("start_seconds")
    if state.get("end_seconds") is not None:
        config["end_sec"] = state.get("end_seconds")
    for per_video in config.get("per_video_configs") or []:
        if isinstance(per_video, dict):
            per_video["output_dir"] = str(output_dir)
            per_video["generate_video"] = bool(generate_video)
    return config


def run_tst_rerun(state: dict[str, Any], output_dir: Path, *, generate_video: bool, parallel: bool, workers: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(output_dir)

    video_paths = [str(Path(p)) for p in state["video_paths"]]
    missing = [p for p in video_paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing staged video files:\n" + "\n".join(missing))

    video_infos = copy.deepcopy(state["video_infos"])
    config = build_rerun_config(state, output_dir, generate_video=generate_video, parallel=parallel, workers=workers)
    session_id = f"tst-timebase-regression-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    session = VisionAnalysisSession(session_id, video_paths, video_infos, config)
    session.is_running = True
    session.status = "analyzing"
    session.start_time = time.time()
    session._run()
    status = str(session.status or "").lower()
    manifest = {
        "session_id": session_id,
        "status": session.status,
        "message": session.message,
        "progress": session.progress,
        "result_files": session.result_files,
        "output_dir": str(output_dir),
        "logs_tail": session.logs[-50:],
    }
    (output_dir / "rerun_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if status != "completed":
        raise RuntimeError(f"TST rerun failed: {session.status} {session.message}")
    return manifest


def frame_dts_from_times(times: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.asarray([], dtype=float)
    if len(times) == 1:
        return np.asarray([FRAME_DT_FALLBACK], dtype=float)
    dt = np.diff(times.astype(float))
    positive = dt[dt > 0]
    fallback = float(np.median(positive)) if len(positive) else FRAME_DT_FALLBACK
    dt = np.where(dt > 0, dt, fallback)
    return np.append(dt, dt[-1]).astype(float)


def bool_series(values: pd.Series) -> np.ndarray:
    if values.dtype == object:
        return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"}).to_numpy(dtype=bool)
    return values.astype(bool).to_numpy()


def numeric_series(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=float)


def load_algorithm_frames(results_root: Path, boris_file: Path, *, window_start: float, window_end: float) -> pd.DataFrame:
    human_intervals = load_human_intervals(boris_file)
    rows: list[pd.DataFrame] = []
    for video_ts, directory in result_dirs(results_root):
        log_paths = list(directory.glob("*_tst_logs.csv"))
        if len(log_paths) != 1:
            raise RuntimeError(f"{directory}: expected one *_tst_logs.csv, found {len(log_paths)}")
        log_df = pd.read_csv(log_paths[0])
        if "time" not in log_df.columns:
            raise RuntimeError(f"{log_paths[0]} does not have a time column.")

        for roi_id, mouse_id in enumerate(VIDEO_MOUSE_MAP[video_ts], start=1):
            roi_df = log_df[log_df["roi_id"] == roi_id].copy().sort_values("time").reset_index(drop=True)
            if roi_df.empty:
                raise RuntimeError(f"{directory}: empty ROI {roi_id} for mouse {mouse_id}")
            time_values = pd.to_numeric(roi_df["time"], errors="coerce").to_numpy(dtype=float)
            mask = (time_values >= window_start - 1e-6) & (time_values <= window_end + 1e-6)
            roi_df = roi_df.loc[mask].copy().reset_index(drop=True)
            if roi_df.empty:
                continue

            times = pd.to_numeric(roi_df["time"], errors="coerce").to_numpy(dtype=float)
            if "dt_s" in roi_df.columns:
                dts = pd.to_numeric(roi_df["dt_s"], errors="coerce").to_numpy(dtype=float)
                fallback = frame_dts_from_times(times)
                dts = np.where(np.isfinite(dts) & (dts > 0), dts, fallback)
            else:
                dts = frame_dts_from_times(times)

            human = np.zeros(len(roi_df), dtype=bool)
            for start, end in human_intervals.get(mouse_id, []):
                human |= (times >= start - FRAME_DT_FALLBACK / 2) & (times < end + FRAME_DT_FALLBACK / 2)

            frame_df = pd.DataFrame(
                {
                    "video_ts": video_ts,
                    "video_dir": directory.name,
                    "roi_id": roi_id,
                    "mouse_id": mouse_id,
                    "time_s": times,
                    "dt_s": dts,
                    "human_struggle": human,
                    "algorithm_struggle": bool_series(roi_df["is_struggling"]) if "is_struggling" in roi_df.columns else False,
                    "algorithm_swing": bool_series(roi_df["is_swinging"]) if "is_swinging" in roi_df.columns else False,
                    "decision_flow_p90_mm_s": numeric_series(roi_df, "decision_flow_p90_mm_s"),
                    "speed_std_mm_s": numeric_series(roi_df, "speed_std_mm_s"),
                    "flow_entropy": numeric_series(roi_df, "flow_entropy"),
                    "acf_peak": numeric_series(roi_df, "acf_peak"),
                }
            )
            rows.append(frame_df)

    if not rows:
        raise RuntimeError(f"No TST frame logs loaded from {results_root}")
    return pd.concat(rows, ignore_index=True)


def metrics_for_frame_df(df: pd.DataFrame) -> dict[str, float]:
    return metrics_from_confusion(
        *confusion_for(
            df["algorithm_struggle"].to_numpy(dtype=bool),
            df["human_struggle"].to_numpy(dtype=bool),
            df["dt_s"].to_numpy(dtype=float),
        )
    )


def rounded_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.floating)):
            out[f"{prefix}_{key}"] = round(float(value), 6)
        else:
            out[f"{prefix}_{key}"] = value
    return out


def grouped_metrics(df: pd.DataFrame, dataset: str, group_cols: list[str]) -> pd.DataFrame:
    records = []
    for key, group in df.groupby(group_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        record = {col: value for col, value in zip(group_cols, key)}
        metrics = metrics_for_frame_df(group)
        record.update({k: round(float(v), 6) for k, v in metrics.items()})
        record["dataset"] = dataset
        record["human_struggle_s"] = round(float(group.loc[group["human_struggle"], "dt_s"].sum()), 6)
        record["algorithm_struggle_s"] = round(float(group.loc[group["algorithm_struggle"], "dt_s"].sum()), 6)
        record["algorithm_swing_s"] = round(float(group.loc[group["algorithm_swing"], "dt_s"].sum()), 6)
        record["evaluated_s"] = round(float(group["dt_s"].sum()), 6)
        records.append(record)
    return pd.DataFrame(records)


def load_timebase_summary(results_root: Path) -> pd.DataFrame:
    rows = []
    for video_ts, directory in result_dirs(results_root):
        summaries = list(directory.glob("*_tst_immobility.csv"))
        if not summaries:
            continue
        summary = pd.read_csv(summaries[0])
        for _, row in summary.iterrows():
            roi_id = int(row.get("roi_id", 0) or 0)
            mouse_ids = VIDEO_MOUSE_MAP.get(video_ts, [])
            mouse_id = mouse_ids[roi_id - 1] if 1 <= roi_id <= len(mouse_ids) else None
            rows.append(
                {
                    "video_ts": video_ts,
                    "video_dir": directory.name,
                    "roi_id": roi_id,
                    "mouse_id": mouse_id,
                    "timebase_source": row.get("timebase_source"),
                    "timebase_median_dt_s": row.get("timebase_median_dt_s"),
                    "timebase_long_gap_gt_50ms": row.get("timebase_long_gap_gt_50ms"),
                    "analysis_fps": row.get("analysis_fps"),
                    "source_fps": row.get("source_fps"),
                }
            )
    return pd.DataFrame(rows)


def write_summary_markdown(
    path: Path,
    *,
    baseline_root: Path,
    new_root: Path,
    boris_file: Path,
    window_start: float,
    window_end: float,
    overall: pd.DataFrame,
    comparison: pd.DataFrame,
    timebase_summary: pd.DataFrame,
    rerun_manifest: dict[str, Any] | None,
) -> None:
    baseline = overall[overall["dataset"] == "baseline"].iloc[0].to_dict()
    new = overall[overall["dataset"] == "new"].iloc[0].to_dict()
    improved = int((comparison["accuracy_delta"] > 0).sum()) if not comparison.empty else 0
    regressed = int((comparison["accuracy_delta"] < 0).sum()) if not comparison.empty else 0
    unchanged = int((comparison["accuracy_delta"] == 0).sum()) if not comparison.empty else 0
    fn_delta = float(new["FN_s"] - baseline["FN_s"])
    fp_delta = float(new["FP_s"] - baseline["FP_s"])

    lines = [
        "# TST timebase regression summary",
        "",
        f"- Baseline root: `{baseline_root}`",
        f"- New root: `{new_root}`",
        f"- BORIS file: `{boris_file}`",
        f"- Evaluation window: {window_start:.0f}-{window_end:.0f}s",
        f"- Rerun status: `{(rerun_manifest or {}).get('status', 'not_run')}`",
        "",
        "## Overall",
        "",
        "| dataset | accuracy | recall | precision | FN_s | FP_s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in (baseline, new):
        lines.append(
            f"| {item['dataset']} | {item['accuracy']:.4f} | {item['struggle_recall']:.4f} | "
            f"{item['struggle_precision']:.4f} | {item['FN_s']:.2f} | {item['FP_s']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Delta",
            "",
            f"- Accuracy delta: {float(new['accuracy'] - baseline['accuracy']):+.4f}",
            f"- Recall delta: {float(new['struggle_recall'] - baseline['struggle_recall']):+.4f}",
            f"- Precision delta: {float(new['struggle_precision'] - baseline['struggle_precision']):+.4f}",
            f"- Missed struggle seconds delta (new - baseline): {fn_delta:+.2f}s",
            f"- False struggle seconds delta (new - baseline): {fp_delta:+.2f}s",
            f"- Per-mouse accuracy improved/regressed/unchanged: {improved}/{regressed}/{unchanged}",
            "",
            "## Timebase",
            "",
        ]
    )
    if timebase_summary.empty:
        lines.append("- No timebase summary fields found.")
    else:
        source_counts = timebase_summary["timebase_source"].fillna("unknown").value_counts().to_dict()
        lines.append(f"- Sources: {source_counts}")
        if "timebase_long_gap_gt_50ms" in timebase_summary:
            gaps = pd.to_numeric(timebase_summary["timebase_long_gap_gt_50ms"], errors="coerce").fillna(0)
            lines.append(f"- Long gaps >50ms, total across ROI summaries: {int(gaps.sum())}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `overall_metrics.csv`",
            "- `per_mouse_metrics.csv`",
            "- `per_video_metrics.csv`",
            "- `per_mouse_comparison.csv`",
            "- `timebase_summary.csv`",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_pair(
    baseline_root: Path,
    new_root: Path,
    boris_file: Path,
    out_dir: Path,
    *,
    window_start: float,
    window_end: float,
    rerun_manifest: dict[str, Any] | None,
) -> None:
    baseline_frames = load_algorithm_frames(baseline_root, boris_file, window_start=window_start, window_end=window_end)
    new_frames = load_algorithm_frames(new_root, boris_file, window_start=window_start, window_end=window_end)

    overall_rows = []
    for dataset, frames in (("baseline", baseline_frames), ("new", new_frames)):
        metrics = metrics_for_frame_df(frames)
        row = {"dataset": dataset}
        row.update({k: round(float(v), 6) for k, v in metrics.items()})
        row["human_struggle_s"] = round(float(frames.loc[frames["human_struggle"], "dt_s"].sum()), 6)
        row["algorithm_struggle_s"] = round(float(frames.loc[frames["algorithm_struggle"], "dt_s"].sum()), 6)
        row["algorithm_swing_s"] = round(float(frames.loc[frames["algorithm_swing"], "dt_s"].sum()), 6)
        row["evaluated_s"] = round(float(frames["dt_s"].sum()), 6)
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)

    per_mouse = pd.concat(
        [
            grouped_metrics(baseline_frames, "baseline", ["video_ts", "roi_id", "mouse_id"]),
            grouped_metrics(new_frames, "new", ["video_ts", "roi_id", "mouse_id"]),
        ],
        ignore_index=True,
    )
    per_video = pd.concat(
        [
            grouped_metrics(baseline_frames, "baseline", ["video_ts"]),
            grouped_metrics(new_frames, "new", ["video_ts"]),
        ],
        ignore_index=True,
    )

    base_mouse = per_mouse[per_mouse["dataset"] == "baseline"].drop(columns=["dataset"])
    new_mouse = per_mouse[per_mouse["dataset"] == "new"].drop(columns=["dataset"])
    comparison = base_mouse.merge(
        new_mouse,
        on=["video_ts", "roi_id", "mouse_id"],
        suffixes=("_baseline", "_new"),
        how="outer",
    )
    for metric in ("accuracy", "struggle_recall", "struggle_precision", "FN_s", "FP_s", "algorithm_struggle_s", "algorithm_swing_s"):
        comparison[f"{metric}_delta"] = comparison[f"{metric}_new"] - comparison[f"{metric}_baseline"]
    comparison = comparison.sort_values(["video_ts", "roi_id"]).reset_index(drop=True)

    timebase_summary = load_timebase_summary(new_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(out_dir / "overall_metrics.csv", index=False, encoding=CSV_ENCODING)
    per_mouse.to_csv(out_dir / "per_mouse_metrics.csv", index=False, encoding=CSV_ENCODING)
    per_video.to_csv(out_dir / "per_video_metrics.csv", index=False, encoding=CSV_ENCODING)
    comparison.to_csv(out_dir / "per_mouse_comparison.csv", index=False, encoding=CSV_ENCODING)
    timebase_summary.to_csv(out_dir / "timebase_summary.csv", index=False, encoding=CSV_ENCODING)
    write_summary_markdown(
        out_dir / "regression_summary.md",
        baseline_root=baseline_root,
        new_root=new_root,
        boris_file=boris_file,
        window_start=window_start,
        window_end=window_end,
        overall=overall,
        comparison=comparison,
        timebase_summary=timebase_summary,
        rerun_manifest=rerun_manifest,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun and evaluate the 2024-11-29 TST timebase regression set.")
    parser.add_argument("--results-root", default=str(ROOT / "data" / "results"))
    parser.add_argument("--baseline-root", default=str(ROOT / "data" / "results"))
    parser.add_argument("--boris-file", default=r"E:\20241129\csds_cvs_ctrl_tst.boris")
    parser.add_argument("--session-state", default="")
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "results" / "tst_timebase_regression"))
    parser.add_argument("--new-results-root", default="")
    parser.add_argument("--rerun", action="store_true", help="Run the 9 videos with the current code before evaluating.")
    parser.add_argument("--generate-video", action="store_true", help="Also export annotated videos during rerun.")
    parser.add_argument("--parallel", action="store_true", help="Run videos in parallel during rerun.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--window-start", type=float, default=HU_WIN_START)
    parser.add_argument("--window-end", type=float, default=HU_WIN_END)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    baseline_root = Path(args.baseline_root)
    boris_file = Path(args.boris_file)
    out_dir = Path(args.out_dir)
    rerun_root = Path(args.new_results_root) if args.new_results_root else out_dir / "rerun_results"

    if not boris_file.exists():
        raise FileNotFoundError(f"BORIS file not found: {boris_file}")

    rerun_manifest = None
    if args.rerun:
        session_state = Path(args.session_state) if args.session_state else latest_session_state(results_root)
        state = load_session_state(session_state)
        rerun_manifest = run_tst_rerun(
            state,
            rerun_root,
            generate_video=args.generate_video,
            parallel=args.parallel,
            workers=args.workers,
        )
    elif not rerun_root.exists():
        raise FileNotFoundError(
            f"New results root does not exist: {rerun_root}. Use --rerun or pass --new-results-root."
        )
    else:
        manifest_path = rerun_root / "rerun_manifest.json"
        if manifest_path.exists():
            try:
                rerun_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                rerun_manifest = None

    evaluate_pair(
        baseline_root,
        rerun_root,
        boris_file,
        out_dir,
        window_start=float(args.window_start),
        window_end=float(args.window_end),
        rerun_manifest=rerun_manifest,
    )
    print(f"Regression report written to: {out_dir / 'regression_summary.md'}")


if __name__ == "__main__":
    main()
