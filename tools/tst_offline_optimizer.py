from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VIDEO_MOUSE_MAP = {
    "11-02-42-882": [1, 2, 3, 4],
    "11-14-37-601": [5, 6, 7, 8],
    "11-28-29-994": [9, 10, 11, 12],
    "11-40-48-715": [13, 14, 15, 16],
    "11-56-10-908": [17, 18, 19, 20],
    "12-08-06-116": [21, 22, 23, 24],
    "12-21-27-977": [25, 26, 27, 28],
    "12-34-22-351": [29, 30, 31, 32],
    "12-44-36-199": [33],
}

TS_RE = re.compile(r"(\d{2}-\d{2}-\d{2}-\d{3})")
HU_WIN_START = 120.0
HU_WIN_END = 360.0
FRAME_DT_FALLBACK = 0.04


@dataclass(frozen=True)
class Candidate:
    min_dur_s: float
    area_thr: float
    aspect_thr: float
    y_thr: float
    tiny_flow_thr: float
    tiny_speed_thr: float
    tiny_entropy_thr: float
    swing_score_thr: float = 4.0
    swing_acf_thr: float = 0.55
    swing_entropy_thr: float = 0.08
    swing_fos_std_thr: float = 3.2

    @property
    def key(self) -> str:
        return (
            f"dur={self.min_dur_s:g}|area={self.area_thr:g}|aspect={self.aspect_thr:g}|"
            f"y={self.y_thr:g}|flow={self.tiny_flow_thr:g}|speed={self.tiny_speed_thr:g}|"
            f"ent={self.tiny_entropy_thr:g}"
        )

    def as_dict(self) -> dict[str, float | str]:
        return {
            "candidate_key": self.key,
            "min_dur_s": self.min_dur_s,
            "area_thr": self.area_thr,
            "aspect_thr": self.aspect_thr,
            "y_thr": self.y_thr,
            "tiny_flow_thr": self.tiny_flow_thr,
            "tiny_speed_thr": self.tiny_speed_thr,
            "tiny_entropy_thr": self.tiny_entropy_thr,
            "swing_score_thr": self.swing_score_thr,
            "swing_acf_thr": self.swing_acf_thr,
            "swing_entropy_thr": self.swing_entropy_thr,
            "swing_fos_std_thr": self.swing_fos_std_thr,
        }


def frame_dts(times: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(times), dtype=float)
    if len(arr) <= 1:
        return np.array([FRAME_DT_FALLBACK] * len(arr), dtype=float)
    d = np.diff(arr)
    positive = d[d > 0]
    fallback = float(np.median(positive)) if len(positive) else FRAME_DT_FALLBACK
    d = np.where(d > 0, d, fallback)
    return np.append(d, d[-1])


def load_human_intervals(boris_file: Path) -> dict[int, list[tuple[float, float]]]:
    with open(boris_file, encoding="utf-8") as fh:
        boris = json.load(fh)

    obs = boris["observations"]["tst"]
    subj_events: dict[str, list] = defaultdict(list)
    for event in obs["events"]:
        subj_events[event[1]].append(event)

    offsets = []
    cumulative = 0.0
    lengths = obs["media_info"]["length"]
    for video_file in list(obs["file"]["1"]):
        length = lengths.get(video_file.replace("/", "\\"), lengths.get(video_file.replace("\\", "/"), 0))
        offsets.append(cumulative)
        cumulative += float(length)

    intervals: dict[int, list[tuple[float, float]]] = {}
    for mouse_id_text, events in subj_events.items():
        mouse_id = int(mouse_id_text)
        events = sorted(events, key=lambda item: float(item[0]))
        if not events:
            continue
        first_time = float(events[0][0])
        video_offset = 0.0
        for offset in reversed(offsets):
            if offset <= first_time + 1e-6:
                video_offset = offset
                break

        mouse_intervals = []
        for idx in range(0, len(events) - 1, 2):
            start = float(events[idx][0]) - video_offset
            end = float(events[idx + 1][0]) - video_offset
            if end > start:
                mouse_intervals.append((start, end))
        intervals[mouse_id] = mouse_intervals
    return intervals


def result_dirs(results_root: Path) -> list[tuple[str, Path]]:
    found = []
    for path in results_root.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"^\d{2}_bandicam 2024-11-29 (\d{2}-\d{2}-\d{2}-\d{3})$", path.name)
        if match and match.group(1) in VIDEO_MOUSE_MAP:
            found.append((match.group(1), path))
    order = {ts: idx for idx, ts in enumerate(VIDEO_MOUSE_MAP)}
    return sorted(found, key=lambda item: order[item[0]])


def numeric_col(df: pd.DataFrame, name: str, default: float = 0.0) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), default, dtype=float)
    return (
        pd.to_numeric(df[name], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .to_numpy(dtype=float)
    )


def bool_col(df: pd.DataFrame, name: str, default: bool = False) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), default, dtype=bool)
    values = df[name]
    if values.dtype == object:
        values = values.astype(str).str.lower().map({"true": True, "false": False}).fillna(default)
    return values.astype(bool).to_numpy()


def load_frame_data(results_root: Path, boris_file: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    human_intervals = load_human_intervals(boris_file)
    rows = []
    full_logs: dict[int, pd.DataFrame] = {}

    for video_ts, directory in result_dirs(results_root):
        logs = list(directory.glob("*_tst_logs.csv"))
        if len(logs) != 1:
            raise RuntimeError(f"{directory}: expected one *_tst_logs.csv, found {len(logs)}")
        log_df = pd.read_csv(logs[0])
        for roi_id, mouse_id in enumerate(VIDEO_MOUSE_MAP[video_ts], start=1):
            roi_df = log_df[log_df["roi_id"] == roi_id].copy().sort_values("time").reset_index(drop=True)
            if roi_df.empty:
                raise RuntimeError(f"{directory}: empty ROI {roi_id} for mouse {mouse_id}")
            roi_df["mouse_id"] = mouse_id
            roi_df["video_ts"] = video_ts
            full_logs[mouse_id] = roi_df

            window = roi_df[
                (roi_df["time"].astype(float) >= HU_WIN_START - 1e-6)
                & (roi_df["time"].astype(float) <= HU_WIN_END + 1e-6)
            ].copy().sort_values("time").reset_index(drop=True)
            times = window["time"].to_numpy(dtype=float)
            dts = frame_dts(times - HU_WIN_START)

            human_struggle = np.zeros(len(window), dtype=bool)
            for start, end in human_intervals.get(mouse_id, []):
                mask = (times >= start - FRAME_DT_FALLBACK / 2) & (times < end + FRAME_DT_FALLBACK / 2)
                human_struggle[mask] = True

            data = {
                "mouse_id": np.full(len(window), mouse_id, dtype=int),
                "video_ts": np.full(len(window), video_ts, dtype=object),
                "roi_id": np.full(len(window), roi_id, dtype=int),
                "time": times,
                "dt": dts,
                "human_struggle": human_struggle,
                "current_struggle": bool_col(window, "is_struggling"),
                "decision_flow_p90_mm_s": numeric_col(window, "decision_flow_p90_mm_s"),
                "speed_std_mm_s": numeric_col(window, "speed_std_mm_s"),
                "flow_entropy": numeric_col(window, "flow_entropy"),
                "swing_score": numeric_col(window, "swing_score"),
                "struggle_score": numeric_col(window, "struggle_score"),
                "acf_peak": numeric_col(window, "acf_peak"),
                "flow_over_speed_std": numeric_col(window, "flow_over_speed_std"),
                "aspect_range_ratio": numeric_col(window, "aspect_range_ratio"),
                "y_range_body": numeric_col(window, "y_range_body"),
                "area_range_ratio": numeric_col(window, "area_range_ratio"),
                "is_swinging": bool_col(window, "is_swinging"),
                "rigid_body_like": bool_col(window, "rigid_body_like"),
                "internal_flow_ok": bool_col(window, "internal_flow_ok"),
            }
            rows.append(pd.DataFrame(data))

    frames = pd.concat(rows, ignore_index=True)
    return frames, full_logs


def metrics_from_confusion(tp: float, fn: float, fp: float, tn: float) -> dict[str, float]:
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total if total else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    false_rate = fp / (fp + tn) if fp + tn else math.nan
    missed_rate = fn / (tp + fn) if tp + fn else math.nan
    return {
        "accuracy": accuracy,
        "struggle_recall": recall,
        "struggle_precision": precision,
        "false_struggle_rate": false_rate,
        "missed_struggle_rate": missed_rate,
        "TP_s": tp,
        "FN_s": fn,
        "FP_s": fp,
        "TN_s": tn,
    }


def confusion_for(pred: np.ndarray, label: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float]:
    pred = np.asarray(pred, dtype=bool)
    label = np.asarray(label, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    tp = float(weights[label & pred].sum())
    fn = float(weights[label & ~pred].sum())
    fp = float(weights[~label & pred].sum())
    tn = float(weights[~label & ~pred].sum())
    return tp, fn, fp, tn


def apply_run_filter(base: np.ndarray, mouse_ids: np.ndarray, dts: np.ndarray, min_dur_s: float) -> np.ndarray:
    output = np.zeros(len(base), dtype=bool)
    for mouse_id in np.unique(mouse_ids):
        idx = np.flatnonzero(mouse_ids == mouse_id)
        values = base[idx]
        local_dts = dts[idx]
        pos = 0
        while pos < len(values):
            if not values[pos]:
                pos += 1
                continue
            end = pos + 1
            while end < len(values) and values[end]:
                end += 1
            if float(local_dts[pos:end].sum()) >= min_dur_s:
                output[idx[pos:end]] = True
            pos = end
    return output


def predict_candidate(frames: pd.DataFrame, candidate: Candidate) -> np.ndarray:
    current = frames["current_struggle"].to_numpy(dtype=bool)
    deform = (
        (frames["area_range_ratio"].to_numpy(dtype=float) >= candidate.area_thr)
        | (frames["aspect_range_ratio"].to_numpy(dtype=float) >= candidate.aspect_thr)
        | (frames["y_range_body"].to_numpy(dtype=float) >= candidate.y_thr)
    )
    tiny_motion = (
        (frames["decision_flow_p90_mm_s"].to_numpy(dtype=float) >= candidate.tiny_flow_thr)
        | (frames["speed_std_mm_s"].to_numpy(dtype=float) >= candidate.tiny_speed_thr)
        | (frames["flow_entropy"].to_numpy(dtype=float) >= candidate.tiny_entropy_thr)
    )
    strong_swing_like = (
        (frames["swing_score"].to_numpy(dtype=float) >= candidate.swing_score_thr)
        & (frames["acf_peak"].to_numpy(dtype=float) >= candidate.swing_acf_thr)
        & (frames["flow_entropy"].to_numpy(dtype=float) <= candidate.swing_entropy_thr)
        & (frames["flow_over_speed_std"].to_numpy(dtype=float) <= candidate.swing_fos_std_thr)
    )
    base = (~current) & deform & tiny_motion & (~strong_swing_like)
    extra = apply_run_filter(
        base,
        frames["mouse_id"].to_numpy(dtype=int),
        frames["dt"].to_numpy(dtype=float),
        candidate.min_dur_s,
    )
    return current | extra


def candidate_grid() -> list[Candidate]:
    candidates = []
    for min_dur_s in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        for area_thr in [0.07, 0.09, 0.11, 0.13]:
            for aspect_thr in [0.12, 0.15, 0.20, 0.25]:
                for y_thr in [0.025, 0.03, 0.04, 0.05]:
                    for tiny_flow_thr, tiny_speed_thr, tiny_entropy_thr in [
                        (8.0, 3.0, 0.04),
                        (12.0, 5.0, 0.05),
                        (15.0, 6.0, 0.06),
                    ]:
                        candidates.append(
                            Candidate(
                                min_dur_s=min_dur_s,
                                area_thr=area_thr,
                                aspect_thr=aspect_thr,
                                y_thr=y_thr,
                                tiny_flow_thr=tiny_flow_thr,
                                tiny_speed_thr=tiny_speed_thr,
                                tiny_entropy_thr=tiny_entropy_thr,
                            )
                        )
    return candidates


def score_metrics(row: pd.Series, current_false_rate: float) -> float:
    false_delta = max(0.0, float(row["false_struggle_rate"]) - current_false_rate)
    return (
        1.20 * float(row["accuracy"])
        + 0.85 * float(row["struggle_recall"])
        - 0.45 * false_delta
        + 0.15 * float(row["net_fixed_s"]) / 1000.0
    )


def per_group_confusions(
    frames: pd.DataFrame,
    pred: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    label = frames["human_struggle"].to_numpy(dtype=bool)
    weights = frames["dt"].to_numpy(dtype=float)
    rows = []
    for group_value in sorted(frames[group_col].unique()):
        mask = frames[group_col].to_numpy() == group_value
        tp, fn, fp, tn = confusion_for(pred[mask], label[mask], weights[mask])
        rows.append({group_col: group_value, **metrics_from_confusion(tp, fn, fp, tn)})
    return pd.DataFrame(rows)


def hardcase_counts(
    candidate_name: str,
    pred_by_mouse: dict[int, np.ndarray],
    full_logs: dict[int, pd.DataFrame],
    segments: pd.DataFrame,
) -> dict[str, int | str]:
    match = 0
    recovered = 0
    false_still = 0
    miss_total = 0
    false_total = 0
    for _, segment in segments.iterrows():
        mouse_id = int(segment["mouse_id"])
        logs = full_logs[mouse_id]
        pred = pred_by_mouse[mouse_id]
        times = logs["time"].to_numpy(dtype=float)
        weights = frame_dts(times)
        start = float(segment["start_time"])
        end = float(segment["end_time"])
        overlap = np.minimum(times + weights, end) - np.maximum(times, start)
        mask = overlap > 1e-9
        if not np.any(mask):
            continue
        frac = float(overlap[mask][pred[mask]].sum() / overlap[mask].sum())
        label = "Struggling" if frac >= 0.5 else "Immobile"
        if label == segment["blind_choice"]:
            match += 1
        if bool(segment["old_missed_struggle_confirmed"]):
            miss_total += 1
            if label == "Struggling":
                recovered += 1
        if bool(segment["old_false_struggle_confirmed"]):
            false_total += 1
            if label == "Struggling":
                false_still += 1
    return {
        "candidate": candidate_name,
        "all72_blind_match_n": match,
        "confirmed_missed_recovered_n": recovered,
        "confirmed_missed_total": miss_total,
        "confirmed_false_still_struggling_n": false_still,
        "confirmed_false_total": false_total,
    }


def full_log_predictions(
    candidate: Candidate | None,
    full_logs: dict[int, pd.DataFrame],
) -> dict[int, np.ndarray]:
    pred_by_mouse = {}
    for mouse_id, logs in full_logs.items():
        if candidate is None:
            pred_by_mouse[mouse_id] = bool_col(logs, "is_struggling")
            continue
        mini = pd.DataFrame(
            {
                "mouse_id": np.full(len(logs), mouse_id, dtype=int),
                "dt": frame_dts(logs["time"].to_numpy(dtype=float)),
                "current_struggle": bool_col(logs, "is_struggling"),
                "decision_flow_p90_mm_s": numeric_col(logs, "decision_flow_p90_mm_s"),
                "speed_std_mm_s": numeric_col(logs, "speed_std_mm_s"),
                "flow_entropy": numeric_col(logs, "flow_entropy"),
                "swing_score": numeric_col(logs, "swing_score"),
                "acf_peak": numeric_col(logs, "acf_peak"),
                "flow_over_speed_std": numeric_col(logs, "flow_over_speed_std"),
                "area_range_ratio": numeric_col(logs, "area_range_ratio"),
                "aspect_range_ratio": numeric_col(logs, "aspect_range_ratio"),
                "y_range_body": numeric_col(logs, "y_range_body"),
            }
        )
        pred_by_mouse[mouse_id] = predict_candidate(mini, candidate)
    return pred_by_mouse


def select_best(candidates: pd.DataFrame, current_false_rate: float, false_budget: float = 0.05) -> pd.Series:
    eligible = candidates[
        (candidates["false_struggle_rate"] <= current_false_rate + false_budget)
        & (candidates["accuracy_delta"] >= -0.002)
        & (candidates["net_fixed_s"] > 0)
    ].copy()
    if eligible.empty:
        eligible = candidates.copy()
    eligible["selection_score"] = eligible.apply(score_metrics, axis=1, current_false_rate=current_false_rate)
    return eligible.sort_values(
        ["selection_score", "accuracy", "struggle_recall", "net_fixed_s"],
        ascending=False,
    ).iloc[0]


def aggregate_confusion_rows(df: pd.DataFrame, group_col: str = "candidate_key") -> pd.DataFrame:
    summed = (
        df.groupby(group_col, as_index=False)[["TP_s", "FN_s", "FP_s", "TN_s"]]
        .sum()
        .reset_index(drop=True)
    )
    metric_rows = []
    for _, row in summed.iterrows():
        metrics = metrics_from_confusion(
            float(row["TP_s"]),
            float(row["FN_s"]),
            float(row["FP_s"]),
            float(row["TN_s"]),
        )
        metric_rows.append({group_col: row[group_col], **metrics})
    return pd.DataFrame(metric_rows)


def aggregate_current_groups(df: pd.DataFrame) -> dict[str, float]:
    totals = df[["TP_s", "FN_s", "FP_s", "TN_s"]].sum()
    return metrics_from_confusion(
        float(totals["TP_s"]),
        float(totals["FN_s"]),
        float(totals["FP_s"]),
        float(totals["TN_s"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline optimizer for TST rerun logs against BORIS labels.")
    parser.add_argument("--results-root", default=r"E:\MouseScope_gui\data\results")
    parser.add_argument("--old-root", default=r"E:\20241129")
    parser.add_argument("--out-dir", default=r"E:\MouseScope_gui\data\results\tst_rerun_comparison")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    old_root = Path(args.old_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, full_logs = load_frame_data(results_root, old_root / "csds_cvs_ctrl_tst.boris")
    label = frames["human_struggle"].to_numpy(dtype=bool)
    weights = frames["dt"].to_numpy(dtype=float)
    current_pred = frames["current_struggle"].to_numpy(dtype=bool)
    current_metrics = metrics_from_confusion(*confusion_for(current_pred, label, weights))

    current_correct = current_pred == label
    candidates = candidate_grid()
    rows = []
    per_mouse_rows = []
    per_video_rows = []
    predictions: dict[str, np.ndarray] = {}

    for idx, candidate in enumerate(candidates, start=1):
        pred = predict_candidate(frames, candidate)
        predictions[candidate.key] = pred
        tp, fn, fp, tn = confusion_for(pred, label, weights)
        metrics = metrics_from_confusion(tp, fn, fp, tn)
        corrected = (~current_correct) & (pred == label)
        damaged = current_correct & (pred != label)
        corrected_s = float(weights[corrected].sum())
        damaged_s = float(weights[damaged].sum())
        extra = pred & (~current_pred)
        row = {
            **candidate.as_dict(),
            **metrics,
            "accuracy_delta": metrics["accuracy"] - current_metrics["accuracy"],
            "recall_delta": metrics["struggle_recall"] - current_metrics["struggle_recall"],
            "false_rate_delta": metrics["false_struggle_rate"] - current_metrics["false_struggle_rate"],
            "corrected_wrong_to_right_s": corrected_s,
            "damaged_right_to_wrong_s": damaged_s,
            "net_fixed_s": corrected_s - damaged_s,
            "extra_struggle_s": float(weights[extra].sum()),
        }
        rows.append(row)

        mouse_groups = per_group_confusions(frames, pred, "mouse_id")
        mouse_groups["candidate_key"] = candidate.key
        per_mouse_rows.append(mouse_groups)
        video_groups = per_group_confusions(frames, pred, "video_ts")
        video_groups["candidate_key"] = candidate.key
        per_video_rows.append(video_groups)

        if idx % 100 == 0:
            print(f"evaluated {idx}/{len(candidates)} candidates", flush=True)

    candidate_df = pd.DataFrame(rows)
    current_false_rate = current_metrics["false_struggle_rate"]
    candidate_df["selection_score"] = candidate_df.apply(score_metrics, axis=1, current_false_rate=current_false_rate)
    candidate_df.sort_values("selection_score", ascending=False).to_csv(
        out_dir / "auto_optimizer_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    per_mouse = pd.concat(per_mouse_rows, ignore_index=True)
    per_video = pd.concat(per_video_rows, ignore_index=True)
    per_mouse.to_csv(out_dir / "auto_optimizer_per_mouse_metrics.csv", index=False, encoding="utf-8-sig")
    per_video.to_csv(out_dir / "auto_optimizer_per_video_metrics.csv", index=False, encoding="utf-8-sig")

    current_per_mouse = per_group_confusions(frames, current_pred, "mouse_id")
    current_per_video = per_group_confusions(frames, current_pred, "video_ts")
    current_per_mouse.to_csv(out_dir / "auto_optimizer_current_per_mouse.csv", index=False, encoding="utf-8-sig")
    current_per_video.to_csv(out_dir / "auto_optimizer_current_per_video.csv", index=False, encoding="utf-8-sig")

    selected = select_best(candidate_df, current_false_rate=current_false_rate)
    selected_candidate = next(c for c in candidates if c.key == selected["candidate_key"])

    # Nested-style validation: select on N-1 videos/mice, evaluate held-out group.
    # This uses cached per-group confusion matrices, so validation does not
    # repeatedly scan every frame for every fold.
    lovo_rows = []
    for video_ts in sorted(frames["video_ts"].unique()):
        train_group = per_video[per_video["video_ts"] != video_ts]
        current_train = aggregate_current_groups(current_per_video[current_per_video["video_ts"] != video_ts])
        fold_df = aggregate_confusion_rows(train_group)
        fold_df["accuracy_delta"] = fold_df["accuracy"] - current_train["accuracy"]
        fold_df["net_fixed_s"] = (
            (fold_df["TP_s"] - current_train["TP_s"]) + (fold_df["TN_s"] - current_train["TN_s"])
        )
        fold_df["false_struggle_rate_current_train"] = current_train["false_struggle_rate"]
        chosen = select_best(
            fold_df,
            current_false_rate=float(fold_df["false_struggle_rate_current_train"].iloc[0]),
        )
        test_row = per_video[
            (per_video["candidate_key"] == chosen["candidate_key"]) & (per_video["video_ts"] == video_ts)
        ].iloc[0]
        test_m = {key: float(test_row[key]) for key in current_metrics}
        cur_row = current_per_video[current_per_video["video_ts"] == video_ts].iloc[0]
        cur_test = {key: float(cur_row[key]) for key in current_metrics}
        lovo_rows.append(
            {
                "heldout_video_ts": video_ts,
                "selected_candidate_key": chosen["candidate_key"],
                **{f"test_{k}": v for k, v in test_m.items()},
                "test_accuracy_delta": test_m["accuracy"] - cur_test["accuracy"],
                "test_recall_delta": test_m["struggle_recall"] - cur_test["struggle_recall"],
                "test_false_rate_delta": test_m["false_struggle_rate"] - cur_test["false_struggle_rate"],
            }
        )
    lovo = pd.DataFrame(lovo_rows)
    lovo.to_csv(out_dir / "auto_optimizer_leave_one_video_out.csv", index=False, encoding="utf-8-sig")

    lomo_rows = []
    for mouse_id in sorted(frames["mouse_id"].unique()):
        train_group = per_mouse[per_mouse["mouse_id"] != mouse_id]
        current_train = aggregate_current_groups(current_per_mouse[current_per_mouse["mouse_id"] != mouse_id])
        fold_df = aggregate_confusion_rows(train_group)
        fold_df["accuracy_delta"] = fold_df["accuracy"] - current_train["accuracy"]
        fold_df["net_fixed_s"] = (
            (fold_df["TP_s"] - current_train["TP_s"]) + (fold_df["TN_s"] - current_train["TN_s"])
        )
        fold_df["false_struggle_rate_current_train"] = current_train["false_struggle_rate"]
        chosen = select_best(
            fold_df,
            current_false_rate=float(fold_df["false_struggle_rate_current_train"].iloc[0]),
        )
        test_row = per_mouse[
            (per_mouse["candidate_key"] == chosen["candidate_key"]) & (per_mouse["mouse_id"] == mouse_id)
        ].iloc[0]
        test_m = {key: float(test_row[key]) for key in current_metrics}
        cur_row = current_per_mouse[current_per_mouse["mouse_id"] == mouse_id].iloc[0]
        cur_test = {key: float(cur_row[key]) for key in current_metrics}
        lomo_rows.append(
            {
                "heldout_mouse_id": int(mouse_id),
                "selected_candidate_key": chosen["candidate_key"],
                **{f"test_{k}": v for k, v in test_m.items()},
                "test_accuracy_delta": test_m["accuracy"] - cur_test["accuracy"],
                "test_recall_delta": test_m["struggle_recall"] - cur_test["struggle_recall"],
                "test_false_rate_delta": test_m["false_struggle_rate"] - cur_test["false_struggle_rate"],
            }
        )
    lomo = pd.DataFrame(lomo_rows)
    lomo.to_csv(out_dir / "auto_optimizer_leave_one_mouse_out.csv", index=False, encoding="utf-8-sig")

    selected_pred = predictions[selected_candidate.key]
    selected_per_mouse = per_group_confusions(frames, selected_pred, "mouse_id")
    selected_per_mouse = selected_per_mouse.merge(
        current_per_mouse[["mouse_id", "accuracy", "struggle_recall", "false_struggle_rate"]],
        on="mouse_id",
        suffixes=("", "_current"),
    )
    selected_per_mouse["accuracy_delta"] = selected_per_mouse["accuracy"] - selected_per_mouse["accuracy_current"]
    selected_per_mouse["recall_delta"] = selected_per_mouse["struggle_recall"] - selected_per_mouse["struggle_recall_current"]
    selected_per_mouse["false_rate_delta"] = (
        selected_per_mouse["false_struggle_rate"] - selected_per_mouse["false_struggle_rate_current"]
    )
    selected_per_mouse.to_csv(out_dir / "auto_optimizer_selected_per_mouse.csv", index=False, encoding="utf-8-sig")

    segments_path = out_dir / "rerun_diff_segments_72.csv"
    hardcase_rows = []
    if segments_path.exists():
        segments = pd.read_csv(segments_path)
        current_full = full_log_predictions(None, full_logs)
        selected_full = full_log_predictions(selected_candidate, full_logs)
        hardcase_rows.append(hardcase_counts("current", current_full, full_logs, segments))
        hardcase_rows.append(hardcase_counts("selected", selected_full, full_logs, segments))
        pd.DataFrame(hardcase_rows).to_csv(
            out_dir / "auto_optimizer_selected_hardcase_counts.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_lines = [
        "TST offline optimizer summary",
        "",
        f"frames: {len(frames)}",
        f"mice: {frames['mouse_id'].nunique()}",
        f"videos: {frames['video_ts'].nunique()}",
        f"candidate_count: {len(candidates)}",
        "",
        "Current:",
        f"  accuracy={current_metrics['accuracy']:.4f}",
        f"  struggle_recall={current_metrics['struggle_recall']:.4f}",
        f"  false_struggle_rate={current_metrics['false_struggle_rate']:.4f}",
        f"  missed_struggle_rate={current_metrics['missed_struggle_rate']:.4f}",
        "",
        "Selected:",
        f"  {selected_candidate.key}",
        f"  accuracy={selected['accuracy']:.4f} ({selected['accuracy_delta']:+.4f})",
        f"  struggle_recall={selected['struggle_recall']:.4f} ({selected['recall_delta']:+.4f})",
        f"  false_struggle_rate={selected['false_struggle_rate']:.4f} ({selected['false_rate_delta']:+.4f})",
        f"  missed_struggle_rate={selected['missed_struggle_rate']:.4f}",
        f"  corrected_wrong_to_right_s={selected['corrected_wrong_to_right_s']:.3f}",
        f"  damaged_right_to_wrong_s={selected['damaged_right_to_wrong_s']:.3f}",
        f"  net_fixed_s={selected['net_fixed_s']:.3f}",
        "",
        "Leave-one-video-out test deltas:",
        f"  mean_accuracy_delta={lovo['test_accuracy_delta'].mean():+.4f}",
        f"  median_accuracy_delta={lovo['test_accuracy_delta'].median():+.4f}",
        f"  mean_recall_delta={lovo['test_recall_delta'].mean():+.4f}",
        f"  mean_false_rate_delta={lovo['test_false_rate_delta'].mean():+.4f}",
        f"  videos_with_accuracy_drop={(lovo['test_accuracy_delta'] < 0).sum()}/{len(lovo)}",
        "",
        "Leave-one-mouse-out test deltas:",
        f"  mean_accuracy_delta={lomo['test_accuracy_delta'].mean():+.4f}",
        f"  median_accuracy_delta={lomo['test_accuracy_delta'].median():+.4f}",
        f"  mean_recall_delta={lomo['test_recall_delta'].mean():+.4f}",
        f"  mean_false_rate_delta={lomo['test_false_rate_delta'].mean():+.4f}",
        f"  mice_with_accuracy_drop={(lomo['test_accuracy_delta'] < 0).sum()}/{len(lomo)}",
    ]
    if hardcase_rows:
        summary_lines.extend(["", "Hardcase counts:"])
        for row in hardcase_rows:
            summary_lines.append(
                "  "
                f"{row['candidate']}: blind_match={row['all72_blind_match_n']}/72, "
                f"confirmed_missed_recovered={row['confirmed_missed_recovered_n']}/{row['confirmed_missed_total']}, "
                f"confirmed_false_still_struggling={row['confirmed_false_still_struggling_n']}/{row['confirmed_false_total']}"
            )

    summary = "\n".join(summary_lines)
    (out_dir / "auto_optimizer_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
