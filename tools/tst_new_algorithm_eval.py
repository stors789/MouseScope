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

from tst_pts_timebase import frame_dts_from_times, load_or_extract_pts


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
class ResidualRule:
    window_frames: int
    resid_body_thr: float
    resid_boundary_thr: float
    hot_thr: float
    xor_thr: float
    area_thr: float
    aspect_thr: float
    y_thr: float
    flow_thr: float
    speed_thr: float
    passive_resid_thr: float
    passive_xor_thr: float

    @property
    def key(self) -> str:
        return (
            f"w={self.window_frames}|rb={self.resid_body_thr:g}|rbd={self.resid_boundary_thr:g}|"
            f"hot={self.hot_thr:g}|xor={self.xor_thr:g}|area={self.area_thr:g}|"
            f"aspect={self.aspect_thr:g}|y={self.y_thr:g}|flow={self.flow_thr:g}|"
            f"speed={self.speed_thr:g}|pr={self.passive_resid_thr:g}|px={self.passive_xor_thr:g}"
        )

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "candidate_key": self.key,
            "window_frames": self.window_frames,
            "resid_body_thr": self.resid_body_thr,
            "resid_boundary_thr": self.resid_boundary_thr,
            "hot_thr": self.hot_thr,
            "xor_thr": self.xor_thr,
            "area_thr": self.area_thr,
            "aspect_thr": self.aspect_thr,
            "y_thr": self.y_thr,
            "flow_thr": self.flow_thr,
            "speed_thr": self.speed_thr,
            "passive_resid_thr": self.passive_resid_thr,
            "passive_xor_thr": self.passive_xor_thr,
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

    intervals = {}
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


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.astype(str).str.lower().map({"true": True, "false": False}).fillna(False).astype(bool)
    return series.astype(bool)


def load_base_logs(results_root: Path, boris_file: Path) -> pd.DataFrame:
    human_intervals = load_human_intervals(boris_file)
    rows = []
    pts_cache_dir = results_root / "tst_pts_timebase"
    for video_ts, directory in result_dirs(results_root):
        log_path = list(directory.glob("*_tst_logs.csv"))[0]
        df = pd.read_csv(log_path)
        df["video_ts"] = video_ts
        video_path = Path(r"E:\20241129") / f"bandicam 2024-11-29 {video_ts}.mp4"
        pts_df = load_or_extract_pts(video_path, pts_cache_dir)
        pts_times = pts_df["pts_time"].to_numpy(dtype=np.float64)
        for roi_id, mouse_id in enumerate(VIDEO_MOUSE_MAP[video_ts], start=1):
            sub = df[df["roi_id"] == roi_id].copy().sort_values("time").reset_index(drop=True)
            frame_numbers = pd.to_numeric(sub["frame"], errors="coerce").fillna(-1).astype(int).to_numpy()
            valid = (frame_numbers >= 0) & (frame_numbers < len(pts_times))
            sub = sub.loc[valid].copy()
            frame_numbers = frame_numbers[valid]
            sub["time_raw_avgfps"] = sub["time"]
            sub["time"] = pts_times[frame_numbers]
            sub["pts_time"] = sub["time"]
            sub = sub[(sub["time"] >= HU_WIN_START - 1e-6) & (sub["time"] <= HU_WIN_END + 1e-6)].copy()
            if sub.empty:
                continue
            sub["mouse_id"] = mouse_id
            sub["dt"] = frame_dts_from_times(sub["time"].to_numpy(dtype=float))
            human = np.zeros(len(sub), dtype=bool)
            times = sub["time"].to_numpy(dtype=float)
            for start, end in human_intervals.get(mouse_id, []):
                human[(times >= start - FRAME_DT_FALLBACK / 2) & (times < end + FRAME_DT_FALLBACK / 2)] = True
            sub["human_struggle"] = human
            sub["current_struggle"] = bool_series(sub["is_struggling"])
            for col in ["is_swinging", "internal_flow_ok", "rigid_body_like"]:
                if col in sub.columns:
                    sub[col] = bool_series(sub[col])
            rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(default)
        .to_numpy(dtype=float)
    )


def load_dataset(results_root: Path, old_root: Path, residual_features: Path) -> pd.DataFrame:
    base = load_base_logs(results_root, old_root / "csds_cvs_ctrl_tst.boris")
    residual = pd.read_csv(residual_features)
    residual_cols = [
        "video_ts",
        "frame",
        "roi_id",
        "detected_residual",
        "area_px",
        "aspect_ratio",
        "solidity",
        "has_prev",
        "translation_speed_mm_s",
        "resid_body_mean",
        "resid_body_p90",
        "resid_boundary_mean",
        "resid_boundary_p90",
        "resid_body_hot_frac",
        "resid_boundary_hot_frac",
        "mask_xor_frac",
        "mask_iou",
        "area_delta_frac",
        "aspect_delta_frac",
        "solidity_delta",
    ]
    residual = residual[residual_cols].copy()
    merged = base.merge(residual, on=["video_ts", "frame", "roi_id"], how="left", validate="one_to_one")
    for col in residual_cols:
        if col in {"video_ts", "frame", "roi_id"}:
            continue
        if col in {"detected_residual", "has_prev"}:
            merged[col] = merged[col].fillna(False).astype(bool)
        else:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return merged.sort_values(["mouse_id", "frame"]).reset_index(drop=True)


def add_rolling_features(df: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    df = df.copy()
    source_cols = [
        "decision_flow_p90_mm_s",
        "speed_std_mm_s",
        "resid_body_p90",
        "resid_boundary_p90",
        "resid_body_hot_frac",
        "resid_boundary_hot_frac",
        "mask_xor_frac",
        "area_range_ratio",
        "aspect_range_ratio",
        "y_range_body",
        "flow_entropy",
    ]
    for col in source_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    grouped = df.groupby("mouse_id", sort=False)
    for window in sorted(set(windows)):
        for col in source_cols:
            roll = grouped[col].rolling(window, min_periods=1)
            df[f"{col}_r{window}_mean"] = roll.mean().reset_index(level=0, drop=True).to_numpy()
            df[f"{col}_r{window}_max"] = roll.max().reset_index(level=0, drop=True).to_numpy()
    return df


def weighted_auc(values: np.ndarray, label: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    label = np.asarray(label, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, label, weights = values[ok], label[ok], weights[ok]
    pos_w = float(weights[label].sum())
    neg_w = float(weights[~label].sum())
    if pos_w <= 0 or neg_w <= 0:
        return math.nan
    order = np.argsort(values)
    values, label, weights = values[order], label[order], weights[order]
    auc_num = 0.0
    neg_seen = 0.0
    idx = 0
    while idx < len(values):
        end = idx + 1
        while end < len(values) and values[end] == values[idx]:
            end += 1
        local_w = weights[idx:end]
        local_y = label[idx:end]
        wp = float(local_w[local_y].sum())
        wn = float(local_w[~local_y].sum())
        auc_num += wp * neg_seen + 0.5 * wp * wn
        neg_seen += wn
        idx = end
    return auc_num / (pos_w * neg_w)


def metrics_from_confusion(tp: float, fn: float, fp: float, tn: float) -> dict[str, float]:
    total = tp + fn + fp + tn
    return {
        "accuracy": (tp + tn) / total if total else math.nan,
        "struggle_recall": tp / (tp + fn) if tp + fn else math.nan,
        "struggle_precision": tp / (tp + fp) if tp + fp else math.nan,
        "false_struggle_rate": fp / (fp + tn) if fp + tn else math.nan,
        "missed_struggle_rate": fn / (tp + fn) if tp + fn else math.nan,
        "TP_s": tp,
        "FN_s": fn,
        "FP_s": fp,
        "TN_s": tn,
    }


def confusion(pred: np.ndarray, label: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float]:
    pred = np.asarray(pred, dtype=bool)
    label = np.asarray(label, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    tp = float(weights[label & pred].sum())
    fn = float(weights[label & ~pred].sum())
    fp = float(weights[~label & pred].sum())
    tn = float(weights[~label & ~pred].sum())
    return tp, fn, fp, tn


def candidate_grid() -> list[ResidualRule]:
    shape_sets = [
        (0.07, 0.15, 0.03),
        (0.09, 0.20, 0.04),
        (0.11, 0.25, 0.05),
    ]
    candidates = []
    for window in [7, 11, 17]:
        for resid_body in [8.0, 10.0, 12.0, 15.0]:
            for resid_boundary in [10.0, 12.0, 15.0, 18.0]:
                for hot_thr in [0.05, 0.08]:
                    for xor_thr in [0.06, 0.10]:
                        for area_thr, aspect_thr, y_thr in shape_sets:
                            for flow_thr, speed_thr in [(12.0, 5.0), (18.0, 8.0)]:
                                candidates.append(
                                    ResidualRule(
                                        window_frames=window,
                                        resid_body_thr=resid_body,
                                        resid_boundary_thr=resid_boundary,
                                        hot_thr=hot_thr,
                                        xor_thr=xor_thr,
                                        area_thr=area_thr,
                                        aspect_thr=aspect_thr,
                                        y_thr=y_thr,
                                        flow_thr=flow_thr,
                                        speed_thr=speed_thr,
                                        passive_resid_thr=8.0,
                                        passive_xor_thr=0.06,
                                    )
                                )
    return candidates


def predict_rule(df: pd.DataFrame, rule: ResidualRule) -> np.ndarray:
    w = rule.window_frames
    flow = df[f"decision_flow_p90_mm_s_r{w}_max"].to_numpy(dtype=float)
    speed = df[f"speed_std_mm_s_r{w}_max"].to_numpy(dtype=float)
    resid_body = df[f"resid_body_p90_r{w}_max"].to_numpy(dtype=float)
    resid_boundary = df[f"resid_boundary_p90_r{w}_max"].to_numpy(dtype=float)
    body_hot = df[f"resid_body_hot_frac_r{w}_max"].to_numpy(dtype=float)
    boundary_hot = df[f"resid_boundary_hot_frac_r{w}_max"].to_numpy(dtype=float)
    xor = df[f"mask_xor_frac_r{w}_max"].to_numpy(dtype=float)
    area = df[f"area_range_ratio_r{w}_max"].to_numpy(dtype=float)
    aspect = df[f"aspect_range_ratio_r{w}_max"].to_numpy(dtype=float)
    y_range = df[f"y_range_body_r{w}_max"].to_numpy(dtype=float)
    acf = numeric(df, "acf_peak")
    swing = numeric(df, "swing_score")
    rigid = df["rigid_body_like"].astype(bool).to_numpy() if "rigid_body_like" in df else np.zeros(len(df), dtype=bool)

    nonrigid_residual = (
        (resid_body >= rule.resid_body_thr)
        | (resid_boundary >= rule.resid_boundary_thr)
        | (body_hot >= rule.hot_thr)
        | (boundary_hot >= rule.hot_thr)
        | (xor >= rule.xor_thr)
    )
    shape_deformation = (area >= rule.area_thr) | (aspect >= rule.aspect_thr) | (y_range >= rule.y_thr)
    weak_motion = (flow >= rule.flow_thr) | (speed >= rule.speed_thr) | (resid_body >= rule.resid_body_thr)

    passive_swing_like = (
        rigid
        & ((swing >= 4.0) | (acf >= 0.55))
        & (resid_body <= rule.passive_resid_thr)
        & (xor <= rule.passive_xor_thr)
        & (~shape_deformation)
    )
    active = (nonrigid_residual & (shape_deformation | weak_motion)) | (shape_deformation & weak_motion)
    return active & (~passive_swing_like)


def per_group_confusions(df: pd.DataFrame, pred: np.ndarray, group_col: str) -> pd.DataFrame:
    label = df["human_struggle"].to_numpy(dtype=bool)
    weights = df["dt"].to_numpy(dtype=float)
    rows = []
    for value in sorted(df[group_col].unique()):
        mask = df[group_col].to_numpy() == value
        rows.append({group_col: value, **metrics_from_confusion(*confusion(pred[mask], label[mask], weights[mask]))})
    return pd.DataFrame(rows)


def aggregate_metrics(rows: pd.DataFrame) -> dict[str, float]:
    sums = rows[["TP_s", "FN_s", "FP_s", "TN_s"]].sum()
    return metrics_from_confusion(float(sums["TP_s"]), float(sums["FN_s"]), float(sums["FP_s"]), float(sums["TN_s"]))


def selection_score(row: pd.Series, current_false_rate: float) -> float:
    false_delta = max(0.0, float(row["false_struggle_rate"]) - current_false_rate)
    return (
        1.4 * float(row["accuracy"])
        + 0.8 * float(row["struggle_recall"])
        - 0.65 * false_delta
        + 0.1 * float(row["net_fixed_s"]) / 1000.0
    )


def select_best(rows: pd.DataFrame, current_false_rate: float, false_budget: float = 0.06) -> pd.Series:
    eligible = rows[
        (rows["false_struggle_rate"] <= current_false_rate + false_budget)
        & (rows["accuracy_delta"] >= -0.002)
        & (rows["net_fixed_s"] > 0)
    ].copy()
    if eligible.empty:
        eligible = rows.copy()
    eligible["selection_score"] = eligible.apply(selection_score, axis=1, current_false_rate=current_false_rate)
    return eligible.sort_values(["selection_score", "accuracy", "struggle_recall"], ascending=False).iloc[0]


def hardcase_counts(df: pd.DataFrame, pred: np.ndarray, hardcase_path: Path) -> dict[str, int]:
    segments = pd.read_csv(hardcase_path)
    work = df[["mouse_id", "time", "dt"]].copy()
    work["pred"] = pred
    match = recovered = false_still = miss_total = false_total = 0
    for _, segment in segments.iterrows():
        mouse_id = int(segment["mouse_id"])
        sub = work[work["mouse_id"] == mouse_id]
        start = float(segment["start_time"])
        end = float(segment["end_time"])
        times = sub["time"].to_numpy(dtype=float)
        dts = sub["dt"].to_numpy(dtype=float)
        overlap = np.minimum(times + dts, end) - np.maximum(times, start)
        mask = overlap > 1e-9
        if not np.any(mask):
            continue
        pred_sub = sub["pred"].to_numpy(dtype=bool)
        frac = float(overlap[mask][pred_sub[mask]].sum() / overlap[mask].sum())
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
        "all72_blind_match_n": match,
        "confirmed_missed_recovered_n": recovered,
        "confirmed_missed_total": miss_total,
        "confirmed_false_still_struggling_n": false_still,
        "confirmed_false_total": false_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a standalone residual-motion TST algorithm.")
    parser.add_argument("--results-root", default=r"E:\MouseScope_gui\data\results")
    parser.add_argument("--old-root", default=r"E:\20241129")
    parser.add_argument(
        "--residual-features",
        default=r"E:\MouseScope_gui\data\results\tst_residual_features\tst_residual_features_120_360.csv",
    )
    parser.add_argument("--out-dir", default=r"E:\MouseScope_gui\data\results\tst_new_algorithm")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(Path(args.results_root), Path(args.old_root), Path(args.residual_features))
    windows = [7, 11, 17, 23]
    df = add_rolling_features(df, windows)
    df.to_csv(out_dir / "standalone_algorithm_dataset.csv", index=False, encoding="utf-8-sig")

    label = df["human_struggle"].to_numpy(dtype=bool)
    weights = df["dt"].to_numpy(dtype=float)
    current_pred = df["current_struggle"].to_numpy(dtype=bool)
    current_metrics = metrics_from_confusion(*confusion(current_pred, label, weights))
    current_correct = current_pred == label

    feature_cols = [
        "resid_body_p90",
        "resid_boundary_p90",
        "resid_body_hot_frac",
        "resid_boundary_hot_frac",
        "mask_xor_frac",
        "area_delta_frac",
        "aspect_delta_frac",
        "translation_speed_mm_s",
        "decision_flow_p90_mm_s",
        "speed_std_mm_s",
        "area_range_ratio",
        "aspect_range_ratio",
        "y_range_body",
    ]
    auc_rows = []
    for col in feature_cols:
        auc = weighted_auc(numeric(df, col), label, weights)
        auc_rows.append({"feature": col, "auc": auc, "auc_abs": max(auc, 1 - auc)})
    pd.DataFrame(auc_rows).sort_values("auc_abs", ascending=False).to_csv(
        out_dir / "standalone_feature_auc.csv",
        index=False,
        encoding="utf-8-sig",
    )

    candidates = candidate_grid()
    rows = []
    per_video_rows = []
    per_mouse_rows = []
    predictions: dict[str, np.ndarray] = {}
    for idx, rule in enumerate(candidates, start=1):
        pred = predict_rule(df, rule)
        predictions[rule.key] = pred
        metrics = metrics_from_confusion(*confusion(pred, label, weights))
        corrected = (~current_correct) & (pred == label)
        damaged = current_correct & (pred != label)
        row = {
            **rule.as_dict(),
            **metrics,
            "accuracy_delta": metrics["accuracy"] - current_metrics["accuracy"],
            "recall_delta": metrics["struggle_recall"] - current_metrics["struggle_recall"],
            "false_rate_delta": metrics["false_struggle_rate"] - current_metrics["false_struggle_rate"],
            "corrected_wrong_to_right_s": float(weights[corrected].sum()),
            "damaged_right_to_wrong_s": float(weights[damaged].sum()),
            "net_fixed_s": float(weights[corrected].sum() - weights[damaged].sum()),
        }
        rows.append(row)
        pv = per_group_confusions(df, pred, "video_ts")
        pv["candidate_key"] = rule.key
        per_video_rows.append(pv)
        pm = per_group_confusions(df, pred, "mouse_id")
        pm["candidate_key"] = rule.key
        per_mouse_rows.append(pm)
        if idx % 1000 == 0:
            print(f"evaluated {idx}/{len(candidates)} residual rules", flush=True)

    candidates_df = pd.DataFrame(rows)
    candidates_df["selection_score"] = candidates_df.apply(
        selection_score,
        axis=1,
        current_false_rate=current_metrics["false_struggle_rate"],
    )
    candidates_df.sort_values("selection_score", ascending=False).to_csv(
        out_dir / "standalone_rule_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    per_video = pd.concat(per_video_rows, ignore_index=True)
    per_mouse = pd.concat(per_mouse_rows, ignore_index=True)
    per_video.to_csv(out_dir / "standalone_rule_per_video.csv", index=False, encoding="utf-8-sig")
    per_mouse.to_csv(out_dir / "standalone_rule_per_mouse.csv", index=False, encoding="utf-8-sig")

    current_per_video = per_group_confusions(df, current_pred, "video_ts")
    current_per_mouse = per_group_confusions(df, current_pred, "mouse_id")
    selected = select_best(candidates_df, current_false_rate=current_metrics["false_struggle_rate"])
    selected_pred = predictions[str(selected["candidate_key"])]

    lovo_rows = []
    for video_ts in sorted(df["video_ts"].unique()):
        train = per_video[per_video["video_ts"] != video_ts]
        train_current = aggregate_metrics(current_per_video[current_per_video["video_ts"] != video_ts])
        train_rows = []
        for key, group in train.groupby("candidate_key"):
            metrics = aggregate_metrics(group)
            train_rows.append(
                {
                    "candidate_key": key,
                    **metrics,
                    "accuracy_delta": metrics["accuracy"] - train_current["accuracy"],
                    "net_fixed_s": (
                        metrics["TP_s"]
                        + metrics["TN_s"]
                        - train_current["TP_s"]
                        - train_current["TN_s"]
                    ),
                }
            )
        chosen = select_best(pd.DataFrame(train_rows), current_false_rate=train_current["false_struggle_rate"])
        test_row = per_video[(per_video["video_ts"] == video_ts) & (per_video["candidate_key"] == chosen["candidate_key"])].iloc[0]
        cur_row = current_per_video[current_per_video["video_ts"] == video_ts].iloc[0]
        lovo_rows.append(
            {
                "heldout_video_ts": video_ts,
                "selected_candidate_key": chosen["candidate_key"],
                "test_accuracy_delta": float(test_row["accuracy"] - cur_row["accuracy"]),
                "test_recall_delta": float(test_row["struggle_recall"] - cur_row["struggle_recall"]),
                "test_false_rate_delta": float(test_row["false_struggle_rate"] - cur_row["false_struggle_rate"]),
                "test_accuracy": float(test_row["accuracy"]),
                "test_struggle_recall": float(test_row["struggle_recall"]),
                "test_false_struggle_rate": float(test_row["false_struggle_rate"]),
            }
        )
    lovo = pd.DataFrame(lovo_rows)
    lovo.to_csv(out_dir / "standalone_leave_one_video_out.csv", index=False, encoding="utf-8-sig")

    lomo_rows = []
    for mouse_id in sorted(df["mouse_id"].unique()):
        train = per_mouse[per_mouse["mouse_id"] != mouse_id]
        train_current = aggregate_metrics(current_per_mouse[current_per_mouse["mouse_id"] != mouse_id])
        train_rows = []
        for key, group in train.groupby("candidate_key"):
            metrics = aggregate_metrics(group)
            train_rows.append(
                {
                    "candidate_key": key,
                    **metrics,
                    "accuracy_delta": metrics["accuracy"] - train_current["accuracy"],
                    "net_fixed_s": (
                        metrics["TP_s"]
                        + metrics["TN_s"]
                        - train_current["TP_s"]
                        - train_current["TN_s"]
                    ),
                }
            )
        chosen = select_best(pd.DataFrame(train_rows), current_false_rate=train_current["false_struggle_rate"])
        test_row = per_mouse[(per_mouse["mouse_id"] == mouse_id) & (per_mouse["candidate_key"] == chosen["candidate_key"])].iloc[0]
        cur_row = current_per_mouse[current_per_mouse["mouse_id"] == mouse_id].iloc[0]
        lomo_rows.append(
            {
                "heldout_mouse_id": int(mouse_id),
                "selected_candidate_key": chosen["candidate_key"],
                "test_accuracy_delta": float(test_row["accuracy"] - cur_row["accuracy"]),
                "test_recall_delta": float(test_row["struggle_recall"] - cur_row["struggle_recall"]),
                "test_false_rate_delta": float(test_row["false_struggle_rate"] - cur_row["false_struggle_rate"]),
                "test_accuracy": float(test_row["accuracy"]),
                "test_struggle_recall": float(test_row["struggle_recall"]),
                "test_false_struggle_rate": float(test_row["false_struggle_rate"]),
            }
        )
    lomo = pd.DataFrame(lomo_rows)
    lomo.to_csv(out_dir / "standalone_leave_one_mouse_out.csv", index=False, encoding="utf-8-sig")

    selected_per_mouse = per_group_confusions(df, selected_pred, "mouse_id").merge(
        current_per_mouse[["mouse_id", "accuracy", "struggle_recall", "false_struggle_rate"]],
        on="mouse_id",
        suffixes=("", "_current"),
    )
    selected_per_mouse["accuracy_delta"] = selected_per_mouse["accuracy"] - selected_per_mouse["accuracy_current"]
    selected_per_mouse["recall_delta"] = (
        selected_per_mouse["struggle_recall"] - selected_per_mouse["struggle_recall_current"]
    )
    selected_per_mouse["false_rate_delta"] = (
        selected_per_mouse["false_struggle_rate"] - selected_per_mouse["false_struggle_rate_current"]
    )
    selected_per_mouse.to_csv(out_dir / "standalone_selected_per_mouse.csv", index=False, encoding="utf-8-sig")

    hardcase_path = Path(args.results_root) / "tst_rerun_comparison" / "rerun_diff_segments_72.csv"
    hard_rows = []
    if hardcase_path.exists():
        hard_rows.append({"candidate": "current", **hardcase_counts(df, current_pred, hardcase_path)})
        hard_rows.append({"candidate": "standalone_selected", **hardcase_counts(df, selected_pred, hardcase_path)})
        pd.DataFrame(hard_rows).to_csv(
            out_dir / "standalone_selected_hardcase_counts.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_lines = [
        "Standalone residual TST algorithm summary",
        "",
        f"rows={len(df)} mice={df['mouse_id'].nunique()} videos={df['video_ts'].nunique()}",
        f"candidate_count={len(candidates)}",
        "",
        "Current module:",
        f"  accuracy={current_metrics['accuracy']:.4f}",
        f"  struggle_recall={current_metrics['struggle_recall']:.4f}",
        f"  false_struggle_rate={current_metrics['false_struggle_rate']:.4f}",
        f"  missed_struggle_rate={current_metrics['missed_struggle_rate']:.4f}",
        "",
        "Standalone selected:",
        f"  {selected['candidate_key']}",
        f"  accuracy={selected['accuracy']:.4f} ({selected['accuracy_delta']:+.4f})",
        f"  struggle_recall={selected['struggle_recall']:.4f} ({selected['recall_delta']:+.4f})",
        f"  false_struggle_rate={selected['false_struggle_rate']:.4f} ({selected['false_rate_delta']:+.4f})",
        f"  missed_struggle_rate={selected['missed_struggle_rate']:.4f}",
        f"  corrected_wrong_to_right_s={selected['corrected_wrong_to_right_s']:.3f}",
        f"  damaged_right_to_wrong_s={selected['damaged_right_to_wrong_s']:.3f}",
        f"  net_fixed_s={selected['net_fixed_s']:.3f}",
        "",
        "Leave-one-video-out:",
        f"  mean_accuracy_delta={lovo['test_accuracy_delta'].mean():+.4f}",
        f"  median_accuracy_delta={lovo['test_accuracy_delta'].median():+.4f}",
        f"  mean_recall_delta={lovo['test_recall_delta'].mean():+.4f}",
        f"  mean_false_rate_delta={lovo['test_false_rate_delta'].mean():+.4f}",
        f"  videos_with_accuracy_drop={(lovo['test_accuracy_delta'] < 0).sum()}/{len(lovo)}",
        "",
        "Leave-one-mouse-out:",
        f"  mean_accuracy_delta={lomo['test_accuracy_delta'].mean():+.4f}",
        f"  median_accuracy_delta={lomo['test_accuracy_delta'].median():+.4f}",
        f"  mean_recall_delta={lomo['test_recall_delta'].mean():+.4f}",
        f"  mean_false_rate_delta={lomo['test_false_rate_delta'].mean():+.4f}",
        f"  mice_with_accuracy_drop={(lomo['test_accuracy_delta'] < 0).sum()}/{len(lomo)}",
    ]
    if hard_rows:
        summary_lines.append("")
        summary_lines.append("Hardcase counts:")
        for row in hard_rows:
            summary_lines.append(
                f"  {row['candidate']}: blind_match={row['all72_blind_match_n']}/72, "
                f"confirmed_missed_recovered={row['confirmed_missed_recovered_n']}/{row['confirmed_missed_total']}, "
                f"confirmed_false_still_struggling={row['confirmed_false_still_struggling_n']}/{row['confirmed_false_total']}"
            )
    summary = "\n".join(summary_lines)
    (out_dir / "standalone_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
