from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "decision_flow_p90_mm_s",
    "speed_std_mm_s",
    "flow_entropy",
    "flow_over_speed_std",
    "struggle_score",
    "swing_score",
    "swing_margin",
    "acf_peak",
    "area_range_ratio",
    "aspect_range_ratio",
    "y_range_body",
    "resid_body_p90",
    "resid_boundary_p90",
    "resid_body_hot_frac",
    "resid_boundary_hot_frac",
    "mask_xor_frac",
    "area_delta_frac",
    "aspect_delta_frac",
    "solidity_delta",
    "translation_speed_mm_s",
    "decision_flow_p90_mm_s_r11_max",
    "speed_std_mm_s_r11_max",
    "area_range_ratio_r11_max",
    "aspect_range_ratio_r11_max",
    "y_range_body_r11_max",
    "resid_body_p90_r11_max",
    "resid_boundary_p90_r11_max",
    "resid_body_hot_frac_r11_max",
    "resid_boundary_hot_frac_r11_max",
    "mask_xor_frac_r11_max",
    "flow_entropy_r11_max",
    "decision_flow_p90_mm_s_r17_max",
    "speed_std_mm_s_r17_max",
    "area_range_ratio_r17_max",
    "aspect_range_ratio_r17_max",
    "y_range_body_r17_max",
    "resid_body_p90_r17_max",
    "resid_boundary_p90_r17_max",
    "resid_body_hot_frac_r17_max",
    "resid_boundary_hot_frac_r17_max",
    "mask_xor_frac_r17_max",
]

LOG_FEATURES = {
    "decision_flow_p90_mm_s",
    "speed_std_mm_s",
    "flow_over_speed_std",
    "resid_body_p90",
    "resid_boundary_p90",
    "translation_speed_mm_s",
    "decision_flow_p90_mm_s_r11_max",
    "speed_std_mm_s_r11_max",
    "resid_body_p90_r11_max",
    "resid_boundary_p90_r11_max",
    "decision_flow_p90_mm_s_r17_max",
    "speed_std_mm_s_r17_max",
    "resid_body_p90_r17_max",
    "resid_boundary_p90_r17_max",
}


def frame_dts(times: np.ndarray) -> np.ndarray:
    if len(times) <= 1:
        return np.array([0.04] * len(times), dtype=float)
    d = np.diff(times)
    positive = d[d > 0]
    fallback = float(np.median(positive)) if len(positive) else 0.04
    d = np.where(d > 0, d, fallback)
    return np.append(d, d[-1])


def metrics(pred: np.ndarray, label: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    label = np.asarray(label, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    tp = float(weights[label & pred].sum())
    fn = float(weights[label & ~pred].sum())
    fp = float(weights[~label & pred].sum())
    tn = float(weights[~label & ~pred].sum())
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


def design_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    cols = []
    for feature in features:
        values = pd.to_numeric(df.get(feature, 0.0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        arr = values.to_numpy(dtype=float)
        if feature in LOG_FEATURES:
            arr = np.log1p(np.maximum(arr, 0.0))
        cols.append(arr)

    for feature in ["rigid_body_like", "internal_flow_ok", "is_swinging", "detected_residual", "has_prev"]:
        if feature in df.columns:
            cols.append(df[feature].astype(bool).to_numpy(dtype=float))
    return np.column_stack(cols).astype(np.float64)


def weighted_mean(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (X * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-12)


def fit_linear_score(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    l2: float = 0.25,
) -> dict[str, np.ndarray]:
    mean = weighted_mean(X, weights)
    centered = X - mean
    std = np.sqrt(weighted_mean(centered * centered, weights))
    std = np.where(std > 1e-6, std, 1.0)
    Z = (X - mean) / std

    pos = y.astype(bool)
    neg = ~pos
    mu_pos = weighted_mean(Z[pos], weights[pos])
    mu_neg = weighted_mean(Z[neg], weights[neg])
    pos_centered = Z[pos] - mu_pos
    neg_centered = Z[neg] - mu_neg
    cov = (
        (pos_centered * weights[pos, None]).T @ pos_centered
        + (neg_centered * weights[neg, None]).T @ neg_centered
    ) / max(float(weights.sum()), 1e-12)
    cov = cov + np.eye(cov.shape[0]) * l2
    coef = np.linalg.solve(cov, mu_pos - mu_neg)
    return {"mean": mean, "std": std, "coef": coef}


def score_model(model: dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
    return ((X - model["mean"]) / model["std"]) @ model["coef"]


def choose_threshold(
    scores: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    current_false_rate: float,
    false_budget: float = 0.06,
) -> tuple[float, dict[str, float]]:
    quantiles = np.linspace(0.02, 0.98, 240)
    thresholds = np.unique(np.quantile(scores[np.isfinite(scores)], quantiles))
    best_thr = float(thresholds[0])
    best_metrics = None
    best_score = -1e18
    for threshold in thresholds:
        pred = scores >= threshold
        m = metrics(pred, y, weights)
        false_delta = max(0.0, m["false_struggle_rate"] - current_false_rate)
        if m["false_struggle_rate"] > current_false_rate + false_budget:
            objective = m["accuracy"] - 0.8 * false_delta
        else:
            objective = m["accuracy"] + 0.25 * m["struggle_recall"] - 0.2 * false_delta
        if objective > best_score:
            best_score = objective
            best_thr = float(threshold)
            best_metrics = m
    assert best_metrics is not None
    return best_thr, best_metrics


def hardcase_counts(df: pd.DataFrame, pred: np.ndarray, hardcase_path: Path) -> dict[str, int]:
    if not hardcase_path.exists():
        return {}
    segments = pd.read_csv(hardcase_path)
    work = df[["mouse_id", "time", "dt"]].copy()
    work["pred"] = pred
    match = recovered = false_still = miss_total = false_total = 0
    for _, segment in segments.iterrows():
        mouse_id = int(segment["mouse_id"])
        sub = work[work["mouse_id"] == mouse_id]
        times = sub["time"].to_numpy(dtype=float)
        dts = sub["dt"].to_numpy(dtype=float)
        start = float(segment["start_time"])
        end = float(segment["end_time"])
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


def run_fold(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    current_false_rate_train: float,
) -> tuple[np.ndarray, float, dict[str, float]]:
    X = design_matrix(df, FEATURES)
    y = df["human_struggle"].to_numpy(dtype=bool)
    weights = df["dt"].to_numpy(dtype=float)
    model = fit_linear_score(X[train_mask], y[train_mask], weights[train_mask])
    train_scores = score_model(model, X[train_mask])
    threshold, _ = choose_threshold(
        train_scores,
        y[train_mask],
        weights[train_mask],
        current_false_rate=current_false_rate_train,
    )
    test_scores = score_model(model, X[test_mask])
    pred = test_scores >= threshold
    return pred, threshold, metrics(pred, y[test_mask], weights[test_mask])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate standalone LDA-style TST score using existing and residual features.")
    parser.add_argument(
        "--dataset",
        default=r"E:\MouseScope_gui\data\results\tst_new_algorithm\standalone_algorithm_dataset.csv",
    )
    parser.add_argument("--out-dir", default=r"E:\MouseScope_gui\data\results\tst_new_algorithm")
    parser.add_argument("--results-root", default=r"E:\MouseScope_gui\data\results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset)
    y = df["human_struggle"].astype(bool).to_numpy()
    weights = df["dt"].to_numpy(dtype=float)
    current = df["current_struggle"].astype(bool).to_numpy()
    current_metrics = metrics(current, y, weights)

    X = design_matrix(df, FEATURES)
    model = fit_linear_score(X, y, weights)
    scores = score_model(model, X)
    threshold, all_train_metrics = choose_threshold(
        scores,
        y,
        weights,
        current_false_rate=current_metrics["false_struggle_rate"],
    )
    pred_all = scores >= threshold
    all_metrics = metrics(pred_all, y, weights)

    coef_rows = []
    feature_names = FEATURES + [name for name in ["rigid_body_like", "internal_flow_ok", "is_swinging", "detected_residual", "has_prev"] if name in df.columns]
    for name, coef in zip(feature_names, model["coef"]):
        coef_rows.append({"feature": name, "coef": float(coef), "abs_coef": abs(float(coef))})
    pd.DataFrame(coef_rows).sort_values("abs_coef", ascending=False).to_csv(
        out_dir / "linear_score_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Leave-one-video-out.
    video_rows = []
    for video_ts in sorted(df["video_ts"].unique()):
        test_mask = df["video_ts"].to_numpy() == video_ts
        train_mask = ~test_mask
        current_train = metrics(current[train_mask], y[train_mask], weights[train_mask])
        current_test = metrics(current[test_mask], y[test_mask], weights[test_mask])
        pred, fold_thr, fold_metrics = run_fold(df, train_mask, test_mask, current_train["false_struggle_rate"])
        video_rows.append(
            {
                "heldout_video_ts": video_ts,
                "threshold": fold_thr,
                **{f"test_{k}": v for k, v in fold_metrics.items()},
                "test_accuracy_delta": fold_metrics["accuracy"] - current_test["accuracy"],
                "test_recall_delta": fold_metrics["struggle_recall"] - current_test["struggle_recall"],
                "test_false_rate_delta": fold_metrics["false_struggle_rate"] - current_test["false_struggle_rate"],
            }
        )
    lovo = pd.DataFrame(video_rows)
    lovo.to_csv(out_dir / "linear_score_leave_one_video_out.csv", index=False, encoding="utf-8-sig")

    # Leave-one-mouse-out.
    mouse_rows = []
    for mouse_id in sorted(df["mouse_id"].unique()):
        test_mask = df["mouse_id"].to_numpy() == mouse_id
        train_mask = ~test_mask
        current_train = metrics(current[train_mask], y[train_mask], weights[train_mask])
        current_test = metrics(current[test_mask], y[test_mask], weights[test_mask])
        pred, fold_thr, fold_metrics = run_fold(df, train_mask, test_mask, current_train["false_struggle_rate"])
        mouse_rows.append(
            {
                "heldout_mouse_id": int(mouse_id),
                "threshold": fold_thr,
                **{f"test_{k}": v for k, v in fold_metrics.items()},
                "test_accuracy_delta": fold_metrics["accuracy"] - current_test["accuracy"],
                "test_recall_delta": fold_metrics["struggle_recall"] - current_test["struggle_recall"],
                "test_false_rate_delta": fold_metrics["false_struggle_rate"] - current_test["false_struggle_rate"],
            }
        )
    lomo = pd.DataFrame(mouse_rows)
    lomo.to_csv(out_dir / "linear_score_leave_one_mouse_out.csv", index=False, encoding="utf-8-sig")

    per_mouse_rows = []
    for mouse_id in sorted(df["mouse_id"].unique()):
        mask = df["mouse_id"].to_numpy() == mouse_id
        m = metrics(pred_all[mask], y[mask], weights[mask])
        cur = metrics(current[mask], y[mask], weights[mask])
        per_mouse_rows.append(
            {
                "mouse_id": int(mouse_id),
                **m,
                "accuracy_current": cur["accuracy"],
                "struggle_recall_current": cur["struggle_recall"],
                "false_struggle_rate_current": cur["false_struggle_rate"],
                "accuracy_delta": m["accuracy"] - cur["accuracy"],
                "recall_delta": m["struggle_recall"] - cur["struggle_recall"],
                "false_rate_delta": m["false_struggle_rate"] - cur["false_struggle_rate"],
            }
        )
    pd.DataFrame(per_mouse_rows).to_csv(out_dir / "linear_score_selected_per_mouse.csv", index=False, encoding="utf-8-sig")

    hardcase_path = Path(args.results_root) / "tst_rerun_comparison" / "rerun_diff_segments_72.csv"
    hard_rows = []
    if hardcase_path.exists():
        hard_rows.append({"candidate": "current", **hardcase_counts(df, current, hardcase_path)})
        hard_rows.append({"candidate": "linear_score", **hardcase_counts(df, pred_all, hardcase_path)})
        pd.DataFrame(hard_rows).to_csv(out_dir / "linear_score_hardcase_counts.csv", index=False, encoding="utf-8-sig")

    summary = "\n".join(
        [
            "TST standalone linear score summary",
            "",
            f"rows={len(df)} mice={df['mouse_id'].nunique()} videos={df['video_ts'].nunique()}",
            "",
            "Current module:",
            f"  accuracy={current_metrics['accuracy']:.4f}",
            f"  struggle_recall={current_metrics['struggle_recall']:.4f}",
            f"  false_struggle_rate={current_metrics['false_struggle_rate']:.4f}",
            f"  missed_struggle_rate={current_metrics['missed_struggle_rate']:.4f}",
            "",
            "Linear score fit on all 33 mice:",
            f"  threshold={threshold:.6f}",
            f"  accuracy={all_metrics['accuracy']:.4f} ({all_metrics['accuracy'] - current_metrics['accuracy']:+.4f})",
            f"  struggle_recall={all_metrics['struggle_recall']:.4f} ({all_metrics['struggle_recall'] - current_metrics['struggle_recall']:+.4f})",
            f"  false_struggle_rate={all_metrics['false_struggle_rate']:.4f} ({all_metrics['false_struggle_rate'] - current_metrics['false_struggle_rate']:+.4f})",
            f"  missed_struggle_rate={all_metrics['missed_struggle_rate']:.4f}",
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
            "",
            "Hardcase counts:",
            *[
                f"  {row['candidate']}: blind_match={row.get('all72_blind_match_n', 'NA')}/72, "
                f"confirmed_missed_recovered={row.get('confirmed_missed_recovered_n', 'NA')}/{row.get('confirmed_missed_total', 'NA')}, "
                f"confirmed_false_still_struggling={row.get('confirmed_false_still_struggling_n', 'NA')}/{row.get('confirmed_false_total', 'NA')}"
                for row in hard_rows
            ],
        ]
    )
    (out_dir / "linear_score_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
