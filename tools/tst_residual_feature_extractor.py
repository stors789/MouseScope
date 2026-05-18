from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tst_pts_timebase import frame_dts_from_times, load_or_extract_pts


TS_RE = re.compile(r"(\d{2}-\d{2}-\d{2}-\d{2}-\d{3})")
TS_SHORT_RE = re.compile(r"(\d{2}-\d{2}-\d{2}-\d{3})")
HU_WIN_START = 120.0
HU_WIN_END = 360.0
MAX_ANALYSIS_DIM = 960

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


def latest_session_state(results_root: Path) -> Path:
    candidates = sorted(
        (results_root / "_work").glob("*/session_state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No session_state.json found under {results_root / '_work'}")
    return candidates[0]


def timestamp_from_path(path: str) -> str:
    match = TS_SHORT_RE.search(path)
    if not match:
        raise ValueError(f"Could not parse timestamp from {path}")
    return match.group(1)


def load_batch_config(session_state_path: Path) -> list[dict]:
    with open(session_state_path, encoding="utf-8") as fh:
        state = json.load(fh)
    video_paths = state.get("video_paths") or []
    configs = (state.get("draft_payload") or {}).get("per_video_configs") or []
    if len(video_paths) != len(configs):
        raise ValueError(
            f"video_paths ({len(video_paths)}) and per_video_configs ({len(configs)}) length mismatch"
        )
    records = []
    for order, (video_path, cfg) in enumerate(zip(video_paths, configs), start=1):
        ts = timestamp_from_path(video_path)
        records.append(
            {
                "video_order": order,
                "video_ts": ts,
                "video_path": str(video_path),
                "config": cfg,
            }
        )
    return records


def scaled_points(points: list[dict], scale: float) -> np.ndarray:
    arr = np.array([[float(p["x"]) * scale, float(p["y"]) * scale] for p in points], dtype=np.float32)
    return np.round(arr).astype(np.int32)


def bbox_from_polygon(points: np.ndarray, width: int, height: int, pad: int = 2) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(points)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(width, x + w + pad)
    y2 = min(height, y + h + pad)
    return x1, y1, x2 - x1, y2 - y1


def largest_detection(mask: np.ndarray, min_area: float) -> dict | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area <= best_area:
            continue
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 1e-9 else 0.0
        x, y, w, h = cv2.boundingRect(contour)
        best = {
            "contour": contour,
            "area": area,
            "cx": float(moments["m10"] / moments["m00"]),
            "cy": float(moments["m01"] / moments["m00"]),
            "bbox": (int(x), int(y), int(w), int(h)),
            "solidity": float(solidity),
            "aspect": float(h / max(w, 1)),
        }
        best_area = area
    return best


def percentile90(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    idx = int(0.9 * (values.size - 1))
    return float(np.partition(values, idx)[idx])


def mask_from_contour(shape: tuple[int, int], contour_full: np.ndarray, x1: int, y1: int) -> np.ndarray:
    shifted = contour_full.copy()
    shifted[:, 0, 0] -= x1
    shifted[:, 0, 1] -= y1
    out = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(out, [shifted], -1, 255, thickness=-1)
    return out


def residual_features(
    prev_gray: np.ndarray | None,
    curr_gray: np.ndarray,
    prev_state: dict | None,
    curr_state: dict | None,
    pixel_to_mm_analysis: float,
    dt_s: float,
    ring_kernel: np.ndarray,
    erode_kernel: np.ndarray,
) -> dict:
    if prev_gray is None or prev_state is None or curr_state is None:
        return {
            "has_prev": False,
            "centroid_dx_px": 0.0,
            "centroid_dy_px": 0.0,
            "translation_speed_mm_s": 0.0,
            "resid_body_mean": 0.0,
            "resid_body_p90": 0.0,
            "resid_boundary_mean": 0.0,
            "resid_boundary_p90": 0.0,
            "resid_body_hot_frac": 0.0,
            "resid_boundary_hot_frac": 0.0,
            "mask_xor_frac": 0.0,
            "mask_iou": 0.0,
            "area_delta_frac": 0.0,
            "aspect_delta_frac": 0.0,
            "solidity_delta": 0.0,
        }

    dx = float(curr_state["cx_full"] - prev_state["cx_full"])
    dy = float(curr_state["cy_full"] - prev_state["cy_full"])
    dt_safe = max(float(dt_s), 1e-6)
    translation_speed = math.hypot(dx, dy) * pixel_to_mm_analysis / dt_safe

    pad = int(max(8, abs(dx) + abs(dy) + 10))
    x_candidates = [
        curr_state["bbox_full"][0],
        curr_state["bbox_full"][0] + curr_state["bbox_full"][2],
        prev_state["bbox_full"][0],
        prev_state["bbox_full"][0] + prev_state["bbox_full"][2],
    ]
    y_candidates = [
        curr_state["bbox_full"][1],
        curr_state["bbox_full"][1] + curr_state["bbox_full"][3],
        prev_state["bbox_full"][1],
        prev_state["bbox_full"][1] + prev_state["bbox_full"][3],
    ]
    h_img, w_img = curr_gray.shape[:2]
    x1 = max(0, int(math.floor(min(x_candidates) - pad)))
    y1 = max(0, int(math.floor(min(y_candidates) - pad)))
    x2 = min(w_img, int(math.ceil(max(x_candidates) + pad)))
    y2 = min(h_img, int(math.ceil(max(y_candidates) + pad)))
    if x2 <= x1 or y2 <= y1:
        return {}

    curr_crop = curr_gray[y1:y2, x1:x2]
    prev_crop = prev_gray[y1:y2, x1:x2]
    affine = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    prev_aligned = cv2.warpAffine(
        prev_crop,
        affine,
        (curr_crop.shape[1], curr_crop.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    curr_mask = mask_from_contour(curr_crop.shape, curr_state["contour_full"], x1, y1)
    prev_mask = mask_from_contour(curr_crop.shape, prev_state["contour_full"], x1, y1)
    prev_mask_aligned = cv2.warpAffine(
        prev_mask,
        affine,
        (curr_crop.shape[1], curr_crop.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    curr_bool = curr_mask > 0
    prev_bool = prev_mask_aligned > 0
    union = curr_bool | prev_bool
    if not np.any(union):
        return {}

    body_eroded = cv2.erode(union.astype(np.uint8) * 255, erode_kernel, iterations=1) > 0
    body_mask = body_eroded if np.any(body_eroded) else union
    dilated = cv2.dilate(union.astype(np.uint8) * 255, ring_kernel, iterations=1) > 0
    eroded = cv2.erode(union.astype(np.uint8) * 255, erode_kernel, iterations=1) > 0
    boundary = dilated & (~eroded)
    if not np.any(boundary):
        boundary = union

    diff = cv2.absdiff(curr_crop, prev_aligned)
    body_vals = diff[body_mask]
    boundary_vals = diff[boundary]
    xor = curr_bool ^ prev_bool
    inter = curr_bool & prev_bool
    union_count = int(union.sum())

    area_delta = abs(float(curr_state["area"]) - float(prev_state["area"])) / max(float(prev_state["area"]), 1.0)
    aspect_delta = abs(float(curr_state["aspect"]) - float(prev_state["aspect"])) / max(
        abs(float(prev_state["aspect"])),
        1e-6,
    )

    hot_thr = 12
    return {
        "has_prev": True,
        "centroid_dx_px": dx,
        "centroid_dy_px": dy,
        "translation_speed_mm_s": float(translation_speed),
        "resid_body_mean": float(body_vals.mean()) if body_vals.size else 0.0,
        "resid_body_p90": percentile90(body_vals),
        "resid_boundary_mean": float(boundary_vals.mean()) if boundary_vals.size else 0.0,
        "resid_boundary_p90": percentile90(boundary_vals),
        "resid_body_hot_frac": float((body_vals >= hot_thr).mean()) if body_vals.size else 0.0,
        "resid_boundary_hot_frac": float((boundary_vals >= hot_thr).mean()) if boundary_vals.size else 0.0,
        "mask_xor_frac": float(xor.sum() / max(union_count, 1)),
        "mask_iou": float(inter.sum() / max(union_count, 1)),
        "area_delta_frac": float(area_delta),
        "aspect_delta_frac": float(aspect_delta),
        "solidity_delta": float(abs(float(curr_state["solidity"]) - float(prev_state["solidity"]))),
    }


def prepare_roi_records(config: dict, scale: float, width: int, height: int, video_ts: str) -> list[dict]:
    rois = config.get("rois") or []
    records = []
    for roi_idx, roi in enumerate(rois, start=1):
        points = scaled_points(roi.get("points") or [], scale)
        if len(points) < 3:
            continue
        x, y, w, h = bbox_from_polygon(points, width, height)
        local_points = points.copy()
        local_points[:, 0] -= x
        local_points[:, 1] -= y
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [local_points], 255)
        mouse_ids = VIDEO_MOUSE_MAP.get(video_ts, [])
        records.append(
            {
                "roi_id": roi_idx,
                "roi_name": roi.get("name") or f"ROI {roi_idx}",
                "mouse_id": mouse_ids[roi_idx - 1] if roi_idx <= len(mouse_ids) else None,
                "polygon": points,
                "bbox": (x, y, w, h),
                "local_polygon": local_points,
                "roi_mask": roi_mask,
                "prev_state": None,
            }
        )
    return records


def process_video(record: dict, out_dir: Path, force: bool = False) -> Path:
    video_ts = record["video_ts"]
    output_path = out_dir / f"residual_features_{video_ts}.csv"
    if output_path.exists() and not force:
        return output_path

    video_path = Path(record["video_path"])
    config = record["config"]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    scale = min(1.0, MAX_ANALYSIS_DIM / max(width, height))
    analysis_width = max(1, int(round(width * scale)))
    analysis_height = max(1, int(round(height * scale)))
    pixel_to_mm = float(config.get("pixel_to_mm", 1.0) or 1.0)
    pixel_to_mm_analysis = pixel_to_mm / max(scale, 1e-6)
    pts_cache_dir = out_dir.parent / "tst_pts_timebase"
    pts_df = load_or_extract_pts(video_path, pts_cache_dir)
    pts_times = pts_df["pts_time"].to_numpy(dtype=np.float64)
    pts_dts = frame_dts_from_times(pts_times)

    rois = prepare_roi_records(config, scale, analysis_width, analysis_height, video_ts)
    if not rois:
        raise RuntimeError(f"No valid ROI in {video_path}")

    threshold = int(config.get("gray_threshold", 127) or 127)
    bg_type = str(config.get("bg_type", "light") or "light").lower()
    invert = bg_type == "dark"
    min_area = float(config.get("min_area", 100) or 100)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (int(config.get("erosion_kernel", 3) or 3), int(config.get("erosion_kernel", 3) or 3)),
    )
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (int(config.get("dilation_kernel", 5) or 5), int(config.get("dilation_kernel", 5) or 5)),
    )
    erosion_iter = int(config.get("erosion_iter", 2) or 0)
    dilation_iter = int(config.get("dilation_iter", 3) or 0)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    residual_erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    start_frame = int(np.searchsorted(pts_times, HU_WIN_START - 1e-9, side="left"))
    end_frame = int(np.searchsorted(pts_times, HU_WIN_END + 1e-9, side="right") - 1)
    start_frame = max(0, min(start_frame, max(total_frames - 1, 0), len(pts_times) - 1))
    end_frame = max(0, min(end_frame, max(total_frames - 1, 0), len(pts_times) - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    rows = []
    prev_gray = None
    t0 = time.time()
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_no > end_frame:
            break
        current_time = float(pts_times[frame_no]) if 0 <= frame_no < len(pts_times) else frame_no / max(fps, 1e-6)
        current_dt = float(pts_dts[frame_no]) if 0 <= frame_no < len(pts_dts) else 1.0 / max(fps, 1e-6)
        if scale != 1.0:
            frame = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if invert:
            _, fg_full = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        else:
            _, fg_full = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        for roi in rois:
            x, y, w, h = roi["bbox"]
            fg_crop = cv2.bitwise_and(fg_full[y : y + h, x : x + w], roi["roi_mask"])
            fg_crop = cv2.morphologyEx(fg_crop, cv2.MORPH_CLOSE, close_kernel)
            if erosion_iter > 0:
                fg_crop = cv2.erode(fg_crop, erode_kernel, iterations=erosion_iter)
            if dilation_iter > 0:
                fg_crop = cv2.dilate(fg_crop, dilate_kernel, iterations=dilation_iter)

            detection = largest_detection(fg_crop, min_area=min_area)
            current_state = None
            if detection is not None:
                contour_full = detection["contour"].copy()
                contour_full[:, 0, 0] += x
                contour_full[:, 0, 1] += y
                bx, by, bw, bh = detection["bbox"]
                current_state = {
                    **detection,
                    "contour_full": contour_full,
                    "bbox_full": (x + bx, y + by, bw, bh),
                    "cx_full": detection["cx"] + x,
                    "cy_full": detection["cy"] + y,
                }

            features = residual_features(
                prev_gray,
                gray,
                roi["prev_state"],
                current_state,
                pixel_to_mm_analysis=pixel_to_mm_analysis,
                dt_s=current_dt,
                ring_kernel=ring_kernel,
                erode_kernel=residual_erode_kernel,
            )
            row = {
                "video_ts": video_ts,
                "video_order": record["video_order"],
                "frame": frame_no,
                "time": round(current_time, 6),
                "pts_time": round(current_time, 6),
                "pts_dt": round(current_dt, 6),
                "roi_id": roi["roi_id"],
                "roi_name": roi["roi_name"],
                "mouse_id": roi["mouse_id"],
                "detected_residual": current_state is not None,
                "area_px": float(current_state["area"]) if current_state else 0.0,
                "aspect_ratio": float(current_state["aspect"]) if current_state else 0.0,
                "solidity": float(current_state["solidity"]) if current_state else 0.0,
                "analysis_scale": scale,
                "analysis_width": analysis_width,
                "analysis_height": analysis_height,
                "fps": fps,
                "pixel_to_mm_analysis": pixel_to_mm_analysis,
                **features,
            }
            rows.append(row)
            roi["prev_state"] = current_state

        prev_gray = gray
        processed += 1
        if processed % 1000 == 0:
            elapsed = time.time() - t0
            print(
                f"{video_ts}: processed {processed}/{max(end_frame - start_frame + 1, 1)} frames "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    cap.release()
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    elapsed = time.time() - t0
    print(f"{video_ts}: wrote {len(df)} rows to {output_path} in {elapsed:.1f}s", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract TST residual/boundary motion features from configured videos.")
    parser.add_argument("--results-root", default=r"E:\MouseScope_gui\data\results")
    parser.add_argument("--out-dir", default=r"E:\MouseScope_gui\data\results\tst_residual_features")
    parser.add_argument("--session-state", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--opencv-threads", type=int, default=0)
    args = parser.parse_args()

    if args.opencv_threads > 0:
        cv2.setNumThreads(args.opencv_threads)

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session_state = Path(args.session_state) if args.session_state else latest_session_state(results_root)
    records = load_batch_config(session_state)

    outputs = []
    for record in records:
        outputs.append(process_video(record, out_dir=out_dir, force=args.force))

    merged = []
    for path in outputs:
        merged.append(pd.read_csv(path))
    all_df = pd.concat(merged, ignore_index=True)
    all_path = out_dir / "tst_residual_features_120_360.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"merged residual features: {len(all_df)} rows -> {all_path}")


if __name__ == "__main__":
    main()
