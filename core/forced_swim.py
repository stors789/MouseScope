#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forced Swim Test tracking and rendering helpers."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from file_utils import ascii_slug, write_image_unicode
from vision_tracker import OpticalFlowAnalyzer


STATE_ACTIVE = "active"
STATE_IMMOBILE = "immobile"
STATE_UNCERTAIN = "uncertain"
STATE_OUTSIDE = "outside"

STATE_COLORS = {
    STATE_ACTIVE: (44, 126, 245),
    STATE_IMMOBILE: (215, 128, 47),
    STATE_UNCERTAIN: (148, 163, 184),
    STATE_OUTSIDE: (203, 213, 225),
}
VIDEO_MOUSE_STATE_COLORS = {
    STATE_ACTIVE: (0, 0, 255),
    STATE_IMMOBILE: (0, 255, 0),
    STATE_UNCERTAIN: (128, 128, 128),
    STATE_OUTSIDE: (100, 100, 100),
}
VIDEO_ROI_COLOR = (255, 214, 10)
VIDEO_ROI_FILL_COLOR = (80, 80, 80)
VIDEO_DIAMETER_COLOR = (0, 255, 255)
VIDEO_WATERLINE_COLOR = (255, 255, 255)
VIDEO_PREVIEW_MOUSE_COLOR = (0, 255, 0)
FOREGROUND_CONTOUR_SHADOW = (0, 0, 0)


def sanitize_name(text: str, fallback: str = "fst") -> str:
    return ascii_slug(text, fallback=fallback, max_length=80)


def _format_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except Exception:
        return None


def _point_xy(point: Any) -> Tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def _coerce_points(points: Any, min_points: int = 3, label: str = "polygon") -> List[List[float]]:
    if not isinstance(points, (list, tuple)) or len(points) < min_points:
        raise ValueError(f"{label} requires at least {min_points} points.")
    out: List[List[float]] = []
    for point in points:
        x, y = _point_xy(point)
        out.append([x, y])
    return out


def _coerce_line(points: Any, label: str = "line") -> List[List[float]]:
    if not isinstance(points, (list, tuple)) or len(points) != 2:
        raise ValueError(f"{label} requires exactly 2 points.")
    return [[*_point_xy(points[0])], [*_point_xy(points[1])]]


def _polygon_mask(shape: Tuple[int, int], points: Sequence[Sequence[float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    polygon = np.asarray(points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0 or window <= 1:
        return values.astype(np.float32, copy=True)
    window = max(1, int(window))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(values.astype(np.float32), (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


class ForcedSwimTracker:
    """Track one mouse in one side-view FST cylinder."""

    def __init__(
        self,
        roi_points: Sequence[Sequence[float]],
        diameter_points: Sequence[Sequence[float]],
        cylinder_diameter_cm: float,
        fps: float = 30.0,
        roi_id: int = 1,
        roi_name: Optional[str] = None,
        time_config: Optional[Dict[str, Any]] = None,
        global_start_time_s: float = 0.0,
        detection_points: Optional[Sequence[Sequence[float]]] = None,
        waterline_points: Optional[Sequence[Sequence[float]]] = None,
        immobility_threshold: Optional[float] = None,
        immobility_high_threshold: Optional[float] = None,
        auto_threshold: bool = True,
        threshold_sensitivity: float = 0.0,
        smoothing_window_s: float = 0.5,
        min_immobility_duration_s: float = 1.0,
        min_active_duration_s: float = 0.3,
        gap_merge_s: float = 0.5,
    ) -> None:
        self.roi_id = int(roi_id)
        self.roi_name = roi_name or f"Cylinder {self.roi_id}"
        self.roi_points = _coerce_points(roi_points, min_points=3, label="FST ROI")
        self.detection_points = (
            _coerce_points(detection_points, min_points=3, label="FST detection mask")
            if detection_points is not None
            else self.roi_points
        )
        self.has_custom_detection_mask = detection_points is not None
        self.diameter_points = _coerce_line(diameter_points, label="Cylinder diameter line")
        self.waterline_points = _coerce_line(waterline_points, label="Waterline") if waterline_points else None
        self.cylinder_diameter_cm = max(0.1, float(cylinder_diameter_cm or 10.0))
        self.fps = max(1.0, float(fps or 30.0))
        self.time_config = self._normalize_time_config(time_config)
        self.global_start_time_s = max(0.0, float(global_start_time_s or 0.0))

        dx = self.diameter_points[0][0] - self.diameter_points[1][0]
        dy = self.diameter_points[0][1] - self.diameter_points[1][1]
        self.diameter_px = max(1.0, float(math.hypot(dx, dy)))
        self.cm_per_px = self.cylinder_diameter_cm / self.diameter_px
        self.pixel_to_mm = self.cm_per_px * 10.0

        self.auto_threshold = bool(auto_threshold)
        self.threshold_sensitivity = float(np.clip(threshold_sensitivity, -3.0, 3.0))
        self.immobility_threshold = float(immobility_threshold) if immobility_threshold is not None else None
        self.immobility_high_threshold = (
            float(immobility_high_threshold) if immobility_high_threshold is not None else None
        )
        self.smoothing_window_s = max(0.0, float(smoothing_window_s or 0.0))
        self.min_immobility_duration_s = max(0.0, float(min_immobility_duration_s or 0.0))
        self.min_active_duration_s = max(0.0, float(min_active_duration_s or 0.0))
        self.gap_merge_s = max(0.0, float(gap_merge_s or 0.0))

        self.roi_mask: Optional[np.ndarray] = None
        self.detection_mask: Optional[np.ndarray] = None
        self.flow_analyzer = OpticalFlowAnalyzer(fps=self.fps, pixel_to_mm=self.pixel_to_mm)
        self.reset_runtime(clear_threshold=False)

    @staticmethod
    def _normalize_time_config(time_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(time_config or {})
        mode = str(cfg.get("mode") or "global").strip().lower()
        if mode not in {"global", "relative", "precise"}:
            mode = "global"
        out: Dict[str, Any] = {"mode": mode, "start": None, "end": None}
        for key in ("start", "end"):
            value = cfg.get(key)
            if value in ("", None):
                continue
            try:
                out[key] = float(value)
            except Exception:
                out[key] = None
        if mode == "global":
            out["start"] = None
            out["end"] = None
        return out

    def reset_runtime(self, clear_threshold: bool = False) -> None:
        self.frame_records: List[Dict[str, Any]] = []
        self.motion_score_samples: List[float] = []
        self.prev_centroid: Optional[Tuple[float, float]] = None
        self.prev_time: Optional[float] = None
        self.prev_area: Optional[float] = None
        self.first_detected_time_abs: Optional[float] = None
        self.online_state = STATE_UNCERTAIN
        self.online_candidate = STATE_UNCERTAIN
        self.online_candidate_frames = 0
        self.flow_analyzer = OpticalFlowAnalyzer(fps=self.fps, pixel_to_mm=self.pixel_to_mm)
        self.live_total_frames = 0
        self.live_detected_frames = 0
        self.live_state_frames = {
            STATE_ACTIVE: 0,
            STATE_IMMOBILE: 0,
            STATE_UNCERTAIN: 0,
            STATE_OUTSIDE: 0,
        }
        self.live_motion_score_sum = 0.0
        self.live_motion_score_count = 0
        self.live_immobility_bouts = 0
        self.live_current_immobility_bout = 0
        self.live_longest_immobility_bout = 0
        self.live_prev_immobile = False
        self.live_latency_to_immobility_s: Optional[float] = None
        if clear_threshold:
            self.immobility_threshold = None
            self.immobility_high_threshold = None

    def _ensure_masks(self, shape: Tuple[int, int]) -> None:
        if self.roi_mask is not None and self.roi_mask.shape[:2] == shape:
            return
        self.roi_mask = _polygon_mask(shape, self.roi_points)
        self.detection_mask = _polygon_mask(shape, self.detection_points)

    def _stats_window_origin_abs(self) -> float:
        mode = self.time_config.get("mode", "global")
        start_cfg = self.time_config.get("start")
        if mode == "precise":
            return float(start_cfg if start_cfg is not None else self.global_start_time_s)
        if mode == "relative" and self.first_detected_time_abs is not None:
            return float(self.first_detected_time_abs + (start_cfg or 0.0))
        return self.global_start_time_s

    def _is_in_stats_window(self, current_time: float) -> bool:
        mode = self.time_config.get("mode", "global")
        start_cfg = self.time_config.get("start")
        end_cfg = self.time_config.get("end")
        if mode == "global":
            return True
        if mode == "precise":
            if start_cfg is not None and current_time < float(start_cfg):
                return False
            if end_cfg is not None and current_time > float(end_cfg):
                return False
            return True
        if mode == "relative":
            if self.first_detected_time_abs is None:
                return False
            elapsed = current_time - self.first_detected_time_abs
            if start_cfg is not None and elapsed < float(start_cfg):
                return False
            if end_cfg is not None and elapsed > float(end_cfg):
                return False
            return True
        return True

    def _select_detection(self, detections: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not detections:
            return None
        return max(detections, key=lambda item: float(item.get("area") or 0.0))

    def _online_update_state(self, score: Optional[float], detected: bool) -> str:
        if not detected or score is None:
            self.online_state = STATE_UNCERTAIN
            self.online_candidate = STATE_UNCERTAIN
            self.online_candidate_frames = 0
            return self.online_state
        low = self.low_threshold
        high = self.high_threshold
        if score <= low:
            candidate = STATE_IMMOBILE
        elif score >= high:
            candidate = STATE_ACTIVE
        else:
            candidate = self.online_state if self.online_state in {STATE_ACTIVE, STATE_IMMOBILE} else STATE_ACTIVE

        if candidate != self.online_candidate:
            self.online_candidate = candidate
            self.online_candidate_frames = 1
        else:
            self.online_candidate_frames += 1

        needed_s = self.min_immobility_duration_s if candidate == STATE_IMMOBILE else self.min_active_duration_s
        needed_frames = max(1, int(round(needed_s * self.fps)))
        if self.online_candidate_frames >= needed_frames:
            self.online_state = candidate
        return self.online_state

    @property
    def low_threshold(self) -> float:
        if self.immobility_threshold is None:
            return 0.08
        return max(0.0001, float(self.immobility_threshold))

    @property
    def high_threshold(self) -> float:
        if self.immobility_high_threshold is None:
            return self.low_threshold * 1.35
        return max(self.low_threshold + 0.0001, float(self.immobility_high_threshold))

    def update(
        self,
        frame: np.ndarray,
        current_time: float,
        gray: np.ndarray,
        bg_subtractor: Any,
        morph_processor: Any,
        detector: Any,
        record: bool = True,
        frame_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._ensure_masks(frame.shape[:2])
        assert self.detection_mask is not None

        raw_mask = bg_subtractor.apply(frame, gray=gray)
        fg_mask = morph_processor.process(raw_mask.copy(), inplace=True)
        fg_mask = cv2.bitwise_and(fg_mask, self.detection_mask)
        detections = detector.detect(fg_mask)
        detection = self._select_detection(detections)

        detected = detection is not None
        centroid_px: Optional[Tuple[float, float]] = None
        bbox = None
        contour = None
        area = 0.0
        speed_cm_s = 0.0
        flow_p90_cm_s = 0.0
        flow_mean_cm_s = 0.0
        shape_change_rate = 0.0
        motion_score: Optional[float] = None

        if detected and detection is not None:
            centroid_px = tuple(float(v) for v in detection["centroid"])
            bbox = detection.get("bbox")
            contour = detection.get("contour")
            area = float(detection.get("area") or 0.0)
            if self.first_detected_time_abs is None:
                self.first_detected_time_abs = float(current_time)

            if self.prev_centroid is not None and self.prev_time is not None:
                dt = max(1.0 / self.fps, float(current_time) - float(self.prev_time))
                dist_px = math.hypot(centroid_px[0] - self.prev_centroid[0], centroid_px[1] - self.prev_centroid[1])
                speed_cm_s = float(dist_px * self.cm_per_px / dt)

            if self.prev_area is not None and self.prev_area > 1.0:
                shape_change_rate = float(abs(area - self.prev_area) / self.prev_area * self.fps)

            flow = None
            if bbox is not None:
                x, y, w, h = [int(v) for v in bbox]
                flow = self.flow_analyzer.analyze(
                    frame,
                    fg_mask,
                    curr_gray=gray,
                    bbox=(x, y, x + w, y + h),
                )
            if flow:
                flow_p90_cm_s = float(flow.get("flow_p90_mm_s", 0.0) or 0.0) / 10.0
                flow_mean_cm_s = float(flow.get("flow_mean_mm_s", 0.0) or 0.0) / 10.0

            flow_diam_s = flow_p90_cm_s / max(self.cylinder_diameter_cm, 1e-6)
            speed_diam_s = speed_cm_s / max(self.cylinder_diameter_cm, 1e-6)
            shape_component = min(1.0, shape_change_rate / 3.0)
            motion_score = float(0.60 * flow_diam_s + 0.25 * speed_diam_s + 0.15 * shape_component)

            self.prev_centroid = centroid_px
            self.prev_time = float(current_time)
            self.prev_area = area
        else:
            self.flow_analyzer.prev_gray = gray

        in_stats_window = self._is_in_stats_window(float(current_time))
        if in_stats_window and detected and motion_score is not None:
            self.motion_score_samples.append(float(motion_score))

        state = self._online_update_state(motion_score, detected) if in_stats_window else STATE_OUTSIDE
        origin = self._stats_window_origin_abs()
        if record and in_stats_window:
            self.live_total_frames += 1
            if detected:
                self.live_detected_frames += 1
            counted_state = state if state in self.live_state_frames else STATE_UNCERTAIN
            self.live_state_frames[counted_state] += 1
            if motion_score is not None:
                self.live_motion_score_sum += float(motion_score)
                self.live_motion_score_count += 1

            is_immobile = counted_state == STATE_IMMOBILE
            if is_immobile:
                self.live_current_immobility_bout += 1
                if not self.live_prev_immobile:
                    self.live_immobility_bouts += 1
                    if self.live_latency_to_immobility_s is None:
                        self.live_latency_to_immobility_s = max(0.0, float(current_time) - origin)
            else:
                self.live_longest_immobility_bout = max(
                    self.live_longest_immobility_bout,
                    self.live_current_immobility_bout,
                )
                self.live_current_immobility_bout = 0
            self.live_prev_immobile = is_immobile

        record_row = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "frame": int(frame_index) if frame_index is not None else None,
            "time_s": round(float(current_time), 3),
            "window_time_s": round(float(current_time - origin), 3) if in_stats_window else None,
            "in_stats_window": bool(in_stats_window),
            "detected": bool(detected),
            "state": state,
            "motion_score": round(float(motion_score), 6) if motion_score is not None else None,
            "threshold_low": round(self.low_threshold, 6),
            "threshold_high": round(self.high_threshold, 6),
            "centroid_x": round(centroid_px[0], 3) if centroid_px else None,
            "centroid_y": round(centroid_px[1], 3) if centroid_px else None,
            "speed_cm_s": round(speed_cm_s, 4),
            "flow_p90_cm_s": round(flow_p90_cm_s, 4),
            "flow_mean_cm_s": round(flow_mean_cm_s, 4),
            "shape_change_rate": round(shape_change_rate, 4),
            "area_px": round(area, 3) if detected else None,
        }
        if bbox:
            x, y, w, h = bbox
            record_row.update({"bbox_x": int(x), "bbox_y": int(y), "bbox_w": int(w), "bbox_h": int(h)})
        else:
            record_row.update({"bbox_x": None, "bbox_y": None, "bbox_w": None, "bbox_h": None})
        if record:
            self.frame_records.append(record_row)

        return {
            **record_row,
            "contour": contour,
            "bbox": bbox,
            "centroid": centroid_px,
            "detection_count": len(detections),
            "diameter_points": self.diameter_points,
            "waterline_points": self.waterline_points,
        }

    def build_live_summary(self) -> Dict[str, Any]:
        total_frames = int(self.live_total_frames)
        detected_frames = int(self.live_detected_frames)
        immobile_frames = int(self.live_state_frames.get(STATE_IMMOBILE, 0))
        active_frames = int(self.live_state_frames.get(STATE_ACTIVE, 0))
        uncertain_frames = max(0, total_frames - immobile_frames - active_frames)

        total_time_s = total_frames / self.fps if self.fps > 0 else 0.0
        detected_time_s = detected_frames / self.fps if self.fps > 0 else 0.0
        immobile_time_s = immobile_frames / self.fps if self.fps > 0 else 0.0
        active_time_s = active_frames / self.fps if self.fps > 0 else 0.0
        uncertain_time_s = uncertain_frames / self.fps if self.fps > 0 else 0.0
        longest_bout_frames = max(self.live_longest_immobility_bout, self.live_current_immobility_bout)
        mean_motion_score = (
            self.live_motion_score_sum / self.live_motion_score_count
            if self.live_motion_score_count > 0
            else None
        )

        return {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "total_time_s": round(total_time_s, 3),
            "detected_frames": detected_frames,
            "total_frames": total_frames,
            "detected_rate_percent": round(detected_frames / max(total_frames, 1) * 100.0, 2),
            "detected_time_s": round(detected_time_s, 3),
            "immobility_time_s": round(immobile_time_s, 3),
            "active_time_s": round(active_time_s, 3),
            "uncertain_time_s": round(uncertain_time_s, 3),
            "immobility_percent": round(immobile_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "active_percent": round(active_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "uncertain_percent": round(uncertain_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "immobility_percent_detected": round(immobile_time_s / max(detected_time_s, 1e-6) * 100.0, 2),
            "immobility_bouts": int(self.live_immobility_bouts),
            "longest_immobility_bout_s": round(longest_bout_frames / self.fps, 3),
            "latency_to_immobility_s": (
                round(self.live_latency_to_immobility_s, 3)
                if self.live_latency_to_immobility_s is not None
                else None
            ),
            "mean_motion_score": round(float(mean_motion_score), 6) if mean_motion_score is not None else None,
            "immobility_threshold": round(self.low_threshold, 6),
            "active_threshold": round(self.high_threshold, 6),
        }

    def auto_configure_thresholds(self, fallback_threshold: float = 0.08) -> None:
        if not self.auto_threshold and self.immobility_threshold is not None:
            if self.immobility_high_threshold is None:
                self.immobility_high_threshold = self.immobility_threshold * 1.35
            return

        scores = np.asarray([v for v in self.motion_score_samples if np.isfinite(v)], dtype=np.float32)
        if scores.size < max(10, int(self.fps * 2)):
            base = float(self.immobility_threshold or fallback_threshold)
        else:
            p20 = float(np.percentile(scores, 20))
            p80 = float(np.percentile(scores, 80))
            if p80 <= p20 + 1e-6:
                base = max(0.005, p20 * 1.25)
            else:
                base = p20 + 0.25 * (p80 - p20)
            # Positive sensitivity makes the detector more likely to call frames immobile.
            base *= float(1.0 + 0.12 * self.threshold_sensitivity)
        self.immobility_threshold = float(max(0.001, base))
        self.immobility_high_threshold = float(max(self.immobility_threshold * 1.18, self.immobility_threshold + 0.002))

    def finalize_states(self) -> None:
        if not self.frame_records:
            return

        valid_indices = [
            idx
            for idx, row in enumerate(self.frame_records)
            if row.get("in_stats_window") and row.get("detected") and row.get("motion_score") is not None
        ]
        for row in self.frame_records:
            if not row.get("in_stats_window"):
                row["state"] = STATE_OUTSIDE
            elif not row.get("detected"):
                row["state"] = STATE_UNCERTAIN

        if not valid_indices:
            return

        scores = np.asarray([float(self.frame_records[idx]["motion_score"]) for idx in valid_indices], dtype=np.float32)
        smooth_window = max(1, int(round(self.smoothing_window_s * self.fps)))
        smoothed = _moving_average(scores, smooth_window)

        states: List[str] = []
        current = STATE_ACTIVE
        for score in smoothed:
            if score <= self.low_threshold:
                current = STATE_IMMOBILE
            elif score >= self.high_threshold:
                current = STATE_ACTIVE
            states.append(current)

        def runs(seq: Sequence[str]) -> List[Tuple[int, int, str]]:
            out: List[Tuple[int, int, str]] = []
            if not seq:
                return out
            start = 0
            for i in range(1, len(seq)):
                if seq[i] != seq[start]:
                    out.append((start, i, seq[start]))
                    start = i
            out.append((start, len(seq), seq[start]))
            return out

        gap_frames = int(round(self.gap_merge_s * self.fps))
        if gap_frames > 0:
            for start, end, state in runs(states):
                if state == STATE_ACTIVE and (end - start) <= gap_frames:
                    prev_state = states[start - 1] if start > 0 else None
                    next_state = states[end] if end < len(states) else None
                    if prev_state == STATE_IMMOBILE and next_state == STATE_IMMOBILE:
                        for i in range(start, end):
                            states[i] = STATE_IMMOBILE

        min_immobile_frames = int(round(self.min_immobility_duration_s * self.fps))
        if min_immobile_frames > 1:
            for start, end, state in runs(states):
                if state == STATE_IMMOBILE and (end - start) < min_immobile_frames:
                    for i in range(start, end):
                        states[i] = STATE_ACTIVE

        min_active_frames = int(round(self.min_active_duration_s * self.fps))
        if min_active_frames > 1:
            for start, end, state in runs(states):
                if state == STATE_ACTIVE and (end - start) < min_active_frames:
                    prev_state = states[start - 1] if start > 0 else None
                    next_state = states[end] if end < len(states) else None
                    if prev_state == STATE_IMMOBILE and next_state == STATE_IMMOBILE:
                        for i in range(start, end):
                            states[i] = STATE_IMMOBILE

        for idx, state, score in zip(valid_indices, states, smoothed):
            self.frame_records[idx]["state"] = state
            self.frame_records[idx]["smoothed_motion_score"] = round(float(score), 6)
            self.frame_records[idx]["threshold_low"] = round(self.low_threshold, 6)
            self.frame_records[idx]["threshold_high"] = round(self.high_threshold, 6)

    def build_summary(self) -> Dict[str, Any]:
        self.finalize_states()
        in_window = [row for row in self.frame_records if row.get("in_stats_window")]
        total_frames = len(in_window)
        detected_frames = sum(1 for row in in_window if row.get("detected"))
        immobile_frames = sum(1 for row in in_window if row.get("state") == STATE_IMMOBILE)
        active_frames = sum(1 for row in in_window if row.get("state") == STATE_ACTIVE)
        uncertain_frames = max(0, total_frames - immobile_frames - active_frames)

        total_time_s = total_frames / self.fps if self.fps > 0 else 0.0
        detected_time_s = detected_frames / self.fps if self.fps > 0 else 0.0
        immobile_time_s = immobile_frames / self.fps if self.fps > 0 else 0.0
        active_time_s = active_frames / self.fps if self.fps > 0 else 0.0
        uncertain_time_s = uncertain_frames / self.fps if self.fps > 0 else 0.0

        bouts = 0
        longest_bout_frames = 0
        latency_s = None
        current_bout = 0
        prev_immobile = False
        origin = self._stats_window_origin_abs()
        for row in in_window:
            is_immobile = row.get("state") == STATE_IMMOBILE
            if is_immobile:
                current_bout += 1
                if not prev_immobile:
                    bouts += 1
                    if latency_s is None:
                        latency_s = max(0.0, float(row.get("time_s") or 0.0) - origin)
            else:
                longest_bout_frames = max(longest_bout_frames, current_bout)
                current_bout = 0
            prev_immobile = is_immobile
        longest_bout_frames = max(longest_bout_frames, current_bout)

        scores = [float(row["motion_score"]) for row in in_window if row.get("motion_score") is not None]
        return {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "time_mode": self.time_config.get("mode", "global"),
            "time_start_s": self.time_config.get("start"),
            "time_end_s": self.time_config.get("end"),
            "analysis_window_start_s": round(origin, 3),
            "total_time_s": round(total_time_s, 3),
            "detected_frames": int(detected_frames),
            "total_frames": int(total_frames),
            "detected_rate_percent": round(detected_frames / max(total_frames, 1) * 100.0, 2),
            "detected_time_s": round(detected_time_s, 3),
            "immobility_time_s": round(immobile_time_s, 3),
            "active_time_s": round(active_time_s, 3),
            "uncertain_time_s": round(uncertain_time_s, 3),
            "immobility_percent": round(immobile_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "active_percent": round(active_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "uncertain_percent": round(uncertain_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "immobility_percent_detected": round(immobile_time_s / max(detected_time_s, 1e-6) * 100.0, 2),
            "immobility_bouts": int(bouts),
            "longest_immobility_bout_s": round(longest_bout_frames / self.fps, 3),
            "latency_to_immobility_s": round(latency_s, 3) if latency_s is not None else None,
            "mean_motion_score": round(float(np.mean(scores)), 6) if scores else None,
            "max_motion_score": round(float(np.max(scores)), 6) if scores else None,
            "immobility_threshold": round(self.low_threshold, 6),
            "active_threshold": round(self.high_threshold, 6),
            "auto_threshold": bool(self.auto_threshold),
            "threshold_sensitivity": round(float(self.threshold_sensitivity), 3),
            "cylinder_diameter_cm": round(self.cylinder_diameter_cm, 3),
            "diameter_px": round(self.diameter_px, 3),
            "cm_per_px": round(self.cm_per_px, 6),
            "has_custom_detection_mask": bool(self.has_custom_detection_mask),
        }

    def build_time_bins(self, bin_seconds: float = 60.0) -> List[Dict[str, Any]]:
        self.finalize_states()
        rows = [row for row in self.frame_records if row.get("in_stats_window")]
        if not rows:
            return []
        bin_seconds = max(1.0, float(bin_seconds or 60.0))
        buckets: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            t = float(row.get("window_time_s") or 0.0)
            idx = int(t // bin_seconds)
            bucket = buckets.setdefault(
                idx,
                {
                    "frames": 0,
                    "detected_frames": 0,
                    "active_frames": 0,
                    "immobile_frames": 0,
                    "uncertain_frames": 0,
                    "motion_scores": [],
                },
            )
            bucket["frames"] += 1
            if row.get("detected"):
                bucket["detected_frames"] += 1
            state = row.get("state")
            if state == STATE_ACTIVE:
                bucket["active_frames"] += 1
            elif state == STATE_IMMOBILE:
                bucket["immobile_frames"] += 1
            else:
                bucket["uncertain_frames"] += 1
            if row.get("motion_score") is not None:
                bucket["motion_scores"].append(float(row["motion_score"]))

        out: List[Dict[str, Any]] = []
        for idx in sorted(buckets):
            bucket = buckets[idx]
            duration_s = bucket["frames"] / self.fps
            immobile_s = bucket["immobile_frames"] / self.fps
            active_s = bucket["active_frames"] / self.fps
            uncertain_s = bucket["uncertain_frames"] / self.fps
            out.append(
                {
                    "roi_id": self.roi_id,
                    "roi_name": self.roi_name,
                    "bin_index": idx,
                    "start_s": round(idx * bin_seconds, 3),
                    "end_s": round((idx + 1) * bin_seconds, 3),
                    "duration_s": round(duration_s, 3),
                    "detected_rate_percent": round(bucket["detected_frames"] / max(bucket["frames"], 1) * 100.0, 2),
                    "immobility_time_s": round(immobile_s, 3),
                    "active_time_s": round(active_s, 3),
                    "uncertain_time_s": round(uncertain_s, 3),
                    "immobility_percent": round(immobile_s / max(duration_s, 1e-6) * 100.0, 2),
                    "mean_motion_score": round(float(np.mean(bucket["motion_scores"])), 6)
                    if bucket["motion_scores"]
                    else None,
                }
            )
        return out


def draw_forced_swim_results(
    frame: np.ndarray,
    tracker_results: Sequence[Tuple[ForcedSwimTracker, Dict[str, Any]]],
    *,
    preview_detection: bool = False,
) -> np.ndarray:
    canvas = frame.copy()
    overlay = canvas.copy()
    for tracker, result in tracker_results:
        pts = np.asarray(tracker.roi_points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], VIDEO_ROI_FILL_COLOR)

    cv2.addWeighted(overlay, 0.08, canvas, 0.92, 0, dst=canvas)

    for tracker, result in tracker_results:
        state = result.get("state") or STATE_UNCERTAIN
        detected = bool(result.get("detected"))
        if preview_detection and detected:
            mouse_color = VIDEO_PREVIEW_MOUSE_COLOR
        else:
            mouse_color = VIDEO_MOUSE_STATE_COLORS.get(state, VIDEO_MOUSE_STATE_COLORS[STATE_UNCERTAIN])
        draw_shadow = not preview_detection
        mouse_line_thickness = 3 if preview_detection else 2
        pts = np.asarray(tracker.roi_points, dtype=np.int32)
        cv2.polylines(canvas, [pts], True, VIDEO_ROI_COLOR, 2, cv2.LINE_AA)

        dpts = np.asarray(tracker.diameter_points, dtype=np.int32)
        cv2.line(canvas, tuple(dpts[0]), tuple(dpts[1]), VIDEO_DIAMETER_COLOR, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(dpts[0]), 4, VIDEO_DIAMETER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(dpts[1]), 4, VIDEO_DIAMETER_COLOR, -1, cv2.LINE_AA)

        if tracker.waterline_points:
            wpts = np.asarray(tracker.waterline_points, dtype=np.int32)
            cv2.line(canvas, tuple(wpts[0]), tuple(wpts[1]), VIDEO_WATERLINE_COLOR, 1, lineType=cv2.LINE_AA)

        contour = result.get("contour")
        if contour is not None:
            if draw_shadow:
                cv2.drawContours(canvas, [contour], -1, FOREGROUND_CONTOUR_SHADOW, 4, cv2.LINE_AA)
            cv2.drawContours(canvas, [contour], -1, mouse_color, mouse_line_thickness, cv2.LINE_AA)
        bbox = result.get("bbox")
        if bbox is not None:
            try:
                x, y, w, h = [int(v) for v in bbox]
                if draw_shadow:
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), FOREGROUND_CONTOUR_SHADOW, 3, cv2.LINE_AA)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), mouse_color, mouse_line_thickness, cv2.LINE_AA)
            except Exception:
                pass
        centroid = result.get("centroid")
        if centroid:
            if draw_shadow:
                cv2.circle(canvas, (int(centroid[0]), int(centroid[1])), 6, FOREGROUND_CONTOUR_SHADOW, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(centroid[0]), int(centroid[1])), 4, mouse_color, -1, cv2.LINE_AA)

        label_x = int(min(max(float(pts[:, 0].min()), 4), canvas.shape[1] - 220))
        label_y = int(max(float(pts[:, 1].min()) - 10, 22))
        score = result.get("motion_score")
        label = f"ROI {tracker.roi_id} {tracker.roi_name} | {state}"
        if score is not None:
            label += f" | score {float(score):.3f}"
        cv2.rectangle(canvas, (label_x - 4, label_y - 18), (label_x + 360, label_y + 6), (20, 20, 20), -1)
        cv2.rectangle(canvas, (label_x - 4, label_y - 18), (label_x + 360, label_y + 6), mouse_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, label[:48], (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, mouse_color, 1, cv2.LINE_AA)

    return canvas


def _draw_axes(image: np.ndarray, x0: int, y0: int, w: int, h: int, label: str) -> None:
    cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (226, 232, 240), 1)
    cv2.putText(image, label[:22], (16, y0 + h // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (51, 65, 85), 1, cv2.LINE_AA)


def render_forced_swim_ethogram(trackers: Sequence[ForcedSwimTracker], path: str) -> bool:
    if not trackers:
        return False
    for tracker in trackers:
        tracker.finalize_states()
    plot_w = 1120
    left = 180
    right = 50
    row_h = 52
    top = 58
    bottom = 52
    width = left + plot_w + right
    height = top + row_h * len(trackers) + bottom
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    image[:, :] = (245, 244, 237)
    cv2.putText(image, "Forced Swim Ethogram", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 19), 2, cv2.LINE_AA)

    max_time = 1.0
    records_by_tracker: List[List[Dict[str, Any]]] = []
    for tracker in trackers:
        rows = [row for row in tracker.frame_records if row.get("in_stats_window")]
        records_by_tracker.append(rows)
        if rows:
            max_time = max(max_time, float(rows[-1].get("window_time_s") or 0.0))

    for idx, (tracker, rows) in enumerate(zip(trackers, records_by_tracker)):
        y = top + idx * row_h
        _draw_axes(image, left, y, plot_w, 24, f"ROI {tracker.roi_id} {tracker.roi_name}")
        if not rows:
            continue
        for row in rows:
            t = float(row.get("window_time_s") or 0.0)
            x = left + int(np.clip(t / max_time, 0.0, 1.0) * plot_w)
            state = row.get("state") or STATE_UNCERTAIN
            color = STATE_COLORS.get(state, STATE_COLORS[STATE_UNCERTAIN])
            cv2.line(image, (x, y + 1), (x, y + 23), color, 1)
        cv2.putText(image, f"{max_time:.1f}s", (left + plot_w - 60, y + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 116, 139), 1, cv2.LINE_AA)

    legend_y = height - 28
    legend_x = left
    for state in (STATE_ACTIVE, STATE_IMMOBILE, STATE_UNCERTAIN):
        cv2.rectangle(image, (legend_x, legend_y - 12), (legend_x + 24, legend_y + 2), STATE_COLORS[state], -1)
        cv2.putText(image, state, (legend_x + 32, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (51, 65, 85), 1, cv2.LINE_AA)
        legend_x += 150
    return write_image_unicode(path, image)


def render_forced_swim_activity_curve(trackers: Sequence[ForcedSwimTracker], path: str) -> bool:
    if not trackers:
        return False
    for tracker in trackers:
        tracker.finalize_states()
    plot_w = 1120
    left = 180
    right = 50
    row_h = 140
    top = 58
    bottom = 52
    width = left + plot_w + right
    height = top + row_h * len(trackers) + bottom
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    image[:, :] = (245, 244, 237)
    cv2.putText(image, "Forced Swim Activity Curve", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 19), 2, cv2.LINE_AA)

    for idx, tracker in enumerate(trackers):
        rows = [row for row in tracker.frame_records if row.get("in_stats_window")]
        y = top + idx * row_h
        plot_h = 92
        _draw_axes(image, left, y, plot_w, plot_h, f"ROI {tracker.roi_id} {tracker.roi_name}")
        if not rows:
            continue
        max_time = max(1.0, float(rows[-1].get("window_time_s") or 0.0))
        scores = [float(row.get("smoothed_motion_score", row.get("motion_score") or 0.0) or 0.0) for row in rows]
        y_max = max(max(scores) if scores else 0.0, tracker.high_threshold * 1.25, 0.05)
        points: List[Tuple[int, int]] = []
        for row, score in zip(rows, scores):
            t = float(row.get("window_time_s") or 0.0)
            x = left + int(np.clip(t / max_time, 0.0, 1.0) * plot_w)
            yy = y + plot_h - int(np.clip(score / y_max, 0.0, 1.0) * plot_h)
            points.append((x, yy))
        if len(points) >= 2:
            cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (37, 99, 235), 2, cv2.LINE_AA)
        low_y = y + plot_h - int(np.clip(tracker.low_threshold / y_max, 0.0, 1.0) * plot_h)
        high_y = y + plot_h - int(np.clip(tracker.high_threshold / y_max, 0.0, 1.0) * plot_h)
        cv2.line(image, (left, low_y), (left + plot_w, low_y), (215, 128, 47), 1, cv2.LINE_AA)
        cv2.line(image, (left, high_y), (left + plot_w, high_y), (44, 126, 245), 1, cv2.LINE_AA)
        cv2.putText(image, f"low {tracker.low_threshold:.3f}", (left + 8, max(y + 12, low_y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 80, 30), 1, cv2.LINE_AA)
        cv2.putText(image, f"max {y_max:.3f}", (left + plot_w - 82, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 116, 139), 1, cv2.LINE_AA)

    return write_image_unicode(path, image)
