#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen legacy Tail Suspension Test tracker.

This module is intentionally independent from newer TST experiments.  The legacy/v1 path
must stay stable so experimental algorithms cannot change its thresholds,
timebase semantics, or swing state machine by accident.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import cv2
import numpy as np


class TSTLegacyOpticalFlowAnalyzer:
    """Legacy Farneback optical-flow feature extractor.

    The output scale is the original frame-normalized v1 scale:
    displacement-per-analyzed-frame multiplied by fps.  Do not switch this to
    per-PTS-delta units; v1 fixed thresholds were calibrated for this scale.
    """

    def __init__(self, fps: float = 30.0, pixel_to_mm: float = 1.0):
        self.fps = float(fps or 30.0)
        self.pixel_to_mm = float(pixel_to_mm or 1.0)
        self.prev_gray = None
        self.frame_count = 0

    def analyze(self, curr_frame, mask, curr_gray=None, bbox=None, mask_offset=None):
        self.frame_count += 1
        if curr_gray is None:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return None

        if bbox is not None:
            x1, y1, x2, y2 = bbox
        else:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                self.prev_gray = curr_gray
                return None
            x_min, y_min = mask.shape[1], mask.shape[0]
            x_max, y_max = 0, 0
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            margin = 10
            x1 = max(0, x_min - margin)
            y1 = max(0, y_min - margin)
            x2 = min(curr_gray.shape[1], x_max + margin)
            y2 = min(curr_gray.shape[0], y_max + margin)

        roi_prev = self.prev_gray[y1:y2, x1:x2]
        roi_curr = curr_gray[y1:y2, x1:x2]
        if roi_prev.size == 0 or roi_curr.size == 0:
            self.prev_gray = curr_gray
            return None

        scale_factor = 1.0
        orig_shape = roi_prev.shape
        if roi_prev.shape[0] > 40 or roi_prev.shape[1] > 40:
            scale_factor = 0.35
            roi_prev = cv2.resize(roi_prev, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            roi_curr = cv2.resize(roi_curr, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            if self.frame_count == 2:
                print(f"  [优化] 光流分辨率: {orig_shape} -> {roi_prev.shape} (缩放 {scale_factor}x)")

        flow = cv2.calcOpticalFlowFarneback(
            roi_prev,
            roi_curr,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        if mask_offset is not None:
            off_x, off_y = mask_offset
            local_y1 = max(0, y1 - off_y)
            local_y2 = min(mask.shape[0], y2 - off_y)
            local_x1 = max(0, x1 - off_x)
            local_x2 = min(mask.shape[1], x2 - off_x)
            raw_mask = mask[local_y1:local_y2, local_x1:local_x2]
        else:
            raw_mask = mask[y1:y2, x1:x2]

        if scale_factor != 1.0:
            flow = flow / scale_factor
            if raw_mask.size > 0:
                local_mask = cv2.resize(raw_mask, (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                local_mask = np.zeros((flow.shape[0], flow.shape[1]), dtype=np.uint8)
        else:
            if raw_mask.size > 0:
                if raw_mask.shape[:2] != flow.shape[:2]:
                    local_mask = cv2.resize(raw_mask, (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    local_mask = raw_mask
            else:
                local_mask = np.zeros_like(flow[:, :, 0], dtype=np.uint8)

        flow_mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        flow_ang = np.arctan2(flow[:, :, 1], flow[:, :, 0])
        mouse_flow_mag = flow_mag[local_mask == 255]
        mouse_flow_ang = flow_ang[local_mask == 255]
        u = flow[:, :, 0][local_mask == 255]
        v = flow[:, :, 1][local_mask == 255]
        if len(mouse_flow_mag) == 0:
            self.prev_gray = curr_gray
            return None

        mag_all = np.sqrt(u ** 2 + v ** 2)
        valid_mag = mag_all > 1e-6
        if int(np.sum(valid_mag)) > 0:
            u_valid = u[valid_mag]
            v_valid = v[valid_mag]
            mag_valid = mag_all[valid_mag]
            n_valid = int(mag_valid.shape[0])
            if n_valid >= 50:
                k = int(0.60 * (n_valid - 1))
                thr = float(np.partition(mag_valid, k)[k])
                strong = mag_valid >= max(1e-6, thr)
            else:
                strong = np.ones_like(mag_valid, dtype=bool)
            if int(np.sum(strong)) < 20:
                strong = np.ones_like(mag_valid, dtype=bool)
            u_s = u_valid[strong]
            v_s = v_valid[strong]
            mag_s = mag_valid[strong]
            u_norm = u_s / mag_s
            v_norm = v_s / mag_s
            r_val = float(np.sqrt(np.mean(u_norm) ** 2 + np.mean(v_norm) ** 2))
            flow_entropy = float(1.0 - r_val)
            mean_u = float(np.mean(u_s))
            mean_v = float(np.mean(v_s))
            mean_flow_mag = float(np.sqrt(mean_u ** 2 + mean_v ** 2))
            avg_individual_mag = float(np.mean(mag_s))
            coherence = float(mean_flow_mag / avg_individual_mag) if avg_individual_mag > 1e-6 else 0.0
        else:
            flow_entropy = 1.0
            coherence = 0.0

        flow_mean_px_frame = float(np.mean(mouse_flow_mag))
        flow_max_px_frame = float(np.max(mouse_flow_mag))
        n = len(mouse_flow_mag)
        if n > 50:
            flow_p50_px_frame = float(np.partition(mouse_flow_mag, n // 2)[n // 2])
            p90_idx = int(n * 0.9)
            flow_p90_px_frame = float(np.partition(mouse_flow_mag, p90_idx)[p90_idx])
        else:
            sorted_mag = np.sort(mouse_flow_mag)
            flow_p50_px_frame = float(sorted_mag[n // 2]) if n > 0 else 0.0
            flow_p90_px_frame = float(sorted_mag[int(n * 0.9)]) if n > 0 else 0.0
        flow_std_px_frame = float(np.std(mouse_flow_mag))

        frame_rate = max(float(self.fps), 1e-6)
        flow_mean_mm_s = flow_mean_px_frame * self.pixel_to_mm * frame_rate
        flow_p90_mm_s = flow_p90_px_frame * self.pixel_to_mm * frame_rate
        flow_max_mm_s = flow_max_px_frame * self.pixel_to_mm * frame_rate
        self.prev_gray = curr_gray

        return {
            "flow_mean_px": flow_mean_px_frame,
            "flow_p50_px": flow_p50_px_frame,
            "flow_p90_px": flow_p90_px_frame,
            "flow_max_px": flow_max_px_frame,
            "flow_std_px": flow_std_px_frame,
            "flow_mean_mm_s": flow_mean_mm_s,
            "flow_p90_mm_s": flow_p90_mm_s,
            "flow_max_mm_s": flow_max_mm_s,
            "flow_entropy": float(flow_entropy),
            "coherence": float(coherence),
            "dominant_angle": float(np.mean(mouse_flow_ang)) if len(mouse_flow_ang) > 0 else 0.0,
        }


class TSTLegacyPeriodicityAnalyzer:
    def __init__(self, fps: float, window_seconds: float = 2.0):
        self.fps = float(fps or 30.0)
        self.win_size = int(self.fps * window_seconds)
        self.x_history = deque(maxlen=self.win_size)
        self.y_history = deque(maxlen=self.win_size)
        self.SWING_FREQ_MIN = 0.8
        self.SWING_FREQ_MAX = 3.0
        self.MIN_AMPLITUDE_MM = 2.0
        self.ACF_THRESHOLD = 0.6

    def update(self, centroid_x, centroid_y=None):
        self.x_history.append(centroid_x)
        if centroid_y is not None:
            self.y_history.append(centroid_y)

    def motion_anisotropy(self):
        if len(self.x_history) < self.win_size or len(self.y_history) < self.win_size:
            return None, None, None
        x = np.asarray(self.x_history, dtype=np.float32)
        y = np.asarray(self.y_history, dtype=np.float32)
        sx = float(np.std(x))
        sy = float(np.std(y))
        return sx, sy, float(sy / (sx + 1e-6))

    def is_swinging_relative(self, scale_px: float, min_amplitude_ratio: float = 0.03):
        if len(self.x_history) < self.win_size:
            return False, 0.0, 0.0
        if scale_px is None or scale_px <= 1e-6:
            return False, 0.0, 0.0
        data = np.asarray(self.x_history, dtype=np.float32)
        amplitude_ratio = float(np.std(data) / float(scale_px))
        is_amplitude_sufficient = amplitude_ratio >= float(min_amplitude_ratio)
        y = data - float(np.mean(data))
        r = np.correlate(y, y, mode="full")
        r = r[r.size // 2:]
        if r[0] <= 0:
            return False, 0.0, 0.0
        r = r / r[0]
        min_lag = int(self.fps / self.SWING_FREQ_MAX)
        max_lag = int(self.fps / self.SWING_FREQ_MIN)
        if min_lag >= len(r) or max_lag >= len(r):
            return False, 0.0, 0.0
        search_window = r[min_lag:max_lag + 1]
        if len(search_window) == 0:
            return False, 0.0, 0.0
        peak_val = float(np.max(search_window))
        true_lag = int(min_lag + int(np.argmax(search_window)))
        est_freq = float(self.fps / true_lag) if true_lag > 0 else 0.0
        is_peak = True
        if 0 < true_lag < len(r) - 1 and (r[true_lag] < r[true_lag - 1] or r[true_lag] < r[true_lag + 1]):
            is_peak = False
        if is_peak and peak_val > self.ACF_THRESHOLD and is_amplitude_sufficient:
            return True, est_freq, peak_val
        return False, est_freq, peak_val


class TSTLegacyMouseTracker:
    def __init__(
        self,
        roi_polygon,
        roi_id,
        bg_subtractor,
        morphology_processor,
        detector,
        flow_analyzer,
        fps: float = 30.0,
        pixel_to_mm: float = 1.0,
        roi_name: str | None = None,
        time_config: dict[str, Any] | None = None,
        win_seconds: float = 0.3,
        flow_thr_hi: float = 50.0,
        flow_thr_lo: float = 35.0,
        speed_std_thr_hi: float = 40.0,
        speed_std_thr_lo: float = 20.0,
        min_active_seconds: float = 0.2,
        calm_seconds: float = 0.5,
        coherence_thr: float = 0.8,
        coherence_calm_seconds: float = 0.5,
        auto_behavior_threshold: bool = False,
        behavior_threshold_sensitivity: float = 0.0,
    ):
        self.roi_polygon = np.asarray(roi_polygon, dtype=np.int32)
        self.roi_id = roi_id
        self.roi_name = roi_name if roi_name else f"ROI {roi_id}"
        self.time_config = time_config
        if self.time_config:
            for key in ("start", "end"):
                if self.time_config.get(key) is not None:
                    try:
                        self.time_config[key] = float(self.time_config[key])
                    except Exception:
                        pass

        self.bg_subtractor = bg_subtractor
        self.morphology_processor = morphology_processor
        self.detector = detector
        self.flow_analyzer = flow_analyzer
        self.fps = float(fps or 30.0)
        self.pixel_to_mm = float(pixel_to_mm or 1.0)

        self.win = max(1, int(win_seconds * self.fps))
        self.min_frames_active = max(1, int(min_active_seconds * self.fps))
        self.calm_frames = max(1, int(calm_seconds * self.fps))
        self.coherence_calm_frames = max(1, int(coherence_calm_seconds * self.fps))

        self.flow_thr_hi = float(flow_thr_hi)
        self.flow_thr_lo = float(flow_thr_lo)
        self.speed_std_thr_hi = float(speed_std_thr_hi)
        self.speed_std_thr_lo = float(speed_std_thr_lo)
        self.coherence_thr = float(coherence_thr)
        self.auto_behavior_threshold = bool(auto_behavior_threshold)
        self.behavior_threshold_sensitivity = float(np.clip(behavior_threshold_sensitivity, -3.0, 3.0))
        self.behavior_threshold_samples: list[dict[str, Any]] = []
        self.collect_behavior_threshold_samples = self.auto_behavior_threshold

        self.roi_mask = None
        self.roi_bbox = None
        self.trajectory = deque(maxlen=100)
        self.last_centroid = None
        self.first_detected_time = None

        self.speeds = deque(maxlen=self.win)
        self.flows = deque(maxlen=self.win)
        self.coherences = deque(maxlen=self.coherence_calm_frames)
        accel_win = max(self.win, int(0.5 * self.fps))
        self._prev_speed = None
        self.accels = deque(maxlen=accel_win)

        self.is_struggling = False
        self.frames_active = 0
        self.calm_counter = 0
        self.moderate_activity_counter = 0
        self.moderate_entry_frames = max(1, int(0.45 * self.fps))
        self.frame_count = 0
        self.detected_count = 0

        self.immobility_stats = {
            "first_frame": 0,
            "total_frames": 0,
            "detected_frames": 0,
            "undetected_frames": 0,
            "immobility_frames": 0,
            "struggling_frames": 0,
            "immobility_episodes": 0,
            "struggling_episodes": 0,
            "latency_to_immobility": None,
            "last_state": None,
        }

        self.periodicity_analyzer = TSTLegacyPeriodicityAnalyzer(self.fps)
        baseline_len = max(5, int(2.0 * self.fps))
        self._bbox_h_history = deque(maxlen=baseline_len)
        self._aspect_ratio_history = deque(maxlen=baseline_len)
        self._baseline_ready_frames = max(5, int(1.0 * self.fps))

        self.is_in_confirmed_swing = False
        self.min_swing_frames = int(1.5 * self.fps)
        self.swing_rigidity_frames = max(5, int(1.0 * self.fps))
        self.swing_buffer = deque(maxlen=self.min_swing_frames + self.swing_rigidity_frames)
        self.swing_evidence = deque(maxlen=self.min_swing_frames)
        self.swing_confirm_hits = max(1, int(0.65 * self.min_swing_frames))
        self.swing_release_hits = max(0, int(0.35 * self.min_swing_frames))
        self.swing_aspect_range_ratio_max = 0.24
        self.swing_y_range_body_max = 0.025
        self.swing_area_range_ratio_max = 0.10
        self._swing_area_history = deque(maxlen=self.swing_rigidity_frames)
        self._swing_aspect_history = deque(maxlen=self.swing_rigidity_frames)
        self._swing_cx_history = deque(maxlen=self.swing_rigidity_frames)
        self._swing_cy_history = deque(maxlen=self.swing_rigidity_frames)
        self._swing_bbox_h_history = deque(maxlen=self.swing_rigidity_frames)
        self.swing_persistence = 0
        self.swing_duration_counter = 0

    def initialize_roi_mask(self, frame_shape):
        self.roi_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [self.roi_polygon], 255)
        x, y, w, h = cv2.boundingRect(self.roi_polygon)
        self.roi_bbox = (x, y, w, h)

    def update(self, frame, current_time=None, gray=None, dt_s=None):
        if not hasattr(self, "_perf_times"):
            self._perf_times = {"roi": [], "bg_sub": [], "morph": [], "detect": [], "flow": [], "stats": []}
        if self.roi_mask is None:
            self.initialize_roi_mask(frame.shape)
        self.frame_count += 1
        dt_value = self._safe_dt(dt_s)
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x, y, w, h = self.roi_bbox
        roi_frame = frame[y:y + h, x:x + w]
        roi_gray = gray[y:y + h, x:x + w]
        if not hasattr(self, "_cropped_roi_mask") or self._cropped_roi_mask is None:
            self._cropped_roi_mask = self.roi_mask[y:y + h, x:x + w]

        fg_mask_raw = self.bg_subtractor.apply(roi_frame, gray=roi_gray)
        fg_mask = cv2.bitwise_and(fg_mask_raw, fg_mask_raw, mask=self._cropped_roi_mask)
        fg_mask = self.morphology_processor.process(fg_mask, inplace=True)
        detections = self.detector.detect(fg_mask)
        if len(detections) == 0:
            self._update_immobility_stats(False, current_time, detected=False)
            self.last_centroid = None
            self.moderate_activity_counter = 0
            if self.flow_analyzer is not None:
                self.flow_analyzer.prev_gray = gray
            return {
                "roi_id": self.roi_id,
                "roi_name": self.roi_name,
                "roi_bbox": self.roi_bbox,
                "detected": False,
                "centroid": None,
                "contour": None,
                "flow": None,
                "is_struggling": False,
                "is_swinging": False,
                "state": "uncertain",
                "is_in_stats_window": self._is_in_stats_window(current_time),
                "dt_s": dt_value,
                "fg_mask": fg_mask,
                "roi_offset": (x, y),
            }

        detection = max(detections, key=lambda d: d.get("area", 0))
        local_centroid = detection["centroid"]
        global_centroid = (local_centroid[0] + x, local_centroid[1] + y)
        local_bbox = detection["bbox"]
        global_bbox = (local_bbox[0] + x, local_bbox[1] + y, local_bbox[2], local_bbox[3])
        local_contour = detection.get("contour")
        global_contour = local_contour + np.array([[x, y]]) if local_contour is not None else None
        detection["centroid"] = global_centroid
        detection["bbox"] = global_bbox
        detection["contour"] = global_contour

        if self.first_detected_time is None:
            self.first_detected_time = current_time if current_time is not None else (self.frame_count / self.fps)

        centroid = detection["centroid"]
        if self.last_centroid is None:
            speed = 0.0
        else:
            dist_px = np.linalg.norm(np.asarray(centroid) - np.asarray(self.last_centroid))
            speed = float(dist_px * self.pixel_to_mm * self.fps)
        self.last_centroid = centroid

        bx, by, bw, bh = detection["bbox"]
        margin = 10
        flow_bbox = (
            max(0, bx - margin),
            max(0, by - margin),
            min(frame.shape[1], bx + bw + margin),
            min(frame.shape[0], by + bh + margin),
        )
        flow_result = self.flow_analyzer.analyze(
            frame,
            fg_mask,
            curr_gray=gray,
            bbox=flow_bbox,
            mask_offset=(x, y),
        )
        if flow_result is None:
            flow_mm_s = 0.0
            coherence = 0.0
        else:
            flow_mm_s = float(flow_result["flow_p90_mm_s"])
            coherence = float(flow_result["coherence"])

        self.speeds.append(speed)
        self.flows.append(flow_mm_s)
        self.coherences.append(coherence)
        if self._prev_speed is None:
            accel = 0.0
        else:
            accel = (speed - self._prev_speed) * self.fps
        self._prev_speed = speed
        self.accels.append(float(accel))

        speed_arr = np.asarray(self.speeds)
        flow_arr = np.asarray(self.flows)
        coherence_arr = np.asarray(self.coherences)
        accel_arr = np.asarray(self.accels) if len(self.accels) > 0 else np.asarray([0.0])
        speed_std = float(np.std(speed_arr)) if len(speed_arr) >= 3 else 0.0
        flow_p90 = float(np.percentile(flow_arr, 90)) if len(flow_arr) >= 3 else flow_mm_s
        accel_std = float(np.std(accel_arr)) if len(accel_arr) >= 3 else 0.0

        self.frames_active += 1
        cx = detection["centroid"][0]
        cy = detection["centroid"][1]
        self.periodicity_analyzer.update(cx, cy)
        _, _, _, roi_h = self.roi_bbox
        self._bbox_h_history.append(float(roi_h))
        scale_px = float(np.median(self._bbox_h_history)) if len(self._bbox_h_history) > 0 else float(roi_h)
        _is_periodic, swing_freq, acf_peak = self.periodicity_analyzer.is_swinging_relative(
            scale_px=scale_px,
            min_amplitude_ratio=0.012,
        )

        amp_ratio = 0.0
        y_over_x = 0.0
        try:
            if len(self.periodicity_analyzer.x_history) > 3 and scale_px > 1e-6:
                amp_ratio = float(np.std(np.asarray(self.periodicity_analyzer.x_history, dtype=np.float32)) / scale_px)
            _, _, yx = self.periodicity_analyzer.motion_anisotropy()
            if yx is not None:
                y_over_x = float(yx)
        except Exception:
            amp_ratio = 0.0
            y_over_x = 0.0

        _, _, body_w, body_h = detection["bbox"]
        aspect_ratio = body_h / (body_w + 1e-6)
        area_value = float(max(detection.get("area", 0), 0))
        bbox_h_value = float(max(body_h, 1))
        self._swing_area_history.append(area_value)
        self._swing_aspect_history.append(float(aspect_ratio))
        self._swing_cx_history.append(float(cx))
        self._swing_cy_history.append(float(cy))
        self._swing_bbox_h_history.append(bbox_h_value)

        aspect_range_ratio = 0.0
        y_range_body = 0.0
        area_range_ratio = 0.0
        rigidity_window_ready = len(self._swing_aspect_history) >= self.swing_rigidity_frames
        if rigidity_window_ready:
            aspect_window = np.asarray(self._swing_aspect_history, dtype=np.float32)
            y_window = np.asarray(self._swing_cy_history, dtype=np.float32)
            area_window = np.asarray(self._swing_area_history, dtype=np.float32)
            h_window = np.asarray(self._swing_bbox_h_history, dtype=np.float32)
            aspect_p10, aspect_p90 = np.percentile(aspect_window, [10, 90])
            y_p10, y_p90 = np.percentile(y_window, [10, 90])
            area_p10, area_p90 = np.percentile(area_window, [10, 90])
            aspect_med = max(abs(float(np.median(aspect_window))), 1e-6)
            h_med = max(abs(float(np.median(h_window))), 1e-6)
            area_med = max(abs(float(np.median(area_window))), 1.0)
            aspect_range_ratio = float(max(0.0, (aspect_p90 - aspect_p10) / aspect_med))
            y_range_body = float(max(0.0, (y_p90 - y_p10) / h_med))
            area_range_ratio = float(max(0.0, (area_p90 - area_p10) / area_med))

        rigid_body_like = bool(
            (not rigidity_window_ready)
            or (
                aspect_range_ratio <= self.swing_aspect_range_ratio_max
                and y_range_body <= self.swing_y_range_body_max
                and area_range_ratio <= self.swing_area_range_ratio_max
            )
        )
        self._aspect_ratio_history.append(float(aspect_ratio))
        baseline_ar = float(np.median(self._aspect_ratio_history)) if len(self._aspect_ratio_history) >= self._baseline_ready_frames else None
        if baseline_ar is None:
            is_elongated = aspect_ratio > 1.8
        else:
            is_elongated = aspect_ratio > max(1.2, baseline_ar * 0.75)

        flow_entropy = float(flow_result.get("flow_entropy", 1.0)) if flow_result else 1.0
        eps = 1e-6
        flow_over_speed = float(flow_p90 / (abs(speed) + eps))
        flow_over_speed_std = float(flow_p90 / (abs(speed_std) + eps))
        coh_med = float(np.median(coherence_arr)) if len(coherence_arr) > 0 else float(coherence)
        ent = float(flow_entropy)
        swing_score = 0.0
        struggle_score = 0.0
        if acf_peak >= 0.80:
            swing_score += 2.5
        elif acf_peak >= 0.70:
            swing_score += 2.0
        elif acf_peak >= 0.60:
            swing_score += 1.0
        if 0.8 <= swing_freq <= 3.0:
            swing_score += 0.8
        if coh_med >= 0.90:
            swing_score += 1.2
        elif coh_med >= 0.80:
            swing_score += 0.8
        else:
            struggle_score += 0.6
        if ent <= 0.15:
            swing_score += 1.2
        elif ent <= 0.30:
            swing_score += 0.8
        elif ent >= 0.55:
            struggle_score += 1.5
        elif ent >= 0.40:
            struggle_score += 0.8
        if is_elongated:
            swing_score += 0.6
        else:
            struggle_score += 0.4
        if flow_over_speed >= 3.0:
            struggle_score += 1.2
        elif flow_over_speed >= 2.0:
            struggle_score += 0.6
        elif flow_over_speed <= 1.2:
            swing_score += 0.4
        if accel_std >= 150.0 and ent >= 0.35:
            struggle_score += 0.8

        freq_ok = 0.8 <= swing_freq <= 3.0
        acf_ok = acf_peak >= 0.45
        amp_ok = amp_ratio >= 0.012
        yx_ok = (y_over_x <= 0.55) if (y_over_x is not None) else True
        coh_ok = coh_med >= 0.88
        ent_ok = ent <= 0.30
        speed_floor_for_fos = 8.0
        fos_ok = True
        if abs(speed) >= speed_floor_for_fos:
            fos_ok = flow_over_speed <= 2.2
        strong_activity_like = False
        swing_margin = swing_score - struggle_score
        struggle_like = (
            (ent >= 0.55)
            or (coh_med <= 0.75)
            or (abs(speed) >= speed_floor_for_fos and flow_over_speed >= 3.0)
        )
        internal_flow_ok = flow_over_speed_std <= 3.2
        motion_swing_candidate = bool(freq_ok and acf_ok and amp_ok and yx_ok and coh_ok and ent_ok and fos_ok and (not struggle_like))
        is_swinging_condition = bool(motion_swing_candidate and rigid_body_like)
        sample_swing_like = bool(is_swinging_condition)
        if self.collect_behavior_threshold_samples and self._is_in_stats_window(current_time) and np.isfinite(flow_p90) and np.isfinite(speed_std):
            self.behavior_threshold_samples.append({
                "flow_p90": float(flow_p90),
                "speed_std": float(speed_std),
                "flow_over_speed": float(flow_over_speed),
                "flow_over_speed_std": float(flow_over_speed_std),
                "entropy": float(ent),
                "coherence": float(coh_med),
                "swing_like": bool(sample_swing_like),
            })

        retroactive_swing_frames = 0
        if not self.is_in_confirmed_swing:
            self.swing_buffer.append({
                "is_struggling": False,
                "swing_candidate": bool(is_swinging_condition),
                "motion_swing_candidate": bool(motion_swing_candidate),
                "in_stats_window": bool(self._is_in_stats_window(current_time)),
            })
        self.swing_evidence.append(1 if is_swinging_condition else 0)
        if not self.is_in_confirmed_swing:
            if len(self.swing_evidence) >= self.min_swing_frames:
                if int(sum(self.swing_evidence)) >= int(self.swing_confirm_hits) and rigid_body_like:
                    self.is_in_confirmed_swing = True
                    retroactive_swing_frames = len(self.swing_buffer)
                    for frame_info in self.swing_buffer:
                        if frame_info.get("in_stats_window", True) and frame_info.get("is_struggling") and self.immobility_stats["struggling_frames"] > 0:
                            self.immobility_stats["struggling_frames"] -= 1
                            self.immobility_stats["immobility_frames"] += 1
                    self.swing_buffer.clear()
                    self.swing_evidence.clear()
        else:
            if len(self.swing_evidence) >= self.min_swing_frames:
                if int(sum(self.swing_evidence)) <= int(self.swing_release_hits):
                    self.is_in_confirmed_swing = False
                    self.swing_evidence.clear()

        is_swinging_state = self.is_in_confirmed_swing
        if is_swinging_state:
            self.is_struggling = False
            self.calm_counter = 0
            self.moderate_activity_counter = 0
            self.swing_persistence = self.fps * 0.2
        else:
            if self.swing_persistence > 0:
                self.swing_persistence -= 1
                self.is_struggling = False
            else:
                was = self.is_struggling
                if not was:
                    enter_by_speed = speed_std > self.speed_std_thr_hi
                    enter_by_flow = flow_p90 > self.flow_thr_hi and flow_over_speed >= 2.5
                    moderate_flow_floor = max(self.flow_thr_lo, self.flow_thr_hi * 0.45)
                    swing_protected = bool(
                        motion_swing_candidate
                        or (swing_score >= 4.0 and ent <= 0.08 and acf_peak >= 0.55 and rigid_body_like)
                    )
                    moderate_by_flow = flow_p90 >= moderate_flow_floor and flow_over_speed >= 2.5 and not swing_protected
                    if moderate_by_flow:
                        self.moderate_activity_counter += 1
                    else:
                        self.moderate_activity_counter = 0
                    enter_by_sustained_flow = self.moderate_activity_counter >= self.moderate_entry_frames
                    if enter_by_speed or enter_by_flow or enter_by_sustained_flow:
                        self.is_struggling = True
                        self.calm_counter = 0
                        self.moderate_activity_counter = 0
                else:
                    self.moderate_activity_counter = 0
                    if flow_p90 < self.flow_thr_lo and speed_std < self.speed_std_thr_lo:
                        self.calm_counter += 1
                    else:
                        self.calm_counter = 0
                    if self.calm_counter >= self.calm_frames:
                        self.is_struggling = False
                        self.calm_counter = 0

        final_struggling = bool(self.is_struggling if self.frames_active >= self.min_frames_active else False)
        if len(self.swing_buffer) > 0:
            self.swing_buffer[-1]["is_struggling"] = final_struggling
        self._update_immobility_stats(final_struggling, current_time)
        self.trajectory.append(detection["centroid"])
        self.detected_count += 1
        effective_is_swinging = is_swinging_state or (self.swing_persistence > 0)

        return {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "roi_bbox": self.roi_bbox,
            "detected": True,
            "centroid": detection["centroid"],
            "contour": detection["contour"],
            "area": detection["area"],
            "bbox": detection["bbox"],
            "flow": flow_result,
            "is_struggling": final_struggling,
            "fg_mask": fg_mask,
            "roi_offset": (0, 0),
            "trajectory": list(self.trajectory),
            "speed": speed,
            "speed_std": speed_std,
            "flow_p90": flow_p90,
            "coherence": coherence,
            "is_swinging": effective_is_swinging,
            "swing_state": {
                "is_confirmed_swing": bool(self.is_in_confirmed_swing),
                "is_effective_swing": bool(effective_is_swinging),
                "is_candidate": bool(is_swinging_condition),
                "motion_candidate": bool(motion_swing_candidate),
                "is_newly_confirmed": bool(retroactive_swing_frames > 0),
                "retroactive_frame_count": int(retroactive_swing_frames),
                "struggle_like": bool(struggle_like),
                "strong_activity_like": bool(strong_activity_like),
                "internal_flow_ok": bool(internal_flow_ok),
                "rigid_body_like": bool(rigid_body_like),
                "rigidity_window_ready": bool(rigidity_window_ready),
                "aspect_range_ratio": float(aspect_range_ratio),
                "y_range_body": float(y_range_body),
                "area_range_ratio": float(area_range_ratio),
                "swing_score": float(swing_score),
                "struggle_score": float(struggle_score),
                "swing_margin": float(swing_margin),
            },
            "calm_counter": self.calm_counter,
            "aspect_ratio": aspect_ratio,
            "flow_entropy": float(flow_entropy),
            "swing_freq": swing_freq,
            "acf_peak": acf_peak,
            "swing_conf": int(sum(self.swing_evidence)) if not self.is_in_confirmed_swing else self.min_swing_frames,
            "amp_ratio": float(amp_ratio),
            "y_over_x": float(y_over_x),
            "flow_over_speed": float(flow_over_speed),
            "flow_over_speed_std": float(flow_over_speed_std),
            "accel_std": float(accel_std),
            "internal_flow_ok": bool(internal_flow_ok),
            "rigid_body_like": bool(rigid_body_like),
            "aspect_range_ratio": float(aspect_range_ratio),
            "y_range_body": float(y_range_body),
            "area_range_ratio": float(area_range_ratio),
            "is_in_stats_window": self._is_in_stats_window(current_time),
            "dt_s": dt_value,
        }

    def get_behavior_thresholds(self):
        return {
            "flow_thr_hi": float(self.flow_thr_hi),
            "flow_thr_lo": float(self.flow_thr_lo),
            "speed_std_thr_hi": float(self.speed_std_thr_hi),
            "speed_std_thr_lo": float(self.speed_std_thr_lo),
        }

    def apply_behavior_thresholds(self, thresholds):
        if not isinstance(thresholds, dict):
            return
        self.flow_thr_hi = float(thresholds.get("flow_thr_hi", self.flow_thr_hi))
        self.flow_thr_lo = float(thresholds.get("flow_thr_lo", self.flow_thr_lo))
        self.speed_std_thr_hi = float(thresholds.get("speed_std_thr_hi", self.speed_std_thr_hi))
        self.speed_std_thr_lo = float(thresholds.get("speed_std_thr_lo", self.speed_std_thr_lo))

    def auto_configure_behavior_thresholds(self):
        if not self.auto_behavior_threshold:
            return False
        samples = [
            sample for sample in self.behavior_threshold_samples
            if not sample.get("swing_like")
            and np.isfinite(sample.get("flow_p90", np.nan))
            and np.isfinite(sample.get("speed_std", np.nan))
        ]
        min_samples = max(30, int(round(self.fps * 3.0)))
        if len(samples) < min_samples:
            return False
        flow_values = np.asarray([sample["flow_p90"] for sample in samples], dtype=np.float32)
        speed_values = np.asarray([sample["speed_std"] for sample in samples], dtype=np.float32)
        if flow_values.size == 0 or speed_values.size == 0:
            return False
        fallback = self.get_behavior_thresholds()
        if float(np.max(flow_values)) < max(8.0, fallback["flow_thr_hi"] * 0.20) and float(np.max(speed_values)) < max(6.0, fallback["speed_std_thr_hi"] * 0.20):
            return False
        sensitivity_factor = float(np.clip(1.0 - 0.10 * self.behavior_threshold_sensitivity, 0.65, 1.35))

        def estimate_hi(values, fallback_hi, max_factor):
            p20, p50, p80, p90 = np.percentile(values, [20, 50, 80, 90])
            if p90 <= p20 + 1e-6:
                base = float(max(p50, p20 * 1.25))
            else:
                base = float(p80)
            hi = base * sensitivity_factor
            return float(np.clip(hi, fallback_hi * 0.60, fallback_hi * max_factor))

        def estimate_lo(values, hi):
            p35 = float(np.percentile(values, 35))
            return float(max(0.001, min(hi * 0.65, max(hi * 0.40, p35))))

        flow_hi = estimate_hi(flow_values, fallback["flow_thr_hi"], 1.60)
        speed_hi = estimate_hi(speed_values, fallback["speed_std_thr_hi"], 1.35)
        self.apply_behavior_thresholds({
            "flow_thr_hi": flow_hi,
            "flow_thr_lo": estimate_lo(flow_values, flow_hi),
            "speed_std_thr_hi": speed_hi,
            "speed_std_thr_lo": estimate_lo(speed_values, speed_hi),
        })
        self.collect_behavior_threshold_samples = False
        return True

    def _is_in_stats_window(self, current_time=None):
        if not self.time_config:
            return True
        if current_time is None:
            current_time = self.frame_count / self.fps
        mode = self.time_config.get("mode", "global")
        start_cfg = self.time_config.get("start")
        end_cfg = self.time_config.get("end")
        if start_cfg == "":
            start_cfg = None
        if end_cfg == "":
            end_cfg = None
        if mode == "global":
            return True
        if mode == "precise":
            if start_cfg is not None and current_time < start_cfg:
                return False
            if end_cfg is not None and current_time > end_cfg:
                return False
            return True
        if mode == "relative":
            if self.first_detected_time is None:
                return False
            elapsed = current_time - self.first_detected_time
            if start_cfg is not None and elapsed < start_cfg:
                return False
            if end_cfg is not None and elapsed > end_cfg:
                return False
            return True
        return True

    def _update_immobility_stats(self, is_struggling, current_time=None, detected=True):
        if not self._is_in_stats_window(current_time):
            return
        stats = self.immobility_stats
        if stats["first_frame"] == 0:
            stats["first_frame"] = self.frame_count
        stats["total_frames"] += 1
        if not detected:
            stats["undetected_frames"] += 1
            stats["last_state"] = None
            return
        stats["detected_frames"] += 1
        if is_struggling and self.frames_active >= self.min_frames_active:
            stats["struggling_frames"] += 1
        else:
            stats["immobility_frames"] += 1
        if stats["last_state"] is not None and stats["last_state"] != is_struggling:
            if is_struggling and self.frames_active >= self.min_frames_active:
                stats["struggling_episodes"] += 1
            elif not is_struggling:
                stats["immobility_episodes"] += 1
                if stats["latency_to_immobility"] is None:
                    stats["latency_to_immobility"] = self.frame_count - stats["first_frame"]
        stats["last_state"] = is_struggling

    def get_immobility_summary(self, fps):
        stats = self.immobility_stats
        if stats["total_frames"] == 0:
            return {
                "roi_id": self.roi_id,
                "roi_name": self.roi_name,
                "time_config": self.time_config,
                "auto_behavior_threshold": bool(self.auto_behavior_threshold),
                "behavior_threshold_sensitivity": round(float(self.behavior_threshold_sensitivity), 3),
                "flow_thr_hi": round(float(self.flow_thr_hi), 3),
                "flow_thr_lo": round(float(self.flow_thr_lo), 3),
                "speed_std_thr_hi": round(float(self.speed_std_thr_hi), 3),
                "speed_std_thr_lo": round(float(self.speed_std_thr_lo), 3),
                "total_time_seconds": 0.0,
                "immobility_time_seconds": 0.0,
                "struggling_time_seconds": 0.0,
                "immobility_percent": 0.0,
                "struggling_percent": 0.0,
                "immobility_episodes": 0,
                "struggling_episodes": 0,
                "immobility_frequency_per_min": 0.0,
                "struggling_frequency_per_min": 0.0,
                "latency_to_immobility_seconds": None,
                "detected_frames": 0,
                "undetected_frames": 0,
                "detected_rate_percent": 0.0,
                "uncertain_time_seconds": 0.0,
                "uncertain_percent": 0.0,
                "total_frames": 0,
            }

        total_time = stats["total_frames"] / fps
        detected_time = stats.get("detected_frames", 0) / fps
        uncertain_time = stats.get("undetected_frames", 0) / fps
        immobile_time = stats["immobility_frames"] / fps
        struggling_time = stats["struggling_frames"] / fps
        immobile_percent = (stats["immobility_frames"] / stats["total_frames"]) * 100
        struggling_percent = (stats["struggling_frames"] / stats["total_frames"]) * 100
        detected_rate = (stats.get("detected_frames", 0) / stats["total_frames"]) * 100
        uncertain_percent = (stats.get("undetected_frames", 0) / stats["total_frames"]) * 100
        duration_minutes = total_time / 60.0
        immobile_freq = stats["immobility_episodes"] / max(duration_minutes, 1 / 60.0)
        struggling_freq = stats["struggling_episodes"] / max(duration_minutes, 1 / 60.0)
        latency_seconds = None
        if stats["latency_to_immobility"] is not None:
            latency_seconds = stats["latency_to_immobility"] / fps
        return {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "time_config": self.time_config,
            "auto_behavior_threshold": bool(self.auto_behavior_threshold),
            "behavior_threshold_sensitivity": round(float(self.behavior_threshold_sensitivity), 3),
            "flow_thr_hi": round(float(self.flow_thr_hi), 3),
            "flow_thr_lo": round(float(self.flow_thr_lo), 3),
            "speed_std_thr_hi": round(float(self.speed_std_thr_hi), 3),
            "speed_std_thr_lo": round(float(self.speed_std_thr_lo), 3),
            "total_time_seconds": round(total_time, 2),
            "detected_time_seconds": round(detected_time, 2),
            "uncertain_time_seconds": round(uncertain_time, 2),
            "immobility_time_seconds": round(immobile_time, 2),
            "struggling_time_seconds": round(struggling_time, 2),
            "detected_rate_percent": round(detected_rate, 1),
            "immobility_percent": round(immobile_percent, 1),
            "struggling_percent": round(struggling_percent, 1),
            "uncertain_percent": round(uncertain_percent, 1),
            "immobility_episodes": stats["immobility_episodes"],
            "struggling_episodes": stats["struggling_episodes"],
            "immobility_frequency_per_min": round(immobile_freq, 2),
            "struggling_frequency_per_min": round(struggling_freq, 2),
            "latency_to_immobility_seconds": round(latency_seconds, 2) if latency_seconds is not None else None,
            "detected_frames": stats.get("detected_frames", 0),
            "undetected_frames": stats.get("undetected_frames", 0),
            "total_frames": stats["total_frames"],
        }

    def _safe_dt(self, dt_s):
        try:
            return max(float(dt_s), 1e-6)
        except Exception:
            return 1.0 / max(float(self.fps), 1e-6)
