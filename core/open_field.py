#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open Field Test helpers.

This module implements arena-level tracking, perspective normalization and
result rendering so the Flask app can focus on workflow orchestration.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from file_utils import ascii_slug, read_image_unicode, write_image_unicode
from spatial_render import (
    append_heatmap_legend,
    classic_border_thickness,
    classic_guide_thickness,
    classic_heatmap,
    classic_marker_radius,
    classic_path_thickness,
    draw_start_end_markers,
    fill_rect_alpha,
    style_color,
)


def _order_quad(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Return 4 corner points in a stable clockwise order."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Open field arena calibration requires exactly 4 points.")

    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]

    # Rotate to a deterministic first point.  The older sum/diff method can
    # reuse the same vertex twice for diamond-shaped arenas where scores tie.
    start = int(np.lexsort((ordered[:, 1], ordered[:, 0] + ordered[:, 1]))[0])
    ordered = np.roll(ordered, -start, axis=0)

    if len(np.unique(np.round(ordered, decimals=3), axis=0)) != 4:
        raise ValueError("Open field arena calibration points must be distinct.")
    if cv2.contourArea(ordered.astype(np.float32)) <= 1.0:
        raise ValueError("Open field arena calibration points form a degenerate quadrilateral.")
    return ordered.astype(np.float32)


def _normalize_polygon(points: Optional[Sequence[Sequence[float]]], min_points: int = 3) -> Optional[np.ndarray]:
    if points is None:
        return None

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < min_points:
        raise ValueError(f"Open field detection mask requires at least {min_points} points.")
    return pts


def sanitize_name(text: str, fallback: str = "arena") -> str:
    return ascii_slug(text, fallback=fallback, max_length=80)


class OpenFieldTracker:
    """Track one mouse inside one calibrated open-field arena."""

    def __init__(
        self,
        arena_points: Sequence[Sequence[float]],
        arena_side_cm: Optional[float] = None,
        arena_width_cm: Optional[float] = None,
        arena_height_cm: Optional[float] = None,
        fps: float = 30.0,
        center_ratio_x: float = 0.5,
        center_ratio_y: float = 0.5,
        wall_margin_cm: float = 8.0,
        movement_threshold_cm_s: float = 2.0,
        roi_id: int = 1,
        roi_name: Optional[str] = None,
        time_config: Optional[Dict[str, Any]] = None,
        global_start_time_s: float = 0.0,
        detection_points: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        if arena_side_cm is not None:
            arena_width_cm = arena_side_cm
            arena_height_cm = arena_side_cm

        if arena_width_cm is None or arena_height_cm is None:
            raise ValueError("Open field tracker requires arena size in centimeters.")

        self.roi_id = int(roi_id)
        self.roi_name = roi_name or f"Arena {self.roi_id}"
        self.time_config = self._normalize_time_config(time_config)
        self.global_start_time_s = max(0.0, float(global_start_time_s or 0.0))

        self.raw_arena_points = arena_points
        self.arena_points = _order_quad(arena_points)
        self.raw_detection_points = detection_points
        self.detection_points = _normalize_polygon(detection_points) if detection_points is not None else self.arena_points.copy()
        self.has_custom_detection_mask = detection_points is not None
        self.arena_width_cm = max(1.0, float(arena_width_cm))
        self.arena_height_cm = max(1.0, float(arena_height_cm))
        self.fps = max(1.0, float(fps))
        self.center_ratio_x = float(np.clip(center_ratio_x, 0.1, 1.0))
        self.center_ratio_y = float(np.clip(center_ratio_y, 0.1, 1.0))
        max_margin = max(0.0, min(self.arena_width_cm, self.arena_height_cm) / 2.0 - 0.1)
        self.wall_margin_cm = float(np.clip(wall_margin_cm, 0.0, max_margin))
        self.movement_threshold_cm_s = max(0.0, float(movement_threshold_cm_s))

        dst_rect = np.array(
            [
                [0.0, 0.0],
                [self.arena_width_cm, 0.0],
                [self.arena_width_cm, self.arena_height_cm],
                [0.0, self.arena_height_cm],
            ],
            dtype=np.float32,
        )
        self.h_img_to_cm = cv2.getPerspectiveTransform(self.arena_points.astype(np.float32), dst_rect)
        self.h_cm_to_img = cv2.getPerspectiveTransform(dst_rect, self.arena_points.astype(np.float32))
        self._center_rect_cm_cache: Optional[Tuple[float, float, float, float]] = None
        self._center_polygon_px_cache: Optional[np.ndarray] = None

        self.detection_mask: Optional[np.ndarray] = None
        self.detection_bbox: Optional[Tuple[int, int, int, int]] = None
        self._cropped_detection_mask: Optional[np.ndarray] = None

        self.total_frames = 0
        self.frames_in_window = 0
        self.detected_frames = 0
        self.detected_frames_in_window = 0
        self.center_frames = 0
        self.outer_frames = 0
        self.wall_frames = 0
        self.moving_frames = 0
        self.immobile_frames = 0
        self.time_in_window_s = 0.0
        self.detected_time_in_window_s = 0.0
        self.center_time_s = 0.0
        self.outer_time_s = 0.0
        self.wall_time_s = 0.0
        self.moving_time_s = 0.0
        self.immobile_time_s = 0.0
        self.center_entries = 0
        self.center_latency_s: Optional[float] = None
        self.total_distance_cm = 0.0
        self.center_distance_cm = 0.0
        self.outer_distance_cm = 0.0
        self.max_speed_cm_s = 0.0
        self.last_speed_cm_s = 0.0
        self.last_zone_label = "Unknown"

        self.first_detected_time_abs: Optional[float] = None
        self.prev_all_arena_xy_cm: Optional[Tuple[float, float]] = None
        self.last_all_arena_xy_cm: Optional[Tuple[float, float]] = None
        self.prev_stats_arena_xy_cm: Optional[Tuple[float, float]] = None
        self.prev_centroid_px: Optional[Tuple[float, float]] = None
        self.prev_in_center_stats = False
        self.missing_frames = 0
        self.max_gap_frames = 5
        self.missing_time_s = 0.0
        self.max_gap_s = self.max_gap_frames / max(float(self.fps), 1e-6)

        self.trajectory_px_live: deque[Tuple[float, float]] = deque(maxlen=900)
        self.trajectory_px_stats: List[Tuple[float, float]] = []
        self.trajectory_cm_stats: List[Tuple[float, float]] = []
        self.frame_records: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize_time_config(time_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(time_config or {})
        mode = str(cfg.get("mode") or "global").strip().lower()
        if mode not in {"global", "precise", "relative"}:
            mode = "global"
        cfg["mode"] = mode
        for key in ("start", "end"):
            value = cfg.get(key)
            if value in ("", None):
                cfg[key] = None
                continue
            try:
                cfg[key] = float(value)
            except Exception:
                cfg[key] = None
        return cfg

    def initialize(self, frame_shape: Tuple[int, ...]) -> None:
        if self.detection_mask is not None:
            return

        self.detection_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.detection_mask, [self.detection_points.astype(np.int32)], 255)
        x, y, w, h = cv2.boundingRect(self.detection_points.astype(np.int32))
        self.detection_bbox = (x, y, w, h)
        self._cropped_detection_mask = self.detection_mask[y : y + h, x : x + w]

    def get_center_rect_cm(self) -> Tuple[float, float, float, float]:
        if self._center_rect_cm_cache is not None:
            return self._center_rect_cm_cache
        center_w = self.arena_width_cm * self.center_ratio_x
        center_h = self.arena_height_cm * self.center_ratio_y
        x0 = (self.arena_width_cm - center_w) / 2.0
        y0 = (self.arena_height_cm - center_h) / 2.0
        self._center_rect_cm_cache = (x0, y0, x0 + center_w, y0 + center_h)
        return self._center_rect_cm_cache

    def cm_rect_to_image_polygon(self, rect_cm: Tuple[float, float, float, float]) -> np.ndarray:
        x0, y0, x1, y1 = rect_cm
        pts = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32)
        return cv2.perspectiveTransform(pts, self.h_cm_to_img)[0].astype(np.int32)

    def get_center_polygon_px(self) -> np.ndarray:
        if self._center_polygon_px_cache is None:
            self._center_polygon_px_cache = self.cm_rect_to_image_polygon(self.get_center_rect_cm())
        return self._center_polygon_px_cache

    def project_point_to_cm(self, point_px: Sequence[float]) -> Tuple[float, float]:
        src = np.array([[[float(point_px[0]), float(point_px[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.h_img_to_cm)[0, 0]
        x_cm = float(np.clip(dst[0], 0.0, self.arena_width_cm))
        y_cm = float(np.clip(dst[1], 0.0, self.arena_height_cm))
        return (x_cm, y_cm)

    @staticmethod
    def _segment_length_in_rect(
        start_xy_cm: Tuple[float, float],
        end_xy_cm: Tuple[float, float],
        rect_cm: Tuple[float, float, float, float],
    ) -> float:
        x0, y0 = (float(start_xy_cm[0]), float(start_xy_cm[1]))
        x1, y1 = (float(end_xy_cm[0]), float(end_xy_cm[1]))
        dx = x1 - x0
        dy = y1 - y0
        total = float(math.hypot(dx, dy))
        if total <= 1e-6:
            return 0.0

        rx0, ry0, rx1, ry1 = rect_cm
        t0 = 0.0
        t1 = 1.0

        for p, q in (
            (-dx, x0 - rx0),
            (dx, rx1 - x0),
            (-dy, y0 - ry0),
            (dy, ry1 - y0),
        ):
            if abs(p) <= 1e-12:
                if q < 0:
                    return 0.0
                continue
            r = q / p
            if p < 0:
                if r > t1:
                    return 0.0
                t0 = max(t0, r)
            else:
                if r < t0:
                    return 0.0
                t1 = min(t1, r)

        if t1 <= t0:
            return 0.0
        return total * max(0.0, min(1.0, t1) - max(0.0, t0))

    def _split_step_distance(
        self,
        start_xy_cm: Tuple[float, float],
        end_xy_cm: Tuple[float, float],
    ) -> Tuple[float, float]:
        total = float(math.hypot(end_xy_cm[0] - start_xy_cm[0], end_xy_cm[1] - start_xy_cm[1]))
        if total <= 1e-6:
            return (0.0, 0.0)

        center_distance_cm = self._segment_length_in_rect(start_xy_cm, end_xy_cm, self.get_center_rect_cm())
        center_distance_cm = float(np.clip(center_distance_cm, 0.0, total))
        outer_distance_cm = max(0.0, total - center_distance_cm)
        return (center_distance_cm, outer_distance_cm)

    def _zone_flags(self, arena_xy_cm: Tuple[float, float]) -> Dict[str, bool]:
        x_cm, y_cm = arena_xy_cm
        cx0, cy0, cx1, cy1 = self.get_center_rect_cm()
        in_center = (cx0 <= x_cm <= cx1) and (cy0 <= y_cm <= cy1)
        in_wall = (
            x_cm <= self.wall_margin_cm
            or x_cm >= (self.arena_width_cm - self.wall_margin_cm)
            or y_cm <= self.wall_margin_cm
            or y_cm >= (self.arena_height_cm - self.wall_margin_cm)
        )
        return {
            "in_center": in_center,
            "in_outer": not in_center,
            "in_wall": in_wall,
        }

    @staticmethod
    def _zone_label(flags: Dict[str, bool]) -> str:
        if flags["in_center"]:
            return "Center"
        if flags["in_wall"]:
            return "Wall"
        return "Outer"

    def _stats_window_origin_abs(self) -> float:
        mode = self.time_config.get("mode", "global")
        start_cfg = self.time_config.get("start")
        if mode == "precise":
            return float(start_cfg or self.global_start_time_s)
        if mode == "relative" and self.first_detected_time_abs is not None:
            return float(self.first_detected_time_abs + float(start_cfg or 0.0))
        return self.global_start_time_s

    def _is_in_stats_window(self, current_time: float) -> bool:
        mode = self.time_config.get("mode", "global")
        start_cfg = self.time_config.get("start")
        end_cfg = self.time_config.get("end")

        if mode == "global":
            return True
        if mode == "precise":
            if start_cfg is not None and current_time < start_cfg:
                return False
            if end_cfg is not None and current_time > end_cfg:
                return False
            return True
        if mode == "relative":
            if self.first_detected_time_abs is None:
                return False
            elapsed = current_time - self.first_detected_time_abs
            if start_cfg is not None and elapsed < start_cfg:
                return False
            if end_cfg is not None and elapsed > end_cfg:
                return False
            return True
        return True

    def detect_mouse(self, frame: np.ndarray, gray: np.ndarray, bg_subtractor, morph_processor, detector):
        self.initialize(frame.shape)
        assert self.detection_bbox is not None
        assert self._cropped_detection_mask is not None

        x, y, w, h = self.detection_bbox
        arena_frame = frame[y : y + h, x : x + w]
        arena_gray = gray[y : y + h, x : x + w]

        if getattr(bg_subtractor, "method", "") == "static":
            full_fg_mask = bg_subtractor.apply(frame, gray=gray)
            fg_mask = full_fg_mask[y : y + h, x : x + w]
        else:
            fg_mask = bg_subtractor.apply(arena_frame, gray=arena_gray)
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=self._cropped_detection_mask)
        fg_mask = morph_processor.process(fg_mask, inplace=True)
        detections = detector.detect(fg_mask)

        if not detections:
            return None, fg_mask

        detection = max(detections, key=lambda item: item.get("area", 0))
        cx, cy = detection["centroid"]
        detection["centroid"] = (cx + x, cy + y)

        bx, by, bw, bh = detection["bbox"]
        detection["bbox"] = (bx + x, by + y, bw, bh)

        contour = detection.get("contour")
        if contour is not None:
            detection["contour"] = contour + np.array([[x, y]])

        return detection, fg_mask

    def update(
        self,
        frame: np.ndarray,
        current_time: float,
        gray: np.ndarray,
        bg_subtractor,
        morph_processor,
        detector,
        dt_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.total_frames += 1
        try:
            current_dt = float(dt_s) if dt_s is not None else 1.0 / max(float(self.fps), 1e-6)
        except (TypeError, ValueError):
            current_dt = 1.0 / max(float(self.fps), 1e-6)
        current_dt = max(current_dt, 1e-6)
        detection, fg_mask = self.detect_mouse(frame, gray, bg_subtractor, morph_processor, detector)

        centroid_px: Optional[Tuple[float, float]] = None
        bbox = None
        contour = None
        detected = detection is not None

        if detection is not None:
            self.detected_frames += 1
            centroid_px = tuple(float(v) for v in detection["centroid"])
            bbox = detection.get("bbox")
            contour = detection.get("contour")
            arena_xy_cm = self.project_point_to_cm(centroid_px)
            self.last_all_arena_xy_cm = arena_xy_cm
            self.prev_centroid_px = centroid_px
            self.missing_frames = 0
            self.missing_time_s = 0.0
            if self.first_detected_time_abs is None:
                self.first_detected_time_abs = float(current_time)
        elif self.last_all_arena_xy_cm is not None and self.missing_time_s < self.max_gap_s:
            arena_xy_cm = self.last_all_arena_xy_cm
            centroid_px = self.prev_centroid_px
            self.missing_frames += 1
            self.missing_time_s += current_dt
        else:
            arena_xy_cm = None
            self.missing_frames += 1
            self.missing_time_s += current_dt

        instant_distance_cm = 0.0
        speed_cm_s = 0.0
        zone_flags = {"in_center": False, "in_outer": False, "in_wall": False}
        zone_label = "Unknown"

        if arena_xy_cm is not None:
            if self.prev_all_arena_xy_cm is not None:
                dx = arena_xy_cm[0] - self.prev_all_arena_xy_cm[0]
                dy = arena_xy_cm[1] - self.prev_all_arena_xy_cm[1]
                instant_distance_cm = float(math.hypot(dx, dy))
                speed_cm_s = instant_distance_cm / current_dt
            self.prev_all_arena_xy_cm = arena_xy_cm
            zone_flags = self._zone_flags(arena_xy_cm)
            zone_label = self._zone_label(zone_flags)
            self.last_zone_label = zone_label
        else:
            self.prev_all_arena_xy_cm = None

        self.last_speed_cm_s = speed_cm_s
        in_stats_window = self._is_in_stats_window(float(current_time))
        stats_step_distance_cm = 0.0
        stats_center_distance_cm = 0.0
        stats_outer_distance_cm = 0.0
        stats_cumulative_distance_cm = self.total_distance_cm

        if in_stats_window:
            self.frames_in_window += 1
            self.time_in_window_s += current_dt
            if detected:
                self.detected_frames_in_window += 1
                self.detected_time_in_window_s += current_dt

            if arena_xy_cm is not None:
                if self.prev_stats_arena_xy_cm is not None:
                    dx = arena_xy_cm[0] - self.prev_stats_arena_xy_cm[0]
                    dy = arena_xy_cm[1] - self.prev_stats_arena_xy_cm[1]
                    stats_step_distance_cm = float(math.hypot(dx, dy))
                    stats_center_distance_cm, stats_outer_distance_cm = self._split_step_distance(
                        self.prev_stats_arena_xy_cm,
                        arena_xy_cm,
                    )
                    self.total_distance_cm += stats_step_distance_cm
                    self.center_distance_cm += stats_center_distance_cm
                    self.outer_distance_cm += stats_outer_distance_cm
                self.prev_stats_arena_xy_cm = arena_xy_cm
                stats_cumulative_distance_cm = self.total_distance_cm
                self.max_speed_cm_s = max(self.max_speed_cm_s, speed_cm_s)

                self.center_frames += int(zone_flags["in_center"])
                self.outer_frames += int(zone_flags["in_outer"])
                self.wall_frames += int(zone_flags["in_wall"])
                if zone_flags["in_center"]:
                    self.center_time_s += current_dt
                if zone_flags["in_outer"]:
                    self.outer_time_s += current_dt
                if zone_flags["in_wall"]:
                    self.wall_time_s += current_dt

                if speed_cm_s >= self.movement_threshold_cm_s:
                    self.moving_frames += 1
                    self.moving_time_s += current_dt
                else:
                    self.immobile_frames += 1
                    self.immobile_time_s += current_dt

                if zone_flags["in_center"] and not self.prev_in_center_stats:
                    self.center_entries += 1
                    if self.center_latency_s is None:
                        self.center_latency_s = float(current_time) - self._stats_window_origin_abs()
                self.prev_in_center_stats = zone_flags["in_center"]

                if centroid_px is not None:
                    self.trajectory_px_live.append(centroid_px)
                    self.trajectory_px_stats.append(centroid_px)
                self.trajectory_cm_stats.append(arena_xy_cm)
            else:
                self.prev_stats_arena_xy_cm = None
                self.prev_in_center_stats = False
                self.immobile_frames += 1
                self.immobile_time_s += current_dt
        else:
            self.prev_stats_arena_xy_cm = None
            self.prev_in_center_stats = False
            stats_cumulative_distance_cm = self.total_distance_cm
            if centroid_px is not None:
                self.trajectory_px_live.append(centroid_px)

        record = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "frame": self.total_frames - 1,
            "time_s": round(float(current_time), 3),
            "dt_s": round(float(current_dt), 6),
            "window_time_s": round(float(current_time - self._stats_window_origin_abs()), 3) if in_stats_window else None,
            "detected": bool(detected),
            "in_stats_window": bool(in_stats_window),
            "x_cm": round(float(arena_xy_cm[0]), 3) if arena_xy_cm is not None else None,
            "y_cm": round(float(arena_xy_cm[1]), 3) if arena_xy_cm is not None else None,
            "speed_cm_s": round(float(speed_cm_s), 3),
            "step_distance_cm": round(float(stats_step_distance_cm), 3),
            "step_center_distance_cm": round(float(stats_center_distance_cm), 3),
            "step_outer_distance_cm": round(float(stats_outer_distance_cm), 3),
            "cumulative_distance_cm": round(float(stats_cumulative_distance_cm), 3),
            "in_center": bool(zone_flags["in_center"]) if in_stats_window else False,
            "in_outer": bool(zone_flags["in_outer"]) if in_stats_window else False,
            "in_wall": bool(zone_flags["in_wall"]) if in_stats_window else False,
            "zone": zone_label if arena_xy_cm is not None else "Unknown",
        }
        self.frame_records.append(record)

        return {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "detected": bool(detected),
            "centroid": centroid_px,
            "bbox": bbox,
            "contour": contour,
            "fg_mask": fg_mask,
            "arena_xy_cm": arena_xy_cm,
            "speed_cm_s": speed_cm_s,
            "step_distance_cm": stats_step_distance_cm,
            "cumulative_distance_cm": self.total_distance_cm,
            "zone_flags": zone_flags,
            "zone_label": zone_label,
            "trajectory_px": list(self.trajectory_px_live),
            "in_stats_window": bool(in_stats_window),
            "record": record,
        }

    def build_summary(self) -> Dict[str, Any]:
        total_time_s = self.time_in_window_s or (self.frames_in_window / self.fps if self.fps > 0 else 0.0)
        center_time_s = self.center_time_s or (self.center_frames / self.fps if self.fps > 0 else 0.0)
        outer_time_s = self.outer_time_s or (self.outer_frames / self.fps if self.fps > 0 else 0.0)
        wall_time_s = self.wall_time_s or (self.wall_frames / self.fps if self.fps > 0 else 0.0)
        moving_time_s = self.moving_time_s or (self.moving_frames / self.fps if self.fps > 0 else 0.0)
        immobile_time_s = self.immobile_time_s or (self.immobile_frames / self.fps if self.fps > 0 else 0.0)
        mean_speed_cm_s = self.total_distance_cm / max(total_time_s, 1e-6)
        detection_rate = (self.detected_time_in_window_s / max(total_time_s, 1e-9) * 100.0) if total_time_s > 0 else 0.0

        summary = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "time_mode": self.time_config.get("mode", "global"),
            "time_start_s": self.time_config.get("start"),
            "time_end_s": self.time_config.get("end"),
            "analysis_window_start_s": round(self._stats_window_origin_abs(), 3),
            "total_time_s": round(total_time_s, 3),
            "detected_frames": int(self.detected_frames_in_window),
            "detected_rate_percent": round(detection_rate, 2),
            "total_distance_cm": round(self.total_distance_cm, 3),
            "center_distance_cm": round(self.center_distance_cm, 3),
            "outer_distance_cm": round(self.outer_distance_cm, 3),
            "mean_speed_cm_s": round(mean_speed_cm_s, 3),
            "max_speed_cm_s": round(self.max_speed_cm_s, 3),
            "center_time_s": round(center_time_s, 3),
            "center_percent": round(center_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "outer_time_s": round(outer_time_s, 3),
            "outer_percent": round(outer_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "wall_time_s": round(wall_time_s, 3),
            "wall_percent": round(wall_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "moving_time_s": round(moving_time_s, 3),
            "immobility_time_s": round(immobile_time_s, 3),
            "center_entries": int(self.center_entries),
            "center_latency_s": round(self.center_latency_s, 3) if self.center_latency_s is not None else None,
            "arena_width_cm": self.arena_width_cm,
            "arena_height_cm": self.arena_height_cm,
            "center_ratio_x": self.center_ratio_x,
            "center_ratio_y": self.center_ratio_y,
            "wall_margin_cm": self.wall_margin_cm,
            "movement_threshold_cm_s": self.movement_threshold_cm_s,
        }
        return summary

    def build_time_bins(self, bin_seconds: float = 300.0) -> List[Dict[str, Any]]:
        rows = [row for row in self.frame_records if row.get("in_stats_window")]
        if not rows:
            return []

        bin_seconds = max(1.0, float(bin_seconds))
        bins: Dict[int, Dict[str, Any]] = {}

        for row in rows:
            t = float(row.get("window_time_s") or 0.0)
            idx = int(t // bin_seconds)
            bucket = bins.setdefault(
                idx,
                {
                    "roi_id": self.roi_id,
                    "roi_name": self.roi_name,
                    "bin_index": idx + 1,
                    "start_s": idx * bin_seconds,
                    "end_s": (idx + 1) * bin_seconds,
                    "frames": 0,
                    "distance_cm": 0.0,
                    "center_distance_cm": 0.0,
                    "outer_distance_cm": 0.0,
                    "center_frames": 0,
                    "wall_frames": 0,
                    "moving_frames": 0,
                    "duration_s": 0.0,
                    "center_time_s": 0.0,
                    "wall_time_s": 0.0,
                    "moving_time_s": 0.0,
                },
            )
            duration = float(row.get("dt_s") or (1.0 / max(self.fps, 1e-6)))
            bucket["frames"] += 1
            bucket["duration_s"] += duration
            bucket["distance_cm"] += float(row.get("step_distance_cm") or 0.0)
            bucket["center_distance_cm"] += float(row.get("step_center_distance_cm") or 0.0)
            bucket["outer_distance_cm"] += float(row.get("step_outer_distance_cm") or 0.0)
            bucket["center_frames"] += int(bool(row.get("in_center")))
            bucket["wall_frames"] += int(bool(row.get("in_wall")))
            bucket["moving_frames"] += int(float(row.get("speed_cm_s") or 0.0) >= self.movement_threshold_cm_s)
            if row.get("in_center"):
                bucket["center_time_s"] += duration
            if row.get("in_wall"):
                bucket["wall_time_s"] += duration
            if float(row.get("speed_cm_s") or 0.0) >= self.movement_threshold_cm_s:
                bucket["moving_time_s"] += duration

        out: List[Dict[str, Any]] = []
        for idx in sorted(bins):
            bucket = bins[idx]
            duration_s = float(bucket.get("duration_s") or (max(bucket["frames"], 1) / self.fps))
            out.append(
                {
                    "roi_id": bucket["roi_id"],
                    "roi_name": bucket["roi_name"],
                    "bin_index": bucket["bin_index"],
                    "start_s": round(bucket["start_s"], 2),
                    "end_s": round(bucket["end_s"], 2),
                    "duration_s": round(duration_s, 3),
                    "distance_cm": round(bucket["distance_cm"], 3),
                    "center_distance_cm": round(bucket["center_distance_cm"], 3),
                    "outer_distance_cm": round(bucket["outer_distance_cm"], 3),
                    "mean_speed_cm_s": round(bucket["distance_cm"] / max(duration_s, 1e-6), 3),
                    "center_time_s": round(bucket["center_time_s"], 3),
                    "wall_time_s": round(bucket["wall_time_s"], 3),
                    "moving_time_s": round(bucket["moving_time_s"], 3),
                }
            )
        return out


def _zone_color(zone_label: str, detected: bool) -> Tuple[int, int, int]:
    if not detected:
        return (120, 120, 120)
    if zone_label == "Center":
        return (34, 197, 94)
    if zone_label == "Wall":
        return (249, 115, 22)
    return (59, 130, 246)

def draw_open_field_results(
    frame: np.ndarray,
    tracker_results: Sequence[Tuple[OpenFieldTracker, Dict[str, Any]]],
    inplace: bool = False,
) -> np.ndarray:
    output = frame if inplace else frame.copy()

    for tracker, result in tracker_results:
        arena_poly = tracker.arena_points.astype(np.int32)
        detection_poly = tracker.detection_points.astype(np.int32)
        center_poly = tracker.get_center_polygon_px()
        zone_label = result.get("zone_label", "Unknown")
        detected = bool(result.get("detected"))
        color = _zone_color(zone_label, detected)

        cv2.polylines(output, [arena_poly], True, color, 2)
        if tracker.has_custom_detection_mask:
            cv2.polylines(output, [detection_poly], True, (255, 214, 10), 1, cv2.LINE_AA)
        cv2.polylines(output, [center_poly], True, (255, 215, 0), 1)

        contour = result.get("contour")
        centroid = result.get("centroid")
        if contour is not None:
            cv2.drawContours(output, [contour], -1, color, 2)
        if centroid is not None:
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(output, (cx, cy), 4, color, -1)

        traj = result.get("trajectory_px") or []
        for idx in range(1, len(traj)):
            pt1 = (int(traj[idx - 1][0]), int(traj[idx - 1][1]))
            pt2 = (int(traj[idx][0]), int(traj[idx][1]))
            cv2.line(output, pt1, pt2, color, 1, cv2.LINE_AA)

        min_x = int(np.min(arena_poly[:, 0]))
        min_y = int(np.min(arena_poly[:, 1]))
        label = tracker.roi_name
        sub = zone_label if result.get("in_stats_window", True) else f"{zone_label} | paused"
        box_w = max(160, int(max(len(label), len(sub)) * 9.5))
        cv2.rectangle(output, (min_x, max(0, min_y - 44)), (min_x + box_w, max(24, min_y + 2)), (8, 15, 28), -1)
        cv2.rectangle(output, (min_x, max(0, min_y - 44)), (min_x + box_w, max(24, min_y + 2)), color, 1)
        cv2.putText(output, label, (min_x + 8, max(16, min_y - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(output, sub, (min_x + 8, max(34, min_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 210, 220), 1, cv2.LINE_AA)

    return output


def _paper_blend_heatmap(norm: np.ndarray, style: Any | None = None) -> np.ndarray:
    return classic_heatmap(norm, style)


def _paper_border_thickness(pixels_per_cm: int) -> int:
    return classic_border_thickness(pixels_per_cm)


def _paper_guide_thickness(pixels_per_cm: int) -> int:
    return classic_guide_thickness(pixels_per_cm)


def _paper_path_thickness(pixels_per_cm: int) -> int:
    return classic_path_thickness(pixels_per_cm)


def _paper_dash_length(pixels_per_cm: int) -> int:
    return max(8, int(round(pixels_per_cm * 0.8)))


def _paper_gap_length(pixels_per_cm: int) -> int:
    return max(4, int(round(pixels_per_cm * 0.35)))


def _paper_draw_dashed_line(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash_length: int,
    gap_length: int,
) -> None:
    start_arr = np.array(start, dtype=np.float32)
    end_arr = np.array(end, dtype=np.float32)
    direction = end_arr - start_arr
    total = float(np.linalg.norm(direction))
    if total <= 1e-6:
        return

    unit = direction / total
    distance = 0.0
    while distance < total:
        seg_start = start_arr + unit * distance
        seg_end = start_arr + unit * min(total, distance + dash_length)
        cv2.line(
            image,
            tuple(np.round(seg_start).astype(int)),
            tuple(np.round(seg_end).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        distance += dash_length + gap_length


def _paper_draw_dashed_rect(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    dash_length: int,
    gap_length: int,
) -> None:
    x0, y0 = top_left
    x1, y1 = bottom_right
    _paper_draw_dashed_line(image, (x0, y0), (x1, y0), color, thickness, dash_length, gap_length)
    _paper_draw_dashed_line(image, (x1, y0), (x1, y1), color, thickness, dash_length, gap_length)
    _paper_draw_dashed_line(image, (x1, y1), (x0, y1), color, thickness, dash_length, gap_length)
    _paper_draw_dashed_line(image, (x0, y1), (x0, y0), color, thickness, dash_length, gap_length)


def render_open_field_heatmap(
    tracker: OpenFieldTracker,
    output_path: str,
    pixels_per_cm: int = 12,
    style: Any | None = None,
) -> Optional[str]:
    points = tracker.trajectory_cm_stats
    if not points:
        return None

    width = max(160, int(round(tracker.arena_width_cm * pixels_per_cm)))
    height = max(160, int(round(tracker.arena_height_cm * pixels_per_cm)))
    heat = np.zeros((height, width), dtype=np.float32)

    for x_cm, y_cm in points:
        px = int(np.clip(round(x_cm * pixels_per_cm), 0, width - 1))
        py = int(np.clip(round(y_cm * pixels_per_cm), 0, height - 1))
        cv2.circle(heat, (px, py), max(4, pixels_per_cm // 2), 1.0, thickness=-1)

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(3, pixels_per_cm), sigmaY=max(3, pixels_per_cm))
    if float(heat.max()) <= 0.0:
        return None

    norm = np.clip(heat / heat.max(), 0.0, 1.0)
    color = _paper_blend_heatmap(norm, style)
    border_thickness = _paper_border_thickness(pixels_per_cm)
    guide_thickness = _paper_guide_thickness(pixels_per_cm)
    heat_outline = style_color(style, "heatmap", "outline")
    heat_border = style_color(style, "heatmap", "border")

    cx0, cy0, cx1, cy1 = tracker.get_center_rect_cm()
    cv2.rectangle(
        color,
        (int(cx0 * pixels_per_cm), int(cy0 * pixels_per_cm)),
        (int(cx1 * pixels_per_cm), int(cy1 * pixels_per_cm)),
        heat_outline,
        guide_thickness,
        cv2.LINE_AA,
    )
    cv2.rectangle(color, (0, 0), (width - 1, height - 1), heat_border, border_thickness, cv2.LINE_AA)
    color = append_heatmap_legend(color, style)

    if write_image_unicode(output_path, color):
        return output_path
    return None


def render_open_field_trackmap(
    tracker: OpenFieldTracker,
    output_path: str,
    pixels_per_cm: int = 12,
    style: Any | None = None,
) -> Optional[str]:
    points_cm = tracker.trajectory_cm_stats
    if not points_cm:
        return None

    width = max(160, int(round(tracker.arena_width_cm * pixels_per_cm)))
    height = max(160, int(round(tracker.arena_height_cm * pixels_per_cm)))
    canvas = np.full((height, width, 3), style_color(style, "trackmap", "background"), dtype=np.uint8)
    border_thickness = _paper_border_thickness(pixels_per_cm)
    guide_thickness = _paper_guide_thickness(pixels_per_cm)
    path_thickness = _paper_path_thickness(pixels_per_cm)
    marker_radius = classic_marker_radius(pixels_per_cm)

    cx0, cy0, cx1, cy1 = tracker.get_center_rect_cm()
    center_top_left = (int(cx0 * pixels_per_cm), int(cy0 * pixels_per_cm))
    center_bottom_right = (int(cx1 * pixels_per_cm), int(cy1 * pixels_per_cm))
    fill_rect_alpha(canvas, center_top_left, center_bottom_right, style_color(style, "trackmap", "special_fill"), alpha=0.92)
    cv2.rectangle(canvas, center_top_left, center_bottom_right, style_color(style, "trackmap", "special_outline"), guide_thickness, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), style_color(style, "trackmap", "border"), border_thickness, cv2.LINE_AA)

    points = []
    for x_cm, y_cm in points_cm:
        px = int(np.clip(round(x_cm * pixels_per_cm), 0, width - 1))
        py = int(np.clip(round(y_cm * pixels_per_cm), 0, height - 1))
        points.append((px, py))

    for idx in range(1, len(points)):
        cv2.line(canvas, points[idx - 1], points[idx], style_color(style, "trackmap", "track"), path_thickness, cv2.LINE_AA)
    draw_start_end_markers(canvas, points, marker_radius, style)

    if write_image_unicode(output_path, canvas):
        return output_path
    return None


def render_open_field_image_grid(
    image_paths: Sequence[str],
    output_path: str,
    title: str = "",
    columns: int = 2,
    padding: int = 18,
    style: Any | None = None,
) -> Optional[str]:
    valid_paths = [path for path in image_paths if path]
    if not valid_paths:
        return None

    images = []
    for path in valid_paths:
        img = read_image_unicode(path)
        if img is not None:
            images.append(img)
    if not images:
        return None

    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    cols = max(1, min(int(columns), len(images)))
    rows = int(math.ceil(len(images) / cols))
    title_h = 46 if title else 0

    canvas_h = title_h + rows * max_h + (rows + 1) * padding
    canvas_w = cols * max_w + (cols + 1) * padding
    canvas = np.full((canvas_h, canvas_w, 3), style_color(style, "trackmap", "grid_background"), dtype=np.uint8)

    if title:
        cv2.putText(canvas, title, (padding, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, style_color(style, "trackmap", "grid_title"), 2, cv2.LINE_AA)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y = title_h + padding + row * (max_h + padding)
        x = padding + col * (max_w + padding)
        canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
        cv2.rectangle(canvas, (x, y), (x + img.shape[1], y + img.shape[0]), style_color(style, "trackmap", "grid_border"), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (canvas_w - 1, canvas_h - 1), style_color(style, "trackmap", "grid_border"), 1, cv2.LINE_AA)

    if write_image_unicode(output_path, canvas):
        return output_path
    return None
