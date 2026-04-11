#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Y-maze helpers for spontaneous alternation analysis based on manually drawn zones.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from file_utils import ascii_slug, read_image_unicode, write_image_unicode
from maze_zone_utils import (
    build_entry_polygon,
    build_union_mask,
    estimate_cm_per_px,
    format_metric,
    format_seconds,
    order_quad_along_axis,
    order_triangle_by_axes,
    polygon_centroid,
    polygons_bbox,
    project_point_homography,
    quad_short_long_edges,
    render_canvas_from_cm_polygons,
    sort_polygon_points,
)
from spatial_render import (
    CLASSIC_BG,
    CLASSIC_BORDER,
    CLASSIC_GUIDE_FILL,
    CLASSIC_GUIDE_OUTLINE,
    CLASSIC_HEAT_OUTLINE,
    CLASSIC_TEXT,
    CLASSIC_TRACK,
    classic_heatmap,
    classic_marker_radius,
    classic_path_thickness,
    draw_start_end_markers,
    fill_poly_alpha,
)


def sanitize_name(text: str, fallback: str = "ymaze") -> str:
    return ascii_slug(text, fallback=fallback, max_length=80)


def _paper_blend_heatmap(norm: np.ndarray) -> np.ndarray:
    return classic_heatmap(norm)


def _zone_color(zone_group: str, detected: bool) -> Tuple[int, int, int]:
    if not detected:
        return (120, 120, 120)
    if zone_group == "center":
        return (34, 197, 94)
    return (59, 130, 246)


def _draw_poly_outline(
    image: np.ndarray,
    poly: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.polylines(image, [poly.astype(np.int32)], True, color, thickness, cv2.LINE_AA)


def _render_canvas_from_image_polygons(
    polygons: Dict[str, np.ndarray],
    padding: int = 24,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    min_x, min_y, max_x, max_y = polygons_bbox(polygons.values())
    width = max(160, int(max_x - min_x + 1 + padding * 2))
    height = max(160, int(max_y - min_y + 1 + padding * 2))
    offset_x = padding - min_x
    offset_y = padding - min_y
    translated = {
        name: (poly + np.array([offset_x, offset_y], dtype=np.float32)).astype(np.int32)
        for name, poly in polygons.items()
    }
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, list(translated.values()), 255)
    return mask, translated


def _cm_point_to_canvas_px(
    point_cm: Tuple[float, float],
    offset_cm: Tuple[float, float],
    pixels_per_cm: int,
    image_shape: Tuple[int, int],
) -> Tuple[int, int]:
    px = int(np.clip(round((float(point_cm[0]) + offset_cm[0]) * float(pixels_per_cm)), 0, image_shape[1] - 1))
    py = int(np.clip(round((float(point_cm[1]) + offset_cm[1]) * float(pixels_per_cm)), 0, image_shape[0] - 1))
    return px, py


def _extend_edge_to_quad(
    start: Sequence[float],
    end: Sequence[float],
    center_point: Sequence[float],
    length_cm: float,
) -> np.ndarray:
    p0 = np.asarray(start, dtype=np.float32).reshape(2)
    p1 = np.asarray(end, dtype=np.float32).reshape(2)
    center = np.asarray(center_point, dtype=np.float32).reshape(2)
    edge = p1 - p0
    norm = float(np.linalg.norm(edge))
    if norm <= 1e-6:
        raise ValueError("Cannot extend a zero-length edge.")
    tangent = edge / norm
    normal_a = np.array([tangent[1], -tangent[0]], dtype=np.float32)
    normal_b = -normal_a
    midpoint = (p0 + p1) * 0.5
    outward = normal_a if float(np.dot(normal_a, midpoint - center)) > 0.0 else normal_b
    offset = outward * float(length_cm)
    return np.vstack([p0, p1, p1 + offset, p0 + offset]).astype(np.float32)


def _project_point_with_transform(
    point: Sequence[float],
    transform: np.ndarray,
) -> Tuple[float, float]:
    src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    if tuple(transform.shape) == (2, 3):
        dst = cv2.transform(src, transform)[0, 0]
        return float(dst[0]), float(dst[1])
    return project_point_homography(point, transform)


class YMazeTracker:
    """Track one mouse in one manually drawn Y-maze and score alternation."""

    ARM_NAMES = ("arm_a", "arm_b", "arm_c")
    ZONE_ORDER = ("center", "arm_a", "arm_b", "arm_c")
    ARM_LABELS = {"arm_a": "A", "arm_b": "B", "arm_c": "C"}
    ZONE_LABELS = {"center": "Center", "arm_a": "Arm A", "arm_b": "Arm B", "arm_c": "Arm C", "Unknown": "Unknown"}

    def __init__(
        self,
        zone_polygons: Dict[str, Sequence[Sequence[float]]],
        detection_polygon: Optional[Sequence[Sequence[float]]] = None,
        arm_length_cm: float = 30.0,
        arm_width_cm: float = 10.0,
        fps: float = 30.0,
        movement_threshold_cm_s: float = 2.0,
        entry_depth_ratio: float = 0.2,
        roi_id: int = 1,
        roi_name: Optional[str] = None,
        time_config: Optional[Dict[str, Any]] = None,
        global_start_time_s: float = 0.0,
    ) -> None:
        missing = [name for name in self.ZONE_ORDER if name not in (zone_polygons or {})]
        if missing:
            raise ValueError(f"Y-maze zones are incomplete: missing {', '.join(missing)}")

        self.roi_id = int(roi_id)
        self.roi_name = roi_name or f"Y-Maze {self.roi_id}"
        self.time_config = self._normalize_time_config(time_config)
        self.global_start_time_s = max(0.0, float(global_start_time_s or 0.0))

        self.arm_length_cm = max(0.5, float(arm_length_cm))
        self.arm_width_cm = max(0.5, float(arm_width_cm))
        self.fps = max(1.0, float(fps))
        self.movement_threshold_cm_s = max(0.0, float(movement_threshold_cm_s))
        self.entry_depth_ratio = float(np.clip(entry_depth_ratio, 0.05, 0.8))

        self.zone_polygons_img = {
            name: sort_polygon_points(zone_polygons[name]).astype(np.float32)
            for name in self.ZONE_ORDER
        }
        if self.zone_polygons_img["center"].shape != (3, 2):
            raise ValueError("Y-maze center zone must use exactly 3 points.")
        for arm_name in self.ARM_NAMES:
            if self.zone_polygons_img[arm_name].shape != (4, 2):
                raise ValueError(f"Y-maze {arm_name} must use exactly 4 points.")
        self.detection_polygon_img = (
            sort_polygon_points(detection_polygon).astype(np.float32)
            if detection_polygon is not None
            else None
        )
        center_centroid = polygon_centroid(self.zone_polygons_img["center"])
        self.entry_polygons_img = {
            name: build_entry_polygon(self.zone_polygons_img[name], center_centroid, self.entry_depth_ratio)
            for name in self.ARM_NAMES
        }
        self.cm_per_px = self._estimate_cm_per_px()
        self.template_polygons_cm = self._build_template_polygons_cm()
        self._zone_img_to_template_h = self._build_zone_template_homographies()

        self.detection_mask: Optional[np.ndarray] = None
        self.detection_bbox: Optional[Tuple[int, int, int, int]] = None
        self._cropped_detection_mask: Optional[np.ndarray] = None

        self.total_frames = 0
        self.frames_in_window = 0
        self.detected_frames = 0
        self.detected_frames_in_window = 0
        self.center_frames = 0
        self.arm_frames = {name: 0 for name in self.ARM_NAMES}
        self.moving_frames = 0
        self.immobile_frames = 0
        self.arm_entries = {name: 0 for name in self.ARM_NAMES}
        self.total_arm_entries = 0
        self.alternation_count = 0
        self.total_distance_cm = 0.0
        self.max_speed_cm_s = 0.0
        self.last_speed_cm_s = 0.0

        self.arm_entry_sequence: List[str] = []
        self.current_entry_arm: Optional[str] = None
        self.first_detected_time_abs: Optional[float] = None
        self.prev_all_centroid_px: Optional[Tuple[float, float]] = None
        self.last_all_centroid_px: Optional[Tuple[float, float]] = None
        self.prev_stats_centroid_px: Optional[Tuple[float, float]] = None
        self.prev_centroid_px: Optional[Tuple[float, float]] = None
        self.missing_frames = 0
        self.max_gap_frames = 5

        self.trajectory_px_live: deque[Tuple[float, float]] = deque(maxlen=1000)
        self.trajectory_px_stats: List[Tuple[float, float]] = []
        self.trajectory_template_cm_stats: List[Tuple[float, float]] = []
        self.frame_records: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize_time_config(time_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(time_config or {})
        mode = str(cfg.get("mode") or "global").strip().lower()
        if mode not in {"global", "precise"}:
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

    def _estimate_cm_per_px(self) -> float:
        samples: List[float] = []
        for arm_name in self.ARM_NAMES:
            short_edges, long_edges = quad_short_long_edges(self.zone_polygons_img[arm_name])
            for edge in short_edges:
                if edge > 1e-6:
                    samples.append(self.arm_width_cm / edge)
            for edge in long_edges:
                if edge > 1e-6:
                    samples.append(self.arm_length_cm / edge)
        return estimate_cm_per_px(samples)

    def _build_template_polygons_cm(self) -> Dict[str, np.ndarray]:
        arm_width = float(self.arm_width_cm)
        arm_length = float(self.arm_length_cm)
        tri_height = arm_width * math.sqrt(3.0) / 2.0
        center = np.array(
            [
                [0.0, 0.0],
                [arm_width, 0.0],
                [arm_width * 0.5, tri_height],
            ],
            dtype=np.float32,
        )
        center_centroid = polygon_centroid(center)
        arm_a = np.array(
            [
                [0.0, -arm_length],
                [arm_width, -arm_length],
                [arm_width, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        arm_b = _extend_edge_to_quad(center[1], center[2], center_centroid, arm_length)
        arm_c = _extend_edge_to_quad(center[2], center[0], center_centroid, arm_length)
        return {
            "center": center,
            "arm_a": arm_a,
            "arm_b": arm_b,
            "arm_c": arm_c,
        }

    def _build_zone_template_homographies(self) -> Dict[str, np.ndarray]:
        center_centroid_src = polygon_centroid(self.zone_polygons_img["center"])
        center_centroid_dst = polygon_centroid(self.template_polygons_cm["center"])
        x_axis_src = polygon_centroid(self.zone_polygons_img["arm_b"]) - polygon_centroid(self.zone_polygons_img["arm_c"])
        y_axis_src = center_centroid_src - polygon_centroid(self.zone_polygons_img["arm_a"])
        x_axis_dst = polygon_centroid(self.template_polygons_cm["arm_b"]) - polygon_centroid(self.template_polygons_cm["arm_c"])
        y_axis_dst = center_centroid_dst - polygon_centroid(self.template_polygons_cm["arm_a"])

        transforms: Dict[str, np.ndarray] = {
            "center": cv2.getAffineTransform(
                order_triangle_by_axes(self.zone_polygons_img["center"], x_axis_src, y_axis_src),
                order_triangle_by_axes(self.template_polygons_cm["center"], x_axis_dst, y_axis_dst),
            )
        }

        for name in self.ARM_NAMES:
            src_axis = polygon_centroid(self.zone_polygons_img[name]) - center_centroid_src
            dst_axis = polygon_centroid(self.template_polygons_cm[name]) - center_centroid_dst
            transforms[name] = cv2.getPerspectiveTransform(
                order_quad_along_axis(self.zone_polygons_img[name], center_centroid_src, src_axis),
                order_quad_along_axis(self.template_polygons_cm[name], center_centroid_dst, dst_axis),
            )
        return transforms

    def project_point_to_template_cm(
        self,
        point_px: Optional[Tuple[float, float]],
        zone_key: Optional[str] = None,
    ) -> Optional[Tuple[float, float]]:
        if point_px is None:
            return None
        zone = zone_key if zone_key in self.ZONE_ORDER else self._zone_name_for_point(point_px)
        if zone not in self._zone_img_to_template_h:
            best_zone = None
            best_score = -float("inf")
            for candidate in self.ZONE_ORDER:
                score = float(cv2.pointPolygonTest(
                    self.zone_polygons_img[candidate],
                    (float(point_px[0]), float(point_px[1])),
                    True,
                ))
                if score > best_score:
                    best_score = score
                    best_zone = candidate
            zone = best_zone
        if zone not in self._zone_img_to_template_h:
            return None
        return _project_point_with_transform(point_px, self._zone_img_to_template_h[zone])

    def initialize(self, frame_shape: Tuple[int, ...]) -> None:
        if self.detection_mask is not None:
            return
        polygons_for_mask = (
            {"detection": self.detection_polygon_img}
            if self.detection_polygon_img is not None
            else self.zone_polygons_img
        )
        mask, bbox, cropped = build_union_mask(frame_shape, polygons_for_mask)
        self.detection_mask = mask
        self.detection_bbox = bbox
        self._cropped_detection_mask = cropped

    def _stats_window_origin_abs(self) -> float:
        start_cfg = self.time_config.get("start")
        if self.time_config.get("mode") == "precise":
            return float(start_cfg or self.global_start_time_s)
        return self.global_start_time_s

    def _is_in_stats_window(self, current_time: float) -> bool:
        mode = self.time_config.get("mode", "global")
        start_cfg = self.time_config.get("start")
        end_cfg = self.time_config.get("end")
        if mode == "global":
            return True
        if start_cfg is not None and current_time < start_cfg:
            return False
        if end_cfg is not None and current_time > end_cfg:
            return False
        return True

    def _zone_name_for_point(self, point_px: Optional[Tuple[float, float]]) -> str:
        if point_px is None:
            return "Unknown"
        test_pt = (float(point_px[0]), float(point_px[1]))
        if cv2.pointPolygonTest(self.zone_polygons_img["center"], test_pt, False) >= 0:
            return "center"
        for name in self.ARM_NAMES:
            if cv2.pointPolygonTest(self.zone_polygons_img[name], test_pt, False) >= 0:
                return name
        return "Unknown"

    def _is_inside_entry_zone(self, arm_name: str, point_px: Optional[Tuple[float, float]]) -> bool:
        if point_px is None or arm_name not in self.entry_polygons_img:
            return False
        return cv2.pointPolygonTest(self.entry_polygons_img[arm_name], (float(point_px[0]), float(point_px[1])), False) >= 0

    def _sequence_tail(self, max_items: int = 6) -> str:
        if not self.arm_entry_sequence:
            return ""
        tail = self.arm_entry_sequence[-max_items:]
        return "-".join(self.ARM_LABELS.get(name, name) for name in tail)

    def detect_mouse(self, frame: np.ndarray, gray: np.ndarray, bg_subtractor, morph_processor, detector):
        self.initialize(frame.shape)
        assert self.detection_bbox is not None
        assert self._cropped_detection_mask is not None

        x, y, w, h = self.detection_bbox
        crop_frame = frame[y : y + h, x : x + w]
        crop_gray = gray[y : y + h, x : x + w]

        if getattr(bg_subtractor, "method", "") == "static":
            full_fg_mask = bg_subtractor.apply(frame, gray=gray)
            fg_mask = full_fg_mask[y : y + h, x : x + w]
        else:
            fg_mask = bg_subtractor.apply(crop_frame, gray=crop_gray)
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
    ) -> Dict[str, Any]:
        self.total_frames += 1
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
            self.last_all_centroid_px = centroid_px
            self.prev_centroid_px = centroid_px
            self.missing_frames = 0
            if self.first_detected_time_abs is None:
                self.first_detected_time_abs = float(current_time)
        elif self.last_all_centroid_px is not None and self.missing_frames < self.max_gap_frames:
            centroid_px = self.last_all_centroid_px
            self.missing_frames += 1
        else:
            self.missing_frames += 1

        speed_cm_s = 0.0
        if centroid_px is not None:
            if self.prev_all_centroid_px is not None:
                step_px = float(
                    math.hypot(
                        centroid_px[0] - self.prev_all_centroid_px[0],
                        centroid_px[1] - self.prev_all_centroid_px[1],
                    )
                )
                speed_cm_s = step_px * self.cm_per_px * self.fps
            self.prev_all_centroid_px = centroid_px
        else:
            self.prev_all_centroid_px = None

        zone_key = self._zone_name_for_point(centroid_px)
        zone_group = "arm" if zone_key in self.ARM_NAMES else "center" if zone_key == "center" else "unknown"
        zone_label = self.ZONE_LABELS.get(zone_key, "Unknown")
        self.last_speed_cm_s = speed_cm_s
        in_stats_window = self._is_in_stats_window(float(current_time))

        stats_step_distance_cm = 0.0
        stats_cumulative_distance_cm = self.total_distance_cm
        arm_entry_event = False
        alternation_event = False
        entry_arm_name = None

        if in_stats_window:
            self.frames_in_window += 1
            if detected:
                self.detected_frames_in_window += 1

            if centroid_px is not None:
                if self.prev_stats_centroid_px is not None:
                    step_px = float(
                        math.hypot(
                            centroid_px[0] - self.prev_stats_centroid_px[0],
                            centroid_px[1] - self.prev_stats_centroid_px[1],
                        )
                    )
                    stats_step_distance_cm = step_px * self.cm_per_px
                    self.total_distance_cm += stats_step_distance_cm
                self.prev_stats_centroid_px = centroid_px
                stats_cumulative_distance_cm = self.total_distance_cm
                self.max_speed_cm_s = max(self.max_speed_cm_s, speed_cm_s)

                if zone_key == "center":
                    self.center_frames += 1
                elif zone_key in self.ARM_NAMES:
                    self.arm_frames[zone_key] += 1

                if speed_cm_s >= self.movement_threshold_cm_s:
                    self.moving_frames += 1
                else:
                    self.immobile_frames += 1

                if zone_key in self.ARM_NAMES and self._is_inside_entry_zone(zone_key, centroid_px):
                    if self.current_entry_arm != zone_key:
                        arm_entry_event = True
                        entry_arm_name = zone_key
                        self.current_entry_arm = zone_key
                        self.arm_entries[zone_key] += 1
                        self.total_arm_entries += 1
                        self.arm_entry_sequence.append(zone_key)
                        if len(self.arm_entry_sequence) >= 3:
                            trio = self.arm_entry_sequence[-3:]
                            if len(set(trio)) == 3:
                                self.alternation_count += 1
                                alternation_event = True
                else:
                    self.current_entry_arm = None

                self.trajectory_px_stats.append(centroid_px)
                template_xy = self.project_point_to_template_cm(centroid_px, zone_key)
                if template_xy is not None:
                    self.trajectory_template_cm_stats.append(template_xy)
                self.trajectory_px_live.append(centroid_px)
            else:
                self.prev_stats_centroid_px = None
                self.current_entry_arm = None
                self.immobile_frames += 1
        else:
            self.prev_stats_centroid_px = None
            self.current_entry_arm = None
            if centroid_px is not None:
                self.trajectory_px_live.append(centroid_px)

        x_cm = float(centroid_px[0] * self.cm_per_px) if centroid_px is not None else None
        y_cm = float(centroid_px[1] * self.cm_per_px) if centroid_px is not None else None
        record = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "frame": self.total_frames - 1,
            "time_s": format_seconds(current_time),
            "window_time_s": format_seconds(current_time - self._stats_window_origin_abs()) if in_stats_window else None,
            "detected": bool(detected),
            "in_stats_window": bool(in_stats_window),
            "x_px": format_metric(centroid_px[0]) if centroid_px is not None else None,
            "y_px": format_metric(centroid_px[1]) if centroid_px is not None else None,
            "x_cm": format_metric(x_cm) if x_cm is not None else None,
            "y_cm": format_metric(y_cm) if y_cm is not None else None,
            "speed_cm_s": format_metric(speed_cm_s),
            "step_distance_cm": format_metric(stats_step_distance_cm),
            "cumulative_distance_cm": format_metric(stats_cumulative_distance_cm),
            "zone_key": zone_key,
            "zone": zone_label,
            "zone_group": zone_group,
            "current_arm": zone_key if zone_key in self.ARM_NAMES else None,
            "in_center": bool(zone_key == "center") if in_stats_window else False,
            "arm_entry_event": bool(arm_entry_event),
            "arm_entry_name": entry_arm_name,
            "alternation_event": bool(alternation_event),
            "alternation_count": int(self.alternation_count),
            "recent_entry_sequence": self._sequence_tail(),
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
            "speed_cm_s": speed_cm_s,
            "zone_key": zone_key,
            "zone_label": zone_label,
            "zone_group": zone_group,
            "arm_entry_event": arm_entry_event,
            "entry_arm_name": entry_arm_name,
            "alternation_event": alternation_event,
            "alternation_count": self.alternation_count,
            "recent_entry_sequence": self._sequence_tail(),
            "trajectory_px": list(self.trajectory_px_live),
            "in_stats_window": bool(in_stats_window),
            "record": record,
        }

    def build_summary(self) -> Dict[str, Any]:
        total_time_s = self.frames_in_window / self.fps if self.fps > 0 else 0.0
        center_time_s = self.center_frames / self.fps if self.fps > 0 else 0.0
        moving_time_s = self.moving_frames / self.fps if self.fps > 0 else 0.0
        immobile_time_s = self.immobile_frames / self.fps if self.fps > 0 else 0.0
        mean_speed_cm_s = self.total_distance_cm / max(total_time_s, 1e-6)
        detection_rate = self.detected_frames_in_window / max(self.frames_in_window, 1) * 100.0
        alternation_opportunities = max(self.total_arm_entries - 2, 0)
        alternation_percent = None
        if alternation_opportunities > 0:
            alternation_percent = self.alternation_count / alternation_opportunities * 100.0

        summary: Dict[str, Any] = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "time_mode": self.time_config.get("mode", "global"),
            "time_start_s": self.time_config.get("start"),
            "time_end_s": self.time_config.get("end"),
            "analysis_window_start_s": format_seconds(self._stats_window_origin_abs()),
            "total_time_s": format_seconds(total_time_s),
            "detected_frames": int(self.detected_frames_in_window),
            "detected_rate_percent": round(detection_rate, 2),
            "total_distance_cm": format_metric(self.total_distance_cm),
            "mean_speed_cm_s": format_metric(mean_speed_cm_s),
            "max_speed_cm_s": format_metric(self.max_speed_cm_s),
            "center_time_s": format_seconds(center_time_s),
            "center_percent": round(center_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "moving_time_s": format_seconds(moving_time_s),
            "immobility_time_s": format_seconds(immobile_time_s),
            "total_arm_entries": int(self.total_arm_entries),
            "alternation_count": int(self.alternation_count),
            "alternation_opportunities": int(alternation_opportunities),
            "alternation_percent": round(alternation_percent, 2) if alternation_percent is not None else None,
            "arm_length_cm": format_metric(self.arm_length_cm),
            "arm_width_cm": format_metric(self.arm_width_cm),
            "entry_depth_ratio": round(self.entry_depth_ratio, 3),
            "movement_threshold_cm_s": round(self.movement_threshold_cm_s, 3),
            "entry_sequence": self._sequence_tail(max_items=12),
            "cm_per_px_est": round(self.cm_per_px, 6),
        }
        for arm_name in self.ARM_NAMES:
            label = self.ARM_LABELS[arm_name].lower()
            arm_time_s = self.arm_frames[arm_name] / self.fps if self.fps > 0 else 0.0
            summary[f"arm_{label}_time_s"] = format_seconds(arm_time_s)
            summary[f"arm_{label}_percent"] = round(arm_time_s / max(total_time_s, 1e-6) * 100.0, 2)
            summary[f"arm_{label}_entries"] = int(self.arm_entries[arm_name])
        return summary

    def build_time_bins(self, bin_seconds: float = 60.0) -> List[Dict[str, Any]]:
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
                    "center_frames": 0,
                    "moving_frames": 0,
                    "alternation_count": 0,
                    "arm_entries": {name: 0 for name in self.ARM_NAMES},
                    "arm_frames": {name: 0 for name in self.ARM_NAMES},
                },
            )
            bucket["frames"] += 1
            bucket["distance_cm"] += float(row.get("step_distance_cm") or 0.0)
            bucket["center_frames"] += int(bool(row.get("in_center")))
            bucket["moving_frames"] += int(float(row.get("speed_cm_s") or 0.0) >= self.movement_threshold_cm_s)
            bucket["alternation_count"] += int(bool(row.get("alternation_event")))
            current_arm = row.get("current_arm")
            if current_arm in self.ARM_NAMES:
                bucket["arm_frames"][current_arm] += 1
            entry_arm_name = row.get("arm_entry_name")
            if entry_arm_name in self.ARM_NAMES:
                bucket["arm_entries"][entry_arm_name] += 1

        out: List[Dict[str, Any]] = []
        for idx in sorted(bins):
            bucket = bins[idx]
            frames = max(bucket["frames"], 1)
            duration_s = frames / self.fps
            row: Dict[str, Any] = {
                "roi_id": bucket["roi_id"],
                "roi_name": bucket["roi_name"],
                "bin_index": bucket["bin_index"],
                "start_s": round(bucket["start_s"], 2),
                "end_s": round(bucket["end_s"], 2),
                "duration_s": format_seconds(duration_s),
                "distance_cm": format_metric(bucket["distance_cm"]),
                "mean_speed_cm_s": format_metric(bucket["distance_cm"] / max(duration_s, 1e-6)),
                "center_time_s": format_seconds(bucket["center_frames"] / self.fps),
                "moving_time_s": format_seconds(bucket["moving_frames"] / self.fps),
                "alternation_count": int(bucket["alternation_count"]),
            }
            total_entries = 0
            for arm_name in self.ARM_NAMES:
                label = self.ARM_LABELS[arm_name].lower()
                row[f"arm_{label}_time_s"] = format_seconds(bucket["arm_frames"][arm_name] / self.fps)
                row[f"arm_{label}_entries"] = int(bucket["arm_entries"][arm_name])
                total_entries += int(bucket["arm_entries"][arm_name])
            row["total_arm_entries"] = total_entries
            opportunities = max(total_entries - 2, 0)
            row["alternation_opportunities"] = opportunities
            row["alternation_percent"] = round(bucket["alternation_count"] / opportunities * 100.0, 2) if opportunities > 0 else None
            out.append(row)
        return out


def draw_y_maze_results(
    frame: np.ndarray,
    tracker_results: Sequence[Tuple[YMazeTracker, Dict[str, Any]]],
    inplace: bool = False,
) -> np.ndarray:
    output = frame if inplace else frame.copy()

    for tracker, result in tracker_results:
        if tracker.detection_polygon_img is not None:
            _draw_poly_outline(output, tracker.detection_polygon_img, (244, 180, 0), 2)
        for zone_name, poly in tracker.zone_polygons_img.items():
            group = "center" if zone_name == "center" else "arm"
            color = _zone_color(group, True)
            thickness = 3 if zone_name == result.get("zone_key") else 2
            _draw_poly_outline(output, poly, color, thickness)

        for arm_name in tracker.ARM_NAMES:
            poly = tracker.zone_polygons_img[arm_name]
            anchor = polygon_centroid(poly)
            cv2.putText(
                output,
                tracker.ARM_LABELS[arm_name],
                (int(anchor[0]) - 8, int(anchor[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (15, 23, 42),
                2,
                cv2.LINE_AA,
            )

        contour = result.get("contour")
        centroid = result.get("centroid")
        color = _zone_color(result.get("zone_group", "unknown"), bool(result.get("detected")))
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

        all_points = np.concatenate(list(tracker.zone_polygons_img.values()), axis=0)
        min_x = int(np.min(all_points[:, 0]))
        min_y = int(np.min(all_points[:, 1]))
        label = tracker.roi_name
        zone_text = str(result.get("zone_label") or "Unknown")
        if not result.get("in_stats_window", True):
            zone_text = f"{zone_text} | paused"
        sub = f"{zone_text} | Alt {int(result.get('alternation_count') or 0)}"
        box_w = max(210, int(max(len(label), len(sub)) * 9.2))
        cv2.rectangle(output, (min_x, max(0, min_y - 44)), (min_x + box_w, max(24, min_y + 2)), (8, 15, 28), -1)
        cv2.rectangle(output, (min_x, max(0, min_y - 44)), (min_x + box_w, max(24, min_y + 2)), color, 1)
        cv2.putText(output, label, (min_x + 8, max(16, min_y - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(output, sub, (min_x + 8, max(34, min_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 210, 220), 1, cv2.LINE_AA)

    return output


def render_y_maze_heatmap(
    tracker: YMazeTracker,
    output_path: str,
    pixels_per_cm: int = 12,
) -> Optional[str]:
    if not tracker.trajectory_template_cm_stats:
        return None

    mask, translated, offset_cm = render_canvas_from_cm_polygons(tracker.template_polygons_cm, pixels_per_cm, padding_cm=2.2)
    heat = np.zeros(mask.shape, dtype=np.float32)

    for point_cm in tracker.trajectory_template_cm_stats:
        px, py = _cm_point_to_canvas_px(point_cm, offset_cm, pixels_per_cm, mask.shape)
        cv2.circle(heat, (px, py), max(4, pixels_per_cm // 2), 1.0, thickness=-1)

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(3, pixels_per_cm), sigmaY=max(3, pixels_per_cm))
    if float(heat.max()) <= 0.0:
        return None

    norm = np.clip(heat / heat.max(), 0.0, 1.0)
    image = _paper_blend_heatmap(norm)
    image[mask == 0] = CLASSIC_BG

    for zone_name, poly in translated.items():
        _draw_poly_outline(image, poly, CLASSIC_HEAT_OUTLINE, 2)
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), CLASSIC_BORDER, 1, cv2.LINE_AA)

    if write_image_unicode(output_path, image):
        return output_path
    return None


def render_y_maze_trackmap(
    tracker: YMazeTracker,
    output_path: str,
    pixels_per_cm: int = 12,
) -> Optional[str]:
    if not tracker.trajectory_template_cm_stats:
        return None

    mask, translated, offset_cm = render_canvas_from_cm_polygons(tracker.template_polygons_cm, pixels_per_cm, padding_cm=2.2)
    canvas = np.full((mask.shape[0], mask.shape[1], 3), CLASSIC_BG, dtype=np.uint8)
    canvas[mask == 0] = CLASSIC_BG

    for zone_name, poly in translated.items():
        group = "center" if zone_name == "center" else "arm"
        fill_color = (226, 242, 227) if group == "center" else CLASSIC_GUIDE_FILL
        outline_color = (94, 170, 106) if group == "center" else CLASSIC_GUIDE_OUTLINE
        fill_poly_alpha(canvas, poly, fill_color, alpha=0.82)
        _draw_poly_outline(canvas, poly, outline_color, 2)

    for arm_name in tracker.ARM_NAMES:
        anchor = polygon_centroid(translated[arm_name])
        cv2.putText(
            canvas,
            tracker.ARM_LABELS[arm_name],
            (int(anchor[0]) - 8, int(anchor[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            CLASSIC_TEXT,
            2,
            cv2.LINE_AA,
        )

    points = []
    for point_cm in tracker.trajectory_template_cm_stats:
        points.append(_cm_point_to_canvas_px(point_cm, offset_cm, pixels_per_cm, canvas.shape[:2]))
    path_thickness = classic_path_thickness(pixels_per_cm)
    marker_radius = classic_marker_radius(pixels_per_cm)
    for idx in range(1, len(points)):
        cv2.line(canvas, points[idx - 1], points[idx], CLASSIC_TRACK, path_thickness, cv2.LINE_AA)
    draw_start_end_markers(canvas, points, marker_radius)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), CLASSIC_BORDER, 1, cv2.LINE_AA)

    if write_image_unicode(output_path, canvas):
        return output_path
    return None


def render_y_maze_image_grid(
    image_paths: Sequence[str],
    output_path: str,
    title: str = "",
    columns: int = 2,
    padding: int = 18,
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
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    if title:
        cv2.putText(canvas, title, (padding, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 41, 59), 2, cv2.LINE_AA)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y = title_h + padding + row * (max_h + padding)
        x = padding + col * (max_w + padding)
        h, w = img.shape[:2]
        canvas[y : y + h, x : x + w] = img
        cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), (226, 232, 240), 1, cv2.LINE_AA)

    if write_image_unicode(output_path, canvas):
        return output_path
    return None
