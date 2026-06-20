#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Light-dark box tracking and rendering helpers."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from file_utils import ascii_slug, read_image_unicode, write_image_unicode
from maze_zone_utils import order_quad_by_axes, project_point_homography, render_canvas_from_cm_polygons
from spatial_render import (
    append_heatmap_legend,
    classic_heatmap,
    classic_marker_radius,
    classic_path_thickness,
    draw_start_end_markers,
    fill_poly_alpha,
    style_color,
)


ZONE_ORDER = ("light", "dark")
OPTIONAL_ZONE_ORDER = ("doorway",)
ZONE_LABELS = {
    "light": "Light",
    "dark": "Dark",
    "doorway": "Doorway",
    "Unknown": "Unknown",
}


def _truthy(value: Any, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off", "disabled", "hidden", "invisible"}:
            return False
        if normalized in {"1", "true", "yes", "on", "enabled", "visible"}:
            return True
    return bool(value)


def sanitize_name(text: str, fallback: str = "ldb") -> str:
    return ascii_slug(text, fallback=fallback, max_length=80)


def _coerce_polygon(points: Sequence[Sequence[float]], min_points: int = 3) -> np.ndarray:
    coerced = []
    for point in ([] if points is None else points):
        if isinstance(point, dict):
            coerced.append([float(point["x"]), float(point["y"])])
        else:
            coerced.append([float(point[0]), float(point[1])])
    pts = np.asarray(coerced, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < min_points:
        raise ValueError(f"Light-dark box zone requires at least {min_points} points.")
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)].astype(np.float32)


def _polygon_centroid(poly: np.ndarray) -> np.ndarray:
    moments = cv2.moments(poly.astype(np.float32))
    if abs(moments.get("m00", 0.0)) > 1e-6:
        return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)
    return np.mean(poly.reshape(-1, 2), axis=0).astype(np.float32)


def _polygons_bbox(polygons: Sequence[np.ndarray]) -> Tuple[float, float, float, float]:
    points = np.concatenate([poly.reshape(-1, 2) for poly in polygons], axis=0)
    return float(np.min(points[:, 0])), float(np.min(points[:, 1])), float(np.max(points[:, 0])), float(np.max(points[:, 1]))


def _quad_from_polygon(poly: np.ndarray) -> np.ndarray:
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if pts.shape == (4, 2):
        return pts.astype(np.float32)
    rect = cv2.minAreaRect(pts)
    return _coerce_polygon(cv2.boxPoints(rect), min_points=4)


def _normalize_axis(axis: Sequence[float], fallback: Sequence[float] = (1.0, 0.0)) -> np.ndarray:
    vec = np.asarray(axis, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        vec = np.asarray(fallback, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _segment_point_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    line = end - start
    denom = float(np.dot(line, line))
    if denom <= 1e-9:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, line) / denom, 0.0, 1.0))
    projected = start + line * t
    return float(np.linalg.norm(point - projected))


def _segment_pair_distance(edge_a: Tuple[np.ndarray, np.ndarray], edge_b: Tuple[np.ndarray, np.ndarray]) -> float:
    a0, a1 = edge_a
    b0, b1 = edge_b
    return float((
        _segment_point_distance(a0, b0, b1)
        + _segment_point_distance(a1, b0, b1)
        + _segment_point_distance(b0, a0, a1)
        + _segment_point_distance(b1, a0, a1)
    ) / 4.0)


def _quad_edges(quad: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    pts = _coerce_polygon(quad, min_points=4)
    return [(pts[idx], pts[(idx + 1) % len(pts)]) for idx in range(len(pts))]


def _closest_shared_edges(
    poly_a: np.ndarray,
    poly_b: np.ndarray,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    edges_a = _quad_edges(_quad_from_polygon(poly_a))
    edges_b = _quad_edges(_quad_from_polygon(poly_b))
    best_pair = (edges_a[0], edges_b[0])
    best_distance = float("inf")
    for edge_a in edges_a:
        for edge_b in edges_b:
            distance = _segment_pair_distance(edge_a, edge_b)
            if distance < best_distance:
                best_distance = distance
                best_pair = (edge_a, edge_b)
    return best_pair


def _closest_shared_edge_axis(poly_a: np.ndarray, poly_b: np.ndarray) -> np.ndarray:
    best_pair = _closest_shared_edges(poly_a, poly_b)
    axis_a = best_pair[0][1] - best_pair[0][0]
    axis_b = best_pair[1][1] - best_pair[1][0]
    if float(np.dot(axis_a, axis_b)) < 0.0:
        axis_b = -axis_b
    return _normalize_axis(axis_a + axis_b, fallback=axis_a)


def _cm_point_to_canvas_px(
    point_cm: Tuple[float, float],
    offset_cm: Tuple[float, float],
    pixels_per_cm: int,
    image_shape: Tuple[int, int],
) -> Tuple[int, int]:
    px = int(np.clip(round((float(point_cm[0]) + offset_cm[0]) * float(pixels_per_cm)), 0, image_shape[1] - 1))
    py = int(np.clip(round((float(point_cm[1]) + offset_cm[1]) * float(pixels_per_cm)), 0, image_shape[0] - 1))
    return px, py


def _format_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except Exception:
        return None


def _format_metric(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except Exception:
        return None


def _safe_percent(part: float, total: float) -> float:
    return round(float(part) / max(float(total), 1e-9) * 100.0, 2)


def _positive_float(value: Any, default: float, minimum: float = 1.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if not math.isfinite(parsed) or parsed < minimum:
        parsed = float(default)
    return max(float(minimum), float(parsed))


def _normalize_length_width_for_shared_edge(length_cm: float, width_cm: float, shared_edge_cm: float) -> Tuple[float, float]:
    """Treat length as the long side when the shared edge clearly matches width."""
    length = float(length_cm)
    width = float(width_cm)
    shared = float(shared_edge_cm)
    if length + 1e-6 < width and shared + 1e-6 >= width:
        return width, length
    return length, width


def _dimension_along_shared_edge(
    poly: np.ndarray,
    shared_axis: np.ndarray,
    length_cm: float,
    width_cm: float,
) -> Tuple[float, float, str]:
    quad = _quad_from_polygon(poly)
    axis = _normalize_axis(shared_axis)
    perp = np.array([-axis[1], axis[0]], dtype=np.float32)
    parallel = quad @ axis
    perpendicular = quad @ perp
    parallel_span = float(np.max(parallel) - np.min(parallel))
    perpendicular_span = float(np.max(perpendicular) - np.min(perpendicular))
    if parallel_span >= perpendicular_span:
        return float(length_cm), float(width_cm), "length"
    return float(width_cm), float(length_cm), "width"


def _edge_center_fraction(poly: np.ndarray, edge: Tuple[np.ndarray, np.ndarray], shared_axis: np.ndarray) -> float:
    quad = _quad_from_polygon(poly)
    axis = _normalize_axis(shared_axis)
    projections = quad @ axis
    span = float(np.max(projections) - np.min(projections))
    if span <= 1e-6:
        return 0.5
    edge_center = (float(np.dot(edge[0], axis)) + float(np.dot(edge[1], axis))) / 2.0
    return float(np.clip((edge_center - float(np.min(projections))) / span, 0.0, 1.0))


def _aligned_shared_offsets(
    side_by_zone: Dict[str, float],
    center_fraction_by_zone: Dict[str, float],
    shared_edge_length_cm: float,
) -> Tuple[Dict[str, float], float, float]:
    centers: Dict[str, float] = {}
    half_shared = max(0.05, float(shared_edge_length_cm) / 2.0)
    for zone_name, side_cm in side_by_zone.items():
        side = max(0.1, float(side_cm))
        center_cm = float(center_fraction_by_zone.get(zone_name, 0.5)) * side
        center_cm = float(np.clip(center_cm, min(half_shared, side / 2.0), max(side - half_shared, side / 2.0)))
        centers[zone_name] = center_cm
    common_center = max(centers.values()) if centers else half_shared
    offsets = {zone_name: common_center - center_cm for zone_name, center_cm in centers.items()}
    min_offset = min(offsets.values()) if offsets else 0.0
    if min_offset < 0.0:
        offsets = {zone_name: offset - min_offset for zone_name, offset in offsets.items()}
        common_center -= min_offset
    cross_extent = max((offsets[name] + float(side_by_zone[name]) for name in side_by_zone), default=shared_edge_length_cm)
    return offsets, float(cross_extent), float(common_center)


def _point_in_polygon(poly: Optional[np.ndarray], point: Tuple[float, float]) -> bool:
    if poly is None:
        return False
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def _build_union_mask(
    frame_shape: Tuple[int, ...],
    polygons: Sequence[np.ndarray],
) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32) for poly in polygons], 255)
    x, y, w, h = cv2.boundingRect(np.concatenate([poly.astype(np.int32) for poly in polygons], axis=0))
    x = max(0, int(x))
    y = max(0, int(y))
    w = max(1, min(int(w), frame_shape[1] - x))
    h = max(1, min(int(h), frame_shape[0] - y))
    return mask, (x, y, w, h), mask[y : y + h, x : x + w]


class LightDarkBoxTracker:
    """Track one mouse in one light-dark box ROI."""

    def __init__(
        self,
        zone_polygons: Dict[str, Sequence[Sequence[float]]],
        detection_polygon: Optional[Sequence[Sequence[float]]] = None,
        arena_width_cm: float = 40.0,
        arena_height_cm: float = 40.0,
        light_box_length_cm: Optional[float] = None,
        light_box_width_cm: Optional[float] = None,
        dark_box_length_cm: Optional[float] = None,
        dark_box_width_cm: Optional[float] = None,
        shared_edge_length_cm: Optional[float] = None,
        fps: float = 30.0,
        door_open_delay_s: float = 3.0,
        transition_min_dwell_s: float = 0.12,
        doorway_reset_s: float = 0.2,
        movement_threshold_cm_s: float = 2.0,
        dark_box_visible: bool = True,
        roi_id: int = 1,
        roi_name: Optional[str] = None,
        time_config: Optional[Dict[str, Any]] = None,
        global_start_time_s: float = 0.0,
        subject_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dark_box_visible = _truthy(dark_box_visible, default=True)
        self.locomotion_reliable = bool(self.dark_box_visible)
        required_zones = ZONE_ORDER if self.dark_box_visible else ("light",)
        missing = [name for name in required_zones if name not in (zone_polygons or {})]
        if missing:
            raise ValueError(f"Light-dark box zones are incomplete: missing {', '.join(missing)}")

        self.roi_id = int(roi_id)
        self.roi_name = roi_name or f"LDB {self.roi_id}"
        self.subject_fields = dict(subject_fields or {})
        self.time_config = self._normalize_time_config(time_config)
        self.global_start_time_s = max(0.0, float(global_start_time_s or 0.0))

        self.zone_polygons_img = {name: _coerce_polygon(zone_polygons[name]) for name in required_zones}
        self.doorway_polygon_img = None
        if zone_polygons.get("doorway"):
            self.doorway_polygon_img = _coerce_polygon(zone_polygons["doorway"])
        self.detection_polygon_img = _coerce_polygon(detection_polygon) if detection_polygon is not None else None

        all_zone_polys = list(self.zone_polygons_img.values())
        self.legacy_arena_width_cm = _positive_float(arena_width_cm, 40.0)
        self.legacy_arena_height_cm = _positive_float(arena_height_cm, 40.0)
        square_legacy_default = abs(self.legacy_arena_width_cm - self.legacy_arena_height_cm) <= 1e-6
        default_chamber_length_cm = self.legacy_arena_width_cm if square_legacy_default else self.legacy_arena_width_cm / 2.0
        default_chamber_width_cm = self.legacy_arena_height_cm / 2.0 if square_legacy_default else self.legacy_arena_height_cm
        self.light_box_length_cm = _positive_float(light_box_length_cm, default_chamber_length_cm)
        self.light_box_width_cm = _positive_float(light_box_width_cm, default_chamber_width_cm)
        self.dark_box_length_cm = _positive_float(dark_box_length_cm, default_chamber_length_cm)
        self.dark_box_width_cm = _positive_float(dark_box_width_cm, default_chamber_width_cm)
        default_shared = min(
            max(self.light_box_length_cm, self.light_box_width_cm),
            max(self.dark_box_length_cm, self.dark_box_width_cm),
        )
        self.shared_edge_length_cm = _positive_float(shared_edge_length_cm, default_shared, minimum=0.1)
        self.light_box_length_cm, self.light_box_width_cm = _normalize_length_width_for_shared_edge(
            self.light_box_length_cm,
            self.light_box_width_cm,
            self.shared_edge_length_cm,
        )
        self.dark_box_length_cm, self.dark_box_width_cm = _normalize_length_width_for_shared_edge(
            self.dark_box_length_cm,
            self.dark_box_width_cm,
            self.shared_edge_length_cm,
        )
        self.template_width_cm = self.legacy_arena_width_cm
        self.template_height_cm = self.legacy_arena_height_cm
        self.template_layout = "none"
        self.template_dark_shared_side = None
        self.template_light_shared_side = None
        self.arena_width_cm = self.legacy_arena_width_cm
        self.arena_height_cm = self.legacy_arena_height_cm
        min_x, min_y, max_x, max_y = _polygons_bbox(all_zone_polys)
        px_w = max(1.0, max_x - min_x)
        px_h = max(1.0, max_y - min_y)
        self.cm_per_px = float((self.arena_width_cm / px_w + self.arena_height_cm / px_h) * 0.5)
        self.template_polygons_cm = self._build_template_polygons_cm()
        self._zone_img_to_template_h = self._build_zone_template_homographies()
        if self.template_polygons_cm:
            self.arena_width_cm = max(1.0, float(self.template_width_cm))
            self.arena_height_cm = max(1.0, float(self.template_height_cm))
        self.fps = max(1.0, float(fps))
        self.door_open_delay_s = max(0.0, float(door_open_delay_s))
        self.transition_min_dwell_s = max(0.0, float(transition_min_dwell_s))
        self.doorway_reset_s = max(0.0, float(doorway_reset_s))
        self.movement_threshold_cm_s = max(0.0, float(movement_threshold_cm_s))

        self.detection_mask: Optional[np.ndarray] = None
        self.detection_bbox: Optional[Tuple[int, int, int, int]] = None
        self._cropped_detection_mask: Optional[np.ndarray] = None

        self.total_frames = 0
        self.frames_in_window = 0
        self.detected_frames = 0
        self.detected_frames_in_window = 0
        self.time_in_window_s = 0.0
        self.detected_time_in_window_s = 0.0
        self.zone_time_s = {"light": 0.0, "dark": 0.0, "doorway": 0.0}
        self.zone_frames = {"light": 0, "dark": 0, "doorway": 0}
        self.zone_entries = {"light": 0, "dark": 0}
        self.transition_count = 0
        self.total_distance_cm = 0.0
        self.zone_distance_cm = {"light": 0.0, "dark": 0.0, "doorway": 0.0}
        self.moving_time_s = 0.0
        self.immobile_time_s = 0.0
        self.moving_frames = 0
        self.immobile_frames = 0
        self.max_speed_cm_s = 0.0
        self.last_speed_cm_s = 0.0
        self.first_light_latency_s: Optional[float] = None
        self.started_outside_dark = False

        self.first_detected_time_abs: Optional[float] = None
        self.prev_all_centroid_px: Optional[Tuple[float, float]] = None
        self.last_all_centroid_px: Optional[Tuple[float, float]] = None
        self.prev_stats_centroid_px: Optional[Tuple[float, float]] = None
        self.prev_all_motion_cm: Optional[Tuple[float, float]] = None
        self.prev_stats_motion_cm: Optional[Tuple[float, float]] = None
        self.prev_centroid_px: Optional[Tuple[float, float]] = None
        self.missing_frames = 0
        self.missing_time_s = 0.0
        self.max_gap_frames = 5
        self.max_gap_s = self.max_gap_frames / max(self.fps, 1e-6)

        self.stable_chamber: Optional[str] = None
        self.candidate_chamber: Optional[str] = None
        self.candidate_dwell_s = 0.0
        self.doorway_streak_s = 0.0
        self.initial_stable_assigned = False

        self.trajectory_px_live: deque[Tuple[float, float]] = deque(maxlen=1000)
        self.trajectory_px_stats: List[Tuple[float, float]] = []
        self.trajectory_template_cm_stats: List[Tuple[float, float]] = []
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
        if self.detection_polygon_img is not None:
            self.detection_mask, self.detection_bbox, _ = _build_union_mask(frame_shape, [self.detection_polygon_img])
            if self.doorway_polygon_img is not None:
                cv2.fillPoly(self.detection_mask, [self.doorway_polygon_img.astype(np.int32)], 0)
            x, y, w, h = self.detection_bbox
            self._cropped_detection_mask = self.detection_mask[y : y + h, x : x + w]
            return

        # By default the doorway is an ignore/buffer band, not a detection zone.
        # This avoids dark door seams being selected as the mouse in LDB videos.
        self.detection_mask, self.detection_bbox, _ = _build_union_mask(frame_shape, list(self.zone_polygons_img.values()))
        if self.doorway_polygon_img is not None:
            cv2.fillPoly(self.detection_mask, [self.doorway_polygon_img.astype(np.int32)], 0)
        x, y, w, h = self.detection_bbox
        self._cropped_detection_mask = self.detection_mask[y : y + h, x : x + w]

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

    def _latency_origin_abs(self) -> float:
        return self._stats_window_origin_abs() + self.door_open_delay_s

    def _build_template_polygons_cm(self) -> Dict[str, np.ndarray]:
        if not self.dark_box_visible or "dark" not in self.zone_polygons_img or "light" not in self.zone_polygons_img:
            return {}
        dark_poly = self.zone_polygons_img["dark"]
        light_poly = self.zone_polygons_img["light"]
        dark_center = _polygon_centroid(dark_poly)
        light_center = _polygon_centroid(light_poly)
        light_edge, dark_edge = _closest_shared_edges(light_poly, dark_poly)
        shared_axis = _closest_shared_edge_axis(light_poly, dark_poly)
        center_fraction_by_zone = {
            "light": _edge_center_fraction(light_poly, light_edge, shared_axis),
            "dark": _edge_center_fraction(dark_poly, dark_edge, shared_axis),
        }
        dark_side_cm, dark_depth_cm, dark_side_label = _dimension_along_shared_edge(
            dark_poly,
            shared_axis,
            self.dark_box_length_cm,
            self.dark_box_width_cm,
        )
        light_side_cm, light_depth_cm, light_side_label = _dimension_along_shared_edge(
            light_poly,
            shared_axis,
            self.light_box_length_cm,
            self.light_box_width_cm,
        )
        self.template_dark_shared_side = dark_side_label
        self.template_light_shared_side = light_side_label
        self.shared_edge_length_cm = min(
            max(0.1, float(self.shared_edge_length_cm)),
            max(0.1, float(dark_side_cm)),
            max(0.1, float(light_side_cm)),
        )

        center_delta = light_center - dark_center
        horizontal = abs(float(center_delta[0])) >= abs(float(center_delta[1]))
        template: Dict[str, np.ndarray] = {}
        if horizontal:
            self.template_layout = "horizontal"
            left_zone, right_zone = ("dark", "light") if float(dark_center[0]) <= float(light_center[0]) else ("light", "dark")
            depth_by_zone = {"dark": dark_depth_cm, "light": light_depth_cm}
            side_by_zone = {"dark": dark_side_cm, "light": light_side_cm}
            cross_offsets, total_height, shared_center = _aligned_shared_offsets(
                side_by_zone,
                center_fraction_by_zone,
                self.shared_edge_length_cm,
            )
            left_depth = depth_by_zone[left_zone]
            x_ranges = {
                left_zone: (0.0, left_depth),
                right_zone: (left_depth, left_depth + depth_by_zone[right_zone]),
            }
            for zone_name in ("dark", "light"):
                y0 = cross_offsets[zone_name]
                y1 = y0 + side_by_zone[zone_name]
                x0, x1 = x_ranges[zone_name]
                template[zone_name] = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            self.template_width_cm = float(left_depth + depth_by_zone[right_zone])
            self.template_height_cm = float(total_height)

            right_center = _polygon_centroid(self.zone_polygons_img[right_zone])
            left_center = _polygon_centroid(self.zone_polygons_img[left_zone])
            self._template_source_x_axis = _normalize_axis(right_center - left_center)
            y_axis = np.array([-self._template_source_x_axis[1], self._template_source_x_axis[0]], dtype=np.float32)
            if float(np.dot(y_axis, np.array([0.0, 1.0], dtype=np.float32))) < 0.0:
                y_axis = -y_axis
            self._template_source_y_axis = _normalize_axis(y_axis, fallback=(0.0, 1.0))

            if self.doorway_polygon_img is not None:
                doorway_w = max(0.5, min(self.template_width_cm * 0.08, 3.0))
                y_mid = shared_center
                y0 = y_mid - self.shared_edge_length_cm / 2.0
                y1 = y_mid + self.shared_edge_length_cm / 2.0
                x0 = left_depth - doorway_w / 2.0
                x1 = left_depth + doorway_w / 2.0
                template["doorway"] = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        else:
            self.template_layout = "vertical"
            top_zone, bottom_zone = ("dark", "light") if float(dark_center[1]) <= float(light_center[1]) else ("light", "dark")
            depth_by_zone = {"dark": dark_depth_cm, "light": light_depth_cm}
            side_by_zone = {"dark": dark_side_cm, "light": light_side_cm}
            cross_offsets, total_width, shared_center = _aligned_shared_offsets(
                side_by_zone,
                center_fraction_by_zone,
                self.shared_edge_length_cm,
            )
            top_depth = depth_by_zone[top_zone]
            y_ranges = {
                top_zone: (0.0, top_depth),
                bottom_zone: (top_depth, top_depth + depth_by_zone[bottom_zone]),
            }
            for zone_name in ("dark", "light"):
                x0 = cross_offsets[zone_name]
                x1 = x0 + side_by_zone[zone_name]
                y0, y1 = y_ranges[zone_name]
                template[zone_name] = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            self.template_width_cm = float(total_width)
            self.template_height_cm = float(top_depth + depth_by_zone[bottom_zone])

            top_center = _polygon_centroid(self.zone_polygons_img[top_zone])
            bottom_center = _polygon_centroid(self.zone_polygons_img[bottom_zone])
            self._template_source_y_axis = _normalize_axis(bottom_center - top_center, fallback=(0.0, 1.0))
            x_axis = np.array([self._template_source_y_axis[1], -self._template_source_y_axis[0]], dtype=np.float32)
            if float(np.dot(x_axis, np.array([1.0, 0.0], dtype=np.float32))) < 0.0:
                x_axis = -x_axis
            self._template_source_x_axis = _normalize_axis(x_axis, fallback=(1.0, 0.0))

            if self.doorway_polygon_img is not None:
                doorway_h = max(0.5, min(self.template_height_cm * 0.08, 3.0))
                x_mid = shared_center
                x0 = x_mid - self.shared_edge_length_cm / 2.0
                x1 = x_mid + self.shared_edge_length_cm / 2.0
                y0 = top_depth - doorway_h / 2.0
                y1 = top_depth + doorway_h / 2.0
                template["doorway"] = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        return template

    def _build_zone_template_homographies(self) -> Dict[str, np.ndarray]:
        if not self.template_polygons_cm:
            return {}
        x_axis = getattr(self, "_template_source_x_axis", None)
        y_axis = getattr(self, "_template_source_y_axis", None)
        if x_axis is None or y_axis is None:
            dark_center = _polygon_centroid(self.zone_polygons_img["dark"])
            light_center = _polygon_centroid(self.zone_polygons_img["light"])
            x_axis = light_center - dark_center
            y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float32)
        x_axis = _normalize_axis(x_axis)
        y_axis = _normalize_axis(y_axis, fallback=(-x_axis[1], x_axis[0]))
        transforms: Dict[str, np.ndarray] = {}
        for zone_name in ("dark", "light"):
            src_quad = order_quad_by_axes(_quad_from_polygon(self.zone_polygons_img[zone_name]), x_axis, y_axis)
            dst_quad = self.template_polygons_cm[zone_name].astype(np.float32)
            transforms[zone_name] = cv2.getPerspectiveTransform(src_quad, dst_quad)
        return transforms

    def _zone_name_for_point(self, point_px: Optional[Tuple[float, float]]) -> str:
        if point_px is None:
            return "dark" if not self.dark_box_visible else "Unknown"
        if _point_in_polygon(self.doorway_polygon_img, point_px):
            return "doorway"
        if _point_in_polygon(self.zone_polygons_img["light"], point_px):
            return "light"
        if self.dark_box_visible and _point_in_polygon(self.zone_polygons_img.get("dark"), point_px):
            return "dark"
        return "Unknown"

    def _point_to_local_cm(self, point_px: Optional[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
        if point_px is None:
            return None, None
        min_x, min_y, _, _ = _polygons_bbox(list(self.zone_polygons_img.values()))
        return (float(point_px[0] - min_x) * self.cm_per_px, float(point_px[1] - min_y) * self.cm_per_px)

    def project_point_to_template_cm(
        self,
        point_px: Optional[Tuple[float, float]],
        zone_key: Optional[str] = None,
    ) -> Optional[Tuple[float, float]]:
        if point_px is None or not self._zone_img_to_template_h:
            return None
        zone = zone_key if zone_key in self._zone_img_to_template_h else None
        if zone is None:
            best_zone = None
            best_score = -float("inf")
            for candidate in ("dark", "light"):
                poly = self.zone_polygons_img.get(candidate)
                if poly is None:
                    continue
                score = float(cv2.pointPolygonTest(poly, (float(point_px[0]), float(point_px[1])), True))
                if score > best_score:
                    best_score = score
                    best_zone = candidate
            zone = best_zone
        if zone not in self._zone_img_to_template_h:
            return None
        x_cm, y_cm = project_point_homography(point_px, self._zone_img_to_template_h[zone])
        return (
            float(np.clip(x_cm, 0.0, self.arena_width_cm)),
            float(np.clip(y_cm, 0.0, self.arena_height_cm)),
        )

    def _motion_point_cm(
        self,
        point_px: Optional[Tuple[float, float]],
        zone_key: Optional[str] = None,
    ) -> Optional[Tuple[float, float]]:
        if point_px is None or not self.locomotion_reliable:
            return None
        projected = self.project_point_to_template_cm(point_px, zone_key)
        if projected is not None:
            return projected
        x_cm, y_cm = self._point_to_local_cm(point_px)
        if x_cm is None or y_cm is None:
            return None
        return float(x_cm), float(y_cm)

    def _doorway_bridge_detection(
        self,
        fg_mask: np.ndarray,
        detector,
        crop_offset: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        if not self.dark_box_visible or self.doorway_polygon_img is None:
            return None

        offset_x, offset_y = crop_offset
        doorway_local = self.doorway_polygon_img.astype(np.float32) - np.array([[offset_x, offset_y]], dtype=np.float32)
        if doorway_local.size == 0:
            return None

        min_area = float(getattr(detector, "min_area", 100.0))
        max_area = float(getattr(detector, "max_area", 50000.0))
        min_piece_area = max(8.0, min_area * 0.25)

        bridge_mask = np.zeros_like(fg_mask, dtype=np.uint8)
        cv2.fillPoly(bridge_mask, [doorway_local.astype(np.int32)], 255)
        _dx, _dy, door_w, door_h = cv2.boundingRect(doorway_local.astype(np.int32))
        bridge_radius = int(max(1, min(6, round(max(1, min(door_w, door_h)) * 0.35))))
        kernel_size = bridge_radius * 2 + 1
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        bridge_mask = cv2.dilate(bridge_mask, bridge_kernel, iterations=1)

        grouped_mask = cv2.bitwise_or(fg_mask, bridge_mask)
        group_contours, _ = cv2.findContours(grouped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Dict[str, Any]] = []

        for group_contour in group_contours:
            group_fill = np.zeros_like(fg_mask, dtype=np.uint8)
            cv2.drawContours(group_fill, [group_contour], -1, 255, thickness=-1)
            raw_group = cv2.bitwise_and(fg_mask, group_fill)
            piece_contours, _ = cv2.findContours(raw_group, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            kept_pieces = []
            zones = set()
            raw_area = 0.0
            for contour in piece_contours:
                area = float(cv2.contourArea(contour))
                if area < min_piece_area:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] == 0:
                    continue
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
                zone = self._zone_name_for_point((cx + offset_x, cy + offset_y))
                if zone not in {"light", "dark"}:
                    continue
                zones.add(zone)
                kept_pieces.append(contour)
                raw_area += area

            if len(kept_pieces) < 2 or not {"light", "dark"}.issubset(zones):
                continue
            if raw_area < min_area or raw_area > max_area:
                continue

            raw_moments = cv2.moments(raw_group)
            if raw_moments["m00"] == 0:
                continue
            points = np.vstack(kept_pieces)
            hull = cv2.convexHull(points)
            hull_area = float(cv2.contourArea(hull))
            x0, y0, bw, bh = cv2.boundingRect(points)
            candidates.append({
                "contour": hull,
                "centroid": (
                    float(raw_moments["m10"] / raw_moments["m00"]),
                    float(raw_moments["m01"] / raw_moments["m00"]),
                ),
                "area": raw_area,
                "bbox": (int(x0), int(y0), int(bw), int(bh)),
                "solidity": raw_area / hull_area if hull_area > 0 else 1.0,
                "doorway_bridge": True,
                "bridge_components": len(kept_pieces),
            })

        if not candidates:
            return None
        return max(candidates, key=lambda item: item.get("area", 0.0))

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
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=self._cropped_detection_mask)
        detections = detector.detect(fg_mask)
        bridge_detection = self._doorway_bridge_detection(fg_mask, detector, (x, y))
        if bridge_detection is not None:
            if detections:
                largest_area = max(float(item.get("area", 0.0)) for item in detections)
                min_area = float(getattr(detector, "min_area", 100.0))
                plausible_bridge_area = max(largest_area * 3.0, largest_area + min_area * 2.0)
                if float(bridge_detection.get("area", 0.0)) <= plausible_bridge_area:
                    detections.append(bridge_detection)
            else:
                detections.append(bridge_detection)
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

    def _update_stable_chamber(
        self,
        zone_key: str,
        current_time: float,
        current_dt: float,
        *,
        detected: bool,
        in_stats_window: bool,
    ) -> Tuple[bool, Optional[str]]:
        if not in_stats_window:
            self.candidate_chamber = None
            self.candidate_dwell_s = 0.0
            self.doorway_streak_s = 0.0
            return False, None

        if zone_key == "doorway":
            self.doorway_streak_s += current_dt
            if self.doorway_streak_s >= self.doorway_reset_s:
                self.candidate_chamber = None
                self.candidate_dwell_s = 0.0
            return False, None
        self.doorway_streak_s = 0.0

        if zone_key not in {"light", "dark"}:
            self.candidate_chamber = None
            self.candidate_dwell_s = 0.0
            return False, None

        if self.candidate_chamber == zone_key:
            self.candidate_dwell_s += current_dt
        else:
            self.candidate_chamber = zone_key
            self.candidate_dwell_s = current_dt

        if self.candidate_dwell_s + 1e-9 < self.transition_min_dwell_s:
            return False, None
        if self.stable_chamber == zone_key:
            return False, None
        if not detected:
            return False, None

        previous = self.stable_chamber
        self.stable_chamber = zone_key
        behavior_started = current_time >= self._latency_origin_abs()
        if not self.initial_stable_assigned:
            self.initial_stable_assigned = True
            if zone_key != "dark":
                self.started_outside_dark = True
            if behavior_started and zone_key == "light":
                self.zone_entries["light"] += 1
                if self.first_light_latency_s is None:
                    self.first_light_latency_s = max(0.0, current_time - self._latency_origin_abs())
                return True, zone_key
            return False, zone_key

        if previous and previous != zone_key and behavior_started:
            self.transition_count += 1
            self.zone_entries[zone_key] += 1
            if zone_key == "light" and self.first_light_latency_s is None:
                self.first_light_latency_s = max(0.0, current_time - self._latency_origin_abs())
            return True, zone_key
        return False, zone_key

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
            current_dt = float(dt_s) if dt_s is not None else 1.0 / max(self.fps, 1e-6)
        except Exception:
            current_dt = 1.0 / max(self.fps, 1e-6)
        current_dt = max(current_dt, 1e-6)

        detection, fg_mask = self.detect_mouse(frame, gray, bg_subtractor, morph_processor, detector)
        centroid_px: Optional[Tuple[float, float]] = None
        bbox = None
        contour = None
        detected = detection is not None
        carried = False
        if detection is not None:
            self.detected_frames += 1
            centroid_px = tuple(float(v) for v in detection["centroid"])
            bbox = detection.get("bbox")
            contour = detection.get("contour")
            self.last_all_centroid_px = centroid_px
            self.prev_centroid_px = centroid_px
            self.missing_frames = 0
            self.missing_time_s = 0.0
            if self.first_detected_time_abs is None:
                self.first_detected_time_abs = float(current_time)
        elif self.dark_box_visible and self.last_all_centroid_px is not None and self.missing_time_s < self.max_gap_s:
            centroid_px = self.last_all_centroid_px
            carried = True
            self.missing_frames += 1
            self.missing_time_s += current_dt
        else:
            self.missing_frames += 1
            self.missing_time_s += current_dt

        zone_key = self._zone_name_for_point(centroid_px)
        zone_label = ZONE_LABELS.get(zone_key, "Unknown")
        motion_point_cm = self._motion_point_cm(centroid_px, zone_key)

        speed_cm_s = 0.0
        if self.locomotion_reliable and motion_point_cm is not None:
            if self.prev_all_motion_cm is not None:
                speed_cm_s = float(math.hypot(
                    motion_point_cm[0] - self.prev_all_motion_cm[0],
                    motion_point_cm[1] - self.prev_all_motion_cm[1],
                )) / current_dt
            self.prev_all_motion_cm = motion_point_cm
        else:
            self.prev_all_motion_cm = None
        if centroid_px is not None:
            self.prev_all_centroid_px = centroid_px
        else:
            self.prev_all_centroid_px = None
        self.last_speed_cm_s = speed_cm_s if self.locomotion_reliable else 0.0

        in_stats_window = self._is_in_stats_window(float(current_time))
        chamber_evidence = bool(detected and not carried) or (not self.dark_box_visible and zone_key == "dark" and not detected)
        transition_event, entry_chamber = self._update_stable_chamber(
            zone_key,
            float(current_time),
            current_dt,
            detected=chamber_evidence,
            in_stats_window=in_stats_window,
        )

        stats_step_distance_cm = 0.0
        if in_stats_window:
            self.frames_in_window += 1
            self.time_in_window_s += current_dt
            if detected:
                self.detected_frames_in_window += 1
                self.detected_time_in_window_s += current_dt
            if zone_key in self.zone_time_s:
                self.zone_time_s[zone_key] += current_dt
                self.zone_frames[zone_key] += 1
            if self.locomotion_reliable and motion_point_cm is not None:
                if self.prev_stats_motion_cm is not None:
                    stats_step_distance_cm = float(math.hypot(
                        motion_point_cm[0] - self.prev_stats_motion_cm[0],
                        motion_point_cm[1] - self.prev_stats_motion_cm[1],
                    ))
                    self.total_distance_cm += stats_step_distance_cm
                    if zone_key in self.zone_distance_cm:
                        self.zone_distance_cm[zone_key] += stats_step_distance_cm
                self.prev_stats_motion_cm = motion_point_cm
                self.prev_stats_centroid_px = centroid_px
                self.max_speed_cm_s = max(self.max_speed_cm_s, speed_cm_s)
                if speed_cm_s >= self.movement_threshold_cm_s:
                    self.moving_frames += 1
                    self.moving_time_s += current_dt
                else:
                    self.immobile_frames += 1
                    self.immobile_time_s += current_dt
                self.trajectory_px_live.append(centroid_px)
                self.trajectory_px_stats.append(centroid_px)
                template_xy = self.project_point_to_template_cm(centroid_px, zone_key)
                if template_xy is not None:
                    self.trajectory_template_cm_stats.append(template_xy)
            else:
                self.prev_stats_centroid_px = None
                self.prev_stats_motion_cm = None
                if self.locomotion_reliable:
                    self.immobile_frames += 1
                    self.immobile_time_s += current_dt
        else:
            self.prev_stats_centroid_px = None
            self.prev_stats_motion_cm = None
            self.stable_chamber = None
            self.candidate_chamber = None
            self.candidate_dwell_s = 0.0
            if self.locomotion_reliable and centroid_px is not None:
                self.trajectory_px_live.append(centroid_px)

        if self.locomotion_reliable and motion_point_cm is not None:
            x_cm, y_cm = motion_point_cm
        else:
            x_cm, y_cm = None, None
        record = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            **self.subject_fields,
            "frame": self.total_frames - 1,
            "time_s": _format_seconds(current_time),
            "dt_s": _format_seconds(current_dt),
            "window_time_s": _format_seconds(current_time - self._stats_window_origin_abs()) if in_stats_window else None,
            "dark_box_visible": bool(self.dark_box_visible),
            "tracking_reliable": bool(self.locomotion_reliable),
            "detected": bool(detected),
            "carried_position": bool(carried),
            "in_stats_window": bool(in_stats_window),
            "x_px": _format_metric(centroid_px[0]) if self.locomotion_reliable and centroid_px is not None else None,
            "y_px": _format_metric(centroid_px[1]) if self.locomotion_reliable and centroid_px is not None else None,
            "x_cm": _format_metric(x_cm),
            "y_cm": _format_metric(y_cm),
            "speed_cm_s": _format_metric(speed_cm_s) if self.locomotion_reliable else None,
            "step_distance_cm": _format_metric(stats_step_distance_cm) if self.locomotion_reliable else None,
            "cumulative_distance_cm": _format_metric(self.total_distance_cm) if self.locomotion_reliable else None,
            "zone_key": zone_key,
            "zone": zone_label,
            "stable_chamber": self.stable_chamber,
            "transition_event": bool(transition_event),
            "entry_chamber": entry_chamber,
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
            "speed_cm_s": speed_cm_s if self.locomotion_reliable else None,
            "step_distance_cm": stats_step_distance_cm if self.locomotion_reliable else None,
            "cumulative_distance_cm": self.total_distance_cm if self.locomotion_reliable else None,
            "zone_key": zone_key,
            "zone_label": zone_label,
            "stable_chamber": self.stable_chamber,
            "transition_event": transition_event,
            "trajectory_px": list(self.trajectory_px_live) if self.locomotion_reliable else [],
            "in_stats_window": bool(in_stats_window),
            "record": record,
        }

    def build_summary(self) -> Dict[str, Any]:
        total_time_s = self.time_in_window_s or (self.frames_in_window / self.fps if self.fps > 0 else 0.0)
        visible_detection_rate = (self.detected_time_in_window_s / max(total_time_s, 1e-9) * 100.0) if total_time_s > 0 else 0.0
        detection_rate = visible_detection_rate if self.locomotion_reliable else None
        mean_speed_cm_s = self.total_distance_cm / max(total_time_s, 1e-6) if self.locomotion_reliable else None
        qc_flags = []
        if not self.dark_box_visible:
            qc_flags.append("dark_box_invisible")
            qc_flags.append("locomotion_unreliable")
        if self.locomotion_reliable and detection_rate is not None and detection_rate < 80.0:
            qc_flags.append("low_detection")
        if self.started_outside_dark:
            qc_flags.append("started_outside_dark")
        if self.transition_count == 0 and total_time_s >= 60.0:
            qc_flags.append("no_transitions")
        summary = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            **self.subject_fields,
            "time_mode": self.time_config.get("mode", "global"),
            "time_start_s": self.time_config.get("start"),
            "time_end_s": self.time_config.get("end"),
            "analysis_window_start_s": _format_seconds(self._stats_window_origin_abs()),
            "door_open_delay_s": _format_seconds(self.door_open_delay_s),
            "dark_box_visible": bool(self.dark_box_visible),
            "tracking_reliable": bool(self.locomotion_reliable),
            "total_time_s": _format_seconds(total_time_s),
            "detected_frames": int(self.detected_frames_in_window),
            "detected_rate_percent": round(detection_rate, 2) if detection_rate is not None else None,
            "visible_light_detection_rate_percent": round(visible_detection_rate, 2),
            "total_distance_cm": _format_metric(self.total_distance_cm) if self.locomotion_reliable else None,
            "mean_speed_cm_s": _format_metric(mean_speed_cm_s),
            "max_speed_cm_s": _format_metric(self.max_speed_cm_s) if self.locomotion_reliable else None,
            "light_time_s": _format_seconds(self.zone_time_s["light"]),
            "dark_time_s": _format_seconds(self.zone_time_s["dark"]),
            "doorway_time_s": _format_seconds(self.zone_time_s["doorway"]) if self.dark_box_visible else None,
            "light_time_percent": _safe_percent(self.zone_time_s["light"], total_time_s),
            "dark_time_percent": _safe_percent(self.zone_time_s["dark"], total_time_s),
            "doorway_time_percent": _safe_percent(self.zone_time_s["doorway"], total_time_s) if self.dark_box_visible else None,
            "light_entries": int(self.zone_entries["light"]),
            "dark_entries": int(self.zone_entries["dark"]),
            "total_transitions": int(self.transition_count),
            "first_light_latency_s": _format_seconds(self.first_light_latency_s),
            "light_distance_cm": _format_metric(self.zone_distance_cm["light"]) if self.locomotion_reliable else None,
            "dark_distance_cm": _format_metric(self.zone_distance_cm["dark"]) if self.locomotion_reliable else None,
            "doorway_distance_cm": _format_metric(self.zone_distance_cm["doorway"]) if self.locomotion_reliable and self.dark_box_visible else None,
            "moving_time_s": _format_seconds(self.moving_time_s) if self.locomotion_reliable else None,
            "immobility_time_s": _format_seconds(self.immobile_time_s) if self.locomotion_reliable else None,
            "transition_min_dwell_s": _format_seconds(self.transition_min_dwell_s),
            "doorway_reset_s": _format_seconds(self.doorway_reset_s) if self.dark_box_visible else None,
            "movement_threshold_cm_s": _format_metric(self.movement_threshold_cm_s) if self.locomotion_reliable else None,
            "light_box_length_cm": _format_metric(self.light_box_length_cm) if self.locomotion_reliable else None,
            "light_box_width_cm": _format_metric(self.light_box_width_cm) if self.locomotion_reliable else None,
            "dark_box_length_cm": _format_metric(self.dark_box_length_cm) if self.locomotion_reliable else None,
            "dark_box_width_cm": _format_metric(self.dark_box_width_cm) if self.locomotion_reliable else None,
            "shared_edge_length_cm": _format_metric(self.shared_edge_length_cm) if self.locomotion_reliable else None,
            "template_layout": self.template_layout if self.locomotion_reliable else None,
            "light_shared_side": self.template_light_shared_side if self.locomotion_reliable else None,
            "dark_shared_side": self.template_dark_shared_side if self.locomotion_reliable else None,
            "arena_width_cm": _format_metric(self.arena_width_cm) if self.locomotion_reliable else None,
            "arena_height_cm": _format_metric(self.arena_height_cm) if self.locomotion_reliable else None,
            "cm_per_px_est": round(self.cm_per_px, 6) if self.locomotion_reliable else None,
            "started_outside_dark": bool(self.started_outside_dark),
            "qc_flags": ";".join(qc_flags),
        }
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
                    **self.subject_fields,
                    "bin_index": idx + 1,
                    "start_s": idx * bin_seconds,
                    "end_s": (idx + 1) * bin_seconds,
                    "duration_s": 0.0,
                    "distance_cm": 0.0,
                    "light_time_s": 0.0,
                    "dark_time_s": 0.0,
                    "doorway_time_s": 0.0,
                    "transitions": 0,
                },
            )
            duration = float(row.get("dt_s") or (1.0 / max(self.fps, 1e-6)))
            zone_key = row.get("zone_key")
            bucket["duration_s"] += duration
            if self.locomotion_reliable:
                bucket["distance_cm"] += float(row.get("step_distance_cm") or 0.0)
            if zone_key in {"light", "dark", "doorway"}:
                bucket[f"{zone_key}_time_s"] += duration
            bucket["transitions"] += int(bool(row.get("transition_event")))
        out: List[Dict[str, Any]] = []
        for idx in sorted(bins):
            bucket = bins[idx]
            duration_s = float(bucket.get("duration_s") or 0.0)
            out.append({
                **bucket,
                "start_s": round(bucket["start_s"], 2),
                "end_s": round(bucket["end_s"], 2),
                "duration_s": _format_seconds(duration_s),
                "distance_cm": _format_metric(bucket["distance_cm"]) if self.locomotion_reliable else None,
                "mean_speed_cm_s": _format_metric(bucket["distance_cm"] / max(duration_s, 1e-6)) if self.locomotion_reliable else None,
                "light_time_s": _format_seconds(bucket["light_time_s"]),
                "dark_time_s": _format_seconds(bucket["dark_time_s"]),
                "doorway_time_s": _format_seconds(bucket["doorway_time_s"]) if self.dark_box_visible else None,
                "transitions": int(bucket["transitions"]),
            })
        return out


def _zone_color(zone_key: str, detected: bool = True) -> Tuple[int, int, int]:
    if not detected:
        return (120, 120, 120)
    if zone_key == "light":
        return (245, 158, 11)
    if zone_key == "dark":
        return (59, 130, 246)
    if zone_key == "doorway":
        return (34, 197, 94)
    return (148, 163, 184)


def draw_light_dark_box_results(
    frame: np.ndarray,
    tracker_results: Sequence[Tuple[LightDarkBoxTracker, Dict[str, Any]]],
    inplace: bool = False,
) -> np.ndarray:
    output = frame if inplace else frame.copy()
    for tracker, result in tracker_results:
        if tracker.detection_polygon_img is not None:
            cv2.polylines(output, [tracker.detection_polygon_img.astype(np.int32)], True, (255, 214, 10), 1, cv2.LINE_AA)
        for zone_name, poly in tracker.zone_polygons_img.items():
            fill_poly_alpha(output, poly.astype(np.int32), _zone_color(zone_name), alpha=0.12)
            cv2.polylines(output, [poly.astype(np.int32)], True, _zone_color(zone_name), 2, cv2.LINE_AA)
            anchor = _polygon_centroid(poly)
            cv2.putText(output, ZONE_LABELS[zone_name], (int(anchor[0]) - 24, int(anchor[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2, cv2.LINE_AA)
        if tracker.doorway_polygon_img is not None:
            fill_poly_alpha(output, tracker.doorway_polygon_img.astype(np.int32), _zone_color("doorway"), alpha=0.18)
            cv2.polylines(output, [tracker.doorway_polygon_img.astype(np.int32)], True, _zone_color("doorway"), 2, cv2.LINE_AA)

        color = _zone_color(str(result.get("zone_key") or "Unknown"), bool(result.get("detected")))
        contour = result.get("contour")
        centroid = result.get("centroid")
        if contour is not None:
            cv2.drawContours(output, [contour], -1, color, 2)
        if centroid is not None:
            cv2.circle(output, (int(centroid[0]), int(centroid[1])), 4, color, -1)
        traj = result.get("trajectory_px") or []
        for idx in range(1, len(traj)):
            cv2.line(output, (int(traj[idx - 1][0]), int(traj[idx - 1][1])), (int(traj[idx][0]), int(traj[idx][1])), color, 1, cv2.LINE_AA)

        all_points = np.concatenate(list(tracker.zone_polygons_img.values()), axis=0)
        min_x = int(np.min(all_points[:, 0]))
        min_y = int(np.min(all_points[:, 1]))
        label = tracker.roi_name
        zone_text = str(result.get("zone_label") or "Unknown")
        if not result.get("in_stats_window", True):
            zone_text = f"{zone_text} | paused"
        sub = f"{zone_text} | T {tracker.transition_count}"
        box_w = max(190, int(max(len(label), len(sub)) * 9.2))
        cv2.rectangle(output, (min_x, max(0, min_y - 44)), (min_x + box_w, max(24, min_y + 2)), (8, 15, 28), -1)
        cv2.rectangle(output, (min_x, max(0, min_y - 44)), (min_x + box_w, max(24, min_y + 2)), color, 1)
        cv2.putText(output, label, (min_x + 8, max(16, min_y - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(output, sub, (min_x + 8, max(34, min_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 210, 220), 1, cv2.LINE_AA)
    return output


def _render_tracker_canvas(
    tracker: LightDarkBoxTracker,
    padding: int = 24,
    style: Any | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[float, float]]:
    polys = list(tracker.zone_polygons_img.values())
    if tracker.doorway_polygon_img is not None:
        polys.append(tracker.doorway_polygon_img)
    min_x, min_y, max_x, max_y = _polygons_bbox(polys)
    width = max(180, int(max_x - min_x + padding * 2))
    height = max(140, int(max_y - min_y + padding * 2))
    offset = np.array([padding - min_x, padding - min_y], dtype=np.float32)
    translated = {name: (poly + offset).astype(np.int32) for name, poly in tracker.zone_polygons_img.items()}
    if tracker.doorway_polygon_img is not None:
        translated["doorway"] = (tracker.doorway_polygon_img + offset).astype(np.int32)
    canvas = np.full((height, width, 3), style_color(style, "trackmap", "background"), dtype=np.uint8)
    return canvas, translated, (float(offset[0]), float(offset[1]))


def _render_tracker_template_canvas(
    tracker: LightDarkBoxTracker,
    pixels_per_cm: int,
    style: Any | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[float, float]]:
    drawable_polygons = {
        name: tracker.template_polygons_cm[name]
        for name in ("dark", "light")
        if name in tracker.template_polygons_cm
    }
    mask, translated, offset_cm = render_canvas_from_cm_polygons(
        drawable_polygons,
        pixels_per_cm,
        padding_cm=2.0,
        min_size_px=200,
    )
    canvas = np.full((mask.shape[0], mask.shape[1], 3), style_color(style, "trackmap", "background"), dtype=np.uint8)
    return canvas, translated, offset_cm


def render_light_dark_box_heatmap(
    tracker: LightDarkBoxTracker,
    output_path: str,
    pixels_per_cm: int = 12,
    style: Any | None = None,
) -> Optional[str]:
    if not tracker.locomotion_reliable or not tracker.trajectory_template_cm_stats or not tracker.template_polygons_cm:
        return None
    canvas, translated, offset_cm = _render_tracker_template_canvas(tracker, pixels_per_cm, style)
    heat = np.zeros(canvas.shape[:2], dtype=np.float32)
    for point_cm in tracker.trajectory_template_cm_stats:
        px, py = _cm_point_to_canvas_px(point_cm, offset_cm, pixels_per_cm, heat.shape)
        cv2.circle(heat, (px, py), max(4, pixels_per_cm // 2), 1.0, thickness=-1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(3, pixels_per_cm), sigmaY=max(3, pixels_per_cm))
    if float(heat.max()) <= 0.0:
        return None
    color = classic_heatmap(np.clip(heat / heat.max(), 0.0, 1.0), style)
    for poly in translated.values():
        cv2.polylines(color, [poly], True, style_color(style, "heatmap", "outline"), 2, cv2.LINE_AA)
    cv2.rectangle(color, (0, 0), (color.shape[1] - 1, color.shape[0] - 1), style_color(style, "heatmap", "border"), 1, cv2.LINE_AA)
    color = append_heatmap_legend(color, style)
    if write_image_unicode(output_path, color):
        return output_path
    return None


def render_light_dark_box_trackmap(
    tracker: LightDarkBoxTracker,
    output_path: str,
    pixels_per_cm: int = 12,
    style: Any | None = None,
) -> Optional[str]:
    if not tracker.locomotion_reliable or not tracker.trajectory_template_cm_stats or not tracker.template_polygons_cm:
        return None
    canvas, translated, offset_cm = _render_tracker_template_canvas(tracker, pixels_per_cm, style)
    for name, poly in translated.items():
        fill_color = style_color(style, "trackmap", "region_fill") if name != "dark" else style_color(style, "trackmap", "dark_region_fill")
        fill_poly_alpha(canvas, poly, fill_color, alpha=0.88)
        cv2.polylines(canvas, [poly], True, style_color(style, "trackmap", "region_outline"), 2, cv2.LINE_AA)
    points = []
    for point_cm in tracker.trajectory_template_cm_stats:
        points.append(_cm_point_to_canvas_px(point_cm, offset_cm, pixels_per_cm, canvas.shape[:2]))
    thickness = classic_path_thickness(pixels_per_cm)
    for idx in range(1, len(points)):
        cv2.line(canvas, points[idx - 1], points[idx], style_color(style, "trackmap", "track"), thickness, cv2.LINE_AA)
    draw_start_end_markers(canvas, points, classic_marker_radius(pixels_per_cm), style)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), style_color(style, "trackmap", "border"), 1, cv2.LINE_AA)
    if write_image_unicode(output_path, canvas):
        return output_path
    return None


def render_light_dark_box_image_grid(
    image_paths: Sequence[str],
    output_path: str,
    title: str = "",
    columns: int = 2,
    padding: int = 18,
    style: Any | None = None,
) -> Optional[str]:
    images = [read_image_unicode(path) for path in image_paths if path]
    images = [img for img in images if img is not None]
    if not images:
        return None
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    cols = max(1, min(int(columns), len(images)))
    rows = int(math.ceil(len(images) / cols))
    title_h = 46 if title else 0
    canvas = np.full((title_h + rows * max_h + (rows + 1) * padding, cols * max_w + (cols + 1) * padding, 3), style_color(style, "trackmap", "grid_background"), dtype=np.uint8)
    if title:
        cv2.putText(canvas, title, (padding, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, style_color(style, "trackmap", "grid_title"), 2, cv2.LINE_AA)
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y = title_h + padding + row * (max_h + padding)
        x = padding + col * (max_w + padding)
        canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
        cv2.rectangle(canvas, (x, y), (x + img.shape[1], y + img.shape[0]), style_color(style, "trackmap", "grid_border"), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), style_color(style, "trackmap", "grid_border"), 1, cv2.LINE_AA)
    if write_image_unicode(output_path, canvas):
        return output_path
    return None
