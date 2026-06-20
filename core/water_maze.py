#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Morris water maze geometry, tracking, metrics, and visual outputs."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from file_utils import ascii_slug, read_image_unicode, write_image_unicode
from spatial_render import (
    append_heatmap_legend,
    classic_heatmap,
    classic_marker_radius,
    classic_path_thickness,
    draw_start_end_markers,
    style_color,
)


Point = Tuple[float, float]
QUADRANTS = ("NE", "NW", "SW", "SE")
QUADRANT_DISPLAY_LABELS = {
    "NE": "I",
    "NW": "II",
    "SW": "III",
    "SE": "IV",
    "Unknown": "Unknown",
}
SEARCH_STRATEGY_LABELS = {
    "direct_path": "Direct Path",
    "directed_search": "Directed Search",
    "focal_search": "Focal Search",
    "indirect_search": "Indirect Search",
    "chaining": "Chaining",
    "scanning": "Scanning",
    "random_search": "Random Search",
    "thigmotaxis": "Thigmotaxis",
    "uncertain": "Uncertain",
}
SEARCH_STRATEGY_LABELS_ZH = {
    "direct_path": "直接路径",
    "directed_search": "定向搜索",
    "focal_search": "局灶搜索",
    "indirect_search": "间接搜索",
    "chaining": "环形搜索",
    "scanning": "扫描搜索",
    "random_search": "随机搜索",
    "thigmotaxis": "趋边搜索",
    "uncertain": "不确定",
}
SEARCH_STRATEGY_SCORES = {
    "thigmotaxis": 0,
    "random_search": 1,
    "scanning": 2,
    "chaining": 3,
    "indirect_search": 4,
    "focal_search": 5,
    "directed_search": 6,
    "direct_path": 7,
    "uncertain": None,
}


def sanitize_name(text: str, fallback: str = "mwm") -> str:
    return ascii_slug(text, fallback=fallback, max_length=80)


def _format_metric(value: Any, digits: int = 3) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, int(digits))


def _format_seconds(value: Any) -> float | None:
    return _format_metric(value, 3)


def _coerce_point(point: Any) -> Point:
    if isinstance(point, dict):
        if isinstance(point.get("point"), dict):
            return _coerce_point(point.get("point"))
        if isinstance(point.get("center"), dict):
            return _coerce_point(point.get("center"))
        return (float(point["x"]), float(point["y"]))
    return (float(point[0]), float(point[1]))


def _coerce_axes(value: Any) -> Point:
    if isinstance(value, dict):
        if value.get("axes_px") is not None:
            return _coerce_axes(value.get("axes_px"))
        if value.get("axes") is not None:
            return _coerce_axes(value.get("axes"))
        if value.get("width") is not None and value.get("height") is not None:
            return (float(value["width"]), float(value["height"]))
        if value.get("w") is not None and value.get("h") is not None:
            return (float(value["w"]), float(value["h"]))
        if value.get("radius_x_px") is not None and value.get("radius_y_px") is not None:
            return (float(value["radius_x_px"]) * 2.0, float(value["radius_y_px"]) * 2.0)
        if value.get("rx") is not None and value.get("ry") is not None:
            return (float(value["rx"]) * 2.0, float(value["ry"]) * 2.0)
    return (float(value[0]), float(value[1]))


def _coerce_points(points: Any, min_points: int = 1) -> list[Point]:
    if not isinstance(points, list) or len(points) < int(min_points):
        raise ValueError(f"requires at least {min_points} points")
    return [_coerce_point(point) for point in points]


def _point_dict(point: Point | None) -> dict[str, float] | None:
    if point is None:
        return None
    return {"x": float(point[0]), "y": float(point[1])}


def _normalize_trial_type(value: Any) -> str:
    raw = str(value or "acquisition_hidden").strip().lower()
    aliases = {
        "acquisition": "acquisition_hidden",
        "hidden": "acquisition_hidden",
        "hidden_platform": "acquisition_hidden",
        "training": "acquisition_hidden",
        "visible": "visible_platform",
        "visible_platform": "visible_platform",
        "probe": "probe",
        "probe_trial": "probe",
        "reversal": "reversal",
    }
    return aliases.get(raw, "acquisition_hidden")


def _normalize_quadrant(value: Any) -> str:
    raw = str(value or "").strip().upper().replace("-", "").replace("_", "")
    aliases = {
        "NORTHEAST": "NE",
        "NORTHWEST": "NW",
        "SOUTHWEST": "SW",
        "SOUTHEAST": "SE",
        "1": "NE",
        "I": "NE",
        "Q1": "NE",
        "QI": "NE",
        "QUADRANT1": "NE",
        "2": "NW",
        "II": "NW",
        "Q2": "NW",
        "QII": "NW",
        "QUADRANT2": "NW",
        "3": "SW",
        "III": "SW",
        "Q3": "SW",
        "QIII": "SW",
        "QUADRANT3": "SW",
        "4": "SE",
        "IV": "SE",
        "Q4": "SE",
        "QIV": "SE",
        "QUADRANT4": "SE",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in QUADRANTS else ""


def _quadrant_display_label(value: Any) -> str:
    normalized = _normalize_quadrant(value)
    if normalized:
        return QUADRANT_DISPLAY_LABELS[normalized]
    return str(value or "Unknown")


def _rotate_point(point: Point, angle_deg: float) -> Point:
    angle = math.radians(float(angle_deg or 0.0))
    c = math.cos(angle)
    s = math.sin(angle)
    x = float(point[0])
    y = float(point[1])
    return (
        x * c - y * s,
        x * s + y * c,
    )


def _ellipse_points(
    center: Point,
    axes: Point,
    angle_deg: float,
    count: int = 160,
) -> np.ndarray:
    a = max(1e-6, float(axes[0]) * 0.5)
    b = max(1e-6, float(axes[1]) * 0.5)
    theta = np.linspace(0.0, 2.0 * math.pi, int(count), endpoint=False)
    local = np.column_stack([a * np.cos(theta), b * np.sin(theta)])
    angle = math.radians(float(angle_deg))
    rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    pts = local @ rot.T + np.asarray(center, dtype=np.float32)
    return pts.astype(np.float32)


def _circle_points(radius_cm: float, count: int = 160) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, int(count), endpoint=False)
    return np.column_stack([float(radius_cm) * np.cos(theta), float(radius_cm) * np.sin(theta)]).astype(np.float32)


def _distance_point_to_segment(point: Point, start: Point, end: Point) -> float:
    p = np.asarray(point, dtype=np.float32)
    a = np.asarray(start, dtype=np.float32)
    b = np.asarray(end, dtype=np.float32)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-9:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    projection = a + t * ab
    return float(np.linalg.norm(p - projection))


def _angle_between_deg(a: Point, b: Point) -> float | None:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-9 or nb <= 1e-9:
        return None
    cos_value = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos_value))


class PoolGeometry:
    """Map the visible pool boundary to a standard circular pool coordinate system."""

    def __init__(
        self,
        boundary_points: Sequence[Sequence[float]] | None = None,
        pool_diameter_cm: float = 120.0,
        circle_center: Sequence[float] | None = None,
        circle_radius_px: float | None = None,
        ellipse_center: Sequence[float] | None = None,
        ellipse_axes_px: Sequence[float] | dict[str, Any] | None = None,
        ellipse_angle_deg: float = 0.0,
        correction_mode: str = "ellipse_affine",
        distortion_k1: float = 0.0,
        distortion_k2: float = 0.0,
    ) -> None:
        self.pool_diameter_cm = max(1.0, float(pool_diameter_cm))
        self.pool_radius_cm = self.pool_diameter_cm * 0.5
        self.correction_mode = str(correction_mode or "ellipse_affine").strip().lower()
        self.distortion_k1 = float(distortion_k1 or 0.0)
        self.distortion_k2 = float(distortion_k2 or 0.0)

        self.raw_boundary_points = np.asarray(boundary_points or [], dtype=np.float32).reshape(-1, 2)
        if ellipse_center is not None and ellipse_axes_px is not None:
            center = _coerce_point(ellipse_center)
            axes = _coerce_axes(ellipse_axes_px)
            axes = (max(1.0, float(axes[0])), max(1.0, float(axes[1])))
            self.raw_boundary_points = _ellipse_points(center, axes, float(ellipse_angle_deg or 0.0), count=96)
            self.ellipse_center_px = (float(center[0]), float(center[1]))
            self.ellipse_axes_px = axes
            self.ellipse_angle_deg = float(ellipse_angle_deg or 0.0)
        elif circle_center is not None and circle_radius_px is not None:
            center = _coerce_point(circle_center)
            radius = max(1.0, float(circle_radius_px))
            self.raw_boundary_points = _ellipse_points(center, (radius * 2.0, radius * 2.0), 0.0, count=80)
            self.ellipse_center_px = (float(center[0]), float(center[1]))
            self.ellipse_axes_px = (radius * 2.0, radius * 2.0)
            self.ellipse_angle_deg = 0.0
        else:
            if self.raw_boundary_points.shape[0] < 5:
                raise ValueError("Water maze pool calibration requires at least 5 boundary points.")
            self.ellipse_center_px, self.ellipse_axes_px, self.ellipse_angle_deg = self._fit_ellipse(self.raw_boundary_points)

        self.distortion_center_px = self.ellipse_center_px
        self.distortion_scale_px = max(float(self.ellipse_axes_px[0]), float(self.ellipse_axes_px[1]), 1.0) * 0.5

        if self.distortion_k1 or self.distortion_k2:
            corrected = np.asarray([self._apply_radial_correction(tuple(point)) for point in self.raw_boundary_points], dtype=np.float32)
            self.ellipse_center_px, self.ellipse_axes_px, self.ellipse_angle_deg = self._fit_ellipse(corrected)
            self.boundary_points_for_fit = corrected
        else:
            self.boundary_points_for_fit = self.raw_boundary_points.copy()

        self.image_to_pool_matrix, self.pool_to_image_matrix = self._build_affine_matrices()
        self.boundary_residual_mean_percent, self.boundary_residual_max_percent = self._boundary_residuals()

    @staticmethod
    def _fit_ellipse(points: np.ndarray) -> tuple[Point, Point, float]:
        ellipse = cv2.fitEllipse(points.astype(np.float32))
        (cx, cy), (w, h), angle = ellipse
        if w <= 1e-6 or h <= 1e-6:
            raise ValueError("Water maze pool boundary fit failed because the ellipse axes are too small.")
        return (float(cx), float(cy)), (float(w), float(h)), float(angle)

    def _build_affine_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        cx, cy = self.ellipse_center_px
        a = max(1e-6, float(self.ellipse_axes_px[0]) * 0.5)
        b = max(1e-6, float(self.ellipse_axes_px[1]) * 0.5)
        radius = self.pool_radius_cm
        angle = math.radians(float(self.ellipse_angle_deg))
        c = math.cos(angle)
        s = math.sin(angle)
        img_to_pool = np.array(
            [
                [radius / a * c, radius / a * s, radius / a * (-c * cx - s * cy)],
                [radius / b * -s, radius / b * c, radius / b * (s * cx - c * cy)],
            ],
            dtype=np.float32,
        )
        pool_to_img = np.array(
            [
                [c * a / radius, -s * b / radius, cx],
                [s * a / radius, c * b / radius, cy],
            ],
            dtype=np.float32,
        )
        return img_to_pool, pool_to_img

    def _apply_radial_correction(self, point: Point) -> Point:
        if not (self.distortion_k1 or self.distortion_k2):
            return (float(point[0]), float(point[1]))
        cx, cy = self.distortion_center_px
        scale = max(float(self.distortion_scale_px), 1e-6)
        dx = (float(point[0]) - cx) / scale
        dy = (float(point[1]) - cy) / scale
        r2 = dx * dx + dy * dy
        factor = 1.0 + self.distortion_k1 * r2 + self.distortion_k2 * r2 * r2
        return (cx + dx * factor * scale, cy + dy * factor * scale)

    def _undo_radial_correction(self, point: Point) -> Point:
        if not (self.distortion_k1 or self.distortion_k2):
            return (float(point[0]), float(point[1]))
        cx, cy = self.distortion_center_px
        scale = max(float(self.distortion_scale_px), 1e-6)
        target = np.asarray([(float(point[0]) - cx) / scale, (float(point[1]) - cy) / scale], dtype=np.float32)
        guess = target.copy()
        for _ in range(8):
            r2 = float(np.dot(guess, guess))
            factor = 1.0 + self.distortion_k1 * r2 + self.distortion_k2 * r2 * r2
            if abs(factor) <= 1e-9:
                break
            guess = target / factor
        return (cx + float(guess[0]) * scale, cy + float(guess[1]) * scale)

    def image_to_pool(self, point_px: Sequence[float]) -> Point:
        corrected = self._apply_radial_correction((float(point_px[0]), float(point_px[1])))
        src = np.array([[[corrected[0], corrected[1]]]], dtype=np.float32)
        dst = cv2.transform(src, self.image_to_pool_matrix)[0, 0]
        x = float(dst[0])
        y = float(dst[1])
        radius = math.hypot(x, y)
        if radius > self.pool_radius_cm:
            scale = self.pool_radius_cm / max(radius, 1e-6)
            x *= scale
            y *= scale
        return (x, y)

    def pool_to_image(self, point_cm: Sequence[float]) -> Point:
        src = np.array([[[float(point_cm[0]), float(point_cm[1])]]], dtype=np.float32)
        corrected = cv2.transform(src, self.pool_to_image_matrix)[0, 0]
        return self._undo_radial_correction((float(corrected[0]), float(corrected[1])))

    def pool_to_image_points(self, points_cm: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        pts = np.asarray(points_cm, dtype=np.float32).reshape(-1, 2)
        return np.asarray([self.pool_to_image(point) for point in pts], dtype=np.float32)

    def image_boundary_points(self, count: int = 160) -> np.ndarray:
        circle = _circle_points(self.pool_radius_cm, count=count)
        return self.pool_to_image_points(circle)

    def _boundary_residuals(self) -> tuple[float, float]:
        if self.raw_boundary_points.size == 0:
            return 0.0, 0.0
        radii = []
        for point in self.raw_boundary_points:
            xy = self.image_to_pool(point)
            radii.append(math.hypot(xy[0], xy[1]))
        residuals = [abs(r - self.pool_radius_cm) / max(self.pool_radius_cm, 1e-6) * 100.0 for r in radii]
        return float(np.mean(residuals)), float(np.max(residuals))

    def as_config_summary(self) -> dict[str, Any]:
        return {
            "pool_diameter_cm": round(self.pool_diameter_cm, 3),
            "pool_radius_cm": round(self.pool_radius_cm, 3),
            "correction_mode": self.correction_mode,
            "ellipse_center_px": _point_dict(self.ellipse_center_px),
            "ellipse_axes_px": {"major_px": round(float(max(self.ellipse_axes_px)), 3), "minor_px": round(float(min(self.ellipse_axes_px)), 3)},
            "ellipse_angle_deg": round(float(self.ellipse_angle_deg), 3),
            "boundary_residual_mean_percent": round(self.boundary_residual_mean_percent, 3),
            "boundary_residual_max_percent": round(self.boundary_residual_max_percent, 3),
            "distortion_k1": round(float(self.distortion_k1), 6),
            "distortion_k2": round(float(self.distortion_k2), 6),
        }


class WaterMazeTracker:
    """Track one mouse in one Morris water maze trial."""

    def __init__(
        self,
        pool_boundary_points: Sequence[Sequence[float]] | None = None,
        pool: dict[str, Any] | None = None,
        platform_point: Sequence[float] | dict[str, Any] | None = None,
        platform: dict[str, Any] | None = None,
        detection_points: Sequence[Sequence[float]] | None = None,
        platform_exclusion_points: Sequence[Sequence[float]] | None = None,
        pool_diameter_cm: float = 120.0,
        platform_radius_cm: float = 5.0,
        trial_type: str = "acquisition_hidden",
        target_quadrant: str | None = None,
        start_position: str = "N",
        day_label: str = "",
        trial_no: int = 1,
        fps: float = 30.0,
        movement_threshold_cm_s: float = 2.0,
        thigmotaxis_band_cm: float = 10.0,
        platform_dwell_s: float = 1.0,
        track_smoothing_max_jump_cm: float = 25.0,
        max_gap_s: float = 0.5,
        quadrant_rotation_deg: float = 0.0,
        whishaw_corridor_width_cm: float | None = None,
        exclude_platform_from_detection: bool = False,
        correction_mode: str = "ellipse_affine",
        distortion_k1: float = 0.0,
        distortion_k2: float = 0.0,
        roi_id: int = 1,
        roi_name: Optional[str] = None,
        time_config: Optional[Dict[str, Any]] = None,
        global_start_time_s: float = 0.0,
    ) -> None:
        pool = dict(pool or {})
        if pool_boundary_points is None:
            pool_boundary_points = pool.get("boundary_points") or pool.get("pool_boundary_points") or pool.get("points")
        circle_center = pool.get("center")
        circle_radius = pool.get("radius_px") or pool.get("radius")
        ellipse_center = pool.get("ellipse_center") or pool.get("center")
        ellipse_axes_px = pool.get("ellipse_axes_px") or pool.get("axes_px") or pool.get("axes")
        if ellipse_axes_px is None and pool.get("radius_x_px") is not None and pool.get("radius_y_px") is not None:
            ellipse_axes_px = {
                "radius_x_px": pool.get("radius_x_px"),
                "radius_y_px": pool.get("radius_y_px"),
            }
        ellipse_angle_deg = float(pool.get("angle_deg") or pool.get("ellipse_angle_deg") or 0.0)
        if pool.get("diameter_cm") is not None:
            pool_diameter_cm = float(pool.get("diameter_cm"))

        self.geometry = PoolGeometry(
            boundary_points=pool_boundary_points,
            pool_diameter_cm=pool_diameter_cm,
            circle_center=circle_center,
            circle_radius_px=circle_radius,
            ellipse_center=ellipse_center if ellipse_axes_px is not None else None,
            ellipse_axes_px=ellipse_axes_px,
            ellipse_angle_deg=ellipse_angle_deg,
            correction_mode=correction_mode,
            distortion_k1=distortion_k1,
            distortion_k2=distortion_k2,
        )
        self.pool_radius_cm = self.geometry.pool_radius_cm
        self.pool_diameter_cm = self.geometry.pool_diameter_cm

        self.roi_id = int(roi_id)
        self.roi_name = roi_name or f"MWM {self.roi_id}"
        self.trial_type = _normalize_trial_type(trial_type)
        self.day_label = str(day_label or "")
        self.trial_no = int(trial_no or 1)
        self.fps = max(1.0, float(fps))
        self.time_config = self._normalize_time_config(time_config)
        self.global_start_time_s = max(0.0, float(global_start_time_s or 0.0))
        self.movement_threshold_cm_s = max(0.0, float(movement_threshold_cm_s))
        self.thigmotaxis_band_cm = float(np.clip(float(thigmotaxis_band_cm), 0.0, self.pool_radius_cm))
        self.platform_dwell_s = max(0.0, float(platform_dwell_s))
        self.track_smoothing_max_jump_cm = max(0.0, float(track_smoothing_max_jump_cm))
        self.max_gap_s = max(0.0, float(max_gap_s))
        self.coordinate_rotation_deg = float(quadrant_rotation_deg or 0.0)
        self.quadrant_rotation_deg = self.coordinate_rotation_deg
        self.start_position = str(start_position or "N").strip().upper()
        self.whishaw_corridor_width_cm = float(
            whishaw_corridor_width_cm
            if whishaw_corridor_width_cm is not None
            else max(self.pool_radius_cm * 0.15, float(platform_radius_cm) * 2.0)
        )
        self.exclude_platform_from_detection = bool(exclude_platform_from_detection)

        platform = dict(platform or {})
        if platform.get("radius_cm") is not None:
            platform_radius_cm = float(platform.get("radius_cm"))
        self.platform_radius_cm = max(0.1, float(platform_radius_cm))
        raw_platform = platform_point or platform.get("point") or platform.get("center")
        if raw_platform is None and platform.get("x") is not None and platform.get("y") is not None:
            raw_platform = platform
        if raw_platform is None:
            raise ValueError("Water maze analysis requires a platform point.")
        self.platform_center_img = _coerce_point(raw_platform)
        self.platform_center_cm = self.geometry.image_to_pool(self.platform_center_img)
        self.target_quadrant = _normalize_quadrant(target_quadrant) or self._quadrant_for_point(self.platform_center_cm)
        self.release_point_cm = self._start_position_to_pool(self.start_position)

        self.raw_detection_points = detection_points
        self.detection_polygon_img = np.asarray(_coerce_points(detection_points, min_points=3), dtype=np.float32) if detection_points is not None else None
        self.platform_exclusion_polygons_img: list[np.ndarray] = []
        if platform_exclusion_points is not None:
            self.platform_exclusion_polygons_img.append(np.asarray(_coerce_points(platform_exclusion_points, min_points=3), dtype=np.float32))

        self.detection_mask: Optional[np.ndarray] = None
        self.detection_bbox: Optional[Tuple[int, int, int, int]] = None
        self._cropped_detection_mask: Optional[np.ndarray] = None
        self.has_custom_detection_mask = self.detection_polygon_img is not None

        self.total_frames = 0
        self.frames_in_window = 0
        self.detected_frames = 0
        self.detected_frames_in_window = 0
        self.detected_time_in_window_s = 0.0
        self.time_in_window_s = 0.0
        self.total_distance_cm = 0.0
        self.path_length_to_platform_cm: Optional[float] = None
        self.max_speed_cm_s = 0.0
        self.last_speed_cm_s = 0.0
        self.moving_time_s = 0.0
        self.floating_time_s = 0.0
        self.thigmotaxis_time_s = 0.0
        self.thigmotaxis_distance_cm = 0.0
        self.whishaw_corridor_time_s = 0.0
        self.whishaw_corridor_distance_cm = 0.0
        self.platform_time_s = 0.0
        self.platform_dwell_streak_s = 0.0
        self.platform_entries = 0
        self.platform_crossings = 0
        self.success = False
        self.escape_latency_s: Optional[float] = None
        self.time_to_first_platform_entry_s: Optional[float] = None
        self.first_platform_crossing_latency_s: Optional[float] = None
        self.distance_to_platform_integral_cm_s = 0.0
        self.cipl_weighted_distance_cm2 = 0.0
        self.heading_error_samples: list[float] = []
        self.initial_heading_error_deg: Optional[float] = None

        self.quadrant_time_s = {key: 0.0 for key in QUADRANTS}
        self.quadrant_distance_cm = {key: 0.0 for key in QUADRANTS}
        self.quadrant_frames = {key: 0 for key in QUADRANTS}

        self.prev_all_pool_xy_cm: Optional[Point] = None
        self.last_all_pool_xy_cm: Optional[Point] = None
        self.prev_stats_pool_xy_cm: Optional[Point] = None
        self.prev_centroid_px: Optional[Point] = None
        self.first_detected_time_abs: Optional[float] = None
        self.prev_in_platform_stats = False
        self.missing_time_s = 0.0
        self.missing_frames = 0
        self.jump_filtered_count = 0

        self.trajectory_px_live: deque[Point] = deque(maxlen=1000)
        self.trajectory_px_stats: list[Point] = []
        self.trajectory_pool_cm_stats: list[Point] = []
        self.frame_records: list[dict[str, Any]] = []
        self.last_quadrant = "Unknown"
        self.qc_flags: set[str] = set()
        if self.geometry.boundary_residual_max_percent > 8.0:
            self.qc_flags.add("pool_boundary_high_residual")

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
            else:
                try:
                    cfg[key] = float(value)
                except Exception:
                    cfg[key] = None
        return cfg

    def _stats_window_origin_abs(self) -> float:
        start_cfg = self.time_config.get("start")
        if self.time_config.get("mode") == "precise":
            return float(start_cfg or self.global_start_time_s)
        if self.time_config.get("mode") == "relative" and self.first_detected_time_abs is not None:
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
        if self.first_detected_time_abs is None:
            return False
        elapsed = current_time - self.first_detected_time_abs
        if start_cfg is not None and elapsed < start_cfg:
            return False
        if end_cfg is not None and elapsed > end_cfg:
            return False
        return True

    def point_to_coordinate_frame(self, point_cm: Point | None) -> Point | None:
        """Return pool coordinates expressed in this ROI's rotated experiment axes."""
        if point_cm is None:
            return None
        return _rotate_point(point_cm, -self.coordinate_rotation_deg)

    def point_from_coordinate_frame(self, point_cm: Point | None) -> Point | None:
        if point_cm is None:
            return None
        return _rotate_point(point_cm, self.coordinate_rotation_deg)

    def coordinate_axis_segments_cm(self) -> tuple[np.ndarray, np.ndarray]:
        radius = float(self.pool_radius_cm)
        horizontal = np.asarray(
            [
                self.point_from_coordinate_frame((-radius, 0.0)),
                self.point_from_coordinate_frame((radius, 0.0)),
            ],
            dtype=np.float32,
        )
        vertical = np.asarray(
            [
                self.point_from_coordinate_frame((0.0, -radius)),
                self.point_from_coordinate_frame((0.0, radius)),
            ],
            dtype=np.float32,
        )
        return horizontal, vertical

    def coordinate_quadrant_label_points_cm(self) -> dict[str, Point]:
        radius = float(self.pool_radius_cm) * 0.56
        coordinate_points = {
            "NE": (radius, -radius),
            "NW": (-radius, -radius),
            "SW": (-radius, radius),
            "SE": (radius, radius),
        }
        return {
            key: self.point_from_coordinate_frame(point) or (0.0, 0.0)
            for key, point in coordinate_points.items()
        }

    def quadrant_label(self, value: Any = None) -> str:
        return _quadrant_display_label(self.target_quadrant if value is None else value)

    def _start_position_to_pool(self, value: str) -> Point:
        mapping = {
            "N": (0.0, -1.0),
            "NE": (math.sqrt(0.5), -math.sqrt(0.5)),
            "E": (1.0, 0.0),
            "SE": (math.sqrt(0.5), math.sqrt(0.5)),
            "S": (0.0, 1.0),
            "SW": (-math.sqrt(0.5), math.sqrt(0.5)),
            "W": (-1.0, 0.0),
            "NW": (-math.sqrt(0.5), -math.sqrt(0.5)),
        }
        unit = mapping.get(str(value or "N").upper(), mapping["N"])
        point = (unit[0] * self.pool_radius_cm, unit[1] * self.pool_radius_cm)
        return self.point_from_coordinate_frame(point) or point

    def _quadrant_for_point(self, point_cm: Point | None) -> str:
        if point_cm is None:
            return "Unknown"
        coordinate_point = self.point_to_coordinate_frame(point_cm)
        if coordinate_point is None:
            return "Unknown"
        x, y = float(coordinate_point[0]), float(coordinate_point[1])
        if x >= 0 and y < 0:
            return "NE"
        if x < 0 and y < 0:
            return "NW"
        if x < 0 and y >= 0:
            return "SW"
        return "SE"

    def _is_thigmotaxis(self, point_cm: Point | None) -> bool:
        if point_cm is None:
            return False
        radial = math.hypot(float(point_cm[0]), float(point_cm[1]))
        return radial >= max(0.0, self.pool_radius_cm - self.thigmotaxis_band_cm)

    def _distance_to_platform(self, point_cm: Point | None) -> float | None:
        if point_cm is None:
            return None
        return float(math.hypot(point_cm[0] - self.platform_center_cm[0], point_cm[1] - self.platform_center_cm[1]))

    def _in_platform(self, point_cm: Point | None) -> bool:
        distance = self._distance_to_platform(point_cm)
        return bool(distance is not None and distance <= self.platform_radius_cm)

    def _in_whishaw_corridor(self, point_cm: Point | None) -> bool:
        if point_cm is None:
            return False
        distance = _distance_point_to_segment(point_cm, self.release_point_cm, self.platform_center_cm)
        return distance <= self.whishaw_corridor_width_cm * 0.5

    def initialize(self, frame_shape: Tuple[int, ...]) -> None:
        if self.detection_mask is not None:
            return
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        if self.detection_polygon_img is not None:
            cv2.fillPoly(mask, [self.detection_polygon_img.astype(np.int32)], 255)
        else:
            boundary = self.geometry.image_boundary_points(count=200).astype(np.int32)
            cv2.fillPoly(mask, [boundary], 255)
        for poly in self.platform_exclusion_polygons_img:
            cv2.fillPoly(mask, [poly.astype(np.int32)], 0)
        if self.exclude_platform_from_detection:
            platform_poly = self.geometry.pool_to_image_points(
                _circle_points(self.platform_radius_cm * 1.25, count=64) + np.asarray(self.platform_center_cm, dtype=np.float32)
            ).astype(np.int32)
            cv2.fillPoly(mask, [platform_poly], 0)

        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            raise ValueError("Water maze detection mask is empty.")
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        self.detection_bbox = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
        self.detection_mask = mask
        self._cropped_detection_mask = mask[y0 : y1 + 1, x0 : x1 + 1]

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
        dt_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.total_frames += 1
        try:
            current_dt = float(dt_s) if dt_s is not None else 1.0 / max(float(self.fps), 1e-6)
        except (TypeError, ValueError):
            current_dt = 1.0 / max(float(self.fps), 1e-6)
        current_dt = max(current_dt, 1e-6)

        detection, fg_mask = self.detect_mouse(frame, gray, bg_subtractor, morph_processor, detector)
        detected = detection is not None
        centroid_px: Optional[Point] = None
        bbox = None
        contour = None
        if detection is not None:
            self.detected_frames += 1
            centroid_px = tuple(float(v) for v in detection["centroid"])
            bbox = detection.get("bbox")
            contour = detection.get("contour")
            pool_xy_cm = self.geometry.image_to_pool(centroid_px)
            self.last_all_pool_xy_cm = pool_xy_cm
            self.prev_centroid_px = centroid_px
            self.missing_frames = 0
            self.missing_time_s = 0.0
            if self.first_detected_time_abs is None:
                self.first_detected_time_abs = float(current_time)
        elif self.last_all_pool_xy_cm is not None and self.missing_time_s < self.max_gap_s:
            pool_xy_cm = self.last_all_pool_xy_cm
            centroid_px = self.prev_centroid_px
            self.missing_frames += 1
            self.missing_time_s += current_dt
        else:
            pool_xy_cm = None
            self.missing_frames += 1
            self.missing_time_s += current_dt

        speed_cm_s = 0.0
        all_step_distance_cm = 0.0
        jump_filtered = False
        if pool_xy_cm is not None:
            if self.prev_all_pool_xy_cm is not None:
                all_step_distance_cm = float(math.hypot(pool_xy_cm[0] - self.prev_all_pool_xy_cm[0], pool_xy_cm[1] - self.prev_all_pool_xy_cm[1]))
                if self.track_smoothing_max_jump_cm and all_step_distance_cm > self.track_smoothing_max_jump_cm:
                    jump_filtered = True
                    self.jump_filtered_count += 1
                    self.qc_flags.add("large_jump_filtered")
                    all_step_distance_cm = 0.0
                speed_cm_s = all_step_distance_cm / current_dt
            self.prev_all_pool_xy_cm = pool_xy_cm
        else:
            self.prev_all_pool_xy_cm = None

        self.last_speed_cm_s = speed_cm_s
        in_stats_window = self._is_in_stats_window(float(current_time))
        quadrant = self._quadrant_for_point(pool_xy_cm)
        self.last_quadrant = quadrant
        in_platform = self._in_platform(pool_xy_cm)
        in_thigmotaxis = self._is_thigmotaxis(pool_xy_cm)
        in_corridor = self._in_whishaw_corridor(pool_xy_cm)
        distance_to_platform = self._distance_to_platform(pool_xy_cm)

        stats_step_distance_cm = 0.0
        stats_cumulative_distance_cm = self.total_distance_cm
        platform_entry_event = False
        platform_crossing_event = False

        if in_stats_window:
            self.frames_in_window += 1
            self.time_in_window_s += current_dt
            if detected:
                self.detected_frames_in_window += 1
                self.detected_time_in_window_s += current_dt

            if pool_xy_cm is not None:
                if self.prev_stats_pool_xy_cm is not None:
                    raw_step = float(math.hypot(pool_xy_cm[0] - self.prev_stats_pool_xy_cm[0], pool_xy_cm[1] - self.prev_stats_pool_xy_cm[1]))
                    if self.track_smoothing_max_jump_cm and raw_step > self.track_smoothing_max_jump_cm:
                        stats_step_distance_cm = 0.0
                        self.qc_flags.add("large_jump_filtered")
                    else:
                        stats_step_distance_cm = raw_step
                        move_vector = (pool_xy_cm[0] - self.prev_stats_pool_xy_cm[0], pool_xy_cm[1] - self.prev_stats_pool_xy_cm[1])
                        target_vector = (self.platform_center_cm[0] - self.prev_stats_pool_xy_cm[0], self.platform_center_cm[1] - self.prev_stats_pool_xy_cm[1])
                        heading = _angle_between_deg(move_vector, target_vector)
                        if heading is not None and stats_step_distance_cm >= 0.5:
                            self.heading_error_samples.append(heading)
                            if self.initial_heading_error_deg is None:
                                self.initial_heading_error_deg = heading
                    self.total_distance_cm += stats_step_distance_cm
                else:
                    start_target = (self.platform_center_cm[0] - pool_xy_cm[0], self.platform_center_cm[1] - pool_xy_cm[1])
                    release_target = (self.platform_center_cm[0] - self.release_point_cm[0], self.platform_center_cm[1] - self.release_point_cm[1])
                    self.initial_heading_error_deg = self.initial_heading_error_deg or _angle_between_deg(release_target, start_target)

                self.prev_stats_pool_xy_cm = pool_xy_cm
                stats_cumulative_distance_cm = self.total_distance_cm
                self.max_speed_cm_s = max(self.max_speed_cm_s, speed_cm_s)

                if speed_cm_s >= self.movement_threshold_cm_s:
                    self.moving_time_s += current_dt
                else:
                    self.floating_time_s += current_dt

                if quadrant in self.quadrant_time_s:
                    self.quadrant_frames[quadrant] += 1
                    self.quadrant_time_s[quadrant] += current_dt
                    self.quadrant_distance_cm[quadrant] += stats_step_distance_cm

                if in_thigmotaxis:
                    self.thigmotaxis_time_s += current_dt
                    self.thigmotaxis_distance_cm += stats_step_distance_cm
                if in_corridor:
                    self.whishaw_corridor_time_s += current_dt
                    self.whishaw_corridor_distance_cm += stats_step_distance_cm

                if distance_to_platform is not None:
                    self.distance_to_platform_integral_cm_s += distance_to_platform * current_dt
                    self.cipl_weighted_distance_cm2 += distance_to_platform * stats_step_distance_cm

                if in_platform:
                    self.platform_time_s += current_dt
                    self.platform_dwell_streak_s += current_dt
                    if not self.prev_in_platform_stats:
                        platform_entry_event = True
                        platform_crossing_event = True
                        self.platform_entries += 1
                        self.platform_crossings += 1
                        latency = float(current_time) - self._stats_window_origin_abs()
                        if self.time_to_first_platform_entry_s is None:
                            self.time_to_first_platform_entry_s = latency
                        if self.first_platform_crossing_latency_s is None:
                            self.first_platform_crossing_latency_s = latency
                    if not self.success and self.platform_dwell_streak_s >= self.platform_dwell_s:
                        self.success = True
                        self.escape_latency_s = float(current_time) - self._stats_window_origin_abs()
                        self.path_length_to_platform_cm = self.total_distance_cm
                else:
                    self.platform_dwell_streak_s = 0.0
                self.prev_in_platform_stats = in_platform

                if centroid_px is not None:
                    self.trajectory_px_live.append(centroid_px)
                    self.trajectory_px_stats.append(centroid_px)
                self.trajectory_pool_cm_stats.append(pool_xy_cm)
            else:
                self.prev_stats_pool_xy_cm = None
                self.prev_in_platform_stats = False
                self.platform_dwell_streak_s = 0.0
                self.floating_time_s += current_dt
        else:
            self.prev_stats_pool_xy_cm = None
            self.prev_in_platform_stats = False
            self.platform_dwell_streak_s = 0.0
            if centroid_px is not None:
                self.trajectory_px_live.append(centroid_px)

        record = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "frame": self.total_frames - 1,
            "time_s": _format_seconds(current_time),
            "dt_s": _format_seconds(current_dt),
            "window_time_s": _format_seconds(float(current_time) - self._stats_window_origin_abs()) if in_stats_window else None,
            "detected": bool(detected),
            "in_stats_window": bool(in_stats_window),
            "x_px": _format_metric(centroid_px[0]) if centroid_px is not None else None,
            "y_px": _format_metric(centroid_px[1]) if centroid_px is not None else None,
            "x_pool_cm": _format_metric(pool_xy_cm[0]) if pool_xy_cm is not None else None,
            "y_pool_cm": _format_metric(pool_xy_cm[1]) if pool_xy_cm is not None else None,
            "radial_distance_cm": _format_metric(math.hypot(pool_xy_cm[0], pool_xy_cm[1])) if pool_xy_cm is not None else None,
            "speed_cm_s": _format_metric(speed_cm_s),
            "step_distance_cm": _format_metric(stats_step_distance_cm),
            "cumulative_distance_cm": _format_metric(stats_cumulative_distance_cm),
            "quadrant": quadrant,
            "quadrant_label": self.quadrant_label(quadrant),
            "target_quadrant": self.target_quadrant,
            "target_quadrant_label": self.quadrant_label(self.target_quadrant),
            "in_target_quadrant": bool(quadrant == self.target_quadrant) if in_stats_window else False,
            "in_platform_zone": bool(in_platform) if in_stats_window else False,
            "platform_entry_event": bool(platform_entry_event),
            "platform_crossing_event": bool(platform_crossing_event),
            "distance_to_platform_cm": _format_metric(distance_to_platform) if distance_to_platform is not None else None,
            "in_thigmotaxis": bool(in_thigmotaxis) if in_stats_window else False,
            "in_whishaw_corridor": bool(in_corridor) if in_stats_window else False,
            "jump_filtered": bool(jump_filtered),
            "success": bool(self.success),
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
            "pool_xy_cm": pool_xy_cm,
            "speed_cm_s": speed_cm_s,
            "step_distance_cm": stats_step_distance_cm,
            "cumulative_distance_cm": self.total_distance_cm,
            "quadrant": quadrant,
            "quadrant_label": self.quadrant_label(quadrant),
            "target_quadrant": self.target_quadrant,
            "target_quadrant_label": self.quadrant_label(self.target_quadrant),
            "in_platform_zone": bool(in_platform),
            "in_thigmotaxis": bool(in_thigmotaxis),
            "in_whishaw_corridor": bool(in_corridor),
            "platform_entries": int(self.platform_entries),
            "success": bool(self.success),
            "escape_latency_s": self.escape_latency_s,
            "trajectory_px": list(self.trajectory_px_live),
            "in_stats_window": bool(in_stats_window),
            "record": record,
        }

    def _search_strategy_result(self, code: str, confidence: float, features: Dict[str, Any]) -> Dict[str, Any]:
        code = code if code in SEARCH_STRATEGY_LABELS else "uncertain"
        basis_keys = [
            "directness",
            "heading_error_deg",
            "target_quadrant_percent",
            "thigmotaxis_percent",
            "near_platform_percent",
            "annulus_percent",
            "coverage_percent",
            "mean_radial_percent",
            "quadrants_visited",
        ]
        basis_parts = []
        for key in basis_keys:
            value = features.get(key)
            if value is None:
                continue
            if isinstance(value, float):
                basis_parts.append(f"{key}={value:.2f}")
            else:
                basis_parts.append(f"{key}={value}")
        return {
            "search_strategy": code,
            "search_strategy_label": SEARCH_STRATEGY_LABELS[code],
            "search_strategy_label_zh": SEARCH_STRATEGY_LABELS_ZH[code],
            "search_strategy_display": f"{SEARCH_STRATEGY_LABELS_ZH[code]} / {SEARCH_STRATEGY_LABELS[code]}",
            "search_strategy_score": SEARCH_STRATEGY_SCORES[code],
            "search_strategy_confidence": round(float(np.clip(confidence, 0.0, 1.0)), 3),
            "search_strategy_basis": "; ".join(basis_parts),
            "search_strategy_directness": _format_metric(features.get("directness")),
            "search_strategy_coverage_percent": _format_metric(features.get("coverage_percent")),
            "search_strategy_near_platform_percent": _format_metric(features.get("near_platform_percent")),
            "search_strategy_annulus_percent": _format_metric(features.get("annulus_percent")),
            "search_strategy_mean_radial_percent": _format_metric(features.get("mean_radial_percent")),
            "search_strategy_quadrants_visited": int(features.get("quadrants_visited") or 0),
            "search_strategy_ideal_path_error_cm": _format_metric(features.get("ideal_path_error_cm")),
            "search_strategy_ideal_path_error_percent": _format_metric(features.get("ideal_path_error_percent")),
        }

    def _trajectory_coverage_percent(self, points: Sequence[Point], bins: int = 8) -> float:
        radius = max(float(self.pool_radius_cm), 1e-6)
        bins = max(4, int(bins))
        cell = radius * 2.0 / bins
        valid_cells = 0
        for ix in range(bins):
            cx = -radius + (ix + 0.5) * cell
            for iy in range(bins):
                cy = -radius + (iy + 0.5) * cell
                if math.hypot(cx, cy) <= radius:
                    valid_cells += 1
        occupied: set[tuple[int, int]] = set()
        for point in points:
            x, y = float(point[0]), float(point[1])
            if math.hypot(x, y) > radius * 1.05:
                continue
            ix = int(np.clip(math.floor((x + radius) / cell), 0, bins - 1))
            iy = int(np.clip(math.floor((y + radius) / cell), 0, bins - 1))
            occupied.add((ix, iy))
        return len(occupied) / max(valid_cells, 1) * 100.0

    def _classify_search_strategy(
        self,
        *,
        total_time_s: float,
        detection_rate: float,
        path_for_efficiency: float,
        heading_error_mean: float | None,
        mean_distance_to_platform: float,
        target_time_s: float,
    ) -> Dict[str, Any]:
        radius = max(float(self.pool_radius_cm), 1e-6)
        points = [
            (float(point[0]), float(point[1]))
            for point in self.trajectory_pool_cm_stats
            if point is not None and math.isfinite(float(point[0])) and math.isfinite(float(point[1]))
        ]
        if len(points) < 4 or total_time_s <= 0.0 or detection_rate < 25.0:
            return self._search_strategy_result(
                "uncertain",
                0.0,
                {
                    "target_quadrant_percent": target_time_s / max(total_time_s, 1e-6) * 100.0,
                    "thigmotaxis_percent": self.thigmotaxis_time_s / max(total_time_s, 1e-6) * 100.0,
                    "quadrants_visited": sum(1 for value in self.quadrant_time_s.values() if value > 0.0),
                },
            )

        platform = (float(self.platform_center_cm[0]), float(self.platform_center_cm[1]))
        first_point = points[0]
        initial_distance = float(math.hypot(first_point[0] - platform[0], first_point[1] - platform[1]))
        actual_path = max(float(path_for_efficiency or 0.0), 1e-6)
        directness = min(1.0, initial_distance / actual_path) if actual_path > 0.0 else 0.0
        ideal_path_error_cm = max(actual_path - initial_distance, 0.0)
        ideal_path_error_percent = ideal_path_error_cm / radius * 100.0

        distances_to_platform = [float(math.hypot(x - platform[0], y - platform[1])) for x, y in points]
        radial_distances = [float(math.hypot(x, y)) for x, y in points]
        near_radius = max(radius * 0.25, float(self.platform_radius_cm) * 2.5)
        near_platform_percent = sum(1 for value in distances_to_platform if value <= near_radius) / len(points) * 100.0
        platform_radial = float(math.hypot(platform[0], platform[1]))
        annulus_band = max(radius * 0.08, float(self.platform_radius_cm) * 1.5, 4.0)
        annulus_percent = sum(1 for value in radial_distances if abs(value - platform_radial) <= annulus_band) / len(points) * 100.0
        mean_radial_percent = (sum(radial_distances) / len(radial_distances)) / radius * 100.0
        coverage_percent = self._trajectory_coverage_percent(points)
        centroid = (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))
        centroid_platform_percent = float(math.hypot(centroid[0] - platform[0], centroid[1] - platform[1])) / radius * 100.0
        spread_percent = (
            math.sqrt(sum((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2 for point in points) / len(points)) / radius * 100.0
        )
        target_quadrant_percent = target_time_s / max(total_time_s, 1e-6) * 100.0
        thigmotaxis_percent = self.thigmotaxis_time_s / max(total_time_s, 1e-6) * 100.0
        whishaw_corridor_percent = self.whishaw_corridor_time_s / max(total_time_s, 1e-6) * 100.0
        mean_distance_percent = mean_distance_to_platform / radius * 100.0
        heading = float(heading_error_mean) if heading_error_mean is not None else None
        heading_for_rule = heading if heading is not None else 180.0
        quadrants_visited = sum(1 for value in self.quadrant_time_s.values() if value > 0.0)

        features = {
            "directness": directness,
            "heading_error_deg": heading,
            "target_quadrant_percent": target_quadrant_percent,
            "thigmotaxis_percent": thigmotaxis_percent,
            "near_platform_percent": near_platform_percent,
            "annulus_percent": annulus_percent,
            "coverage_percent": coverage_percent,
            "mean_radial_percent": mean_radial_percent,
            "quadrants_visited": quadrants_visited,
            "ideal_path_error_cm": ideal_path_error_cm,
            "ideal_path_error_percent": ideal_path_error_percent,
        }

        if thigmotaxis_percent >= 65.0 or (thigmotaxis_percent >= 50.0 and mean_radial_percent >= 80.0):
            confidence = 0.55 + min(max(thigmotaxis_percent - 50.0, 0.0), 45.0) / 100.0
            return self._search_strategy_result("thigmotaxis", confidence, features)
        if directness >= 0.80 and heading_for_rule <= 45.0 and mean_distance_percent <= 65.0:
            confidence = 0.55 + min(max(directness - 0.75, 0.0), 0.25) + min(max(45.0 - heading_for_rule, 0.0), 45.0) / 200.0
            return self._search_strategy_result("direct_path", confidence, features)
        if directness >= 0.55 and mean_distance_percent <= 80.0 and (whishaw_corridor_percent >= 45.0 or heading_for_rule <= 70.0):
            confidence = 0.5 + min(max(directness - 0.45, 0.0), 0.35) + min(max(whishaw_corridor_percent - 35.0, 0.0), 40.0) / 200.0
            return self._search_strategy_result("directed_search", confidence, features)
        if near_platform_percent >= 30.0 or (centroid_platform_percent <= 35.0 and spread_percent <= 45.0) or (target_quadrant_percent >= 50.0 and self.platform_crossings > 0):
            confidence = 0.5 + min(max(near_platform_percent - 25.0, 0.0), 40.0) / 120.0
            return self._search_strategy_result("focal_search", confidence, features)
        if annulus_percent >= 60.0 and quadrants_visited >= 3 and coverage_percent >= 18.0 and thigmotaxis_percent < 55.0:
            confidence = 0.5 + min(max(annulus_percent - 55.0, 0.0), 40.0) / 120.0
            return self._search_strategy_result("chaining", confidence, features)
        if 10.0 <= coverage_percent <= 45.0 and mean_radial_percent <= 75.0 and thigmotaxis_percent < 50.0:
            confidence = 0.45 + min(max(coverage_percent - 10.0, 0.0), 25.0) / 120.0
            return self._search_strategy_result("scanning", confidence, features)
        if target_quadrant_percent >= 38.0 or mean_distance_percent <= 75.0 or self.platform_crossings > 0:
            confidence = 0.45 + min(max(target_quadrant_percent - 30.0, 0.0), 30.0) / 150.0
            return self._search_strategy_result("indirect_search", confidence, features)
        confidence = 0.45 + min(max(coverage_percent - 25.0, 0.0), 35.0) / 140.0
        return self._search_strategy_result("random_search", confidence, features)

    def build_summary(self) -> Dict[str, Any]:
        total_time_s = self.time_in_window_s or (self.frames_in_window / self.fps if self.fps > 0 else 0.0)
        mean_speed_cm_s = self.total_distance_cm / max(total_time_s, 1e-6)
        detection_rate = self.detected_time_in_window_s / max(total_time_s, 1e-9) * 100.0 if total_time_s > 0 else 0.0
        mean_distance_to_platform = self.distance_to_platform_integral_cm_s / max(total_time_s, 1e-6)
        cipl = self.cipl_weighted_distance_cm2 / max(self.total_distance_cm, 1e-6) if self.total_distance_cm > 0 else mean_distance_to_platform
        straight_distance = float(math.hypot(self.platform_center_cm[0] - self.release_point_cm[0], self.platform_center_cm[1] - self.release_point_cm[1]))
        path_for_efficiency = self.path_length_to_platform_cm if self.path_length_to_platform_cm is not None else self.total_distance_cm
        path_efficiency = straight_distance / max(path_for_efficiency, 1e-6) if path_for_efficiency > 0 else None
        heading_error_mean = sum(self.heading_error_samples) / len(self.heading_error_samples) if self.heading_error_samples else None
        target_time_s = self.quadrant_time_s.get(self.target_quadrant, 0.0)
        target_distance_cm = self.quadrant_distance_cm.get(self.target_quadrant, 0.0)

        summary: Dict[str, Any] = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "trial_type": self.trial_type,
            "day_label": self.day_label,
            "trial_no": self.trial_no,
            "time_mode": self.time_config.get("mode", "global"),
            "time_start_s": self.time_config.get("start"),
            "time_end_s": self.time_config.get("end"),
            "analysis_window_start_s": _format_seconds(self._stats_window_origin_abs()),
            "total_time_s": _format_seconds(total_time_s),
            "detected_frames": int(self.detected_frames_in_window),
            "detected_rate_percent": round(detection_rate, 2),
            "success": bool(self.success),
            "escape_latency_s": _format_seconds(self.escape_latency_s),
            "path_length_to_platform_cm": _format_metric(self.path_length_to_platform_cm),
            "time_to_first_platform_entry_s": _format_seconds(self.time_to_first_platform_entry_s),
            "latency_to_first_platform_crossing_s": _format_seconds(self.first_platform_crossing_latency_s),
            "platform_entries": int(self.platform_entries),
            "platform_crossings": int(self.platform_crossings),
            "platform_dwell_time_s": _format_seconds(self.platform_time_s),
            "time_in_platform_zone_s": _format_seconds(self.platform_time_s),
            "total_distance_cm": _format_metric(self.total_distance_cm),
            "mean_swim_speed_cm_s": _format_metric(mean_speed_cm_s),
            "max_swim_speed_cm_s": _format_metric(self.max_speed_cm_s),
            "path_efficiency": _format_metric(path_efficiency),
            "heading_error_deg": _format_metric(heading_error_mean),
            "initial_heading_error_deg": _format_metric(self.initial_heading_error_deg),
            "mean_distance_to_platform_cm": _format_metric(mean_distance_to_platform),
            "distance_to_platform_mean_cm": _format_metric(mean_distance_to_platform),
            "gallagher_proximity_score": _format_metric(mean_distance_to_platform),
            "cipl": _format_metric(cipl),
            "target_quadrant": self.target_quadrant,
            "target_quadrant_label": self.quadrant_label(self.target_quadrant),
            "target_quadrant_time_s": _format_seconds(target_time_s),
            "target_quadrant_percent": round(target_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "target_quadrant_distance_cm": _format_metric(target_distance_cm),
            "thigmotaxis_time_s": _format_seconds(self.thigmotaxis_time_s),
            "thigmotaxis_percent": round(self.thigmotaxis_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "thigmotaxis_distance_cm": _format_metric(self.thigmotaxis_distance_cm),
            "whishaw_corridor_time_s": _format_seconds(self.whishaw_corridor_time_s),
            "whishaw_corridor_percent": round(self.whishaw_corridor_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "whishaw_corridor_distance_cm": _format_metric(self.whishaw_corridor_distance_cm),
            "moving_time_s": _format_seconds(self.moving_time_s),
            "floating_time_s": _format_seconds(self.floating_time_s),
            "pool_diameter_cm": _format_metric(self.pool_diameter_cm),
            "platform_radius_cm": _format_metric(self.platform_radius_cm),
            "platform_x_pool_cm": _format_metric(self.platform_center_cm[0]),
            "platform_y_pool_cm": _format_metric(self.platform_center_cm[1]),
            "start_position": self.start_position,
            "release_x_pool_cm": _format_metric(self.release_point_cm[0]),
            "release_y_pool_cm": _format_metric(self.release_point_cm[1]),
            "coordinate_rotation_deg": _format_metric(self.coordinate_rotation_deg),
            "quadrant_rotation_deg": _format_metric(self.quadrant_rotation_deg),
            "thigmotaxis_band_cm": _format_metric(self.thigmotaxis_band_cm),
            "platform_dwell_s": _format_seconds(self.platform_dwell_s),
            "track_smoothing_max_jump_cm": _format_metric(self.track_smoothing_max_jump_cm),
            "jump_filtered_count": int(self.jump_filtered_count),
            "movement_threshold_cm_s": _format_metric(self.movement_threshold_cm_s),
            "qc_flags": "|".join(sorted(self.qc_flags)),
        }
        summary.update(
            self._classify_search_strategy(
                total_time_s=total_time_s,
                detection_rate=detection_rate,
                path_for_efficiency=float(path_for_efficiency or 0.0),
                heading_error_mean=heading_error_mean,
                mean_distance_to_platform=float(mean_distance_to_platform),
                target_time_s=float(target_time_s),
            )
        )
        for key in QUADRANTS:
            time_s = self.quadrant_time_s[key]
            summary[f"quadrant_{key.lower()}_time_s"] = _format_seconds(time_s)
            summary[f"quadrant_{key.lower()}_percent"] = round(time_s / max(total_time_s, 1e-6) * 100.0, 2)
            summary[f"quadrant_{key.lower()}_distance_cm"] = _format_metric(self.quadrant_distance_cm[key])
        summary.update(self.geometry.as_config_summary())
        return summary

    def build_time_bins(self, bin_seconds: float = 10.0) -> List[Dict[str, Any]]:
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
                    "duration_s": 0.0,
                    "distance_cm": 0.0,
                    "platform_time_s": 0.0,
                    "platform_crossings": 0,
                    "thigmotaxis_time_s": 0.0,
                    "whishaw_corridor_time_s": 0.0,
                    "quadrant_time_s": {key: 0.0 for key in QUADRANTS},
                },
            )
            duration = float(row.get("dt_s") or (1.0 / max(self.fps, 1e-6)))
            bucket["duration_s"] += duration
            bucket["distance_cm"] += float(row.get("step_distance_cm") or 0.0)
            bucket["platform_time_s"] += duration if row.get("in_platform_zone") else 0.0
            bucket["platform_crossings"] += int(bool(row.get("platform_crossing_event")))
            bucket["thigmotaxis_time_s"] += duration if row.get("in_thigmotaxis") else 0.0
            bucket["whishaw_corridor_time_s"] += duration if row.get("in_whishaw_corridor") else 0.0
            quadrant = row.get("quadrant")
            if quadrant in QUADRANTS:
                bucket["quadrant_time_s"][quadrant] += duration

        out: List[Dict[str, Any]] = []
        for idx in sorted(bins):
            bucket = bins[idx]
            duration_s = float(bucket["duration_s"])
            item: Dict[str, Any] = {
                "roi_id": bucket["roi_id"],
                "roi_name": bucket["roi_name"],
                "bin_index": bucket["bin_index"],
                "start_s": round(bucket["start_s"], 2),
                "end_s": round(bucket["end_s"], 2),
                "duration_s": _format_seconds(duration_s),
                "distance_cm": _format_metric(bucket["distance_cm"]),
                "mean_swim_speed_cm_s": _format_metric(bucket["distance_cm"] / max(duration_s, 1e-6)),
                "platform_time_s": _format_seconds(bucket["platform_time_s"]),
                "platform_crossings": int(bucket["platform_crossings"]),
                "thigmotaxis_time_s": _format_seconds(bucket["thigmotaxis_time_s"]),
                "whishaw_corridor_time_s": _format_seconds(bucket["whishaw_corridor_time_s"]),
            }
            for key in QUADRANTS:
                value = bucket["quadrant_time_s"][key]
                item[f"quadrant_{key.lower()}_time_s"] = _format_seconds(value)
                item[f"quadrant_{key.lower()}_percent"] = round(value / max(duration_s, 1e-6) * 100.0, 2)
            out.append(item)
        return out


def _zone_color(result: dict[str, Any]) -> Tuple[int, int, int]:
    if not result.get("detected"):
        return (120, 120, 120)
    if result.get("in_platform_zone"):
        return (34, 197, 94)
    if result.get("in_thigmotaxis"):
        return (249, 115, 22)
    return (59, 130, 246)


def _draw_pool_polyline(output: np.ndarray, tracker: WaterMazeTracker, points_cm: np.ndarray, color: Tuple[int, int, int], thickness: int) -> None:
    pts = tracker.geometry.pool_to_image_points(points_cm).astype(np.int32)
    cv2.polylines(output, [pts], True, color, thickness, cv2.LINE_AA)


def draw_water_maze_results(
    frame: np.ndarray,
    tracker_results: Sequence[Tuple[WaterMazeTracker, Dict[str, Any]]],
    inplace: bool = False,
) -> np.ndarray:
    output = frame if inplace else frame.copy()
    for tracker, result in tracker_results:
        color = _zone_color(result)
        boundary = tracker.geometry.image_boundary_points(count=180).astype(np.int32)
        cv2.polylines(output, [boundary], True, color, 2, cv2.LINE_AA)
        if tracker.has_custom_detection_mask and tracker.detection_polygon_img is not None:
            cv2.polylines(output, [tracker.detection_polygon_img.astype(np.int32)], True, (245, 158, 11), 1, cv2.LINE_AA)

        radius = tracker.pool_radius_cm
        for axis in tracker.coordinate_axis_segments_cm():
            pts = tracker.geometry.pool_to_image_points(axis).astype(np.int32)
            cv2.line(output, tuple(pts[0]), tuple(pts[1]), (148, 163, 184), 1, cv2.LINE_AA)
        for key, label_point_cm in tracker.coordinate_quadrant_label_points_cm().items():
            label_img = tracker.geometry.pool_to_image(label_point_cm)
            cv2.putText(
                output,
                tracker.quadrant_label(key),
                (int(label_img[0]) - 8, int(label_img[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (226, 232, 240),
                1,
                cv2.LINE_AA,
            )

        thig_radius = max(0.0, radius - tracker.thigmotaxis_band_cm)
        if thig_radius > 0:
            _draw_pool_polyline(output, tracker, _circle_points(thig_radius, count=160), (245, 158, 11), 1)

        platform_circle = _circle_points(tracker.platform_radius_cm, count=80) + np.asarray(tracker.platform_center_cm, dtype=np.float32)
        _draw_pool_polyline(output, tracker, platform_circle, (34, 197, 94), 2)

        release_img = tracker.geometry.pool_to_image(tracker.release_point_cm)
        platform_img = tracker.geometry.pool_to_image(tracker.platform_center_cm)
        cv2.line(output, (int(release_img[0]), int(release_img[1])), (int(platform_img[0]), int(platform_img[1])), (14, 165, 233), 1, cv2.LINE_AA)

        contour = result.get("contour")
        centroid = result.get("centroid")
        if contour is not None:
            cv2.drawContours(output, [contour], -1, color, 2)
        if centroid is not None:
            cv2.circle(output, (int(centroid[0]), int(centroid[1])), 4, color, -1)
        traj = result.get("trajectory_px") or []
        for idx in range(1, len(traj)):
            cv2.line(
                output,
                (int(traj[idx - 1][0]), int(traj[idx - 1][1])),
                (int(traj[idx][0]), int(traj[idx][1])),
                color,
                1,
                cv2.LINE_AA,
            )

        min_x = int(np.min(boundary[:, 0]))
        min_y = int(np.min(boundary[:, 1]))
        label = tracker.roi_name
        sub = f"{result.get('quadrant', 'Unknown')} | entries {int(result.get('platform_entries') or 0)}"
        if result.get("success"):
            sub = f"Platform | {tracker.escape_latency_s:.1f}s" if tracker.escape_latency_s is not None else "Platform"
        if not result.get("in_stats_window", True):
            sub = f"{sub} | paused"
        box_w = max(210, int(max(len(label), len(sub)) * 9.0))
        top = max(0, min_y - 44)
        cv2.rectangle(output, (min_x, top), (min_x + box_w, top + 42), (8, 15, 28), -1)
        cv2.rectangle(output, (min_x, top), (min_x + box_w, top + 42), color, 1)
        cv2.putText(output, label, (min_x + 8, top + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(output, sub, (min_x + 8, top + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 210, 220), 1, cv2.LINE_AA)
    return output


def _pool_canvas(
    tracker: WaterMazeTracker,
    pixels_per_cm: int = 6,
    padding_cm: float = 6.0,
    style: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, Point]:
    radius = float(tracker.pool_radius_cm)
    pad = float(padding_cm)
    size = max(240, int(round((radius * 2.0 + pad * 2.0) * pixels_per_cm)))
    center = (size * 0.5, size * 0.5)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(round(radius * pixels_per_cm)), 255, -1, cv2.LINE_AA)
    canvas = np.full((size, size, 3), style_color(style, "trackmap", "background"), dtype=np.uint8)
    return canvas, mask, center


def _pool_cm_to_canvas(point_cm: Point, center: Point, pixels_per_cm: int, shape: Tuple[int, int]) -> tuple[int, int]:
    x = int(np.clip(round(center[0] + float(point_cm[0]) * pixels_per_cm), 0, shape[1] - 1))
    y = int(np.clip(round(center[1] + float(point_cm[1]) * pixels_per_cm), 0, shape[0] - 1))
    return x, y


def _display_cm_to_canvas(tracker: WaterMazeTracker, point_cm: Point, center: Point, pixels_per_cm: int, shape: Tuple[int, int]) -> tuple[int, int]:
    display_point = tracker.point_to_coordinate_frame(point_cm) or point_cm
    return _pool_cm_to_canvas(display_point, center, pixels_per_cm, shape)


def _draw_standard_pool_guides(
    canvas: np.ndarray,
    tracker: WaterMazeTracker,
    center: Point,
    pixels_per_cm: int,
    style: Any | None = None,
) -> None:
    radius_px = int(round(tracker.pool_radius_cm * pixels_per_cm))
    center_i = (int(round(center[0])), int(round(center[1])))
    cv2.circle(canvas, center_i, radius_px, style_color(style, "trackmap", "border"), 2, cv2.LINE_AA)
    cv2.line(canvas, (center_i[0] - radius_px, center_i[1]), (center_i[0] + radius_px, center_i[1]), style_color(style, "trackmap", "region_outline"), 1, cv2.LINE_AA)
    cv2.line(canvas, (center_i[0], center_i[1] - radius_px), (center_i[0], center_i[1] + radius_px), style_color(style, "trackmap", "region_outline"), 1, cv2.LINE_AA)
    inner = max(0, int(round((tracker.pool_radius_cm - tracker.thigmotaxis_band_cm) * pixels_per_cm)))
    if inner:
        cv2.circle(canvas, center_i, inner, style_color(style, "trackmap", "thigmotaxis"), 1, cv2.LINE_AA)
    label_points = {
        "NE": (tracker.pool_radius_cm * 0.5, -tracker.pool_radius_cm * 0.5),
        "NW": (-tracker.pool_radius_cm * 0.5, -tracker.pool_radius_cm * 0.5),
        "SW": (-tracker.pool_radius_cm * 0.5, tracker.pool_radius_cm * 0.5),
        "SE": (tracker.pool_radius_cm * 0.5, tracker.pool_radius_cm * 0.5),
    }
    for key, label_point in label_points.items():
        label_px = _pool_cm_to_canvas(label_point, center, pixels_per_cm, canvas.shape[:2])
        cv2.putText(canvas, tracker.quadrant_label(key), (label_px[0] - 8, label_px[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, style_color(style, "trackmap", "text"), 1, cv2.LINE_AA)
    platform_px = _display_cm_to_canvas(tracker, tracker.platform_center_cm, center, pixels_per_cm, canvas.shape[:2])
    cv2.circle(canvas, platform_px, max(2, int(round(tracker.platform_radius_cm * pixels_per_cm))), style_color(style, "trackmap", "platform"), 2, cv2.LINE_AA)
    release_px = _display_cm_to_canvas(tracker, tracker.release_point_cm, center, pixels_per_cm, canvas.shape[:2])
    cv2.circle(canvas, release_px, max(3, int(round(0.8 * pixels_per_cm))), style_color(style, "trackmap", "end"), -1, cv2.LINE_AA)
    cv2.putText(canvas, tracker.quadrant_label(), (platform_px[0] + 6, platform_px[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, style_color(style, "trackmap", "text"), 1, cv2.LINE_AA)


def render_water_maze_heatmap(
    tracker: WaterMazeTracker,
    output_path: str,
    pixels_per_cm: int = 6,
    style: Any | None = None,
) -> Optional[str]:
    if not tracker.trajectory_pool_cm_stats:
        return None
    canvas, mask, center = _pool_canvas(tracker, pixels_per_cm=pixels_per_cm, style=style)
    heat = np.zeros(mask.shape, dtype=np.float32)
    for point in tracker.trajectory_pool_cm_stats:
        px = _display_cm_to_canvas(tracker, point, center, pixels_per_cm, heat.shape)
        cv2.circle(heat, px, max(4, pixels_per_cm), 1.0, thickness=-1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(3, pixels_per_cm * 1.3), sigmaY=max(3, pixels_per_cm * 1.3))
    if float(heat.max()) <= 0.0:
        return None
    norm = np.clip(heat / heat.max(), 0.0, 1.0)
    image = classic_heatmap(norm, style)
    image[mask == 0] = style_color(style, "heatmap", "outside")
    _draw_standard_pool_guides(image, tracker, center, pixels_per_cm, style)
    image = append_heatmap_legend(image, style)
    if write_image_unicode(output_path, image):
        return output_path
    return None


def render_water_maze_trackmap(
    tracker: WaterMazeTracker,
    output_path: str,
    pixels_per_cm: int = 6,
    style: Any | None = None,
) -> Optional[str]:
    if not tracker.trajectory_pool_cm_stats:
        return None
    canvas, mask, center = _pool_canvas(tracker, pixels_per_cm=pixels_per_cm, style=style)
    canvas[mask == 0] = style_color(style, "trackmap", "background")
    cv2.circle(canvas, (int(center[0]), int(center[1])), int(round(tracker.pool_radius_cm * pixels_per_cm)), style_color(style, "trackmap", "region_fill"), -1, cv2.LINE_AA)
    points = [_display_cm_to_canvas(tracker, point, center, pixels_per_cm, canvas.shape[:2]) for point in tracker.trajectory_pool_cm_stats]
    _draw_standard_pool_guides(canvas, tracker, center, pixels_per_cm, style)
    path_thickness = classic_path_thickness(pixels_per_cm)
    marker_radius = classic_marker_radius(pixels_per_cm)
    for idx in range(1, len(points)):
        cv2.line(canvas, points[idx - 1], points[idx], style_color(style, "trackmap", "track"), path_thickness, cv2.LINE_AA)
    draw_start_end_markers(canvas, points, marker_radius, style)
    if write_image_unicode(output_path, canvas):
        return output_path
    return None


def render_water_maze_image_grid(
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
        image = read_image_unicode(path)
        if image is not None:
            images.append(image)
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
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        y = title_h + padding + row * (max_h + padding)
        x = padding + col * (max_w + padding)
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        cv2.rectangle(canvas, (x, y), (x + image.shape[1] - 1, y + image.shape[0] - 1), style_color(style, "trackmap", "grid_border"), 1, cv2.LINE_AA)
    if write_image_unicode(output_path, canvas):
        return output_path
    return None
