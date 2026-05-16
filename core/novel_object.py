#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Novel Object Recognition (NOR) helpers.

This module provides per-ROI tracking, object-exploration metrics, perspective
normalization, and visualization helpers for a multi-ROI NOR workflow.
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from file_utils import ascii_slug, write_image_unicode
from spatial_render import (
    CLASSIC_BG,
    CLASSIC_BORDER,
    CLASSIC_GUIDE_FILL,
    CLASSIC_GUIDE_OUTLINE,
    CLASSIC_HEAT_OUTLINE,
    CLASSIC_TEXT,
    CLASSIC_TRACK,
    classic_border_thickness,
    classic_guide_thickness,
    classic_heatmap,
    classic_marker_radius,
    classic_path_thickness,
    draw_start_end_markers,
    fill_rect_alpha,
)

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None
    ImageDraw = None
    ImageFont = None


def _order_quad(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Return 4 points ordered as TL, TR, BR, BL."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Novel object recognition arena calibration requires exactly 4 points.")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def _normalize_polygon(points: Optional[Sequence[Sequence[float]]], min_points: int = 3) -> Optional[np.ndarray]:
    if points is None:
        return None

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < min_points:
        raise ValueError(f"Novel object detection mask requires at least {min_points} points.")
    return pts


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _unit_vector(vec: Sequence[float]) -> Optional[np.ndarray]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        return None
    norm = float(np.linalg.norm(arr[:2]))
    if norm < 1e-6:
        return None
    return arr[:2] / norm


def _angle_deg(vec_a: Sequence[float], vec_b: Sequence[float]) -> Optional[float]:
    a = _unit_vector(vec_a)
    b = _unit_vector(vec_b)
    if a is None or b is None:
        return None
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def sanitize_name(text: str, fallback: str = "nor") -> str:
    return ascii_slug(text, fallback=fallback, max_length=80)


_TEXT_FONT_CACHE: Dict[int, Any] = {}
_TEXT_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\simkai.ttf",
]


def _safe_ascii_text(text: Any) -> str:
    return str(text or "").encode("ascii", "replace").decode("ascii")


def _bgr_to_rgb(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (int(color[2]), int(color[1]), int(color[0]))


def _short_text(text: Any, max_chars: int = 8) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    if max_chars <= 1:
        return value[:max_chars]
    return value[: max_chars - 1] + "…"


def _get_text_font(size: int):
    if ImageFont is None:
        return None

    size = max(12, int(size))
    cached = _TEXT_FONT_CACHE.get(size)
    if cached is not None:
        return cached

    font = None
    for path in _TEXT_FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            font = ImageFont.truetype(path, size=size, encoding="utf-8")
            break
        except Exception:
            continue

    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    _TEXT_FONT_CACHE[size] = font
    return font


class _UnicodeTextPainter:
    """Draw Unicode text on OpenCV images, falling back to ASCII-safe labels if needed."""

    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self._pil_ready = Image is not None and ImageDraw is not None and ImageFont is not None

    def text(
        self,
        position: Tuple[int, int],
        text: Any,
        color: Tuple[int, int, int],
        font_size: int = 18,
        stroke_width: int = 1,
        stroke_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        label = str(text or "")
        x = int(position[0])
        y = int(position[1])

        font = _get_text_font(font_size)
        if not self._pil_ready or font is None:
            cv2.putText(
                self.image,
                _safe_ascii_text(label),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.45, font_size / 32.0),
                color,
                max(1, stroke_width),
                cv2.LINE_AA,
            )
            return

        # Convert the current image on each draw so earlier/later OpenCV drawing is preserved.
        pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text(
            (x, y),
            label,
            font=font,
            fill=_bgr_to_rgb(color),
            stroke_width=max(0, int(stroke_width)),
            stroke_fill=_bgr_to_rgb(stroke_color),
        )
        self.image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    def measure(self, text: Any, font_size: int = 18) -> Tuple[int, int]:
        label = str(text or "")
        font = _get_text_font(font_size)
        if not self._pil_ready or font is None:
            return (max(12, int(len(label) * font_size * 0.6)), max(12, int(font_size)))

        try:
            bbox = font.getbbox(label, stroke_width=1)
            return (max(1, int(bbox[2] - bbox[0])), max(1, int(bbox[3] - bbox[1])))
        except Exception:
            return (max(12, int(len(label) * font_size * 0.6)), max(12, int(font_size)))

    def finish(self) -> np.ndarray:
        return self.image


def _preview_status_text(tracker: "NovelObjectTracker", result: Dict[str, Any]) -> str:
    if not result.get("detected"):
        return "未检测到"

    current_idx = result.get("current_object_idx")
    if current_idx is not None and 0 <= int(current_idx) < len(tracker.objects):
        return f"探索: {_short_text(tracker.objects[int(current_idx)]['name'], 8)}"

    near_flags = result.get("near_flags") or []
    for idx, is_near in enumerate(near_flags):
        if is_near and idx < len(tracker.objects):
            return f"靠近: {_short_text(tracker.objects[idx]['name'], 8)}"

    return "状态: 场地"


class NovelObjectTracker:
    """Track one mouse inside one calibrated NOR arena."""

    def __init__(
        self,
        arena_points: Sequence[Sequence[float]],
        object_points: Sequence[Dict[str, Any]],
        arena_side_cm: Optional[float] = None,
        arena_width_cm: Optional[float] = None,
        arena_height_cm: Optional[float] = None,
        fps: float = 30.0,
        roi_id: int = 1,
        roi_name: Optional[str] = None,
        phase: str = "recognition",
        time_config: Optional[Dict[str, Any]] = None,
        global_start_time_s: float = 0.0,
        detection_points: Optional[Sequence[Sequence[float]]] = None,
        center_ratio_x: float = 0.5,
        center_ratio_y: float = 0.5,
        interaction_radius_cm: float = 4.0,
        exploration_max_speed_cm_s: float = 12.0,
        use_head_direction: bool = True,
        head_angle_threshold_deg: float = 50.0,
        head_confidence_threshold: float = 0.35,
        body_gray_threshold: int = 95,
        mouse_is_light: bool = False,
        movement_threshold_cm_s: float = 2.0,
        min_valid_exploration_s: float = 20.0,
    ) -> None:
        if arena_side_cm is not None:
            arena_width_cm = arena_side_cm
            arena_height_cm = arena_side_cm

        if arena_width_cm is None or arena_height_cm is None:
            raise ValueError("Novel object tracker requires arena size in centimeters.")

        if len(object_points) != 2:
            raise ValueError("Novel object tracker requires exactly 2 object points per ROI.")

        self.roi_id = int(roi_id)
        self.roi_name = roi_name or f"ROI {self.roi_id}"
        self.phase = self._normalize_phase(phase)
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
        self.interaction_radius_cm = max(0.5, float(interaction_radius_cm))
        self.exploration_max_speed_cm_s = max(0.0, float(exploration_max_speed_cm_s))
        self.use_head_direction = bool(use_head_direction)
        self.head_angle_threshold_deg = float(np.clip(head_angle_threshold_deg, 5.0, 180.0))
        self.head_confidence_threshold = _clip01(head_confidence_threshold)
        self.body_gray_threshold = int(np.clip(int(body_gray_threshold), 0, 255))
        self.mouse_is_light = bool(mouse_is_light)
        self.movement_threshold_cm_s = max(0.0, float(movement_threshold_cm_s))
        self.min_valid_exploration_s = max(0.0, float(min_valid_exploration_s))

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

        self.objects = self._normalize_objects(object_points)
        self._center_rect_cm_cache: Optional[Tuple[float, float, float, float]] = None
        self._center_polygon_px_cache: Optional[np.ndarray] = None
        self._object_circle_polygon_cache: Dict[Tuple[float, float, float, int], np.ndarray] = {}

        self.detection_mask: Optional[np.ndarray] = None
        self.detection_bbox: Optional[Tuple[int, int, int, int]] = None
        self._cropped_detection_mask: Optional[np.ndarray] = None

        self.total_frames = 0
        self.frames_in_window = 0
        self.detected_frames = 0
        self.detected_frames_in_window = 0
        self.center_frames = 0
        self.moving_frames = 0
        self.immobile_frames = 0
        self.center_entries = 0
        self.center_latency_s: Optional[float] = None
        self.total_distance_cm = 0.0
        self.max_speed_cm_s = 0.0
        self.last_speed_cm_s = 0.0

        self.first_detected_time_abs: Optional[float] = None
        self.prev_all_arena_xy_cm: Optional[Tuple[float, float]] = None
        self.prev_stats_arena_xy_cm: Optional[Tuple[float, float]] = None
        self.last_all_arena_xy_cm: Optional[Tuple[float, float]] = None
        self.prev_centroid_px: Optional[Tuple[float, float]] = None
        self.prev_in_center_stats = False
        self.missing_frames = 0
        self.max_gap_frames = 5
        self.prev_head_vector_cm: Optional[Tuple[float, float]] = None
        self.last_head_point_px: Optional[Tuple[float, float]] = None
        self.last_head_point_cm: Optional[Tuple[float, float]] = None
        self.head_estimated_frames = 0
        self.head_confident_frames = 0
        self.head_supported_exploration_frames = 0
        self.fallback_exploration_frames = 0

        self.object_zone_frames = [0, 0]
        self.object_exploration_frames = [0, 0]
        self.object_zone_entries = [0, 0]
        self.object_bouts = [0, 0]
        self.object_latency_s: List[Optional[float]] = [None, None]
        self.prev_zone_flags = [False, False]
        self.prev_explore_flags = [False, False]
        self.current_exploring_idx: Optional[int] = None
        self.first_explored_object_idx: Optional[int] = None

        self.trajectory_px_live: deque[Tuple[float, float]] = deque(maxlen=450)
        self.trajectory_cm_stats: List[Tuple[float, float]] = []
        self.frame_records: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize_phase(phase: str) -> str:
        value = str(phase or "recognition").strip().lower()
        if value in {"training", "familiarization", "sample"}:
            return "familiarization"
        return "recognition"

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

    def _normalize_objects(self, object_points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, obj in enumerate(object_points):
            point = obj.get("point", obj)
            if isinstance(point, dict):
                x_px = float(point["x"])
                y_px = float(point["y"])
            else:
                x_px = float(point[0])
                y_px = float(point[1])

            role = str(obj.get("role") or "").strip().lower()
            if self.phase == "recognition":
                if role not in {"novel", "familiar"}:
                    role = "novel" if idx == 1 else "familiar"
            else:
                role = "familiar"

            default_name = f"物体 {idx + 1}"
            if self.phase == "recognition":
                default_name = "新奇物体" if role == "novel" else "熟悉物体"
            name = str(obj.get("name") or default_name).strip() or default_name
            cm_point = self.project_point_to_cm((x_px, y_px))
            out.append(
                {
                    "id": idx + 1,
                    "name": name,
                    "role": role,
                    "point_px": (x_px, y_px),
                    "point_cm": cm_point,
                }
            )

        if self.phase == "recognition":
            roles = [item["role"] for item in out]
            if roles.count("novel") != 1 or roles.count("familiar") != 1:
                out[0]["role"] = "familiar"
                out[1]["role"] = "novel"
        return out

    def initialize(self, frame_shape: Tuple[int, ...]) -> None:
        if self.detection_mask is not None:
            return

        self.detection_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.detection_mask, [self.detection_points.astype(np.int32)], 255)
        x, y, w, h = cv2.boundingRect(self.detection_points.astype(np.int32))
        self.detection_bbox = (x, y, w, h)
        self._cropped_detection_mask = self.detection_mask[y : y + h, x : x + w]

    def project_point_to_cm(self, point_px: Sequence[float]) -> Tuple[float, float]:
        src = np.array([[[float(point_px[0]), float(point_px[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, self.h_img_to_cm)[0, 0]
        x_cm = float(np.clip(dst[0], 0.0, self.arena_width_cm))
        y_cm = float(np.clip(dst[1], 0.0, self.arena_height_cm))
        return (x_cm, y_cm)

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

    def object_circle_to_image_polygon(self, center_cm: Tuple[float, float], radius_cm: float, samples: int = 40) -> np.ndarray:
        key = (round(float(center_cm[0]), 6), round(float(center_cm[1]), 6), round(float(radius_cm), 6), int(samples))
        cached = self._object_circle_polygon_cache.get(key)
        if cached is not None:
            return cached
        angles = np.linspace(0.0, 2.0 * math.pi, num=max(16, samples), endpoint=False, dtype=np.float32)
        circle = np.stack(
            [
                center_cm[0] + radius_cm * np.cos(angles),
                center_cm[1] + radius_cm * np.sin(angles),
            ],
            axis=1,
        ).astype(np.float32)
        circle[:, 0] = np.clip(circle[:, 0], 0.0, self.arena_width_cm)
        circle[:, 1] = np.clip(circle[:, 1], 0.0, self.arena_height_cm)
        poly = cv2.perspectiveTransform(circle.reshape(1, -1, 2), self.h_cm_to_img)[0]
        self._object_circle_polygon_cache[key] = poly.astype(np.int32)
        return self._object_circle_polygon_cache[key]

    def _zone_flags(self, arena_xy_cm: Tuple[float, float]) -> Dict[str, bool]:
        x_cm, y_cm = arena_xy_cm
        cx0, cy0, cx1, cy1 = self.get_center_rect_cm()
        in_center = (cx0 <= x_cm <= cx1) and (cy0 <= y_cm <= cy1)
        return {
            "in_center": in_center,
            "in_outer": not in_center,
        }

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

    @staticmethod
    def _proximity_label(exploring_idx: Optional[int], near_flags: Sequence[bool], objects: Sequence[Dict[str, Any]]) -> str:
        if exploring_idx is not None:
            return f"Exploring {objects[exploring_idx]['name']}"
        for idx, is_near in enumerate(near_flags):
            if is_near:
                return f"Near {objects[idx]['name']}"
        return "Arena"

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

    def _object_distances_cm(
        self,
        contour: Optional[np.ndarray],
        arena_xy_cm: Optional[Tuple[float, float]],
    ) -> List[Optional[float]]:
        distances: List[Optional[float]] = []

        contour_cm = None
        if contour is not None and isinstance(contour, np.ndarray) and contour.size > 0:
            pts = contour.reshape(-1, 1, 2).astype(np.float32)
            contour_cm = cv2.perspectiveTransform(pts, self.h_img_to_cm).reshape(-1, 2)
            contour_cm[:, 0] = np.clip(contour_cm[:, 0], 0.0, self.arena_width_cm)
            contour_cm[:, 1] = np.clip(contour_cm[:, 1], 0.0, self.arena_height_cm)

        for obj in self.objects:
            ox, oy = obj["point_cm"]
            if contour_cm is not None and contour_cm.size > 0:
                dists = np.sqrt((contour_cm[:, 0] - ox) ** 2 + (contour_cm[:, 1] - oy) ** 2)
                distances.append(float(np.min(dists)))
            elif arena_xy_cm is not None:
                distances.append(float(math.hypot(arena_xy_cm[0] - ox, arena_xy_cm[1] - oy)))
            else:
                distances.append(None)
        return distances

    @staticmethod
    def _contour_pointiness(points_px: np.ndarray, index: int, step: int) -> float:
        n = int(points_px.shape[0])
        if n < max(8, step * 2 + 1):
            return 0.0

        prev_pt = points_px[(index - step) % n]
        pt = points_px[index % n]
        next_pt = points_px[(index + step) % n]

        v1 = prev_pt - pt
        v2 = next_pt - pt
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cos_theta = float(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
        angle = float(math.degrees(math.acos(cos_theta)))
        return _clip01((180.0 - angle) / 180.0)

    @staticmethod
    def _largest_mask_component(mask: np.ndarray, min_area: int = 1) -> Optional[Dict[str, Any]]:
        if mask is None or mask.size == 0:
            return None
        binary = (mask > 0).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        best_label = None
        best_area = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= int(min_area) and area > best_area:
                best_label = label
                best_area = area
        if best_label is None:
            return None

        component = np.zeros_like(mask, dtype=np.uint8)
        component[labels == best_label] = 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        return {
            "mask": component,
            "contour": contour,
            "area": best_area,
            "centroid": (float(centroids[best_label][0]), float(centroids[best_label][1])),
            "bbox": tuple(int(v) for v in stats[best_label, :4]),
        }

    @staticmethod
    def _mask_contour_axis_ratio(contour: np.ndarray) -> float:
        points = contour.reshape(-1, 2).astype(np.float32)
        if points.shape[0] < 4:
            return 1.0
        centered = points - np.mean(points, axis=0)
        if float(np.max(np.linalg.norm(centered, axis=1))) < 1e-3:
            return 1.0
        try:
            cov = np.cov(centered.T)
            eigvals, _ = np.linalg.eigh(cov)
            eigvals = np.sort(eigvals)[::-1]
            return float((eigvals[0] + 1e-6) / (eigvals[1] + 1e-6)) if eigvals.size > 1 else 999.0
        except Exception:
            return 1.0

    @staticmethod
    def _radial_concentration(points: np.ndarray, origin: np.ndarray) -> float:
        vectors = points.astype(np.float32) - origin.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > 1e-3
        if int(np.count_nonzero(valid)) < 3:
            return 0.0
        units = vectors[valid] / norms[valid, None]
        return _clip01(float(np.linalg.norm(np.mean(units, axis=0))))

    def _select_body_component(self, local_gray: np.ndarray, whole_mask: np.ndarray) -> Optional[Dict[str, Any]]:
        whole_area = int(cv2.countNonZero(whole_mask))
        if whole_area < 20:
            return None

        whole_moments = cv2.moments(whole_mask)
        if whole_moments["m00"] == 0:
            return None
        whole_centroid = np.array(
            [whole_moments["m10"] / whole_moments["m00"], whole_moments["m01"] / whole_moments["m00"]],
            dtype=np.float32,
        )

        close_size = int(np.clip(round(math.sqrt(float(whole_area)) * 0.055), 3, 11))
        if close_size % 2 == 0:
            close_size += 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        candidates: List[Dict[str, Any]] = []
        threshold_specs = (("light_body", cv2.THRESH_BINARY),) if self.mouse_is_light else (("dark_body", cv2.THRESH_BINARY_INV),)
        for polarity, threshold_type in threshold_specs:
            _, raw = cv2.threshold(local_gray, self.body_gray_threshold, 255, threshold_type)
            body_mask = cv2.bitwise_and(raw, whole_mask)
            body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, close_kernel)
            body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, open_kernel)
            body_mask = cv2.bitwise_and(body_mask, whole_mask)
            component = self._largest_mask_component(body_mask, min_area=max(12, int(round(whole_area * 0.12))))
            if component is None:
                continue

            body_area = int(component["area"])
            body_ratio = body_area / max(float(whole_area), 1.0)
            if body_ratio < 0.16 or body_ratio > 0.98:
                continue

            centroid = np.asarray(component["centroid"], dtype=np.float32)
            center_distance = float(np.linalg.norm(centroid - whole_centroid))
            body_radius = max(math.sqrt(body_area / math.pi), 1.0)
            central_score = _clip01(1.0 - center_distance / max(body_radius * 1.8, 1.0))
            ratio_score = _clip01(1.0 - abs(body_ratio - 0.74) / 0.50)
            if body_ratio > 0.94:
                ratio_score *= 0.45
            axis_ratio = self._mask_contour_axis_ratio(component["contour"])
            axis_score = _clip01((axis_ratio - 1.0) / 2.5)
            score = 0.58 * ratio_score + 0.28 * central_score + 0.14 * axis_score

            component.update(
                {
                    "polarity": polarity,
                    "body_ratio": body_ratio,
                    "whole_area": whole_area,
                    "axis_ratio": axis_ratio,
                    "score": score,
                }
            )
            candidates.append(component)

        if not candidates:
            return None
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[0]

    def _select_tail_component(
        self,
        tail_region: np.ndarray,
        body_mask: np.ndarray,
        body_contour: np.ndarray,
        body_centroid: np.ndarray,
        body_radius: float,
        whole_area: int,
    ) -> Optional[Dict[str, Any]]:
        if tail_region is None or int(cv2.countNonZero(tail_region)) < 3:
            return None

        dist_to_body = cv2.distanceTransform(cv2.bitwise_not(body_mask), cv2.DIST_L2, 5)
        root_band_px = float(np.clip(round(body_radius * 0.30), 4.0, 10.0))

        tail_core = np.zeros_like(tail_region, dtype=np.uint8)
        tail_core[(tail_region > 0) & (dist_to_body >= root_band_px)] = 255
        if int(cv2.countNonZero(tail_core)) < 3:
            return None

        body_points = body_contour.reshape(-1, 2).astype(np.float32)
        if body_points.shape[0] < 4:
            return None

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((tail_core > 0).astype(np.uint8), 8)
        candidates: List[Dict[str, Any]] = []
        min_area = max(4, int(round(whole_area * 0.002)))
        max_area = max(20, int(round(whole_area * 0.28)))

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue

            component_mask = np.zeros_like(tail_core, dtype=np.uint8)
            component_mask[labels == label] = 255
            ys, xs = np.where(component_mask > 0)
            if xs.size < 3:
                continue

            points = np.stack([xs, ys], axis=1).astype(np.float32)
            dist_values = dist_to_body[ys, xs]
            proximal = points[int(np.argmin(dist_values))]
            tail_side_unit = _unit_vector(proximal - body_centroid)
            if tail_side_unit is None:
                continue

            body_projections = (body_points - body_centroid) @ tail_side_unit
            max_body_projection = float(np.max(body_projections))
            root_pool = body_points[body_projections >= max_body_projection - max(2.0, body_radius * 0.20)]
            if root_pool.size == 0:
                root_pool = body_points
            root = root_pool[int(np.argmin(np.linalg.norm(root_pool - proximal, axis=1)))]

            tip = points[int(np.argmax(np.linalg.norm(points - root, axis=1)))]
            tail_length = float(np.linalg.norm(tip - root))
            if tail_length < max(12.0, body_radius * 0.65):
                continue
            tail_unit = _unit_vector(tip - root)
            root_unit = _unit_vector(root - body_centroid)
            if tail_unit is None or root_unit is None:
                continue

            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            tail_contour = max(contours, key=cv2.contourArea)

            axis_ratio = self._mask_contour_axis_ratio(tail_contour)
            bbox_w = max(1, int(stats[label, cv2.CC_STAT_WIDTH]))
            bbox_h = max(1, int(stats[label, cv2.CC_STAT_HEIGHT]))
            bbox_aspect = max(bbox_w, bbox_h) / max(1.0, float(min(bbox_w, bbox_h)))
            area_ratio = area / max(float(whole_area), 1.0)

            directional_score = _clip01((self._radial_concentration(points, body_centroid) - 0.38) / 0.52)
            length_score = _clip01((tail_length / max(body_radius, 1e-6) - 0.30) / 1.20)
            slenderness = (tail_length * tail_length) / max(float(area), 1.0)
            thin_score = max(
                _clip01((axis_ratio - 1.7) / 5.5),
                _clip01((bbox_aspect - 1.7) / 5.5),
                _clip01((slenderness - 3.0) / 22.0),
            )
            area_score = _clip01(1.0 - abs(area_ratio - 0.07) / 0.18)
            away_score = _clip01(
                (
                    float(np.linalg.norm(tip - body_centroid))
                    - float(np.linalg.norm(root - body_centroid))
                )
                / max(body_radius * 1.2, 1.0)
            )
            outward_score = _clip01((float(np.dot(tail_unit, root_unit)) + 0.15) / 1.15)

            score = (
                0.24 * length_score
                + 0.23 * thin_score
                + 0.21 * directional_score
                + 0.18 * outward_score
                + 0.08 * away_score
                + 0.06 * area_score
            )
            candidates.append(
                {
                    "mask": component_mask,
                    "contour": tail_contour,
                    "area": area,
                    "area_ratio": area_ratio,
                    "axis_ratio": axis_ratio,
                    "root": root.astype(np.float32),
                    "tip": tip.astype(np.float32),
                    "length": tail_length,
                    "score": score,
                    "touch_score": 1.0,
                    "length_score": length_score,
                    "thin_score": thin_score,
                    "area_score": area_score,
                    "directional_score": directional_score,
                    "away_score": away_score,
                    "outward_score": outward_score,
                    "root_band_px": root_band_px,
                }
            )

        if not candidates:
            return None
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[0]

    def _segment_body_tail_from_gray(
        self,
        contour: np.ndarray,
        gray: Optional[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        if gray is None or contour is None:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 2 or h <= 2:
            return None
        pad = 2
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(gray.shape[1], x + w + pad)
        y1 = min(gray.shape[0], y + h + pad)
        if x1 <= x0 or y1 <= y0:
            return None

        local_gray = gray[y0:y1, x0:x1]
        shifted_contour = contour.astype(np.int32) - np.array([[[x0, y0]]], dtype=np.int32)
        whole_mask = np.zeros(local_gray.shape[:2], dtype=np.uint8)
        cv2.drawContours(whole_mask, [shifted_contour], -1, 255, thickness=-1)
        whole_area = int(cv2.countNonZero(whole_mask))
        if whole_area < 20:
            return None

        body = self._select_body_component(local_gray, whole_mask)
        if body is None:
            return None

        body_mask = body["mask"]
        body_contour_local = body["contour"]
        body_centroid_local = np.asarray(body["centroid"], dtype=np.float32)
        body_area = int(body["area"])
        body_radius = max(math.sqrt(body_area / math.pi), 1.0)

        tail_region = cv2.bitwise_and(whole_mask, cv2.bitwise_not(body_mask))
        if cv2.countNonZero(tail_region) > 0:
            tail_region = cv2.bitwise_and(tail_region, whole_mask)

        tail = self._select_tail_component(
            tail_region,
            body_mask,
            body_contour_local,
            body_centroid_local,
            body_radius,
            whole_area,
        )

        def _to_global_point(point: Sequence[float]) -> Tuple[float, float]:
            return (float(point[0]) + float(x0), float(point[1]) + float(y0))

        body_contour_global = body_contour_local + np.array([[[x0, y0]]], dtype=np.int32)
        body_centroid_px = _to_global_point(body_centroid_local)
        body_centroid_cm = self.project_point_to_cm(body_centroid_px)

        result: Dict[str, Any] = {
            "body_contour": body_contour_global,
            "body_centroid_px": body_centroid_px,
            "body_centroid_cm": body_centroid_cm,
            "body_area_px": body_area,
            "body_ratio": float(body["body_ratio"]),
            "body_threshold_polarity": body["polarity"],
            "body_axis_ratio": float(body["axis_ratio"]),
            "whole_area_px": whole_area,
        }

        if tail is None:
            return result

        tail_tip_px = _to_global_point(tail["tip"])
        tail_root_px = _to_global_point(tail["root"])
        tail_contour_global = tail["contour"] + np.array([[[x0, y0]]], dtype=np.int32)
        head_unit = _unit_vector(body_centroid_local - tail["root"])
        if head_unit is None:
            return result

        body_points = body_contour_local.reshape(-1, 2).astype(np.float32)
        projections = (body_points - body_centroid_local) @ head_unit
        if projections.size == 0:
            return result
        nose_local = body_points[int(np.argmax(projections))]
        nose_projection = float(np.max(projections))
        if nose_projection < max(3.0, body_radius * 0.18):
            return result

        nose_px = _to_global_point(nose_local)
        nose_cm = self.project_point_to_cm(nose_px)
        vector_cm = (
            float(nose_cm[0] - body_centroid_cm[0]),
            float(nose_cm[1] - body_centroid_cm[1]),
        )
        if _unit_vector(vector_cm) is None:
            return result

        body_score = _clip01(0.65 * body["score"] + 0.35 * _clip01((body["body_ratio"] - 0.18) / 0.55))
        nose_score = _clip01((nose_projection / max(body_radius, 1e-6) - 0.18) / 0.75)
        tail_score = _clip01(
            0.30 * tail["length_score"]
            + 0.30 * tail["thin_score"]
            + 0.18 * tail["area_score"]
            + 0.12 * tail["touch_score"]
            + 0.10 * tail["score"]
        )
        confidence = _clip01(0.18 + 0.34 * tail_score + 0.26 * body_score + 0.22 * nose_score)

        result.update(
            {
                "nose_point_px": nose_px,
                "nose_point_cm": nose_cm,
                "vector_cm": vector_cm,
                "tail_point_px": tail_tip_px,
                "tail_root_px": tail_root_px,
                "tail_contour": tail_contour_global,
                "tail_confidence": tail_score,
                "tail_area_ratio": float(tail["area_ratio"]),
                "tail_axis_ratio": float(tail["axis_ratio"]),
                "tail_length_px": float(tail["length"]),
                "confidence": confidence,
            }
        )
        return result

    def _head_state_from_body_tail_segment(self, segment: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not segment or segment.get("nose_point_px") is None or segment.get("vector_cm") is None:
            return None

        body_centroid_cm = segment.get("body_centroid_cm")
        nose_cm = segment.get("nose_point_cm")
        if body_centroid_cm is None or nose_cm is None:
            return None

        nose_distances_cm: List[Optional[float]] = []
        object_angles_deg: List[Optional[float]] = []
        for obj in self.objects:
            nose_vec = (
                float(obj["point_cm"][0] - nose_cm[0]),
                float(obj["point_cm"][1] - nose_cm[1]),
            )
            nose_distances_cm.append(float(math.hypot(nose_vec[0], nose_vec[1])))
            body_vec = (
                float(obj["point_cm"][0] - body_centroid_cm[0]),
                float(obj["point_cm"][1] - body_centroid_cm[1]),
            )
            object_angles_deg.append(_angle_deg(segment["vector_cm"], body_vec))

        return {
            "point_px": segment["nose_point_px"],
            "point_cm": nose_cm,
            "vector_cm": segment["vector_cm"],
            "tail_point_px": segment.get("tail_point_px"),
            "tail_point_cm": self.project_point_to_cm(segment["tail_point_px"]) if segment.get("tail_point_px") is not None else None,
            "tail_root_px": segment.get("tail_root_px"),
            "confidence": float(segment.get("confidence") or 0.0),
            "estimated": True,
            "axis_ratio": segment.get("body_axis_ratio"),
            "method": "body_tail_threshold",
            "body_contour": segment.get("body_contour"),
            "body_centroid": segment.get("body_centroid_px"),
            "body_centroid_cm": body_centroid_cm,
            "body_threshold_polarity": segment.get("body_threshold_polarity"),
            "tail_contour": segment.get("tail_contour"),
            "tail_confidence": segment.get("tail_confidence"),
            "nose_distances_cm": nose_distances_cm,
            "object_angles_deg": object_angles_deg,
        }

    def _estimate_head_direction_pca_endpoints(
        self,
        contour: np.ndarray,
        centroid_px: Tuple[float, float],
        origin_cm: Tuple[float, float],
        motion_vector_cm: Optional[Tuple[float, float]],
        body_segment: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        points_px = contour.reshape(-1, 2).astype(np.float32)
        if points_px.shape[0] < 10:
            return None

        centroid_arr = np.asarray(centroid_px, dtype=np.float32)
        centered = points_px - centroid_arr
        if float(np.max(np.linalg.norm(centered, axis=1))) < 1e-3:
            return None

        try:
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
        except Exception:
            return None
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        major_axis = _unit_vector(eigvecs[:, order[0]])
        if major_axis is None:
            return None

        axis_ratio = float((eigvals[0] + 1e-6) / (eigvals[1] + 1e-6)) if eigvals.size > 1 else 999.0
        projections = centered @ major_axis
        if float(np.max(projections) - np.min(projections)) < 1e-3:
            return None

        candidate_count = int(np.clip(points_px.shape[0] // 12, 4, 12))
        sorted_idx = np.argsort(projections)
        step = max(2, min(10, max(2, points_px.shape[0] // 16)))
        projection_scale = max(float(np.max(np.abs(projections))), 1e-6)
        movement_unit = _unit_vector(motion_vector_cm) if motion_vector_cm is not None else None
        previous_head_unit = _unit_vector(self.prev_head_vector_cm) if self.prev_head_vector_cm is not None else None

        body_to_object_vectors = []
        body_to_object_distances = []
        for obj in self.objects:
            vec = (
                float(obj["point_cm"][0] - origin_cm[0]),
                float(obj["point_cm"][1] - origin_cm[1]),
            )
            body_to_object_vectors.append(vec)
            body_to_object_distances.append(float(math.hypot(vec[0], vec[1])))

        def _support_score(
            unit_head: np.ndarray,
            obj_vec: Optional[Tuple[float, float]] = None,
            obj_distance: Optional[float] = None,
        ) -> Tuple[float, float, float]:
            continuity = 0.5
            if previous_head_unit is not None:
                continuity = (float(np.dot(unit_head, previous_head_unit)) + 1.0) * 0.5

            motion = 0.5
            if movement_unit is not None:
                motion = (float(np.dot(unit_head, movement_unit)) + 1.0) * 0.5

            object_align = 0.5
            if obj_vec is not None and obj_distance is not None and obj_distance <= self.interaction_radius_cm * 2.0:
                obj_unit = _unit_vector(obj_vec)
                if obj_unit is not None:
                    proximity_weight = _clip01(1.0 - obj_distance / max(self.interaction_radius_cm * 2.0, 1e-6))
                    object_align = 0.5 * (1.0 - proximity_weight) + ((float(np.dot(unit_head, obj_unit)) + 1.0) * 0.5) * proximity_weight
            return continuity, motion, object_align

        def _pick_candidate(indices: np.ndarray) -> Optional[Dict[str, Any]]:
            best = None
            for raw_idx in indices.tolist():
                idx = int(raw_idx)
                point_px = (float(points_px[idx][0]), float(points_px[idx][1]))
                point_cm = self.project_point_to_cm(point_px)
                vector_cm = (
                    float(point_cm[0] - origin_cm[0]),
                    float(point_cm[1] - origin_cm[1]),
                )
                unit_head = _unit_vector(vector_cm)
                if unit_head is None:
                    continue

                pointiness = self._contour_pointiness(points_px, idx, step)
                extremeness = abs(float(projections[idx])) / projection_scale
                nearest_idx = int(
                    min(
                        range(len(self.objects)),
                        key=lambda obj_idx: math.hypot(
                            point_cm[0] - self.objects[obj_idx]["point_cm"][0],
                            point_cm[1] - self.objects[obj_idx]["point_cm"][1],
                        ),
                    )
                )
                continuity, motion, object_align = _support_score(
                    unit_head,
                    body_to_object_vectors[nearest_idx],
                    body_to_object_distances[nearest_idx],
                )
                score = (
                    0.35 * pointiness
                    + 0.20 * extremeness
                    + 0.20 * continuity
                    + 0.15 * motion
                    + 0.10 * object_align
                )
                candidate = {
                    "idx": idx,
                    "point_px": point_px,
                    "point_cm": point_cm,
                    "vector_cm": vector_cm,
                    "pointiness": pointiness,
                    "continuity": continuity,
                    "motion": motion,
                    "object_align": object_align,
                    "score": score,
                }
                if best is None or candidate["score"] > best["score"]:
                    best = candidate
            return best

        forward = _pick_candidate(sorted_idx[-candidate_count:])
        backward = _pick_candidate(sorted_idx[:candidate_count])
        if forward is None and backward is None:
            return None
        if backward is None:
            winner = forward
            other = None
        elif forward is None:
            winner = backward
            other = None
        else:
            winner, other = (forward, backward) if forward["score"] >= backward["score"] else (backward, forward)
        if winner is None:
            return None

        score_margin = winner["score"] - (other["score"] if other is not None else 0.0)
        support_strength = max(winner["continuity"], winner["motion"], winner["object_align"])
        elongation = _clip01((axis_ratio - 1.2) / 2.0)
        confidence = _clip01(
            0.38 * _clip01(score_margin)
            + 0.24 * elongation
            + 0.20 * winner["pointiness"]
            + 0.18 * support_strength
        )

        nose_distances_cm: List[Optional[float]] = []
        object_angles_deg: List[Optional[float]] = []
        for obj in self.objects:
            nose_vec = (
                float(obj["point_cm"][0] - winner["point_cm"][0]),
                float(obj["point_cm"][1] - winner["point_cm"][1]),
            )
            nose_distances_cm.append(float(math.hypot(nose_vec[0], nose_vec[1])))
            body_vec = (
                float(obj["point_cm"][0] - origin_cm[0]),
                float(obj["point_cm"][1] - origin_cm[1]),
            )
            object_angles_deg.append(_angle_deg(winner["vector_cm"], body_vec))

        return {
            "point_px": winner["point_px"],
            "point_cm": winner["point_cm"],
            "vector_cm": winner["vector_cm"],
            "tail_point_px": None,
            "tail_point_cm": None,
            "tail_root_px": None,
            "confidence": confidence,
            "estimated": True,
            "axis_ratio": axis_ratio,
            "method": "body_pca_legacy" if body_segment is not None else "pca_legacy",
            "body_contour": body_segment.get("body_contour") if body_segment is not None else None,
            "body_centroid": body_segment.get("body_centroid_px") if body_segment is not None else None,
            "body_centroid_cm": body_segment.get("body_centroid_cm") if body_segment is not None else None,
            "body_threshold_polarity": body_segment.get("body_threshold_polarity") if body_segment is not None else None,
            "tail_contour": body_segment.get("tail_contour") if body_segment is not None else None,
            "tail_confidence": body_segment.get("tail_confidence") if body_segment is not None else None,
            "nose_distances_cm": nose_distances_cm,
            "object_angles_deg": object_angles_deg,
        }

    def _estimate_head_direction(
        self,
        contour: Optional[np.ndarray],
        centroid_px: Optional[Tuple[float, float]],
        arena_xy_cm: Optional[Tuple[float, float]],
        motion_vector_cm: Optional[Tuple[float, float]],
        gray: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        empty = {
            "point_px": None,
            "point_cm": None,
            "vector_cm": None,
            "tail_point_px": None,
            "tail_point_cm": None,
            "tail_root_px": None,
            "confidence": 0.0,
            "estimated": False,
            "axis_ratio": None,
            "method": None,
            "body_contour": None,
            "body_centroid": None,
            "body_centroid_cm": None,
            "body_threshold_polarity": None,
            "tail_contour": None,
            "tail_confidence": None,
            "nose_distances_cm": [None for _ in self.objects],
            "object_angles_deg": [None for _ in self.objects],
        }

        if contour is None or centroid_px is None or arena_xy_cm is None:
            return empty

        body_segment = self._segment_body_tail_from_gray(contour, gray)
        body_tail_state = self._head_state_from_body_tail_segment(body_segment) if body_segment is not None else None
        if body_tail_state is not None:
            return body_tail_state

        if body_segment is not None and body_segment.get("body_contour") is not None:
            contour = body_segment["body_contour"]
            centroid_px = tuple(float(v) for v in body_segment.get("body_centroid_px", centroid_px))
            arena_xy_cm = tuple(float(v) for v in body_segment.get("body_centroid_cm", arena_xy_cm))

        legacy_state = self._estimate_head_direction_pca_endpoints(
            contour,
            centroid_px,
            arena_xy_cm,
            motion_vector_cm,
            body_segment=body_segment,
        )
        if legacy_state is not None:
            return legacy_state

        return empty

    def update(
        self,
        frame: np.ndarray,
        current_time: float,
        gray: np.ndarray,
        bg_subtractor,
        morph_processor,
        detector,
        include_trajectory: bool = True,
    ) -> Dict[str, Any]:
        self.total_frames += 1
        detection, fg_mask = self.detect_mouse(frame, gray, bg_subtractor, morph_processor, detector)

        detected = detection is not None
        contour = detection.get("contour") if detection else None
        centroid_px: Optional[Tuple[float, float]] = None
        bbox = detection.get("bbox") if detection else None
        arena_xy_cm: Optional[Tuple[float, float]] = None

        if detected:
            self.detected_frames += 1
            centroid_px = tuple(float(v) for v in detection["centroid"])
            arena_xy_cm = self.project_point_to_cm(centroid_px)
            self.last_all_arena_xy_cm = arena_xy_cm
            self.prev_centroid_px = centroid_px
            self.missing_frames = 0
            if self.first_detected_time_abs is None:
                self.first_detected_time_abs = float(current_time)
        elif self.last_all_arena_xy_cm is not None and self.missing_frames < self.max_gap_frames:
            arena_xy_cm = self.last_all_arena_xy_cm
            centroid_px = self.prev_centroid_px
            self.missing_frames += 1
        else:
            self.missing_frames += 1

        instant_distance_cm = 0.0
        speed_cm_s = 0.0
        zone_flags = {"in_center": False, "in_outer": False}
        motion_vector_cm: Optional[Tuple[float, float]] = None

        if arena_xy_cm is not None:
            if self.prev_all_arena_xy_cm is not None:
                dx = arena_xy_cm[0] - self.prev_all_arena_xy_cm[0]
                dy = arena_xy_cm[1] - self.prev_all_arena_xy_cm[1]
                instant_distance_cm = float(math.hypot(dx, dy))
                speed_cm_s = instant_distance_cm * self.fps
                motion_vector_cm = (float(dx), float(dy))
            self.prev_all_arena_xy_cm = arena_xy_cm
            zone_flags = self._zone_flags(arena_xy_cm)
        else:
            self.prev_all_arena_xy_cm = None

        self.last_speed_cm_s = speed_cm_s
        in_stats_window = self._is_in_stats_window(float(current_time))
        stats_step_distance_cm = 0.0
        stats_cumulative_distance_cm = self.total_distance_cm

        if in_stats_window:
            self.frames_in_window += 1
            if detected:
                self.detected_frames_in_window += 1

            if arena_xy_cm is not None:
                if self.prev_stats_arena_xy_cm is not None:
                    dx = arena_xy_cm[0] - self.prev_stats_arena_xy_cm[0]
                    dy = arena_xy_cm[1] - self.prev_stats_arena_xy_cm[1]
                    stats_step_distance_cm = float(math.hypot(dx, dy))
                    self.total_distance_cm += stats_step_distance_cm
                self.prev_stats_arena_xy_cm = arena_xy_cm
                stats_cumulative_distance_cm = self.total_distance_cm
                self.max_speed_cm_s = max(self.max_speed_cm_s, speed_cm_s)

                self.center_frames += int(zone_flags["in_center"])
                if speed_cm_s >= self.movement_threshold_cm_s:
                    self.moving_frames += 1
                else:
                    self.immobile_frames += 1

                if zone_flags["in_center"] and not self.prev_in_center_stats:
                    self.center_entries += 1
                    if self.center_latency_s is None:
                        self.center_latency_s = float(current_time) - self._stats_window_origin_abs()
                self.prev_in_center_stats = zone_flags["in_center"]

                if centroid_px is not None:
                    self.trajectory_px_live.append(centroid_px)
                self.trajectory_cm_stats.append(arena_xy_cm)
            else:
                self.prev_stats_arena_xy_cm = None
                self.prev_in_center_stats = False
                self.immobile_frames += 1
                if centroid_px is not None:
                    self.trajectory_px_live.append(centroid_px)
        else:
            self.prev_stats_arena_xy_cm = None
            self.prev_in_center_stats = False
            if centroid_px is not None:
                self.trajectory_px_live.append(centroid_px)

        head_state = self._estimate_head_direction(contour, centroid_px, arena_xy_cm, motion_vector_cm, gray=gray)
        head_confidence = float(head_state.get("confidence") or 0.0)
        head_confident = bool(
            self.use_head_direction
            and head_state.get("estimated")
            and head_confidence >= self.head_confidence_threshold
        )
        if head_state.get("estimated"):
            self.last_head_point_px = tuple(float(v) for v in head_state["point_px"])
            self.last_head_point_cm = tuple(float(v) for v in head_state["point_cm"])
            if head_confident and head_state.get("vector_cm") is not None:
                self.prev_head_vector_cm = tuple(float(v) for v in head_state["vector_cm"])
        else:
            self.last_head_point_px = None
            self.last_head_point_cm = None

        if self.use_head_direction and in_stats_window and head_state.get("estimated"):
            self.head_estimated_frames += 1
            if head_confident:
                self.head_confident_frames += 1

        distances_cm = self._object_distances_cm(contour, arena_xy_cm)
        nose_distances_cm = head_state.get("nose_distances_cm") or [None for _ in self.objects]
        object_angles_deg = head_state.get("object_angles_deg") or [None for _ in self.objects]
        near_flags = [
            bool(detected) and dist is not None and dist <= self.interaction_radius_cm
            for dist in distances_cm
        ]
        head_aligned_flags = [
            bool(detected)
            and nose_distances_cm[idx] is not None
            and float(nose_distances_cm[idx]) <= self.interaction_radius_cm
            and object_angles_deg[idx] is not None
            and float(object_angles_deg[idx]) <= self.head_angle_threshold_deg
            for idx in range(len(self.objects))
        ]

        exploring_idx = None
        exploration_mode = None
        if detected and in_stats_window and speed_cm_s <= self.exploration_max_speed_cm_s:
            candidates = [idx for idx, is_near in enumerate(near_flags) if is_near]
            if candidates:
                if self.use_head_direction:
                    if head_confident:
                        oriented = [idx for idx in candidates if head_aligned_flags[idx]]
                        if oriented:
                            exploring_idx = min(
                                oriented,
                                key=lambda idx: (
                                    nose_distances_cm[idx] if nose_distances_cm[idx] is not None else 1e9,
                                    distances_cm[idx] if distances_cm[idx] is not None else 1e9,
                                ),
                            )
                            exploration_mode = "head"
                    else:
                        exploring_idx = min(
                            candidates,
                            key=lambda idx: distances_cm[idx] if distances_cm[idx] is not None else 1e9,
                        )
                        exploration_mode = "fallback"
                else:
                    exploring_idx = min(
                        candidates,
                        key=lambda idx: distances_cm[idx] if distances_cm[idx] is not None else 1e9,
                    )
                    exploration_mode = "proximity"

        if in_stats_window:
            for idx, is_near in enumerate(near_flags):
                if is_near:
                    self.object_zone_frames[idx] += 1
                    if not self.prev_zone_flags[idx]:
                        self.object_zone_entries[idx] += 1

            if exploring_idx is not None:
                self.object_exploration_frames[exploring_idx] += 1
                if exploration_mode == "head":
                    self.head_supported_exploration_frames += 1
                elif exploration_mode == "fallback":
                    self.fallback_exploration_frames += 1
                if self.object_latency_s[exploring_idx] is None:
                    self.object_latency_s[exploring_idx] = max(0.0, float(current_time) - self._stats_window_origin_abs())
                if self.first_explored_object_idx is None:
                    self.first_explored_object_idx = exploring_idx

            current_explore_flags = [exploring_idx == idx for idx in range(len(self.objects))]
            for idx, flag in enumerate(current_explore_flags):
                if flag and not self.prev_explore_flags[idx]:
                    self.object_bouts[idx] += 1
            self.prev_zone_flags = near_flags
            self.prev_explore_flags = current_explore_flags
        else:
            self.prev_zone_flags = [False, False]
            self.prev_explore_flags = [False, False]

        self.current_exploring_idx = exploring_idx
        proximity_label = self._proximity_label(exploring_idx, near_flags, self.objects)

        record = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "frame": self.total_frames - 1,
            "time_s": round(float(current_time), 3),
            "window_time_s": round(float(current_time - self._stats_window_origin_abs()), 3) if in_stats_window else None,
            "detected": bool(detected),
            "in_stats_window": bool(in_stats_window),
            "x_cm": round(float(arena_xy_cm[0]), 3) if arena_xy_cm is not None else None,
            "y_cm": round(float(arena_xy_cm[1]), 3) if arena_xy_cm is not None else None,
            "speed_cm_s": round(float(speed_cm_s), 3),
            "step_distance_cm": round(float(stats_step_distance_cm), 3),
            "cumulative_distance_cm": round(float(stats_cumulative_distance_cm), 3),
            "in_center": bool(zone_flags["in_center"]) if in_stats_window else False,
            "zone_label": proximity_label if arena_xy_cm is not None else "Unknown",
            "current_object": self.objects[exploring_idx]["name"] if exploring_idx is not None else None,
            "nose_x_cm": round(float(head_state["point_cm"][0]), 3) if head_state.get("point_cm") is not None else None,
            "nose_y_cm": round(float(head_state["point_cm"][1]), 3) if head_state.get("point_cm") is not None else None,
            "body_x_cm": round(float(head_state["body_centroid_cm"][0]), 3) if head_state.get("body_centroid_cm") is not None else None,
            "body_y_cm": round(float(head_state["body_centroid_cm"][1]), 3) if head_state.get("body_centroid_cm") is not None else None,
            "head_direction_confidence": round(head_confidence, 3) if head_state.get("estimated") else None,
            "head_direction_confident": bool(head_confident) if in_stats_window else False,
            "head_direction_method": head_state.get("method") if head_state.get("estimated") else None,
            "body_threshold_polarity": head_state.get("body_threshold_polarity"),
            "tail_confidence": round(float(head_state["tail_confidence"]), 3) if head_state.get("tail_confidence") is not None else None,
            "exploration_mode": exploration_mode,
        }

        for idx, obj in enumerate(self.objects):
            prefix = f"object_{idx + 1}"
            record[f"{prefix}_name"] = obj["name"]
            record[f"{prefix}_role"] = obj["role"]
            record[f"{prefix}_distance_cm"] = round(float(distances_cm[idx]), 3) if distances_cm[idx] is not None else None
            record[f"{prefix}_nose_distance_cm"] = round(float(nose_distances_cm[idx]), 3) if nose_distances_cm[idx] is not None else None
            record[f"{prefix}_head_angle_deg"] = round(float(object_angles_deg[idx]), 3) if object_angles_deg[idx] is not None else None
            record[f"{prefix}_near"] = bool(near_flags[idx]) if in_stats_window else False
            record[f"{prefix}_head_aligned"] = bool(head_aligned_flags[idx]) if in_stats_window else False
            record[f"{prefix}_exploring"] = bool(exploring_idx == idx) if in_stats_window else False

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
            "zone_label": proximity_label,
            "trajectory_px": list(self.trajectory_px_live) if include_trajectory else [],
            "in_stats_window": bool(in_stats_window),
            "object_distances_cm": distances_cm,
            "object_nose_distances_cm": nose_distances_cm,
            "object_head_angles_deg": object_angles_deg,
            "near_flags": near_flags,
            "head_aligned_flags": head_aligned_flags,
            "current_object_idx": exploring_idx,
            "current_object_name": self.objects[exploring_idx]["name"] if exploring_idx is not None else None,
            "current_object_role": self.objects[exploring_idx]["role"] if exploring_idx is not None else None,
            "head_point": head_state.get("point_px"),
            "head_point_cm": head_state.get("point_cm"),
            "body_contour": head_state.get("body_contour"),
            "body_centroid": head_state.get("body_centroid"),
            "tail_contour": head_state.get("tail_contour"),
            "tail_point": head_state.get("tail_point_px"),
            "tail_root_point": head_state.get("tail_root_px"),
            "head_direction_confidence": head_confidence if head_state.get("estimated") else None,
            "head_direction_confident": bool(head_confident),
            "head_direction_method": head_state.get("method"),
            "exploration_mode": exploration_mode,
            "record": record,
        }

    def build_summary(self) -> Dict[str, Any]:
        total_time_s = self.frames_in_window / self.fps if self.fps > 0 else 0.0
        center_time_s = self.center_frames / self.fps if self.fps > 0 else 0.0
        moving_time_s = self.moving_frames / self.fps if self.fps > 0 else 0.0
        immobile_time_s = self.immobile_frames / self.fps if self.fps > 0 else 0.0
        mean_speed_cm_s = self.total_distance_cm / max(total_time_s, 1e-6)
        detection_rate = self.detected_frames_in_window / max(self.frames_in_window, 1) * 100.0
        head_estimated_rate = self.head_estimated_frames / max(self.frames_in_window, 1) * 100.0
        head_confident_rate = self.head_confident_frames / max(self.frames_in_window, 1) * 100.0

        summary: Dict[str, Any] = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "phase": self.phase,
            "time_mode": self.time_config.get("mode", "global"),
            "time_start_s": self.time_config.get("start"),
            "time_end_s": self.time_config.get("end"),
            "analysis_window_start_s": round(self._stats_window_origin_abs(), 3),
            "total_time_s": round(total_time_s, 3),
            "detected_frames": int(self.detected_frames_in_window),
            "detected_rate_percent": round(detection_rate, 2),
            "total_distance_cm": round(self.total_distance_cm, 3),
            "mean_speed_cm_s": round(mean_speed_cm_s, 3),
            "max_speed_cm_s": round(self.max_speed_cm_s, 3),
            "center_time_s": round(center_time_s, 3),
            "center_percent": round(center_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            "moving_time_s": round(moving_time_s, 3),
            "immobility_time_s": round(immobile_time_s, 3),
            "center_entries": int(self.center_entries),
            "center_latency_s": round(self.center_latency_s, 3) if self.center_latency_s is not None else None,
            "interaction_radius_cm": self.interaction_radius_cm,
            "exploration_max_speed_cm_s": self.exploration_max_speed_cm_s,
            "use_head_direction": self.use_head_direction,
            "head_angle_threshold_deg": self.head_angle_threshold_deg,
            "head_confidence_threshold": self.head_confidence_threshold,
            "body_gray_threshold": self.body_gray_threshold,
            "head_estimated_rate_percent": round(head_estimated_rate, 2),
            "head_confident_rate_percent": round(head_confident_rate, 2),
            "movement_threshold_cm_s": self.movement_threshold_cm_s,
            "min_valid_exploration_s": self.min_valid_exploration_s,
            "arena_width_cm": self.arena_width_cm,
            "arena_height_cm": self.arena_height_cm,
        }

        total_exploration = 0.0
        role_times: Dict[str, float] = {}
        for idx, obj in enumerate(self.objects):
            exploration_s = self.object_exploration_frames[idx] / self.fps
            zone_s = self.object_zone_frames[idx] / self.fps
            total_exploration += exploration_s
            role_times[obj["role"]] = role_times.get(obj["role"], 0.0) + exploration_s

            prefix = f"object_{idx + 1}"
            summary[f"{prefix}_name"] = obj["name"]
            summary[f"{prefix}_role"] = obj["role"]
            summary[f"{prefix}_exploration_s"] = round(exploration_s, 3)
            summary[f"{prefix}_zone_time_s"] = round(zone_s, 3)
            summary[f"{prefix}_zone_entries"] = int(self.object_zone_entries[idx])
            summary[f"{prefix}_bouts"] = int(self.object_bouts[idx])
            summary[f"{prefix}_latency_s"] = round(self.object_latency_s[idx], 3) if self.object_latency_s[idx] is not None else None

        summary["total_object_exploration_s"] = round(total_exploration, 3)
        summary["head_supported_exploration_s"] = round(self.head_supported_exploration_frames / self.fps, 3)
        summary["fallback_exploration_s"] = round(self.fallback_exploration_frames / self.fps, 3)
        summary["meets_min_exploration"] = bool(total_exploration >= self.min_valid_exploration_s)
        summary["first_explored_object"] = self.objects[self.first_explored_object_idx]["name"] if self.first_explored_object_idx is not None else None
        summary["first_explored_role"] = self.objects[self.first_explored_object_idx]["role"] if self.first_explored_object_idx is not None else None

        if self.phase == "recognition":
            familiar_time = float(role_times.get("familiar", 0.0))
            novel_time = float(role_times.get("novel", 0.0))
            total_role_exploration = familiar_time + novel_time
            discrimination_index = None
            preference_index_percent = None
            if total_role_exploration > 0:
                discrimination_index = (novel_time - familiar_time) / total_role_exploration
                preference_index_percent = novel_time / total_role_exploration * 100.0

            summary["familiar_exploration_s"] = round(familiar_time, 3)
            summary["novel_exploration_s"] = round(novel_time, 3)
            summary["discrimination_index"] = round(discrimination_index, 4) if discrimination_index is not None else None
            summary["preference_index_percent"] = round(preference_index_percent, 2) if preference_index_percent is not None else None
            summary["difference_seconds"] = round(novel_time - familiar_time, 3)
            summary["novel_preference"] = bool(novel_time > familiar_time)
        else:
            obj1 = self.object_exploration_frames[0] / self.fps
            obj2 = self.object_exploration_frames[1] / self.fps
            balance_index = None
            if (obj1 + obj2) > 0:
                balance_index = (obj1 - obj2) / (obj1 + obj2)
            summary["object_balance_index"] = round(balance_index, 4) if balance_index is not None else None

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
                    "center_frames": 0,
                    "moving_frames": 0,
                    "object_exploration_frames": [0, 0],
                },
            )
            bucket["frames"] += 1
            bucket["distance_cm"] += float(row.get("step_distance_cm") or 0.0)
            bucket["center_frames"] += int(bool(row.get("in_center")))
            bucket["moving_frames"] += int(float(row.get("speed_cm_s") or 0.0) >= self.movement_threshold_cm_s)
            for obj_idx in range(2):
                bucket["object_exploration_frames"][obj_idx] += int(bool(row.get(f"object_{obj_idx + 1}_exploring")))

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
                "duration_s": round(duration_s, 3),
                "distance_cm": round(bucket["distance_cm"], 3),
                "mean_speed_cm_s": round(bucket["distance_cm"] / max(duration_s, 1e-6), 3),
                "center_time_s": round(bucket["center_frames"] / self.fps, 3),
                "moving_time_s": round(bucket["moving_frames"] / self.fps, 3),
            }

            total_exp = 0.0
            role_times: Dict[str, float] = {}
            for obj_idx, obj in enumerate(self.objects):
                exp_s = bucket["object_exploration_frames"][obj_idx] / self.fps
                row[f"object_{obj_idx + 1}_exploration_s"] = round(exp_s, 3)
                total_exp += exp_s
                role_times[obj["role"]] = role_times.get(obj["role"], 0.0) + exp_s

            row["total_object_exploration_s"] = round(total_exp, 3)
            if self.phase == "recognition":
                fam = role_times.get("familiar", 0.0)
                nov = role_times.get("novel", 0.0)
                total_role = fam + nov
                row["familiar_exploration_s"] = round(fam, 3)
                row["novel_exploration_s"] = round(nov, 3)
                row["discrimination_index"] = round((nov - fam) / total_role, 4) if total_role > 0 else None
                row["preference_index_percent"] = round(nov / total_role * 100.0, 2) if total_role > 0 else None
            out.append(row)
        return out


def _object_color(role: str) -> Tuple[int, int, int]:
    value = str(role or "").strip().lower()
    if value == "novel":
        return (59, 130, 246)
    if value == "familiar":
        return (249, 115, 22)
    return (148, 163, 184)


def _paper_role_color(role: str) -> Tuple[int, int, int]:
    value = str(role or "").strip().lower()
    if value == "novel":
        return (43, 142, 242)
    if value == "familiar":
        return (167, 121, 78)
    return (148, 163, 184)


def _paper_blend_heatmap(norm: np.ndarray) -> np.ndarray:
    return classic_heatmap(norm)


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


def draw_novel_object_results(
    frame: np.ndarray,
    tracker_results: Sequence[Tuple[NovelObjectTracker, Dict[str, Any]]],
    inplace: bool = False,
) -> np.ndarray:
    output = frame if inplace else frame.copy()

    for tracker, result in tracker_results:
        arena_poly = tracker.arena_points.astype(np.int32)
        detection_poly = tracker.detection_points.astype(np.int32)
        center_poly = tracker.get_center_polygon_px()
        current_idx = result.get("current_object_idx")

        cv2.polylines(output, [arena_poly], True, (96, 165, 250), 2)
        if tracker.has_custom_detection_mask:
            cv2.polylines(output, [detection_poly], True, (255, 214, 10), 1, cv2.LINE_AA)
        cv2.polylines(output, [center_poly], True, (212, 212, 212), 1)

        for idx, obj in enumerate(tracker.objects):
            color = _object_color(obj["role"])
            poly = tracker.object_circle_to_image_polygon(obj["point_cm"], tracker.interaction_radius_cm)
            thickness = 3 if idx == current_idx else 2
            cv2.polylines(output, [poly], True, color, thickness, cv2.LINE_AA)
            px = int(obj["point_px"][0])
            py = int(obj["point_px"][1])
            cv2.circle(output, (px, py), 4, color, -1)

        contour = result.get("contour")
        centroid = result.get("centroid")
        if contour is not None:
            cv2.drawContours(output, [contour], -1, (34, 197, 94), 2)
        tail_contour = result.get("tail_contour")
        if tail_contour is not None:
            cv2.drawContours(output, [tail_contour], -1, (249, 121, 232), 2, cv2.LINE_AA)
        body_contour = result.get("body_contour")
        if body_contour is not None:
            cv2.drawContours(output, [body_contour], -1, (241, 185, 16), 2, cv2.LINE_AA)
        if centroid is not None:
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(output, (cx, cy), 4, (34, 197, 94), -1)
        body_centroid = result.get("body_centroid")
        if body_centroid is not None:
            cv2.circle(output, (int(body_centroid[0]), int(body_centroid[1])), 4, (241, 185, 16), -1)
        tail_point = result.get("tail_point")
        if tail_point is not None:
            cv2.circle(output, (int(tail_point[0]), int(tail_point[1])), 4, (36, 36, 239), -1, cv2.LINE_AA)
        tail_root = result.get("tail_root_point")
        if tail_root is not None:
            cv2.circle(output, (int(tail_root[0]), int(tail_root[1])), 4, (249, 121, 232), -1, cv2.LINE_AA)

        head_point = result.get("head_point")
        head_confidence = float(result.get("head_direction_confidence") or 0.0)
        if tracker.use_head_direction and centroid is not None and head_point is not None:
            head_pt = (int(head_point[0]), int(head_point[1]))
            line_origin = body_centroid if body_centroid is not None else centroid
            center_pt = (int(line_origin[0]), int(line_origin[1]))
            if head_confidence >= tracker.head_confidence_threshold:
                line_color = (21, 204, 250)
                thickness = 2
            else:
                line_color = (148, 163, 184)
                thickness = 1
            cv2.line(output, center_pt, head_pt, line_color, thickness, cv2.LINE_AA)
            cv2.circle(output, head_pt, 5, (21, 204, 250), -1, cv2.LINE_AA)

        traj = result.get("trajectory_px") or []
        if len(traj) > 360:
            stride = max(1, int(math.ceil(len(traj) / 360)))
            sampled_traj = list(traj[::stride])
            if sampled_traj and sampled_traj[-1] != traj[-1]:
                sampled_traj.append(traj[-1])
            traj = sampled_traj
        for idx in range(1, len(traj)):
            pt1 = (int(traj[idx - 1][0]), int(traj[idx - 1][1]))
            pt2 = (int(traj[idx][0]), int(traj[idx][1]))
            cv2.line(output, pt1, pt2, (34, 197, 94), 1, cv2.LINE_AA)

    return output


def render_novel_object_heatmap(
    tracker: NovelObjectTracker,
    output_path: str,
    pixels_per_cm: int = 12,
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
    color = _paper_blend_heatmap(norm)
    border_thickness = _paper_border_thickness(pixels_per_cm)
    guide_thickness = _paper_guide_thickness(pixels_per_cm)
    cv2.rectangle(color, (0, 0), (width - 1, height - 1), CLASSIC_BORDER, border_thickness, cv2.LINE_AA)

    cx0, cy0, cx1, cy1 = tracker.get_center_rect_cm()
    cv2.rectangle(
        color,
        (int(cx0 * pixels_per_cm), int(cy0 * pixels_per_cm)),
        (int(cx1 * pixels_per_cm), int(cy1 * pixels_per_cm)),
        CLASSIC_HEAT_OUTLINE,
        guide_thickness,
        cv2.LINE_AA,
    )

    for obj in tracker.objects:
        ox = int(np.clip(round(obj["point_cm"][0] * pixels_per_cm), 0, width - 1))
        oy = int(np.clip(round(obj["point_cm"][1] * pixels_per_cm), 0, height - 1))
        radius = int(round(tracker.interaction_radius_cm * pixels_per_cm))
        color_bgr = _paper_role_color(obj["role"])
        cv2.circle(color, (ox, oy), radius, color_bgr, border_thickness, cv2.LINE_AA)
        cv2.circle(color, (ox, oy), max(2, border_thickness), color_bgr, -1, cv2.LINE_AA)

    if write_image_unicode(output_path, color):
        return output_path
    return None


def render_novel_object_trackmap(
    tracker: NovelObjectTracker,
    output_path: str,
    pixels_per_cm: int = 12,
) -> Optional[str]:
    points_cm = tracker.trajectory_cm_stats
    if not points_cm:
        return None

    width = max(160, int(round(tracker.arena_width_cm * pixels_per_cm)))
    height = max(160, int(round(tracker.arena_height_cm * pixels_per_cm)))
    canvas = np.full((height, width, 3), CLASSIC_BG, dtype=np.uint8)
    border_thickness = _paper_border_thickness(pixels_per_cm)
    guide_thickness = _paper_guide_thickness(pixels_per_cm)
    path_thickness = _paper_path_thickness(pixels_per_cm)
    marker_radius = classic_marker_radius(pixels_per_cm)

    cx0, cy0, cx1, cy1 = tracker.get_center_rect_cm()
    center_top_left = (int(cx0 * pixels_per_cm), int(cy0 * pixels_per_cm))
    center_bottom_right = (int(cx1 * pixels_per_cm), int(cy1 * pixels_per_cm))
    fill_rect_alpha(canvas, center_top_left, center_bottom_right, CLASSIC_GUIDE_FILL, alpha=0.92)
    cv2.rectangle(canvas, center_top_left, center_bottom_right, CLASSIC_GUIDE_OUTLINE, guide_thickness, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), CLASSIC_BORDER, border_thickness, cv2.LINE_AA)

    points = []
    for x_cm, y_cm in points_cm:
        px = int(np.clip(round(x_cm * pixels_per_cm), 0, width - 1))
        py = int(np.clip(round(y_cm * pixels_per_cm), 0, height - 1))
        points.append((px, py))

    for idx in range(1, len(points)):
        cv2.line(canvas, points[idx - 1], points[idx], CLASSIC_TRACK, path_thickness, cv2.LINE_AA)
    draw_start_end_markers(canvas, points, marker_radius)

    for obj in tracker.objects:
        ox = int(np.clip(round(obj["point_cm"][0] * pixels_per_cm), 0, width - 1))
        oy = int(np.clip(round(obj["point_cm"][1] * pixels_per_cm), 0, height - 1))
        radius = int(round(tracker.interaction_radius_cm * pixels_per_cm))
        color = _paper_role_color(obj["role"])
        cv2.circle(canvas, (ox, oy), radius, color, border_thickness, cv2.LINE_AA)
        cv2.circle(canvas, (ox, oy), max(2, border_thickness), color, -1, cv2.LINE_AA)

    if write_image_unicode(output_path, canvas):
        return output_path
    return None
