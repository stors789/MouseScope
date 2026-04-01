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


def _normalize_polygon(points: Optional[Sequence[Sequence[float]]], min_points: int = 4) -> Optional[np.ndarray]:
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

        self.trajectory_px_live: deque[Tuple[float, float]] = deque(maxlen=1200)
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
        center_w = self.arena_width_cm * self.center_ratio_x
        center_h = self.arena_height_cm * self.center_ratio_y
        x0 = (self.arena_width_cm - center_w) / 2.0
        y0 = (self.arena_height_cm - center_h) / 2.0
        return (x0, y0, x0 + center_w, y0 + center_h)

    def cm_rect_to_image_polygon(self, rect_cm: Tuple[float, float, float, float]) -> np.ndarray:
        x0, y0, x1, y1 = rect_cm
        pts = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32)
        return cv2.perspectiveTransform(pts, self.h_cm_to_img)[0].astype(np.int32)

    def object_circle_to_image_polygon(self, center_cm: Tuple[float, float], radius_cm: float, samples: int = 40) -> np.ndarray:
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
        return poly.astype(np.int32)

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

    def _estimate_head_direction(
        self,
        contour: Optional[np.ndarray],
        centroid_px: Optional[Tuple[float, float]],
        arena_xy_cm: Optional[Tuple[float, float]],
        motion_vector_cm: Optional[Tuple[float, float]],
    ) -> Dict[str, Any]:
        empty = {
            "point_px": None,
            "point_cm": None,
            "vector_cm": None,
            "confidence": 0.0,
            "estimated": False,
            "axis_ratio": None,
            "nose_distances_cm": [None for _ in self.objects],
            "object_angles_deg": [None for _ in self.objects],
        }

        if contour is None or centroid_px is None or arena_xy_cm is None:
            return empty

        points_px = contour.reshape(-1, 2).astype(np.float32)
        if points_px.shape[0] < 10:
            return empty

        centroid_arr = np.asarray(centroid_px, dtype=np.float32)
        centered = points_px - centroid_arr
        if float(np.max(np.linalg.norm(centered, axis=1))) < 1e-3:
            return empty

        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        major_axis = eigvecs[:, order[0]]
        major_axis = _unit_vector(major_axis)
        if major_axis is None:
            return empty

        axis_ratio = float((eigvals[0] + 1e-6) / (eigvals[1] + 1e-6)) if eigvals.size > 1 else 999.0
        projections = centered @ major_axis
        if float(np.max(projections) - np.min(projections)) < 1e-3:
            return empty

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
                float(obj["point_cm"][0] - arena_xy_cm[0]),
                float(obj["point_cm"][1] - arena_xy_cm[1]),
            )
            body_to_object_vectors.append(vec)
            body_to_object_distances.append(float(math.hypot(vec[0], vec[1])))

        def _support_score(unit_head: np.ndarray, obj_vec: Optional[Tuple[float, float]] = None, obj_distance: Optional[float] = None) -> Tuple[float, float, float]:
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
            best: Optional[Dict[str, Any]] = None
            for raw_idx in indices.tolist():
                idx = int(raw_idx)
                point_px = (float(points_px[idx][0]), float(points_px[idx][1]))
                point_cm = self.project_point_to_cm(point_px)
                vector_cm = (
                    float(point_cm[0] - arena_xy_cm[0]),
                    float(point_cm[1] - arena_xy_cm[1]),
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
                    "extremeness": extremeness,
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
            return empty
        if backward is None:
            winner = forward
            other = None
        elif forward is None:
            winner = backward
            other = None
        else:
            winner, other = (forward, backward) if forward["score"] >= backward["score"] else (backward, forward)

        assert winner is not None
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
                float(obj["point_cm"][0] - arena_xy_cm[0]),
                float(obj["point_cm"][1] - arena_xy_cm[1]),
            )
            object_angles_deg.append(_angle_deg(winner["vector_cm"], body_vec))

        return {
            "point_px": winner["point_px"],
            "point_cm": winner["point_cm"],
            "vector_cm": winner["vector_cm"],
            "confidence": confidence,
            "estimated": True,
            "axis_ratio": axis_ratio,
            "nose_distances_cm": nose_distances_cm,
            "object_angles_deg": object_angles_deg,
        }

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

        head_state = self._estimate_head_direction(contour, centroid_px, arena_xy_cm, motion_vector_cm)
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
            "head_direction_confidence": round(head_confidence, 3) if head_state.get("estimated") else None,
            "head_direction_confident": bool(head_confident) if in_stats_window else False,
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
            "trajectory_px": list(self.trajectory_px_live),
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
            "head_direction_confidence": head_confidence if head_state.get("estimated") else None,
            "head_direction_confident": bool(head_confident),
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


_PAPER_BG = (255, 255, 255)
_PAPER_BORDER = (18, 18, 18)
_PAPER_GUIDE = (110, 110, 110)
_PAPER_TRAJECTORY = (58, 58, 58)
_PAPER_HEAT_STOPS = np.array([0.0, 0.12, 0.35, 0.62, 1.0], dtype=np.float32)
_PAPER_HEAT_COLORS = np.array(
    [
        (255, 255, 255),
        (208, 240, 255),
        (145, 217, 253),
        (84, 160, 253),
        (38, 44, 196),
    ],
    dtype=np.float32,
)


def _paper_role_color(role: str) -> Tuple[int, int, int]:
    value = str(role or "").strip().lower()
    if value == "novel":
        return (43, 142, 242)
    if value == "familiar":
        return (167, 121, 78)
    return (148, 163, 184)


def _paper_blend_heatmap(norm: np.ndarray) -> np.ndarray:
    flat = np.clip(norm, 0.0, 1.0).reshape(-1)
    scaled = np.power(flat, 0.82)
    scaled = np.where(scaled > 0.02, scaled, 0.0)
    channels = [
        np.interp(scaled, _PAPER_HEAT_STOPS, _PAPER_HEAT_COLORS[:, channel])
        for channel in range(3)
    ]
    heat = np.stack(channels, axis=1)
    return heat.reshape(norm.shape[0], norm.shape[1], 3).astype(np.uint8)


def _paper_border_thickness(pixels_per_cm: int) -> int:
    return max(2, int(round(pixels_per_cm / 6.0)))


def _paper_guide_thickness(pixels_per_cm: int) -> int:
    return max(1, _paper_border_thickness(pixels_per_cm) - 1)


def _paper_path_thickness(pixels_per_cm: int) -> int:
    return max(1, int(round(pixels_per_cm / 10.0)))


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
        center_poly = tracker.cm_rect_to_image_polygon(tracker.get_center_rect_cm())
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
        if centroid is not None:
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(output, (cx, cy), 4, (34, 197, 94), -1)

        head_point = result.get("head_point")
        head_confidence = float(result.get("head_direction_confidence") or 0.0)
        if tracker.use_head_direction and centroid is not None and head_point is not None:
            head_pt = (int(head_point[0]), int(head_point[1]))
            center_pt = (int(centroid[0]), int(centroid[1]))
            if head_confidence >= tracker.head_confidence_threshold:
                line_color = (255, 214, 10) if result.get("exploration_mode") == "head" else (16, 185, 241)
                thickness = 2
            else:
                line_color = (148, 163, 184)
                thickness = 1
            cv2.line(output, center_pt, head_pt, line_color, thickness, cv2.LINE_AA)
            cv2.circle(output, head_pt, 3, line_color, -1)

        traj = result.get("trajectory_px") or []
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
    dash_length = _paper_dash_length(pixels_per_cm)
    gap_length = _paper_gap_length(pixels_per_cm)
    cv2.rectangle(color, (0, 0), (width - 1, height - 1), _PAPER_BORDER, border_thickness, cv2.LINE_AA)

    cx0, cy0, cx1, cy1 = tracker.get_center_rect_cm()
    _paper_draw_dashed_rect(
        color,
        (int(cx0 * pixels_per_cm), int(cy0 * pixels_per_cm)),
        (int(cx1 * pixels_per_cm), int(cy1 * pixels_per_cm)),
        _PAPER_GUIDE,
        guide_thickness,
        dash_length,
        gap_length,
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
    canvas = np.full((height, width, 3), _PAPER_BG, dtype=np.uint8)
    border_thickness = _paper_border_thickness(pixels_per_cm)
    guide_thickness = _paper_guide_thickness(pixels_per_cm)
    path_thickness = _paper_path_thickness(pixels_per_cm)
    dash_length = _paper_dash_length(pixels_per_cm)
    gap_length = _paper_gap_length(pixels_per_cm)

    cx0, cy0, cx1, cy1 = tracker.get_center_rect_cm()
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), _PAPER_BORDER, border_thickness, cv2.LINE_AA)
    _paper_draw_dashed_rect(
        canvas,
        (int(cx0 * pixels_per_cm), int(cy0 * pixels_per_cm)),
        (int(cx1 * pixels_per_cm), int(cy1 * pixels_per_cm)),
        _PAPER_GUIDE,
        guide_thickness,
        dash_length,
        gap_length,
    )

    points = []
    for x_cm, y_cm in points_cm:
        px = int(np.clip(round(x_cm * pixels_per_cm), 0, width - 1))
        py = int(np.clip(round(y_cm * pixels_per_cm), 0, height - 1))
        points.append((px, py))

    for idx in range(1, len(points)):
        cv2.line(canvas, points[idx - 1], points[idx], _PAPER_TRAJECTORY, path_thickness, cv2.LINE_AA)

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
