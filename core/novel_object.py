#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Novel Object Recognition (NOR) helpers.

This module provides per-ROI tracking, proximity-based object exploration
metrics, perspective normalization, and visualization helpers for a multi-ROI
NOR workflow.
"""

from __future__ import annotations

import math
import os
import re
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

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


def sanitize_name(text: str, fallback: str = "nor") -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(text or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:80] or fallback


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

        if arena_xy_cm is not None:
            if self.prev_all_arena_xy_cm is not None:
                dx = arena_xy_cm[0] - self.prev_all_arena_xy_cm[0]
                dy = arena_xy_cm[1] - self.prev_all_arena_xy_cm[1]
                instant_distance_cm = float(math.hypot(dx, dy))
                speed_cm_s = instant_distance_cm * self.fps
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

        distances_cm = self._object_distances_cm(contour, arena_xy_cm)
        near_flags = [
            bool(detected) and dist is not None and dist <= self.interaction_radius_cm
            for dist in distances_cm
        ]

        exploring_idx = None
        if detected and in_stats_window and speed_cm_s <= self.exploration_max_speed_cm_s:
            candidates = [idx for idx, is_near in enumerate(near_flags) if is_near]
            if candidates:
                exploring_idx = min(
                    candidates,
                    key=lambda idx: distances_cm[idx] if distances_cm[idx] is not None else 1e9,
                )

        if in_stats_window:
            for idx, is_near in enumerate(near_flags):
                if is_near:
                    self.object_zone_frames[idx] += 1
                    if not self.prev_zone_flags[idx]:
                        self.object_zone_entries[idx] += 1

            if exploring_idx is not None:
                self.object_exploration_frames[exploring_idx] += 1
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
        }

        for idx, obj in enumerate(self.objects):
            prefix = f"object_{idx + 1}"
            record[f"{prefix}_name"] = obj["name"]
            record[f"{prefix}_role"] = obj["role"]
            record[f"{prefix}_distance_cm"] = round(float(distances_cm[idx]), 3) if distances_cm[idx] is not None else None
            record[f"{prefix}_near"] = bool(near_flags[idx]) if in_stats_window else False
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
            "near_flags": near_flags,
            "current_object_idx": exploring_idx,
            "current_object_name": self.objects[exploring_idx]["name"] if exploring_idx is not None else None,
            "current_object_role": self.objects[exploring_idx]["role"] if exploring_idx is not None else None,
            "record": record,
        }

    def build_summary(self) -> Dict[str, Any]:
        total_time_s = self.frames_in_window / self.fps if self.fps > 0 else 0.0
        center_time_s = self.center_frames / self.fps if self.fps > 0 else 0.0
        moving_time_s = self.moving_frames / self.fps if self.fps > 0 else 0.0
        immobile_time_s = self.immobile_frames / self.fps if self.fps > 0 else 0.0
        mean_speed_cm_s = self.total_distance_cm / max(total_time_s, 1e-6)
        detection_rate = self.detected_frames_in_window / max(self.frames_in_window, 1) * 100.0

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
    img = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.rectangle(color, (0, 0), (width - 1, height - 1), (255, 255, 255), 2)

    cx0, cy0, cx1, cy1 = tracker.get_center_rect_cm()
    cv2.rectangle(
        color,
        (int(cx0 * pixels_per_cm), int(cy0 * pixels_per_cm)),
        (int(cx1 * pixels_per_cm), int(cy1 * pixels_per_cm)),
        (255, 255, 255),
        1,
    )

    for obj in tracker.objects:
        ox = int(np.clip(round(obj["point_cm"][0] * pixels_per_cm), 0, width - 1))
        oy = int(np.clip(round(obj["point_cm"][1] * pixels_per_cm), 0, height - 1))
        radius = int(round(tracker.interaction_radius_cm * pixels_per_cm))
        color_bgr = _object_color(obj["role"])
        cv2.circle(color, (ox, oy), radius, color_bgr, 2)

    cv2.imwrite(output_path, color)
    return output_path


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
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)

    cx0, cy0, cx1, cy1 = tracker.get_center_rect_cm()
    cv2.rectangle(
        canvas,
        (int(cx0 * pixels_per_cm), int(cy0 * pixels_per_cm)),
        (int(cx1 * pixels_per_cm), int(cy1 * pixels_per_cm)),
        (218, 234, 255),
        thickness=-1,
    )
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (70, 70, 70), 2)
    cv2.rectangle(
        canvas,
        (int(cx0 * pixels_per_cm), int(cy0 * pixels_per_cm)),
        (int(cx1 * pixels_per_cm), int(cy1 * pixels_per_cm)),
        (66, 133, 244),
        2,
    )

    points = []
    for x_cm, y_cm in points_cm:
        px = int(np.clip(round(x_cm * pixels_per_cm), 0, width - 1))
        py = int(np.clip(round(y_cm * pixels_per_cm), 0, height - 1))
        points.append((px, py))

    for idx in range(1, len(points)):
        cv2.line(canvas, points[idx - 1], points[idx], (34, 91, 179), 1, cv2.LINE_AA)

    if points:
        cv2.circle(canvas, points[0], 5, (16, 185, 129), -1)
        cv2.circle(canvas, points[-1], 5, (229, 57, 53), -1)

    for obj in tracker.objects:
        ox = int(np.clip(round(obj["point_cm"][0] * pixels_per_cm), 0, width - 1))
        oy = int(np.clip(round(obj["point_cm"][1] * pixels_per_cm), 0, height - 1))
        radius = int(round(tracker.interaction_radius_cm * pixels_per_cm))
        color = _object_color(obj["role"])
        cv2.circle(canvas, (ox, oy), radius, color, 2)

    cv2.imwrite(output_path, canvas)
    return output_path
