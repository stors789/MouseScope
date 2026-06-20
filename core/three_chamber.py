#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-chamber social test tracking and rendering helpers."""

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
    fill_poly_alpha,
    style_color,
)


CHAMBERS = ("left", "center", "right")
SIDES = ("left", "right")
DOORS = ("left_door", "right_door")
PHASES = ("habituation_center", "habituation_all", "sociability", "social_novelty")
VALID_PHASES = set(PHASES) | {"ignore"}
ROLE_EMPTY = {"", "none", "empty_cup", "empty_object"}
ROLE_SOCIAL = {"social_stranger_1", "stranger_1", "social"}
ROLE_FAMILIAR = {"familiar_stranger_1", "familiar", "stranger_1"}
ROLE_NOVEL = {"novel_stranger_2", "novel", "stranger_2"}
ZONE_LABELS = {
    "left": "Left",
    "center": "Center",
    "right": "Right",
    "left_door": "Left Door",
    "right_door": "Right Door",
    "Unknown": "Unknown",
}


def sanitize_name(text: str, fallback: str = "three_chamber") -> str:
    return ascii_slug(text, fallback=fallback, max_length=80)


def _coerce_point(point: Any) -> Tuple[float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"])
    return float(point[0]), float(point[1])


def _coerce_polygon(points: Sequence[Sequence[float]], min_points: int = 3) -> np.ndarray:
    coerced = [_coerce_point(point) for point in (points or [])]
    pts = np.asarray(coerced, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < min_points:
        raise ValueError(f"Three-chamber zone requires at least {min_points} points.")
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


def _point_in_polygon(poly: Optional[np.ndarray], point: Tuple[float, float]) -> bool:
    return poly is not None and cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


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


def _preference_index(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    denom = float(a) + float(b)
    if denom <= 1e-9:
        return None
    return round((float(a) - float(b)) / denom, 4)


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


def _normalize_role(role: Any) -> str:
    value = str(role or "").strip().lower()
    aliases = {
        "stranger1": "social_stranger_1",
        "stranger_1": "social_stranger_1",
        "object": "empty_object",
        "empty": "empty_object",
        "empty_cup": "empty_object",
        "familiar": "familiar_stranger_1",
        "novel": "novel_stranger_2",
        "stranger2": "novel_stranger_2",
        "stranger_2": "novel_stranger_2",
    }
    return aliases.get(value, value)


def _role_for_category(role: str, category: str) -> bool:
    role = _normalize_role(role)
    if category == "social":
        return role in ROLE_SOCIAL
    if category == "object":
        return role in ROLE_EMPTY
    if category == "familiar":
        return role in ROLE_FAMILIAR
    if category == "novel":
        return role in ROLE_NOVEL
    return False


class ThreeChamberTracker:
    """Track one test mouse in one three-chamber apparatus."""

    def __init__(
        self,
        zone_polygons: Dict[str, Sequence[Sequence[float]]],
        objects: Sequence[Dict[str, Any]],
        detection_polygon: Optional[Sequence[Sequence[float]]] = None,
        stimulus_exclusion_polygons: Optional[Sequence[Sequence[Sequence[float]]]] = None,
        role_schedule: Optional[Dict[str, Dict[str, str]]] = None,
        protocol_segments: Optional[Sequence[Dict[str, Any]]] = None,
        analysis_mode: str = "single_phase",
        selected_phase: str = "sociability",
        arena_width_cm: float = 60.0,
        arena_height_cm: float = 40.0,
        fps: float = 30.0,
        interaction_radius_cm: float = 2.0,
        cup_exclusion_radius_cm: float = 1.5,
        use_head_direction: bool = False,
        head_angle_threshold_deg: float = 60.0,
        head_confidence_threshold: float = 0.35,
        transition_min_dwell_s: float = 0.12,
        center_reset_s: float = 0.2,
        movement_threshold_cm_s: float = 2.0,
        roi_id: int = 1,
        roi_name: Optional[str] = None,
        time_config: Optional[Dict[str, Any]] = None,
        global_start_time_s: float = 0.0,
        subject_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        missing = [name for name in CHAMBERS if name not in (zone_polygons or {})]
        if missing:
            raise ValueError(f"Three-chamber zones are incomplete: missing {', '.join(missing)}")

        self.roi_id = int(roi_id)
        self.roi_name = roi_name or f"3CST {self.roi_id}"
        self.subject_fields = dict(subject_fields or {})
        self.time_config = self._normalize_time_config(time_config)
        self.global_start_time_s = max(0.0, float(global_start_time_s or 0.0))

        self.zone_polygons_img = {name: _coerce_polygon(zone_polygons[name]) for name in CHAMBERS}
        self.door_polygons_img = {
            name: _coerce_polygon(zone_polygons[name])
            for name in DOORS
            if zone_polygons.get(name)
        }
        self.detection_polygon_img = _coerce_polygon(detection_polygon) if detection_polygon is not None else None
        self.stimulus_exclusion_polygons_img = [
            _coerce_polygon(poly)
            for poly in (stimulus_exclusion_polygons or [])
            if poly
        ]

        self.objects = self._normalize_objects(objects)
        if set(self.objects) != {"left", "right"}:
            raise ValueError("Three-chamber analysis requires left and right cup/object points.")

        all_zone_polys = list(self.zone_polygons_img.values())
        self.arena_width_cm = max(1.0, float(arena_width_cm))
        self.arena_height_cm = max(1.0, float(arena_height_cm))
        min_x, min_y, max_x, max_y = _polygons_bbox(all_zone_polys)
        self._origin_px = (min_x, min_y)
        px_w = max(1.0, max_x - min_x)
        px_h = max(1.0, max_y - min_y)
        self.cm_per_px = float((self.arena_width_cm / px_w + self.arena_height_cm / px_h) * 0.5)
        self.px_per_cm = 1.0 / max(self.cm_per_px, 1e-9)

        self.fps = max(1.0, float(fps))
        self.analysis_mode = "full_protocol" if str(analysis_mode or "").strip().lower() == "full_protocol" else "single_phase"
        self.selected_phase = self._normalize_phase(selected_phase, default="sociability")
        self.protocol_segments = self._normalize_protocol_segments(protocol_segments)
        self.role_schedule = self._normalize_role_schedule(role_schedule)

        self.interaction_radius_cm = max(0.1, float(interaction_radius_cm))
        self.cup_exclusion_radius_cm = max(0.0, float(cup_exclusion_radius_cm))
        self.interaction_radius_px = self.interaction_radius_cm * self.px_per_cm
        self.cup_exclusion_radius_px = self.cup_exclusion_radius_cm * self.px_per_cm
        self.use_head_direction = bool(use_head_direction)
        self.head_angle_threshold_deg = max(0.0, float(head_angle_threshold_deg))
        self.head_confidence_threshold = max(0.0, min(1.0, float(head_confidence_threshold)))
        self.transition_min_dwell_s = max(0.0, float(transition_min_dwell_s))
        self.center_reset_s = max(0.0, float(center_reset_s))
        self.movement_threshold_cm_s = max(0.0, float(movement_threshold_cm_s))

        self.detection_mask: Optional[np.ndarray] = None
        self.exclusion_mask: Optional[np.ndarray] = None
        self.detection_bbox: Optional[Tuple[int, int, int, int]] = None
        self._cropped_detection_mask: Optional[np.ndarray] = None
        self._cropped_exclusion_mask: Optional[np.ndarray] = None

        self.total_frames = 0
        self.detected_frames = 0
        self.first_detected_time_abs: Optional[float] = None
        self.prev_all_centroid_px: Optional[Tuple[float, float]] = None
        self.last_all_centroid_px: Optional[Tuple[float, float]] = None
        self.prev_centroid_px: Optional[Tuple[float, float]] = None
        self.missing_frames = 0
        self.missing_time_s = 0.0
        self.max_gap_frames = 5
        self.max_gap_s = self.max_gap_frames / max(self.fps, 1e-6)
        self.last_speed_cm_s = 0.0

        self.segment_stats = {segment["id"]: self._new_segment_stats(segment) for segment in self.protocol_segments if segment.get("analyze")}
        self.trajectory_px_live: deque[Tuple[float, float]] = deque(maxlen=1200)
        self.trajectory_px_stats: List[Tuple[float, float]] = []
        self.frame_records: List[Dict[str, Any]] = []
        self.last_active_segment_id: Optional[str] = None

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

    @staticmethod
    def _normalize_phase(value: Any, default: str = "sociability") -> str:
        phase = str(value or default).strip().lower()
        return phase if phase in VALID_PHASES else default

    def _normalize_objects(self, objects: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for obj in objects or []:
            if not isinstance(obj, dict):
                continue
            side = str(obj.get("side") or "").strip().lower()
            if side not in SIDES:
                continue
            point = obj.get("point") or obj.get("center") or obj.get("position")
            if point is None and obj.get("x") is not None and obj.get("y") is not None:
                point = {"x": obj.get("x"), "y": obj.get("y")}
            if point is None:
                continue
            out[side] = {
                "side": side,
                "point": _coerce_point(point),
                "name": str(obj.get("name") or f"{side.title()} cup"),
                "role": _normalize_role(obj.get("role")),
            }
        return out

    def _normalize_protocol_segments(self, segments: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if self.analysis_mode == "full_protocol":
            for idx, raw in enumerate(segments or []):
                if not isinstance(raw, dict):
                    continue
                phase = self._normalize_phase(raw.get("phase"), default="ignore")
                try:
                    start_s = float(raw.get("start_s"))
                    end_s = float(raw.get("end_s"))
                except Exception:
                    continue
                if end_s <= start_s:
                    continue
                analyze = bool(raw.get("analyze", phase != "ignore")) and phase != "ignore"
                normalized.append({
                    "id": str(raw.get("id") or f"{phase}_{idx + 1}"),
                    "phase": phase,
                    "start_s": start_s,
                    "end_s": end_s,
                    "analyze": analyze,
                })
            normalized.sort(key=lambda item: (item["start_s"], item["end_s"]))
            if normalized:
                return normalized

        start_s = self.time_config.get("start") if self.time_config.get("mode") == "precise" else None
        end_s = self.time_config.get("end") if self.time_config.get("mode") == "precise" else None
        return [{
            "id": self.selected_phase,
            "phase": self.selected_phase,
            "start_s": float(start_s) if start_s is not None else -1.0e12,
            "end_s": float(end_s) if end_s is not None else 1.0e12,
            "analyze": True,
        }]

    def _normalize_role_schedule(self, schedule: Optional[Dict[str, Dict[str, str]]]) -> Dict[str, Dict[str, str]]:
        out: Dict[str, Dict[str, str]] = {}
        if isinstance(schedule, dict):
            for phase, mapping in schedule.items():
                phase_key = self._normalize_phase(phase, default="")
                if phase_key not in PHASES:
                    continue
                if not isinstance(mapping, dict):
                    continue
                out[phase_key] = {
                    side: _normalize_role(mapping.get(side))
                    for side in SIDES
                }
        object_roles = {side: _normalize_role(obj.get("role")) for side, obj in self.objects.items()}
        for phase in PHASES:
            if phase not in out:
                out[phase] = dict(object_roles)
        if not any(out.get("sociability", {}).values()):
            out["sociability"] = {"left": "social_stranger_1", "right": "empty_object"}
        if not any(out.get("social_novelty", {}).values()):
            out["social_novelty"] = {"left": "familiar_stranger_1", "right": "novel_stranger_2"}
        if not any(out.get("habituation_all", {}).values()):
            out["habituation_all"] = {"left": "empty_object", "right": "empty_object"}
        if not any(out.get("habituation_center", {}).values()):
            out["habituation_center"] = {"left": "", "right": ""}
        return out

    def _new_segment_stats(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "segment": dict(segment),
            "frames": 0,
            "detected_frames": 0,
            "duration_s": 0.0,
            "detected_time_s": 0.0,
            "chamber_time_s": {name: 0.0 for name in CHAMBERS},
            "chamber_frames": {name: 0 for name in CHAMBERS},
            "chamber_entries": {name: 0 for name in CHAMBERS},
            "transitions": 0,
            "total_distance_cm": 0.0,
            "moving_time_s": 0.0,
            "immobile_time_s": 0.0,
            "max_speed_cm_s": 0.0,
            "stable_chamber": None,
            "candidate_chamber": None,
            "candidate_dwell_s": 0.0,
            "prev_stats_centroid_px": None,
            "interaction_time_s": {side: 0.0 for side in SIDES},
            "interaction_bouts": {side: 0 for side in SIDES},
            "interaction_active": {side: False for side in SIDES},
            "head_directed_sniff_time_s": {side: 0.0 for side in SIDES},
            "proximity_only_time_s": {side: 0.0 for side in SIDES},
            "chamber_latency_s": {side: None for side in SIDES},
            "interaction_latency_s": {side: None for side in SIDES},
            "qc_flags": set(),
        }

    def initialize(self, frame_shape: Tuple[int, ...]) -> None:
        if self.detection_mask is not None:
            return
        polygons = [self.detection_polygon_img] if self.detection_polygon_img is not None else list(self.zone_polygons_img.values())
        self.detection_mask, self.detection_bbox, self._cropped_detection_mask = _build_union_mask(frame_shape, polygons)
        self.exclusion_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        for side, obj in self.objects.items():
            if self.cup_exclusion_radius_px > 0:
                cv2.circle(self.exclusion_mask, tuple(np.round(obj["point"]).astype(int)), int(round(self.cup_exclusion_radius_px)), 255, thickness=-1)
        for poly in self.stimulus_exclusion_polygons_img:
            cv2.fillPoly(self.exclusion_mask, [poly.astype(np.int32)], 255)
        x, y, w, h = self.detection_bbox
        self._cropped_exclusion_mask = self.exclusion_mask[y : y + h, x : x + w]

    def _stats_window_origin_abs(self) -> float:
        mode = self.time_config.get("mode", "global")
        start_cfg = self.time_config.get("start")
        if mode == "precise":
            return float(start_cfg or self.global_start_time_s)
        if mode == "relative" and self.first_detected_time_abs is not None:
            return float(self.first_detected_time_abs + float(start_cfg or 0.0))
        return self.global_start_time_s

    def _is_in_time_config(self, current_time: float) -> bool:
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

    def _segment_for_time(self, current_time: float) -> Optional[Dict[str, Any]]:
        if not self._is_in_time_config(current_time):
            return None
        last_index = len(self.protocol_segments) - 1
        for index, segment in enumerate(self.protocol_segments):
            start_s = float(segment["start_s"])
            end_s = float(segment["end_s"])
            in_segment = start_s <= current_time < end_s
            if index == last_index and current_time == end_s:
                in_segment = True
            if in_segment:
                return segment if segment.get("analyze") else None
        return None

    def _zone_name_for_point(self, point_px: Optional[Tuple[float, float]]) -> str:
        if point_px is None:
            return "Unknown"
        for door in DOORS:
            if _point_in_polygon(self.door_polygons_img.get(door), point_px):
                return door
        if _point_in_polygon(self.zone_polygons_img["center"], point_px):
            return "center"
        for side in SIDES:
            if _point_in_polygon(self.zone_polygons_img[side], point_px):
                return side
        return "Unknown"

    def _point_to_local_cm(self, point_px: Optional[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
        if point_px is None:
            return None, None
        return (float(point_px[0] - self._origin_px[0]) * self.cm_per_px, float(point_px[1] - self._origin_px[1]) * self.cm_per_px)

    def _is_inside_exclusion(self, point_px: Tuple[float, float]) -> bool:
        if self.cup_exclusion_radius_px > 0:
            for side, obj in self.objects.items():
                dx = float(point_px[0]) - float(obj["point"][0])
                dy = float(point_px[1]) - float(obj["point"][1])
                if math.hypot(dx, dy) <= self.cup_exclusion_radius_px:
                    return True
        for poly in self.stimulus_exclusion_polygons_img:
            if _point_in_polygon(poly, point_px):
                return True
        return False

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
        if self._cropped_exclusion_mask is not None and np.any(self._cropped_exclusion_mask):
            fg_mask = cv2.bitwise_and(fg_mask, cv2.bitwise_not(self._cropped_exclusion_mask))
        fg_mask = morph_processor.process(fg_mask, inplace=True)
        detections = detector.detect(fg_mask)
        if not detections:
            return None, fg_mask
        candidates = []
        for item in detections:
            cx, cy = item["centroid"]
            global_centroid = (float(cx + x), float(cy + y))
            if self._is_inside_exclusion(global_centroid):
                continue
            candidates.append(item)
        if not candidates:
            return None, fg_mask
        detection = max(candidates, key=lambda item: item.get("area", 0))
        cx, cy = detection["centroid"]
        detection["centroid"] = (cx + x, cy + y)
        bx, by, bw, bh = detection["bbox"]
        detection["bbox"] = (bx + x, by + y, bw, bh)
        contour = detection.get("contour")
        if contour is not None:
            detection["contour"] = contour + np.array([[x, y]])
        return detection, fg_mask

    def _update_chamber_state(self, stats: Dict[str, Any], chamber: str, current_dt: float, current_time: float, detected: bool) -> Tuple[bool, Optional[str]]:
        if chamber not in CHAMBERS:
            stats["candidate_chamber"] = None
            stats["candidate_dwell_s"] = 0.0
            return False, None
        if stats["candidate_chamber"] == chamber:
            stats["candidate_dwell_s"] += current_dt
        else:
            stats["candidate_chamber"] = chamber
            stats["candidate_dwell_s"] = current_dt
        if stats["candidate_dwell_s"] + 1e-9 < self.transition_min_dwell_s or not detected:
            return False, None
        if stats["stable_chamber"] == chamber:
            return False, None
        previous = stats["stable_chamber"]
        stats["stable_chamber"] = chamber
        if previous is not None:
            stats["transitions"] += 1
            stats["chamber_entries"][chamber] += 1
        elif chamber != "center":
            stats["chamber_entries"][chamber] += 1
        if chamber in SIDES and stats["chamber_latency_s"][chamber] is None:
            stats["chamber_latency_s"][chamber] = max(0.0, current_time - float(stats["segment"]["start_s"]))
        return previous is not None and previous != chamber, chamber

    def _head_directed(self, previous_px: Optional[Tuple[float, float]], centroid_px: Tuple[float, float], cup_px: Tuple[float, float], speed_cm_s: float) -> bool:
        if not self.use_head_direction or previous_px is None:
            return False
        move = np.array([centroid_px[0] - previous_px[0], centroid_px[1] - previous_px[1]], dtype=np.float32)
        target = np.array([cup_px[0] - centroid_px[0], cup_px[1] - centroid_px[1]], dtype=np.float32)
        move_norm = float(np.linalg.norm(move))
        target_norm = float(np.linalg.norm(target))
        if move_norm <= 1e-6 or target_norm <= 1e-6:
            return False
        cos_value = float(np.dot(move, target) / (move_norm * target_norm))
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_value))))
        confidence = min(1.0, float(speed_cm_s) / max(self.movement_threshold_cm_s * 2.0, 1e-6))
        return angle <= self.head_angle_threshold_deg and confidence >= self.head_confidence_threshold

    def _update_interactions(
        self,
        stats: Dict[str, Any],
        centroid_px: Optional[Tuple[float, float]],
        chamber: str,
        current_dt: float,
        current_time: float,
        speed_cm_s: float,
    ) -> Dict[str, bool]:
        active = {side: False for side in SIDES}
        previous_px = stats.get("prev_stats_centroid_px")
        if centroid_px is None:
            for side in SIDES:
                stats["interaction_active"][side] = False
            return active
        for side, obj in self.objects.items():
            cup = obj["point"]
            dist_px = math.hypot(float(centroid_px[0]) - float(cup[0]), float(centroid_px[1]) - float(cup[1]))
            is_active = chamber == side and dist_px <= self.interaction_radius_px
            active[side] = bool(is_active)
            if is_active:
                stats["interaction_time_s"][side] += current_dt
                if not stats["interaction_active"][side]:
                    stats["interaction_bouts"][side] += 1
                if stats["interaction_latency_s"][side] is None:
                    stats["interaction_latency_s"][side] = max(0.0, current_time - float(stats["segment"]["start_s"]))
                if self._head_directed(previous_px, centroid_px, cup, speed_cm_s):
                    stats["head_directed_sniff_time_s"][side] += current_dt
                else:
                    stats["proximity_only_time_s"][side] += current_dt
            stats["interaction_active"][side] = bool(is_active)
        return active

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
        elif self.last_all_centroid_px is not None and self.missing_time_s < self.max_gap_s:
            centroid_px = self.last_all_centroid_px
            carried = True
            self.missing_frames += 1
            self.missing_time_s += current_dt
        else:
            self.missing_frames += 1
            self.missing_time_s += current_dt

        speed_cm_s = 0.0
        if centroid_px is not None:
            if self.prev_all_centroid_px is not None:
                step_px = math.hypot(centroid_px[0] - self.prev_all_centroid_px[0], centroid_px[1] - self.prev_all_centroid_px[1])
                speed_cm_s = step_px * self.cm_per_px / current_dt
            self.prev_all_centroid_px = centroid_px
            self.trajectory_px_live.append(centroid_px)
        else:
            self.prev_all_centroid_px = None
        self.last_speed_cm_s = speed_cm_s

        zone_key = self._zone_name_for_point(centroid_px)
        chamber = zone_key if zone_key in CHAMBERS else "Unknown"
        segment = self._segment_for_time(float(current_time))
        segment_id = segment.get("id") if segment else None
        phase = segment.get("phase") if segment else "ignore"
        stats = self.segment_stats.get(segment_id) if segment_id else None
        in_stats_window = bool(stats is not None)
        transition_event = False
        entry_chamber = None
        interaction_active = {side: False for side in SIDES}
        step_distance_cm = 0.0

        if stats is not None:
            stats["frames"] += 1
            stats["duration_s"] += current_dt
            if detected:
                stats["detected_frames"] += 1
                stats["detected_time_s"] += current_dt
            if chamber in CHAMBERS:
                stats["chamber_time_s"][chamber] += current_dt
                stats["chamber_frames"][chamber] += 1
            if centroid_px is not None:
                prev_stats_px = stats.get("prev_stats_centroid_px")
                if prev_stats_px is not None:
                    step_px = math.hypot(centroid_px[0] - prev_stats_px[0], centroid_px[1] - prev_stats_px[1])
                    step_distance_cm = step_px * self.cm_per_px
                    stats["total_distance_cm"] += step_distance_cm
                transition_event, entry_chamber = self._update_chamber_state(stats, chamber, current_dt, float(current_time), bool(detected and not carried))
                stats["max_speed_cm_s"] = max(stats["max_speed_cm_s"], speed_cm_s)
                if speed_cm_s >= self.movement_threshold_cm_s:
                    stats["moving_time_s"] += current_dt
                else:
                    stats["immobile_time_s"] += current_dt
                interaction_active = self._update_interactions(stats, centroid_px, chamber, current_dt, float(current_time), speed_cm_s)
                stats["prev_stats_centroid_px"] = centroid_px
                self.trajectory_px_stats.append(centroid_px)
            else:
                stats["prev_stats_centroid_px"] = None
                for side in SIDES:
                    stats["interaction_active"][side] = False
                stats["immobile_time_s"] += current_dt
        self.last_active_segment_id = segment_id

        x_cm, y_cm = self._point_to_local_cm(centroid_px)
        record = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            **self.subject_fields,
            "frame": self.total_frames - 1,
            "time_s": _format_seconds(current_time),
            "dt_s": _format_seconds(current_dt),
            "analysis_mode": self.analysis_mode,
            "segment_id": segment_id,
            "phase": phase,
            "segment_time_s": _format_seconds(float(current_time) - float(segment["start_s"])) if segment else None,
            "detected": bool(detected),
            "carried_position": bool(carried),
            "in_stats_window": bool(in_stats_window),
            "x_px": _format_metric(centroid_px[0]) if centroid_px is not None else None,
            "y_px": _format_metric(centroid_px[1]) if centroid_px is not None else None,
            "x_cm": _format_metric(x_cm),
            "y_cm": _format_metric(y_cm),
            "speed_cm_s": _format_metric(speed_cm_s),
            "step_distance_cm": _format_metric(step_distance_cm),
            "zone_key": zone_key,
            "zone": ZONE_LABELS.get(zone_key, "Unknown"),
            "stable_chamber": stats.get("stable_chamber") if stats else None,
            "transition_event": bool(transition_event),
            "entry_chamber": entry_chamber,
            "left_interaction": bool(interaction_active.get("left")),
            "right_interaction": bool(interaction_active.get("right")),
            "left_role": self.role_schedule.get(phase, {}).get("left"),
            "right_role": self.role_schedule.get(phase, {}).get("right"),
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
            "zone_label": ZONE_LABELS.get(zone_key, "Unknown"),
            "chamber": chamber,
            "phase": phase,
            "segment_id": segment_id,
            "transition_event": transition_event,
            "interaction_active": interaction_active,
            "trajectory_px": list(self.trajectory_px_live),
            "in_stats_window": bool(in_stats_window),
            "record": record,
        }

    def _side_for_role(self, phase: str, category: str) -> Optional[str]:
        roles = self.role_schedule.get(phase, {})
        for side in SIDES:
            if _role_for_category(roles.get(side), category):
                return side
        return None

    def _role_metric(self, stats: Dict[str, Any], category: str, source: str) -> Optional[float]:
        phase = stats["segment"]["phase"]
        side = self._side_for_role(phase, category)
        if side not in SIDES:
            return None
        if source == "chamber":
            return float(stats["chamber_time_s"].get(side, 0.0))
        if source == "interaction":
            return float(stats["interaction_time_s"].get(side, 0.0))
        return None

    def _role_latency(self, stats: Dict[str, Any], category: str, source: str) -> Optional[float]:
        phase = stats["segment"]["phase"]
        side = self._side_for_role(phase, category)
        if side not in SIDES:
            return None
        if source == "chamber":
            return stats["chamber_latency_s"].get(side)
        if source == "interaction":
            return stats["interaction_latency_s"].get(side)
        return None

    def build_phase_summaries(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []
        for segment in self.protocol_segments:
            if not segment.get("analyze"):
                continue
            stats = self.segment_stats.get(segment["id"])
            if not stats:
                continue
            duration_s = float(stats["duration_s"] or 0.0)
            detection_rate = stats["detected_time_s"] / max(duration_s, 1e-9) * 100.0 if duration_s > 0 else 0.0
            mean_speed = stats["total_distance_cm"] / max(duration_s, 1e-6)
            roles = self.role_schedule.get(segment["phase"], {})
            social_chamber = self._role_metric(stats, "social", "chamber")
            object_chamber = self._role_metric(stats, "object", "chamber")
            social_interaction = self._role_metric(stats, "social", "interaction")
            object_interaction = self._role_metric(stats, "object", "interaction")
            novel_chamber = self._role_metric(stats, "novel", "chamber")
            familiar_chamber = self._role_metric(stats, "familiar", "chamber")
            novel_interaction = self._role_metric(stats, "novel", "interaction")
            familiar_interaction = self._role_metric(stats, "familiar", "interaction")
            qc_flags = set(stats["qc_flags"])
            if detection_rate < 80.0:
                qc_flags.add("low_detection")
            if duration_s >= 60.0 and stats["transitions"] == 0:
                qc_flags.add("low_exploration")
            if segment["phase"] == "habituation_all":
                left = stats["chamber_time_s"]["left"]
                right = stats["chamber_time_s"]["right"]
                side_total = left + right
                if side_total > 1e-9 and max(left, right) / side_total >= 0.7:
                    qc_flags.add("habituation_side_bias")
            if segment["phase"] == "sociability" and (self._side_for_role("sociability", "social") is None or self._side_for_role("sociability", "object") is None):
                qc_flags.add("missing_sociability_role")
            if segment["phase"] == "social_novelty" and (self._side_for_role("social_novelty", "novel") is None or self._side_for_role("social_novelty", "familiar") is None):
                qc_flags.add("missing_novelty_role")
            summary = {
                "roi_id": self.roi_id,
                "roi_name": self.roi_name,
                **self.subject_fields,
                "analysis_mode": self.analysis_mode,
                "phase": segment["phase"],
                "segment_id": segment["id"],
                "segment_start_s": _format_seconds(segment["start_s"]) if abs(float(segment["start_s"])) < 1e11 else self.time_config.get("start"),
                "segment_end_s": _format_seconds(segment["end_s"]) if abs(float(segment["end_s"])) < 1e11 else self.time_config.get("end"),
                "segment_duration_s": _format_seconds(duration_s),
                "total_time_s": _format_seconds(duration_s),
                "detected_frames": int(stats["detected_frames"]),
                "detected_rate_percent": round(detection_rate, 2),
                "total_distance_cm": _format_metric(stats["total_distance_cm"]),
                "mean_speed_cm_s": _format_metric(mean_speed),
                "max_speed_cm_s": _format_metric(stats["max_speed_cm_s"]),
                "left_time_s": _format_seconds(stats["chamber_time_s"]["left"]),
                "center_time_s": _format_seconds(stats["chamber_time_s"]["center"]),
                "right_time_s": _format_seconds(stats["chamber_time_s"]["right"]),
                "left_time_percent": _safe_percent(stats["chamber_time_s"]["left"], duration_s),
                "center_time_percent": _safe_percent(stats["chamber_time_s"]["center"], duration_s),
                "right_time_percent": _safe_percent(stats["chamber_time_s"]["right"], duration_s),
                "left_entries": int(stats["chamber_entries"]["left"]),
                "center_entries": int(stats["chamber_entries"]["center"]),
                "right_entries": int(stats["chamber_entries"]["right"]),
                "total_transitions": int(stats["transitions"]),
                "left_interaction_time_s": _format_seconds(stats["interaction_time_s"]["left"]),
                "right_interaction_time_s": _format_seconds(stats["interaction_time_s"]["right"]),
                "left_interaction_bouts": int(stats["interaction_bouts"]["left"]),
                "right_interaction_bouts": int(stats["interaction_bouts"]["right"]),
                "left_role": roles.get("left", ""),
                "right_role": roles.get("right", ""),
                "social_chamber_time_s": _format_seconds(social_chamber),
                "object_chamber_time_s": _format_seconds(object_chamber),
                "social_interaction_time_s": _format_seconds(social_interaction),
                "object_interaction_time_s": _format_seconds(object_interaction),
                "sociability_index": _preference_index(social_chamber, object_chamber),
                "sociability_interaction_index": _preference_index(social_interaction, object_interaction),
                "novel_chamber_time_s": _format_seconds(novel_chamber),
                "familiar_chamber_time_s": _format_seconds(familiar_chamber),
                "novel_interaction_time_s": _format_seconds(novel_interaction),
                "familiar_interaction_time_s": _format_seconds(familiar_interaction),
                "social_novelty_index": _preference_index(novel_chamber, familiar_chamber),
                "social_novelty_interaction_index": _preference_index(novel_interaction, familiar_interaction),
                "latency_to_social_chamber_s": _format_seconds(self._role_latency(stats, "social", "chamber")),
                "latency_to_novel_chamber_s": _format_seconds(self._role_latency(stats, "novel", "chamber")),
                "latency_to_social_interaction_s": _format_seconds(self._role_latency(stats, "social", "interaction")),
                "latency_to_novel_interaction_s": _format_seconds(self._role_latency(stats, "novel", "interaction")),
                "head_directed_sniff_time_s": _format_seconds(sum(stats["head_directed_sniff_time_s"].values())) if self.use_head_direction else None,
                "proximity_only_time_s": _format_seconds(sum(stats["proximity_only_time_s"].values())),
                "moving_time_s": _format_seconds(stats["moving_time_s"]),
                "immobility_time_s": _format_seconds(stats["immobile_time_s"]),
                "qc_flags": ";".join(sorted(qc_flags)),
            }
            summaries.append(summary)
        return summaries

    def build_summary(self) -> Dict[str, Any]:
        phases = self.build_phase_summaries()
        total_time = sum(float(row.get("total_time_s") or 0.0) for row in phases)
        detected_time = 0.0
        total_distance = 0.0
        max_speed = 0.0
        chamber_totals = {name: 0.0 for name in CHAMBERS}
        entries = {name: 0 for name in CHAMBERS}
        total_transitions = 0
        qc_flags = set()
        by_phase = {row.get("phase"): row for row in phases}
        for row in phases:
            duration = float(row.get("total_time_s") or 0.0)
            detected_time += duration * float(row.get("detected_rate_percent") or 0.0) / 100.0
            total_distance += float(row.get("total_distance_cm") or 0.0)
            max_speed = max(max_speed, float(row.get("max_speed_cm_s") or 0.0))
            for chamber in CHAMBERS:
                chamber_totals[chamber] += float(row.get(f"{chamber}_time_s") or 0.0)
                entries[chamber] += int(row.get(f"{chamber}_entries") or 0)
            total_transitions += int(row.get("total_transitions") or 0)
            for flag in str(row.get("qc_flags") or "").split(";"):
                if flag:
                    qc_flags.add(flag)
        if self.analysis_mode == "full_protocol" and "social_novelty" not in by_phase:
            qc_flags.add("missing_social_novelty")
        sociability = by_phase.get("sociability", {})
        novelty = by_phase.get("social_novelty", {})
        summary = {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            **self.subject_fields,
            "analysis_mode": self.analysis_mode,
            "phase": "protocol",
            "segment_id": "protocol",
            "phase_count": len(phases),
            "total_time_s": _format_seconds(total_time),
            "detected_rate_percent": round(detected_time / max(total_time, 1e-9) * 100.0, 2) if total_time > 0 else 0.0,
            "total_distance_cm": _format_metric(total_distance),
            "mean_speed_cm_s": _format_metric(total_distance / max(total_time, 1e-6)),
            "max_speed_cm_s": _format_metric(max_speed),
            "left_time_s": _format_seconds(chamber_totals["left"]),
            "center_time_s": _format_seconds(chamber_totals["center"]),
            "right_time_s": _format_seconds(chamber_totals["right"]),
            "left_entries": int(entries["left"]),
            "center_entries": int(entries["center"]),
            "right_entries": int(entries["right"]),
            "total_transitions": int(total_transitions),
            "sociability_index": sociability.get("sociability_index"),
            "sociability_interaction_index": sociability.get("sociability_interaction_index"),
            "social_chamber_time_s": sociability.get("social_chamber_time_s"),
            "object_chamber_time_s": sociability.get("object_chamber_time_s"),
            "social_interaction_time_s": sociability.get("social_interaction_time_s"),
            "object_interaction_time_s": sociability.get("object_interaction_time_s"),
            "social_novelty_index": novelty.get("social_novelty_index"),
            "social_novelty_interaction_index": novelty.get("social_novelty_interaction_index"),
            "novel_chamber_time_s": novelty.get("novel_chamber_time_s"),
            "familiar_chamber_time_s": novelty.get("familiar_chamber_time_s"),
            "novel_interaction_time_s": novelty.get("novel_interaction_time_s"),
            "familiar_interaction_time_s": novelty.get("familiar_interaction_time_s"),
            "latency_to_social_chamber_s": sociability.get("latency_to_social_chamber_s"),
            "latency_to_novel_chamber_s": novelty.get("latency_to_novel_chamber_s"),
            "latency_to_social_interaction_s": sociability.get("latency_to_social_interaction_s"),
            "latency_to_novel_interaction_s": novelty.get("latency_to_novel_interaction_s"),
            "head_directed_sniff_time_s": _format_seconds(sum(float(row.get("head_directed_sniff_time_s") or 0.0) for row in phases)) if self.use_head_direction else None,
            "proximity_only_time_s": _format_seconds(sum(float(row.get("proximity_only_time_s") or 0.0) for row in phases)),
            "arena_width_cm": _format_metric(self.arena_width_cm),
            "arena_height_cm": _format_metric(self.arena_height_cm),
            "interaction_radius_cm": _format_metric(self.interaction_radius_cm),
            "cup_exclusion_radius_cm": _format_metric(self.cup_exclusion_radius_cm),
            "transition_min_dwell_s": _format_seconds(self.transition_min_dwell_s),
            "center_reset_s": _format_seconds(self.center_reset_s),
            "movement_threshold_cm_s": _format_metric(self.movement_threshold_cm_s),
            "cm_per_px_est": round(self.cm_per_px, 6),
            "qc_flags": ";".join(sorted(qc_flags)),
        }
        return summary

    def build_time_bins(self, bin_seconds: float = 60.0) -> List[Dict[str, Any]]:
        rows = [row for row in self.frame_records if row.get("in_stats_window")]
        if not rows:
            return []
        bin_seconds = max(1.0, float(bin_seconds))
        bins: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for row in rows:
            segment_id = str(row.get("segment_id") or "")
            phase = str(row.get("phase") or "")
            t = float(row.get("segment_time_s") or 0.0)
            idx = int(t // bin_seconds)
            key = (segment_id, idx)
            bucket = bins.setdefault(
                key,
                {
                    "roi_id": self.roi_id,
                    "roi_name": self.roi_name,
                    **self.subject_fields,
                    "analysis_mode": self.analysis_mode,
                    "phase": phase,
                    "segment_id": segment_id,
                    "bin_index": idx + 1,
                    "start_s": idx * bin_seconds,
                    "end_s": (idx + 1) * bin_seconds,
                    "duration_s": 0.0,
                    "distance_cm": 0.0,
                    "left_time_s": 0.0,
                    "center_time_s": 0.0,
                    "right_time_s": 0.0,
                    "left_interaction_time_s": 0.0,
                    "right_interaction_time_s": 0.0,
                    "transitions": 0,
                },
            )
            duration = float(row.get("dt_s") or (1.0 / max(self.fps, 1e-6)))
            zone = row.get("zone_key")
            bucket["duration_s"] += duration
            bucket["distance_cm"] += float(row.get("step_distance_cm") or 0.0)
            if zone in CHAMBERS:
                bucket[f"{zone}_time_s"] += duration
            for side in SIDES:
                if row.get(f"{side}_interaction"):
                    bucket[f"{side}_interaction_time_s"] += duration
            bucket["transitions"] += int(bool(row.get("transition_event")))
        out: List[Dict[str, Any]] = []
        for key in sorted(bins):
            bucket = bins[key]
            duration_s = float(bucket.get("duration_s") or 0.0)
            out.append({
                **bucket,
                "start_s": round(bucket["start_s"], 2),
                "end_s": round(bucket["end_s"], 2),
                "duration_s": _format_seconds(duration_s),
                "distance_cm": _format_metric(bucket["distance_cm"]),
                "mean_speed_cm_s": _format_metric(bucket["distance_cm"] / max(duration_s, 1e-6)),
                "left_time_s": _format_seconds(bucket["left_time_s"]),
                "center_time_s": _format_seconds(bucket["center_time_s"]),
                "right_time_s": _format_seconds(bucket["right_time_s"]),
                "left_interaction_time_s": _format_seconds(bucket["left_interaction_time_s"]),
                "right_interaction_time_s": _format_seconds(bucket["right_interaction_time_s"]),
                "transitions": int(bucket["transitions"]),
            })
        return out


def _zone_color(zone_key: str, detected: bool = True) -> Tuple[int, int, int]:
    if not detected:
        return (120, 120, 120)
    if zone_key == "left":
        return (59, 130, 246)
    if zone_key == "center":
        return (34, 197, 94)
    if zone_key == "right":
        return (249, 115, 22)
    if zone_key in DOORS:
        return (14, 165, 233)
    return (148, 163, 184)


def _role_short(role: str) -> str:
    role = _normalize_role(role)
    return {
        "social_stranger_1": "S1",
        "empty_object": "Empty",
        "familiar_stranger_1": "Fam",
        "novel_stranger_2": "S2",
    }.get(role, role or "-")


def draw_three_chamber_results(
    frame: np.ndarray,
    tracker_results: Sequence[Tuple[ThreeChamberTracker, Dict[str, Any]]],
    inplace: bool = False,
) -> np.ndarray:
    output = frame if inplace else frame.copy()
    for tracker, result in tracker_results:
        phase = str(result.get("phase") or "ignore")
        roles = tracker.role_schedule.get(phase, {})
        if tracker.detection_polygon_img is not None:
            cv2.polylines(output, [tracker.detection_polygon_img.astype(np.int32)], True, (255, 214, 10), 1, cv2.LINE_AA)
        for name, poly in tracker.zone_polygons_img.items():
            fill_poly_alpha(output, poly.astype(np.int32), _zone_color(name), alpha=0.10)
            thickness = 3 if name == result.get("zone_key") else 2
            cv2.polylines(output, [poly.astype(np.int32)], True, _zone_color(name), thickness, cv2.LINE_AA)
            anchor = _polygon_centroid(poly)
            cv2.putText(output, ZONE_LABELS[name], (int(anchor[0]) - 24, int(anchor[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 23, 42), 2, cv2.LINE_AA)
        for name, poly in tracker.door_polygons_img.items():
            fill_poly_alpha(output, poly.astype(np.int32), _zone_color(name), alpha=0.18)
            cv2.polylines(output, [poly.astype(np.int32)], True, _zone_color(name), 2, cv2.LINE_AA)
        for side, obj in tracker.objects.items():
            cup = tuple(np.round(obj["point"]).astype(int))
            cv2.circle(output, cup, max(3, int(round(tracker.interaction_radius_px))), _zone_color(side), 1, cv2.LINE_AA)
            if tracker.cup_exclusion_radius_px > 0:
                cv2.circle(output, cup, max(2, int(round(tracker.cup_exclusion_radius_px))), (244, 63, 94), 1, cv2.LINE_AA)
            cv2.circle(output, cup, 4, _zone_color(side), -1, cv2.LINE_AA)
            cv2.putText(output, _role_short(roles.get(side, "")), (cup[0] + 6, cup[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (8, 15, 28), 2, cv2.LINE_AA)
        for poly in tracker.stimulus_exclusion_polygons_img:
            cv2.polylines(output, [poly.astype(np.int32)], True, (244, 63, 94), 1, cv2.LINE_AA)

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
            zone_text = f"{zone_text} | gap"
        sub = f"{phase} | {zone_text}"
        box_w = max(240, int(max(len(label), len(sub)) * 8.8))
        cv2.rectangle(output, (min_x, max(0, min_y - 44)), (min_x + box_w, max(24, min_y + 2)), (8, 15, 28), -1)
        cv2.rectangle(output, (min_x, max(0, min_y - 44)), (min_x + box_w, max(24, min_y + 2)), color, 1)
        cv2.putText(output, label, (min_x + 8, max(16, min_y - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(output, sub, (min_x + 8, max(34, min_y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 220), 1, cv2.LINE_AA)
    return output


def _render_tracker_canvas(
    tracker: ThreeChamberTracker,
    padding: int = 24,
    style: Any | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[float, float]]:
    polys = list(tracker.zone_polygons_img.values()) + list(tracker.door_polygons_img.values())
    min_x, min_y, max_x, max_y = _polygons_bbox(polys)
    width = max(220, int(max_x - min_x + padding * 2))
    height = max(160, int(max_y - min_y + padding * 2))
    offset = np.array([padding - min_x, padding - min_y], dtype=np.float32)
    translated = {name: (poly + offset).astype(np.int32) for name, poly in tracker.zone_polygons_img.items()}
    for name, poly in tracker.door_polygons_img.items():
        translated[name] = (poly + offset).astype(np.int32)
    return np.full((height, width, 3), style_color(style, "trackmap", "background"), dtype=np.uint8), translated, (float(offset[0]), float(offset[1]))


def render_three_chamber_heatmap(
    tracker: ThreeChamberTracker,
    output_path: str,
    pixels_per_cm: int = 12,
    style: Any | None = None,
) -> Optional[str]:
    if not tracker.trajectory_px_stats:
        return None
    canvas, translated, offset = _render_tracker_canvas(tracker, style=style)
    heat = np.zeros(canvas.shape[:2], dtype=np.float32)
    for x, y in tracker.trajectory_px_stats:
        px = int(np.clip(round(x + offset[0]), 0, heat.shape[1] - 1))
        py = int(np.clip(round(y + offset[1]), 0, heat.shape[0] - 1))
        cv2.circle(heat, (px, py), max(4, pixels_per_cm // 2), 1.0, thickness=-1)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(3, pixels_per_cm), sigmaY=max(3, pixels_per_cm))
    if float(heat.max()) <= 0.0:
        return None
    image = classic_heatmap(np.clip(heat / heat.max(), 0.0, 1.0), style)
    for poly in translated.values():
        cv2.polylines(image, [poly], True, style_color(style, "heatmap", "outline"), 2, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), style_color(style, "heatmap", "border"), 1, cv2.LINE_AA)
    image = append_heatmap_legend(image, style)
    if write_image_unicode(output_path, image):
        return output_path
    return None


def render_three_chamber_trackmap(
    tracker: ThreeChamberTracker,
    output_path: str,
    pixels_per_cm: int = 12,
    style: Any | None = None,
) -> Optional[str]:
    if not tracker.trajectory_px_stats:
        return None
    canvas, translated, offset = _render_tracker_canvas(tracker, style=style)
    for name, poly in translated.items():
        if name in DOORS:
            continue
        fill_poly_alpha(canvas, poly, style_color(style, "trackmap", "region_fill"), alpha=0.86)
        cv2.polylines(canvas, [poly], True, style_color(style, "trackmap", "region_outline"), 2, cv2.LINE_AA)
    points = []
    for x, y in tracker.trajectory_px_stats:
        points.append((int(np.clip(round(x + offset[0]), 0, canvas.shape[1] - 1)), int(np.clip(round(y + offset[1]), 0, canvas.shape[0] - 1))))
    thickness = classic_path_thickness(pixels_per_cm)
    for idx in range(1, len(points)):
        cv2.line(canvas, points[idx - 1], points[idx], style_color(style, "trackmap", "track"), thickness, cv2.LINE_AA)
    draw_start_end_markers(canvas, points, classic_marker_radius(pixels_per_cm), style)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), style_color(style, "trackmap", "border"), 1, cv2.LINE_AA)
    if write_image_unicode(output_path, canvas):
        return output_path
    return None


def render_three_chamber_image_grid(
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
