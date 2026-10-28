"""Barnes arena calibration, observed nose-ROI visits, and trial metrics."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from file_utils import write_image_unicode
from maze_zone_utils import build_union_mask
from water_maze import PoolGeometry

ALGORITHM_VERSION = "barnes-1"


def point_xy(value):
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"))
    try:
        x, y = map(float, value)
    except (TypeError, ValueError):
        raise ValueError("坐标必须包含有限的 x、y 数值。") from None
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("坐标必须是有限数值。")
    return x, y


def positive(value, name, *, allow_zero=False):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}必须为数值。") from None
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name}必须是{'非负' if allow_zero else '正'}有限数值。")
    return value


def metric(value):
    return round(float(value), 6) if value is not None else None


class BarnesArena:
    def __init__(
        self, arena_points, holes, target_hole_id, diameter_cm=90.0, hole_radius_cm=2.0
    ):
        self.diameter_cm = positive(diameter_cm, "圆盘直径")
        self.radius_cm = self.diameter_cm / 2
        self.hole_radius_cm = positive(hole_radius_cm, "洞口半径")
        self.points = [point_xy(p) for p in arena_points]
        if len(self.points) < 5:
            raise ValueError("请在圆盘边界标定至少5个点。")
        try:
            self.geometry = PoolGeometry(self.points, pool_diameter_cm=self.diameter_cm)
        except cv2.error as exc:
            raise ValueError("圆盘边界无法拟合，请重新标定边界点。") from exc
        if not isinstance(holes, list) or len(holes) < 2:
            raise ValueError("请标出全部洞位，至少需要2个洞。")
        self.holes = []
        ids = set()
        for raw in holes:
            hole_id = str(raw.get("id", "")).strip()
            if not hole_id or hole_id in ids:
                raise ValueError("洞位编号不能为空或重复。")
            ids.add(hole_id)
            image_point = point_xy(raw.get("point"))
            center = self.to_cm(image_point)
            radius = positive(raw.get("radius_cm", self.hole_radius_cm), "洞口半径")
            if math.hypot(*center) > self.radius_cm + 1e-3:
                raise ValueError(f"洞位 {hole_id} 不在标定圆盘内。")
            for other in self.holes:
                if math.dist(center, other["center_cm"]) < radius + other["radius_cm"]:
                    raise ValueError("洞口ROI重叠，请核对洞位和实际洞口半径。")
            self.holes.append(
                dict(id=hole_id, point=image_point, center_cm=center, radius_cm=radius)
            )
        self.target_hole_id = str(target_hole_id).strip()
        if self.target_hole_id not in ids:
            raise ValueError("目标洞必须对应一个已标定洞位。")
        self.target = next(h for h in self.holes if h["id"] == self.target_hole_id)
        self.circular_ids = [
            h["id"]
            for h in sorted(
                self.holes,
                key=lambda h: math.atan2(h["center_cm"][1], h["center_cm"][0]),
            )
        ]

    def to_cm(self, point):
        # PoolGeometry's public projection clamps outside points to the rim.
        # A nose outside the platform must not be moved into a hole ROI.
        p = np.array([[point_xy(point)]], np.float32)
        mapped = cv2.transform(p, self.geometry.image_to_pool_matrix)[0, 0]
        return float(mapped[0]), float(mapped[1])

    def to_image(self, point):
        return self.geometry.pool_to_image(point)

    def hole_at(self, point_cm):
        if point_cm is None:
            return None
        return next(
            (
                h["id"]
                for h in self.holes
                if math.dist(point_cm, h["center_cm"]) <= h["radius_cm"]
            ),
            None,
        )

    def image_outline(self):
        return self.geometry.image_boundary_points()


def estimate_silhouette_landmarks(contour):
    """Conservative tail/body geometry; unknown head stays unknown.

    Quality is a heuristic, not a calibrated probability of correctness.
    This function deliberately has no knowledge of holes or their target.
    """
    if contour is None or len(contour) < 3:
        return {"keypoints": {}, "source": "classical", "reason": "no_contour"}
    x, y, w, h = cv2.boundingRect(contour)
    local = np.zeros((h + 4, w + 4), np.uint8)
    shifted = contour.astype(np.int32) - np.array([[[x - 2, y - 2]]], np.int32)
    cv2.drawContours(local, [shifted], -1, 255, -1)
    m = cv2.moments(local)
    if m["m00"] < 20:
        return {"keypoints": {}, "source": "classical", "reason": "small_contour"}
    radius = math.sqrt(cv2.countNonZero(local) / math.pi)
    size = max(3, int(radius * 0.5) // 2 * 2 + 1)
    opened = cv2.morphologyEx(
        local,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
    )
    body_contours, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not body_contours:
        return {"keypoints": {}, "source": "classical", "reason": "body_unresolved"}
    body_contour = max(body_contours, key=cv2.contourArea)
    bm = cv2.moments(body_contour)
    if bm["m00"] <= 0:
        return {"keypoints": {}, "source": "classical", "reason": "body_unresolved"}
    center = np.array([bm["m10"] / bm["m00"], bm["m01"] / bm["m00"]], np.float32)
    body = np.zeros_like(local)
    cv2.drawContours(body, [body_contour], -1, 255, -1)
    tail_mask = cv2.bitwise_and(local, cv2.bitwise_not(body))
    _, labels, sizes, _ = cv2.connectedComponentsWithStats(tail_mask)
    candidates = []
    for label in range(1, len(sizes)):
        ys, xs = np.where(labels == label)
        if len(xs) < max(5, cv2.countNonZero(local) * 0.015):
            continue
        pixels = np.column_stack([xs, ys]).astype(np.float32)
        distances = np.linalg.norm(pixels - center, axis=1)
        eigenvalues = np.linalg.eigvalsh(np.cov(pixels.T))
        elongation = float(eigenvalues[-1] / max(eigenvalues[0], 0.1))
        if elongation < 4 or float(distances.max()) < radius * 1.2:
            continue
        candidates.append(
            (float(distances.max()), pixels[int(distances.argmin())], elongation)
        )
    offset = np.array([x - 2, y - 2], np.float32)

    def keypoint(p, confidence):
        return dict(
            x=float(p[0] + offset[0]),
            y=float(p[1] + offset[1]),
            confidence=float(confidence),
            visible=True,
        )

    points = {"body_center": keypoint(center, 1.0)}
    if not candidates:
        return dict(keypoints=points, source="classical", reason="tail_unresolved")
    _, root, elongation = max(candidates, key=lambda c: c[0])
    direction = center - root
    length = float(np.linalg.norm(direction))
    if length < radius * 0.25:
        return dict(keypoints=points, source="classical", reason="head_ambiguous")
    direction /= length
    vertices = body_contour.reshape(-1, 2).astype(np.float32)
    projection = (vertices - center) @ direction
    nose = vertices[int(projection.argmax())]
    quality = min(
        0.95,
        0.35
        + 0.3 * min(1.0, elongation / 20.0)
        + 0.3 * min(1.0, float(projection.max()) / max(radius, 1.0)),
    )
    points.update(nose=keypoint(nose, quality), tail_base=keypoint(root, quality))
    return dict(keypoints=points, source="classical", reason="silhouette_tail_geometry")


def validate_manual_events(events, arena, start_s=0.0, end_s=None):
    if events is None:
        return None
    if not isinstance(events, list):
        raise ValueError("人工洞位事件必须为列表。")
    cleaned = []
    previous = -math.inf
    for raw in events:
        t = positive(raw.get("time_s"), "人工事件时间", allow_zero=True)
        hole = str(raw.get("hole_id", "")).strip()
        if (
            hole not in arena.circular_ids
            or t <= previous
            or t < start_s
            or (end_s is not None and t > end_s)
        ):
            raise ValueError("人工事件必须时间递增、位于分析时段内，且洞位已标定。")
        cleaned.append(dict(time_s=t, hole_id=hole))
        previous = t
    return cleaned


class BarnesScorer:
    def __init__(
        self,
        arena,
        *,
        roi_id=1,
        roi_name="Barnes 1",
        start_s=0.0,
        end_s=None,
        trial_type="acquisition",
        visit_confirm_s=0.1,
        visit_reset_s=0.1,
        head_confidence_threshold=0.6,
        tracking_max_jump_cm=12.0,
        manual_escape_time_s=None,
        manual_events=None,
        tracking_backend="classical",
        escape_absence_s=0.5,
        escape_zone_radius_cm=6.0,
    ):
        if trial_type not in {"acquisition", "reversal", "probe"}:
            raise ValueError("不支持的巴恩斯试次类型。")
        self.arena = arena
        self.roi_id = int(roi_id)
        self.roi_name = str(roi_name)
        self.start_s = positive(start_s, "开始时间", allow_zero=True)
        self.end_s = (
            None if end_s is None else positive(end_s, "结束时间", allow_zero=True)
        )
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("结束时间必须晚于开始时间。")
        self.trial_type = trial_type
        self.backend = tracking_backend
        self.confirm_s = positive(visit_confirm_s, "洞位确认时长", allow_zero=True)
        self.reset_s = positive(visit_reset_s, "撤出确认时长", allow_zero=True)
        self.confidence = float(head_confidence_threshold)
        if not 0 <= self.confidence <= 1:
            raise ValueError("关键点质量阈值必须在0到1之间。")
        self.max_jump = positive(tracking_max_jump_cm, "最大追踪跳变", allow_zero=True)
        self.escape_absence_s = positive(escape_absence_s, "入箱候选消失时长")
        self.escape_zone_radius_cm = positive(escape_zone_radius_cm, "入箱候选邻域")
        self.escape_time = (
            None
            if manual_escape_time_s is None
            else positive(manual_escape_time_s, "人工确认入箱时间", allow_zero=True)
        )
        if self.escape_time is not None and (
            trial_type == "probe"
            or self.escape_time < self.start_s
            or (self.end_s is not None and self.escape_time > self.end_s)
        ):
            raise ValueError(
                "人工入箱时间必须位于训练试次的分析时段内。探测试次不设置入箱。"
            )
        self.manual_events = validate_manual_events(
            manual_events, arena, self.start_s, self.end_s
        )
        if self.escape_time is not None and any(
            e["time_s"] > self.escape_time for e in self.manual_events or []
        ):
            raise ValueError("人工洞位事件不能晚于已确认入箱时间。")
        self._manual_index = 0
        self.events = []
        self.records = []
        self.hole_time = {h["id"]: 0.0 for h in arena.holes}
        self.distance = 0.0
        self.duration = 0.0
        self.body_time = 0.0
        self.head_time = 0.0
        self.target_sector_time = 0.0
        self.max_speed = 0.0
        self.qc = set()
        self._previous_body = None
        self._first_body = None
        self._last_body = None
        self._pending = None
        self._pending_start = 0.0
        self._pending_path = 0.0
        self._pending_duration = 0.0
        self._active = None
        self._outside_duration = 0.0
        self._missing_duration = 0.0
        self.escape_candidate = None
        self._near_target_before_missing = False
        self._last_t = None
        self._last_dt = 0.0
        self._completed_until = None
        self.first_target = None
        self.primary_path = None

    def _point(self, observation, name, threshold):
        raw = (observation or {}).get("keypoints", {}).get(name)
        if not isinstance(raw, dict) or not raw.get("visible", True):
            return None
        try:
            confidence = float(raw.get("confidence", 0.0))
            if not math.isfinite(confidence) or not threshold <= confidence <= 1:
                return None
            return point_xy(raw)
        except (TypeError, ValueError):
            return None

    def _emit(self, hole, time_s, path, source):
        event = dict(
            event_index=len(self.events) + 1,
            roi_id=self.roi_id,
            roi_name=self.roi_name,
            time_s=metric(time_s),
            confirmed_time_s=metric(time_s if source == "manual" else self._last_t),
            latency_s=metric(time_s - self.start_s),
            hole_id=hole,
            target=hole == self.arena.target_hole_id,
            source=source,
            path_length_cm=metric(path),
            criterion=(
                "manual_annotation" if source == "manual" else "observed_nose_roi_entry"
            ),
        )
        self.events.append(event)
        if event["target"] and self.first_target is None:
            self.first_target = event
            self.primary_path = path
        return event

    def update(self, time_s, dt_s, observation):
        t = float(time_s)
        dt = positive(dt_s, "帧间隔")
        if not math.isfinite(t) or (self._last_t is not None and t <= self._last_t):
            raise ValueError("源时间戳必须为有限递增数值。")
        previous_time = self._last_t
        self._last_t = t
        in_window = (
            t >= self.start_s
            and (self.end_s is None or t < self.end_s)
            and (self.escape_time is None or t < self.escape_time)
        )
        if in_window:
            for limit in (self.end_s, self.escape_time):
                if limit is not None:
                    dt = min(dt, limit - t)
        self._last_dt = dt
        body_px = self._point(
            observation,
            "body_center",
            self.confidence if self.backend == "deeplabcut" else 0.0,
        )
        nose_px = self._point(observation, "nose", self.confidence)
        body = self.arena.to_cm(body_px) if body_px is not None else None
        nose = self.arena.to_cm(nose_px) if nose_px is not None else None
        if body is not None and math.hypot(*body) > self.arena.radius_cm:
            body = None
            body_px = None
            nose = None
            nose_px = None
        jump = (
            body is not None
            and self._previous_body is not None
            and self.max_jump > 0
            and math.dist(body, self._previous_body) > self.max_jump
        )
        if jump:
            self.qc.add("body_jump_rejected")
            body = None
            body_px = None
            nose = None
            nose_px = None
        hole = self.arena.hole_at(nose)
        emitted = None
        speed = 0.0
        step = 0.0
        if in_window:
            self.duration += dt
            if body is not None:
                self.body_time += dt
                if self._previous_body is not None:
                    step = math.dist(body, self._previous_body)
                    self.distance += step
                    speed = (
                        step / (t - previous_time) if previous_time is not None else 0.0
                    )
                self._first_body = (
                    self._first_body if self._first_body is not None else body
                )
                self._last_body = body
                self.max_speed = max(self.max_speed, speed)
                if self.escape_candidate is not None:
                    self.qc.add("body_reappeared_after_escape_candidate")
                    self.escape_candidate = None
                target_angle = math.atan2(
                    self.arena.target["center_cm"][1], self.arena.target["center_cm"][0]
                )
                body_angle = math.atan2(body[1], body[0])
                difference = (body_angle - target_angle + math.pi) % (
                    2 * math.pi
                ) - math.pi
                if abs(difference) <= math.pi / 4:
                    self.target_sector_time += dt
                self._near_target_before_missing = (
                    hole == self.arena.target_hole_id
                    and math.dist(body, self.arena.target["center_cm"])
                    <= self.escape_zone_radius_cm
                )
                self._missing_duration = 0.0
            else:
                self._missing_duration += dt
                if (
                    self._near_target_before_missing
                    and self._missing_duration >= self.escape_absence_s
                    and self.trial_type != "probe"
                    and self.escape_time is None
                ):
                    self.escape_candidate = t - self._missing_duration + dt
                    self.qc.add("escape_candidate_requires_confirmation")
            if nose is not None:
                self.head_time += dt
                if hole is not None:
                    self.hole_time[hole] += dt
            if self.manual_events is not None:
                while (
                    self._manual_index < len(self.manual_events)
                    and self.manual_events[self._manual_index]["time_s"] <= t
                ):
                    e = self.manual_events[self._manual_index]
                    emitted = self._emit(
                        e["hole_id"], e["time_s"], self.distance, "manual"
                    )
                    self._manual_index += 1
            elif nose is None:
                self._pending = None
                self._pending_duration = 0.0
                self._outside_duration = 0.0
                # Missing data is not an observed withdrawal from a hole.
                self.qc.add("unobserved_nose_frames")
            elif hole == self._active and hole is not None:
                self._pending = None
                self._pending_duration = 0.0
                self._outside_duration = 0.0
            elif hole is None:
                self._pending = None
                self._pending_duration = 0.0
                self._outside_duration += dt
                if self._outside_duration + 1e-9 >= self.reset_s:
                    self._active = None
            else:
                self._outside_duration = 0.0
                if hole != self._pending:
                    self._pending = hole
                    self._pending_start = t
                    self._pending_duration = 0.0
                    self._pending_path = self.distance
                self._pending_duration += dt
                if self._pending_duration + 1e-9 >= self.confirm_s:
                    emitted = self._emit(
                        hole, self._pending_start, self._pending_path, self.backend
                    )
                    self._active = hole
                    self._pending = None
                    self._pending_duration = 0.0
        self._previous_body = body if in_window else None
        record = dict(
            roi_id=self.roi_id,
            roi_name=self.roi_name,
            time_s=metric(t),
            dt_s=metric(dt),
            in_stats_window=in_window,
            detected=body is not None,
            nose_observed=nose is not None,
            tracking_backend=self.backend,
            landmark_source=(observation or {}).get("source", self.backend),
            x_px=metric(body_px[0]) if body_px is not None else None,
            y_px=metric(body_px[1]) if body_px is not None else None,
            x_cm=metric(body[0]) if body is not None else None,
            y_cm=metric(body[1]) if body is not None else None,
            nose_x_px=metric(nose_px[0]) if nose_px is not None else None,
            nose_y_px=metric(nose_px[1]) if nose_px is not None else None,
            head_confidence=metric(
                ((observation or {}).get("keypoints", {}).get("nose") or {}).get(
                    "confidence"
                )
            ),
            hole_id=hole,
            hole_visit_event=emitted is not None,
            hole_visit_count=len(self.events),
            speed_cm_s=metric(speed),
            step_distance_cm=metric(step),
            cumulative_distance_cm=metric(self.distance),
            escape_candidate_time_s=metric(self.escape_candidate),
        )
        self.records.append(record)
        return record

    def strategy_candidate(self):
        visits = [
            e["hole_id"]
            for e in self.events
            if self.first_target is None or e["time_s"] <= self.first_target["time_s"]
        ]
        if (
            len(visits) < 3
            or self.duration <= 0
            or self.head_time / self.duration < 0.8
        ):
            return "uncertain"
        indices = [self.arena.circular_ids.index(h) for h in visits]
        n = len(self.arena.holes)
        moves = [(b - a) % n for a, b in zip(indices, indices[1:]) if a != b]
        adjacent = [m for m in moves if m in {1, n - 1}]
        if (
            moves
            and len(adjacent) / len(moves) >= 0.75
            and max(adjacent.count(1), adjacent.count(n - 1)) / len(moves) >= 0.75
        ):
            return "serial_candidate"
        if self.first_target is not None and len(visits) <= 4:
            return "spatial_candidate"
        return "mixed_or_random_candidate"

    def build_summary(self):
        flags = set(self.qc)
        if self._completed_until is not None and self._manual_index < len(
            self.manual_events or []
        ):
            flags.add("manual_events_outside_observed_video")
        observed_until = (
            self._completed_until if self._completed_until is not None else self._last_t
        )
        escape_observed = (
            self.escape_time is not None
            and observed_until is not None
            and observed_until + 1e-6 >= self.escape_time
        )
        before_target = [
            e
            for e in self.events
            if self.first_target is None or e["time_s"] < self.first_target["time_s"]
        ]
        errors = sum(not e["target"] for e in self.events)
        if self.head_time < self.duration * 0.8:
            flags.add("low_observed_nose_coverage")
        if self.body_time < self.duration * 0.9:
            flags.add("incomplete_observed_path")
        return dict(
            algorithm_version=ALGORITHM_VERSION,
            roi_id=self.roi_id,
            roi_name=self.roi_name,
            trial_type=self.trial_type,
            tracking_backend=self.backend,
            event_source="manual" if self.manual_events is not None else self.backend,
            visit_criterion="observed_nose_roi_entry",
            target_hole_id=self.arena.target_hole_id,
            hole_count=len(self.arena.holes),
            analysis_start_s=metric(self.start_s),
            analysis_duration_s=metric(self.duration),
            total_distance_cm=metric(self.distance),
            mean_speed_cm_s=(
                metric(self.distance / self.body_time) if self.body_time else None
            ),
            max_speed_cm_s=metric(self.max_speed),
            body_observed_percent=(
                metric(self.body_time / self.duration * 100) if self.duration else None
            ),
            nose_observed_percent=(
                metric(self.head_time / self.duration * 100) if self.duration else None
            ),
            target_sector_time_s=metric(self.target_sector_time),
            target_sector_percent=(
                metric(self.target_sector_time / self.duration * 100)
                if self.duration
                else None
            ),
            hole_visit_count=len(self.events),
            hole_visit_sequence="-".join(e["hole_id"] for e in self.events),
            target_hole_visits=sum(e["target"] for e in self.events),
            primary_errors=sum(not e["target"] for e in before_target),
            primary_errors_censored=self.first_target is None,
            primary_latency_s=(
                self.first_target["latency_s"] if self.first_target else None
            ),
            primary_latency_censored=self.first_target is None,
            primary_path_length_cm=metric(self.primary_path),
            total_errors=errors,
            total_errors_censored=self.trial_type != "probe" and not escape_observed,
            total_latency_s=(
                metric(self.escape_time - self.start_s) if escape_observed else None
            ),
            total_latency_censored=self.trial_type != "probe" and not escape_observed,
            escape_status=(
                "not_applicable"
                if self.trial_type == "probe"
                else ("confirmed_manual" if escape_observed else "not_confirmed")
            ),
            escape_candidate_time_s=metric(self.escape_candidate),
            search_strategy_candidate=self.strategy_candidate(),
            search_strategy_rule="circular-adjacency-v1-exploratory",
            qc_flags=";".join(sorted(flags)),
        )

    def finish(self):
        self._completed_until = (
            self._last_t + self._last_dt if self._last_t is not None else self.start_s
        )
        if self.end_s is not None:
            self._completed_until = min(self._completed_until, self.end_s)
        limit = (
            min(self._completed_until, self.escape_time)
            if self.escape_time is not None
            else self._completed_until
        )
        while (
            self.manual_events is not None
            and self._manual_index < len(self.manual_events)
            and self.manual_events[self._manual_index]["time_s"] <= limit + 1e-6
        ):
            event = self.manual_events[self._manual_index]
            self._emit(event["hole_id"], event["time_s"], self.distance, "manual")
            self._manual_index += 1
        return self.build_summary()

    def hole_summary(self):
        return [
            dict(
                roi_id=self.roi_id,
                roi_name=self.roi_name,
                hole_id=h["id"],
                target=h["id"] == self.arena.target_hole_id,
                visits=sum(e["hole_id"] == h["id"] for e in self.events),
                nose_occupancy_s=metric(self.hole_time[h["id"]]),
            )
            for h in self.arena.holes
        ]


class BarnesTracker:
    """One arena's frame observations; geometry and event scoring stay separate."""

    def __init__(self, arena, *, detection_points=None, fps=30.0, **scoring):
        self.arena = arena
        self.arena_points = np.asarray(arena.image_outline(), np.float32)
        self.detection_points = (
            None
            if detection_points is None or len(detection_points) == 0
            else np.asarray([point_xy(p) for p in detection_points], np.float32)
        )
        self.scorer = BarnesScorer(arena, **scoring)
        self.roi_id = self.scorer.roi_id
        self.roi_name = self.scorer.roi_name
        self.fps = float(fps)
        self.detection_mask = None
        self.detection_bbox = None
        self.frame_records = self.scorer.records
        self.last_keypoints = {}

    def initialize(self, frame_shape):
        if self.detection_mask is not None:
            return
        polygon = (
            self.detection_points
            if self.detection_points is not None
            else self.arena_points
        )
        self.detection_mask, self.detection_bbox, self._cropped_detection_mask = (
            build_union_mask(frame_shape, {"arena": polygon})
        )

    def detect_landmarks(self, frame, gray, bg, morph, detector):
        self.initialize(frame.shape)
        x, y, w, h = self.detection_bbox
        foreground = bg.apply(frame, gray=gray)
        mask = cv2.bitwise_and(
            foreground[y : y + h, x : x + w], self._cropped_detection_mask
        )
        detections = detector.detect(morph.process(mask, inplace=True))
        if not detections:
            return dict(keypoints={}, source="classical", reason="no_mouse_detected")
        detection = max(detections, key=lambda d: d["area"])
        contour = detection.get("contour")
        if contour is None:
            return dict(keypoints={}, source="classical", reason="no_mouse_contour")
        return estimate_silhouette_landmarks(contour + np.array([[[x, y]]], np.int32))

    def update(
        self,
        frame,
        time_s,
        gray,
        bg,
        morph,
        detector,
        *,
        dt_s=None,
        pose_observation=None,
    ):
        if self.scorer.backend == "deeplabcut":
            observation = pose_observation or dict(keypoints={}, source="deeplabcut")
        else:
            observation = self.detect_landmarks(frame, gray, bg, morph, detector)
        self.last_keypoints = observation.get("keypoints") or {}
        result = self.scorer.update(
            time_s, dt_s if dt_s is not None else 1 / self.fps, observation
        )
        result["landmark_reason"] = observation.get("reason")
        return result

    def build_summary(self):
        return self.scorer.build_summary()

    def build_time_bins(self, bin_seconds=60.0):
        width = positive(bin_seconds, "分箱长度")
        bins = {}
        for row in self.frame_records:
            if not row["in_stats_window"]:
                continue
            index = int(max(0, row["time_s"] - self.scorer.start_s) // width)
            b = bins.setdefault(
                index,
                dict(
                    roi_id=self.roi_id,
                    roi_name=self.roi_name,
                    bin_index=index + 1,
                    start_s=metric(self.scorer.start_s + index * width),
                    end_s=metric(self.scorer.start_s + (index + 1) * width),
                    observed_body_s=0.0,
                    observed_nose_s=0.0,
                    observed_distance_cm=0.0,
                    hole_visits=0,
                    target_visits=0,
                ),
            )
            b["observed_body_s"] += row["dt_s"] if row["detected"] else 0.0
            b["observed_nose_s"] += row["dt_s"] if row["nose_observed"] else 0.0
            b["observed_distance_cm"] += row["step_distance_cm"]
        for event in self.scorer.events:
            index = int(max(0, event["time_s"] - self.scorer.start_s) // width)
            if index in bins:
                bins[index]["hole_visits"] += 1
                bins[index]["target_visits"] += int(event["target"])
        return list(bins.values())


def hole_outline(arena, hole, count=32):
    return np.asarray(
        [
            arena.to_image(
                (
                    hole["center_cm"][0] + hole["radius_cm"] * math.cos(t),
                    hole["center_cm"][1] + hole["radius_cm"] * math.sin(t),
                )
            )
            for t in np.linspace(0, 2 * math.pi, count, endpoint=False)
        ],
        np.int32,
    )


def draw_barnes_results(frame, trackers, results):
    out = frame.copy()
    for tracker, result in zip(trackers, results):
        cv2.polylines(
            out,
            [tracker.arena_points.astype(np.int32)],
            True,
            (60, 180, 80),
            2,
            cv2.LINE_AA,
        )
        for hole in tracker.arena.holes:
            target = hole["id"] == tracker.arena.target_hole_id
            color = (40, 190, 40) if target else (230, 160, 30)
            cv2.polylines(
                out, [hole_outline(tracker.arena, hole)], True, color, 2, cv2.LINE_AA
            )
            x, y = map(round, hole["point"])
            cv2.putText(
                out,
                hole["id"],
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        for name, p in tracker.last_keypoints.items():
            if (
                p.get("visible", True)
                and p.get("confidence", 0) >= tracker.scorer.confidence
            ):
                cv2.circle(
                    out,
                    (round(p["x"]), round(p["y"])),
                    3,
                    (40, 40, 240) if name == "nose" else (180, 80, 230),
                    -1,
                )
        anchor = tracker.arena.geometry.ellipse_center_px
        label = f"{tracker.roi_name} | visits {result['hole_visit_count']} | {tracker.scorer.backend}"
        cv2.putText(
            out,
            label,
            (max(3, int(anchor[0]) - 100), max(22, int(anchor[1]))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (30, 70, 240),
            1,
            cv2.LINE_AA,
        )
    return out


def render_barnes_map(tracker, path, *, heatmap=False, style=None):
    from spatial_render import classic_heatmap, style_color

    radius = tracker.arena.radius_cm
    scale = 220.0 / radius
    size = 520
    center = np.array([size / 2, size / 2])
    out = np.empty((size, size, 3), np.uint8)
    out[:] = style_color(style, "trackmap", "background")
    ink = style_color(style, "trackmap", "text")
    histogram = np.zeros((size, size), np.float32)

    def pixel(point):
        return tuple(np.round(np.asarray(point) * scale + center).astype(int))

    previous = None
    first = None
    last = None
    for row in tracker.frame_records:
        if not row["in_stats_window"] or not row["detected"]:
            previous = None
            continue
        p = pixel((row["x_cm"], row["y_cm"]))
        if not 0 <= p[0] < size or not 0 <= p[1] < size:
            previous = None
            continue
        first = p if first is None else first
        last = p
        if heatmap:
            histogram[p[1], p[0]] += row["dt_s"]
        elif previous is not None:
            cv2.line(
                out,
                previous,
                p,
                style_color(style, "trackmap", "track"),
                2,
                cv2.LINE_AA,
            )
        previous = p
    if heatmap:
        blurred = cv2.GaussianBlur(histogram, (0, 0), sigmaX=5)
        if blurred.max() > 0:
            colored = classic_heatmap(blurred / blurred.max(), style)
            alpha = np.clip(blurred / blurred.max(), 0, 1)[:, :, None]
            out = np.uint8(out * (1 - alpha) + colored * alpha)
    cv2.circle(
        out,
        tuple(center.astype(int)),
        round(radius * scale),
        ink,
        2,
        cv2.LINE_AA,
    )
    for hole in tracker.arena.holes:
        p = pixel(hole["center_cm"])
        target = hole["id"] == tracker.arena.target_hole_id
        cv2.circle(
            out,
            p,
            max(3, round(hole["radius_cm"] * scale)),
            style_color(style, "trackmap", "platform") if target else ink,
            2,
        )
        cv2.putText(
            out,
            hole["id"],
            (p[0] + 4, p[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            ink,
            1,
        )
    if not heatmap:
        if first is not None:
            cv2.circle(out, first, 5, style_color(style, "trackmap", "start"), -1)
        if last is not None:
            cv2.circle(out, last, 5, style_color(style, "trackmap", "end"), -1)
    cv2.putText(
        out,
        "Observed body occupancy" if heatmap else "Observed body path",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        ink,
        1,
    )
    return bool(write_image_unicode(path, out))
