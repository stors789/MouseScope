#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np


def sort_polygon_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("Polygon points must be shaped as Nx2.")
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    return pts[order].astype(np.float32)


def polygon_centroid(points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    return np.mean(pts, axis=0).astype(np.float32)


def polygon_edge_lengths(points: Sequence[Sequence[float]]) -> List[float]:
    pts = sort_polygon_points(points)
    lengths: List[float] = []
    for idx in range(len(pts)):
        nxt = pts[(idx + 1) % len(pts)]
        lengths.append(float(np.linalg.norm(nxt - pts[idx])))
    return lengths


def quad_short_long_edges(points: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    lengths = sorted(polygon_edge_lengths(points))
    if len(lengths) != 4:
        raise ValueError("Quadrilateral requires exactly 4 edges.")
    return lengths[:2], lengths[2:]


def polygon_bbox(points: Sequence[Sequence[float]]) -> Tuple[int, int, int, int]:
    pts = np.asarray(points, dtype=np.float32)
    min_xy = np.floor(np.min(pts, axis=0)).astype(int)
    max_xy = np.ceil(np.max(pts, axis=0)).astype(int)
    return int(min_xy[0]), int(min_xy[1]), int(max_xy[0]), int(max_xy[1])


def polygons_bbox(polygons: Iterable[Sequence[Sequence[float]]]) -> Tuple[int, int, int, int]:
    boxes = [polygon_bbox(poly) for poly in polygons]
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return min_x, min_y, max_x, max_y


def polygons_bbox_float(polygons: Iterable[Sequence[Sequence[float]]]) -> Tuple[float, float, float, float]:
    polys = [np.asarray(poly, dtype=np.float32) for poly in polygons]
    if not polys:
        raise ValueError("At least one polygon is required.")
    stacked = np.concatenate(polys, axis=0)
    min_xy = np.min(stacked, axis=0)
    max_xy = np.max(stacked, axis=0)
    return float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])


def shift_polygon(points: Sequence[Sequence[float]], dx: float, dy: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    return (pts + np.array([dx, dy], dtype=np.float32)).astype(np.float32)


def point_in_polygon(points: Sequence[Sequence[float]], point: Sequence[float]) -> bool:
    poly = np.asarray(points, dtype=np.float32)
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def normalize_axis(vec: Sequence[float]) -> np.ndarray:
    axis = np.asarray(vec, dtype=np.float32).reshape(2)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (axis / norm).astype(np.float32)


def order_quad_by_axes(
    points: Sequence[Sequence[float]],
    x_axis: Sequence[float],
    y_axis: Sequence[float],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Quadrilateral requires exactly 4 points.")

    center = np.mean(pts, axis=0)
    axis_x = normalize_axis(x_axis)
    axis_y = normalize_axis(y_axis)
    rel = pts - center
    proj_x = rel @ axis_x
    proj_y = rel @ axis_y
    sums = proj_x + proj_y
    diffs = proj_x - proj_y

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmax(diffs)]
    ordered[3] = pts[np.argmin(diffs)]
    return ordered


def order_triangle_by_axes(
    points: Sequence[Sequence[float]],
    x_axis: Sequence[float],
    y_axis: Sequence[float],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (3, 2):
        raise ValueError("Triangle requires exactly 3 points.")

    center = np.mean(pts, axis=0)
    axis_x = normalize_axis(x_axis)
    axis_y = normalize_axis(y_axis)
    rel = pts - center
    proj_x = rel @ axis_x
    proj_y = rel @ axis_y

    bottom_idx = int(np.argmax(proj_y))
    top_indices = [idx for idx in range(3) if idx != bottom_idx]
    top_left_idx, top_right_idx = sorted(top_indices, key=lambda idx: float(proj_x[idx]))

    return np.vstack([pts[top_left_idx], pts[top_right_idx], pts[bottom_idx]]).astype(np.float32)


def order_quad_along_axis(
    points: Sequence[Sequence[float]],
    center_point: Sequence[float],
    axis_unit: Sequence[float],
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Quadrilateral requires exactly 4 points.")

    origin = np.asarray(center_point, dtype=np.float32).reshape(2)
    axis = normalize_axis(axis_unit)
    perp = np.array([-axis[1], axis[0]], dtype=np.float32)
    rel = pts - origin
    axis_proj = rel @ axis
    perp_proj = rel @ perp
    axis_order = np.argsort(axis_proj)
    near_idx = axis_order[:2]
    far_idx = axis_order[2:]
    near = pts[near_idx][np.argsort(perp_proj[near_idx])]
    far = pts[far_idx][np.argsort(perp_proj[far_idx])]
    return np.vstack([near[0], near[1], far[1], far[0]]).astype(np.float32)


def project_point_homography(
    point: Sequence[float],
    homography: np.ndarray,
) -> Tuple[float, float]:
    src = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)[0, 0]
    return float(dst[0]), float(dst[1])


def render_canvas_from_cm_polygons(
    polygons_cm: Dict[str, Sequence[Sequence[float]]],
    pixels_per_cm: int,
    padding_cm: float = 2.0,
    min_size_px: int = 200,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[float, float]]:
    min_x, min_y, max_x, max_y = polygons_bbox_float(polygons_cm.values())
    offset_x = float(padding_cm - min_x)
    offset_y = float(padding_cm - min_y)
    width_cm = float(max_x - min_x + padding_cm * 2.0)
    height_cm = float(max_y - min_y + padding_cm * 2.0)
    width_px = max(int(min_size_px), int(math.ceil(width_cm * pixels_per_cm)) + 1)
    height_px = max(int(min_size_px), int(math.ceil(height_cm * pixels_per_cm)) + 1)

    translated: Dict[str, np.ndarray] = {}
    for name, poly in polygons_cm.items():
        pts = np.asarray(poly, dtype=np.float32).copy()
        pts[:, 0] = (pts[:, 0] + offset_x) * float(pixels_per_cm)
        pts[:, 1] = (pts[:, 1] + offset_y) * float(pixels_per_cm)
        translated[name] = np.round(pts).astype(np.int32)

    mask = np.zeros((height_px, width_px), dtype=np.uint8)
    cv2.fillPoly(mask, list(translated.values()), 255)
    return mask, translated, (offset_x, offset_y)


def clip_polygon_with_half_plane(
    polygon: Sequence[Sequence[float]],
    axis_unit: Sequence[float],
    threshold: float,
    origin: Sequence[float],
) -> np.ndarray:
    axis = normalize_axis(axis_unit)
    origin_pt = np.asarray(origin, dtype=np.float32).reshape(2)
    pts = [np.asarray(p, dtype=np.float32).reshape(2) for p in sort_polygon_points(polygon)]
    if not pts:
        return np.empty((0, 2), dtype=np.float32)

    def signed_distance(pt: np.ndarray) -> float:
        return float(np.dot(pt - origin_pt, axis) - threshold)

    output: List[np.ndarray] = []
    prev = pts[-1]
    prev_dist = signed_distance(prev)
    prev_inside = prev_dist >= 0.0
    for curr in pts:
        curr_dist = signed_distance(curr)
        curr_inside = curr_dist >= 0.0
        if curr_inside != prev_inside:
            denom = prev_dist - curr_dist
            if abs(denom) > 1e-6:
                t = prev_dist / denom
                inter = prev + (curr - prev) * float(t)
                output.append(inter.astype(np.float32))
        if curr_inside:
            output.append(curr.astype(np.float32))
        prev = curr
        prev_dist = curr_dist
        prev_inside = curr_inside

    if len(output) < 3:
        return np.empty((0, 2), dtype=np.float32)
    return sort_polygon_points(output)


def build_entry_polygon(
    arm_polygon: Sequence[Sequence[float]],
    center_point: Sequence[float],
    entry_depth_ratio: float,
) -> np.ndarray:
    arm_poly = sort_polygon_points(arm_polygon)
    center_pt = np.asarray(center_point, dtype=np.float32).reshape(2)
    arm_center = polygon_centroid(arm_poly)
    axis = normalize_axis(arm_center - center_pt)
    projections = [float(np.dot(point - center_pt, axis)) for point in arm_poly]
    near_proj = min(projections)
    far_proj = max(projections)
    arm_span = max(far_proj - near_proj, 1e-3)
    threshold = near_proj + arm_span * float(np.clip(entry_depth_ratio, 0.05, 0.95))
    clipped = clip_polygon_with_half_plane(arm_poly, axis, threshold, center_pt)
    if len(clipped) >= 3:
        return clipped.astype(np.float32)
    return arm_poly.astype(np.float32)


def build_union_mask(
    frame_shape: Tuple[int, ...],
    zone_polygons: Dict[str, Sequence[Sequence[float]]],
) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    polys = [sort_polygon_points(poly).astype(np.int32) for poly in zone_polygons.values()]
    cv2.fillPoly(mask, polys, 255)
    min_x, min_y, max_x, max_y = polygons_bbox(polys)
    x = max(0, min_x)
    y = max(0, min_y)
    w = max(1, max_x - min_x + 1)
    h = max(1, max_y - min_y + 1)
    return mask, (x, y, w, h), mask[y : y + h, x : x + w]


def estimate_cm_per_px(samples_cm_per_px: Sequence[float]) -> float:
    values = [float(v) for v in samples_cm_per_px if v and np.isfinite(v) and float(v) > 0]
    if not values:
        return 1.0
    return float(np.median(np.asarray(values, dtype=np.float32)))


def format_seconds(seconds: float) -> float:
    return round(float(seconds), 3)


def format_metric(value: float) -> float:
    return round(float(value), 3)
