from __future__ import annotations

import math
from typing import Any

import numpy as np


def coerce_point_list(points: Any, min_points: int = 4) -> list[list[float]]:
    if not isinstance(points, list) or len(points) < int(min_points):
        raise ValueError(f"requires at least {min_points} points")
    parsed: list[list[float]] = []
    for point in points:
        if isinstance(point, dict):
            parsed.append([float(point["x"]), float(point["y"])])
        else:
            parsed.append([float(point[0]), float(point[1])])
    return parsed


def coerce_point_xy(point: Any) -> list[float]:
    if isinstance(point, dict):
        if isinstance(point.get("point"), dict):
            return coerce_point_xy(point.get("point"))
        if isinstance(point.get("center"), dict):
            return coerce_point_xy(point.get("center"))
        return [float(point["x"]), float(point["y"])]
    return [float(point[0]), float(point[1])]


def circle_boundary_points(center: Any, radius: Any, count: int = 48) -> list[list[float]]:
    cx, cy = coerce_point_xy(center)
    radius = max(1.0, float(radius))
    return [
        [
            cx + radius * math.cos(2.0 * math.pi * idx / int(count)),
            cy + radius * math.sin(2.0 * math.pi * idx / int(count)),
        ]
        for idx in range(int(count))
    ]


def ellipse_boundary_points(pool: dict[str, Any] | None, count: int = 64) -> list[list[float]]:
    pool = dict(pool or {})
    center = pool.get("ellipse_center") or pool.get("center")
    if center is None:
        return []
    cx, cy = coerce_point_xy(center)
    axes = pool.get("ellipse_axes_px") or pool.get("axes_px") or pool.get("axes")
    if isinstance(axes, dict):
        if axes.get("radius_x_px") is not None and axes.get("radius_y_px") is not None:
            rx = float(axes.get("radius_x_px"))
            ry = float(axes.get("radius_y_px"))
        else:
            rx = float(axes.get("width") or axes.get("w") or 0.0) * 0.5
            ry = float(axes.get("height") or axes.get("h") or 0.0) * 0.5
    elif axes is not None:
        rx = float(axes[0]) * 0.5
        ry = float(axes[1]) * 0.5
    else:
        rx = float(pool.get("radius_x_px") or pool.get("rx") or 0.0)
        ry = float(pool.get("radius_y_px") or pool.get("ry") or 0.0)
    if rx <= 0 or ry <= 0:
        radius = pool.get("radius_px") or pool.get("radius")
        if radius is None:
            return []
        rx = ry = float(radius)
    angle = float(pool.get("angle_deg") or pool.get("ellipse_angle_deg") or 0.0) * math.pi / 180.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return [
        [
            cx + rx * math.cos(2.0 * math.pi * idx / int(count)) * cos_a - ry * math.sin(2.0 * math.pi * idx / int(count)) * sin_a,
            cy + rx * math.cos(2.0 * math.pi * idx / int(count)) * sin_a + ry * math.sin(2.0 * math.pi * idx / int(count)) * cos_a,
        ]
        for idx in range(int(count))
    ]


def normalize_water_maze_pool_payload(pool: dict[str, Any] | None, diameter_cm: Any = None) -> dict[str, Any]:
    payload = dict(pool or {})
    if not payload:
        return payload
    if diameter_cm is not None:
        payload["diameter_cm"] = float(diameter_cm)
    if payload.get("ellipse_center") is None and payload.get("center") is not None:
        payload["ellipse_center"] = payload.get("center")
    if payload.get("ellipse_axes_px") is None and payload.get("radius_x_px") is not None and payload.get("radius_y_px") is not None:
        payload["ellipse_axes_px"] = {
            "radius_x_px": payload.get("radius_x_px"),
            "radius_y_px": payload.get("radius_y_px"),
        }
    if payload.get("ellipse_angle_deg") is None and payload.get("angle_deg") is not None:
        payload["ellipse_angle_deg"] = payload.get("angle_deg")
    return payload


def sort_polygon_points(points: Any) -> list[list[float]]:
    pts = np.asarray(points, dtype=np.float32)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)].astype(float).tolist()


def coerce_quad_points(points: Any) -> list[list[float]]:
    parsed = coerce_point_list(points, min_points=4)
    if len(parsed) != 4:
        raise ValueError("requires exactly 4 points")
    return sort_polygon_points(parsed)


def coerce_triangle_points(points: Any) -> list[list[float]]:
    parsed = coerce_point_list(points, min_points=3)
    if len(parsed) != 3:
        raise ValueError("requires exactly 3 points")
    return sort_polygon_points(parsed)


def coerce_zone_polygons(raw_zones: Any, required_keys: tuple[str, ...], experiment_name: str) -> dict[str, list[list[float]]]:
    if not isinstance(raw_zones, dict):
        raise ValueError(f"{experiment_name} ROI zones must be an object.")
    parsed: dict[str, list[list[float]]] = {}
    for key in required_keys:
        parsed[key] = coerce_quad_points(raw_zones.get(key))
    return parsed


def coerce_zone_polygons_with_schema(
    raw_zones: Any,
    zone_point_counts: dict[str, int],
    experiment_name: str,
) -> dict[str, list[list[float]]]:
    if not isinstance(raw_zones, dict):
        raise ValueError(f"{experiment_name} ROI zones must be an object.")
    parsed: dict[str, list[list[float]]] = {}
    for key, expected_count in zone_point_counts.items():
        if int(expected_count) == 3:
            try:
                parsed[key] = coerce_triangle_points(raw_zones.get(key))
            except Exception as exc:
                raise ValueError(f'Zone "{key}" requires exactly {expected_count} points.') from exc
        elif int(expected_count) == 4:
            try:
                parsed[key] = coerce_quad_points(raw_zones.get(key))
            except Exception as exc:
                raise ValueError(f'Zone "{key}" requires exactly {expected_count} points.') from exc
        else:
            parsed[key] = coerce_point_list(raw_zones.get(key), min_points=int(expected_count))
            if len(parsed[key]) != int(expected_count):
                raise ValueError(f"zone {key} requires exactly {expected_count} points")
    return parsed
