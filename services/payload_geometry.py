from __future__ import annotations

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
