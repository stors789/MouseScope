from __future__ import annotations

import base64
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

CORE_DIR = Path(__file__).resolve().parents[1] / "core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from core.visual_style import normalize_visual_style
from elevated_plus import ElevatedPlusTracker, render_elevated_plus_heatmap, render_elevated_plus_trackmap
from light_dark_box import LightDarkBoxTracker, render_light_dark_box_heatmap, render_light_dark_box_trackmap
from novel_object import NovelObjectTracker, render_novel_object_heatmap, render_novel_object_trackmap
from open_field import OpenFieldTracker, render_open_field_heatmap, render_open_field_trackmap
from three_chamber import ThreeChamberTracker, render_three_chamber_heatmap, render_three_chamber_trackmap
from water_maze import WaterMazeTracker, render_water_maze_heatmap, render_water_maze_trackmap
from barnes_maze import BarnesArena, BarnesTracker, render_barnes_map
from y_maze import YMazeTracker, _extend_edge_to_quad, render_y_maze_heatmap, render_y_maze_trackmap


PREVIEW_EXPERIMENTS = (
    {"key": "open_field", "label": "旷场", "short_label": "OFT"},
    {"key": "y_maze", "label": "Y 迷宫", "short_label": "Y-maze"},
    {"key": "elevated_plus", "label": "高架十字迷宫", "short_label": "EPM"},
    {"key": "light_dark", "label": "明暗箱", "short_label": "LDB"},
    {"key": "novel_object", "label": "新物体识别", "short_label": "NOR"},
    {"key": "water_maze", "label": "水迷宫", "short_label": "MWM"},
    {"key": "barnes_maze", "label": "巴恩斯迷宫", "short_label": "Barnes"},
)


def preview_experiment_payload() -> list[dict[str, str]]:
    return [dict(item) for item in PREVIEW_EXPERIMENTS]


def _sample_path(points: list[tuple[float, float]], steps: int = 20) -> list[tuple[float, float]]:
    sampled: list[tuple[float, float]] = []
    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        for step in range(steps):
            t = step / float(steps)
            sampled.append((x0 * (1.0 - t) + x1 * t, y0 * (1.0 - t) + y1 * t))
    sampled.append(points[-1])
    return sampled


def _polygon_center(poly: np.ndarray) -> tuple[float, float]:
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def _scale_polygons(polygons: dict[str, np.ndarray], scale: float = 8.0, pad: float = 80.0) -> dict[str, list[list[float]]]:
    all_points = np.concatenate([np.asarray(poly, dtype=np.float32).reshape(-1, 2) for poly in polygons.values()], axis=0)
    min_xy = np.min(all_points, axis=0)
    offset = np.array([pad, pad], dtype=np.float32) - min_xy * float(scale)
    return {
        name: (np.asarray(poly, dtype=np.float32) * float(scale) + offset).astype(float).tolist()
        for name, poly in polygons.items()
    }


def _standard_y_polygons_cm(arm_width: float = 10.0, arm_length: float = 30.0) -> dict[str, np.ndarray]:
    tri_height = arm_width * math.sqrt(3.0) / 2.0
    center = np.array(
        [
            [0.0, 0.0],
            [arm_width, 0.0],
            [arm_width * 0.5, tri_height],
        ],
        dtype=np.float32,
    )
    center_point = np.mean(center, axis=0).astype(np.float32)
    return {
        "center": center,
        "arm_a": np.array(
            [
                [0.0, -arm_length],
                [arm_width, -arm_length],
                [arm_width, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "arm_b": _extend_edge_to_quad(center[1], center[2], center_point, arm_length),
        "arm_c": _extend_edge_to_quad(center[2], center[0], center_point, arm_length),
    }


def _standard_epm_polygons_cm(center_size: float = 5.0, arm_length: float = 30.0) -> dict[str, np.ndarray]:
    c = float(center_size)
    arm = float(arm_length)
    return {
        "center": np.array([[arm, arm], [arm + c, arm], [arm + c, arm + c], [arm, arm + c]], dtype=np.float32),
        "open_1": np.array([[arm + c, arm], [arm * 2.0 + c, arm], [arm * 2.0 + c, arm + c], [arm + c, arm + c]], dtype=np.float32),
        "open_2": np.array([[0.0, arm], [arm, arm], [arm, arm + c], [0.0, arm + c]], dtype=np.float32),
        "closed_1": np.array([[arm, 0.0], [arm + c, 0.0], [arm + c, arm], [arm, arm]], dtype=np.float32),
        "closed_2": np.array([[arm, arm + c], [arm + c, arm + c], [arm + c, arm * 2.0 + c], [arm, arm * 2.0 + c]], dtype=np.float32),
    }


def _build_open_field() -> tuple[Any, Callable[..., Any], Callable[..., Any], int]:
    tracker = OpenFieldTracker(
        arena_points=[[0, 0], [420, 0], [420, 420], [0, 420]],
        arena_width_cm=40,
        arena_height_cm=40,
        center_ratio_x=0.5,
        center_ratio_y=0.5,
    )
    tracker.trajectory_cm_stats = _sample_path(
        [(4, 6), (34, 7), (33, 18), (22, 20), (12, 32), (35, 35), (7, 24), (17, 10), (29, 30)],
        steps=18,
    )
    return tracker, render_open_field_trackmap, render_open_field_heatmap, 12


def _build_novel_object() -> tuple[Any, Callable[..., Any], Callable[..., Any], int]:
    tracker = NovelObjectTracker(
        arena_points=[[0, 0], [420, 0], [420, 420], [0, 420]],
        object_points=[
            {"point": {"x": 126, "y": 145}, "role": "familiar"},
            {"point": {"x": 294, "y": 280}, "role": "novel"},
        ],
        arena_width_cm=40,
        arena_height_cm=40,
        center_ratio_x=0.5,
        center_ratio_y=0.5,
        interaction_radius_cm=4,
    )
    tracker.trajectory_cm_stats = _sample_path(
        [(6, 8), (11, 14), (15, 13), (25, 20), (30, 27), (27, 33), (16, 30), (11, 16), (30, 10)],
        steps=18,
    )
    return tracker, render_novel_object_trackmap, render_novel_object_heatmap, 12


def _build_y_maze() -> tuple[Any, Callable[..., Any], Callable[..., Any], int]:
    template = _standard_y_polygons_cm()
    tracker = YMazeTracker(_scale_polygons(template), arm_length_cm=30, arm_width_cm=10)
    centers = {name: _polygon_center(poly) for name, poly in tracker.template_polygons_cm.items()}
    tracker.trajectory_template_cm_stats = _sample_path(
        [
            centers["arm_a"],
            centers["center"],
            centers["arm_b"],
            centers["center"],
            centers["arm_c"],
            centers["center"],
            (centers["arm_a"][0], centers["arm_a"][1] - 8.0),
        ],
        steps=18,
    )
    return tracker, render_y_maze_trackmap, render_y_maze_heatmap, 12


def _build_elevated_plus() -> tuple[Any, Callable[..., Any], Callable[..., Any], int]:
    template = _standard_epm_polygons_cm()
    tracker = ElevatedPlusTracker(
        _scale_polygons(template),
        center_size_cm=5,
        open_arm_length_cm=30,
        closed_arm_length_cm=30,
    )
    centers = {name: _polygon_center(poly) for name, poly in tracker.template_polygons_cm.items()}
    tracker.trajectory_template_cm_stats = _sample_path(
        [
            centers["open_2"],
            centers["center"],
            centers["closed_1"],
            centers["center"],
            centers["open_1"],
            centers["center"],
            centers["closed_2"],
            centers["center"],
        ],
        steps=16,
    )
    return tracker, render_elevated_plus_trackmap, render_elevated_plus_heatmap, 12


def _build_light_dark() -> tuple[Any, Callable[..., Any], Callable[..., Any], int]:
    tracker = LightDarkBoxTracker(
        zone_polygons={
            "dark": [[0, 0], [200, 0], [200, 400], [0, 400]],
            "light": [[200, 0], [400, 0], [400, 400], [200, 400]],
        },
        arena_width_cm=40,
        arena_height_cm=40,
        light_box_length_cm=20,
        light_box_width_cm=40,
        dark_box_length_cm=20,
        dark_box_width_cm=40,
        shared_edge_length_cm=40,
        door_open_delay_s=0,
    )
    tracker.template_width_cm = 40.0
    tracker.template_height_cm = 40.0
    tracker.arena_width_cm = 40.0
    tracker.arena_height_cm = 40.0
    tracker.template_polygons_cm = {
        "dark": np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 40.0], [0.0, 40.0]], dtype=np.float32),
        "light": np.array([[20.0, 0.0], [40.0, 0.0], [40.0, 40.0], [20.0, 40.0]], dtype=np.float32),
    }
    tracker.trajectory_template_cm_stats = _sample_path(
        [(5, 8), (16, 17), (24, 20), (35, 10), (30, 31), (18, 34), (8, 24), (22, 15), (34, 30)],
        steps=18,
    )
    return tracker, render_light_dark_box_trackmap, render_light_dark_box_heatmap, 12


def _build_three_chamber() -> tuple[Any, Callable[..., Any], Callable[..., Any], int]:
    tracker = ThreeChamberTracker(
        zone_polygons={
            "left": [[0, 0], [190, 0], [190, 300], [0, 300]],
            "center": [[190, 0], [310, 0], [310, 300], [190, 300]],
            "right": [[310, 0], [500, 0], [500, 300], [310, 300]],
            "left_door": [[185, 120], [195, 120], [195, 180], [185, 180]],
            "right_door": [[305, 120], [315, 120], [315, 180], [305, 180]],
        },
        objects=[
            {"side": "left", "point": {"x": 70, "y": 150}, "role": "social_stranger_1"},
            {"side": "right", "point": {"x": 430, "y": 150}, "role": "empty_object"},
        ],
        arena_width_cm=60,
        arena_height_cm=36,
    )
    tracker.trajectory_px_stats = _sample_path(
        [(245, 150), (150, 80), (70, 145), (200, 215), (275, 155), (390, 210), (430, 150), (320, 80), (245, 150)],
        steps=14,
    )
    return tracker, render_three_chamber_trackmap, render_three_chamber_heatmap, 12


def _build_water_maze() -> tuple[Any, Callable[..., Any], Callable[..., Any], int]:
    tracker = WaterMazeTracker(
        pool={"center": [300, 300], "radius_px": 250, "diameter_cm": 120},
        platform_point={"x": 382, "y": 218},
        pool_diameter_cm=120,
        platform_radius_cm=5,
        start_position="S",
        thigmotaxis_band_cm=10,
    )
    tracker.trajectory_pool_cm_stats = _sample_path(
        [(-5, 52), (-36, 32), (-45, -4), (-18, -28), (20, -30), (35, -16), (18, 4), (12, -18)],
        steps=18,
    )
    return tracker, render_water_maze_trackmap, render_water_maze_heatmap, 6


def _build_barnes():
    angles = np.linspace(0, 2 * math.pi, 32, endpoint=False)
    points = [[300 + 250 * math.cos(t), 300 + 210 * math.sin(t)] for t in angles]
    holes = [dict(id=str(i+1), point=[300+210*math.cos(t),300+176*math.sin(t)])
             for i,t in enumerate(angles[::4])]
    tracker = BarnesTracker(BarnesArena(points, holes, '1'), tracking_max_jump_cm=0)
    for i,(x,y) in enumerate(_sample_path([(-5,10),(-25,-8),(-18,-27),(14,-22),(30,2)],steps=20)):
        px,py = tracker.arena.to_image((x,y))
        tracker.scorer.update(i*.1,.1,dict(keypoints=dict(body_center=dict(x=px,y=py,confidence=1,visible=True))))
    track = lambda t,p,**kw: render_barnes_map(t,p,style=kw.get('style'))
    heat = lambda t,p,**kw: render_barnes_map(t,p,heatmap=True,style=kw.get('style'))
    return tracker,track,heat,1


BUILDERS: dict[str, Callable[[], tuple[Any, Callable[..., Any], Callable[..., Any], int]]] = {
    "barnes_maze": _build_barnes,
    "open_field": _build_open_field,
    "novel_object": _build_novel_object,
    "y_maze": _build_y_maze,
    "elevated_plus": _build_elevated_plus,
    "light_dark": _build_light_dark,
    "three_chamber": _build_three_chamber,
    "water_maze": _build_water_maze,
}


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_visual_style_preview(style: Any | None, experiment: str | None = None) -> dict[str, Any]:
    experiment_key = str(experiment or "open_field").strip().lower()
    if experiment_key not in BUILDERS:
        experiment_key = "open_field"
    normalized = normalize_visual_style(style)

    tracker, track_renderer, heat_renderer, pixels_per_cm = BUILDERS[experiment_key]()
    with tempfile.TemporaryDirectory(prefix="mousescope_style_preview_") as tmp:
        tmpdir = Path(tmp)
        track_path = tmpdir / "trackmap.png"
        heat_path = tmpdir / "heatmap.png"
        if not track_renderer(tracker, str(track_path), pixels_per_cm=pixels_per_cm, style=normalized):
            raise RuntimeError("Track map preview could not be rendered.")
        if not heat_renderer(tracker, str(heat_path), pixels_per_cm=pixels_per_cm, style=normalized):
            raise RuntimeError("Heatmap preview could not be rendered.")
        return {
            "experiment": experiment_key,
            "trackmap": _image_data_url(track_path),
            "heatmap": _image_data_url(heat_path),
        }
