from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable


def _points_count(points) -> int:
    return len(points or [])


def _roi_points_complete(roi: dict[str, Any] | None) -> bool:
    return isinstance(roi, dict) and _points_count(roi.get("points")) >= 4


def _mask_valid(roi: dict[str, Any] | None) -> bool:
    if not isinstance(roi, dict):
        return True
    mask_points = roi.get("detection_points") or []
    return not mask_points or len(mask_points) >= 4


def _oft_video_complete(config: dict[str, Any] | None) -> tuple[bool, list[str]]:
    rois = (config or {}).get("rois") or []
    if not rois:
        return False, ["缺少 ROI 配置"]
    for idx, roi in enumerate(rois, start=1):
        if not _roi_points_complete(roi):
            return False, [f"ROI {idx} 缺少 4 个 arena 角点"]
        if not _mask_valid(roi):
            return False, [f"ROI {idx} 的检测遮罩少于 4 个点"]
    return True, []


def _nor_video_complete(config: dict[str, Any] | None) -> tuple[bool, list[str]]:
    rois = (config or {}).get("rois") or []
    if not rois:
        return False, ["缺少 ROI 配置"]
    for idx, roi in enumerate(rois, start=1):
        if not _roi_points_complete(roi):
            return False, [f"ROI {idx} 缺少 4 个 arena 角点"]
        objects = roi.get("objects") or []
        if len(objects) != 2:
            return False, [f"ROI {idx} 缺少 2 个物体点"]
        if not _mask_valid(roi):
            return False, [f"ROI {idx} 的检测遮罩少于 4 个点"]
    return True, []


def _tst_video_complete(config: dict[str, Any] | None) -> tuple[bool, list[str]]:
    rois = (config or {}).get("rois") or []
    if not rois:
        return False, ["缺少 ROI 配置"]
    if not config or not config.get("pixel_to_mm"):
        return False, ["缺少体长标定"]
    for idx, roi in enumerate(rois, start=1):
        if not _roi_points_complete(roi):
            return False, [f"ROI {idx} 缺少多边形点"]
    return True, []


def _maze_zone_complete(zone_points, expected_points: int = 4) -> bool:
    return isinstance(zone_points, list) and len(zone_points) == int(expected_points)


def _maze_video_complete(
    config: dict[str, Any] | None,
    required_keys: tuple[str, ...],
    label_prefix: str,
    zone_point_counts: dict[str, int] | None = None,
) -> tuple[bool, list[str]]:
    rois = (config or {}).get("rois") or []
    if not rois:
        return False, ["缺少 ROI 配置"]
    counts = zone_point_counts or {key: 4 for key in required_keys}
    for idx, roi in enumerate(rois, start=1):
        zones = (roi or {}).get("zones") or {}
        for key in required_keys:
            expected_points = int(counts.get(key, 4))
            if not _maze_zone_complete(zones.get(key), expected_points):
                return False, [f"ROI {idx} 缺少区域 {key}（需要 {expected_points} 个点）"]
        if not _mask_valid(roi):
            return False, [f"ROI {idx} 的{label_prefix}检测遮罩少于 4 个点"]
    return True, []


def _epm_video_complete(config: dict[str, Any] | None) -> tuple[bool, list[str]]:
    return _maze_video_complete(
        config,
        ("center", "open_1", "open_2", "closed_1", "closed_2"),
        "EPM",
    )


def _ymaze_video_complete(config: dict[str, Any] | None) -> tuple[bool, list[str]]:
    return _maze_video_complete(
        config,
        ("center", "arm_a", "arm_b", "arm_c"),
        "Y 迷宫",
        {"center": 3, "arm_a": 4, "arm_b": 4, "arm_c": 4},
    )


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    short_label: str
    manual_module: str
    calibrate_template: str
    monitor_template: str
    validator: Callable[[dict[str, Any] | None], tuple[bool, list[str]]]
    maze_module: str | None = None


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "tst": ExperimentSpec(
        key="tst",
        label="悬尾实验 Tail Suspension Test",
        short_label="TST",
        manual_module="tst",
        calibrate_template="calibrate.html",
        monitor_template="monitor.html",
        validator=_tst_video_complete,
    ),
    "open_field": ExperimentSpec(
        key="open_field",
        label="旷场实验 Open Field Test",
        short_label="OFT",
        manual_module="oft",
        calibrate_template="calibrate_openfield.html",
        monitor_template="monitor_openfield.html",
        validator=_oft_video_complete,
    ),
    "novel_object": ExperimentSpec(
        key="novel_object",
        label="新奇物体识别 Novel Object Recognition",
        short_label="NOR",
        manual_module="nor",
        calibrate_template="calibrate_nor.html",
        monitor_template="monitor_nor.html",
        validator=_nor_video_complete,
    ),
    "elevated_plus": ExperimentSpec(
        key="elevated_plus",
        label="高架十字迷宫 Elevated Plus Maze",
        short_label="EPM",
        manual_module="epm",
        calibrate_template="calibrate_maze.html",
        monitor_template="monitor_maze.html",
        maze_module="epm",
        validator=_epm_video_complete,
    ),
    "y_maze": ExperimentSpec(
        key="y_maze",
        label="Y 迷宫 Y Maze",
        short_label="Y-Maze",
        manual_module="ymaze",
        calibrate_template="calibrate_maze.html",
        monitor_template="monitor_maze.html",
        maze_module="ymaze",
        validator=_ymaze_video_complete,
    ),
}


def normalize_experiment_type(raw_value: str | None) -> str:
    value = str(raw_value or "").strip().lower()
    return value if value in EXPERIMENT_SPECS else "tst"


def get_experiment_spec(experiment_type: str | None) -> ExperimentSpec:
    return EXPERIMENT_SPECS[normalize_experiment_type(experiment_type)]


def get_manual_module(experiment_type: str | None) -> str:
    return get_experiment_spec(experiment_type).manual_module


def list_experiments() -> list[ExperimentSpec]:
    return list(EXPERIMENT_SPECS.values())


def extract_per_video_configs(draft_payload: dict[str, Any] | None, total_videos: int) -> list[dict[str, Any]]:
    payload = draft_payload or {}
    raw = payload.get("per_video_configs") if isinstance(payload, dict) else None
    configs = copy.deepcopy(raw) if isinstance(raw, list) else []
    while len(configs) < total_videos:
        configs.append({})
    return configs[:total_videos]


def summarize_batch_state(experiment_type: str | None, draft_payload: dict[str, Any] | None, video_infos: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    spec = get_experiment_spec(experiment_type)
    video_infos = list(video_infos or [])
    per_video_configs = extract_per_video_configs(draft_payload, len(video_infos))
    cards: list[dict[str, Any]] = []
    for idx, info in enumerate(video_infos):
        config = per_video_configs[idx] if idx < len(per_video_configs) else {}
        complete, missing_items = spec.validator(config)
        cards.append(
            {
                "index": idx,
                "name": info.get("original_name") or f"Video {idx + 1}",
                "video_info": {
                    "fps": info.get("fps"),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "total_frames": info.get("total_frames"),
                },
                "complete": bool(complete),
                "missing_items": missing_items,
            }
        )
    return cards
