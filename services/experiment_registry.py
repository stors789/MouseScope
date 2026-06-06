from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Callable


def _points_count(points) -> int:
    return len(points or [])


def _roi_points_complete(roi: dict[str, Any] | None, min_points: int = 4) -> bool:
    return isinstance(roi, dict) and _points_count(roi.get("points")) >= int(min_points)


def _mask_valid(roi: dict[str, Any] | None) -> bool:
    if not isinstance(roi, dict):
        return True
    mask_points = roi.get("detection_points") or []
    return not mask_points or len(mask_points) >= 3


def _oft_video_complete(config: dict[str, Any] | None) -> tuple[bool, list[str]]:
    rois = (config or {}).get("rois") or []
    if not rois:
        return False, ["缺少 ROI 配置"]
    for idx, roi in enumerate(rois, start=1):
        if not _roi_points_complete(roi):
            return False, [f"ROI {idx} 缺少 4 个 arena 角点"]
        if not _mask_valid(roi):
            return False, [f"ROI {idx} 的检测遮罩少于 3 个点"]
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
            return False, [f"ROI {idx} 的检测遮罩少于 3 个点"]
    return True, []


def _tst_video_complete(config: dict[str, Any] | None) -> tuple[bool, list[str]]:
    rois = (config or {}).get("rois") or []
    if not rois:
        return False, ["缺少 ROI 配置"]
    if not config or not config.get("pixel_to_mm"):
        return False, ["缺少体长标定"]
    for idx, roi in enumerate(rois, start=1):
        if not _roi_points_complete(roi, min_points=3):
            return False, [f"ROI {idx} 缺少多边形点（至少 3 个）"]
    return True, []


def _fst_video_complete(config: dict[str, Any] | None) -> tuple[bool, list[str]]:
    rois = (config or {}).get("rois") or []
    if not rois:
        return False, ["缺少圆筒 ROI 配置"]
    for idx, roi in enumerate(rois, start=1):
        if not isinstance(roi, dict) or _points_count(roi.get("points")) < 3:
            return False, [f"ROI {idx} 缺少圆筒多边形点（至少 3 个）"]
        diameter_points = (roi or {}).get("diameter_points") or []
        if len(diameter_points) != 2:
            return False, [f"ROI {idx} 缺少圆筒直径线"]
        try:
            diameter_cm = float((roi or {}).get("cylinder_diameter_cm") or (config or {}).get("cylinder_diameter_cm") or 0)
        except Exception:
            diameter_cm = 0.0
        if diameter_cm <= 0:
            return False, [f"ROI {idx} 缺少圆筒实际直径"]
        if not _mask_valid(roi):
            return False, [f"ROI {idx} 的检测遮罩少于 3 个点"]
    return True, []


def _water_maze_boundary_points(roi: dict[str, Any] | None):
    if not isinstance(roi, dict):
        return []
    pool = roi.get("pool") if isinstance(roi.get("pool"), dict) else {}
    return (
        roi.get("pool_boundary_points")
        or roi.get("boundary_points")
        or roi.get("points")
        or pool.get("boundary_points")
        or pool.get("pool_boundary_points")
        or pool.get("points")
        or []
    )


def _water_maze_has_circle_pool(roi: dict[str, Any] | None) -> bool:
    if not isinstance(roi, dict):
        return False
    pool = roi.get("pool") if isinstance(roi.get("pool"), dict) else {}
    center = pool.get("center") or roi.get("pool_center")
    radius = pool.get("radius_px") or pool.get("radius") or roi.get("pool_radius_px")
    try:
        return center is not None and float(radius) > 0
    except Exception:
        return False


def _water_maze_has_ellipse_pool(roi: dict[str, Any] | None) -> bool:
    if not isinstance(roi, dict):
        return False
    pool = roi.get("pool") if isinstance(roi.get("pool"), dict) else {}
    center = pool.get("ellipse_center") or pool.get("center") or roi.get("pool_center")
    rx = pool.get("radius_x_px") or pool.get("rx")
    ry = pool.get("radius_y_px") or pool.get("ry")
    axes = pool.get("ellipse_axes_px") or pool.get("axes_px") or pool.get("axes")
    try:
        if isinstance(axes, dict):
            rx = rx or axes.get("radius_x_px") or axes.get("rx") or (float(axes.get("width") or axes.get("w") or 0) * 0.5)
            ry = ry or axes.get("radius_y_px") or axes.get("ry") or (float(axes.get("height") or axes.get("h") or 0) * 0.5)
        elif isinstance(axes, (list, tuple)) and len(axes) >= 2:
            rx = rx or float(axes[0]) * 0.5
            ry = ry or float(axes[1]) * 0.5
        return center is not None and float(rx) > 0 and float(ry) > 0
    except Exception:
        return False


def _water_maze_has_platform(roi: dict[str, Any] | None) -> bool:
    if not isinstance(roi, dict):
        return False
    platform = roi.get("platform") if isinstance(roi.get("platform"), dict) else {}
    if roi.get("platform_point") is not None:
        return True
    if platform.get("point") is not None or platform.get("center") is not None:
        return True
    return platform.get("x") is not None and platform.get("y") is not None


def _water_maze_video_complete(config: dict[str, Any] | None) -> tuple[bool, list[str]]:
    config = config or {}
    rois = config.get("rois") or []
    if not rois:
        return False, ["Missing water maze pool ROI"]
    for idx, roi in enumerate(rois, start=1):
        if not isinstance(roi, dict):
            return False, [f"ROI {idx} must be an object"]
        boundary = _water_maze_boundary_points(roi)
        if _points_count(boundary) < 5 and not (_water_maze_has_circle_pool(roi) or _water_maze_has_ellipse_pool(roi)):
            return False, [f"ROI {idx} needs a pool ellipse or at least 5 boundary points"]
        if not _water_maze_has_platform(roi):
            return False, [f"ROI {idx} needs a platform point"]
        pool = roi.get("pool") if isinstance(roi.get("pool"), dict) else {}
        try:
            diameter_cm = float(roi.get("pool_diameter_cm") or pool.get("diameter_cm") or config.get("pool_diameter_cm") or 0)
        except Exception:
            diameter_cm = 0.0
        if diameter_cm <= 0:
            return False, [f"ROI {idx} needs a real pool diameter"]
        if not _mask_valid(roi):
            return False, [f"ROI {idx} detection mask needs at least 3 points"]
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
            return False, [f"ROI {idx} 的{label_prefix}检测遮罩少于 3 个点"]
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


def _polygon_complete(zone_points, min_points: int = 3) -> bool:
    return isinstance(zone_points, list) and len(zone_points) >= int(min_points)


def _truthy(value, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off", "disabled", "hidden", "invisible"}:
            return False
        if normalized in {"1", "true", "yes", "on", "enabled", "visible"}:
            return True
    return bool(value)


def _light_dark_video_complete(config: dict[str, Any] | None) -> tuple[bool, list[str]]:
    rois = (config or {}).get("rois") or []
    dark_box_visible = _truthy((config or {}).get("dark_box_visible"), True)
    if not rois:
        return False, ["缺少明暗箱 ROI 配置"]
    for idx, roi in enumerate(rois, start=1):
        zones = (roi or {}).get("zones") or {}
        if not _polygon_complete(zones.get("light")):
            return False, [f"ROI {idx} 缺少明室区域 light（至少 3 个点）"]
        if dark_box_visible and not _polygon_complete(zones.get("dark")):
            return False, [f"ROI {idx} 缺少暗室区域 dark（至少 3 个点）"]
        if zones.get("doorway") and not _polygon_complete(zones.get("doorway")):
            return False, [f"ROI {idx} 的门区 doorway 少于 3 个点"]
        if not _mask_valid(roi):
            return False, [f"ROI {idx} 的明暗箱检测遮罩少于 3 个点"]
    return True, []


def _segment_times_valid(segments) -> bool:
    if not isinstance(segments, list):
        return False
    previous_end = None
    for segment in sorted((item for item in segments if isinstance(item, dict)), key=lambda item: float(item.get("start_s") or 0.0)):
        try:
            start = float(segment.get("start_s"))
            end = float(segment.get("end_s"))
        except Exception:
            return False
        if end <= start:
            return False
        if previous_end is not None and start < previous_end:
            return False
        previous_end = end
    return True


def _three_chamber_video_complete(config: dict[str, Any] | None) -> tuple[bool, list[str]]:
    config = config or {}
    analysis_mode = str(config.get("analysis_mode") or "single_phase").strip().lower()
    rois = config.get("rois") or []
    if not rois:
        return False, ["缺少三箱 ROI 配置"]
    if analysis_mode == "full_protocol":
        segments = config.get("protocol_segments") or []
        if not _segment_times_valid(segments):
            return False, ["完整流程时间轴缺失、重叠或起止时间无效"]
        if not any((item or {}).get("analyze", (item or {}).get("phase") != "ignore") and (item or {}).get("phase") != "ignore" for item in segments if isinstance(item, dict)):
            return False, ["完整流程至少需要一个待分析阶段"]
    else:
        phase = str(config.get("selected_phase") or config.get("phase") or "").strip().lower()
        if phase not in {"habituation_center", "habituation_all", "sociability", "social_novelty"}:
            return False, ["单阶段模式需要选择有效阶段"]
    for idx, roi in enumerate(rois, start=1):
        zones = (roi or {}).get("zones") or {}
        for key in ("left", "center", "right"):
            if not _polygon_complete(zones.get(key)):
                return False, [f"ROI {idx} 缺少区域 {key}（至少 3 个点）"]
        objects = (roi or {}).get("objects") or []
        sides = {
            str((obj or {}).get("side") or "").strip().lower()
            for obj in objects
            if isinstance(obj, dict) and ((obj.get("point") is not None) or (obj.get("x") is not None and obj.get("y") is not None))
        }
        if not {"left", "right"}.issubset(sides):
            return False, [f"ROI {idx} 缺少左/右 cup 点"]
        if not _mask_valid(roi):
            return False, [f"ROI {idx} 的三箱检测遮罩少于 3 个点"]
        if analysis_mode == "full_protocol" and roi.get("protocol_segments") and not _segment_times_valid(roi.get("protocol_segments")):
            return False, [f"ROI {idx} 的单独时间轴重叠或起止时间无效"]
    return True, []


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


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on", "enabled"}
_EXPERIMENT_FEATURE_FLAGS = {
    "three_chamber": ("MOUSESCOPE_ENABLE_3CST", "MOUSESCOPE_ENABLE_THREE_CHAMBER"),
}


def _env_flag_enabled(names: tuple[str, ...]) -> bool:
    return any(str(os.environ.get(name, "")).strip().lower() in _TRUTHY_ENV_VALUES for name in names)


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
    "forced_swim": ExperimentSpec(
        key="forced_swim",
        label="强迫游泳 Forced Swim Test",
        short_label="FST",
        manual_module="fst",
        calibrate_template="calibrate_forced_swim.html",
        monitor_template="monitor_forced_swim.html",
        validator=_fst_video_complete,
    ),
    "water_maze": ExperimentSpec(
        key="water_maze",
        label="水迷宫 Morris Water Maze",
        short_label="MWM",
        manual_module="mwm",
        calibrate_template="calibrate_water_maze.html",
        monitor_template="monitor_water_maze.html",
        validator=_water_maze_video_complete,
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
    "light_dark_box": ExperimentSpec(
        key="light_dark_box",
        label="明暗箱 Light-Dark Box",
        short_label="LDB",
        manual_module="ldb",
        calibrate_template="calibrate_light_dark.html",
        monitor_template="monitor_light_dark.html",
        validator=_light_dark_video_complete,
    ),
    "three_chamber": ExperimentSpec(
        key="three_chamber",
        label="三箱社交 Three-Chamber Social Test",
        short_label="3CST",
        manual_module="three_chamber",
        calibrate_template="calibrate_three_chamber.html",
        monitor_template="monitor_three_chamber.html",
        validator=_three_chamber_video_complete,
    ),
}


def is_experiment_enabled(experiment_type: str | None) -> bool:
    value = str(experiment_type or "").strip().lower()
    if value not in EXPERIMENT_SPECS:
        return False
    flags = _EXPERIMENT_FEATURE_FLAGS.get(value)
    return not flags or _env_flag_enabled(flags)


def is_experiment_release_hidden(experiment_type: str | None) -> bool:
    value = str(experiment_type or "").strip().lower()
    return value in EXPERIMENT_SPECS and not is_experiment_enabled(value)


def is_three_chamber_enabled() -> bool:
    return is_experiment_enabled("three_chamber")


def normalize_experiment_type(raw_value: str | None, include_unreleased: bool = False) -> str:
    value = str(raw_value or "").strip().lower()
    if value not in EXPERIMENT_SPECS:
        return "tst"
    if include_unreleased or is_experiment_enabled(value):
        return value
    return "tst"


def get_experiment_spec(experiment_type: str | None, include_unreleased: bool = True) -> ExperimentSpec:
    return EXPERIMENT_SPECS[normalize_experiment_type(experiment_type, include_unreleased=include_unreleased)]


def get_manual_module(experiment_type: str | None) -> str:
    return get_experiment_spec(experiment_type).manual_module


def list_experiments(include_unreleased: bool = False) -> list[ExperimentSpec]:
    specs = list(EXPERIMENT_SPECS.values())
    if include_unreleased:
        return specs
    return [spec for spec in specs if is_experiment_enabled(spec.key)]


def extract_per_video_configs(draft_payload: dict[str, Any] | None, total_videos: int) -> list[dict[str, Any]]:
    payload = draft_payload or {}
    raw = payload.get("per_video_configs") if isinstance(payload, dict) else None
    configs = copy.deepcopy(raw) if isinstance(raw, list) else []
    while len(configs) < total_videos:
        configs.append({})
    return configs[:total_videos]


def _project_subject_ids(project_context: dict[str, Any] | None) -> set[str] | None:
    if not isinstance(project_context, dict) or not project_context.get("project_id"):
        return None
    subjects = project_context.get("subject_candidates") or []
    return {
        str(subject.get("subject_id") or "").strip()
        for subject in subjects
        if isinstance(subject, dict) and str(subject.get("subject_id") or "").strip()
    }


def _project_subject_binding_missing(config: dict[str, Any] | None, valid_subject_ids: set[str] | None) -> list[str]:
    if valid_subject_ids is None:
        return []
    rois = (config or {}).get("rois") or []
    if not rois:
        return []
    if not valid_subject_ids:
        return ["项目中没有可绑定的小鼠"]
    missing: list[str] = []
    for idx, roi in enumerate(rois, start=1):
        subject_id = str((roi or {}).get("subject_id") or "").strip() if isinstance(roi, dict) else ""
        if not subject_id or subject_id not in valid_subject_ids:
            missing.append(f"ROI {idx} 未绑定项目小鼠")
    return missing


def summarize_batch_state(
    experiment_type: str | None,
    draft_payload: dict[str, Any] | None,
    video_infos: list[dict[str, Any]] | None,
    project_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    spec = get_experiment_spec(experiment_type)
    video_infos = list(video_infos or [])
    per_video_configs = extract_per_video_configs(draft_payload, len(video_infos))
    valid_subject_ids = _project_subject_ids(project_context)
    cards: list[dict[str, Any]] = []
    for idx, info in enumerate(video_infos):
        config = per_video_configs[idx] if idx < len(per_video_configs) else {}
        complete, missing_items = spec.validator(config)
        subject_missing = _project_subject_binding_missing(config, valid_subject_ids)
        if subject_missing:
            complete = False
            missing_items = list(missing_items) + subject_missing
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
