from __future__ import annotations

import copy
from typing import Any


def _base_session_fields(sess: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_dir": sess.get("output_dir"),
        "start_sec": float(sess["start_seconds"]) if sess.get("start_seconds") else None,
        "end_sec": float(sess["end_seconds"]) if sess.get("end_seconds") else None,
        "fps_override": sess.get("fps_override"),
    }


def _iter_video_configs(sess: dict[str, Any], data: dict[str, Any], builder) -> list[dict[str, Any]]:
    per_video_configs = []
    raw_per_video_configs = data.get("per_video_configs", []) if isinstance(data, dict) else []
    for idx, _ in enumerate(sess.get("video_infos", [])):
        raw_cfg = raw_per_video_configs[idx] if idx < len(raw_per_video_configs) and isinstance(raw_per_video_configs[idx], dict) else {}
        per_video_configs.append(builder(raw_cfg))
    return per_video_configs


def build_open_field_config(sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    default_bg_method = data.get("bg_method", "threshold")
    default_bg_type = data.get("bg_type", "light")
    default_gray_threshold = int(data.get("gray_threshold", 127))
    default_erosion_kernel = int(data.get("erosion_kernel", 3))
    default_erosion_iter = int(data.get("erosion_iter", 1))
    default_dilation_kernel = int(data.get("dilation_kernel", 5))
    default_dilation_iter = int(data.get("dilation_iter", 2))
    default_min_area = int(data.get("min_area", 120))
    default_arena_side_cm = float(data.get("arena_side_cm", data.get("arena_width_cm", 50.0)))
    default_center_ratio_x = float(data.get("center_ratio_x", 0.5))
    default_center_ratio_y = float(data.get("center_ratio_y", 0.5))
    default_wall_margin_cm = float(data.get("wall_margin_cm", 8.0))
    default_movement_threshold_cm_s = float(data.get("movement_threshold_cm_s", 2.0))
    default_bin_seconds = float(data.get("bin_seconds", 300.0))

    per_video_configs = _iter_video_configs(
        sess,
        data,
        lambda raw_cfg: {
            "bg_method": raw_cfg.get("bg_method", default_bg_method),
            "gray_threshold": int(raw_cfg.get("gray_threshold", default_gray_threshold)),
            "invert": str(raw_cfg.get("bg_type", default_bg_type)).lower() == "dark",
            "erosion_kernel": int(raw_cfg.get("erosion_kernel", default_erosion_kernel)),
            "erosion_iter": int(raw_cfg.get("erosion_iter", default_erosion_iter)),
            "dilation_kernel": int(raw_cfg.get("dilation_kernel", default_dilation_kernel)),
            "dilation_iter": int(raw_cfg.get("dilation_iter", default_dilation_iter)),
            "min_area": int(raw_cfg.get("min_area", default_min_area)),
            "arena_side_cm": float(raw_cfg.get("arena_side_cm", default_arena_side_cm)),
            "rois": copy.deepcopy(raw_cfg.get("rois", data.get("rois", []))),
            "center_ratio_x": float(raw_cfg.get("center_ratio_x", default_center_ratio_x)),
            "center_ratio_y": float(raw_cfg.get("center_ratio_y", default_center_ratio_y)),
            "wall_margin_cm": float(raw_cfg.get("wall_margin_cm", default_wall_margin_cm)),
            "movement_threshold_cm_s": float(raw_cfg.get("movement_threshold_cm_s", default_movement_threshold_cm_s)),
            "bin_seconds": float(raw_cfg.get("bin_seconds", default_bin_seconds)),
        },
    )

    return {
        "bg_method": default_bg_method,
        "gray_threshold": default_gray_threshold,
        "invert": str(default_bg_type).lower() == "dark",
        "erosion_kernel": default_erosion_kernel,
        "erosion_iter": default_erosion_iter,
        "dilation_kernel": default_dilation_kernel,
        "dilation_iter": default_dilation_iter,
        "min_area": default_min_area,
        "max_area": int(data.get("max_area", 50000)),
        "rois": data.get("rois", []),
        "arena_points": data.get("arena_points", []),
        "arena_side_cm": default_arena_side_cm,
        "per_video_configs": per_video_configs,
        "center_ratio_x": default_center_ratio_x,
        "center_ratio_y": default_center_ratio_y,
        "wall_margin_cm": default_wall_margin_cm,
        "movement_threshold_cm_s": default_movement_threshold_cm_s,
        "bin_seconds": default_bin_seconds,
        "generate_video": data.get("generate_video", True),
        **_base_session_fields(sess),
    }


def build_novel_object_config(sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    default_phase = data.get("phase", "recognition")
    default_bg_method = data.get("bg_method", "threshold")
    default_bg_type = data.get("bg_type", "light")
    default_gray_threshold = int(data.get("gray_threshold", 127))
    default_erosion_kernel = int(data.get("erosion_kernel", 3))
    default_erosion_iter = int(data.get("erosion_iter", 1))
    default_dilation_kernel = int(data.get("dilation_kernel", 5))
    default_dilation_iter = int(data.get("dilation_iter", 2))
    default_min_area = int(data.get("min_area", 120))
    default_arena_width_cm = float(data.get("arena_width_cm", data.get("arena_side_cm", 50.0)))
    default_arena_height_cm = float(data.get("arena_height_cm", data.get("arena_side_cm", 50.0)))
    default_center_ratio_x = float(data.get("center_ratio_x", 0.5))
    default_center_ratio_y = float(data.get("center_ratio_y", 0.5))
    default_interaction_radius_cm = float(data.get("interaction_radius_cm", 4.0))
    default_exploration_max_speed_cm_s = float(data.get("exploration_max_speed_cm_s", 12.0))
    default_use_head_direction = bool(data.get("use_head_direction", True))
    default_head_angle_threshold_deg = float(data.get("head_angle_threshold_deg", 50.0))
    default_head_confidence_threshold = float(data.get("head_confidence_threshold", 0.35))
    default_movement_threshold_cm_s = float(data.get("movement_threshold_cm_s", 2.0))
    default_min_valid_exploration_s = float(data.get("min_valid_exploration_s", 20.0))
    default_bin_seconds = float(data.get("bin_seconds", 300.0))

    per_video_configs = _iter_video_configs(
        sess,
        data,
        lambda raw_cfg: {
            "bg_method": raw_cfg.get("bg_method", default_bg_method),
            "gray_threshold": int(raw_cfg.get("gray_threshold", default_gray_threshold)),
            "invert": str(raw_cfg.get("bg_type", default_bg_type)).lower() == "dark",
            "erosion_kernel": int(raw_cfg.get("erosion_kernel", default_erosion_kernel)),
            "erosion_iter": int(raw_cfg.get("erosion_iter", default_erosion_iter)),
            "dilation_kernel": int(raw_cfg.get("dilation_kernel", default_dilation_kernel)),
            "dilation_iter": int(raw_cfg.get("dilation_iter", default_dilation_iter)),
            "min_area": int(raw_cfg.get("min_area", default_min_area)),
            "phase": raw_cfg.get("phase", default_phase),
            "arena_width_cm": float(raw_cfg.get("arena_width_cm", default_arena_width_cm)),
            "arena_height_cm": float(raw_cfg.get("arena_height_cm", default_arena_height_cm)),
            "rois": copy.deepcopy(raw_cfg.get("rois", data.get("rois", []))),
            "center_ratio_x": float(raw_cfg.get("center_ratio_x", default_center_ratio_x)),
            "center_ratio_y": float(raw_cfg.get("center_ratio_y", default_center_ratio_y)),
            "interaction_radius_cm": float(raw_cfg.get("interaction_radius_cm", default_interaction_radius_cm)),
            "exploration_max_speed_cm_s": float(raw_cfg.get("exploration_max_speed_cm_s", default_exploration_max_speed_cm_s)),
            "use_head_direction": bool(raw_cfg.get("use_head_direction", default_use_head_direction)),
            "head_angle_threshold_deg": float(raw_cfg.get("head_angle_threshold_deg", default_head_angle_threshold_deg)),
            "head_confidence_threshold": float(raw_cfg.get("head_confidence_threshold", default_head_confidence_threshold)),
            "movement_threshold_cm_s": float(raw_cfg.get("movement_threshold_cm_s", default_movement_threshold_cm_s)),
            "min_valid_exploration_s": float(raw_cfg.get("min_valid_exploration_s", default_min_valid_exploration_s)),
            "bin_seconds": float(raw_cfg.get("bin_seconds", default_bin_seconds)),
        },
    )

    return {
        "bg_method": default_bg_method,
        "gray_threshold": default_gray_threshold,
        "invert": str(default_bg_type).lower() == "dark",
        "erosion_kernel": default_erosion_kernel,
        "erosion_iter": default_erosion_iter,
        "dilation_kernel": default_dilation_kernel,
        "dilation_iter": default_dilation_iter,
        "min_area": default_min_area,
        "max_area": int(data.get("max_area", 50000)),
        "phase": default_phase,
        "rois": data.get("rois", []),
        "per_video_configs": per_video_configs,
        "arena_width_cm": default_arena_width_cm,
        "arena_height_cm": default_arena_height_cm,
        "center_ratio_x": default_center_ratio_x,
        "center_ratio_y": default_center_ratio_y,
        "interaction_radius_cm": default_interaction_radius_cm,
        "exploration_max_speed_cm_s": default_exploration_max_speed_cm_s,
        "use_head_direction": default_use_head_direction,
        "head_angle_threshold_deg": default_head_angle_threshold_deg,
        "head_confidence_threshold": default_head_confidence_threshold,
        "movement_threshold_cm_s": default_movement_threshold_cm_s,
        "min_valid_exploration_s": default_min_valid_exploration_s,
        "bin_seconds": default_bin_seconds,
        "generate_video": data.get("generate_video", True),
        **_base_session_fields(sess),
    }


def build_elevated_plus_config(sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    default_bg_method = data.get("bg_method", "threshold")
    default_bg_type = data.get("bg_type", "light")
    default_gray_threshold = int(data.get("gray_threshold", 127))
    default_erosion_kernel = int(data.get("erosion_kernel", 3))
    default_erosion_iter = int(data.get("erosion_iter", 1))
    default_dilation_kernel = int(data.get("dilation_kernel", 5))
    default_dilation_iter = int(data.get("dilation_iter", 2))
    default_min_area = int(data.get("min_area", 120))
    default_center_size_cm = float(data.get("center_size_cm", 5.0))
    default_open_arm_length_cm = float(data.get("open_arm_length_cm", 30.0))
    default_closed_arm_length_cm = float(data.get("closed_arm_length_cm", 30.0))
    default_open_axis = data.get("open_axis", "horizontal")
    default_movement_threshold_cm_s = float(data.get("movement_threshold_cm_s", 2.0))
    default_entry_depth_ratio = float(data.get("entry_depth_ratio", 0.2))
    default_bin_seconds = float(data.get("bin_seconds", 60.0))

    per_video_configs = _iter_video_configs(
        sess,
        data,
        lambda raw_cfg: {
            "bg_method": raw_cfg.get("bg_method", default_bg_method),
            "gray_threshold": int(raw_cfg.get("gray_threshold", default_gray_threshold)),
            "invert": str(raw_cfg.get("bg_type", default_bg_type)).lower() == "dark",
            "erosion_kernel": int(raw_cfg.get("erosion_kernel", default_erosion_kernel)),
            "erosion_iter": int(raw_cfg.get("erosion_iter", default_erosion_iter)),
            "dilation_kernel": int(raw_cfg.get("dilation_kernel", default_dilation_kernel)),
            "dilation_iter": int(raw_cfg.get("dilation_iter", default_dilation_iter)),
            "min_area": int(raw_cfg.get("min_area", default_min_area)),
            "center_size_cm": float(raw_cfg.get("center_size_cm", default_center_size_cm)),
            "open_arm_length_cm": float(raw_cfg.get("open_arm_length_cm", default_open_arm_length_cm)),
            "closed_arm_length_cm": float(raw_cfg.get("closed_arm_length_cm", default_closed_arm_length_cm)),
            "open_axis": raw_cfg.get("open_axis", default_open_axis),
            "rois": copy.deepcopy(raw_cfg.get("rois", data.get("rois", []))),
            "movement_threshold_cm_s": float(raw_cfg.get("movement_threshold_cm_s", default_movement_threshold_cm_s)),
            "entry_depth_ratio": float(raw_cfg.get("entry_depth_ratio", default_entry_depth_ratio)),
            "bin_seconds": float(raw_cfg.get("bin_seconds", default_bin_seconds)),
        },
    )

    return {
        "bg_method": default_bg_method,
        "gray_threshold": default_gray_threshold,
        "invert": str(default_bg_type).lower() == "dark",
        "erosion_kernel": default_erosion_kernel,
        "erosion_iter": default_erosion_iter,
        "dilation_kernel": default_dilation_kernel,
        "dilation_iter": default_dilation_iter,
        "min_area": default_min_area,
        "max_area": int(data.get("max_area", 50000)),
        "rois": data.get("rois", []),
        "per_video_configs": per_video_configs,
        "center_size_cm": default_center_size_cm,
        "open_arm_length_cm": default_open_arm_length_cm,
        "closed_arm_length_cm": default_closed_arm_length_cm,
        "open_axis": default_open_axis,
        "movement_threshold_cm_s": default_movement_threshold_cm_s,
        "entry_depth_ratio": default_entry_depth_ratio,
        "bin_seconds": default_bin_seconds,
        "generate_video": data.get("generate_video", True),
        **_base_session_fields(sess),
    }


def build_y_maze_config(sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    default_bg_method = data.get("bg_method", "threshold")
    default_bg_type = data.get("bg_type", "light")
    default_gray_threshold = int(data.get("gray_threshold", 127))
    default_erosion_kernel = int(data.get("erosion_kernel", 3))
    default_erosion_iter = int(data.get("erosion_iter", 1))
    default_dilation_kernel = int(data.get("dilation_kernel", 5))
    default_dilation_iter = int(data.get("dilation_iter", 2))
    default_min_area = int(data.get("min_area", 120))
    default_arm_length_cm = float(data.get("arm_length_cm", 30.0))
    default_arm_width_cm = float(data.get("arm_width_cm", 10.0))
    default_movement_threshold_cm_s = float(data.get("movement_threshold_cm_s", 2.0))
    default_entry_depth_ratio = float(data.get("entry_depth_ratio", 0.2))
    default_bin_seconds = float(data.get("bin_seconds", 60.0))

    per_video_configs = _iter_video_configs(
        sess,
        data,
        lambda raw_cfg: {
            "bg_method": raw_cfg.get("bg_method", default_bg_method),
            "gray_threshold": int(raw_cfg.get("gray_threshold", default_gray_threshold)),
            "invert": str(raw_cfg.get("bg_type", default_bg_type)).lower() == "dark",
            "erosion_kernel": int(raw_cfg.get("erosion_kernel", default_erosion_kernel)),
            "erosion_iter": int(raw_cfg.get("erosion_iter", default_erosion_iter)),
            "dilation_kernel": int(raw_cfg.get("dilation_kernel", default_dilation_kernel)),
            "dilation_iter": int(raw_cfg.get("dilation_iter", default_dilation_iter)),
            "min_area": int(raw_cfg.get("min_area", default_min_area)),
            "arm_length_cm": float(raw_cfg.get("arm_length_cm", default_arm_length_cm)),
            "arm_width_cm": float(raw_cfg.get("arm_width_cm", default_arm_width_cm)),
            "rois": copy.deepcopy(raw_cfg.get("rois", data.get("rois", []))),
            "movement_threshold_cm_s": float(raw_cfg.get("movement_threshold_cm_s", default_movement_threshold_cm_s)),
            "entry_depth_ratio": float(raw_cfg.get("entry_depth_ratio", default_entry_depth_ratio)),
            "bin_seconds": float(raw_cfg.get("bin_seconds", default_bin_seconds)),
        },
    )

    return {
        "bg_method": default_bg_method,
        "gray_threshold": default_gray_threshold,
        "invert": str(default_bg_type).lower() == "dark",
        "erosion_kernel": default_erosion_kernel,
        "erosion_iter": default_erosion_iter,
        "dilation_kernel": default_dilation_kernel,
        "dilation_iter": default_dilation_iter,
        "min_area": default_min_area,
        "max_area": int(data.get("max_area", 50000)),
        "rois": data.get("rois", []),
        "per_video_configs": per_video_configs,
        "arm_length_cm": default_arm_length_cm,
        "arm_width_cm": default_arm_width_cm,
        "movement_threshold_cm_s": default_movement_threshold_cm_s,
        "entry_depth_ratio": default_entry_depth_ratio,
        "bin_seconds": default_bin_seconds,
        "generate_video": data.get("generate_video", True),
        **_base_session_fields(sess),
    }


def build_tst_config(sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    default_bg_method = data.get("bg_method", "threshold")
    default_bg_type = data.get("bg_type", "light")
    default_gray_threshold = int(data.get("gray_threshold", 127))
    default_erosion_kernel = int(data.get("erosion_kernel", 3))
    default_erosion_iter = int(data.get("erosion_iter", 2))
    default_dilation_kernel = int(data.get("dilation_kernel", 5))
    default_dilation_iter = int(data.get("dilation_iter", 3))
    default_min_area = int(data.get("min_area", 100))

    per_video_configs = _iter_video_configs(
        sess,
        data,
        lambda raw_cfg: {
            "bg_method": raw_cfg.get("bg_method", default_bg_method),
            "gray_threshold": int(raw_cfg.get("gray_threshold", default_gray_threshold)),
            "invert": str(raw_cfg.get("bg_type", default_bg_type)).lower() == "dark",
            "erosion_kernel": int(raw_cfg.get("erosion_kernel", default_erosion_kernel)),
            "erosion_iter": int(raw_cfg.get("erosion_iter", default_erosion_iter)),
            "dilation_kernel": int(raw_cfg.get("dilation_kernel", default_dilation_kernel)),
            "dilation_iter": int(raw_cfg.get("dilation_iter", default_dilation_iter)),
            "min_area": int(raw_cfg.get("min_area", default_min_area)),
            "pixel_to_mm": float(raw_cfg.get("pixel_to_mm", data.get("pixel_to_mm", 1.0))),
            "rois": copy.deepcopy(raw_cfg.get("rois", data.get("rois", []))),
        },
    )

    return {
        "bg_method": default_bg_method,
        "gray_threshold": default_gray_threshold,
        "invert": str(default_bg_type).lower() == "dark",
        "erosion_kernel": default_erosion_kernel,
        "erosion_iter": default_erosion_iter,
        "dilation_kernel": default_dilation_kernel,
        "dilation_iter": default_dilation_iter,
        "min_area": default_min_area,
        "pixel_to_mm": float(data.get("pixel_to_mm", 1.0)),
        "rois": data.get("rois", []),
        "per_video_configs": per_video_configs,
        "generate_video": data.get("generate_video", True),
        **_base_session_fields(sess),
    }


def build_analysis_config(experiment_type: str | None, sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    exp = str(experiment_type or "").strip().lower()
    if exp == "open_field":
        return build_open_field_config(sess, data)
    if exp == "novel_object":
        return build_novel_object_config(sess, data)
    if exp == "elevated_plus":
        return build_elevated_plus_config(sess, data)
    if exp == "y_maze":
        return build_y_maze_config(sess, data)
    return build_tst_config(sess, data)
