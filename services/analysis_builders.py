from __future__ import annotations

import copy
import math
from typing import Any

from services.video_orientation import orientation_override_degrees
from services.experiment_defaults import merge_experiment_defaults


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
        config = builder(raw_cfg)
        runtime_keys = {"start_sec", "end_sec", "fps_override", "orientation_override"}
        if any(key in raw_cfg for key in runtime_keys):
            config.update(_per_video_runtime_settings(raw_cfg))
        per_video_configs.append(config)
    return per_video_configs


def _optional_finite_float(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _per_video_runtime_settings(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep batch-only timing/FPS/orientation overrides across all assays."""
    start_sec = _optional_finite_float(raw_cfg.get("start_sec"))
    end_sec = _optional_finite_float(raw_cfg.get("end_sec"))
    fps_override = _optional_finite_float(raw_cfg.get("fps_override"))
    return {
        "start_sec": max(0.0, start_sec) if start_sec is not None else None,
        "end_sec": max(0.0, end_sec) if end_sec is not None else None,
        "fps_override": fps_override if fps_override and 0.001 <= fps_override <= 1000.0 else None,
        "orientation_override": orientation_override_degrees(raw_cfg.get("orientation_override")),
    }


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false", "no", "off", "disabled", "hidden", "invisible"}:
            return False
        if normalized in {"1", "true", "yes", "on", "enabled", "visible"}:
            return True
    return bool(value)


def _tracking_backend(value: Any) -> str:
    backend = str(value or "classical").strip().lower()
    return (
        backend
        if backend in {"classical", "deeplabcut", "auto", "hybrid"}
        else "classical"
    )


def _pose_tracking_fields(
    raw: dict[str, Any],
    defaults: dict[str, Any],
    *,
    allow_hybrid: bool = False,
) -> dict[str, Any]:
    requested = _tracking_backend(raw.get("tracking_backend", defaults.get("tracking_backend")))
    if requested == "hybrid" and not allow_hybrid:
        requested = "classical"
    try:
        pose_batch_size = int(
            raw.get("pose_batch_size", defaults.get("pose_batch_size", 4))
        )
    except (TypeError, ValueError):
        pose_batch_size = 4
    return {
        "tracking_backend": requested,
        # Route startup resolves this requested backend against the optional
        # DLC sidecar and selected immutable model version.
        "tracking_backend_effective": "classical",
        "tracking_fallback_reason": (
            None if requested == "classical" else "deeplabcut_analysis_adapter_pending"
        ),
        "pose_model_profile_id": str(
            raw.get("pose_model_profile_id", defaults.get("pose_model_profile_id")) or ""
        ).strip() or None,
        "pose_model_version_id": str(
            raw.get("pose_model_version_id", defaults.get("pose_model_version_id")) or ""
        ).strip() or None,
        "pose_confidence_threshold": float(
            raw.get("pose_confidence_threshold", defaults.get("pose_confidence_threshold", 0.8))
        ),
        "pose_batch_size": max(1, min(16, pose_batch_size)),
    }


def _nor_hybrid_fields(
    raw: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pose_trigger_margin_cm": max(
            0.0,
            min(
                20.0,
                float(
                    raw.get(
                        "pose_trigger_margin_cm",
                        defaults.get("pose_trigger_margin_cm", 2.0),
                    )
                ),
            ),
        ),
        "pose_crop_padding_ratio": max(
            0.0,
            min(
                3.0,
                float(
                    raw.get(
                        "pose_crop_padding_ratio",
                        defaults.get("pose_crop_padding_ratio", 0.5),
                    )
                ),
            ),
        ),
        "pose_cache_enabled": _coerce_bool(
            raw.get("pose_cache_enabled", defaults.get("pose_cache_enabled")),
            True,
        ),
    }


def build_open_field_config(sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    tracking_defaults = _pose_tracking_fields({}, data)
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
            **_pose_tracking_fields(raw_cfg, tracking_defaults),
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
        **tracking_defaults,
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
    tracking_defaults = {
        **_pose_tracking_fields({}, data, allow_hybrid=True),
        **_nor_hybrid_fields({}, data),
    }
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
    default_pose_head_hold_s = float(data.get("pose_head_hold_s", 0.12))
    default_pose_head_smoothing = float(data.get("pose_head_smoothing", 0.35))
    default_body_gray_threshold = int(data.get("body_gray_threshold", 95))
    default_movement_threshold_cm_s = float(data.get("movement_threshold_cm_s", 2.0))
    default_min_valid_exploration_s = float(data.get("min_valid_exploration_s", 20.0))
    default_bin_seconds = float(data.get("bin_seconds", 300.0))

    per_video_configs = _iter_video_configs(
        sess,
        data,
        lambda raw_cfg: {
            **_pose_tracking_fields(
                raw_cfg,
                tracking_defaults,
                allow_hybrid=True,
            ),
            **_nor_hybrid_fields(raw_cfg, tracking_defaults),
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
            "pose_head_hold_s": float(raw_cfg.get("pose_head_hold_s", default_pose_head_hold_s)),
            "pose_head_smoothing": float(raw_cfg.get("pose_head_smoothing", default_pose_head_smoothing)),
            "body_gray_threshold": int(raw_cfg.get("body_gray_threshold", default_body_gray_threshold)),
            "movement_threshold_cm_s": float(raw_cfg.get("movement_threshold_cm_s", default_movement_threshold_cm_s)),
            "min_valid_exploration_s": float(raw_cfg.get("min_valid_exploration_s", default_min_valid_exploration_s)),
            "bin_seconds": float(raw_cfg.get("bin_seconds", default_bin_seconds)),
        },
    )

    return {
        **tracking_defaults,
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
        "pose_head_hold_s": default_pose_head_hold_s,
        "pose_head_smoothing": default_pose_head_smoothing,
        "body_gray_threshold": default_body_gray_threshold,
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
    default_tracking_max_jump_cm = float(data.get("tracking_max_jump_cm", 12.0))
    default_occlusion_hold_s = float(data.get("occlusion_hold_s", 0.4))
    default_zone_confirm_s = float(data.get("zone_confirm_s", 0.12))
    default_center_reset_s = float(data.get("center_reset_s", 0.2))
    default_require_center_passage = _coerce_bool(data.get("require_center_passage"), True)
    default_temporal_bg_samples = int(data.get("temporal_bg_samples", 90))
    default_temporal_diff_threshold = int(data.get("temporal_diff_threshold", 25))
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
            "tracking_max_jump_cm": float(raw_cfg.get("tracking_max_jump_cm", default_tracking_max_jump_cm)),
            "occlusion_hold_s": float(raw_cfg.get("occlusion_hold_s", default_occlusion_hold_s)),
            "zone_confirm_s": float(raw_cfg.get("zone_confirm_s", default_zone_confirm_s)),
            "center_reset_s": float(raw_cfg.get("center_reset_s", default_center_reset_s)),
            "require_center_passage": _coerce_bool(
                raw_cfg.get("require_center_passage"),
                default_require_center_passage,
            ),
            "temporal_bg_samples": int(raw_cfg.get("temporal_bg_samples", default_temporal_bg_samples)),
            "temporal_diff_threshold": int(raw_cfg.get("temporal_diff_threshold", default_temporal_diff_threshold)),
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
        "tracking_max_jump_cm": default_tracking_max_jump_cm,
        "occlusion_hold_s": default_occlusion_hold_s,
        "zone_confirm_s": default_zone_confirm_s,
        "center_reset_s": default_center_reset_s,
        "require_center_passage": default_require_center_passage,
        "temporal_bg_samples": default_temporal_bg_samples,
        "temporal_diff_threshold": default_temporal_diff_threshold,
        "bin_seconds": default_bin_seconds,
        "generate_video": data.get("generate_video", True),
        **_base_session_fields(sess),
    }


def build_light_dark_box_config(sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    default_bg_method = data.get("bg_method", "threshold")
    default_bg_type = data.get("bg_type", "light")
    default_gray_threshold = int(data.get("gray_threshold", 127))
    default_erosion_kernel = int(data.get("erosion_kernel", 3))
    default_erosion_iter = int(data.get("erosion_iter", 1))
    default_dilation_kernel = int(data.get("dilation_kernel", 5))
    default_dilation_iter = int(data.get("dilation_iter", 2))
    default_min_area = int(data.get("min_area", 120))
    default_arena_width_cm = float(data.get("arena_width_cm", 40.0))
    default_arena_height_cm = float(data.get("arena_height_cm", 40.0))
    def _default_ldb_dims(arena_width_cm: float, arena_height_cm: float) -> tuple[float, float, float]:
        if abs(float(arena_width_cm) - float(arena_height_cm)) <= 1e-6:
            side = max(float(arena_width_cm), float(arena_height_cm))
            return side, side / 2.0, side
        return float(arena_width_cm) / 2.0, float(arena_height_cm), float(arena_height_cm)

    def _normalize_ldb_dims(length_cm: float, width_cm: float, shared_edge_cm: float) -> tuple[float, float]:
        length = float(length_cm)
        width = float(width_cm)
        shared = float(shared_edge_cm)
        if length + 1e-6 < width and shared + 1e-6 >= width:
            return width, length
        return length, width

    default_length_cm, default_width_cm, default_shared_cm = _default_ldb_dims(default_arena_width_cm, default_arena_height_cm)
    default_light_box_length_cm = float(data.get("light_box_length_cm", default_length_cm))
    default_light_box_width_cm = float(data.get("light_box_width_cm", default_width_cm))
    default_dark_box_length_cm = float(data.get("dark_box_length_cm", default_length_cm))
    default_dark_box_width_cm = float(data.get("dark_box_width_cm", default_width_cm))
    default_shared_edge_length_cm = float(data.get(
        "shared_edge_length_cm",
        min(
            max(default_light_box_length_cm, default_light_box_width_cm),
            max(default_dark_box_length_cm, default_dark_box_width_cm),
            default_shared_cm,
        ),
    ))
    default_light_box_length_cm, default_light_box_width_cm = _normalize_ldb_dims(
        default_light_box_length_cm,
        default_light_box_width_cm,
        default_shared_edge_length_cm,
    )
    default_dark_box_length_cm, default_dark_box_width_cm = _normalize_ldb_dims(
        default_dark_box_length_cm,
        default_dark_box_width_cm,
        default_shared_edge_length_cm,
    )
    default_door_open_delay_s = float(data.get("door_open_delay_s", 3.0))
    default_transition_min_dwell_s = float(data.get("transition_min_dwell_s", 0.12))
    default_doorway_reset_s = float(data.get("doorway_reset_s", 0.2))
    default_movement_threshold_cm_s = float(data.get("movement_threshold_cm_s", 2.0))
    default_bin_seconds = float(data.get("bin_seconds", 60.0))
    default_dark_box_visible = _coerce_bool(data.get("dark_box_visible"), True)

    def _float_cfg(raw_cfg: dict[str, Any], key: str, fallback: float) -> float:
        try:
            return float(raw_cfg.get(key, fallback))
        except Exception:
            return float(fallback)

    def _per_video_light_dark_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
        arena_width_cm = _float_cfg(raw_cfg, "arena_width_cm", default_arena_width_cm)
        arena_height_cm = _float_cfg(raw_cfg, "arena_height_cm", default_arena_height_cm)
        fallback_length_cm, fallback_width_cm, fallback_shared_cm = _default_ldb_dims(arena_width_cm, arena_height_cm)
        light_box_length_cm = _float_cfg(
            raw_cfg,
            "light_box_length_cm",
            default_light_box_length_cm if "light_box_length_cm" in data else fallback_length_cm,
        )
        light_box_width_cm = _float_cfg(
            raw_cfg,
            "light_box_width_cm",
            default_light_box_width_cm if "light_box_width_cm" in data else fallback_width_cm,
        )
        dark_box_length_cm = _float_cfg(
            raw_cfg,
            "dark_box_length_cm",
            default_dark_box_length_cm if "dark_box_length_cm" in data else fallback_length_cm,
        )
        dark_box_width_cm = _float_cfg(
            raw_cfg,
            "dark_box_width_cm",
            default_dark_box_width_cm if "dark_box_width_cm" in data else fallback_width_cm,
        )
        shared_edge_length_cm = _float_cfg(
            raw_cfg,
            "shared_edge_length_cm",
            default_shared_edge_length_cm if "shared_edge_length_cm" in data else min(
                max(light_box_length_cm, light_box_width_cm),
                max(dark_box_length_cm, dark_box_width_cm),
                fallback_shared_cm,
            ),
        )
        light_box_length_cm, light_box_width_cm = _normalize_ldb_dims(
            light_box_length_cm,
            light_box_width_cm,
            shared_edge_length_cm,
        )
        dark_box_length_cm, dark_box_width_cm = _normalize_ldb_dims(
            dark_box_length_cm,
            dark_box_width_cm,
            shared_edge_length_cm,
        )
        return {
            "dark_box_visible": _coerce_bool(raw_cfg.get("dark_box_visible"), default_dark_box_visible),
            "bg_method": raw_cfg.get("bg_method", default_bg_method),
            "gray_threshold": int(raw_cfg.get("gray_threshold", default_gray_threshold)),
            "invert": str(raw_cfg.get("bg_type", default_bg_type)).lower() == "dark",
            "erosion_kernel": int(raw_cfg.get("erosion_kernel", default_erosion_kernel)),
            "erosion_iter": int(raw_cfg.get("erosion_iter", default_erosion_iter)),
            "dilation_kernel": int(raw_cfg.get("dilation_kernel", default_dilation_kernel)),
            "dilation_iter": int(raw_cfg.get("dilation_iter", default_dilation_iter)),
            "min_area": int(raw_cfg.get("min_area", default_min_area)),
            "arena_width_cm": arena_width_cm,
            "arena_height_cm": arena_height_cm,
            "light_box_length_cm": light_box_length_cm,
            "light_box_width_cm": light_box_width_cm,
            "dark_box_length_cm": dark_box_length_cm,
            "dark_box_width_cm": dark_box_width_cm,
            "shared_edge_length_cm": shared_edge_length_cm,
            "door_open_delay_s": float(raw_cfg.get("door_open_delay_s", default_door_open_delay_s)),
            "transition_min_dwell_s": float(raw_cfg.get("transition_min_dwell_s", default_transition_min_dwell_s)),
            "doorway_reset_s": float(raw_cfg.get("doorway_reset_s", default_doorway_reset_s)),
            "movement_threshold_cm_s": float(raw_cfg.get("movement_threshold_cm_s", default_movement_threshold_cm_s)),
            "bin_seconds": float(raw_cfg.get("bin_seconds", default_bin_seconds)),
            "rois": copy.deepcopy(raw_cfg.get("rois", data.get("rois", []))),
        }

    per_video_configs = _iter_video_configs(
        sess,
        data,
        _per_video_light_dark_config,
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
        "arena_width_cm": default_arena_width_cm,
        "arena_height_cm": default_arena_height_cm,
        "light_box_length_cm": default_light_box_length_cm,
        "light_box_width_cm": default_light_box_width_cm,
        "dark_box_length_cm": default_dark_box_length_cm,
        "dark_box_width_cm": default_dark_box_width_cm,
        "shared_edge_length_cm": default_shared_edge_length_cm,
        "door_open_delay_s": default_door_open_delay_s,
        "transition_min_dwell_s": default_transition_min_dwell_s,
        "doorway_reset_s": default_doorway_reset_s,
        "movement_threshold_cm_s": default_movement_threshold_cm_s,
        "bin_seconds": default_bin_seconds,
        "dark_box_visible": default_dark_box_visible,
        "rois": data.get("rois", []),
        "per_video_configs": per_video_configs,
        "generate_video": data.get("generate_video", True),
        **_base_session_fields(sess),
    }


def _default_three_chamber_segments() -> list[dict[str, Any]]:
    return [
        {"id": "habituation_center", "phase": "habituation_center", "start_s": 0.0, "end_s": 300.0, "analyze": True},
        {"id": "gap_1", "phase": "ignore", "start_s": 300.0, "end_s": 420.0, "analyze": False},
        {"id": "habituation_all", "phase": "habituation_all", "start_s": 420.0, "end_s": 1020.0, "analyze": True},
        {"id": "gap_2", "phase": "ignore", "start_s": 1020.0, "end_s": 1140.0, "analyze": False},
        {"id": "sociability", "phase": "sociability", "start_s": 1140.0, "end_s": 1740.0, "analyze": True},
        {"id": "gap_3", "phase": "ignore", "start_s": 1740.0, "end_s": 1860.0, "analyze": False},
        {"id": "social_novelty", "phase": "social_novelty", "start_s": 1860.0, "end_s": 2460.0, "analyze": True},
    ]


def _default_three_chamber_roles() -> dict[str, dict[str, str]]:
    return {
        "habituation_center": {"left": "", "right": ""},
        "habituation_all": {"left": "empty_object", "right": "empty_object"},
        "sociability": {"left": "social_stranger_1", "right": "empty_object"},
        "social_novelty": {"left": "familiar_stranger_1", "right": "novel_stranger_2"},
    }


def build_three_chamber_config(sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    default_analysis_mode = data.get("analysis_mode", "single_phase")
    default_selected_phase = data.get("selected_phase", data.get("phase", "sociability"))
    default_protocol_segments = copy.deepcopy(data.get("protocol_segments") or ([] if default_analysis_mode != "full_protocol" else _default_three_chamber_segments()))
    default_role_schedule = copy.deepcopy(data.get("role_schedule") or _default_three_chamber_roles())
    default_bg_method = data.get("bg_method", "threshold")
    default_bg_type = data.get("bg_type", "light")
    default_gray_threshold = int(data.get("gray_threshold", 127))
    default_erosion_kernel = int(data.get("erosion_kernel", 3))
    default_erosion_iter = int(data.get("erosion_iter", 1))
    default_dilation_kernel = int(data.get("dilation_kernel", 5))
    default_dilation_iter = int(data.get("dilation_iter", 2))
    default_min_area = int(data.get("min_area", 120))
    default_arena_width_cm = float(data.get("arena_width_cm", 60.0))
    default_arena_height_cm = float(data.get("arena_height_cm", 40.0))
    default_interaction_radius_cm = float(data.get("interaction_radius_cm", 2.0))
    default_cup_exclusion_radius_cm = float(data.get("cup_exclusion_radius_cm", 1.5))
    default_use_head_direction = bool(data.get("use_head_direction", False))
    default_head_angle_threshold_deg = float(data.get("head_angle_threshold_deg", 60.0))
    default_head_confidence_threshold = float(data.get("head_confidence_threshold", 0.35))
    default_transition_min_dwell_s = float(data.get("transition_min_dwell_s", 0.12))
    default_center_reset_s = float(data.get("center_reset_s", 0.2))
    default_movement_threshold_cm_s = float(data.get("movement_threshold_cm_s", 2.0))
    default_bin_seconds = float(data.get("bin_seconds", 60.0))

    per_video_configs = _iter_video_configs(
        sess,
        data,
        lambda raw_cfg: {
            "analysis_mode": raw_cfg.get("analysis_mode", default_analysis_mode),
            "selected_phase": raw_cfg.get("selected_phase", raw_cfg.get("phase", default_selected_phase)),
            "phase": raw_cfg.get("phase", raw_cfg.get("selected_phase", default_selected_phase)),
            "protocol_segments": copy.deepcopy(raw_cfg.get("protocol_segments", default_protocol_segments)),
            "role_schedule": copy.deepcopy(raw_cfg.get("role_schedule", default_role_schedule)),
            "bg_method": raw_cfg.get("bg_method", default_bg_method),
            "gray_threshold": int(raw_cfg.get("gray_threshold", default_gray_threshold)),
            "invert": str(raw_cfg.get("bg_type", default_bg_type)).lower() == "dark",
            "erosion_kernel": int(raw_cfg.get("erosion_kernel", default_erosion_kernel)),
            "erosion_iter": int(raw_cfg.get("erosion_iter", default_erosion_iter)),
            "dilation_kernel": int(raw_cfg.get("dilation_kernel", default_dilation_kernel)),
            "dilation_iter": int(raw_cfg.get("dilation_iter", default_dilation_iter)),
            "min_area": int(raw_cfg.get("min_area", default_min_area)),
            "arena_width_cm": float(raw_cfg.get("arena_width_cm", default_arena_width_cm)),
            "arena_height_cm": float(raw_cfg.get("arena_height_cm", default_arena_height_cm)),
            "interaction_radius_cm": float(raw_cfg.get("interaction_radius_cm", default_interaction_radius_cm)),
            "cup_exclusion_radius_cm": float(raw_cfg.get("cup_exclusion_radius_cm", default_cup_exclusion_radius_cm)),
            "use_head_direction": bool(raw_cfg.get("use_head_direction", default_use_head_direction)),
            "head_angle_threshold_deg": float(raw_cfg.get("head_angle_threshold_deg", default_head_angle_threshold_deg)),
            "head_confidence_threshold": float(raw_cfg.get("head_confidence_threshold", default_head_confidence_threshold)),
            "transition_min_dwell_s": float(raw_cfg.get("transition_min_dwell_s", default_transition_min_dwell_s)),
            "center_reset_s": float(raw_cfg.get("center_reset_s", default_center_reset_s)),
            "movement_threshold_cm_s": float(raw_cfg.get("movement_threshold_cm_s", default_movement_threshold_cm_s)),
            "bin_seconds": float(raw_cfg.get("bin_seconds", default_bin_seconds)),
            "rois": copy.deepcopy(raw_cfg.get("rois", data.get("rois", []))),
        },
    )
    return {
        "analysis_mode": default_analysis_mode,
        "selected_phase": default_selected_phase,
        "phase": default_selected_phase,
        "protocol_segments": default_protocol_segments,
        "role_schedule": default_role_schedule,
        "bg_method": default_bg_method,
        "gray_threshold": default_gray_threshold,
        "invert": str(default_bg_type).lower() == "dark",
        "erosion_kernel": default_erosion_kernel,
        "erosion_iter": default_erosion_iter,
        "dilation_kernel": default_dilation_kernel,
        "dilation_iter": default_dilation_iter,
        "min_area": default_min_area,
        "max_area": int(data.get("max_area", 50000)),
        "arena_width_cm": default_arena_width_cm,
        "arena_height_cm": default_arena_height_cm,
        "interaction_radius_cm": default_interaction_radius_cm,
        "cup_exclusion_radius_cm": default_cup_exclusion_radius_cm,
        "use_head_direction": default_use_head_direction,
        "head_angle_threshold_deg": default_head_angle_threshold_deg,
        "head_confidence_threshold": default_head_confidence_threshold,
        "transition_min_dwell_s": default_transition_min_dwell_s,
        "center_reset_s": default_center_reset_s,
        "movement_threshold_cm_s": default_movement_threshold_cm_s,
        "bin_seconds": default_bin_seconds,
        "rois": data.get("rois", []),
        "per_video_configs": per_video_configs,
        "generate_video": data.get("generate_video", True),
        **_base_session_fields(sess),
    }


def build_tst_config(sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    def _resolve_lis_primary(raw_cfg: dict[str, Any], default: bool = True) -> bool:
        if "use_tst_lis_primary" in raw_cfg:
            return bool(raw_cfg.get("use_tst_lis_primary"))
        if "enable_tst_lis" in raw_cfg:
            return bool(raw_cfg.get("enable_tst_lis"))
        if "use_tst_legacy_primary" in raw_cfg:
            return not bool(raw_cfg.get("use_tst_legacy_primary", False))
        return bool(default)

    default_bg_method = data.get("bg_method", "threshold")
    default_bg_type = data.get("bg_type", "light")
    default_gray_threshold = int(data.get("gray_threshold", 127))
    default_erosion_kernel = int(data.get("erosion_kernel", 3))
    default_erosion_iter = int(data.get("erosion_iter", 2))
    default_dilation_kernel = int(data.get("dilation_kernel", 5))
    default_dilation_iter = int(data.get("dilation_iter", 3))
    default_min_area = int(data.get("min_area", 100))
    default_max_area = int(data.get("max_area", 50000))
    default_behavior_threshold_sensitivity = float(data.get("behavior_threshold_sensitivity", 0.0))
    default_use_tst_lis_primary = _resolve_lis_primary(data, False)
    default_use_tst_legacy_primary = not default_use_tst_lis_primary
    default_auto_behavior_threshold = (
        False if default_use_tst_lis_primary else bool(data.get("auto_behavior_threshold", False))
    )
    default_enable_tst_lis = default_use_tst_lis_primary

    def _build_video_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
        use_tst_lis_primary = _resolve_lis_primary(raw_cfg, default_use_tst_lis_primary)
        auto_behavior_threshold = (
            False
            if use_tst_lis_primary
            else bool(raw_cfg.get("auto_behavior_threshold", default_auto_behavior_threshold))
        )
        return {
            "bg_method": raw_cfg.get("bg_method", default_bg_method),
            "gray_threshold": int(raw_cfg.get("gray_threshold", default_gray_threshold)),
            "invert": str(raw_cfg.get("bg_type", default_bg_type)).lower() == "dark",
            "erosion_kernel": int(raw_cfg.get("erosion_kernel", default_erosion_kernel)),
            "erosion_iter": int(raw_cfg.get("erosion_iter", default_erosion_iter)),
            "dilation_kernel": int(raw_cfg.get("dilation_kernel", default_dilation_kernel)),
            "dilation_iter": int(raw_cfg.get("dilation_iter", default_dilation_iter)),
            "min_area": int(raw_cfg.get("min_area", default_min_area)),
            "max_area": int(raw_cfg.get("max_area", default_max_area)),
            "auto_behavior_threshold": auto_behavior_threshold,
            "behavior_threshold_sensitivity": float(raw_cfg.get("behavior_threshold_sensitivity", default_behavior_threshold_sensitivity)),
            "use_tst_lis_primary": use_tst_lis_primary,
            "use_tst_legacy_primary": not use_tst_lis_primary,
            "enable_tst_lis": use_tst_lis_primary,
            "pixel_to_mm": float(raw_cfg.get("pixel_to_mm", data.get("pixel_to_mm", 1.0))),
            "rois": copy.deepcopy(raw_cfg.get("rois", data.get("rois", []))),
        }

    per_video_configs = _iter_video_configs(
        sess,
        data,
        _build_video_config,
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
        "max_area": default_max_area,
        "auto_behavior_threshold": default_auto_behavior_threshold,
        "behavior_threshold_sensitivity": default_behavior_threshold_sensitivity,
        "use_tst_lis_primary": default_use_tst_lis_primary,
        "use_tst_legacy_primary": default_use_tst_legacy_primary,
        "enable_tst_lis": default_enable_tst_lis,
        "pixel_to_mm": float(data.get("pixel_to_mm", 1.0)),
        "rois": data.get("rois", []),
        "per_video_configs": per_video_configs,
        "generate_video": data.get("generate_video", True),
        **_base_session_fields(sess),
    }


def build_forced_swim_config(sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    default_bg_method = data.get("bg_method", "threshold")
    default_bg_type = data.get("bg_type", "light")
    default_gray_threshold = int(data.get("gray_threshold", 127))
    default_erosion_kernel = int(data.get("erosion_kernel", 3))
    default_erosion_iter = int(data.get("erosion_iter", 1))
    default_dilation_kernel = int(data.get("dilation_kernel", 5))
    default_dilation_iter = int(data.get("dilation_iter", 2))
    default_min_area = int(data.get("min_area", 80))
    default_max_area = int(data.get("max_area", 50000))
    default_cylinder_diameter_cm = float(data.get("cylinder_diameter_cm", 10.0))
    default_auto_threshold = bool(data.get("auto_threshold", True))
    default_immobility_threshold = data.get("immobility_threshold")
    default_immobility_high_threshold = data.get("immobility_high_threshold")
    default_threshold_sensitivity = float(data.get("threshold_sensitivity", 0.0))
    default_smoothing_window_s = float(data.get("smoothing_window_s", 0.5))
    default_min_immobility_duration_s = float(data.get("min_immobility_duration_s", 1.0))
    default_min_active_duration_s = float(data.get("min_active_duration_s", 0.3))
    default_gap_merge_s = float(data.get("gap_merge_s", 0.5))
    default_bin_seconds = float(data.get("bin_seconds", 60.0))

    def _float_or_none(value):
        if value in ("", None):
            return None
        try:
            return float(value)
        except Exception:
            return None

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
            "max_area": int(raw_cfg.get("max_area", default_max_area)),
            "cylinder_diameter_cm": float(raw_cfg.get("cylinder_diameter_cm", default_cylinder_diameter_cm)),
            "rois": copy.deepcopy(raw_cfg.get("rois", data.get("rois", []))),
            "auto_threshold": bool(raw_cfg.get("auto_threshold", default_auto_threshold)),
            "immobility_threshold": _float_or_none(raw_cfg.get("immobility_threshold", default_immobility_threshold)),
            "immobility_high_threshold": _float_or_none(raw_cfg.get("immobility_high_threshold", default_immobility_high_threshold)),
            "threshold_sensitivity": float(raw_cfg.get("threshold_sensitivity", default_threshold_sensitivity)),
            "smoothing_window_s": float(raw_cfg.get("smoothing_window_s", default_smoothing_window_s)),
            "min_immobility_duration_s": float(raw_cfg.get("min_immobility_duration_s", default_min_immobility_duration_s)),
            "min_active_duration_s": float(raw_cfg.get("min_active_duration_s", default_min_active_duration_s)),
            "gap_merge_s": float(raw_cfg.get("gap_merge_s", default_gap_merge_s)),
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
        "max_area": default_max_area,
        "cylinder_diameter_cm": default_cylinder_diameter_cm,
        "rois": data.get("rois", []),
        "per_video_configs": per_video_configs,
        "auto_threshold": default_auto_threshold,
        "immobility_threshold": _float_or_none(default_immobility_threshold),
        "immobility_high_threshold": _float_or_none(default_immobility_high_threshold),
        "threshold_sensitivity": default_threshold_sensitivity,
        "smoothing_window_s": default_smoothing_window_s,
        "min_immobility_duration_s": default_min_immobility_duration_s,
        "min_active_duration_s": default_min_active_duration_s,
        "gap_merge_s": default_gap_merge_s,
        "bin_seconds": default_bin_seconds,
        "generate_video": data.get("generate_video", True),
        **_base_session_fields(sess),
    }


def build_water_maze_config(sess: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    default_trial_type = data.get("trial_type", "acquisition_hidden")
    default_day_label = data.get("day_label", "")
    default_trial_no = int(data.get("trial_no", 1))
    default_pool_diameter_cm = float(data.get("pool_diameter_cm", 120.0))
    default_platform_radius_cm = float(data.get("platform_radius_cm", 5.0))
    default_target_quadrant = data.get("target_quadrant", "")
    default_start_position = data.get("start_position", "N")
    default_quadrant_rotation_deg = float(data.get("quadrant_rotation_deg", 0.0))
    default_correction_mode = data.get("correction_mode", "ellipse_affine")
    default_distortion_k1 = float(data.get("distortion_k1", 0.0))
    default_distortion_k2 = float(data.get("distortion_k2", 0.0))
    default_thigmotaxis_band_cm = float(data.get("thigmotaxis_band_cm", 10.0))
    default_platform_dwell_s = float(data.get("platform_dwell_s", 1.0))
    default_max_jump_cm = float(data.get("track_smoothing_max_jump_cm", data.get("max_jump_cm", 25.0)))
    default_max_gap_s = float(data.get("max_gap_s", 0.5))
    default_movement_threshold_cm_s = float(data.get("movement_threshold_cm_s", 2.0))
    default_whishaw_corridor_width_cm = data.get("whishaw_corridor_width_cm")
    default_exclude_platform = _coerce_bool(data.get("exclude_platform_from_detection"), False)
    default_bin_seconds = float(data.get("bin_seconds", 10.0))
    default_bg_method = data.get("bg_method", "threshold")
    default_bg_type = data.get("bg_type", "light")
    default_gray_threshold = int(data.get("gray_threshold", 127))
    default_erosion_kernel = int(data.get("erosion_kernel", 3))
    default_erosion_iter = int(data.get("erosion_iter", 1))
    default_dilation_kernel = int(data.get("dilation_kernel", 5))
    default_dilation_iter = int(data.get("dilation_iter", 2))
    default_min_area = int(data.get("min_area", 80))
    default_max_area = int(data.get("max_area", 50000))

    def _float_or_none(value):
        if value in ("", None):
            return None
        try:
            return float(value)
        except Exception:
            return None

    per_video_configs = _iter_video_configs(
        sess,
        data,
        lambda raw_cfg: {
            "trial_type": raw_cfg.get("trial_type", default_trial_type),
            "day_label": raw_cfg.get("day_label", default_day_label),
            "trial_no": int(raw_cfg.get("trial_no", default_trial_no)),
            "pool_diameter_cm": float(raw_cfg.get("pool_diameter_cm", default_pool_diameter_cm)),
            "platform_radius_cm": float(raw_cfg.get("platform_radius_cm", default_platform_radius_cm)),
            "target_quadrant": raw_cfg.get("target_quadrant", default_target_quadrant),
            "start_position": raw_cfg.get("start_position", default_start_position),
            "quadrant_rotation_deg": float(raw_cfg.get("quadrant_rotation_deg", default_quadrant_rotation_deg)),
            "correction_mode": raw_cfg.get("correction_mode", default_correction_mode),
            "distortion_k1": float(raw_cfg.get("distortion_k1", default_distortion_k1)),
            "distortion_k2": float(raw_cfg.get("distortion_k2", default_distortion_k2)),
            "thigmotaxis_band_cm": float(raw_cfg.get("thigmotaxis_band_cm", default_thigmotaxis_band_cm)),
            "platform_dwell_s": float(raw_cfg.get("platform_dwell_s", default_platform_dwell_s)),
            "track_smoothing_max_jump_cm": float(raw_cfg.get("track_smoothing_max_jump_cm", raw_cfg.get("max_jump_cm", default_max_jump_cm))),
            "max_gap_s": float(raw_cfg.get("max_gap_s", default_max_gap_s)),
            "movement_threshold_cm_s": float(raw_cfg.get("movement_threshold_cm_s", default_movement_threshold_cm_s)),
            "whishaw_corridor_width_cm": _float_or_none(raw_cfg.get("whishaw_corridor_width_cm", default_whishaw_corridor_width_cm)),
            "exclude_platform_from_detection": _coerce_bool(raw_cfg.get("exclude_platform_from_detection"), default_exclude_platform),
            "bin_seconds": float(raw_cfg.get("bin_seconds", default_bin_seconds)),
            "bg_method": raw_cfg.get("bg_method", default_bg_method),
            "gray_threshold": int(raw_cfg.get("gray_threshold", default_gray_threshold)),
            "invert": str(raw_cfg.get("bg_type", default_bg_type)).lower() == "dark",
            "erosion_kernel": int(raw_cfg.get("erosion_kernel", default_erosion_kernel)),
            "erosion_iter": int(raw_cfg.get("erosion_iter", default_erosion_iter)),
            "dilation_kernel": int(raw_cfg.get("dilation_kernel", default_dilation_kernel)),
            "dilation_iter": int(raw_cfg.get("dilation_iter", default_dilation_iter)),
            "min_area": int(raw_cfg.get("min_area", default_min_area)),
            "max_area": int(raw_cfg.get("max_area", default_max_area)),
            "rois": copy.deepcopy(raw_cfg.get("rois", data.get("rois", []))),
        },
    )

    return {
        "trial_type": default_trial_type,
        "day_label": default_day_label,
        "trial_no": default_trial_no,
        "pool_diameter_cm": default_pool_diameter_cm,
        "platform_radius_cm": default_platform_radius_cm,
        "target_quadrant": default_target_quadrant,
        "start_position": default_start_position,
        "quadrant_rotation_deg": default_quadrant_rotation_deg,
        "correction_mode": default_correction_mode,
        "distortion_k1": default_distortion_k1,
        "distortion_k2": default_distortion_k2,
        "thigmotaxis_band_cm": default_thigmotaxis_band_cm,
        "platform_dwell_s": default_platform_dwell_s,
        "track_smoothing_max_jump_cm": default_max_jump_cm,
        "max_gap_s": default_max_gap_s,
        "movement_threshold_cm_s": default_movement_threshold_cm_s,
        "whishaw_corridor_width_cm": _float_or_none(default_whishaw_corridor_width_cm),
        "exclude_platform_from_detection": default_exclude_platform,
        "bin_seconds": default_bin_seconds,
        "bg_method": default_bg_method,
        "gray_threshold": default_gray_threshold,
        "invert": str(default_bg_type).lower() == "dark",
        "erosion_kernel": default_erosion_kernel,
        "erosion_iter": default_erosion_iter,
        "dilation_kernel": default_dilation_kernel,
        "dilation_iter": default_dilation_iter,
        "min_area": default_min_area,
        "max_area": default_max_area,
        "rois": data.get("rois", []),
        "per_video_configs": per_video_configs,
        "generate_video": data.get("generate_video", True),
        **_base_session_fields(sess),
    }


def build_analysis_config(
    experiment_type: str | None,
    sess: dict[str, Any],
    data: dict[str, Any],
    global_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    exp = str(experiment_type or "").strip().lower()
    data = merge_experiment_defaults(exp, data, global_defaults)
    if exp == "open_field":
        return build_open_field_config(sess, data)
    if exp == "novel_object":
        return build_novel_object_config(sess, data)
    if exp == "elevated_plus":
        return build_elevated_plus_config(sess, data)
    if exp == "y_maze":
        return build_y_maze_config(sess, data)
    if exp == "light_dark_box":
        return build_light_dark_box_config(sess, data)
    if exp == "three_chamber":
        return build_three_chamber_config(sess, data)
    if exp == "forced_swim":
        return build_forced_swim_config(sess, data)
    if exp == "water_maze":
        return build_water_maze_config(sess, data)
    return build_tst_config(sess, data)
