"""Barnes-only settings and trial geometry, shared by preview and analysis."""

from __future__ import annotations

import copy
import math

BARNES_DEFAULTS = {
    "tracking_backend": "classical",
    "pose_confidence_threshold": 0.8,
    "pose_batch_size": 4,
    "head_confidence_threshold": 0.6,
    "trial_type": "acquisition",
    "day_label": "",
    "trial_no": 1,
    "arena_diameter_cm": 90.0,
    "hole_count": 20,
    "hole_radius_cm": 2.0,
    "visit_confirm_s": 0.1,
    "visit_reset_s": 0.1,
    "tracking_max_jump_cm": 12.0,
    "escape_absence_s": 0.5,
    "escape_zone_radius_cm": 6.0,
    "bin_seconds": 60.0,
    "bg_method": "temporal",
    "bg_type": "light",
    "gray_threshold": 127,
    "temporal_bg_samples": 90,
    "temporal_diff_threshold": 25,
    "min_area": 120,
    "max_area": 50000,
    "erosion_kernel": 3,
    "erosion_iter": 1,
    "dilation_kernel": 5,
    "dilation_iter": 2,
    "generate_video": True,
}


def normalise_barnes_roi(raw, config, index=0):
    from barnes_maze import BarnesArena, BarnesScorer, point_xy

    if not isinstance(raw, dict):
        raise ValueError("巴恩斯ROI必须为对象。")
    roi = copy.deepcopy(raw)
    diameter = float(
        roi.get("arena_diameter_cm", config.get("arena_diameter_cm", 90.0))
    )
    radius = float(roi.get("hole_radius_cm", config.get("hole_radius_cm", 2.0)))
    arena = BarnesArena(
        roi.get("arena_points") or [],
        roi.get("holes") or [],
        roi.get("target_hole_id", ""),
        diameter,
        radius,
    )
    expected = int(roi.get("hole_count", config.get("hole_count", 20)))
    if len(arena.holes) != expected:
        raise ValueError(
            f"请标出全部洞位：当前{len(arena.holes)}个，设置为{expected}个。"
        )
    roi.update(
        name=str(roi.get("name") or f"Barnes {index+1}"),
        roi_id=index + 1,
        arena_points=[list(p) for p in arena.points],
        holes=[
            dict(id=h["id"], point=list(h["point"]), radius_cm=h["radius_cm"])
            for h in arena.holes
        ],
        target_hole_id=arena.target_hole_id,
        arena_diameter_cm=diameter,
        hole_radius_cm=radius,
        hole_count=expected,
    )
    if roi.get("detection_points"):
        if len(roi["detection_points"]) < 3:
            raise ValueError("检测遮罩至少需要3个点。")
        roi["detection_points"] = [list(point_xy(p)) for p in roi["detection_points"]]
    timing = roi.get("time_config") or {"mode": "global", "start": None, "end": None}
    if timing.get("mode") not in {"global", "precise"}:
        raise ValueError("ROI时间模式无效。")
    start = float(config.get("start_sec") or 0.0)
    end = config.get("end_sec")
    if timing.get("mode") == "precise":
        if timing.get("start") is not None:
            start = max(start, float(timing["start"]))
        if timing.get("end") is not None:
            end = (
                min(float(end), float(timing["end"]))
                if end is not None
                else float(timing["end"])
            )
    roi["time_config"] = timing
    # Construct the scorer now so malformed manual times/events fail before a run.
    BarnesScorer(
        arena,
        start_s=start,
        end_s=end,
        trial_type=config.get("trial_type", "acquisition"),
        manual_escape_time_s=roi.get("manual_escape_time_s"),
        manual_events=roi.get("manual_hole_events"),
    )
    return roi


def build_barnes_config(sess, data):
    def build(raw):
        config = copy.deepcopy(BARNES_DEFAULTS)
        config.update({k: raw[k] for k in BARNES_DEFAULTS if k in raw})
        for key in ["pose_model_profile_id", "pose_model_version_id"]:
            config[key] = str(raw.get(key) or "").strip() or None
        if config["tracking_backend"] not in {"classical", "deeplabcut"}:
            raise ValueError("巴恩斯追踪后端只能选择经典视觉或DLC。")
        if config["trial_type"] not in {"acquisition", "reversal", "probe"}:
            raise ValueError("巴恩斯试次类型无效。")
        positive_fields = {
            "arena_diameter_cm",
            "hole_count",
            "hole_radius_cm",
            "escape_absence_s",
            "escape_zone_radius_cm",
            "bin_seconds",
            "min_area",
            "max_area",
            "pose_batch_size",
            "temporal_bg_samples",
            "erosion_kernel",
            "dilation_kernel",
            "trial_no",
        }
        nonnegative = {
            "visit_confirm_s",
            "visit_reset_s",
            "tracking_max_jump_cm",
            "erosion_iter",
            "dilation_iter",
        }
        for key in (
            positive_fields
            | nonnegative
            | {
                "head_confidence_threshold",
                "pose_confidence_threshold",
                "gray_threshold",
                "temporal_diff_threshold",
            }
        ):
            value = float(config[key])
            if (
                not math.isfinite(value)
                or value < 0
                or (key in positive_fields and value == 0)
            ):
                raise ValueError(f"{key}数值无效。")
            config[key] = value
        if any(
            not 0 <= config[k] <= 1
            for k in ["head_confidence_threshold", "pose_confidence_threshold"]
        ):
            raise ValueError("关键点阈值必须在0到1之间。")
        if (
            not 1 <= config["pose_batch_size"] <= 16
            or int(config["pose_batch_size"]) != config["pose_batch_size"]
        ):
            raise ValueError("DLC批大小必须是1到16的整数。")
        if config["min_area"] > config["max_area"]:
            raise ValueError("最小面积不能大于最大面积。")
        if (
            not 2 <= config["hole_count"] <= 200
            or int(config["hole_count"]) != config["hole_count"]
        ):
            raise ValueError("洞数必须是2到200的整数。")
        for key in [
            "hole_count",
            "pose_batch_size",
            "temporal_bg_samples",
            "erosion_kernel",
            "dilation_kernel",
            "erosion_iter",
            "dilation_iter",
            "min_area",
            "max_area",
            "gray_threshold",
            "temporal_diff_threshold",
            "trial_no",
        ]:
            if int(config[key]) != config[key]:
                raise ValueError(f"{key}必须是整数。")
            config[key] = int(config[key])
        if config["bg_type"] not in {"light", "dark"}:
            raise ValueError("背景色只能为浅色或深色。")
        for key in ["erosion_kernel", "dilation_kernel"]:
            if config[key] > 99:
                raise ValueError("形态学核大小不能超过99。")
        for key in ["gray_threshold", "temporal_diff_threshold"]:
            if not 0 <= config[key] <= 255:
                raise ValueError(f"{key}必须在0到255之间。")
        if config["bg_method"] not in {
            "threshold",
            "temporal",
            "static",
            "mog2",
            "knn",
        }:
            raise ValueError("背景方法无效。")
        config["invert"] = str(config["bg_type"]).lower() == "dark"
        config["start_sec"] = float(
            raw.get("start_sec", sess.get("start_seconds")) or 0.0
        )
        end = raw.get("end_sec", sess.get("end_seconds"))
        config["end_sec"] = None if end in (None, "") else float(end)
        config["fps_override"] = raw.get("fps_override", sess.get("fps_override"))
        config["orientation_override"] = raw.get("orientation_override")
        config.update(
            fast_scoring=False,
            spatial_downsample=False,
            temporal_downsample=False,
            tracking_backend_effective=config["tracking_backend"],
        )
        config["rois"] = [
            normalise_barnes_roi(r, config, i)
            for i, r in enumerate(raw.get("rois") or [])
        ]
        return config

    global_config = build(data)
    per = data.get("per_video_configs") or []
    global_config["per_video_configs"] = [
        build(dict(data, **(per[i] if i < len(per) else {})))
        for i in range(len(sess.get("video_infos") or []))
    ]
    global_config["output_dir"] = sess.get("output_dir")
    return global_config
