from __future__ import annotations

import math
from typing import Any


def optional_finite_float(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def effective_video_duration_seconds(
    video_info: dict[str, Any] | None,
    fps_override: Any = None,
) -> float | None:
    """Return duration after applying an optional user FPS reinterpretation."""
    info = video_info if isinstance(video_info, dict) else {}
    frames = optional_finite_float(info.get("total_frames"))
    override = optional_finite_float(fps_override)
    if frames and frames > 0 and override and override > 0:
        return frames / override
    duration = optional_finite_float(info.get("duration_seconds"))
    if duration and duration > 0:
        return duration
    fps = optional_finite_float(info.get("detected_fps") or info.get("fps"))
    if frames and frames > 0 and fps and fps > 0:
        return frames / fps
    return None


def effective_roi_window(
    video_start: float | None,
    video_end: float | None,
    time_config: dict[str, Any] | None,
) -> tuple[float, float | None]:
    """Return the ROI statistics window clipped to the video's analysis window."""
    outer_start = max(0.0, float(video_start or 0.0))
    outer_end = optional_finite_float(video_end)
    cfg = time_config if isinstance(time_config, dict) else {}
    mode = str(cfg.get("mode") or "global").strip().lower()
    if mode != "precise":
        return outer_start, outer_end

    roi_start = optional_finite_float(cfg.get("start"))
    roi_end = optional_finite_float(cfg.get("end"))
    effective_start = max(outer_start, roi_start if roi_start is not None else outer_start)
    if outer_end is None:
        effective_end = roi_end
    elif roi_end is None:
        effective_end = outer_end
    else:
        effective_end = min(outer_end, roi_end)
    return effective_start, effective_end


def validate_roi_windows(
    rois: Any,
    *,
    video_start: float | None,
    video_end: float | None,
    video_label: str = "视频",
) -> list[str]:
    errors: list[str] = []
    if not isinstance(rois, list):
        return errors

    for index, roi in enumerate(rois, start=1):
        if not isinstance(roi, dict):
            continue
        cfg = roi.get("time_config")
        if not isinstance(cfg, dict):
            continue
        mode = str(cfg.get("mode") or "global").strip().lower()
        if mode == "global":
            continue
        if mode not in {"precise", "relative"}:
            errors.append(f"{video_label} ROI {index} 的时间模式无效")
            continue

        start = optional_finite_float(cfg.get("start"))
        end = optional_finite_float(cfg.get("end"))
        if cfg.get("start") not in (None, "") and start is None:
            errors.append(f"{video_label} ROI {index} 的开始时间必须是有限数字")
            continue
        if cfg.get("end") not in (None, "") and end is None:
            errors.append(f"{video_label} ROI {index} 的结束时间必须是有限数字")
            continue
        if start is not None and start < 0:
            errors.append(f"{video_label} ROI {index} 的开始时间不能小于 0")
            continue
        if end is not None and end <= 0:
            errors.append(f"{video_label} ROI {index} 的结束时间必须大于 0")
            continue
        if start is not None and end is not None and end <= start:
            errors.append(f"{video_label} ROI {index} 的结束时间必须大于开始时间")
            continue

        if mode == "precise":
            effective_start, effective_end = effective_roi_window(video_start, video_end, cfg)
            if effective_end is not None and effective_end <= effective_start:
                errors.append(f"{video_label} ROI {index} 的时间窗与视频级时间窗没有交集")
        elif mode == "relative" and video_end is not None and start is not None:
            available = max(0.0, float(video_end) - max(0.0, float(video_start or 0.0)))
            if start >= available:
                errors.append(
                    f"{video_label} ROI {index} 的相对开始时间超出视频级时间窗，无法产生统计结果"
                )
    return errors


def validate_analysis_windows(
    payload: dict[str, Any] | None,
    video_infos: list[dict[str, Any]] | None,
    *,
    session_start: float | None = None,
    session_end: float | None = None,
) -> list[str]:
    data = payload if isinstance(payload, dict) else {}
    infos = list(video_infos or [])
    raw_configs = data.get("per_video_configs")
    configs = raw_configs if isinstance(raw_configs, list) else []
    errors: list[str] = []

    def is_provided(value: Any) -> bool:
        return value is not None and not (isinstance(value, str) and not value.strip())

    for index, info in enumerate(infos):
        config = configs[index] if index < len(configs) and isinstance(configs[index], dict) else {}
        if not configs and index == 0:
            config = data
        label = f"第 {index + 1} 个视频"
        raw_start = config.get("start_sec") if "start_sec" in config else session_start
        raw_end = config.get("end_sec") if "end_sec" in config else session_end
        raw_fps = config.get("fps_override")
        start = optional_finite_float(raw_start)
        end = optional_finite_float(raw_end)
        fps_override = optional_finite_float(raw_fps)

        if is_provided(raw_start) and start is None:
            errors.append(f"{label} 的开始时间必须是有限数字")
        elif start is not None and start < 0:
            errors.append(f"{label} 的开始时间不能小于 0")
        if is_provided(raw_end) and end is None:
            errors.append(f"{label} 的结束时间必须是有限数字")
        elif end is not None and end <= 0:
            errors.append(f"{label} 的结束时间必须大于 0")
        if start is not None and end is not None and end <= start:
            errors.append(f"{label} 的结束时间必须大于开始时间")
        if is_provided(raw_fps) and fps_override is None:
            errors.append(f"{label} 的 FPS 必须是有限数字")
        elif fps_override is not None and not (0.001 <= fps_override <= 1000.0):
            errors.append(f"{label} 的 FPS 必须在 0.001 到 1000 之间")

        duration = effective_video_duration_seconds(info, fps_override)
        if duration is not None:
            if start is not None and start >= duration:
                errors.append(f"{label} 的开始时间必须小于视频时长（{duration:.3f} 秒）")
            if end is not None and end > duration + 0.001:
                errors.append(f"{label} 的结束时间不能超过视频时长（{duration:.3f} 秒）")
        if end is None:
            end = duration
        errors.extend(
            validate_roi_windows(
                config.get("rois"),
                video_start=start,
                video_end=end,
                video_label=label,
            )
        )
    return errors
