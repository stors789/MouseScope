from __future__ import annotations

import concurrent.futures
import copy
import json
import logging
import os
import queue
import sys
import threading
import time
from contextlib import contextmanager
from collections import deque
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from vision_tracker import (
    BackgroundSubtractor,
    MorphologyProcessor,
    MouseDetector,
    OpticalFlowAnalyzer,
    MouseTracker,
    draw_results,
    render_tst_activity_curve,
    render_tst_ethogram,
    create_static_background,
    create_temporal_background,
)
from open_field import (
    OpenFieldTracker,
    draw_open_field_results,
    render_open_field_heatmap,
    render_open_field_image_grid,
    render_open_field_trackmap,
    sanitize_name,
)
from novel_object import (
    NovelObjectTracker,
    draw_novel_object_results,
    render_novel_object_exploration_cumulative,
    render_novel_object_heatmap,
    render_novel_object_trackmap,
)
from elevated_plus import (
    ElevatedPlusTracker,
    draw_elevated_plus_results,
    render_elevated_plus_heatmap,
    render_elevated_plus_trackmap,
    render_elevated_plus_image_grid,
    sanitize_name as sanitize_epm_name,
)
from y_maze import (
    YMazeTracker,
    draw_y_maze_results,
    render_y_maze_heatmap,
    render_y_maze_trackmap,
    render_y_maze_image_grid,
    sanitize_name as sanitize_y_maze_name,
)
from light_dark_box import (
    LightDarkBoxTracker,
    draw_light_dark_box_results,
    render_light_dark_box_heatmap,
    render_light_dark_box_trackmap,
    render_light_dark_box_image_grid,
    sanitize_name as sanitize_light_dark_name,
)
from three_chamber import (
    ThreeChamberTracker,
    draw_three_chamber_results,
    render_three_chamber_heatmap,
    render_three_chamber_trackmap,
    render_three_chamber_image_grid,
    sanitize_name as sanitize_three_chamber_name,
)
from water_maze import (
    WaterMazeTracker,
    draw_water_maze_results,
    render_water_maze_heatmap,
    render_water_maze_trackmap,
    render_water_maze_image_grid,
    sanitize_name as sanitize_water_maze_name,
)
from forced_swim import (
    ForcedSwimTracker,
    draw_forced_swim_results,
    render_forced_swim_activity_curve,
    render_forced_swim_ethogram,
)
from services.payload_geometry import (
    coerce_point_list as _coerce_point_list,
    coerce_zone_polygons as _coerce_zone_polygons,
    coerce_zone_polygons_with_schema as _coerce_zone_polygons_with_schema,
)
from services.video_export import create_annotated_video_writer
from services.video_metadata import probe_video_info
from services.video_orientation import open_video_capture
from services.video_timebase import build_video_timebase
from services.resource_policy import batch_worker_count, current_resource_policy


logger = logging.getLogger(__name__)
_RESOURCE_POLICY = current_resource_policy()
_VIDEO_WORKER_SEMAPHORE = threading.BoundedSemaphore(_RESOURCE_POLICY.max_parallel_videos)
RESULT_DIR: Path | str | None = None
HAS_PANDAS = False

def _missing_runtime(*_args, **_kwargs):
    raise RuntimeError('Analysis runtime has not been configured.')

china_now: Callable[[], Any] = _missing_runtime
_prepare_output_paths: Callable[..., dict[str, Any]] = _missing_runtime
_export_result_file: Callable[..., str | None] = _missing_runtime
_write_dataframe_csv: Callable[..., bool] = _missing_runtime
_bi_label: Callable[[str, str], str] = _missing_runtime
_report_value: Callable[..., str] = _missing_runtime
_report_time_mode_label: Callable[[Any], str] = _missing_runtime
_report_phase_label: Callable[[Any], str] = _missing_runtime
_write_report_metric_lines: Callable[..., None] = _missing_runtime
_analysis_status_observer: Callable[[Any], None] | None = None

DETECTION_MASK_MIN_POINTS = 3
EPM_ZONE_KEYS = ('center', 'open_1', 'open_2', 'closed_1', 'closed_2')
YMAZE_ZONE_POINT_COUNTS = {'center': 3, 'arm_a': 4, 'arm_b': 4, 'arm_c': 4}

_SESSION_MODULE_NAMES = (
    'services.tst_session',
    'services.open_field_session',
    'services.forced_swim_session',
    'services.novel_object_session',
    'services.elevated_plus_session',
    'services.y_maze_session',
    'services.light_dark_session',
    'services.three_chamber_session',
    'services.water_maze_session',
    'services.barnes_session',
)

_RUNTIME_BINDING_NAMES = (
    'RESULT_DIR',
    'HAS_PANDAS',
    'china_now',
    '_prepare_output_paths',
    '_export_result_file',
    '_write_dataframe_csv',
    '_bi_label',
    '_report_value',
    '_report_time_mode_label',
    '_report_phase_label',
    '_write_report_metric_lines',
    '_analysis_status_observer',
)


def _propagate_runtime_bindings() -> None:
    for module_name in _SESSION_MODULE_NAMES:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for name in _RUNTIME_BINDING_NAMES:
            setattr(module, name, globals()[name])


def configure_analysis_runtime(**kwargs: Any) -> None:
    global RESULT_DIR, HAS_PANDAS
    global china_now, _prepare_output_paths, _export_result_file, _write_dataframe_csv
    global _bi_label, _report_value, _report_time_mode_label, _report_phase_label, _write_report_metric_lines
    global _analysis_status_observer

    RESULT_DIR = kwargs['RESULT_DIR']
    HAS_PANDAS = bool(kwargs.get('HAS_PANDAS', False))
    china_now = kwargs['china_now']
    _prepare_output_paths = kwargs['_prepare_output_paths']
    _export_result_file = kwargs['_export_result_file']
    _write_dataframe_csv = kwargs['_write_dataframe_csv']
    _bi_label = kwargs['_bi_label']
    _report_value = kwargs['_report_value']
    _report_time_mode_label = kwargs['_report_time_mode_label']
    _report_phase_label = kwargs['_report_phase_label']
    _write_report_metric_lines = kwargs['_write_report_metric_lines']
    _analysis_status_observer = kwargs.get('_analysis_status_observer')
    _propagate_runtime_bindings()


def _notify_analysis_status(analysis_session: Any) -> None:
    if _analysis_status_observer is None:
        return
    try:
        _analysis_status_observer(analysis_session)
    except Exception:
        logger.exception("Analysis status observer failed for session %s", getattr(analysis_session, "session_id", ""))


def _run_with_status_observer(analysis_session: Any) -> None:
    try:
        analysis_session._run()
    finally:
        _notify_analysis_status(analysis_session)


def _analysis_stdout_logs_enabled() -> bool:
    raw = str(os.environ.get('MOUSESCOPE_ANALYSIS_STDOUT_LOGS', '')).strip().lower()
    return raw not in {'0', 'false', 'no', 'off', 'quiet'}


# ==================== 分析引擎 ====================

class FrameCachedBackgroundSubtractor:
    """Cache repeated threshold/static masks for the same frame."""

    CACHEABLE_METHODS = {"threshold", "static", "temporal"}

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.method = getattr(wrapped, "method", None)
        self._enabled = self.method in self.CACHEABLE_METHODS
        self._frame_token: Any = None
        self._cache: dict[Any, np.ndarray] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def begin_frame(self, token: Any) -> None:
        if token != self._frame_token:
            self._frame_token = token
            self._cache.clear()

    @staticmethod
    def _array_key(array: Any) -> tuple[Any, ...]:
        if array is None or not hasattr(array, "__array_interface__"):
            return (id(array),)
        interface = array.__array_interface__
        data = interface.get("data") or (0, False)
        return (
            id(array),
            int(data[0] or 0),
            tuple(getattr(array, "shape", ())),
            tuple(getattr(array, "strides", ()) or ()),
            str(getattr(array, "dtype", "")),
        )

    def apply(self, frame, learning_rate=-1, gray=None):
        if not self._enabled:
            return self._wrapped.apply(frame, learning_rate=learning_rate, gray=gray)

        key = (
            self._frame_token,
            self._array_key(frame),
            self._array_key(gray),
            float(learning_rate or 0.0),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()

        mask = self._wrapped.apply(frame, learning_rate=learning_rate, gray=gray)
        self._cache[key] = mask.copy()
        return mask


def enable_frame_cache(bg_subtractor: Any) -> Any:
    if isinstance(bg_subtractor, FrameCachedBackgroundSubtractor):
        return bg_subtractor
    return FrameCachedBackgroundSubtractor(bg_subtractor)


def begin_frame_cache(bg_subtractor: Any, token: Any) -> None:
    if not hasattr(bg_subtractor, "begin_frame"):
        return
    try:
        bg_subtractor.begin_frame(token)
    except Exception:
        logger.debug("Frame cache begin_frame failed; continuing without cache update.", exc_info=True)


def build_standard_processors(
    video_path,
    vision_config,
    *,
    static_frames=15,
    erosion_default=1,
    dilation_default=2,
    min_area_default=120,
    analysis_size=None,
    area_scale=1.0,
    morphology_scale=1.0,
    temporal_start_frame=0,
    temporal_end_frame=None,
):
    def _scaled_area(value, default):
        try:
            base = float(value if value is not None else default)
        except Exception:
            base = float(default)
        return max(1, int(round(base * max(float(area_scale or 1.0), 1e-6))))

    def _scaled_kernel(value, default):
        try:
            base = int(value if value is not None else default)
        except Exception:
            base = int(default)
        scaled = max(1, int(round(base * max(float(morphology_scale or 1.0), 1e-6))))
        if scaled > 1 and scaled % 2 == 0:
            scaled += 1
        return scaled

    bg_method = vision_config.get('bg_method', 'threshold')
    if bg_method == 'threshold':
        bg_subtractor = BackgroundSubtractor(
            method='threshold',
            threshold=vision_config.get('gray_threshold', 127),
            invert=vision_config.get('invert', False),
        )
    elif bg_method == 'static':
        bg = create_static_background(
            video_path,
            num_frames=static_frames,
            orientation_override=vision_config.get('orientation_override'),
        )
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR) if len(bg.shape) == 2 else bg
        if analysis_size:
            target_w, target_h = int(analysis_size[0]), int(analysis_size[1])
            if target_w > 0 and target_h > 0 and (bg_rgb.shape[1], bg_rgb.shape[0]) != (target_w, target_h):
                bg_rgb = cv2.resize(bg_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
    elif bg_method == 'temporal':
        temporal_bg, temporal_noise, _ = create_temporal_background(
            video_path,
            num_samples=int(vision_config.get('temporal_bg_samples', 90)),
            start_frame=temporal_start_frame,
            end_frame=temporal_end_frame,
            orientation_override=vision_config.get('orientation_override'),
            analysis_size=analysis_size,
        )
        bg_subtractor = BackgroundSubtractor(
            method='temporal',
            threshold=int(vision_config.get('temporal_diff_threshold', 25)),
            static_bg=temporal_bg,
            temporal_noise_map=temporal_noise,
        )
    else:
        bg_subtractor = BackgroundSubtractor(method=bg_method)

    bg_subtractor = enable_frame_cache(bg_subtractor)
    morph_processor = MorphologyProcessor(
        erosion_kernel=_scaled_kernel(vision_config.get('erosion_kernel'), 3),
        erosion_iterations=vision_config.get('erosion_iter', erosion_default),
        dilation_kernel=_scaled_kernel(vision_config.get('dilation_kernel'), 5),
        dilation_iterations=vision_config.get('dilation_iter', dilation_default),
    )
    detector = MouseDetector(
        min_area=_scaled_area(vision_config.get('min_area'), min_area_default),
        max_area=_scaled_area(vision_config.get('max_area'), 50000),
    )
    return bg_subtractor, morph_processor, detector


class AnalysisSessionBase:
    """Shared lifecycle and bookkeeping for behavior-analysis sessions."""

    batch_progress_template = "Processing video {index}/{total}: {name}"
    batch_complete_template = "All {total} videos finished."
    error_template = "Analysis error: {error}"
    error_log_prefix = ""

    def _iter_prefetched_frame_batches(
        self,
        cap,
        *,
        start_frame,
        end_frame,
        sampling,
        batch_size,
        prepare_gray=True,
    ):
        prepared = queue.Queue(maxsize=2)
        stop_event = threading.Event()
        sentinel = object()

        def put(item):
            while not stop_event.is_set():
                try:
                    prepared.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce():
            batch = []
            source_frames_done = 0
            try:
                while self.is_running and not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    try:
                        current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    except Exception:
                        current_frame_no = start_frame + source_frames_done
                    if current_frame_no > end_frame:
                        break
                    analysis_frame = self._resize_for_sampling(frame, sampling)
                    batch.append(
                        {
                            'source_frame_no': current_frame_no,
                            'frame': analysis_frame,
                            'gray': (
                                cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
                                if prepare_gray
                                else None
                            ),
                        }
                    )
                    skipped = self._grab_skipped_frames(
                        cap,
                        current_frame_no,
                        end_frame,
                        sampling['frame_stride'],
                    )
                    covered = 1 + skipped
                    batch[-1]['covered_source_frames'] = covered
                    source_frames_done += covered
                    if len(batch) >= batch_size:
                        if not put(batch):
                            return
                        batch = []
                if batch:
                    put(batch)
            except BaseException as exc:
                put(exc)
            finally:
                put(sentinel)

        producer = threading.Thread(
            target=produce,
            daemon=True,
            name=f"frame-prefetch-{getattr(self, 'session_id', 'analysis')}",
        )
        producer.start()
        try:
            while True:
                item = prepared.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop_event.set()
            producer.join(timeout=5.0)

    @staticmethod
    def _truthy(value, default=True):
        if value is None:
            return bool(default)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {'0', 'false', 'no', 'off', 'disabled'}:
                return False
            if normalized in {'1', 'true', 'yes', 'on', 'enabled'}:
                return True
        return bool(value)

    def _resolve_video_sampling(
        self,
        width,
        height,
        fps,
        vision_config,
        *,
        label='Video',
        temporal_default=True,
        max_dim_default=960,
        target_fps_default=30.0,
    ):
        width = max(1, int(width or 1))
        height = max(1, int(height or 1))
        fps = max(1e-6, float(fps or 30.0))
        fast_default = self._truthy(vision_config.get('fast_scoring', True), True)
        spatial_enabled = self._truthy(vision_config.get('spatial_downsample'), fast_default)
        temporal_enabled = self._truthy(vision_config.get('temporal_downsample'), fast_default and temporal_default)

        try:
            max_dim = float(
                vision_config.get('analysis_max_dim')
                or vision_config.get('fast_analysis_max_dim')
                or vision_config.get('downsample_max_dim')
                or max_dim_default
            )
        except Exception:
            max_dim = float(max_dim_default)
        try:
            target_fps = float(
                vision_config.get('analysis_target_fps')
                or vision_config.get('fast_analysis_target_fps')
                or vision_config.get('downsample_target_fps')
                or target_fps_default
            )
        except Exception:
            target_fps = float(target_fps_default)
        try:
            max_stride = int(vision_config.get('analysis_max_frame_stride') or 4)
        except Exception:
            max_stride = 4
        max_stride = max(1, min(max_stride, 8))

        scale = 1.0
        if spatial_enabled and max_dim >= 240 and max(width, height) > max_dim:
            scale = max_dim / float(max(width, height))

        analysis_width = max(1, int(round(width * scale)))
        analysis_height = max(1, int(round(height * scale)))
        scale_x = analysis_width / float(width)
        scale_y = analysis_height / float(height)

        frame_stride = 1
        if temporal_enabled and target_fps > 0 and fps > target_fps * 1.15:
            frame_stride = max(1, int(round(fps / target_fps)))
            frame_stride = min(frame_stride, max_stride)

        effective_fps = fps / max(frame_stride, 1)
        sampling = {
            'label': label,
            'source_width': width,
            'source_height': height,
            'analysis_width': analysis_width,
            'analysis_height': analysis_height,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'scale': min(scale_x, scale_y),
            'frame_stride': frame_stride,
            'source_fps': fps,
            'effective_fps': effective_fps,
            'spatial_enabled': spatial_enabled,
            'temporal_enabled': temporal_enabled,
            'active': (analysis_width != width or analysis_height != height or frame_stride > 1),
        }
        return sampling

    def _log_video_sampling(self, sampling):
        if not sampling.get('active'):
            return
        self._log(
            f"{sampling.get('label', 'Video')} fast scoring: "
            f"{sampling['source_width']}x{sampling['source_height']}@{sampling['source_fps']:.2f}fps -> "
            f"{sampling['analysis_width']}x{sampling['analysis_height']}@{sampling['effective_fps']:.2f}fps "
            f"(scale {sampling['scale']:.3f}, stride {sampling['frame_stride']})."
        )

    def _ensure_analysis_frames_processed(self, *, video_path, label, frame_idx, source_frames_done):
        try:
            sample_count = int(frame_idx or 0)
        except Exception:
            sample_count = 0
        try:
            source_count = int(source_frames_done or 0)
        except Exception:
            source_count = 0
        if sample_count > 0 and source_count > 0:
            return
        message = (
            f"{label} frame reading failed: OpenCV opened the video but decoded 0 frames. "
            "Please check the video codec/OpenCV backend, or transcode the video to a standard H.264 MP4 and retry. "
            f"Video: {video_path}"
        )
        self._log(f"Frame decoding guard: {message}", force_console=True)
        raise RuntimeError(message)

    @classmethod
    def _scale_geometry_value(cls, value, scale_x, scale_y):
        def _scale_number(item, factor):
            try:
                return float(item) * float(factor)
            except Exception:
                return item

        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float32) * np.asarray([scale_x, scale_y], dtype=np.float32)
        if isinstance(value, (list, tuple)):
            if len(value) == 2:
                try:
                    return [float(value[0]) * scale_x, float(value[1]) * scale_y]
                except Exception:
                    pass
            return [cls._scale_geometry_value(item, scale_x, scale_y) for item in value]
        if isinstance(value, dict):
            scaled = {}
            for key, item in value.items():
                key_str = str(key)
                if key_str in {'x', 'cx', 'left', 'right'}:
                    scaled[key] = _scale_number(item, scale_x)
                elif key_str in {'y', 'cy', 'top', 'bottom'}:
                    scaled[key] = _scale_number(item, scale_y)
                elif key_str in {'radius_x_px', 'rx_px', 'width_px'}:
                    scaled[key] = _scale_number(item, scale_x)
                elif key_str in {'radius_y_px', 'ry_px', 'height_px'}:
                    scaled[key] = _scale_number(item, scale_y)
                elif key_str in {'radius_px', 'diameter_px'}:
                    scaled[key] = _scale_number(item, (float(scale_x) + float(scale_y)) * 0.5)
                elif key_str == 'radius':
                    scaled[key] = _scale_number(item, (float(scale_x) + float(scale_y)) * 0.5)
                else:
                    scaled[key] = cls._scale_geometry_value(item, scale_x, scale_y)
            return scaled
        return value

    def _scale_roi_configs_for_sampling(self, roi_configs, sampling):
        scale_x = float(sampling.get('scale_x', 1.0))
        scale_y = float(sampling.get('scale_y', 1.0))
        if abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6:
            return copy.deepcopy(roi_configs)
        return self._scale_geometry_value(copy.deepcopy(roi_configs), scale_x, scale_y)

    @staticmethod
    def _resize_for_sampling(frame, sampling):
        target = (int(sampling.get('analysis_width') or frame.shape[1]), int(sampling.get('analysis_height') or frame.shape[0]))
        if (frame.shape[1], frame.shape[0]) == target:
            return frame
        return cv2.resize(frame, target, interpolation=cv2.INTER_AREA)

    def _build_processors_for_sampling(
        self,
        video_path,
        vision_config,
        sampling,
        *,
        static_frames=15,
        erosion_default=1,
        dilation_default=2,
        min_area_default=120,
        temporal_start_frame=0,
        temporal_end_frame=None,
    ):
        return build_standard_processors(
            video_path,
            vision_config,
            static_frames=static_frames,
            erosion_default=erosion_default,
            dilation_default=dilation_default,
            min_area_default=min_area_default,
            analysis_size=(sampling['analysis_width'], sampling['analysis_height']),
            area_scale=float(sampling.get('scale_x', 1.0)) * float(sampling.get('scale_y', 1.0)),
            morphology_scale=min(float(sampling.get('scale_x', 1.0)), float(sampling.get('scale_y', 1.0))),
            temporal_start_frame=temporal_start_frame,
            temporal_end_frame=temporal_end_frame,
        )

    @staticmethod
    def _grab_skipped_frames(cap, current_frame_no, end_frame, frame_stride):
        skipped = 0
        stride = max(1, int(frame_stride or 1))
        while skipped < stride - 1:
            if current_frame_no + skipped + 1 > end_frame:
                break
            if not cap.grab():
                break
            skipped += 1
        return skipped

    @staticmethod
    def _annotate_sample_record(result, current_frame_no, sample_index, sampling):
        record = result.get('record') if isinstance(result, dict) else None
        if not isinstance(record, dict):
            return
        original_sample = record.get('frame')
        record['analysis_sample'] = original_sample if original_sample is not None else sample_index
        record['frame'] = int(current_frame_no)
        record['analysis_frame_stride'] = int(sampling.get('frame_stride', 1))
        record['analysis_spatial_scale'] = round(float(sampling.get('scale', 1.0)), 6)

    @staticmethod
    def _add_sampling_summary_fields(summary, sampling):
        summary['source_fps'] = round(float(sampling.get('source_fps', 0.0)), 6)
        summary['analysis_fps'] = round(float(sampling.get('effective_fps', 0.0)), 6)
        summary['analysis_frame_stride'] = int(sampling.get('frame_stride', 1))
        summary['analysis_spatial_scale'] = round(float(sampling.get('scale', 1.0)), 6)
        summary['analysis_width_px'] = int(sampling.get('analysis_width', 0))
        summary['analysis_height_px'] = int(sampling.get('analysis_height', 0))
        return summary

    def _build_timebase_for_video(self, video_path, fps, total_frames, *, label='Video', prefer_pts=True):
        timebase = build_video_timebase(
            video_path,
            fps=fps,
            total_frames=total_frames,
            prefer_pts=bool(prefer_pts),
            logger=self._log,
        )
        if int(total_frames or 0) <= 0 and timebase.frame_count > 0:
            total_frames = timebase.frame_count
        summary = timebase.summary()
        self._log(
            f"{label} timebase: {summary['source']}, "
            f"median dt {float(summary['median_dt']):.4f}s, "
            f"long gaps >50ms {int(summary['long_gap_gt_50ms'])}"
        )
        return timebase, summary, int(total_frames or 0)

    @staticmethod
    def _time_range_to_frames(timebase, total_frames, start_sec=None, end_sec=None):
        total_frames = int(max(0, total_frames or 0))
        start_frame = timebase.frame_at_or_after(float(start_sec)) if start_sec is not None else 0
        end_frame = timebase.frame_before(float(end_sec)) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(int(start_frame), max(total_frames - 1, 0)))
        end_frame = max(0, min(int(end_frame), max(total_frames - 1, 0)))
        if end_frame < start_frame:
            end_frame = start_frame
        return start_frame, end_frame

    @staticmethod
    def _add_timebase_summary_fields(summary, timebase_summary):
        summary['timebase_source'] = timebase_summary.get('source')
        summary['timebase_median_dt_s'] = round(float(timebase_summary.get('median_dt') or 0.0), 6)
        summary['timebase_long_gap_gt_50ms'] = int(timebase_summary.get('long_gap_gt_50ms') or 0)
        return summary

    def _lock(self):
        lock = getattr(self, "_state_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._state_lock = lock
        return lock

    @staticmethod
    def _clamp_progress_fraction(value):
        try:
            fraction = float(value or 0.0)
        except Exception:
            fraction = 0.0
        return max(0.0, min(1.0, fraction))

    @staticmethod
    def _format_progress_message(message, progress):
        if message is None:
            return None
        try:
            return str(message).format(progress=progress)
        except Exception:
            return str(message)

    def _begin_batch_progress(self, total, *, aggregate=False):
        total = max(1, int(total or 1))
        with self._lock():
            self._batch_progress_aggregate = bool(aggregate)
            self._batch_progress_total = total
            self._batch_video_progress = [0.0 for _ in range(total)] if aggregate else None
            if aggregate:
                self.progress = 0

    def _set_video_progress(self, batch_index, phase_fraction, message=None):
        phase_fraction = self._clamp_progress_fraction(phase_fraction)
        with self._lock():
            progress_slots = getattr(self, '_batch_video_progress', None)
            aggregate = bool(getattr(self, '_batch_progress_aggregate', False))
            if batch_index is not None:
                try:
                    index = int(batch_index)
                except Exception:
                    index = 0
                if aggregate and isinstance(progress_slots, list) and 0 <= index < len(progress_slots):
                    progress_slots[index] = max(float(progress_slots[index]), phase_fraction)
                    progress = int(sum(progress_slots) / max(len(progress_slots), 1) * 100)
                else:
                    total = max(1, len(getattr(self, 'video_paths', []) or []))
                    progress = int((index + phase_fraction) / total * 100)
            else:
                progress = int(phase_fraction * 100)

            progress = max(0, min(100, progress))
            self.progress = progress
            formatted = self._format_progress_message(message, progress)
            if formatted is not None:
                self.message = formatted
            return progress

    def _mark_batch_video_completed(self, batch_index, message=None):
        return self._set_video_progress(batch_index, 1.0, message)

    def start(self):
        with self._lock():
            thread = getattr(self, "_thread", None)
            if bool(getattr(self, "is_running", False)) or (thread is not None and thread.is_alive()):
                raise RuntimeError("Analysis is already running.")
            self.is_running = True
            self.status = 'analyzing'
            self.start_time = time.time()
            self.end_time = None
            self._speed_frame_counter = getattr(self, '_speed_frame_counter', None)
            self._status_sample_time = None
            self._status_sample_frames = None
            self._recent_processing_fps = 0.0
            self.config.setdefault('resource_policy', _RESOURCE_POLICY.as_dict())
            thread = threading.Thread(target=_run_with_status_observer, args=(self,), daemon=True)
            self._thread = thread
        self._log(
            "Resource policy: "
            f"{_RESOURCE_POLICY.max_parallel_videos} concurrent videos, "
            f"{_RESOURCE_POLICY.opencv_threads} OpenCV threads, "
            f"{_RESOURCE_POLICY.video_queue_target_mb} MB writer queue target."
        )
        _notify_analysis_status(self)
        thread.start()

    def _log(self, message, *, force_console=False):
        timestamp = china_now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        with self._lock():
            self.logs.append(log_entry)
        if force_console or _analysis_stdout_logs_enabled():
            try:
                print(log_entry)
            except UnicodeEncodeError:
                encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
                safe_entry = log_entry.encode(encoding, errors='replace').decode(encoding, errors='replace')
                print(safe_entry)

    def _append_result_file(self, path, group=None):
        if not path or not os.path.exists(path):
            return
        with self._lock():
            if path not in self.result_files:
                self.result_files.append(path)
            if group and hasattr(self, 'output_groups'):
                self.output_groups.setdefault(group, [])
                if path not in self.output_groups[group]:
                    self.output_groups[group].append(path)

    def _advance_frame_counters(self, count=1, *, processed=True):
        try:
            frame_count = max(0, int(count or 0))
        except Exception:
            frame_count = 0
        if frame_count <= 0:
            return
        with self._lock():
            if processed:
                self.processed_frames += frame_count
            self._speed_frame_counter = float(getattr(self, '_speed_frame_counter', 0.0) or 0.0) + frame_count

    def _config_for_video(self, video_index):
        vision_config = copy.deepcopy(self.config)
        per_video_configs = vision_config.pop('per_video_configs', None) or []
        video_index = int(video_index or 0)
        if 0 <= video_index < len(per_video_configs):
            per_video_config = per_video_configs[video_index] or {}
            for key, value in per_video_config.items():
                vision_config[key] = copy.deepcopy(value)
        return vision_config

    def _run(self):
        try:
            if len(self.video_paths) == 1:
                with self._video_worker_slot() as acquired:
                    if acquired:
                        self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                total = len(self.video_paths)
                parallel_enabled = bool(self.config.get('parallel_videos', True))
                requested_workers = self.config.get('batch_workers') or os.environ.get('MOUSESCOPE_BATCH_WORKERS')
                worker_count = batch_worker_count(total, requested_workers)
                self.config.setdefault('resource_policy', _RESOURCE_POLICY.as_dict())['batch_workers'] = worker_count
                aggregate_progress = parallel_enabled and worker_count > 1
                self._begin_batch_progress(total, aggregate=aggregate_progress)

                if parallel_enabled and worker_count > 1:
                    self._log(f"Batch parallel enabled: {worker_count} workers for {total} videos.")

                    def _worker(index, path, info):
                        if not self.is_running:
                            return
                        with self._video_worker_slot() as acquired:
                            if not acquired or not self.is_running:
                                return
                            self._log(self.batch_progress_template.format(
                                index=index + 1,
                                total=total,
                                name=info.get('original_name', ''),
                            ))
                            self._analyze_single_video(path, info, batch_index=index)

                    completed = 0
                    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix='MouseScopeBatch') as executor:
                        futures = {
                            executor.submit(_worker, i, path, info): i
                            for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos))
                        }
                        for future in concurrent.futures.as_completed(futures):
                            index = futures[future]
                            future.result()
                            completed += 1
                            if self.is_running:
                                self._mark_batch_video_completed(index)
                            if not self.is_running:
                                break
                else:
                    for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                        if not self.is_running:
                            break
                        self._log(self.batch_progress_template.format(
                            index=i + 1,
                            total=total,
                            name=info.get('original_name', ''),
                        ))
                        with self._video_worker_slot() as acquired:
                            if not acquired:
                                break
                            self._analyze_single_video(path, info, batch_index=i)
                        self.progress = int((i + 1) / total * 100)

                if self.is_running:
                    self.status = 'completed'
                    self.progress = 100
                    self.end_time = time.time()
                    self.message = self.batch_complete_template.format(total=total)
                    self._log(self.message)
        except Exception as e:
            with self._lock():
                self.status = 'error'
                self.message = self.error_template.format(error=str(e))
                self.end_time = time.time()
                self.is_running = False
            self._log(f"{self.error_log_prefix}{self.message}", force_console=True)
            import traceback
            self._log(traceback.format_exc(), force_console=True)
        else:
            with self._lock():
                status = str(getattr(self, 'status', '') or '').lower()
                if self.is_running and status not in {'completed', 'error', 'stopped', 'interrupted'}:
                    self.status = 'completed'
                    self.progress = 100
                    self.message = self.message or 'Analysis completed.'
                    self.end_time = time.time()
                    status = 'completed'
                if status in {'completed', 'error', 'stopped', 'interrupted'}:
                    self.is_running = False
                    if self.end_time is None:
                        self.end_time = time.time()
        finally:
            with self._lock():
                status = str(getattr(self, 'status', '') or '').lower()
                if status in {'completed', 'error', 'stopped', 'interrupted'}:
                    self.is_running = False
                    if self.end_time is None:
                        self.end_time = time.time()

    def stop(self):
        with self._lock():
            status = str(getattr(self, 'status', '') or '').lower()
            if status in {'completed', 'error', 'stopped', 'interrupted'}:
                return False
            thread = getattr(self, '_thread', None)
            thread_alive = bool(thread is not None and thread.is_alive())
            if not self.is_running and not thread_alive and status not in {'waiting', 'analyzing', 'preprocessing'}:
                return False
            self.is_running = False
            self.status = 'stopped'
            self.end_time = time.time()
        _notify_analysis_status(self)
        return True

    @contextmanager
    def _video_worker_slot(self):
        acquired = False
        try:
            while self.is_running and not acquired:
                acquired = _VIDEO_WORKER_SEMAPHORE.acquire(timeout=0.25)
            yield acquired
        finally:
            if acquired:
                _VIDEO_WORKER_SEMAPHORE.release()

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        speed_counter = getattr(self, '_speed_frame_counter', None)
        if speed_counter is None:
            speed_frames = float(self.processed_frames or 0.0)
        else:
            speed_frames = max(float(speed_counter or 0.0), float(self.processed_frames or 0.0))
        average_fps = speed_frames / elapsed if elapsed > 0 else 0

        now = time.time()
        previous_time = getattr(self, '_status_sample_time', None)
        previous_frames = getattr(self, '_status_sample_frames', None)
        recent_fps = float(getattr(self, '_recent_processing_fps', 0.0) or 0.0)
        if previous_time is None or previous_frames is None:
            recent_fps = average_fps
        else:
            dt = max(0.0, now - float(previous_time))
            if dt >= 0.25:
                recent_fps = max(0.0, (speed_frames - float(previous_frames)) / dt)
        self._status_sample_time = now
        self._status_sample_frames = speed_frames
        self._recent_processing_fps = recent_fps

        processing_fps = recent_fps if self.status == 'analyzing' else average_fps
        payload = {
            'session_id': self.session_id,
            'experiment_type': self.experiment_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'logs': self.logs[-10:],
            'realtime_data': self.realtime_data,
            'result_files': self.result_files,
            'stats': {
                'processed_frames': self.processed_frames,
                'elapsed_time': elapsed,
                'fps': processing_fps,
                'recent_fps': recent_fps,
                'average_fps': average_fps,
                'speed_frames': int(speed_frames),
            },
        }
        if hasattr(self, 'live_metrics'):
            payload['experiment_stats'] = self.live_metrics
        return payload

    def get_download_file_map(self):
        if hasattr(self, 'output_groups'):
            return self.output_groups
        return {
            'video': [f for f in self.result_files if f.endswith('.mp4')],
        }


__all__ = [name for name in globals() if not (name.startswith('__') and name.endswith('__'))]
