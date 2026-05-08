from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
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
        pass


def _run_with_status_observer(analysis_session: Any) -> None:
    try:
        analysis_session._run()
    finally:
        _notify_analysis_status(analysis_session)


# ==================== 分析引擎 ====================

class AnalysisSessionBase:
    """Shared lifecycle and bookkeeping for behavior-analysis sessions."""

    batch_progress_template = "Processing video {index}/{total}: {name}"
    batch_complete_template = "All {total} videos finished."
    error_template = "Analysis error: {error}"
    error_log_prefix = ""

    def start(self):
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        _notify_analysis_status(self)
        thread = threading.Thread(target=_run_with_status_observer, args=(self,), daemon=True)
        thread.start()

    def _log(self, message):
        timestamp = china_now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

    def _append_result_file(self, path, group=None):
        if not path or not os.path.exists(path):
            return
        if path not in self.result_files:
            self.result_files.append(path)
        if group and hasattr(self, 'output_groups'):
            self.output_groups.setdefault(group, [])
            if path not in self.output_groups[group]:
                self.output_groups[group].append(path)

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
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                total = len(self.video_paths)
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(self.batch_progress_template.format(
                        index=i + 1,
                        total=total,
                        name=info.get('original_name', ''),
                    ))
                    self._analyze_single_video(path, info, batch_index=i)
                    self.progress = int((i + 1) / total * 100)

                if self.is_running:
                    self.status = 'completed'
                    self.end_time = time.time()
                    self.message = self.batch_complete_template.format(total=total)
                    self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = self.error_template.format(error=str(e))
            self._log(f"{self.error_log_prefix}{self.message}")
            import traceback
            self._log(traceback.format_exc())

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()
        _notify_analysis_status(self)

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
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
