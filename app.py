#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MouseScope Behavior Analysis Platform — 简化版 Flask 应用
去除：用户认证、数据库、YOLO、多语言
保留：视频上传、ROI标定、传统CV分析、结果下载
"""

import os
import sys
import stringprep  # Ensure IDNA codec dependencies are loaded in frozen builds.

try:
    import encodings.idna  # noqa: F401
except Exception:
    encodings = None

def _configure_console_streams():
    """Avoid startup crashes when Windows console encoding cannot represent log symbols."""
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, 'reconfigure'):
            continue
        try:
            stream.reconfigure(errors='replace')
        except Exception:
            pass


_configure_console_streams()

# ==================== 性能环境变量（必须在 import numpy/cv2 之前） ====================
# 让底层 BLAS/OpenMP 库充分利用所有 CPU 核心
_cpu_count = str(os.cpu_count() or 4)
os.environ.setdefault('OMP_NUM_THREADS', _cpu_count)
os.environ.setdefault('OPENBLAS_NUM_THREADS', _cpu_count)
os.environ.setdefault('MKL_NUM_THREADS', _cpu_count)

import secrets
import json
import copy
import cv2
import numpy as np
import base64
import threading
import time
import shutil
import zipfile
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import deque
import subprocess
import platform
from urllib.parse import urlsplit
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, session
)

# 将 core/ 加入路径（兼容 PyInstaller 打包环境）
if getattr(sys, 'frozen', False):
    # 打包后：core/ 在 sys._MEIPASS 解压目录下
    _base = sys._MEIPASS
    sys.path.insert(0, os.path.join(_base, 'core'))
    sys.path.insert(0, _base)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from vision_tracker import (
    BackgroundSubtractor,
    MorphologyProcessor,
    MouseDetector,
    OpticalFlowAnalyzer,
    MouseTracker,
    draw_results,
    create_static_background
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
from file_utils import (
    build_ascii_file_name,
    clean_display_name,
    clean_display_stem,
    contains_non_ascii,
    normalized_path_key,
)
from performance_utils import enable_opencv_optimizations
from services.session_store import SessionStore
from services.experiment_registry import (
    get_experiment_spec,
    get_manual_module,
    list_experiments,
    normalize_experiment_type,
    summarize_batch_state,
)
from services.analysis_builders import build_analysis_config

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠ pandas 未安装，将无法生成 CSV 报告")

# ==================== 配置 ====================
# PyInstaller 打包兼容：资源文件路径
if getattr(sys, 'frozen', False):
    # 打包后的 exe 运行环境
    BUNDLE_DIR = Path(sys._MEIPASS)          # 资源解压目录（templates, static 等）
    BASE_DIR = Path(sys.executable).parent   # exe 所在目录（用于存放用户数据）
else:
    BUNDLE_DIR = Path(__file__).parent
    BASE_DIR = BUNDLE_DIR

DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = DATA_DIR / 'uploads'
RESULT_DIR = DATA_DIR / 'results'
WORK_DIR = RESULT_DIR / '_work'

for d in [DATA_DIR, UPLOAD_DIR, RESULT_DIR, WORK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHINA_TZ = timezone(timedelta(hours=8))
CSV_ENCODING = 'utf-8-sig'
VALID_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
DETECTION_MASK_MIN_POINTS = 3
EPM_ZONE_KEYS = ('center', 'open_1', 'open_2', 'closed_1', 'closed_2')
YMAZE_ZONE_KEYS = ('center', 'arm_a', 'arm_b', 'arm_c')
YMAZE_ZONE_POINT_COUNTS = {'center': 3, 'arm_a': 4, 'arm_b': 4, 'arm_c': 4}
staged_original_names = {}
SESSION_PENDING_TTL_SECONDS = 2 * 60 * 60
SESSION_FINISHED_TTL_SECONDS = 12 * 60 * 60
SESSION_ERROR_TTL_SECONDS = 24 * 60 * 60
DEBUG_STATIC_BG_CACHE_TTL_SECONDS = 30 * 60
DEBUG_STATIC_BG_CACHE_MAX_ITEMS = 32
LICENSE_EXEMPT_ENDPOINTS = {
    'static',
    'index',
    'upload_page',
    'manual',
    'manual_open_field',
    'manual_novel_object',
    'license_page',
    'api_get_license',
    'api_accept_license',
}

def china_now():
    return datetime.now(CHINA_TZ)


def _now_ts():
    return time.time()


def _bi_label(zh_text, en_text):
    return f"{zh_text} ({en_text})"


def _report_value(value, suffix=''):
    if value is None or value == '':
        return '-'
    return f"{value}{suffix}"


def _report_time_mode_label(mode):
    labels = {
        'global': ('跟随全局', 'Global'),
        'relative': ('相对检测时间', 'Relative Detection Time'),
        'precise': ('精确时间段', 'Precise Time Range'),
    }
    zh_text, en_text = labels.get(str(mode or 'global').lower(), ('时间模式', str(mode or 'global')))
    return _bi_label(zh_text, en_text)


def _report_phase_label(phase):
    value = str(phase or 'recognition').strip().lower()
    labels = {
        'recognition': ('识别期', 'Recognition'),
        'training': ('熟悉期', 'Familiarization'),
        'familiarization': ('熟悉期', 'Familiarization'),
        'sample': ('熟悉期', 'Familiarization'),
    }
    zh_text, en_text = labels.get(value, ('阶段', value or 'recognition'))
    return _bi_label(zh_text, en_text)


def _write_report_metric_lines(handle, summary, metric_specs, indent='  '):
    for zh_text, en_text, key, suffix in metric_specs:
        handle.write(f"{indent}{_bi_label(zh_text, en_text)}: {_report_value(summary.get(key), suffix)}\n")


def _register_original_name(path, original_name):
    staged_original_names[normalized_path_key(path)] = clean_display_name(original_name, fallback='video')


def _get_original_name_for_path(path):
    return staged_original_names.get(normalized_path_key(path))


def _safe_display_stem(original_name, fallback='video'):
    return clean_display_stem(original_name, fallback=fallback)


def _build_ascii_result_stem(original_name, session_id, fallback='video'):
    display_stem = _safe_display_stem(original_name, fallback=fallback)
    digest = hashlib.sha1(str(original_name or display_stem).encode('utf-8', 'ignore')).hexdigest()[:10]
    stem = re.sub(r'[^0-9A-Za-z._-]+', '_', display_stem.encode('ascii', 'ignore').decode('ascii')).strip('._-')
    stem = (stem or fallback)[:48]
    session_prefix = re.sub(r'[^0-9A-Za-z]+', '', str(session_id or ''))[:12] or 'session'
    return f"{session_prefix}_{stem}_{digest}"


def _build_batch_video_subdir(original_name, batch_index, batch_count, fallback='video'):
    total = max(int(batch_count or 1), 1)
    index = max(int(batch_index or 0), 0)
    digits = max(2, len(str(total)))
    display_stem = _safe_display_stem(original_name, fallback=fallback)
    return f"{index + 1:0{digits}d}_{display_stem}"


def _prepare_output_paths(session_id, original_name, suffix, result_folder=None, fallback='video', batch_index=None, batch_count=1):
    final_dir = Path(_resolve_result_dir(result_folder))
    if int(batch_count or 1) > 1 and batch_index is not None:
        final_dir = final_dir / _build_batch_video_subdir(
            original_name,
            batch_index=batch_index,
            batch_count=batch_count,
            fallback=fallback,
        )
    final_dir.mkdir(parents=True, exist_ok=True)
    work_dir = WORK_DIR / str(session_id)
    work_dir.mkdir(parents=True, exist_ok=True)

    display_stem = _safe_display_stem(original_name, fallback=fallback)
    ascii_stem = _build_ascii_result_stem(original_name, session_id=session_id, fallback=fallback)
    return {
        'final_dir': final_dir,
        'work_dir': work_dir,
        'display_stem': display_stem,
        'work_base': str(work_dir / f"{ascii_stem}_{suffix}"),
        'final_base': str(final_dir / f"{display_stem}_{suffix}"),
    }


def _result_zip_arcname(path, result_dir):
    path_abs = os.path.abspath(str(path))
    result_dir_abs = os.path.abspath(str(result_dir))
    try:
        common = os.path.commonpath([path_abs, result_dir_abs])
    except ValueError:
        common = None
    if common and normalized_path_key(common) == normalized_path_key(result_dir_abs):
        return os.path.relpath(path_abs, result_dir_abs).replace('\\', '/')
    return os.path.basename(path_abs)


def _export_result_file(work_path, final_path):
    if not work_path or not os.path.exists(work_path):
        return None

    final_path = str(final_path)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    if normalized_path_key(work_path) != normalized_path_key(final_path):
        shutil.copy2(work_path, final_path)
    return final_path


def _write_dataframe_csv(rows, path):
    if not HAS_PANDAS:
        return False
    pd.DataFrame(rows).to_csv(path, index=False, encoding=CSV_ENCODING)
    return os.path.exists(path)


def _probe_video_info(video_path, fps_override=None):
    cap = cv2.VideoCapture(str(video_path))
    detected_fps = 30.0
    width = 0
    height = 0
    total_frames = 0
    if cap.isOpened():
        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {
        'fps': float(fps_override) if fps_override else float(detected_fps),
        'detected_fps': float(detected_fps),
        'width': int(width),
        'height': int(height),
        'total_frames': int(total_frames),
    }


def _extract_video_info_from_entry(video_info, fps_override=None):
    info = dict(video_info or {})
    fallback = _probe_video_info(info.get('path', ''), fps_override=fps_override)
    return {
        'fps': float(info.get('fps', fallback['fps'])),
        'detected_fps': float(info.get('detected_fps', fallback['detected_fps'])),
        'width': int(info.get('width', fallback['width'])),
        'height': int(info.get('height', fallback['height'])),
        'total_frames': int(info.get('total_frames', fallback['total_frames'])),
    }


def _update_video_entry_info(video_info, fps_override=None):
    video_info.update(_probe_video_info(video_info.get('path', ''), fps_override=fps_override))
    return video_info


def _resolve_result_dir(output_dir):
    if not output_dir:
        return str(RESULT_DIR)

    raw = os.path.expandvars(str(output_dir).strip())
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


def _save_uploaded_video(file_storage, prefix):
    original_name = clean_display_name(file_storage.filename, fallback='video.mp4')
    staged_dir = UPLOAD_DIR / '_staged' / str(prefix)
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_name = build_ascii_file_name(original_name, prefix=str(prefix), fallback='video')
    save_path = staged_dir / staged_name
    file_storage.save(str(save_path))
    _register_original_name(save_path, original_name)
    return str(save_path), original_name


def _stage_local_video_if_needed(path, session_id, original_name=None):
    source = Path(path).expanduser()
    display_name = clean_display_name(original_name or _get_original_name_for_path(source) or source.name, fallback='video')
    source_path = str(source)
    if not contains_non_ascii(source_path):
        return source_path, display_name

    staged_dir = UPLOAD_DIR / '_staged' / str(session_id)
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_name = build_ascii_file_name(display_name, prefix=str(session_id), fallback='video')
    staged_path = staged_dir / staged_name
    if not staged_path.exists() or staged_path.stat().st_size != source.stat().st_size:
        shutil.copy2(source, staged_path)
    _register_original_name(staged_path, display_name)
    return str(staged_path), display_name

# ==================== 性能优化 ====================
enable_opencv_optimizations()

# ==================== Flask 应用 ====================
app = Flask(__name__,
            template_folder=str(BUNDLE_DIR / 'templates'),
            static_folder=str(BUNDLE_DIR / 'static'))
app.secret_key = secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

# 会话存储（运行时 + 磁盘快照）
analysis_sessions = SessionStore(
    upload_dir=UPLOAD_DIR,
    work_dir=WORK_DIR,
    pending_ttl_seconds=SESSION_PENDING_TTL_SECONDS,
    finished_ttl_seconds=SESSION_FINISHED_TTL_SECONDS,
    error_ttl_seconds=SESSION_ERROR_TTL_SECONDS,
    now_fn=_now_ts,
)
debug_static_bg_cache = {}


def _experiment_type_to_manual_module(experiment_type):
    return get_manual_module(experiment_type)


def _safe_internal_return_to(value):
    raw = str(value or '').strip()
    if not raw:
        return None

    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        return None

    path = parsed.path or ''
    if not path.startswith('/') or path.startswith('//'):
        return None

    safe = path
    if parsed.query:
        safe += f'?{parsed.query}'
    if parsed.fragment:
        safe += f'#{parsed.fragment}'
    return safe


def _current_request_target():
    query = request.query_string.decode('utf-8', 'ignore') if request.query_string else ''
    return f'{request.path}?{query}' if query else request.path


def _request_session_id():
    view_args = getattr(request, 'view_args', None) or {}
    return view_args.get('session_id')


def _request_experiment_type():
    session_id = _request_session_id()
    if not session_id:
        return None

    sess = analysis_sessions.get(session_id) or {}
    if sess.get('experiment_type'):
        return sess.get('experiment_type')

    analysis = sess.get('analysis')
    if analysis is not None:
        return getattr(analysis, 'experiment_type', None)
    return None


def _session_video_info(sess, video_idx=0):
    video_infos = sess.get('video_infos') or []
    if not video_infos:
        return sess.get('video_info') or {
            'fps': 30.0,
            'detected_fps': 30.0,
            'width': 0,
            'height': 0,
            'total_frames': 0,
        }
    index = min(max(int(video_idx or 0), 0), len(video_infos) - 1)
    return _extract_video_info_from_entry(video_infos[index], fps_override=sess.get('fps_override'))


def _mark_session_accessed(sess):
    if isinstance(sess, dict):
        sess['last_accessed_at'] = _now_ts()


def _cleanup_session_artifacts(session_id):
    for target in (
        UPLOAD_DIR / '_staged' / str(session_id),
        WORK_DIR / str(session_id),
    ):
        try:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
        except Exception:
            pass


def _cleanup_expired_runtime_state():
    now = _now_ts()

    expired_session_ids = []
    for session_id, sess in list(analysis_sessions.items()):
        last_access = float(sess.get('last_accessed_at') or sess.get('created_at') or now)
        analysis = sess.get('analysis')
        if analysis is None:
            ttl = SESSION_PENDING_TTL_SECONDS
        else:
            status = str(getattr(analysis, 'status', '') or '').lower()
            is_running = bool(getattr(analysis, 'is_running', False))
            if is_running or status in {'waiting', 'analyzing'}:
                continue
            ttl = SESSION_FINISHED_TTL_SECONDS if status in {'completed', 'stopped'} else SESSION_ERROR_TTL_SECONDS

        if now - last_access > ttl:
            expired_session_ids.append(session_id)

    for session_id in expired_session_ids:
        analysis_sessions.pop(session_id, None)
        _cleanup_session_artifacts(session_id)

    for cache_key, entry in list(debug_static_bg_cache.items()):
        if isinstance(entry, dict):
            last_access = float(entry.get('last_accessed_at') or now)
        else:
            last_access = now
        if now - last_access > DEBUG_STATIC_BG_CACHE_TTL_SECONDS:
            debug_static_bg_cache.pop(cache_key, None)

    if len(debug_static_bg_cache) > DEBUG_STATIC_BG_CACHE_MAX_ITEMS:
        ordered = sorted(
            debug_static_bg_cache.items(),
            key=lambda item: float(item[1].get('last_accessed_at', 0.0)) if isinstance(item[1], dict) else 0.0,
        )
        overflow = len(debug_static_bg_cache) - DEBUG_STATIC_BG_CACHE_MAX_ITEMS
        for cache_key, _ in ordered[:overflow]:
            debug_static_bg_cache.pop(cache_key, None)


def _license_not_accepted_response():
    message = 'License agreement must be accepted before using MouseScope.'
    if request.path.startswith('/api/') or request.method != 'GET':
        return jsonify({'success': False, 'error': message}), 403
    flash(message, 'warning')
    return redirect(url_for('license_page'))


@app.before_request
def before_request_guard():
    _cleanup_expired_runtime_state()

    session_id = _request_session_id()
    if session_id:
        sess = analysis_sessions.get(session_id)
        if sess:
            _mark_session_accessed(sess)

    endpoint = request.endpoint or ''
    if endpoint in LICENSE_EXEMPT_ENDPOINTS or endpoint.startswith('static'):
        return None
    if session.get('license_accepted'):
        return None
    return _license_not_accepted_response()


@app.context_processor
def inject_navigation_context():
    endpoint = request.endpoint or ''
    manual_return_to = _safe_internal_return_to(request.args.get('return_to'))
    valid_manual_modules = {spec.manual_module for spec in list_experiments()}

    if endpoint == 'manual':
        manual_module = str(request.args.get('module', 'tst')).strip().lower()
        if manual_module not in valid_manual_modules:
            manual_module = 'tst'
        manual_nav_target = None
        manual_nav_rel = None
        if manual_return_to:
            manual_nav_url = url_for('manual', module=manual_module, return_to=manual_return_to)
        else:
            manual_nav_url = url_for('manual', module=manual_module)
    else:
        manual_module = _experiment_type_to_manual_module(_request_experiment_type())
        current_target = _safe_internal_return_to(_current_request_target())
        if current_target:
            manual_nav_url = url_for('manual', module=manual_module, return_to=current_target)
        else:
            manual_nav_url = url_for('manual', module=manual_module)
        manual_nav_target = '_blank' if endpoint in {'calibrate', 'monitor'} else None
        manual_nav_rel = 'noopener noreferrer' if manual_nav_target else None

    return {
        'manual_nav_url': manual_nav_url,
        'manual_nav_target': manual_nav_target,
        'manual_nav_rel': manual_nav_rel,
        'manual_return_to': manual_return_to,
    }


def _coerce_point_list(points, min_points=4):
    parsed = []
    for point in points or []:
        if isinstance(point, dict):
            parsed.append([float(point['x']), float(point['y'])])
        else:
            parsed.append([float(point[0]), float(point[1])])
    if len(parsed) < min_points:
        raise ValueError(f'Polygon requires at least {min_points} points.')
    return parsed


def _sort_polygon_points(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError('Polygon points must be Nx2.')
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    order = np.argsort(angles)
    return pts[order].astype(np.float32).tolist()


def _coerce_quad_points(points):
    parsed = _coerce_point_list(points, min_points=4)
    if len(parsed) != 4:
        raise ValueError('Zone requires exactly 4 points.')
    return _sort_polygon_points(parsed)


def _coerce_triangle_points(points):
    parsed = _coerce_point_list(points, min_points=3)
    if len(parsed) != 3:
        raise ValueError('Zone requires exactly 3 points.')
    return _sort_polygon_points(parsed)


def _coerce_zone_polygons(raw_zones, required_keys, experiment_name):
    if not isinstance(raw_zones, dict):
        raise ValueError(f'{experiment_name} ROI zones must be an object.')
    parsed = {}
    for key in required_keys:
        parsed[key] = _coerce_quad_points(raw_zones.get(key))
    return parsed


def _coerce_zone_polygons_with_schema(raw_zones, zone_point_counts, experiment_name):
    if not isinstance(raw_zones, dict):
        raise ValueError(f'{experiment_name} ROI zones must be an object.')
    parsed = {}
    for key, expected_points in zone_point_counts.items():
        expected_points = int(expected_points)
        if int(expected_points) == 3:
            try:
                parsed[key] = _coerce_triangle_points(raw_zones.get(key))
            except Exception as exc:
                raise ValueError(f'Zone "{key}" requires exactly {expected_points} points.') from exc
            continue
        try:
            parsed[key] = _coerce_quad_points(raw_zones.get(key))
        except Exception as exc:
            raise ValueError(f'Zone "{key}" requires exactly {expected_points} points.') from exc
    return parsed


def _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert):
    if bg_method == 'threshold':
        return BackgroundSubtractor(method='threshold', threshold=gray_threshold, invert=invert)
    if bg_method == 'static':
        cache_key = os.path.abspath(video_path)
        cache_entry = debug_static_bg_cache.get(cache_key)
        if isinstance(cache_entry, dict):
            static_bg = cache_entry.get('bg')
        else:
            static_bg = cache_entry
        if static_bg is None:
            static_bg = create_static_background(video_path, num_frames=15)
        debug_static_bg_cache[cache_key] = {
            'bg': static_bg,
            'last_accessed_at': _now_ts(),
        }
        bg_rgb = cv2.cvtColor(static_bg, cv2.COLOR_GRAY2BGR) if len(static_bg.shape) == 2 else static_bg
        return BackgroundSubtractor(method='static', static_bg=bg_rgb)
    return BackgroundSubtractor(method=bg_method)

# ==================== Jinja2 过滤器 ====================
@app.template_filter('filesizeformat')
def filesizeformat(value):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"

@app.template_filter('basename')
def basename_filter(path):
    """获取文件名"""
    return os.path.basename(path)


# ==================== 分析引擎 ====================
class VisionAnalysisSession:
    """视觉分析会话（无数据库、无YOLO）"""

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths  # list of paths
        self.video_infos = video_infos  # list of {path, original_name, ...}
        self.config = config
        self.experiment_type = 'tst'

        self.status = 'waiting'
        self.progress = 0
        self.message = '等待开始...'
        self.logs = []
        self.realtime_data = []
        self.result_files = []
        self.processed_frames = 0
        self.is_running = False
        self.start_time = None
        self.end_time = None
        self.current_frame = None
        self.csv_path = None
        self.immobility_csv_path = None
        self.immobility_txt_path = None

    def _config_for_video(self, video_index):
        vision_config = copy.deepcopy(self.config)
        per_video_configs = vision_config.pop('per_video_configs', None) or []
        video_index = int(video_index or 0)
        if 0 <= video_index < len(per_video_configs):
            per_video_config = per_video_configs[video_index] or {}
            for key, value in per_video_config.items():
                vision_config[key] = copy.deepcopy(value)
        return vision_config

    def start(self):
        """在后台线程中启动分析"""
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        """分析主循环"""
        try:
            if len(self.video_paths) == 1:
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                # 批量处理
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(f"正在处理视频 {i+1}/{len(self.video_paths)}: {info.get('original_name', '')}")
                    self._analyze_single_video(path, info, batch_index=i)
                    base_progress = int((i + 1) / len(self.video_paths) * 100)
                    self.progress = base_progress

                self.status = 'completed'
                self.end_time = time.time()
                self.message = f'全部 {len(self.video_paths)} 个视频分析完成！'
                self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = f'分析出错: {str(e)}'
            self._log(f"❌ {self.message}")
            import traceback
            self._log(traceback.format_exc())

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        """分析单个视频"""
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)

        # 结果文件夹（优先使用自定义输出目录）
        result_folder = vision_config.get('output_dir') or str(RESULT_DIR)
        output_info = _prepare_output_paths(
            self.session_id,
            video_info.get('original_name', 'video'),
            suffix='tst',
            result_folder=result_folder,
            fallback='video',
            batch_index=video_index,
            batch_count=len(self.video_paths),
        )

        video_name = output_info['display_stem']
        output_path = f"{output_info['work_base']}_analyzed.mp4"
        final_output_path = f"{output_info['final_base']}_analyzed.mp4"

        generate_video = vision_config.get('generate_video', True)

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"❌ 无法打开视频: {video_path}")
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps_override:
            self._log(f"视频信息: {width}x{height}, {fps:.1f}fps (手动指定, OpenCV检测={detected_fps:.1f}), {total_frames}帧")
        else:
            self._log(f"视频信息: {width}x{height}, {fps:.1f}fps, {total_frames}帧")

        # 创建处理器
        bg_method = vision_config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=vision_config.get('gray_threshold', 127),
                invert=vision_config.get('invert', False)
            )
        elif bg_method == 'static':
            try:
                bg = create_static_background(video_path, num_frames=10)
                bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
                bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
            except:
                bg_subtractor = BackgroundSubtractor(method='threshold', threshold=127)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=vision_config.get('erosion_kernel', 3),
            erosion_iterations=vision_config.get('erosion_iter', 2),
            dilation_kernel=vision_config.get('dilation_kernel', 5),
            dilation_iterations=vision_config.get('dilation_iter', 3)
        )

        detector = MouseDetector(
            min_area=vision_config.get('min_area', 100),
            max_area=50000
        )

        pixel_to_mm = vision_config.get('pixel_to_mm', 1.0)

        # 创建追踪器
        trackers = []
        rois_config = vision_config.get('rois', [])
        if isinstance(rois_config, str):
            try:
                rois_config = json.loads(rois_config)
            except:
                rois_config = []

        for roi_idx, roi_item in enumerate(rois_config):
            if isinstance(roi_item, str):
                try:
                    roi_item = json.loads(roi_item)
                except:
                    continue

            if isinstance(roi_item, dict) and 'points' in roi_item:
                roi_points = roi_item['points']
                roi_name = roi_item.get('name', f"ROI {roi_idx+1}")
                time_config = roi_item.get('time_config')
            else:
                roi_points = roi_item
                roi_name = f"ROI {roi_idx+1}"
                time_config = None

            if isinstance(roi_points, str):
                try:
                    roi_points = json.loads(roi_points)
                except:
                    continue

            roi_polygon = [[p['x'], p['y']] for p in roi_points]

            flow_analyzer = OpticalFlowAnalyzer(fps=fps, pixel_to_mm=pixel_to_mm)

            tracker = MouseTracker(
                roi_polygon=roi_polygon,
                roi_id=roi_idx + 1,
                bg_subtractor=bg_subtractor,
                morphology_processor=morph_processor,
                detector=detector,
                flow_analyzer=flow_analyzer,
                use_yolo_seg=False,
                fps=fps,
                pixel_to_mm=pixel_to_mm,
                roi_name=roi_name,
                time_config=time_config,
                win_seconds=0.3,
                flow_thr_hi=50.0,
                flow_thr_lo=35.0,
                speed_std_thr_hi=40.0,
                speed_std_thr_lo=20.0,
                min_active_seconds=0.2,
                calm_seconds=0.4,
                coherence_thr=0.8,
                coherence_calm_seconds=0.5,
            )
            trackers.append(tracker)

        if not trackers:
            self._log("❌ 没有有效的 ROI 配置")
            return

        self._log(f"创建了 {len(trackers)} 个追踪器")

        # 视频编码
        writer = None
        if generate_video:
            codecs_to_try = [
                ('avc1', 'H264'), ('H264', 'H264'), ('X264', 'X264'), ('mp4v', 'MPEG-4'),
            ]
            for codec, codec_name in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if test_writer.isOpened():
                        writer = test_writer
                        self._log(f"视频编码器: {codec_name}")
                        break
                    else:
                        test_writer.release()
                except:
                    pass

        # 时间范围
        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"从第 {start_frame} 帧开始分析 ({start_frame/fps:.1f}秒)")

        frames_to_process = end_frame - start_frame + 1

        # 优化视频读取缓冲区大小
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        # 异步视频写入（独立线程，避免编码阻塞分析）
        from queue import Queue
        write_queue = Queue(maxsize=120)  # 最多缓存120帧
        write_done = threading.Event()

        def _video_writer_thread():
            """后台线程：从队列取帧并写入视频文件"""
            while True:
                item = write_queue.get()
                if item is None:  # 结束信号
                    break
                if writer is not None:
                    writer.write(item)
            write_done.set()

        writer_thread = None
        if generate_video and writer is not None:
            writer_thread = threading.Thread(target=_video_writer_thread, daemon=True)
            writer_thread.start()

        # 视频写入缓冲区（用于摆动状态回溯修正）
        buffer_seconds = 2.0
        buffer_size = int(fps * buffer_seconds)
        video_buffer = deque(maxlen=buffer_size + 10)
        prev_swing_states = {}

        frame_idx = 0
        logs = []

        self._log(f"开始分析... (CPU核心: {os.cpu_count()}, OpenCV线程: {cv2.getNumThreads()})")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except:
                current_frame_no = start_frame + frame_idx

            if end_frame is not None and current_frame_no > end_frame:
                break

            current_time = current_frame_no / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 追踪每个 ROI
            results = []
            for tracker in trackers:
                result = tracker.update(frame, current_time=current_time, gray=gray)
                results.append(result)

            # 记录日志
            for result in results:
                if result.get('detected'):
                    log_entry = {
                        'frame': current_frame_no,
                        'time': round(current_time, 3),
                        'roi_id': result['roi_id'],
                        'roi_name': result.get('roi_name', ''),
                        'is_struggling': result.get('is_struggling', False),
                    }
                    if result.get('flow'):
                        log_entry['flow_p90_mm_s'] = round(result['flow'].get('flow_p90_mm_s', 0), 2)
                    logs.append(log_entry)

            # 实时数据更新（每50帧）
            if frame_idx % 50 == 0:
                self.realtime_data = []
                for result in results:
                    if result.get('detected'):
                        self.realtime_data.append({
                            'roi_id': result['roi_id'],
                            'roi_name': result.get('roi_name', ''),
                            'is_struggling': result.get('is_struggling', False),
                            'flow_value': result.get('flow', {}).get('flow_p90_mm_s', 0) if result.get('flow') else 0,
                        })

            # 处理摆动状态
            for result in results:
                roi_id = result['roi_id']
                current_swing = result.get('swing_state', {}).get('is_confirmed_swing', False)
                prev_swing = prev_swing_states.get(roi_id, False)

                if current_swing and not prev_swing:
                    # 摆动开始 — 回溯修正缓冲区
                    for buf_item in video_buffer:
                        for r in buf_item['results']:
                            if r['roi_id'] == roi_id:
                                r['is_swinging'] = True
                                r['is_struggling'] = False

                prev_swing_states[roi_id] = current_swing

            # 绘制结果 & 异步写入
            if generate_video:
                result_frame = draw_results(frame.copy(), results)
                video_buffer.append({'frame': result_frame, 'results': results})

                if len(video_buffer) >= buffer_size:
                    oldest = video_buffer.popleft()
                    # 非阻塞写入：放入队列由后台线程处理
                    write_queue.put(oldest['frame'])

            # 更新进度
            frame_idx += 1
            self.processed_frames = frame_idx

            if frame_idx % 30 == 0:
                if batch_index is not None:
                    batch_progress = int(batch_index / len(self.video_paths) * 100)
                    video_progress = int(frame_idx / max(frames_to_process, 1) * 100 / len(self.video_paths))
                    self.progress = batch_progress + video_progress
                else:
                    self.progress = int(frame_idx / max(frames_to_process, 1) * 100)
                self.message = f'帧 {frame_idx}/{frames_to_process} ({self.progress}%)'

            # 保存当前帧（用于实时预览）
            if frame_idx % 10 == 0:
                self.current_frame = frame

        # 刷新缓冲区中剩余的帧
        if generate_video:
            while len(video_buffer) > 0:
                item = video_buffer.popleft()
                write_queue.put(item['frame'])

        # 通知写入线程结束并等待完成
        if writer_thread is not None:
            write_queue.put(None)  # 发送结束信号
            writer_thread.join(timeout=30)

        cap.release()
        if writer is not None:
            writer.release()

        # 过滤空ROI
        valid_trackers = []
        for tracker in trackers:
            detection_rate = tracker.detected_count / max(frame_idx, 1)
            if detection_rate >= 0.01 or tracker.detected_count > 30:
                valid_trackers.append(tracker)
                self._log(f"ROI {tracker.roi_id} ({tracker.roi_name}): 检测 {tracker.detected_count}/{frame_idx} 帧 ({detection_rate*100:.1f}%) ✓")

        if not valid_trackers and trackers:
            valid_trackers = trackers

        # 保存 CSV
        if len(logs) > 0 and HAS_PANDAS:
            csv_work_path = output_path.replace('.mp4', '_logs.csv')
            if _write_dataframe_csv(logs, csv_work_path):
                csv_path = _export_result_file(csv_work_path, f"{output_info['final_base']}_logs.csv")
                if csv_path:
                    self.result_files.append(csv_path)
                    self.csv_path = csv_path
                    self._log(f"日志已保存: {csv_path}")

        # 保存不动时间统计
        immobility_summaries = []
        for tracker in valid_trackers:
            summary = tracker.get_immobility_summary(fps)
            if summary:
                immobility_summaries.append(summary)

        if len(immobility_summaries) > 0 and HAS_PANDAS:
            immobility_csv_work = output_path.replace('.mp4', '_immobility.csv')
            if _write_dataframe_csv(immobility_summaries, immobility_csv_work):
                immobility_csv = _export_result_file(
                    immobility_csv_work,
                    f"{output_info['final_base']}_immobility.csv",
                )
                if immobility_csv:
                    self.result_files.append(immobility_csv)
                    self.immobility_csv_path = immobility_csv

            txt_work_path = output_path.replace('.mp4', '_report.txt')
            self._generate_txt_report(immobility_summaries, txt_work_path)
            txt_path = _export_result_file(txt_work_path, f"{output_info['final_base']}_report.txt")
            if txt_path:
                self.result_files.append(txt_path)
                self.immobility_txt_path = txt_path

        # 保存分析视频
        if generate_video and os.path.exists(output_path):
            exported_video = _export_result_file(output_path, final_output_path)
            if exported_video:
                file_size = os.path.getsize(exported_video) / (1024 * 1024)
                self._log(f"分析视频已保存: {exported_video} ({file_size:.1f} MB)")
                self.result_files.append(exported_video)

        if batch_index is None:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = '分析完成！'

        elapsed = time.time() - video_start_time
        self._log(f"视频处理完成，耗时: {elapsed:.1f}秒，处理 {frame_idx} 帧")

        import gc
        gc.collect()

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        return {
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
            }
        }

    def get_download_file_map(self):
        return {
            'video': [f for f in self.result_files if f.endswith('.mp4')],
            'csv': [self.csv_path] if self.csv_path else [],
            'immobility_csv': [self.immobility_csv_path] if self.immobility_csv_path else [],
            'report': [self.immobility_txt_path] if self.immobility_txt_path else [],
        }

    def _log(self, message):
        timestamp = china_now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

    def _generate_txt_report(self, immobility_summaries, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 72 + "\n")
                f.write("MouseScope 悬尾实验报告 (Tail Suspension Test Report)\n")
                f.write("=" * 72 + "\n\n")
                f.write(f"{_bi_label('分析时间', 'Analysis Time')}: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{_bi_label('ROI 数量', 'ROI Count')}: {len(immobility_summaries)}\n\n")
                for i, summary in enumerate(immobility_summaries, 1):
                    roi_name = summary.get('roi_name', f"ROI {summary['roi_id']}")
                    f.write(f"{_bi_label('对象', 'Subject')} {i}: {roi_name}\n")
                    _write_report_metric_lines(f, summary, [
                        ('总时长', 'Total Time', 'total_time_seconds', ' s'),
                        ('不动时间', 'Immobility Time', 'immobility_time_seconds', ' s'),
                        ('不动比例', 'Immobility Percent', 'immobility_percent', '%'),
                        ('挣扎时间', 'Struggling Time', 'struggling_time_seconds', ' s'),
                        ('挣扎比例', 'Struggling Percent', 'struggling_percent', '%'),
                    ])
                    if summary.get('latency_to_immobility_seconds') is not None:
                        f.write(
                            f"  {_bi_label('首次不动潜伏期', 'Latency to First Immobility')}: "
                            f"{_report_value(summary.get('latency_to_immobility_seconds'), ' s')}\n"
                        )
                    f.write("\n")
        except Exception as e:
            print(f"生成TXT报告失败: {e}")


# ==================== Open Field ====================

class OpenFieldMultiArenaAnalysisSession:
    """Multi-arena Open Field Test analysis session."""

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths
        self.video_infos = video_infos
        self.config = config
        self.experiment_type = 'open_field'

        self.status = 'waiting'
        self.progress = 0
        self.message = 'Waiting to start...'
        self.logs = []
        self.realtime_data = []
        self.result_files = []
        self.processed_frames = 0
        self.is_running = False
        self.start_time = None
        self.end_time = None
        self.current_frame = None
        self.live_metrics = {}
        self.output_groups = {
            'video': [],
            'track_csv': [],
            'summary_csv': [],
            'bins_csv': [],
            'report': [],
            'heatmap': [],
            'trackmap': [],
        }

    def start(self):
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        thread = threading.Thread(target=self._run, daemon=True)
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
        if group and path not in self.output_groups[group]:
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

    def _parse_rois(self, config=None):
        active_config = config or self.config
        rois = active_config.get('rois') or []
        if isinstance(rois, str):
            rois = json.loads(rois)

        if not rois and active_config.get('arena_points'):
            rois = [{
                'points': active_config.get('arena_points'),
                'name': 'Arena 1',
                'time_config': {'mode': 'global', 'start': None, 'end': None},
            }]

        parsed = []
        for idx, roi in enumerate(rois):
            points = roi.get('points', roi) if isinstance(roi, dict) else roi
            if len(points) != 4:
                raise ValueError(f'ROI {idx + 1} must contain exactly 4 arena corner points.')

            parsed_points = []
            for point in points:
                if isinstance(point, dict):
                    parsed_points.append([float(point['x']), float(point['y'])])
                else:
                    parsed_points.append([float(point[0]), float(point[1])])

            detection_points = None
            if isinstance(roi, dict) and roi.get('detection_points'):
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)

            if isinstance(roi, dict):
                roi_name = str(roi.get('name') or f'Arena {idx + 1}').strip() or f'Arena {idx + 1}'
                time_config = roi.get('time_config') or {'mode': 'global', 'start': None, 'end': None}
            else:
                roi_name = f'Arena {idx + 1}'
                time_config = {'mode': 'global', 'start': None, 'end': None}

            parsed.append({
                'roi_id': idx + 1,
                'name': roi_name,
                'points': parsed_points,
                'detection_points': detection_points,
                'time_config': time_config,
            })

        if not parsed:
            raise ValueError('Open field analysis requires at least one ROI.')
        return parsed

    @staticmethod
    def _build_live_metrics(rows):
        if not rows:
            return {}
        return {
            'roi_count': len(rows),
            'detected_roi_count': sum(1 for row in rows if row.get('detected')),
            'total_distance_cm': round(sum(float(row.get('distance_cm', 0.0) or 0.0) for row in rows), 3),
            'mean_center_percent': round(
                sum(float(row.get('center_percent', 0.0) or 0.0) for row in rows) / max(len(rows), 1),
                3,
            ),
        }

    def _run(self):
        try:
            if len(self.video_paths) == 1:
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(f"Processing OFT video {i + 1}/{len(self.video_paths)}: {info.get('original_name', '')}")
                    self._analyze_single_video(path, info, batch_index=i)
                    self.progress = int((i + 1) / len(self.video_paths) * 100)

                if self.is_running:
                    self.status = 'completed'
                    self.end_time = time.time()
                    self.message = f'All {len(self.video_paths)} OFT videos finished.'
                    self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = f'Analysis error: {str(e)}'
            self._log(self.message)
            import traceback
            self._log(traceback.format_exc())

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
        result_folder = vision_config.get('output_dir') or str(RESULT_DIR)
        output_info = _prepare_output_paths(
            self.session_id,
            video_info.get('original_name', 'video'),
            suffix='oft',
            result_folder=result_folder,
            fallback='video',
            batch_index=video_index,
            batch_count=len(self.video_paths),
        )

        video_name = output_info['display_stem']
        output_base = output_info['work_base']
        final_base = output_info['final_base']
        output_video_path = f"{output_base}_analyzed.mp4"
        final_output_video_path = f"{final_base}_analyzed.mp4"
        generate_video = bool(vision_config.get('generate_video', True))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to open video: {video_path}")
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log(f"OFT video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

        bg_method = vision_config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=vision_config.get('gray_threshold', 127),
                invert=vision_config.get('invert', False),
            )
        elif bg_method == 'static':
            bg = create_static_background(video_path, num_frames=15)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=vision_config.get('erosion_kernel', 3),
            erosion_iterations=vision_config.get('erosion_iter', 1),
            dilation_kernel=vision_config.get('dilation_kernel', 5),
            dilation_iterations=vision_config.get('dilation_iter', 2),
        )
        detector = MouseDetector(
            min_area=vision_config.get('min_area', 120),
            max_area=vision_config.get('max_area', 50000),
        )

        arena_side_cm = float(vision_config.get('arena_side_cm', 50.0))
        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        roi_configs = self._parse_rois(vision_config)

        trackers = []
        for roi in roi_configs:
            trackers.append(
                OpenFieldTracker(
                    arena_points=roi['points'],
                    detection_points=roi.get('detection_points'),
                    arena_side_cm=arena_side_cm,
                    fps=fps,
                    center_ratio_x=float(vision_config.get('center_ratio_x', 0.5)),
                    center_ratio_y=float(vision_config.get('center_ratio_y', 0.5)),
                    wall_margin_cm=float(vision_config.get('wall_margin_cm', 8.0)),
                    movement_threshold_cm_s=float(vision_config.get('movement_threshold_cm_s', 2.0)),
                    roi_id=roi['roi_id'],
                    roi_name=roi['name'],
                    time_config=roi.get('time_config'),
                    global_start_time_s=global_start_time_s,
                )
            )
        self._log(f"Configured {len(trackers)} OFT arenas")

        writer = None
        if generate_video:
            for codec in ('avc1', 'H264', 'mp4v'):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    if test_writer.isOpened():
                        writer = test_writer
                        self._log(f"OFT video codec: {codec}")
                        break
                    test_writer.release()
                except Exception:
                    pass

        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start OFT analysis from frame {start_frame} ({start_frame / max(fps, 1e-6):.1f}s)")

        frames_to_process = max(0, end_frame - start_frame + 1)
        frame_idx = 0

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + frame_idx

            if current_frame_no > end_frame:
                break

            current_time = current_frame_no / max(fps, 1e-6)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = [tracker.update(frame, current_time, gray, bg_subtractor, morph_processor, detector) for tracker in trackers]

            if frame_idx % 15 == 0:
                rows = []
                for tracker, result in zip(trackers, results):
                    summary = tracker.build_summary()
                    rows.append({
                        'roi_id': tracker.roi_id,
                        'roi_name': tracker.roi_name,
                        'detected': bool(result.get('detected')),
                        'zone': result.get('zone_label', 'Unknown'),
                        'speed_cm_s': round(float(result.get('speed_cm_s', 0.0)), 3),
                        'distance_cm': summary['total_distance_cm'],
                        'center_percent': summary['center_percent'],
                        'center_entries': summary['center_entries'],
                        'wall_time_s': summary['wall_time_s'],
                        'in_stats_window': bool(result.get('in_stats_window', True)),
                    })
                self.realtime_data = rows
                self.live_metrics = self._build_live_metrics(rows)

            if generate_video and writer is not None:
                result_frame = draw_open_field_results(frame, list(zip(trackers, results)))
                writer.write(result_frame)
                if frame_idx % 10 == 0:
                    self.current_frame = result_frame

            frame_idx += 1
            self.processed_frames += 1

            if frame_idx % 20 == 0:
                if batch_index is not None:
                    batch_progress = int(batch_index / len(self.video_paths) * 100)
                    video_progress = int(frame_idx / max(frames_to_process, 1) * 100 / len(self.video_paths))
                    self.progress = batch_progress + video_progress
                else:
                    self.progress = int(frame_idx / max(frames_to_process, 1) * 100)
                self.message = f'Frame {frame_idx}/{frames_to_process} ({self.progress}%)'

        cap.release()
        if writer is not None:
            writer.release()

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
            summary['video_name'] = video_info.get('original_name', video_name)
            summary_rows.append(summary)

            for row in tracker.frame_records:
                record = dict(row)
                record['video_name'] = video_info.get('original_name', video_name)
                track_rows.append(record)

            for row in tracker.build_time_bins(bin_seconds=float(vision_config.get('bin_seconds', 300.0))):
                bucket = dict(row)
                bucket['video_name'] = video_info.get('original_name', video_name)
                bins_rows.append(bucket)

        if HAS_PANDAS and track_rows:
            track_csv_work_path = f"{output_base}_track.csv"
            if _write_dataframe_csv(track_rows, track_csv_work_path):
                track_csv_path = _export_result_file(track_csv_work_path, f"{final_base}_track.csv")
                self._append_result_file(track_csv_path, 'track_csv')

        if HAS_PANDAS and summary_rows:
            summary_csv_work_path = f"{output_base}_summary.csv"
            if _write_dataframe_csv(summary_rows, summary_csv_work_path):
                summary_csv_path = _export_result_file(summary_csv_work_path, f"{final_base}_summary.csv")
                self._append_result_file(summary_csv_path, 'summary_csv')

        if HAS_PANDAS and bins_rows:
            bins_csv_work_path = f"{output_base}_bins.csv"
            if _write_dataframe_csv(bins_rows, bins_csv_work_path):
                bins_csv_path = _export_result_file(bins_csv_work_path, f"{final_base}_bins.csv")
                self._append_result_file(bins_csv_path, 'bins_csv')

        report_txt_work_path = f"{output_base}_report.txt"
        self._generate_txt_report(summary_rows, report_txt_work_path)
        report_txt_path = _export_result_file(report_txt_work_path, f"{final_base}_report.txt")
        self._append_result_file(report_txt_path, 'report')

        heatmap_work_paths = []
        heatmap_final_paths = []
        trackmap_work_paths = []
        trackmap_final_paths = []
        for tracker in trackers:
            roi_slug = sanitize_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"arena_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"

            if render_open_field_heatmap(tracker, heatmap_work_path):
                exported_heatmap = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported_heatmap)
                self._append_result_file(exported_heatmap)
            else:
                self._log(f"Failed to save OFT heatmap: {heatmap_final_path}")

            if render_open_field_trackmap(tracker, trackmap_work_path):
                exported_trackmap = _export_result_file(trackmap_work_path, trackmap_final_path)
                trackmap_work_paths.append(trackmap_work_path)
                trackmap_final_paths.append(exported_trackmap)
                self._append_result_file(exported_trackmap)
            else:
                self._log(f"Failed to save OFT trackmap: {trackmap_final_path}")

        if heatmap_work_paths:
            if len(heatmap_final_paths) == 1:
                self._append_result_file(heatmap_final_paths[0], 'heatmap')
            else:
                heatmap_grid_work_path = f"{output_base}_heatmaps.png"
                heatmap_grid_final_path = f"{final_base}_heatmaps.png"
                if render_open_field_image_grid(heatmap_work_paths, heatmap_grid_work_path, title=f"{video_name} | Heatmaps"):
                    self._append_result_file(_export_result_file(heatmap_grid_work_path, heatmap_grid_final_path), 'heatmap')
                else:
                    self._log(f"Failed to save OFT heatmap grid: {heatmap_grid_final_path}")

        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                trackmap_grid_work_path = f"{output_base}_trackmaps.png"
                trackmap_grid_final_path = f"{final_base}_trackmaps.png"
                if render_open_field_image_grid(trackmap_work_paths, trackmap_grid_work_path, title=f"{video_name} | Trajectory Maps"):
                    self._append_result_file(_export_result_file(trackmap_grid_work_path, trackmap_grid_final_path), 'trackmap')
                else:
                    self._log(f"Failed to save OFT trackmap grid: {trackmap_grid_final_path}")

        if generate_video and os.path.exists(output_video_path):
            exported_video = _export_result_file(output_video_path, final_output_video_path)
            self._append_result_file(exported_video, 'video')
            if exported_video:
                self._log(f"OFT annotated video saved: {exported_video}")

        self.realtime_data = [
            {
                'roi_id': row['roi_id'],
                'roi_name': row['roi_name'],
                'detected': True,
                'zone': 'Completed',
                'speed_cm_s': row['mean_speed_cm_s'],
                'distance_cm': row['total_distance_cm'],
                'center_percent': row['center_percent'],
                'center_entries': row['center_entries'],
                'wall_time_s': row['wall_time_s'],
                'in_stats_window': True,
            }
            for row in summary_rows
        ]
        self.live_metrics = self._build_live_metrics(self.realtime_data)

        if batch_index is None and self.is_running:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = 'Open field analysis completed.'

        elapsed = time.time() - video_start_time
        self._log(f"OFT video processed in {elapsed:.1f}s, {frame_idx} frames")

        import gc
        gc.collect()

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        return {
            'session_id': self.session_id,
            'experiment_type': self.experiment_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'logs': self.logs[-10:],
            'realtime_data': self.realtime_data,
            'result_files': self.result_files,
            'experiment_stats': self.live_metrics,
            'stats': {
                'processed_frames': self.processed_frames,
                'elapsed_time': elapsed,
                'fps': processing_fps,
            }
        }

    def get_download_file_map(self):
        return self.output_groups

    def _generate_txt_report(self, summaries, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 68 + "\n")
                f.write("MouseScope 旷场实验报告 (Open Field Test Report)\n")
                f.write("=" * 68 + "\n\n")
                f.write(f"{_bi_label('分析时间', 'Analysis Time')}: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if summaries:
                    f.write(f"{_bi_label('视频', 'Video')}: {summaries[0].get('video_name', '')}\n")
                f.write(f"{_bi_label('场地数量', 'Arena Count')}: {len(summaries)}\n\n")

                for summary in summaries:
                    f.write(f"[ROI {summary.get('roi_id')}] {summary.get('roi_name', '')}\n")
                    f.write(f"  {_bi_label('统计模式', 'Time Mode')}: {_report_time_mode_label(summary.get('time_mode', 'global'))}\n")
                    if summary.get('time_start_s') is not None:
                        f.write(f"  {_bi_label('起始时间', 'Window Start')}: {_report_value(summary.get('time_start_s'), ' s')}\n")
                    if summary.get('time_end_s') is not None:
                        f.write(f"  {_bi_label('结束时间', 'Window End')}: {_report_value(summary.get('time_end_s'), ' s')}\n")
                    _write_report_metric_lines(f, summary, [
                        ('总时长', 'Total Time', 'total_time_s', ' s'),
                        ('检测率', 'Detected Rate', 'detected_rate_percent', '%'),
                        ('总距离', 'Total Distance', 'total_distance_cm', ' cm'),
                        ('平均速度', 'Mean Speed', 'mean_speed_cm_s', ' cm/s'),
                        ('最大速度', 'Max Speed', 'max_speed_cm_s', ' cm/s'),
                        ('中心区时间', 'Center Time', 'center_time_s', ' s'),
                        ('中心区比例', 'Center Percent', 'center_percent', '%'),
                        ('中心区进入次数', 'Center Entries', 'center_entries', ''),
                        ('中心区潜伏期', 'Center Latency', 'center_latency_s', ' s'),
                        ('壁区时间', 'Wall Time', 'wall_time_s', ' s'),
                        ('壁区比例', 'Wall Percent', 'wall_percent', '%'),
                        ('外周区时间', 'Outer Time', 'outer_time_s', ' s'),
                        ('运动时间', 'Moving Time', 'moving_time_s', ' s'),
                        ('静止时间', 'Immobility Time', 'immobility_time_s', ' s'),
                    ])
                    f.write("\n")
        except Exception as e:
            print(f"Failed to write OFT report: {e}")


OpenFieldAnalysisSession = OpenFieldMultiArenaAnalysisSession


class NovelObjectMultiArenaAnalysisSession:
    """Multi-ROI Novel Object Recognition analysis session."""

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths
        self.video_infos = video_infos
        self.config = config
        self.experiment_type = 'novel_object'

        self.status = 'waiting'
        self.progress = 0
        self.message = 'Waiting to start...'
        self.logs = []
        self.realtime_data = []
        self.result_files = []
        self.processed_frames = 0
        self.is_running = False
        self.start_time = None
        self.end_time = None
        self.current_frame = None
        self.live_metrics = {}
        self.output_groups = {
            'video': [],
            'track_csv': [],
            'summary_csv': [],
            'bins_csv': [],
            'report': [],
            'heatmap': [],
            'trackmap': [],
        }

    def start(self):
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        thread = threading.Thread(target=self._run, daemon=True)
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
        if group and path not in self.output_groups[group]:
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

    def _parse_rois(self, config=None):
        active_config = config or self.config
        rois = active_config.get('rois') or []
        if isinstance(rois, str):
            rois = json.loads(rois)

        parsed = []
        for idx, roi in enumerate(rois):
            points = roi.get('points', roi) if isinstance(roi, dict) else roi
            if len(points) != 4:
                raise ValueError(f'ROI {idx + 1} must contain exactly 4 arena corner points.')

            parsed_points = []
            for point in points:
                if isinstance(point, dict):
                    parsed_points.append([float(point['x']), float(point['y'])])
                else:
                    parsed_points.append([float(point[0]), float(point[1])])

            detection_points = None
            if isinstance(roi, dict) and roi.get('detection_points'):
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)

            objects = roi.get('objects') if isinstance(roi, dict) else None
            if not objects or len(objects) != 2:
                raise ValueError(f'ROI {idx + 1} must contain exactly 2 object points.')

            parsed_objects = []
            for obj_idx, obj in enumerate(objects):
                point = obj.get('point', obj) if isinstance(obj, dict) else obj
                if isinstance(point, dict):
                    point_xy = [float(point['x']), float(point['y'])]
                else:
                    point_xy = [float(point[0]), float(point[1])]

                parsed_objects.append({
                    'point': point_xy,
                    'name': (obj.get('name') if isinstance(obj, dict) else None) or f'Object {obj_idx + 1}',
                    'role': (obj.get('role') if isinstance(obj, dict) else None) or ('novel' if obj_idx == 1 else 'familiar'),
                })

            if isinstance(roi, dict):
                roi_name = str(roi.get('name') or f'ROI {idx + 1}').strip() or f'ROI {idx + 1}'
                time_config = roi.get('time_config') or {'mode': 'global', 'start': None, 'end': None}
            else:
                roi_name = f'ROI {idx + 1}'
                time_config = {'mode': 'global', 'start': None, 'end': None}

            parsed.append({
                'roi_id': idx + 1,
                'name': roi_name,
                'points': parsed_points,
                'detection_points': detection_points,
                'objects': parsed_objects,
                'time_config': time_config,
            })

        if not parsed:
            raise ValueError('Novel object recognition requires at least one ROI.')
        return parsed

    @staticmethod
    def _build_live_metrics(rows):
        if not rows:
            return {}
        valid_dis = [float(row['discrimination_index']) for row in rows if row.get('discrimination_index') is not None]
        return {
            'roi_count': len(rows),
            'detected_roi_count': sum(1 for row in rows if row.get('detected')),
            'total_distance_cm': round(sum(float(row.get('distance_cm', 0.0) or 0.0) for row in rows), 3),
            'total_exploration_s': round(sum(float(row.get('total_exploration_s', 0.0) or 0.0) for row in rows), 3),
            'mean_discrimination_index': round(sum(valid_dis) / len(valid_dis), 4) if valid_dis else None,
        }

    def _run(self):
        try:
            if len(self.video_paths) == 1:
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(f"Processing NOR video {i + 1}/{len(self.video_paths)}: {info.get('original_name', '')}")
                    self._analyze_single_video(path, info, batch_index=i)
                    self.progress = int((i + 1) / len(self.video_paths) * 100)

                if self.is_running:
                    self.status = 'completed'
                    self.end_time = time.time()
                    self.message = f'All {len(self.video_paths)} NOR videos finished.'
                    self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = f'Analysis error: {str(e)}'
            self._log(self.message)
            import traceback
            self._log(traceback.format_exc())

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
        result_folder = vision_config.get('output_dir') or str(RESULT_DIR)
        output_info = _prepare_output_paths(
            self.session_id,
            video_info.get('original_name', 'video'),
            suffix='nor',
            result_folder=result_folder,
            fallback='video',
            batch_index=video_index,
            batch_count=len(self.video_paths),
        )

        video_name = output_info['display_stem']
        output_base = output_info['work_base']
        final_base = output_info['final_base']
        output_video_path = f"{output_base}_analyzed.mp4"
        final_output_video_path = f"{final_base}_analyzed.mp4"
        generate_video = bool(vision_config.get('generate_video', True))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to open video: {video_path}")
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log(f"NOR video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

        bg_method = vision_config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=vision_config.get('gray_threshold', 127),
                invert=vision_config.get('invert', False),
            )
        elif bg_method == 'static':
            bg = create_static_background(video_path, num_frames=15)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=vision_config.get('erosion_kernel', 3),
            erosion_iterations=vision_config.get('erosion_iter', 1),
            dilation_kernel=vision_config.get('dilation_kernel', 5),
            dilation_iterations=vision_config.get('dilation_iter', 2),
        )
        detector = MouseDetector(
            min_area=vision_config.get('min_area', 120),
            max_area=vision_config.get('max_area', 50000),
        )

        phase = vision_config.get('phase', 'recognition')
        arena_width_cm = float(vision_config.get('arena_width_cm', vision_config.get('arena_side_cm', 50.0)))
        arena_height_cm = float(vision_config.get('arena_height_cm', vision_config.get('arena_side_cm', 50.0)))
        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        roi_configs = self._parse_rois(vision_config)

        trackers = []
        for roi in roi_configs:
            trackers.append(
                NovelObjectTracker(
                    arena_points=roi['points'],
                    object_points=roi['objects'],
                    detection_points=roi.get('detection_points'),
                    arena_width_cm=arena_width_cm,
                    arena_height_cm=arena_height_cm,
                    fps=fps,
                    roi_id=roi['roi_id'],
                    roi_name=roi['name'],
                    phase=phase,
                    time_config=roi.get('time_config'),
                    global_start_time_s=global_start_time_s,
                    center_ratio_x=float(vision_config.get('center_ratio_x', 0.5)),
                    center_ratio_y=float(vision_config.get('center_ratio_y', 0.5)),
                    interaction_radius_cm=float(vision_config.get('interaction_radius_cm', 4.0)),
                    exploration_max_speed_cm_s=float(vision_config.get('exploration_max_speed_cm_s', 12.0)),
                    use_head_direction=bool(vision_config.get('use_head_direction', True)),
                    head_angle_threshold_deg=float(vision_config.get('head_angle_threshold_deg', 50.0)),
                    head_confidence_threshold=float(vision_config.get('head_confidence_threshold', 0.35)),
                    movement_threshold_cm_s=float(vision_config.get('movement_threshold_cm_s', 2.0)),
                    min_valid_exploration_s=float(vision_config.get('min_valid_exploration_s', 20.0)),
                )
            )
        self._log(f"Configured {len(trackers)} NOR arenas")

        writer = None
        if generate_video:
            for codec in ('avc1', 'H264', 'mp4v'):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    if test_writer.isOpened():
                        writer = test_writer
                        self._log(f"NOR video codec: {codec}")
                        break
                    test_writer.release()
                except Exception:
                    pass

        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start NOR analysis from frame {start_frame} ({start_frame / max(fps, 1e-6):.1f}s)")

        frames_to_process = max(0, end_frame - start_frame + 1)
        frame_idx = 0

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + frame_idx

            if current_frame_no > end_frame:
                break

            current_time = current_frame_no / max(fps, 1e-6)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = [tracker.update(frame, current_time, gray, bg_subtractor, morph_processor, detector) for tracker in trackers]

            if frame_idx % 15 == 0:
                rows = []
                for tracker, result in zip(trackers, results):
                    summary = tracker.build_summary()
                    rows.append({
                        'roi_id': tracker.roi_id,
                        'roi_name': tracker.roi_name,
                        'detected': bool(result.get('detected')),
                        'current_object': result.get('current_object_name') or '-',
                        'current_role': result.get('current_object_role') or '-',
                        'distance_cm': summary['total_distance_cm'],
                        'total_exploration_s': summary['total_object_exploration_s'],
                        'novel_exploration_s': summary.get('novel_exploration_s'),
                        'familiar_exploration_s': summary.get('familiar_exploration_s'),
                        'discrimination_index': summary.get('discrimination_index'),
                        'preference_index_percent': summary.get('preference_index_percent'),
                        'meets_min_exploration': bool(summary.get('meets_min_exploration', False)),
                        'in_stats_window': bool(result.get('in_stats_window', True)),
                    })
                self.realtime_data = rows
                self.live_metrics = self._build_live_metrics(rows)

            if generate_video and writer is not None:
                result_frame = draw_novel_object_results(frame, list(zip(trackers, results)))
                writer.write(result_frame)
                if frame_idx % 10 == 0:
                    self.current_frame = result_frame

            frame_idx += 1
            self.processed_frames += 1

            if frame_idx % 20 == 0:
                if batch_index is not None:
                    batch_progress = int(batch_index / len(self.video_paths) * 100)
                    video_progress = int(frame_idx / max(frames_to_process, 1) * 100 / len(self.video_paths))
                    self.progress = batch_progress + video_progress
                else:
                    self.progress = int(frame_idx / max(frames_to_process, 1) * 100)
                self.message = f'Frame {frame_idx}/{frames_to_process} ({self.progress}%)'

        cap.release()
        if writer is not None:
            writer.release()

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
            summary['video_name'] = video_info.get('original_name', video_name)
            summary_rows.append(summary)

            for row in tracker.frame_records:
                record = dict(row)
                record['video_name'] = video_info.get('original_name', video_name)
                track_rows.append(record)

            for row in tracker.build_time_bins(bin_seconds=float(vision_config.get('bin_seconds', 300.0))):
                bucket = dict(row)
                bucket['video_name'] = video_info.get('original_name', video_name)
                bins_rows.append(bucket)

        if HAS_PANDAS and track_rows:
            track_csv_work_path = f"{output_base}_track.csv"
            if _write_dataframe_csv(track_rows, track_csv_work_path):
                track_csv_path = _export_result_file(track_csv_work_path, f"{final_base}_track.csv")
                self._append_result_file(track_csv_path, 'track_csv')

        if HAS_PANDAS and summary_rows:
            summary_csv_work_path = f"{output_base}_summary.csv"
            if _write_dataframe_csv(summary_rows, summary_csv_work_path):
                summary_csv_path = _export_result_file(summary_csv_work_path, f"{final_base}_summary.csv")
                self._append_result_file(summary_csv_path, 'summary_csv')

        if HAS_PANDAS and bins_rows:
            bins_csv_work_path = f"{output_base}_bins.csv"
            if _write_dataframe_csv(bins_rows, bins_csv_work_path):
                bins_csv_path = _export_result_file(bins_csv_work_path, f"{final_base}_bins.csv")
                self._append_result_file(bins_csv_path, 'bins_csv')

        report_txt_work_path = f"{output_base}_report.txt"
        self._generate_txt_report(summary_rows, report_txt_work_path)
        report_txt_path = _export_result_file(report_txt_work_path, f"{final_base}_report.txt")
        self._append_result_file(report_txt_path, 'report')

        heatmap_work_paths = []
        heatmap_final_paths = []
        trackmap_work_paths = []
        trackmap_final_paths = []
        for tracker in trackers:
            roi_slug = sanitize_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"roi_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"

            if render_novel_object_heatmap(tracker, heatmap_work_path):
                exported_heatmap = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported_heatmap)
                self._append_result_file(exported_heatmap)
            else:
                self._log(f"Failed to save NOR heatmap: {heatmap_final_path}")

            if render_novel_object_trackmap(tracker, trackmap_work_path):
                exported_trackmap = _export_result_file(trackmap_work_path, trackmap_final_path)
                trackmap_work_paths.append(trackmap_work_path)
                trackmap_final_paths.append(exported_trackmap)
                self._append_result_file(exported_trackmap)
            else:
                self._log(f"Failed to save NOR trackmap: {trackmap_final_path}")

        if heatmap_work_paths:
            if len(heatmap_final_paths) == 1:
                self._append_result_file(heatmap_final_paths[0], 'heatmap')
            else:
                heatmap_grid_work_path = f"{output_base}_heatmaps.png"
                heatmap_grid_final_path = f"{final_base}_heatmaps.png"
                if render_open_field_image_grid(heatmap_work_paths, heatmap_grid_work_path, title=f"{video_name} | NOR Heatmaps"):
                    self._append_result_file(_export_result_file(heatmap_grid_work_path, heatmap_grid_final_path), 'heatmap')
                else:
                    self._log(f"Failed to save NOR heatmap grid: {heatmap_grid_final_path}")

        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                trackmap_grid_work_path = f"{output_base}_trackmaps.png"
                trackmap_grid_final_path = f"{final_base}_trackmaps.png"
                if render_open_field_image_grid(trackmap_work_paths, trackmap_grid_work_path, title=f"{video_name} | NOR Trajectory Maps"):
                    self._append_result_file(_export_result_file(trackmap_grid_work_path, trackmap_grid_final_path), 'trackmap')
                else:
                    self._log(f"Failed to save NOR trackmap grid: {trackmap_grid_final_path}")

        if generate_video and os.path.exists(output_video_path):
            exported_video = _export_result_file(output_video_path, final_output_video_path)
            self._append_result_file(exported_video, 'video')
            if exported_video:
                self._log(f"NOR annotated video saved: {exported_video}")

        self.realtime_data = [
            {
                'roi_id': row['roi_id'],
                'roi_name': row['roi_name'],
                'detected': True,
                'current_object': 'Completed',
                'current_role': '-',
                'distance_cm': row['total_distance_cm'],
                'total_exploration_s': row['total_object_exploration_s'],
                'novel_exploration_s': row.get('novel_exploration_s'),
                'familiar_exploration_s': row.get('familiar_exploration_s'),
                'discrimination_index': row.get('discrimination_index'),
                'preference_index_percent': row.get('preference_index_percent'),
                'meets_min_exploration': bool(row.get('meets_min_exploration', False)),
                'in_stats_window': True,
            }
            for row in summary_rows
        ]
        self.live_metrics = self._build_live_metrics(self.realtime_data)

        if batch_index is None and self.is_running:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = 'Novel object recognition analysis completed.'

        elapsed = time.time() - video_start_time
        self._log(f"NOR video processed in {elapsed:.1f}s, {frame_idx} frames")

        import gc
        gc.collect()

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        return {
            'session_id': self.session_id,
            'experiment_type': self.experiment_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'logs': self.logs[-10:],
            'realtime_data': self.realtime_data,
            'result_files': self.result_files,
            'experiment_stats': self.live_metrics,
            'stats': {
                'processed_frames': self.processed_frames,
                'elapsed_time': elapsed,
                'fps': processing_fps,
            }
        }

    def get_download_file_map(self):
        return self.output_groups

    def _generate_txt_report(self, summaries, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 72 + "\n")
                f.write("MouseScope 新奇物体识别报告 (Novel Object Recognition Report)\n")
                f.write("=" * 72 + "\n\n")
                f.write(f"{_bi_label('分析时间', 'Analysis Time')}: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if summaries:
                    f.write(f"{_bi_label('视频', 'Video')}: {summaries[0].get('video_name', '')}\n")
                    f.write(f"{_bi_label('实验阶段', 'Phase')}: {_report_phase_label(summaries[0].get('phase', 'recognition'))}\n")
                f.write(f"{_bi_label('ROI 数量', 'ROI Count')}: {len(summaries)}\n")
                f.write(
                    f"{_bi_label('统计规则', 'Scoring Rule')}: "
                    "探索判定优先使用轮廓接近与头部朝向门控；当鼻尖估计置信度不足时回退到接近规则 "
                    "(Exploration uses contour proximity with head-direction gating when the nose estimate is confident; "
                    "low-confidence frames fall back to proximity.)\n\n"
                )

                for summary in summaries:
                    f.write(f"[ROI {summary.get('roi_id')}] {summary.get('roi_name', '')}\n")
                    f.write(f"  {_bi_label('统计模式', 'Time Mode')}: {_report_time_mode_label(summary.get('time_mode', 'global'))}\n")
                    if summary.get('time_start_s') is not None:
                        f.write(f"  {_bi_label('起始时间', 'Window Start')}: {_report_value(summary.get('time_start_s'), ' s')}\n")
                    if summary.get('time_end_s') is not None:
                        f.write(f"  {_bi_label('结束时间', 'Window End')}: {_report_value(summary.get('time_end_s'), ' s')}\n")

                    _write_report_metric_lines(f, summary, [
                        ('总时长', 'Total Time', 'total_time_s', ' s'),
                        ('检测率', 'Detected Rate', 'detected_rate_percent', '%'),
                        ('总距离', 'Total Distance', 'total_distance_cm', ' cm'),
                        ('平均速度', 'Mean Speed', 'mean_speed_cm_s', ' cm/s'),
                        ('中心区时间', 'Center Time', 'center_time_s', ' s'),
                        ('中心区比例', 'Center Percent', 'center_percent', '%'),
                        ('物体 1 探索时间', 'Object 1 Exploration', 'object_1_exploration_s', ' s'),
                        ('物体 2 探索时间', 'Object 2 Exploration', 'object_2_exploration_s', ' s'),
                        ('总探索时间', 'Total Exploration', 'total_object_exploration_s', ' s'),
                        ('头部支持探索时间', 'Head-Supported Exploration', 'head_supported_exploration_s', ' s'),
                        ('回退探索时间', 'Fallback Exploration', 'fallback_exploration_s', ' s'),
                        ('新奇物体探索时间', 'Novel Exploration', 'novel_exploration_s', ' s'),
                        ('熟悉物体探索时间', 'Familiar Exploration', 'familiar_exploration_s', ' s'),
                        ('辨别指数', 'Discrimination Index', 'discrimination_index', ''),
                        ('偏好指数', 'Preference Index', 'preference_index_percent', '%'),
                        ('头部估计率', 'Head Estimated Rate', 'head_estimated_rate_percent', '%'),
                        ('头部高置信率', 'Head Confident Rate', 'head_confident_rate_percent', '%'),
                        ('头部角度阈值', 'Head Angle Threshold', 'head_angle_threshold_deg', ' deg'),
                    ])
                    f.write("\n")
        except Exception as e:
            print(f"Failed to write NOR report: {e}")


class ElevatedPlusMultiArenaAnalysisSession:
    """Multi-ROI Elevated Plus Maze analysis session."""

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths
        self.video_infos = video_infos
        self.config = config
        self.experiment_type = 'elevated_plus'

        self.status = 'waiting'
        self.progress = 0
        self.message = 'Waiting to start...'
        self.logs = []
        self.realtime_data = []
        self.result_files = []
        self.processed_frames = 0
        self.is_running = False
        self.start_time = None
        self.end_time = None
        self.current_frame = None
        self.live_metrics = {}
        self.output_groups = {
            'video': [],
            'track_csv': [],
            'summary_csv': [],
            'bins_csv': [],
            'report': [],
            'heatmap': [],
            'trackmap': [],
        }

    def start(self):
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        thread = threading.Thread(target=self._run, daemon=True)
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
        if group and path not in self.output_groups[group]:
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

    def _parse_rois(self, config=None):
        active_config = config or self.config
        rois = active_config.get('rois') or []
        if isinstance(rois, str):
            rois = json.loads(rois)

        parsed = []
        for idx, roi in enumerate(rois):
            if not isinstance(roi, dict):
                raise ValueError(f'EPM ROI {idx + 1} must be an object.')
            zone_polygons = _coerce_zone_polygons(roi.get('zones'), EPM_ZONE_KEYS, 'EPM')

            roi_name = str(roi.get('name') or f'EPM {idx + 1}').strip() or f'EPM {idx + 1}'
            time_config = roi.get('time_config') or {'mode': 'global', 'start': None, 'end': None}
            detection_points = None
            if isinstance(roi, dict) and roi.get('detection_points'):
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)

            parsed.append({
                'roi_id': idx + 1,
                'name': roi_name,
                'zones': zone_polygons,
                'time_config': time_config,
                'detection_points': detection_points,
            })

        if not parsed:
            raise ValueError('Elevated plus maze analysis requires at least one ROI.')
        return parsed

    @staticmethod
    def _build_live_metrics(rows):
        if not rows:
            return {}
        valid_open = [float(row['open_time_percent_total']) for row in rows if row.get('open_time_percent_total') is not None]
        return {
            'roi_count': len(rows),
            'detected_roi_count': sum(1 for row in rows if row.get('detected')),
            'total_distance_cm': round(sum(float(row.get('distance_cm', 0.0) or 0.0) for row in rows), 3),
            'mean_open_time_percent_total': round(sum(valid_open) / len(valid_open), 3) if valid_open else None,
            'total_open_entries': sum(int(row.get('open_entries') or 0) for row in rows),
        }

    def _run(self):
        try:
            if len(self.video_paths) == 1:
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(f"Processing EPM video {i + 1}/{len(self.video_paths)}: {info.get('original_name', '')}")
                    self._analyze_single_video(path, info, batch_index=i)
                    self.progress = int((i + 1) / len(self.video_paths) * 100)

                if self.is_running:
                    self.status = 'completed'
                    self.end_time = time.time()
                    self.message = f'All {len(self.video_paths)} EPM videos finished.'
                    self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = f'Analysis error: {str(e)}'
            self._log(self.message)
            import traceback
            self._log(traceback.format_exc())

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
        result_folder = vision_config.get('output_dir') or str(RESULT_DIR)
        output_info = _prepare_output_paths(
            self.session_id,
            video_info.get('original_name', 'video'),
            suffix='epm',
            result_folder=result_folder,
            fallback='video',
            batch_index=video_index,
            batch_count=len(self.video_paths),
        )

        video_name = output_info['display_stem']
        output_base = output_info['work_base']
        final_base = output_info['final_base']
        output_video_path = f"{output_base}_analyzed.mp4"
        final_output_video_path = f"{final_base}_analyzed.mp4"
        generate_video = bool(vision_config.get('generate_video', True))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to open video: {video_path}")
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log(f"EPM video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

        bg_method = vision_config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=vision_config.get('gray_threshold', 127),
                invert=vision_config.get('invert', False),
            )
        elif bg_method == 'static':
            bg = create_static_background(video_path, num_frames=15)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=vision_config.get('erosion_kernel', 3),
            erosion_iterations=vision_config.get('erosion_iter', 1),
            dilation_kernel=vision_config.get('dilation_kernel', 5),
            dilation_iterations=vision_config.get('dilation_iter', 2),
        )
        detector = MouseDetector(
            min_area=vision_config.get('min_area', 120),
            max_area=vision_config.get('max_area', 50000),
        )

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        roi_configs = self._parse_rois(vision_config)

        trackers = []
        for roi in roi_configs:
            trackers.append(
                ElevatedPlusTracker(
                    zone_polygons=roi['zones'],
                    detection_polygon=roi.get('detection_points'),
                    center_size_cm=float(vision_config.get('center_size_cm', 5.0)),
                    open_arm_length_cm=float(vision_config.get('open_arm_length_cm', 30.0)),
                    closed_arm_length_cm=float(vision_config.get('closed_arm_length_cm', 30.0)),
                    fps=fps,
                    movement_threshold_cm_s=float(vision_config.get('movement_threshold_cm_s', 2.0)),
                    entry_depth_ratio=float(vision_config.get('entry_depth_ratio', 0.2)),
                    roi_id=roi['roi_id'],
                    roi_name=roi['name'],
                    time_config=roi.get('time_config'),
                    global_start_time_s=global_start_time_s,
                )
            )
        self._log(f"Configured {len(trackers)} EPM mazes")

        writer = None
        if generate_video:
            for codec in ('avc1', 'H264', 'mp4v'):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    if test_writer.isOpened():
                        writer = test_writer
                        self._log(f"EPM video codec: {codec}")
                        break
                    test_writer.release()
                except Exception:
                    pass

        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start EPM analysis from frame {start_frame} ({start_frame / max(fps, 1e-6):.1f}s)")

        frames_to_process = max(0, end_frame - start_frame + 1)
        frame_idx = 0

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + frame_idx

            if current_frame_no > end_frame:
                break

            current_time = current_frame_no / max(fps, 1e-6)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = [tracker.update(frame, current_time, gray, bg_subtractor, morph_processor, detector) for tracker in trackers]

            if frame_idx % 15 == 0:
                rows = []
                for tracker, result in zip(trackers, results):
                    summary = tracker.build_summary()
                    rows.append({
                        'roi_id': tracker.roi_id,
                        'roi_name': tracker.roi_name,
                        'detected': bool(result.get('detected')),
                        'zone': result.get('zone_label', 'Unknown'),
                        'speed_cm_s': round(float(result.get('speed_cm_s', 0.0)), 3),
                        'distance_cm': summary['total_distance_cm'],
                        'open_time_percent_total': summary.get('open_time_percent_total'),
                        'open_entries': summary.get('open_entries'),
                        'closed_entries': summary.get('closed_entries'),
                        'open_entry_percent': summary.get('open_entry_percent'),
                        'in_stats_window': bool(result.get('in_stats_window', True)),
                    })
                self.realtime_data = rows
                self.live_metrics = self._build_live_metrics(rows)

            if generate_video and writer is not None:
                result_frame = draw_elevated_plus_results(frame, list(zip(trackers, results)))
                writer.write(result_frame)
                if frame_idx % 10 == 0:
                    self.current_frame = result_frame

            frame_idx += 1
            self.processed_frames += 1

            if frame_idx % 20 == 0:
                if batch_index is not None:
                    batch_progress = int(batch_index / len(self.video_paths) * 100)
                    video_progress = int(frame_idx / max(frames_to_process, 1) * 100 / len(self.video_paths))
                    self.progress = batch_progress + video_progress
                else:
                    self.progress = int(frame_idx / max(frames_to_process, 1) * 100)
                self.message = f'Frame {frame_idx}/{frames_to_process} ({self.progress}%)'

        cap.release()
        if writer is not None:
            writer.release()

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
            summary['video_name'] = video_info.get('original_name', video_name)
            summary_rows.append(summary)

            for row in tracker.frame_records:
                record = dict(row)
                record['video_name'] = video_info.get('original_name', video_name)
                track_rows.append(record)

            for row in tracker.build_time_bins(bin_seconds=float(vision_config.get('bin_seconds', 60.0))):
                bucket = dict(row)
                bucket['video_name'] = video_info.get('original_name', video_name)
                bins_rows.append(bucket)

        if HAS_PANDAS and track_rows:
            track_csv_work_path = f"{output_base}_track.csv"
            if _write_dataframe_csv(track_rows, track_csv_work_path):
                track_csv_path = _export_result_file(track_csv_work_path, f"{final_base}_track.csv")
                self._append_result_file(track_csv_path, 'track_csv')

        if HAS_PANDAS and summary_rows:
            summary_csv_work_path = f"{output_base}_summary.csv"
            if _write_dataframe_csv(summary_rows, summary_csv_work_path):
                summary_csv_path = _export_result_file(summary_csv_work_path, f"{final_base}_summary.csv")
                self._append_result_file(summary_csv_path, 'summary_csv')

        if HAS_PANDAS and bins_rows:
            bins_csv_work_path = f"{output_base}_bins.csv"
            if _write_dataframe_csv(bins_rows, bins_csv_work_path):
                bins_csv_path = _export_result_file(bins_csv_work_path, f"{final_base}_bins.csv")
                self._append_result_file(bins_csv_path, 'bins_csv')

        report_txt_work_path = f"{output_base}_report.txt"
        self._generate_txt_report(summary_rows, report_txt_work_path)
        report_txt_path = _export_result_file(report_txt_work_path, f"{final_base}_report.txt")
        self._append_result_file(report_txt_path, 'report')

        heatmap_work_paths = []
        heatmap_final_paths = []
        trackmap_work_paths = []
        trackmap_final_paths = []
        for tracker in trackers:
            roi_slug = sanitize_epm_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"epm_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"

            if render_elevated_plus_heatmap(tracker, heatmap_work_path):
                exported_heatmap = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported_heatmap)
                self._append_result_file(exported_heatmap)
            else:
                self._log(f"Failed to save EPM heatmap: {heatmap_final_path}")

            if render_elevated_plus_trackmap(tracker, trackmap_work_path):
                exported_trackmap = _export_result_file(trackmap_work_path, trackmap_final_path)
                trackmap_work_paths.append(trackmap_work_path)
                trackmap_final_paths.append(exported_trackmap)
                self._append_result_file(exported_trackmap)
            else:
                self._log(f"Failed to save EPM trackmap: {trackmap_final_path}")

        if heatmap_work_paths:
            if len(heatmap_final_paths) == 1:
                self._append_result_file(heatmap_final_paths[0], 'heatmap')
            else:
                heatmap_grid_work_path = f"{output_base}_heatmaps.png"
                heatmap_grid_final_path = f"{final_base}_heatmaps.png"
                if render_elevated_plus_image_grid(heatmap_work_paths, heatmap_grid_work_path, title=f"{video_name} | EPM Heatmaps"):
                    self._append_result_file(_export_result_file(heatmap_grid_work_path, heatmap_grid_final_path), 'heatmap')
                else:
                    self._log(f"Failed to save EPM heatmap grid: {heatmap_grid_final_path}")

        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                trackmap_grid_work_path = f"{output_base}_trackmaps.png"
                trackmap_grid_final_path = f"{final_base}_trackmaps.png"
                if render_elevated_plus_image_grid(trackmap_work_paths, trackmap_grid_work_path, title=f"{video_name} | EPM Trajectory Maps"):
                    self._append_result_file(_export_result_file(trackmap_grid_work_path, trackmap_grid_final_path), 'trackmap')
                else:
                    self._log(f"Failed to save EPM trackmap grid: {trackmap_grid_final_path}")

        if generate_video and os.path.exists(output_video_path):
            exported_video = _export_result_file(output_video_path, final_output_video_path)
            self._append_result_file(exported_video, 'video')
            if exported_video:
                self._log(f"EPM annotated video saved: {exported_video}")

        self.realtime_data = [
            {
                'roi_id': row['roi_id'],
                'roi_name': row['roi_name'],
                'detected': True,
                'zone': 'Completed',
                'speed_cm_s': row['mean_speed_cm_s'],
                'distance_cm': row['total_distance_cm'],
                'open_time_percent_total': row.get('open_time_percent_total'),
                'open_entries': row.get('open_entries'),
                'closed_entries': row.get('closed_entries'),
                'open_entry_percent': row.get('open_entry_percent'),
                'in_stats_window': True,
            }
            for row in summary_rows
        ]
        self.live_metrics = self._build_live_metrics(self.realtime_data)

        if batch_index is None and self.is_running:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = 'Elevated plus maze analysis completed.'

        elapsed = time.time() - video_start_time
        self._log(f"EPM video processed in {elapsed:.1f}s, {frame_idx} frames")

        import gc
        gc.collect()

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        return {
            'session_id': self.session_id,
            'experiment_type': self.experiment_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'logs': self.logs[-10:],
            'realtime_data': self.realtime_data,
            'result_files': self.result_files,
            'experiment_stats': self.live_metrics,
            'stats': {
                'processed_frames': self.processed_frames,
                'elapsed_time': elapsed,
                'fps': processing_fps,
            }
        }

    def get_download_file_map(self):
        return self.output_groups

    def _generate_txt_report(self, summaries, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 72 + "\n")
                f.write("MouseScope 高架十字迷宫报告 (Elevated Plus Maze Report)\n")
                f.write("=" * 72 + "\n\n")
                f.write(f"{_bi_label('分析时间', 'Analysis Time')}: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if summaries:
                    f.write(f"{_bi_label('视频', 'Video')}: {summaries[0].get('video_name', '')}\n")
                f.write(f"{_bi_label('ROI 数量', 'ROI Count')}: {len(summaries)}\n")
                f.write(
                    f"{_bi_label('统计规则', 'Scoring Rule')}: "
                    "当质心越过各臂内部入口阈值时记为一次入臂 "
                    "(Arm entries are scored when the centroid crosses the inner entry threshold inside each arm.)\n\n"
                )

                for summary in summaries:
                    f.write(f"[ROI {summary.get('roi_id')}] {summary.get('roi_name', '')}\n")
                    _write_report_metric_lines(f, summary, [
                        ('总时长', 'Total Time', 'total_time_s', ' s'),
                        ('检测率', 'Detected Rate', 'detected_rate_percent', '%'),
                        ('总距离', 'Total Distance', 'total_distance_cm', ' cm'),
                        ('平均速度', 'Mean Speed', 'mean_speed_cm_s', ' cm/s'),
                        ('开放臂时间', 'Open Time', 'open_time_s', ' s'),
                        ('封闭臂时间', 'Closed Time', 'closed_time_s', ' s'),
                        ('中心区时间', 'Center Time', 'center_time_s', ' s'),
                        ('开放臂时间占比（总时长）', 'Open Time % (Total)', 'open_time_percent_total', '%'),
                        ('开放臂时间占比（仅双臂）', 'Open Time % (Arms Only)', 'open_time_percent_arms_only', '%'),
                        ('开放臂进入次数', 'Open Entries', 'open_entries', ''),
                        ('封闭臂进入次数', 'Closed Entries', 'closed_entries', ''),
                        ('开放臂进入占比', 'Open Entry %', 'open_entry_percent', '%'),
                        ('首次进入开放臂潜伏期', 'First Open-Entry Latency', 'first_open_entry_latency_s', ' s'),
                    ])
                    f.write("\n")
        except Exception as e:
            print(f"Failed to write EPM report: {e}")


class YMazeMultiArenaAnalysisSession:
    """Multi-ROI Y-maze spontaneous alternation analysis session."""

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths
        self.video_infos = video_infos
        self.config = config
        self.experiment_type = 'y_maze'

        self.status = 'waiting'
        self.progress = 0
        self.message = 'Waiting to start...'
        self.logs = []
        self.realtime_data = []
        self.result_files = []
        self.processed_frames = 0
        self.is_running = False
        self.start_time = None
        self.end_time = None
        self.current_frame = None
        self.live_metrics = {}
        self.output_groups = {
            'video': [],
            'track_csv': [],
            'summary_csv': [],
            'bins_csv': [],
            'report': [],
            'heatmap': [],
            'trackmap': [],
        }

    def start(self):
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        thread = threading.Thread(target=self._run, daemon=True)
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
        if group and path not in self.output_groups[group]:
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

    def _parse_rois(self, config=None):
        active_config = config or self.config
        rois = active_config.get('rois') or []
        if isinstance(rois, str):
            rois = json.loads(rois)

        parsed = []
        for idx, roi in enumerate(rois):
            if not isinstance(roi, dict):
                raise ValueError(f'Y-maze ROI {idx + 1} must be an object.')
            zone_polygons = _coerce_zone_polygons_with_schema(roi.get('zones'), YMAZE_ZONE_POINT_COUNTS, 'Y-maze')

            roi_name = str(roi.get('name') or f'Y-Maze {idx + 1}').strip() or f'Y-Maze {idx + 1}'
            time_config = roi.get('time_config') or {'mode': 'global', 'start': None, 'end': None}
            detection_points = None
            if isinstance(roi, dict) and roi.get('detection_points'):
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)
            parsed.append({
                'roi_id': idx + 1,
                'name': roi_name,
                'zones': zone_polygons,
                'time_config': time_config,
                'detection_points': detection_points,
            })

        if not parsed:
            raise ValueError('Y-maze analysis requires at least one ROI.')
        return parsed

    @staticmethod
    def _build_live_metrics(rows):
        if not rows:
            return {}
        valid_alt = [float(row['alternation_percent']) for row in rows if row.get('alternation_percent') is not None]
        return {
            'roi_count': len(rows),
            'detected_roi_count': sum(1 for row in rows if row.get('detected')),
            'total_distance_cm': round(sum(float(row.get('distance_cm', 0.0) or 0.0) for row in rows), 3),
            'total_arm_entries': sum(int(row.get('total_arm_entries') or 0) for row in rows),
            'mean_alternation_percent': round(sum(valid_alt) / len(valid_alt), 3) if valid_alt else None,
        }

    def _run(self):
        try:
            if len(self.video_paths) == 1:
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(f"Processing Y-maze video {i + 1}/{len(self.video_paths)}: {info.get('original_name', '')}")
                    self._analyze_single_video(path, info, batch_index=i)
                    self.progress = int((i + 1) / len(self.video_paths) * 100)

                if self.is_running:
                    self.status = 'completed'
                    self.end_time = time.time()
                    self.message = f'All {len(self.video_paths)} Y-maze videos finished.'
                    self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = f'Analysis error: {str(e)}'
            self._log(self.message)
            import traceback
            self._log(traceback.format_exc())

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
        result_folder = vision_config.get('output_dir') or str(RESULT_DIR)
        output_info = _prepare_output_paths(
            self.session_id,
            video_info.get('original_name', 'video'),
            suffix='ymaze',
            result_folder=result_folder,
            fallback='video',
            batch_index=video_index,
            batch_count=len(self.video_paths),
        )

        video_name = output_info['display_stem']
        output_base = output_info['work_base']
        final_base = output_info['final_base']
        output_video_path = f"{output_base}_analyzed.mp4"
        final_output_video_path = f"{final_base}_analyzed.mp4"
        generate_video = bool(vision_config.get('generate_video', True))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to open video: {video_path}")
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log(f"Y-maze video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

        bg_method = vision_config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=vision_config.get('gray_threshold', 127),
                invert=vision_config.get('invert', False),
            )
        elif bg_method == 'static':
            bg = create_static_background(video_path, num_frames=15)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=vision_config.get('erosion_kernel', 3),
            erosion_iterations=vision_config.get('erosion_iter', 1),
            dilation_kernel=vision_config.get('dilation_kernel', 5),
            dilation_iterations=vision_config.get('dilation_iter', 2),
        )
        detector = MouseDetector(
            min_area=vision_config.get('min_area', 120),
            max_area=vision_config.get('max_area', 50000),
        )

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        roi_configs = self._parse_rois(vision_config)

        trackers = []
        for roi in roi_configs:
            trackers.append(
                YMazeTracker(
                    zone_polygons=roi['zones'],
                    detection_polygon=roi.get('detection_points'),
                    arm_length_cm=float(vision_config.get('arm_length_cm', 30.0)),
                    arm_width_cm=float(vision_config.get('arm_width_cm', 10.0)),
                    fps=fps,
                    movement_threshold_cm_s=float(vision_config.get('movement_threshold_cm_s', 2.0)),
                    entry_depth_ratio=float(vision_config.get('entry_depth_ratio', 0.2)),
                    roi_id=roi['roi_id'],
                    roi_name=roi['name'],
                    time_config=roi.get('time_config'),
                    global_start_time_s=global_start_time_s,
                )
            )
        self._log(f"Configured {len(trackers)} Y-maze arenas")

        writer = None
        if generate_video:
            for codec in ('avc1', 'H264', 'mp4v'):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    if test_writer.isOpened():
                        writer = test_writer
                        self._log(f"Y-maze video codec: {codec}")
                        break
                    test_writer.release()
                except Exception:
                    pass

        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start Y-maze analysis from frame {start_frame} ({start_frame / max(fps, 1e-6):.1f}s)")

        frames_to_process = max(0, end_frame - start_frame + 1)
        frame_idx = 0

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + frame_idx

            if current_frame_no > end_frame:
                break

            current_time = current_frame_no / max(fps, 1e-6)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = [tracker.update(frame, current_time, gray, bg_subtractor, morph_processor, detector) for tracker in trackers]

            if frame_idx % 15 == 0:
                rows = []
                for tracker, result in zip(trackers, results):
                    summary = tracker.build_summary()
                    rows.append({
                        'roi_id': tracker.roi_id,
                        'roi_name': tracker.roi_name,
                        'detected': bool(result.get('detected')),
                        'zone': result.get('zone_label', 'Unknown'),
                        'speed_cm_s': round(float(result.get('speed_cm_s', 0.0)), 3),
                        'distance_cm': summary['total_distance_cm'],
                        'total_arm_entries': summary.get('total_arm_entries'),
                        'alternation_count': summary.get('alternation_count'),
                        'alternation_percent': summary.get('alternation_percent'),
                        'in_stats_window': bool(result.get('in_stats_window', True)),
                    })
                self.realtime_data = rows
                self.live_metrics = self._build_live_metrics(rows)

            if generate_video and writer is not None:
                result_frame = draw_y_maze_results(frame, list(zip(trackers, results)))
                writer.write(result_frame)
                if frame_idx % 10 == 0:
                    self.current_frame = result_frame

            frame_idx += 1
            self.processed_frames += 1

            if frame_idx % 20 == 0:
                if batch_index is not None:
                    batch_progress = int(batch_index / len(self.video_paths) * 100)
                    video_progress = int(frame_idx / max(frames_to_process, 1) * 100 / len(self.video_paths))
                    self.progress = batch_progress + video_progress
                else:
                    self.progress = int(frame_idx / max(frames_to_process, 1) * 100)
                self.message = f'Frame {frame_idx}/{frames_to_process} ({self.progress}%)'

        cap.release()
        if writer is not None:
            writer.release()

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
            summary['video_name'] = video_info.get('original_name', video_name)
            summary_rows.append(summary)

            for row in tracker.frame_records:
                record = dict(row)
                record['video_name'] = video_info.get('original_name', video_name)
                track_rows.append(record)

            for row in tracker.build_time_bins(bin_seconds=float(vision_config.get('bin_seconds', 60.0))):
                bucket = dict(row)
                bucket['video_name'] = video_info.get('original_name', video_name)
                bins_rows.append(bucket)

        if HAS_PANDAS and track_rows:
            track_csv_work_path = f"{output_base}_track.csv"
            if _write_dataframe_csv(track_rows, track_csv_work_path):
                track_csv_path = _export_result_file(track_csv_work_path, f"{final_base}_track.csv")
                self._append_result_file(track_csv_path, 'track_csv')

        if HAS_PANDAS and summary_rows:
            summary_csv_work_path = f"{output_base}_summary.csv"
            if _write_dataframe_csv(summary_rows, summary_csv_work_path):
                summary_csv_path = _export_result_file(summary_csv_work_path, f"{final_base}_summary.csv")
                self._append_result_file(summary_csv_path, 'summary_csv')

        if HAS_PANDAS and bins_rows:
            bins_csv_work_path = f"{output_base}_bins.csv"
            if _write_dataframe_csv(bins_rows, bins_csv_work_path):
                bins_csv_path = _export_result_file(bins_csv_work_path, f"{final_base}_bins.csv")
                self._append_result_file(bins_csv_path, 'bins_csv')

        report_txt_work_path = f"{output_base}_report.txt"
        self._generate_txt_report(summary_rows, report_txt_work_path)
        report_txt_path = _export_result_file(report_txt_work_path, f"{final_base}_report.txt")
        self._append_result_file(report_txt_path, 'report')

        heatmap_work_paths = []
        heatmap_final_paths = []
        trackmap_work_paths = []
        trackmap_final_paths = []
        for tracker in trackers:
            roi_slug = sanitize_y_maze_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"ymaze_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"

            if render_y_maze_heatmap(tracker, heatmap_work_path):
                exported_heatmap = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported_heatmap)
                self._append_result_file(exported_heatmap)
            else:
                self._log(f"Failed to save Y-maze heatmap: {heatmap_final_path}")

            if render_y_maze_trackmap(tracker, trackmap_work_path):
                exported_trackmap = _export_result_file(trackmap_work_path, trackmap_final_path)
                trackmap_work_paths.append(trackmap_work_path)
                trackmap_final_paths.append(exported_trackmap)
                self._append_result_file(exported_trackmap)
            else:
                self._log(f"Failed to save Y-maze trackmap: {trackmap_final_path}")

        if heatmap_work_paths:
            if len(heatmap_final_paths) == 1:
                self._append_result_file(heatmap_final_paths[0], 'heatmap')
            else:
                heatmap_grid_work_path = f"{output_base}_heatmaps.png"
                heatmap_grid_final_path = f"{final_base}_heatmaps.png"
                if render_y_maze_image_grid(heatmap_work_paths, heatmap_grid_work_path, title=f"{video_name} | Y-maze Heatmaps"):
                    self._append_result_file(_export_result_file(heatmap_grid_work_path, heatmap_grid_final_path), 'heatmap')
                else:
                    self._log(f"Failed to save Y-maze heatmap grid: {heatmap_grid_final_path}")

        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                trackmap_grid_work_path = f"{output_base}_trackmaps.png"
                trackmap_grid_final_path = f"{final_base}_trackmaps.png"
                if render_y_maze_image_grid(trackmap_work_paths, trackmap_grid_work_path, title=f"{video_name} | Y-maze Trajectory Maps"):
                    self._append_result_file(_export_result_file(trackmap_grid_work_path, trackmap_grid_final_path), 'trackmap')
                else:
                    self._log(f"Failed to save Y-maze trackmap grid: {trackmap_grid_final_path}")

        if generate_video and os.path.exists(output_video_path):
            exported_video = _export_result_file(output_video_path, final_output_video_path)
            self._append_result_file(exported_video, 'video')
            if exported_video:
                self._log(f"Y-maze annotated video saved: {exported_video}")

        self.realtime_data = [
            {
                'roi_id': row['roi_id'],
                'roi_name': row['roi_name'],
                'detected': True,
                'zone': 'Completed',
                'speed_cm_s': row['mean_speed_cm_s'],
                'distance_cm': row['total_distance_cm'],
                'total_arm_entries': row.get('total_arm_entries'),
                'alternation_count': row.get('alternation_count'),
                'alternation_percent': row.get('alternation_percent'),
                'in_stats_window': True,
            }
            for row in summary_rows
        ]
        self.live_metrics = self._build_live_metrics(self.realtime_data)

        if batch_index is None and self.is_running:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = 'Y-maze analysis completed.'

        elapsed = time.time() - video_start_time
        self._log(f"Y-maze video processed in {elapsed:.1f}s, {frame_idx} frames")

        import gc
        gc.collect()

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        return {
            'session_id': self.session_id,
            'experiment_type': self.experiment_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'logs': self.logs[-10:],
            'realtime_data': self.realtime_data,
            'result_files': self.result_files,
            'experiment_stats': self.live_metrics,
            'stats': {
                'processed_frames': self.processed_frames,
                'elapsed_time': elapsed,
                'fps': processing_fps,
            }
        }

    def get_download_file_map(self):
        return self.output_groups

    def _generate_txt_report(self, summaries, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 72 + "\n")
                f.write("MouseScope Y 迷宫报告 (Y-Maze Report)\n")
                f.write("=" * 72 + "\n\n")
                f.write(f"{_bi_label('分析时间', 'Analysis Time')}: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if summaries:
                    f.write(f"{_bi_label('视频', 'Video')}: {summaries[0].get('video_name', '')}\n")
                f.write(f"{_bi_label('ROI 数量', 'ROI Count')}: {len(summaries)}\n")
                f.write(
                    f"{_bi_label('统计规则', 'Scoring Rule')}: "
                    "连续 3 次入臂且 3 个臂均不相同时记为一次自发交替 "
                    "(Spontaneous alternation is scored when 3 consecutive arm entries are all different.)\n\n"
                )

                for summary in summaries:
                    f.write(f"[ROI {summary.get('roi_id')}] {summary.get('roi_name', '')}\n")
                    _write_report_metric_lines(f, summary, [
                        ('总时长', 'Total Time', 'total_time_s', ' s'),
                        ('检测率', 'Detected Rate', 'detected_rate_percent', '%'),
                        ('总距离', 'Total Distance', 'total_distance_cm', ' cm'),
                        ('平均速度', 'Mean Speed', 'mean_speed_cm_s', ' cm/s'),
                        ('中心区时间', 'Center Time', 'center_time_s', ' s'),
                        ('A 臂进入次数', 'Arm A Entries', 'arm_a_entries', ''),
                        ('B 臂进入次数', 'Arm B Entries', 'arm_b_entries', ''),
                        ('C 臂进入次数', 'Arm C Entries', 'arm_c_entries', ''),
                        ('总入臂次数', 'Total Arm Entries', 'total_arm_entries', ''),
                        ('交替次数', 'Alternation Count', 'alternation_count', ''),
                        ('交替机会数', 'Alternation Opportunities', 'alternation_opportunities', ''),
                        ('交替百分比', 'Alternation Percent', 'alternation_percent', '%'),
                    ])
                    f.write("\n")
        except Exception as e:
            print(f"Failed to write Y-maze report: {e}")

# ==================== 路由 ====================

@app.route('/')
def index():
    """首页"""
    return render_template('home.html')


@app.route('/upload')
def upload_page():
    """上传视频页面"""
    return render_template('upload.html')


@app.route('/manual')
def manual():
    """使用手册"""
    manual_type = request.args.get('module', 'tst').strip().lower()
    if manual_type not in {'tst', 'oft', 'nor', 'epm', 'ymaze'}:
        manual_type = 'tst'
    return render_template('manual.html', manual_type=manual_type)


@app.route('/manual/open_field')
def manual_open_field():
    """Open field manual."""
    return_to = _safe_internal_return_to(request.args.get('return_to'))
    if return_to:
        return redirect(url_for('manual', module='oft', return_to=return_to))
    return redirect(url_for('manual', module='oft'))


@app.route('/manual/novel_object')
def manual_novel_object():
    """Novel object recognition manual."""
    return_to = _safe_internal_return_to(request.args.get('return_to'))
    if return_to:
        return redirect(url_for('manual', module='nor', return_to=return_to))
    return redirect(url_for('manual', module='nor'))


@app.route('/license')
def license_page():
    """许可协议页面"""
    license_path = BUNDLE_DIR / 'LICENSE'
    license_text = ''
    if license_path.exists():
        license_text = license_path.read_text(encoding='utf-8')
    return render_template('license.html', license_text=license_text)


@app.route('/api/license', methods=['GET'])
def api_get_license():
    """获取许可协议内容"""
    license_path = BUNDLE_DIR / 'LICENSE'
    if license_path.exists():
        return jsonify({'success': True, 'text': license_path.read_text(encoding='utf-8')})
    return jsonify({'success': False, 'error': '许可协议文件不存在'}), 404


@app.route('/api/accept_license', methods=['POST'])
def api_accept_license():
    """用户同意许可协议"""
    session['license_accepted'] = True
    return jsonify({'success': True})


@app.route('/api/upload_files', methods=['POST'])
def api_upload_files():
    """处理拖拽/选择的文件上传，返回服务器路径"""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'error': '没有接收到文件'}), 400

    saved_paths = []
    saved_names = []
    upload_id = secrets.token_hex(8)

    for f in files:
        if f and f.filename:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in VALID_VIDEO_EXTS:
                continue
            save_path, original_name = _save_uploaded_video(f, upload_id)
            saved_paths.append(save_path)
            saved_names.append(original_name)

    if not saved_paths:
        return jsonify({'success': False, 'error': '没有有效的视频文件'}), 400

    return jsonify({'success': True, 'paths': saved_paths, 'names': saved_names})


@app.route('/do_upload', methods=['POST'])
def do_upload():
    """处理视频上传"""
    session_id = secrets.token_hex(16)
    uploaded_files = request.files.getlist('videos')
    local_paths_raw = request.form.get('local_paths', '').strip()
    experiment_type = normalize_experiment_type(request.form.get('experiment_type', 'tst'))

    video_paths = []
    video_infos = []

    # 处理上传的文件
    for f in uploaded_files:
        if f and f.filename:
            save_path, original_name = _save_uploaded_video(f, session_id)
            video_paths.append(save_path)
            video_infos.append({'path': save_path, 'original_name': original_name})

    # 处理本地路径
    if local_paths_raw:
        for line in local_paths_raw.split('\n'):
            p = line.strip().strip('"').strip("'").strip('“”').strip('‘’').strip()
            if not p:
                continue
            p = Path(p).expanduser()
            if p.is_file() and p.suffix.lower() in VALID_VIDEO_EXTS:
                staged_path, original_name = _stage_local_video_if_needed(
                    p,
                    session_id,
                    original_name=_get_original_name_for_path(p) or p.name,
                )
                video_paths.append(staged_path)
                video_infos.append({'path': staged_path, 'original_name': original_name, 'source_path': str(p)})
            elif p.is_dir():
                for vf in sorted(p.iterdir()):
                    if vf.is_file() and vf.suffix.lower() in VALID_VIDEO_EXTS:
                        staged_path, original_name = _stage_local_video_if_needed(vf, session_id, original_name=vf.name)
                        video_paths.append(staged_path)
                        video_infos.append({'path': staged_path, 'original_name': original_name, 'source_path': str(vf)})

    if not video_paths:
        flash('请选择或输入至少一个视频文件', 'error')
        return redirect(url_for('upload_page'))

    # 获取时间配置
    start_seconds = request.form.get('start_seconds', '').strip()
    end_seconds = request.form.get('end_seconds', '').strip()
    output_dir = request.form.get('output_dir', '').strip().strip('"').strip("'")
    resolved_output_dir = _resolve_result_dir(output_dir) if output_dir else None
    fps_override = request.form.get('fps_override', '').strip()
    fps_override_value = float(fps_override) if fps_override else None

    for info in video_infos:
        _update_video_entry_info(info, fps_override=fps_override_value)

    # 读取第一个视频的基本信息
    video_info_data = _extract_video_info_from_entry(
        video_infos[0],
        fps_override=fps_override_value,
    )

    # 保存到会话
    analysis_sessions[session_id] = {
        'experiment_type': experiment_type,
        'video_paths': video_paths,
        'video_infos': video_infos,
        'video_info': video_info_data,
        'start_seconds': start_seconds if start_seconds else None,
        'end_seconds': end_seconds if end_seconds else None,
        'output_dir': resolved_output_dir,
        'fps_override': fps_override_value,
        'is_batch': len(video_paths) > 1,
        'draft_payload': None,
        'draft_updated_at': None,
        'selected_video_idx': 0,
        'created_at': _now_ts(),
        'last_accessed_at': _now_ts(),
    }

    if len(video_paths) > 1:
        return redirect(url_for('batch_workspace', session_id=session_id))
    return redirect(url_for('calibrate', session_id=session_id))


def _render_calibrate_page(session_id, sess, video_idx=0, launch_mode='direct'):
    sess = analysis_sessions.get(session_id)
    if not sess:
        flash('会话不存在或已过期，请重新上传', 'error')
        return redirect(url_for('index'))

    total_videos = len(sess.get('video_infos') or [])
    if total_videos <= 0:
        flash('会话中没有可用视频，请重新上传', 'error')
        return redirect(url_for('upload_page'))

    video_idx = min(max(int(video_idx or 0), 0), total_videos - 1)
    video_list = []
    if sess['is_batch']:
        for i, info in enumerate(sess['video_infos']):
            video_list.append({
                'index': i,
                'name': info['original_name'],
                'video_info': _extract_video_info_from_entry(info, fps_override=sess.get('fps_override')),
            })

    spec = get_experiment_spec(sess.get('experiment_type'))
    return render_template(
        spec.calibrate_template,
        session_id=session_id,
        video_info=_session_video_info(sess, video_idx),
        video_info_list=[_extract_video_info_from_entry(info, fps_override=sess.get('fps_override')) for info in sess['video_infos']],
        video_path=sess['video_paths'][video_idx],
        is_batch=sess['is_batch'],
        video_list=video_list,
        experiment_type=sess.get('experiment_type', 'tst'),
        maze_module=spec.maze_module,
        experiment_spec=spec,
        current_video_index=video_idx,
        launch_mode=launch_mode,
        batch_workspace_url=url_for('batch_workspace', session_id=session_id) if sess.get('is_batch') else None,
        draft_payload=sess.get('draft_payload'),
    )


@app.route('/batch/<session_id>')
def batch_workspace(session_id):
    sess = analysis_sessions.get(session_id)
    if not sess:
        flash('会话不存在或已过期，请重新上传', 'error')
        return redirect(url_for('index'))
    if not sess.get('is_batch'):
        return redirect(url_for('calibrate', session_id=session_id))
    spec = get_experiment_spec(sess.get('experiment_type'))
    cards = summarize_batch_state(
        sess.get('experiment_type'),
        sess.get('draft_payload'),
        sess.get('video_infos'),
    )
    complete_count = sum(1 for card in cards if card.get('complete'))
    return render_template(
        'batch_workspace.html',
        session_id=session_id,
        experiment_spec=spec,
        cards=cards,
        complete_count=complete_count,
        total_videos=len(cards),
        draft_updated_at=sess.get('draft_updated_at'),
        start_seconds=sess.get('start_seconds'),
        end_seconds=sess.get('end_seconds'),
        output_dir=sess.get('output_dir'),
        fps_override=sess.get('fps_override'),
        batch_state_url=url_for('api_batch_state', session_id=session_id),
    )


@app.route('/calibrate/<session_id>')
def calibrate(session_id):
    """标定页面（ROI + 参数调试）"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        flash('会话不存在或已过期，请重新上传', 'error')
        return redirect(url_for('index'))
    if sess.get('is_batch'):
        selected_video_idx = int(sess.get('selected_video_idx') or 0)
        return redirect(url_for('calibrate_video', session_id=session_id, video_idx=selected_video_idx))
    return _render_calibrate_page(session_id, sess, video_idx=0, launch_mode='direct')


@app.route('/calibrate/<session_id>/video/<int:video_idx>')
def calibrate_video(session_id, video_idx):
    sess = analysis_sessions.get(session_id)
    if not sess:
        flash('会话不存在或已过期，请重新上传', 'error')
        return redirect(url_for('index'))
    if not sess.get('is_batch'):
        return redirect(url_for('calibrate', session_id=session_id))
    sess['selected_video_idx'] = int(video_idx)
    analysis_sessions.persist(session_id)
    return _render_calibrate_page(session_id, sess, video_idx=video_idx, launch_mode='workspace')


@app.route('/monitor/<session_id>')
def monitor(session_id):
    """监控分析进度"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        flash('Session expired. Please start again.', 'error')
        return redirect(url_for('index'))
    spec = get_experiment_spec(sess.get('experiment_type'))
    return render_template(
        spec.monitor_template,
        session_id=session_id,
        maze_module=spec.maze_module,
        experiment_spec=spec,
        is_batch=sess.get('is_batch', False),
    )


# ==================== API 路由 ====================

@app.route('/api/get_first_frame/<session_id>')
def get_first_frame(session_id):
    """获取视频帧（用于标定页面）"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    frame_idx = int(request.args.get('frame', 0))
    video_idx = int(request.args.get('video', 0))

    video_path = sess['video_paths'][min(video_idx, len(sess['video_paths'])-1)]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': '无法读取帧'}), 500

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': img_b64,
        'frame_idx': frame_idx,
        'total_frames': _session_video_info(sess, video_idx)['total_frames'],
        'video_info': _session_video_info(sess, video_idx),
    })


@app.route('/api/batch_state/<session_id>')
def api_batch_state(session_id):
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404
    cards = summarize_batch_state(
        sess.get('experiment_type'),
        sess.get('draft_payload'),
        sess.get('video_infos'),
    )
    complete_count = sum(1 for card in cards if card.get('complete'))
    return jsonify({
        'session_id': session_id,
        'experiment_type': sess.get('experiment_type'),
        'is_batch': bool(sess.get('is_batch')),
        'cards': cards,
        'complete_count': complete_count,
        'total_videos': len(cards),
        'can_start': bool(cards) and complete_count == len(cards),
        'draft_updated_at': sess.get('draft_updated_at'),
        'selected_video_idx': int(sess.get('selected_video_idx') or 0),
    })


@app.route('/api/draft/<session_id>', methods=['GET', 'POST'])
def api_draft_payload(session_id):
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    if request.method == 'GET':
        return jsonify({
            'session_id': session_id,
            'draft_payload': sess.get('draft_payload'),
            'draft_updated_at': sess.get('draft_updated_at'),
            'selected_video_idx': int(sess.get('selected_video_idx') or 0),
        })

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': '草稿数据格式无效'}), 400

    selected_video_idx = payload.pop('_selected_video_idx', None)
    analysis_sessions.save_draft(session_id, payload, selected_video_idx=selected_video_idx)
    cards = summarize_batch_state(
        sess.get('experiment_type'),
        payload,
        sess.get('video_infos'),
    )
    return jsonify({
        'success': True,
        'draft_updated_at': sess.get('draft_updated_at'),
        'cards': cards,
    })


@app.route('/api/draft_action/<session_id>', methods=['POST'])
def api_draft_action(session_id):
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    payload = request.get_json(silent=True) or {}
    action = str(payload.get('action') or '').strip().lower()
    draft_payload = copy.deepcopy(sess.get('draft_payload') or {})
    total_videos = len(sess.get('video_infos') or [])
    per_video_configs = draft_payload.get('per_video_configs')
    if not isinstance(per_video_configs, list):
        per_video_configs = []
    while len(per_video_configs) < total_videos:
        per_video_configs.append({})

    if action == 'copy':
        source_index = int(payload.get('source_index', -1))
        target_indices = payload.get('target_indices') or []
        if not (0 <= source_index < total_videos):
            return jsonify({'error': '源视频索引无效'}), 400
        if not isinstance(target_indices, list) or not target_indices:
            return jsonify({'error': '目标视频索引无效'}), 400
        source_config = copy.deepcopy(per_video_configs[source_index] or {})
        for raw_idx in target_indices:
            idx = int(raw_idx)
            if 0 <= idx < total_videos:
                per_video_configs[idx] = copy.deepcopy(source_config)
    elif action == 'reset':
        target_index = int(payload.get('target_index', -1))
        if not (0 <= target_index < total_videos):
            return jsonify({'error': '目标视频索引无效'}), 400
        per_video_configs[target_index] = {}
    else:
        return jsonify({'error': '不支持的草稿操作'}), 400

    draft_payload['per_video_configs'] = per_video_configs
    analysis_sessions.save_draft(session_id, draft_payload)
    cards = summarize_batch_state(
        sess.get('experiment_type'),
        draft_payload,
        sess.get('video_infos'),
    )
    return jsonify({
        'success': True,
        'draft_updated_at': sess.get('draft_updated_at'),
        'cards': cards,
    })


def _open_field_debug_preview(frame, data, video_path):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 120))
    max_area = int(data.get('max_area', 50000))
    rois = data.get('rois') or []
    if not rois and data.get('arena_points'):
        rois = [{
            'points': data.get('arena_points'),
            'name': 'Arena 1',
            'time_config': {'mode': 'global', 'start': None, 'end': None},
        }]
    if not rois:
        return jsonify({'error': 'Open field preview requires at least one ROI.'}), 400

    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter
    )
    det = MouseDetector(min_area=min_area, max_area=max_area)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    arena_side_cm = float(data.get('arena_side_cm', data.get('arena_width_cm', 50.0)))

    tracker_results = []
    detected_count = 0
    for idx, roi in enumerate(rois):
        points = roi.get('points', roi) if isinstance(roi, dict) else roi
        if len(points) != 4:
            return jsonify({'error': f'ROI {idx + 1} must contain exactly 4 corner points.'}), 400

        tracker = OpenFieldTracker(
            arena_points=[[p['x'], p['y']] if isinstance(p, dict) else p for p in points],
            detection_points=(([[p['x'], p['y']] if isinstance(p, dict) else p for p in roi.get('detection_points', [])]) if isinstance(roi, dict) and roi.get('detection_points') else None),
            arena_side_cm=arena_side_cm,
            fps=30.0,
            center_ratio_x=float(data.get('center_ratio_x', 0.5)),
            center_ratio_y=float(data.get('center_ratio_y', 0.5)),
            wall_margin_cm=float(data.get('wall_margin_cm', 8.0)),
            movement_threshold_cm_s=float(data.get('movement_threshold_cm_s', 2.0)),
            roi_id=idx + 1,
            roi_name=(roi.get('name') if isinstance(roi, dict) else None) or f'Arena {idx + 1}',
            time_config=(roi.get('time_config') if isinstance(roi, dict) else None),
        )
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))

    result_frame = draw_open_field_results(frame, tracker_results)

    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detected': detected_count > 0,
        'detections': detected_count,
        'arena_count': len(tracker_results),
    })


def _novel_object_debug_preview(frame, data, video_path):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 120))
    rois = data.get('rois') or []
    if not rois:
        return jsonify({'error': 'Novel object preview requires at least one ROI.'}), 400

    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter
    )
    det = MouseDetector(min_area=min_area, max_area=50000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tracker_results = []
    detected_count = 0
    phase = data.get('phase', 'recognition')
    arena_width_cm = float(data.get('arena_width_cm', data.get('arena_side_cm', 50.0)))
    arena_height_cm = float(data.get('arena_height_cm', data.get('arena_side_cm', 50.0)))

    for idx, roi in enumerate(rois):
        points = roi.get('points', roi) if isinstance(roi, dict) else roi
        if len(points) != 4:
            return jsonify({'error': f'ROI {idx + 1} must contain exactly 4 arena corner points.'}), 400

        objects = roi.get('objects') if isinstance(roi, dict) else None
        if not objects or len(objects) != 2:
            return jsonify({'error': f'ROI {idx + 1} must contain exactly 2 object points.'}), 400

        tracker = NovelObjectTracker(
            arena_points=[[p['x'], p['y']] if isinstance(p, dict) else p for p in points],
            object_points=objects,
            detection_points=(([[p['x'], p['y']] if isinstance(p, dict) else p for p in roi.get('detection_points', [])]) if isinstance(roi, dict) and roi.get('detection_points') else None),
            arena_width_cm=arena_width_cm,
            arena_height_cm=arena_height_cm,
            fps=30.0,
            roi_id=idx + 1,
            roi_name=(roi.get('name') if isinstance(roi, dict) else None) or f'ROI {idx + 1}',
            phase=phase,
            time_config=(roi.get('time_config') if isinstance(roi, dict) else None),
            center_ratio_x=float(data.get('center_ratio_x', 0.5)),
            center_ratio_y=float(data.get('center_ratio_y', 0.5)),
            interaction_radius_cm=float(data.get('interaction_radius_cm', 4.0)),
            exploration_max_speed_cm_s=float(data.get('exploration_max_speed_cm_s', 12.0)),
            use_head_direction=bool(data.get('use_head_direction', True)),
            head_angle_threshold_deg=float(data.get('head_angle_threshold_deg', 50.0)),
            head_confidence_threshold=float(data.get('head_confidence_threshold', 0.35)),
            movement_threshold_cm_s=float(data.get('movement_threshold_cm_s', 2.0)),
            min_valid_exploration_s=float(data.get('min_valid_exploration_s', 20.0)),
        )
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))

    result_frame = draw_novel_object_results(frame, tracker_results)
    exploring_count = sum(1 for _, result in tracker_results if result.get('current_object_name'))

    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detected': detected_count > 0,
        'detections': detected_count,
        'roi_count': len(tracker_results),
        'exploring': exploring_count,
    })


def _elevated_plus_debug_preview(frame, data, video_path):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 120))
    rois = data.get('rois') or []
    if not rois:
        return jsonify({'error': 'Elevated plus maze preview requires at least one ROI.'}), 400

    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter
    )
    det = MouseDetector(min_area=min_area, max_area=50000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tracker_results = []
    detected_count = 0
    for idx, roi in enumerate(rois):
        if not isinstance(roi, dict):
            return jsonify({'error': f'EPM ROI {idx + 1} must be an object.'}), 400
        try:
            zone_polygons = _coerce_zone_polygons(roi.get('zones'), EPM_ZONE_KEYS, 'EPM')
        except Exception as exc:
            return jsonify({'error': f'EPM ROI {idx + 1}: {exc}'}), 400
        detection_points = None
        if isinstance(roi, dict) and roi.get('detection_points'):
            try:
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)
            except Exception as exc:
                return jsonify({'error': f'EPM ROI {idx + 1}: detection mask {exc}'}), 400

        tracker = ElevatedPlusTracker(
            zone_polygons=zone_polygons,
            detection_polygon=detection_points,
            center_size_cm=float(data.get('center_size_cm', 5.0)),
            open_arm_length_cm=float(data.get('open_arm_length_cm', 30.0)),
            closed_arm_length_cm=float(data.get('closed_arm_length_cm', 30.0)),
            fps=30.0,
            movement_threshold_cm_s=float(data.get('movement_threshold_cm_s', 2.0)),
            entry_depth_ratio=float(data.get('entry_depth_ratio', 0.2)),
            roi_id=idx + 1,
            roi_name=(roi.get('name') if isinstance(roi, dict) else None) or f'EPM {idx + 1}',
            time_config=(roi.get('time_config') if isinstance(roi, dict) else None),
        )
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))

    result_frame = draw_elevated_plus_results(frame, tracker_results)
    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detected': detected_count > 0,
        'detections': detected_count,
        'roi_count': len(tracker_results),
    })


def _y_maze_debug_preview(frame, data, video_path):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 120))
    rois = data.get('rois') or []
    if not rois:
        return jsonify({'error': 'Y-maze preview requires at least one ROI.'}), 400

    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter
    )
    det = MouseDetector(min_area=min_area, max_area=50000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tracker_results = []
    detected_count = 0
    for idx, roi in enumerate(rois):
        if not isinstance(roi, dict):
            return jsonify({'error': f'Y-maze ROI {idx + 1} must be an object.'}), 400
        try:
            zone_polygons = _coerce_zone_polygons_with_schema(roi.get('zones'), YMAZE_ZONE_POINT_COUNTS, 'Y-maze')
        except Exception as exc:
            return jsonify({'error': f'Y-maze ROI {idx + 1}: {exc}'}), 400
        detection_points = None
        if isinstance(roi, dict) and roi.get('detection_points'):
            try:
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)
            except Exception as exc:
                return jsonify({'error': f'Y-maze ROI {idx + 1}: detection mask {exc}'}), 400

        tracker = YMazeTracker(
            zone_polygons=zone_polygons,
            detection_polygon=detection_points,
            arm_length_cm=float(data.get('arm_length_cm', 30.0)),
            arm_width_cm=float(data.get('arm_width_cm', 10.0)),
            fps=30.0,
            movement_threshold_cm_s=float(data.get('movement_threshold_cm_s', 2.0)),
            entry_depth_ratio=float(data.get('entry_depth_ratio', 0.2)),
            roi_id=idx + 1,
            roi_name=(roi.get('name') if isinstance(roi, dict) else None) or f'Y-Maze {idx + 1}',
            time_config=(roi.get('time_config') if isinstance(roi, dict) else None),
        )
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))

    result_frame = draw_y_maze_results(frame, tracker_results)
    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detected': detected_count > 0,
        'detections': detected_count,
        'roi_count': len(tracker_results),
    })


@app.route('/api/debug_preview/<session_id>', methods=['POST'])
def debug_preview(session_id):
    """参数调试预览"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    data = request.get_json()
    frame_idx = int(data.get('frame', 0))
    video_idx = int(data.get('video', 0))

    video_path = sess['video_paths'][min(video_idx, len(sess['video_paths'])-1)]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': '无法读取帧'}), 500

    # 使用传统检测
    if sess.get('experiment_type') == 'open_field':
        return _open_field_debug_preview(frame, data, video_path)
    if sess.get('experiment_type') == 'novel_object':
        return _novel_object_debug_preview(frame, data, video_path)
    if sess.get('experiment_type') == 'elevated_plus':
        return _elevated_plus_debug_preview(frame, data, video_path)
    if sess.get('experiment_type') == 'y_maze':
        return _y_maze_debug_preview(frame, data, video_path)

    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')

    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 2))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 3))
    min_area = int(data.get('min_area', 100))

    # ROI 掩码
    rois = data.get('rois', [])

    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter
    )
    det = MouseDetector(min_area=min_area, max_area=50000)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = bg_sub.apply(frame, gray=gray)
    fg_mask = morph.process(fg_mask)

    # 应用 ROI 掩码
    if rois:
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for roi in rois:
            pts = roi if isinstance(roi, list) else roi.get('points', [])
            if pts:
                polygon = np.array([[p['x'], p['y']] for p in pts], dtype=np.int32)
                cv2.fillPoly(roi_mask, [polygon], 255)
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

    detections = det.detect(fg_mask)

    # 绘制结果
    result_frame = frame.copy()
    for d in detections:
        cx, cy = d['centroid']
        cv2.circle(result_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        if d.get('contour') is not None:
            cv2.drawContours(result_frame, [d['contour']], -1, (0, 255, 0), 2)

    # 绘制 ROI
    if rois:
        for roi in rois:
            pts = roi if isinstance(roi, list) else roi.get('points', [])
            if pts:
                polygon = np.array([[p['x'], p['y']] for p in pts], dtype=np.int32)
                cv2.polylines(result_frame, [polygon], True, (255, 255, 0), 2)

    # 返回两张图
    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detections': len(detections),
    })


@app.route('/api/start_analysis/<session_id>', methods=['POST'])
def start_analysis(session_id):
    """启动分析"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}
    use_saved_draft = bool(data.get('use_saved_draft'))
    if use_saved_draft:
        draft_payload = copy.deepcopy(sess.get('draft_payload') or {})
        if not draft_payload:
            return jsonify({'error': '当前没有可用于启动分析的已保存配置'}), 400
        data = draft_payload
    elif not data and sess.get('draft_payload'):
        data = copy.deepcopy(sess.get('draft_payload') or {})

    if isinstance(data, dict) and data:
        selected_video_idx = data.get('video', sess.get('selected_video_idx'))
        analysis_sessions.save_draft(session_id, data, selected_video_idx=selected_video_idx)

    experiment_type = str(sess.get('experiment_type') or 'tst').strip().lower()
    vision_config = build_analysis_config(experiment_type, sess, data)
    analysis_class_map = {
        'tst': VisionAnalysisSession,
        'open_field': OpenFieldMultiArenaAnalysisSession,
        'novel_object': NovelObjectMultiArenaAnalysisSession,
        'elevated_plus': ElevatedPlusMultiArenaAnalysisSession,
        'y_maze': YMazeMultiArenaAnalysisSession,
    }
    analysis_class = analysis_class_map.get(experiment_type, VisionAnalysisSession)
    analysis = analysis_class(
        session_id=session_id,
        video_paths=sess['video_paths'],
        video_infos=sess['video_infos'],
        config=vision_config
    )

    analysis_sessions.set_analysis(session_id, analysis)
    analysis.start()

    return jsonify({'success': True, 'session_id': session_id})


@app.route('/api/status/<session_id>')
def get_status(session_id):
    """获取分析状态"""
    sess = analysis_sessions.get(session_id)
    if not sess or 'analysis' not in sess:
        return jsonify({'error': '会话不存在'}), 404
    return jsonify(sess['analysis'].get_status())


@app.route('/api/stop/<session_id>', methods=['POST'])
def stop_analysis(session_id):
    """停止分析"""
    sess = analysis_sessions.get(session_id)
    if sess and 'analysis' in sess:
        sess['analysis'].stop()
        return jsonify({'message': '已发送停止命令'})
    return jsonify({'error': '会话不存在'}), 404


@app.route('/api/download_all/<session_id>')
def download_all(session_id):
    """下载所有结果（ZIP）"""
    sess = analysis_sessions.get(session_id)
    if not sess or 'analysis' not in sess:
        return jsonify({'error': '会话不存在'}), 404

    analysis = sess['analysis']
    if not analysis.result_files:
        return jsonify({'error': '没有结果文件'}), 404

    result_dir = _resolve_result_dir(sess.get('output_dir'))
    zip_path = str(RESULT_DIR / f'{session_id}_results.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in analysis.result_files:
            if os.path.exists(f):
                zf.write(f, _result_zip_arcname(f, result_dir))

    return send_file(zip_path, as_attachment=True, download_name='MouseScope_results.zip')


@app.route('/api/download/<session_id>/<file_type>')
def download_file(session_id, file_type):
    """下载单个结果文件"""
    sess = analysis_sessions.get(session_id)
    if not sess or 'analysis' not in sess:
        return jsonify({'error': '会话不存在'}), 404

    analysis = sess['analysis']
    if hasattr(analysis, 'get_download_file_map'):
        file_map = analysis.get_download_file_map()
    else:
        file_map = {
            'video': [f for f in analysis.result_files if f.endswith('.mp4')],
            'csv': [analysis.csv_path] if analysis.csv_path else [],
            'immobility_csv': [analysis.immobility_csv_path] if analysis.immobility_csv_path else [],
            'report': [analysis.immobility_txt_path] if analysis.immobility_txt_path else [],
        }

    files = file_map.get(file_type, [])
    files = [f for f in files if f and os.path.exists(f)]

    if not files:
        return jsonify({'error': '文件不存在'}), 404

    if len(files) == 1:
        return send_file(files[0], as_attachment=True)

    result_dir = _resolve_result_dir(sess.get('output_dir'))
    os.makedirs(result_dir, exist_ok=True)
    zip_path = os.path.join(result_dir, f'{session_id}_{file_type}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, _result_zip_arcname(path, result_dir))
    return send_file(zip_path, as_attachment=True)


@app.route('/api/open_folder/<session_id>', methods=['POST'])
def open_folder(session_id):
    """用系统文件管理器打开结果文件夹（桌面端专属功能）"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    # 确定结果文件夹
    result_dir = _resolve_result_dir(sess.get('output_dir'))

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    try:
        system = platform.system()
        if system == 'Linux':
            subprocess.Popen(['xdg-open', result_dir])
        elif system == 'Darwin':
            subprocess.Popen(['open', result_dir])
        elif system == 'Windows':
            os.startfile(result_dir)
        else:
            return jsonify({'error': f'不支持的操作系统: {system}'}), 500

        return jsonify({'success': True, 'path': result_dir})
    except Exception as e:
        return jsonify({'error': f'打开文件夹失败: {str(e)}', 'path': result_dir}), 500


@app.route('/api/result_dir/<session_id>')
def get_result_dir(session_id):
    """获取结果文件夹路径"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    result_dir = _resolve_result_dir(sess.get('output_dir'))
    return jsonify({'path': result_dir})





# ==================== 启动 ====================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MouseScope Behavior Analysis Platform')
    parser.add_argument('--port', type=int, default=8080, help='端口号')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='监听地址')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    args = parser.parse_args()
    host_explicit = any(arg == '--host' or str(arg).startswith('--host=') for arg in sys.argv[1:])
    if (not host_explicit) and (not args.host or str(args.host).strip() == '0.0.0.0'):
        args.host = '127.0.0.1'

    # 获取本机局域网IP
    def get_local_ip():
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return '127.0.0.1'

    local_ip = get_local_ip()
    bind_host = str(args.host or '127.0.0.1').strip()
    browser_host = '127.0.0.1' if bind_host in {'0.0.0.0', '::', ''} else bind_host
    browse_url = f'http://{browser_host}:{args.port}'

    print(f"\n{'='*60}")
    print(f"  MouseScope Behavior Analysis Platform")
    print(f"  访问地址: {browse_url}")
    if bind_host not in {'127.0.0.1', 'localhost', '::1'}:
        print(f"  WARNING: remote access enabled on host {bind_host}.")
    print(f"{'='*60}\n")

    if not args.no_browser:
        import webbrowser
        import socket as _socket

        def _wait_and_open(url, host, port, timeout=30):
            """等待 Flask 端口真正就绪后再打开浏览器，避免竞态条件"""
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    target_host = browser_host if host in {'0.0.0.0', '::'} else host
                    with _socket.create_connection((target_host, port), timeout=0.5):
                        pass
                    webbrowser.open(url)
                    return
                except OSError:
                    time.sleep(0.2)
            # 超时仍未就绪，兜底直接打开
            webbrowser.open(url)

        threading.Thread(
            target=_wait_and_open,
            args=(browse_url, args.host, args.port),
            daemon=True
        ).start()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
