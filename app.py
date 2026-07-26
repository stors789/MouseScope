#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MouseScope Behavior Analysis Platform — 简化版 Flask 应用
去除：用户认证、数据库、YOLO、多语言
保留：视频上传、ROI标定、传统CV分析、结果下载
"""

import os
import sys
import logging
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


def _configure_http_access_logging():
    """Show backend request logs by default during local desktop runs."""
    raw = str(os.environ.get('MOUSESCOPE_HTTP_ACCESS_LOG', '')).strip().lower()
    enabled = raw not in {'0', 'false', 'no', 'off', 'quiet'}
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(name)s: %(message)s')
    logging.getLogger('werkzeug').setLevel(logging.INFO if enabled else logging.WARNING)


_configure_http_access_logging()

# ==================== 性能环境变量（必须在 import numpy/cv2 之前） ====================
# 让底层 BLAS/OpenMP 库充分利用所有 CPU 核心
from services.resource_policy import current_resource_policy

_startup_resource_policy = current_resource_policy()
_native_thread_count = str(_startup_resource_policy.opencv_threads)
os.environ.setdefault('OMP_NUM_THREADS', _native_thread_count)
os.environ.setdefault('OPENBLAS_NUM_THREADS', _native_thread_count)
os.environ.setdefault('MKL_NUM_THREADS', _native_thread_count)

import secrets
import cv2
import errno
import threading
import time
import shutil
import hashlib
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from flask import (
    Flask, request, redirect, url_for,
    flash, jsonify, session
)
from werkzeug.exceptions import RequestEntityTooLarge

# 将 core/ 加入路径（兼容 PyInstaller 打包环境）
if getattr(sys, 'frozen', False):
    # 打包后：core/ 在 sys._MEIPASS 解压目录下
    _base = sys._MEIPASS
    sys.path.insert(0, os.path.join(_base, 'core'))
    sys.path.insert(0, _base)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from file_utils import (
    build_ascii_file_name,
    clean_display_name,
    clean_display_stem,
    contains_non_ascii,
    normalized_path_key,
)
from performance_utils import enable_opencv_optimizations
from services.session_store import SessionStore
from services.analysis_history import AnalysisHistoryStore
from services.project_store import ProjectStore
from services.video_metadata import probe_video_info
from services.experiment_registry import (
    get_manual_module,
    is_three_chamber_enabled,
    list_experiments,
)
from services.analysis_sessions import configure_analysis_runtime
from services.debug_preview import cleanup_debug_preview_cache
from services.visual_style_store import VisualStyleStore
from services.experiment_defaults_store import ExperimentDefaultsStore
from services.reproducibility import finalize_reproducibility_artifacts
from services.version import APP_VERSION, APP_VERSION_TAG
from services.app_paths import resolve_application_data_directory

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠ pandas 未安装，将无法生成 CSV 报告")

# ==================== 配置 ====================
# PyInstaller 打包兼容：资源文件路径
IS_FROZEN = bool(getattr(sys, 'frozen', False))
if IS_FROZEN:
    # 打包后的 exe 运行环境
    BUNDLE_DIR = Path(sys._MEIPASS)          # 资源解压目录（templates, static 等）
    BASE_DIR = Path(sys.executable).parent   # exe 所在目录（用于存放用户数据）
else:
    BUNDLE_DIR = Path(__file__).parent
    BASE_DIR = BUNDLE_DIR

DATA_DIR = resolve_application_data_directory(BASE_DIR, frozen=IS_FROZEN)
UPLOAD_DIR = DATA_DIR / 'uploads'
RESULT_DIR = DATA_DIR / 'results'
WORK_DIR = RESULT_DIR / '_work'
HISTORY_DIR = DATA_DIR / 'history'
PROJECTS_DIR = DATA_DIR / 'projects'

for d in [DATA_DIR, UPLOAD_DIR, RESULT_DIR, WORK_DIR, HISTORY_DIR, PROJECTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHINA_TZ = timezone(timedelta(hours=8))
CSV_ENCODING = 'utf-8-sig'
VALID_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
staged_original_names = {}
SESSION_PENDING_TTL_SECONDS = 2 * 60 * 60
SESSION_FINISHED_TTL_SECONDS = 12 * 60 * 60
SESSION_ERROR_TTL_SECONDS = 24 * 60 * 60
DEBUG_STATIC_BG_CACHE_TTL_SECONDS = 30 * 60
DEBUG_STATIC_BG_CACHE_MAX_ITEMS = 32
UPLOAD_FREE_SPACE_BUFFER_BYTES = 512 * 1024 * 1024
UPLOAD_ORPHAN_STAGING_TTL_SECONDS = 24 * 60 * 60
LICENSE_EXEMPT_ENDPOINTS = {
    'static',
    'license_page',
    'api_get_license',
    'api_accept_license',
    'desktop_quit',
}


class UploadStorageError(RuntimeError):
    """Raised when MouseScope cannot safely store an uploaded video."""

    def __init__(self, message, path=None, required_bytes=None, free_bytes=None):
        super().__init__(message)
        self.path = path
        self.required_bytes = required_bytes
        self.free_bytes = free_bytes


def china_now():
    return datetime.now(CHINA_TZ)


def _load_or_create_secret_key():
    env_value = os.environ.get('MOUSESCOPE_SECRET_KEY')
    if env_value:
        return env_value

    secret_path = DATA_DIR / '.flask_secret_key'
    try:
        if secret_path.exists():
            value = secret_path.read_text(encoding='utf-8').strip()
            if value:
                return value
        value = secrets.token_hex(32)
        secret_path.write_text(value, encoding='utf-8')
        return value
    except Exception:
        return secrets.token_hex(32)


def _now_ts():
    return time.time()


def _safe_runtime_ts(value, default=None):
    if value is None or value == '':
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            text = value.strip()
            if text.endswith('Z'):
                text = f"{text[:-1]}+00:00"
            try:
                return datetime.fromisoformat(text).timestamp()
            except ValueError:
                return default
        return default


def _bi_label(zh_text, en_text):
    return f"{zh_text} ({en_text})"


def _report_value(value, suffix=''):
    if value is None or value == '':
        return '-'
    return f"{value}{suffix}"


def _format_storage_bytes(value):
    if value is None:
        return '未知'
    size = float(max(int(value), 0))
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _nearest_existing_path(path):
    current = Path(path)
    if current.is_file():
        current = current.parent
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _free_bytes_for_path(path):
    try:
        usage = shutil.disk_usage(str(_nearest_existing_path(path)))
        return int(usage.free)
    except Exception:
        return None


def _storage_error_message(path, required_bytes=None, free_bytes=None):
    needed = int(required_bytes or 0) + UPLOAD_FREE_SPACE_BUFFER_BYTES
    required_text = _format_storage_bytes(needed) if needed else '未知'
    free_text = _format_storage_bytes(free_bytes)
    target = Path(path)
    return (
        f"磁盘空间不足，无法保存上传视频。当前可用空间约 {free_text}，"
        f"本次保存建议预留 {required_text}。请清理 MouseScope 所在磁盘的 "
        f"data/uploads 或 data/results 目录，或把软件/输出目录放到空间更大的磁盘后重试。"
        f"保存位置: {target}"
    )


def _ensure_upload_storage_available(path, required_bytes=None):
    free_bytes = _free_bytes_for_path(path)
    if free_bytes is None:
        return
    required = int(required_bytes or 0)
    threshold = required + UPLOAD_FREE_SPACE_BUFFER_BYTES
    if free_bytes < threshold:
        raise UploadStorageError(
            _storage_error_message(path, required, free_bytes),
            path=str(path),
            required_bytes=required,
            free_bytes=free_bytes,
        )


def _is_no_space_error(exc):
    if not isinstance(exc, OSError):
        return False
    return (
        getattr(exc, 'errno', None) == errno.ENOSPC
        or getattr(exc, 'winerror', None) == 112
        or 'No space left on device' in str(exc)
        or 'not enough space' in str(exc).lower()
    )


def _raise_upload_storage_error(path, required_bytes=None, cause=None):
    free_bytes = _free_bytes_for_path(path)
    raise UploadStorageError(
        _storage_error_message(path, required_bytes, free_bytes),
        path=str(path),
        required_bytes=required_bytes,
        free_bytes=free_bytes,
    ) from cause


def _remove_partial_file(path):
    try:
        target = Path(path)
        if target.exists() and target.is_file():
            target.unlink()
    except Exception:
        pass


def _uploaded_file_size(file_storage):
    size = getattr(file_storage, 'content_length', None)
    if size:
        return int(size)

    stream = getattr(file_storage, 'stream', None)
    if stream is None:
        return None
    try:
        position = stream.tell()
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(position)
        return int(size)
    except Exception:
        return None


def _unique_path_for_upload(directory, filename):
    candidate = Path(directory) / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    for index in range(2, 10000):
        next_candidate = candidate.with_name(f"{stem}_{index}{suffix}")
        if not next_candidate.exists():
            return next_candidate
    return candidate.with_name(f"{stem}_{secrets.token_hex(4)}{suffix}")


def _path_is_within(path, parent):
    try:
        path_abs = os.path.abspath(os.fspath(path))
        parent_abs = os.path.abspath(os.fspath(parent))
        common = os.path.commonpath([path_abs, parent_abs])
        return normalized_path_key(common) == normalized_path_key(parent_abs)
    except Exception:
        return False


def _staged_upload_root():
    return UPLOAD_DIR / '_staged'


def _staged_upload_owner(path):
    root = _staged_upload_root()
    if not _path_is_within(path, root):
        return None
    try:
        relative = os.path.relpath(os.path.abspath(os.fspath(path)), os.path.abspath(os.fspath(root)))
        parts = Path(relative).parts
        return parts[0] if parts else None
    except Exception:
        return None


def _cleanup_empty_staged_parents(start_dir):
    root = _staged_upload_root()
    current = Path(start_dir)
    while _path_is_within(current, root) and normalized_path_key(current) != normalized_path_key(root):
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _discard_uploaded_staging_file(path):
    target = Path(path)
    if not _path_is_within(target, _staged_upload_root()):
        return False
    try:
        if target.is_file():
            target.unlink()
            _cleanup_empty_staged_parents(target.parent)
            return True
    except Exception:
        return False
    return False


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


def _probe_video_info(video_path, fps_override=None, orientation_override=None):
    return probe_video_info(
        video_path,
        fps_override=fps_override,
        orientation_override=orientation_override,
    )


def _extract_video_info_from_entry(video_info, fps_override=None):
    info = dict(video_info or {})
    fallback = _probe_video_info(
        info.get('path', ''),
        fps_override=fps_override,
        orientation_override=info.get('orientation_override'),
    )
    return {
        'fps': float(info.get('fps') or fallback['fps']),
        'detected_fps': float(info.get('detected_fps') or fallback['detected_fps']),
        # Geometry is always re-probed so sessions saved by an older version
        # (which stored coded rather than display dimensions) are corrected.
        'width': int(fallback.get('width') or info.get('width') or 0),
        'height': int(fallback.get('height') or info.get('height') or 0),
        'total_frames': int(fallback.get('total_frames') or info.get('total_frames') or 0),
        'duration_seconds': fallback.get('duration_seconds') or info.get('duration_seconds'),
        'orientation_degrees': int(fallback.get('orientation_degrees') or info.get('orientation_degrees') or 0),
        'orientation_source': fallback.get('orientation_source') or info.get('orientation_source', 'none'),
        'metadata_orientation_degrees': int(fallback.get('metadata_orientation_degrees') or info.get('metadata_orientation_degrees') or 0),
        'metadata_orientation_source': fallback.get('metadata_orientation_source') or info.get('metadata_orientation_source', 'none'),
        'orientation_override': info.get('orientation_override'),
    }


def _update_video_entry_info(video_info, fps_override=None):
    video_info.update(
        _probe_video_info(
            video_info.get('path', ''),
            fps_override=fps_override,
            orientation_override=video_info.get('orientation_override'),
        )
    )
    return video_info


def _opencv_can_open_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0:
            return True
        ret, _frame = cap.read()
        return bool(ret)
    finally:
        cap.release()


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
    save_path = _unique_path_for_upload(staged_dir, staged_name)
    expected_size = _uploaded_file_size(file_storage)
    _ensure_upload_storage_available(staged_dir, expected_size)
    try:
        file_storage.save(str(save_path))
    except OSError as exc:
        _remove_partial_file(save_path)
        if _is_no_space_error(exc):
            _raise_upload_storage_error(staged_dir, expected_size, cause=exc)
        raise
    _register_original_name(save_path, original_name)
    return str(save_path), original_name


def _adopt_uploaded_staging_file(source, session_id, display_name):
    staged_root = _staged_upload_root()
    if not _path_is_within(source, staged_root):
        return None

    owner = _staged_upload_owner(source)
    if owner and len(owner) == 32 and owner != str(session_id):
        return None

    session_dir = staged_root / str(session_id)
    if _path_is_within(source, session_dir):
        return str(source), display_name

    source = Path(source)
    previous_parent = source.parent
    session_dir.mkdir(parents=True, exist_ok=True)
    staged_name = build_ascii_file_name(display_name, prefix=str(session_id), fallback='video')
    staged_path = _unique_path_for_upload(session_dir, staged_name)

    try:
        shutil.move(str(source), str(staged_path))
    except OSError as exc:
        _remove_partial_file(staged_path)
        if _is_no_space_error(exc):
            required_bytes = source.stat().st_size if source.exists() else None
            _raise_upload_storage_error(session_dir, required_bytes, cause=exc)
        raise

    _cleanup_empty_staged_parents(previous_parent)
    _register_original_name(staged_path, display_name)
    return str(staged_path), display_name


def _stage_local_video_if_needed(path, session_id, original_name=None):
    source = Path(path).expanduser()
    display_name = clean_display_name(original_name or _get_original_name_for_path(source) or source.name, fallback='video')
    adopted = _adopt_uploaded_staging_file(source, session_id, display_name)
    if adopted:
        return adopted

    source_path = str(source)
    if not contains_non_ascii(source_path) or _opencv_can_open_video(source):
        return source_path, display_name

    staged_dir = UPLOAD_DIR / '_staged' / str(session_id)
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_name = build_ascii_file_name(display_name, prefix=str(session_id), fallback='video')
    staged_path = _unique_path_for_upload(staged_dir, staged_name)
    if not staged_path.exists() or staged_path.stat().st_size != source.stat().st_size:
        required_bytes = source.stat().st_size
        try:
            os.link(source, staged_path)
        except OSError:
            _ensure_upload_storage_available(staged_dir, required_bytes)
            try:
                shutil.copy2(source, staged_path)
            except OSError as exc:
                _remove_partial_file(staged_path)
                if _is_no_space_error(exc):
                    _raise_upload_storage_error(staged_dir, required_bytes, cause=exc)
                raise
    _register_original_name(staged_path, display_name)
    return str(staged_path), display_name

# ==================== 性能优化 ====================
enable_opencv_optimizations()

# ==================== Flask 应用 ====================
app = Flask(__name__,
            template_folder=str(BUNDLE_DIR / 'templates'),
            static_folder=str(BUNDLE_DIR / 'static'))
app.secret_key = _load_or_create_secret_key()
app.config['MAX_CONTENT_LENGTH'] = None
app.config['DESKTOP_MODE'] = bool(getattr(sys, 'frozen', False))


def _desktop_exit_process():
    """End the frozen desktop backend after the HTTP response is delivered."""
    time.sleep(0.25)
    os._exit(0)


@app.post('/api/desktop/quit')
def desktop_quit():
    if not app.config.get('DESKTOP_MODE'):
        return jsonify({'success': False, 'error': 'Desktop quit is unavailable in server mode.'}), 404
    if request.remote_addr not in {'127.0.0.1', '::1'}:
        return jsonify({'success': False, 'error': 'Desktop quit is restricted to this computer.'}), 403

    callback = app.config.get('DESKTOP_EXIT_CALLBACK') or _desktop_exit_process
    threading.Thread(target=callback, name='mousescope-desktop-exit', daemon=True).start()
    return jsonify({'success': True})


@app.errorhandler(RequestEntityTooLarge)
def handle_request_entity_too_large(_exc):
    message = '上传请求过大，已被运行环境拒绝。请拆分批次，或改用本地路径方式选择文件。'
    wants_json = (
        request.path.startswith('/api/')
        or request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        or request.accept_mimetypes.best == 'application/json'
    )
    if wants_json:
        return jsonify({'success': False, 'error': message}), 413
    flash(message, 'error')
    return redirect(url_for('upload_page'))


# 会话存储（运行时 + 磁盘快照）
analysis_sessions = SessionStore(
    upload_dir=UPLOAD_DIR,
    work_dir=WORK_DIR,
    pending_ttl_seconds=SESSION_PENDING_TTL_SECONDS,
    finished_ttl_seconds=SESSION_FINISHED_TTL_SECONDS,
    error_ttl_seconds=SESSION_ERROR_TTL_SECONDS,
    now_fn=_now_ts,
)
if analysis_sessions.restored_session_ids:
    logging.getLogger(__name__).info(
        "Restored %d pending analysis configuration session(s).",
        len(analysis_sessions.restored_session_ids),
    )
for restore_error in analysis_sessions.restore_errors:
    logging.getLogger(__name__).warning("Session snapshot recovery: %s", restore_error)

analysis_history = AnalysisHistoryStore(
    db_path=HISTORY_DIR / 'analysis_history.sqlite',
    now_fn=_now_ts,
    default_result_dir=RESULT_DIR,
)
analysis_history.mark_interrupted_running()
visual_style_store = VisualStyleStore(DATA_DIR / 'visual_style.json')
experiment_defaults_store = ExperimentDefaultsStore(DATA_DIR / 'experiment_defaults.json')

project_store = ProjectStore(
    projects_root=PROJECTS_DIR,
    now_fn=_now_ts,
)


def _analysis_status_observer(analysis):
    session_id = getattr(analysis, 'session_id', None)
    if not session_id:
        return
    sess = analysis_sessions.get(session_id)
    # A task can finish just after the user deletes its record. Never recreate
    # deleted history/project state from the late worker callback.
    if sess is None:
        return
    try:
        finalize_reproducibility_artifacts(analysis, sess)
    except Exception:
        logging.getLogger(__name__).exception(
            "Failed to finalize reproducibility artifacts for session %s.",
            session_id,
        )
    record = analysis_history.upsert_runtime(session_id, sess, analysis=analysis)
    project_context = sess.get('project_context') if isinstance(sess, dict) and isinstance(sess.get('project_context'), dict) else None
    if project_context:
        status = str(record.get('status') or '').lower()
        assignment_ids = project_context.get('assignment_ids') or []
        progress = int(record.get('progress') or 0)
        if status == 'completed':
            progress = 100
        project_store.upsert_project_run(
            project_id=project_context.get('project_id'),
            project_run_id=project_context.get('project_run_id'),
            run_id=session_id,
            session_id=session_id,
            experiment_type=record.get('experiment_type') or sess.get('experiment_type'),
            assignment_ids=assignment_ids,
            output_dir=record.get('output_dir') or sess.get('output_dir') or '',
            status=status or 'unknown',
            progress=progress,
            message=str(record.get('message') or ''),
            result_files=record.get('result_files') or [],
            finished_at=record.get('finished_at'),
        )
        if status in {'completed', 'error', 'stopped', 'interrupted'}:
            project_store.set_assignments_status(
                project_context.get('project_id'),
                assignment_ids,
                status,
                latest_run_id=session_id,
            )
            if status == 'completed':
                project_store.cleanup_completed_upload_caches(project_context.get('project_id'), assignment_ids)


configure_analysis_runtime(
    RESULT_DIR=RESULT_DIR,
    HAS_PANDAS=HAS_PANDAS,
    china_now=china_now,
    _prepare_output_paths=_prepare_output_paths,
    _export_result_file=_export_result_file,
    _write_dataframe_csv=_write_dataframe_csv,
    _bi_label=_bi_label,
    _report_value=_report_value,
    _report_time_mode_label=_report_time_mode_label,
    _report_phase_label=_report_phase_label,
    _write_report_metric_lines=_write_report_metric_lines,
    _analysis_status_observer=_analysis_status_observer,
)


def _experiment_type_to_manual_module(experiment_type):
    return get_manual_module(experiment_type)


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


def _cleanup_orphan_upload_staging(now):
    staged_root = _staged_upload_root()
    if not staged_root.exists():
        return

    active_session_ids = {str(session_id) for session_id, _sess in analysis_sessions.items()}
    for target in staged_root.iterdir():
        if not target.is_dir() or target.name in active_session_ids:
            continue
        try:
            age_seconds = now - target.stat().st_mtime
        except OSError:
            continue
        if age_seconds <= UPLOAD_ORPHAN_STAGING_TTL_SECONDS:
            continue
        try:
            shutil.rmtree(target, ignore_errors=True)
        except Exception:
            pass


def _cleanup_expired_runtime_state():
    now = _now_ts()

    expired_session_ids = []
    for session_id, sess in list(analysis_sessions.items()):
        last_access = _safe_runtime_ts(sess.get('last_accessed_at') or sess.get('created_at'), now)
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
        analysis_sessions.delete(session_id)

    _cleanup_orphan_upload_staging(now)
    cleanup_debug_preview_cache(now, DEBUG_STATIC_BG_CACHE_TTL_SECONDS, DEBUG_STATIC_BG_CACHE_MAX_ITEMS)


def _license_not_accepted_response():
    message = 'License agreement must be accepted before using MouseScope.'
    if request.path.startswith('/api/') or request.method != 'GET':
        return jsonify({'success': False, 'error': message}), 403
    flash(message, 'warning')
    next_url = request.full_path.rstrip('?') if request.query_string else request.path
    return redirect(url_for('license_page', next=next_url))


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
    valid_manual_modules = {spec.manual_module for spec in list_experiments()}

    if endpoint == 'manual':
        manual_module = str(request.args.get('module', 'tst')).strip().lower()
        if manual_module not in valid_manual_modules:
            manual_module = 'tst'
        manual_nav_target = None
        manual_nav_rel = None
        manual_nav_url = url_for('manual', module=manual_module)
    else:
        manual_module = _experiment_type_to_manual_module(_request_experiment_type())
        manual_nav_url = url_for('manual', module=manual_module)
        manual_nav_target = '_blank' if endpoint in {'calibrate', 'monitor'} else None
        manual_nav_rel = 'noopener noreferrer' if manual_nav_target else None

    return {
        'app_version': APP_VERSION,
        'app_version_tag': APP_VERSION_TAG,
        'desktop_mode': bool(app.config.get('DESKTOP_MODE')),
        'enable_three_chamber': is_three_chamber_enabled(),
        'manual_nav_url': manual_nav_url,
        'manual_nav_target': manual_nav_target,
        'manual_nav_rel': manual_nav_rel,
    }


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


# ==================== Route Registration ====================
from types import SimpleNamespace
from routes.pages import register_page_routes
from routes.api import register_api_routes
from routes.projects import register_project_page_routes
from routes.project_api import register_project_api_routes

route_context = SimpleNamespace(
    APP_VERSION=APP_VERSION,
    DATA_DIR=DATA_DIR,
    UploadStorageError=UploadStorageError,
    analysis_sessions=analysis_sessions,
    analysis_history=analysis_history,
    project_store=project_store,
    visual_style_store=visual_style_store,
    experiment_defaults_store=experiment_defaults_store,
    BUNDLE_DIR=BUNDLE_DIR,
    VALID_VIDEO_EXTS=VALID_VIDEO_EXTS,
    _cleanup_session_artifacts=_cleanup_session_artifacts,
    RESULT_DIR=RESULT_DIR,
    _extract_video_info_from_entry=_extract_video_info_from_entry,
    _get_original_name_for_path=_get_original_name_for_path,
    _now_ts=_now_ts,
    _resolve_result_dir=_resolve_result_dir,
    _result_zip_arcname=_result_zip_arcname,
    _save_uploaded_video=_save_uploaded_video,
    _session_video_info=_session_video_info,
    _stage_local_video_if_needed=_stage_local_video_if_needed,
    _update_video_entry_info=_update_video_entry_info,
    _discard_uploaded_staging_file=_discard_uploaded_staging_file,
)
register_page_routes(app, route_context)
register_api_routes(app, route_context)
register_project_page_routes(app, route_context)
register_project_api_routes(app, route_context)


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
