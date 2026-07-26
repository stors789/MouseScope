from __future__ import annotations

import base64
import copy
import logging
import math
import os
import platform
import re
import secrets
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
from flask import Response, after_this_request, jsonify, request, send_file, stream_with_context, url_for

from services.api_errors import build_api_error
from services.browser_video import BrowserVideoError, ensure_browser_video
from services.analysis_builders import build_analysis_config
from services.analysis_windows import effective_video_duration_seconds, validate_analysis_windows, validate_roi_windows
from services.analysis_sessions import (
    ElevatedPlusMultiArenaAnalysisSession,
    ForcedSwimMultiCylinderAnalysisSession,
    LightDarkBoxMultiArenaAnalysisSession,
    NovelObjectMultiArenaAnalysisSession,
    OpenFieldMultiArenaAnalysisSession,
    ThreeChamberSocialAnalysisSession,
    VisionAnalysisSession,
    WaterMazeMultiPoolAnalysisSession,
    YMazeMultiArenaAnalysisSession,
)
from services.debug_preview import build_debug_preview_response
from services.experiment_registry import get_experiment_spec, is_experiment_release_hidden, summarize_batch_state
from services.experiment_defaults import ExperimentDefaultsValidationError
from services.output_runs import prepare_session_run_output
from services.reproducibility import (
    DEFAULT_RANDOM_SEED,
    ReproducibilityError,
    apply_deterministic_seed,
    build_run_snapshot,
    find_run_snapshot,
    load_run_snapshot,
    reproducibility_status,
    resolved_config_for_replay,
    validate_replay_snapshot,
    write_run_snapshot,
)
from services.resource_policy import resolved_run_resource_policy
from services.video_frame_reader import read_video_frame
from services.video_orientation import orientation_override_degrees
from services.visual_style_preview import preview_experiment_payload, render_visual_style_preview
from services.y_maze_roi_templates import RoiTemplateValidationError


logger = logging.getLogger(__name__)


def register_api_routes(app, ctx) -> None:
    analysis_sessions = ctx.analysis_sessions
    analysis_history = ctx.analysis_history
    project_store = getattr(ctx, "project_store", None)
    visual_style_store = getattr(ctx, "visual_style_store", None)
    experiment_defaults_store = getattr(ctx, "experiment_defaults_store", None)
    y_maze_roi_template_store = getattr(ctx, "y_maze_roi_template_store", None)
    manual_video_cache_dir = Path(getattr(ctx, "DATA_DIR", tempfile.gettempdir())) / "manual_video_cache"
    app_version = str(getattr(ctx, "APP_VERSION", ""))
    RESULT_DIR = ctx.RESULT_DIR
    _resolve_result_dir = ctx._resolve_result_dir
    _result_zip_arcname = ctx._result_zip_arcname
    _session_video_info = ctx._session_video_info
    _update_video_entry_info = ctx._update_video_entry_info

    runtime_setting_keys = ("start_sec", "end_sec", "fps_override", "orientation_override")

    def _api_error(
        message: str | None = None,
        status: int = 400,
        *,
        code: str | None = None,
        message_en: str | None = None,
        **extra,
    ):
        return jsonify(build_api_error(message, code=code, message_en=message_en, **extra)), status

    def _detach_exact_replay_after_edit(sess: dict) -> bool:
        """Turn an edited replay draft into an ordinary custom configuration."""

        if not isinstance(sess.get('replay_resolved_config'), dict):
            return False
        sess.pop('replay_resolved_config', None)
        sess.pop('replay_of', None)
        sess['replay_detached_by_edit'] = True
        return True

    def _preserve_per_video_runtime_settings(incoming: dict, saved: dict, total_videos: int) -> dict:
        """Calibration pages only serialize assay fields; retain video setup fields."""
        raw_incoming = incoming.get("per_video_configs") if isinstance(incoming, dict) else None
        raw_saved = saved.get("per_video_configs") if isinstance(saved, dict) else None
        if not isinstance(raw_incoming, list) or not isinstance(raw_saved, list):
            return incoming
        for index in range(min(total_videos, len(raw_incoming), len(raw_saved))):
            config = raw_incoming[index]
            previous = raw_saved[index]
            if not isinstance(config, dict) or not isinstance(previous, dict):
                continue
            for key in runtime_setting_keys:
                if key not in config and key in previous:
                    config[key] = copy.deepcopy(previous[key])
        return incoming

    def _batch_cards(sess: dict):
        return summarize_batch_state(
            sess.get('experiment_type'),
            sess.get('draft_payload'),
            sess.get('video_infos'),
            project_context=sess.get('project_context'),
        )

    def _ensure_writable_output_directory(value: str | os.PathLike) -> str:
        """Create and verify an output directory without leaving probe files."""
        path = Path(value)
        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f'{path} is not a directory')
        path.mkdir(parents=True, exist_ok=True)
        fd = None
        probe_path = None
        try:
            fd, probe_path = tempfile.mkstemp(prefix='.mousescope-write-test-', dir=str(path))
        finally:
            if fd is not None:
                os.close(fd)
            if probe_path:
                try:
                    os.unlink(probe_path)
                except FileNotFoundError:
                    pass
        return str(path)

    def _output_directory_error(exc: Exception):
        return _api_error(
            '输出目录不可用，请选择一个可写文件夹。',
            400,
            code='output_dir_invalid',
            message_en='The output location is not a writable folder.',
            detail=str(exc),
        )

    def _task_search_blob(task: dict) -> str:
        fields = [
            task.get('run_id'),
            task.get('session_id'),
            task.get('experiment_type'),
            task.get('experiment_label'),
            task.get('experiment_short_label'),
            task.get('project_id'),
            task.get('project_name'),
            task.get('project_run_id'),
            task.get('primary_video_name'),
            task.get('message'),
            task.get('output_dir'),
        ]
        fields.extend(task.get('video_names') or [])
        fields.extend(task.get('experiment_types') or [])
        fields.extend(task.get('experiment_short_labels') or [])
        return ' '.join(str(item or '') for item in fields).lower()

    def _task_matches_filters(task: dict, query: str, status_filter: str, experiment_filter: str) -> bool:
        if query and query.lower() not in _task_search_blob(task):
            return False
        status = str(task.get('status') or '').lower()
        if status_filter:
            if status_filter == 'active':
                if status not in {'waiting', 'analyzing'}:
                    return False
            elif status != status_filter:
                return False
        if experiment_filter:
            experiment_values = {str(task.get('experiment_type') or '').lower()}
            experiment_values.update(str(item or '').lower() for item in (task.get('experiment_types') or []))
            if experiment_filter not in experiment_values:
                return False
        return True

    @app.route('/api/y_maze_roi_templates', methods=['GET'])
    def get_y_maze_roi_templates():
        if y_maze_roi_template_store is None:
            return _api_error('ROI 模板存储未配置。', 500)
        return jsonify({'templates': y_maze_roi_template_store.list_templates()})

    @app.route('/api/y_maze_roi_templates', methods=['POST'])
    def save_y_maze_roi_template():
        if y_maze_roi_template_store is None:
            return _api_error('ROI 模板存储未配置。', 500)
        data = request.get_json(silent=True) or {}
        try:
            saved = y_maze_roi_template_store.save_template(
                data.get('name'),
                {'rois': data.get('rois'), 'geometry': data.get('geometry')},
            )
        except RoiTemplateValidationError as exc:
            return _api_error(str(exc), 400)
        return jsonify({'success': True, 'template': saved})

    @app.route('/api/y_maze_roi_templates/<template_id>', methods=['DELETE'])
    def delete_y_maze_roi_template(template_id):
        if y_maze_roi_template_store is None:
            return _api_error('ROI 模板存储未配置。', 500)
        if not y_maze_roi_template_store.delete_template(template_id):
            return _api_error('ROI 模板不存在。', 404)
        return jsonify({'success': True})

    def _decorate_task_urls(task: dict) -> dict:
        session_id = task.get('session_id')
        run_id = task.get('run_id')
        if task.get('can_monitor') and session_id:
            task['monitor_url'] = url_for('monitor', session_id=session_id)
        if task.get('can_configure') and session_id:
            task['configure_url'] = url_for('analysis_config', session_id=session_id)
        if task.get('can_stop') and session_id:
            task['stop_url'] = url_for('stop_analysis', session_id=session_id)
        if task.get('can_download'):
            if task.get('is_live') and task.get('can_monitor') and session_id:
                task['download_all_url'] = url_for('download_all', session_id=session_id)
            elif run_id:
                task['download_all_url'] = url_for('history_download_all', run_id=run_id)
        if task.get('can_open_folder'):
            if task.get('is_live') and session_id:
                task['open_folder_url'] = url_for('open_folder', session_id=session_id)
            elif run_id:
                task['open_folder_url'] = url_for('history_open_folder', run_id=run_id)
        if task.get('project_id'):
            task['project_url'] = url_for('project_home', project_id=task.get('project_id'))
        if task.get('can_delete') and run_id and not task.get('project_id'):
            task['delete_url'] = url_for('delete_task_record', run_id=run_id)
        project_latest_run_id = str(task.get('project_latest_run_id') or '').strip()
        replay_source_id = project_latest_run_id or str(run_id or '').strip()
        replay_record = analysis_history.get_record(project_latest_run_id) if project_latest_run_id else task
        replay_snapshot = find_run_snapshot(replay_record or task)
        if (
            replay_source_id
            and replay_snapshot
            and str(task.get('status') or '').lower() in {'completed', 'error', 'stopped', 'interrupted'}
        ):
            task['can_replay'] = True
            task['replay_url'] = url_for('replay_task', run_id=replay_source_id)
        replay_result_status = reproducibility_status(replay_record or task)
        if replay_result_status:
            task['reproducibility_status'] = replay_result_status
        return task

    def _project_subject_binding_error(sess: dict, payload: dict) -> str:
        project_context = sess.get('project_context') if isinstance(sess.get('project_context'), dict) else None
        if not project_context:
            return ''

        project_id = project_context.get('project_id')
        subjects = project_context.get('subject_candidates') or []
        if project_store is not None and project_id:
            try:
                try:
                    project_exists = bool(project_store.get_project(project_id, include_summary=False))
                except TypeError:
                    project_exists = bool(project_store.get_project(project_id))
                if project_exists:
                    fresh_subjects = project_store.subject_candidates(project_id)
                    subjects = fresh_subjects
            except Exception:
                pass

        valid_subject_ids = {
            str(subject.get('subject_id') or '').strip()
            for subject in subjects
            if isinstance(subject, dict) and str(subject.get('subject_id') or '').strip()
        }
        if not valid_subject_ids:
            return '请先在项目中添加小鼠，然后为每个 ROI 选择对应小鼠。'

        video_infos = sess.get('video_infos') or []
        video_assignment_map = project_context.get('video_assignment_map') or {}
        unit_map = project_context.get('unit_map') or {}

        def video_name(index: int) -> str:
            if 0 <= index < len(video_infos):
                info = video_infos[index] or {}
                return str(info.get('original_name') or info.get('name') or f'Video {index + 1}')
            return f'Video {index + 1}'

        def check_rois(rois, video_index: int) -> str:
            if not isinstance(rois, list) or not rois:
                return ''
            assignment_id = video_assignment_map.get(str(video_index))
            allowed_units = {
                str(unit.get('unit_id') or ''): unit
                for unit in (unit_map.get(assignment_id) or [])
                if isinstance(unit, dict) and str(unit.get('unit_id') or '')
            }
            for roi_index, roi in enumerate(rois, start=1):
                subject_id = str((roi or {}).get('subject_id') or '').strip() if isinstance(roi, dict) else ''
                if not subject_id or subject_id not in valid_subject_ids:
                    return f"{video_index + 1}. {video_name(video_index)}: ROI {roi_index} 未绑定项目小鼠，请从已有小鼠中选择。"
                project_unit_id = str((roi or {}).get('project_unit_id') or '').strip() if isinstance(roi, dict) else ''
                if project_unit_id:
                    unit = allowed_units.get(project_unit_id)
                    if not unit or str(unit.get('subject_id') or '').strip() != subject_id:
                        return (
                            f"{video_index + 1}. {video_name(video_index)}: ROI {roi_index} 的实验单元"
                            "不属于该视频，请重新选择项目小鼠。"
                        )
            return ''

        raw_configs = payload.get('per_video_configs') if isinstance(payload, dict) else None
        if isinstance(raw_configs, list):
            for index, config in enumerate(raw_configs):
                if not isinstance(config, dict):
                    continue
                error = check_rois(config.get('rois'), index)
                if error:
                    return error

        root_rois = payload.get('rois') if isinstance(payload, dict) else None
        if isinstance(root_rois, list):
            try:
                current_index = int(payload.get('video', sess.get('selected_video_idx') or 0))
            except Exception:
                current_index = int(sess.get('selected_video_idx') or 0)
            error = check_rois(root_rois, current_index)
            if error:
                return error

        return ''

    def _project_task_indexes() -> tuple[dict[str, dict], dict[str, dict]]:
        lookup: dict[str, dict] = {}
        project_tasks: dict[str, dict] = {}
        if project_store is None:
            return lookup, project_tasks
        try:
            projects = project_store.list_projects(include_archived=True)
        except Exception:
            logger.exception("Failed to list projects while building task index.")
            return lookup, project_tasks
        for project in projects:
            project_id = str(project.get('project_id') or '')
            if not project_id:
                continue
            try:
                runs = project_store.list_project_runs(project_id)
            except Exception:
                logger.debug("Failed to list project runs: project_id=%s", project_id, exc_info=True)
                runs = []
            for run in runs:
                run_id = str(run.get('run_id') or run.get('session_id') or '').strip()
                if not run_id:
                    continue
                assignment_ids = [str(item) for item in (run.get('assignment_ids') or []) if item]
                try:
                    assignments = project_store.get_assignments(project_id, assignment_ids) if assignment_ids else []
                except Exception:
                    logger.debug("Failed to read project assignments: project_id=%s", project_id, exc_info=True)
                    assignments = []
                video_names = [str(item.get('original_name') or '') for item in assignments if item.get('original_name')]
                result_files = [str(path) for path in (run.get('result_files') or []) if path]
                existing_result_files = [path for path in result_files if os.path.isfile(path)]
                output_dir = str(run.get('output_dir') or '')
                output_exists = bool(output_dir and os.path.isdir(output_dir))
                try:
                    spec = get_experiment_spec(str(run.get('experiment_type') or 'tst'))
                except Exception:
                    spec = get_experiment_spec('tst')
                meta = {
                    'project_id': project_id,
                    'project_name': project.get('name') or '',
                    'project_run_id': run.get('project_run_id') or '',
                    'task_scope': 'project',
                }
                lookup[run_id] = meta
                session_key = str(run.get('session_id') or '').strip()
                if session_key:
                    lookup[session_key] = meta
                project_tasks[run_id] = {
                    'run_id': run_id,
                    'session_id': session_key or run_id,
                    'experiment_type': spec.key,
                    'experiment_label': spec.label,
                    'experiment_short_label': spec.short_label,
                    'status': run.get('status') or 'configured',
                    'progress': int(run.get('progress') or 0),
                    'message': run.get('message') or '',
                    'video_names': video_names,
                    'video_paths': [str(item.get('project_path') or item.get('source_path') or '') for item in assignments],
                    'primary_video_name': video_names[0] if video_names else (project.get('name') or run_id),
                    'video_count': len(video_names),
                    'output_dir': output_dir,
                    'result_files': result_files,
                    'result_file_count': len(existing_result_files),
                    'has_results': bool(existing_result_files or output_exists),
                    'has_output_dir': output_exists,
                    'config_summary': {'video_count': len(video_names)},
                    'is_batch': len(video_names) > 1,
                    'created_at': run.get('created_at'),
                    'started_at': None,
                    'finished_at': run.get('finished_at'),
                    'updated_at': run.get('updated_at') or run.get('created_at'),
                    'duration_seconds': None,
                    'error_message': '',
                    'source': 'project',
                    'is_live': False,
                    'can_monitor': False,
                    'can_stop': False,
                    'can_download': False,
                    'can_open_folder': False,
                    'can_configure': False,
                    'can_delete': False,
                    **meta,
                }
        return lookup, project_tasks

    def _ordered_unique(values) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values or []:
            text = str(value or '').strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _is_project_task(task: dict) -> bool:
        return bool(task.get('project_id') or task.get('project_run_id') or task.get('task_scope') == 'project')

    def _project_summary_status(runs: list[dict]) -> str:
        statuses = [str(run.get('status') or '').lower() for run in runs]
        if any(status in {'analyzing', 'running'} for status in statuses):
            return 'analyzing'
        if any(status == 'waiting' for status in statuses):
            return 'waiting'
        latest_status = statuses[0] if statuses else ''
        return latest_status or 'configured'

    def _project_experiment_summary(experiment_types: list[str]) -> tuple[str, str, str, list[str]]:
        specs = []
        for experiment_type in experiment_types:
            try:
                specs.append(get_experiment_spec(experiment_type))
            except Exception:
                continue
        if not specs:
            return '', '', '', []
        short_labels = _ordered_unique(spec.short_label for spec in specs)
        if len(specs) == 1:
            return specs[0].key, specs[0].label, specs[0].short_label, short_labels
        shown = short_labels[:4]
        extra = len(short_labels) - len(shown)
        short_text = ' / '.join(shown) + (f' +{extra}' if extra > 0 else '')
        return '', 'Multiple experiments', short_text, short_labels

    def _aggregate_project_tasks(tasks: list[dict]) -> list[dict]:
        project_groups: dict[str, list[dict]] = {}
        passthrough: list[dict] = []
        for task in tasks:
            if not _is_project_task(task):
                passthrough.append(task)
                continue
            project_id = str(task.get('project_id') or '').strip()
            if not project_id:
                passthrough.append(task)
                continue
            project_groups.setdefault(project_id, []).append(task)

        cards: list[dict] = []
        for project_id, runs in project_groups.items():
            runs.sort(
                key=lambda item: (
                    _safe_timestamp(item.get('updated_at') or item.get('created_at')) or 0,
                    _safe_timestamp(item.get('created_at')) or 0,
                    str(item.get('run_id') or item.get('session_id') or ''),
                ),
                reverse=True,
            )
            latest = runs[0]
            video_names = _ordered_unique(
                name
                for run in runs
                for name in (run.get('video_names') or ([run.get('primary_video_name')] if run.get('primary_video_name') else []))
            )
            video_paths = _ordered_unique(path for run in runs for path in (run.get('video_paths') or []))
            experiment_types = _ordered_unique(run.get('experiment_type') for run in runs)
            experiment_type, experiment_label, experiment_short_label, experiment_short_labels = _project_experiment_summary(experiment_types)
            result_files = _ordered_unique(path for run in runs for path in (run.get('result_files') or []))
            existing_result_files = [path for path in result_files if os.path.isfile(path)]
            output_dirs = _ordered_unique(run.get('output_dir') for run in runs)
            output_exists = any(path and os.path.isdir(path) for path in output_dirs)
            latest_created = max((_safe_timestamp(run.get('created_at')) or 0) for run in runs)
            latest_updated = max((_safe_timestamp(run.get('updated_at') or run.get('created_at')) or 0) for run in runs)
            progress_values = [int(run.get('progress') or 0) for run in runs]
            status = _project_summary_status(runs)
            if status == 'completed':
                progress = 100
            elif latest.get('progress') is not None and latest.get('progress') != '':
                progress = int(latest.get('progress') or 0)
            else:
                progress = int(sum(progress_values) / len(progress_values)) if progress_values else 0

            cards.append({
                'run_id': f'project:{project_id}',
                'session_id': latest.get('session_id') or latest.get('run_id') or '',
                'experiment_type': experiment_type,
                'experiment_types': experiment_types,
                'experiment_label': experiment_label,
                'experiment_short_label': experiment_short_label,
                'experiment_short_labels': experiment_short_labels,
                'status': status,
                'progress': max(0, min(100, int(progress or 0))),
                'message': latest.get('message') or '',
                'video_names': video_names,
                'video_paths': video_paths,
                'primary_video_name': video_names[0] if video_names else '',
                'video_count': len(video_names),
                'output_dir': output_dirs[0] if len(output_dirs) == 1 else '',
                'result_files': result_files,
                'result_file_count': len(existing_result_files),
                'has_results': bool(existing_result_files or output_exists),
                'has_output_dir': bool(len(output_dirs) == 1 and output_exists),
                'config_summary': {'video_count': len(video_names), 'project_run_count': len(runs)},
                'project_run_count': len(runs),
                'project_latest_run_id': latest.get('run_id') or latest.get('session_id') or '',
                'project_id': project_id,
                'project_name': latest.get('project_name') or '',
                'project_run_id': '',
                'task_scope': 'project',
                'is_batch': len(video_names) > 1 or len(runs) > 1,
                'created_at': latest_created or latest.get('created_at'),
                'started_at': latest.get('started_at'),
                'finished_at': latest.get('finished_at'),
                'updated_at': latest_updated or latest.get('updated_at') or latest.get('created_at'),
                'sort_at': latest_created or latest_updated,
                'duration_seconds': latest.get('duration_seconds'),
                'error_message': latest.get('error_message') or '',
                'source': 'project-summary',
                'is_live': any(bool(run.get('is_live')) for run in runs),
                'can_monitor': False,
                'can_stop': False,
                'can_download': False,
                'can_open_folder': False,
                'can_configure': False,
                'can_delete': False,
            })
        return passthrough + cards

    def _history_output_dir(record: dict) -> str:
        output_dir = str(record.get('output_dir') or '').strip()
        if output_dir:
            return output_dir
        for path in record.get('result_files') or []:
            if path and os.path.exists(path):
                return str(Path(path).parent)
        return ''

    def _history_result_files(record: dict) -> list[str]:
        files = [str(path) for path in record.get('result_files') or [] if path and os.path.isfile(path)]
        if files:
            return files
        output_dir = _history_output_dir(record)
        if not output_dir or not os.path.isdir(output_dir):
            return []
        created_at = _safe_timestamp(record.get('created_at'))
        finished_at = _safe_timestamp(record.get('finished_at') or record.get('updated_at'))
        collected = []
        for root, _dirs, names in os.walk(output_dir):
            for name in names:
                path = os.path.join(root, name)
                if not _is_history_fallback_result_file(path, created_at, finished_at):
                    continue
                if os.path.isfile(path):
                    collected.append(path)
        return collected

    def _safe_timestamp(value) -> float | None:
        try:
            if value is None or value == '':
                return None
            return float(value)
        except Exception:
            return None

    def _nonnegative_int(value, label: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f'{label} 参数必须是非负整数')
        return max(0, parsed)

    def _is_history_fallback_result_file(path: str, created_at: float | None, finished_at: float | None) -> bool:
        suffix = Path(path).suffix.lower()
        if suffix == '.zip':
            return False
        if suffix not in {'.mp4', '.csv', '.txt', '.png', '.jpg', '.jpeg'}:
            return False
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return False
        if created_at is not None and mtime < created_at - 60:
            return False
        if finished_at is not None and mtime > finished_at + 6 * 60 * 60:
            return False
        return True

    def _temp_zip_path(prefix: str) -> str:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        safe_prefix = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(prefix or 'results'))[:80] or 'results'
        fd, path = tempfile.mkstemp(prefix=f'{safe_prefix}_', suffix='.zip', dir=str(RESULT_DIR))
        os.close(fd)
        return path

    def _send_zip_and_cleanup(zip_path: str, download_name: str):
        file_size = os.path.getsize(zip_path)
        safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(download_name or 'MouseScope_results.zip'))

        def _stream_zip():
            try:
                with open(zip_path, 'rb') as handle:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        yield chunk
            finally:
                try:
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                except Exception:
                    logger.exception('Failed to remove temporary download archive: %s', zip_path)

        response = Response(stream_with_context(_stream_zip()), mimetype='application/zip')
        response.content_length = file_size
        response.headers['Content-Disposition'] = f'attachment; filename="{safe_name}"'
        return response

    @app.route('/api/tasks')
    def api_tasks():
        query = str(request.args.get('q') or '').strip().lower()
        status_filter = str(request.args.get('status') or '').strip().lower()
        experiment_filter = str(request.args.get('experiment') or '').strip().lower()
        try:
            limit = int(request.args.get('limit') or 200)
        except (TypeError, ValueError):
            return _api_error('任务数量参数无效', 400, code='invalid_task_limit')
        limit = max(1, min(limit, 1000))
        project_lookup, project_only_tasks = _project_task_indexes()

        merged: dict[str, dict] = {}
        for task in analysis_history.list_records():
            meta = project_lookup.get(str(task.get('run_id') or '')) or project_lookup.get(str(task.get('session_id') or ''))
            if meta:
                task = {**task, **meta}
            merged[task['run_id']] = task

        for session_id, sess in analysis_sessions.items():
            task = analysis_history.build_runtime_task(session_id, sess)
            if sess.get('analysis') is not None:
                analysis_history.upsert_runtime(session_id, sess, analysis=sess.get('analysis'))
            meta = project_lookup.get(str(task.get('run_id') or '')) or project_lookup.get(str(task.get('session_id') or ''))
            if meta:
                task = {**task, **meta}
            previous = merged.get(task['run_id']) or {}
            merged[task['run_id']] = {**previous, **task}

        for run_id, task in project_only_tasks.items():
            if run_id not in merged and str(task.get('session_id') or '') not in merged:
                merged[run_id] = task

        tasks = _aggregate_project_tasks(list(merged.values()))
        tasks = [_decorate_task_urls(task) for task in tasks]
        tasks.sort(
            key=lambda item: (
                _safe_timestamp(item.get('sort_at') or item.get('created_at') or item.get('updated_at')) or 0,
                _safe_timestamp(item.get('created_at') or item.get('updated_at')) or 0,
                str(item.get('run_id') or item.get('session_id') or ''),
            ),
            reverse=True,
        )

        all_count = len(tasks)
        active_count = sum(1 for task in tasks if str(task.get('status') or '').lower() in {'waiting', 'analyzing'})
        completed_count = sum(1 for task in tasks if str(task.get('status') or '').lower() == 'completed')
        interrupted_count = sum(1 for task in tasks if str(task.get('status') or '').lower() == 'interrupted')
        error_count = sum(1 for task in tasks if str(task.get('status') or '').lower() == 'error')

        filtered = [
            task for task in tasks
            if _task_matches_filters(task, query, status_filter, experiment_filter)
        ]
        filtered_total = len(filtered)
        limited_tasks = filtered[:limit]

        return jsonify({
            'success': True,
            'tasks': [] if request.args.get('summary') else limited_tasks,
            'limit': limit,
            'returned': len(limited_tasks),
            'has_more': filtered_total > len(limited_tasks),
            'counts': {
                'all': all_count,
                'active': active_count,
                'completed': completed_count,
                'interrupted': interrupted_count,
                'error': error_count,
                'filtered': filtered_total,
            },
        })

    @app.route('/api/tasks/<run_id>', methods=['DELETE'])
    def delete_task_record(run_id):
        record = analysis_history.get_record(run_id)
        session_id = (record or {}).get('session_id') or run_id
        sess = analysis_sessions.get(session_id)
        removed_runtime = False

        if sess:
            if analysis_sessions.has_active_analysis(session_id):
                return _api_error('任务仍在运行，请先停止分析。', 400, code='task_running')
            analysis_sessions.delete(session_id)
            removed_runtime = True

        deleted_history = analysis_history.delete_record(run_id)
        if session_id != run_id:
            deleted_history = analysis_history.delete_record(session_id) or deleted_history

        if not deleted_history and not removed_runtime:
            return _api_error('任务记录不存在', 404, code='task_not_found')

        return jsonify({
            'success': True,
            'run_id': run_id,
            'session_id': session_id,
            'deleted_history': bool(deleted_history),
            'removed_runtime': bool(removed_runtime),
        })

    @app.route('/api/tasks/<run_id>/replay', methods=['POST'])
    def replay_task(run_id):
        """Create a new session that uses the exact resolved configuration of a prior run."""

        record = analysis_history.get_record(run_id)
        if not record:
            return _api_error('找不到要重跑的任务记录。', 404, code='task_not_found')
        if str(record.get('status') or '').lower() in {'waiting', 'analyzing'}:
            return _api_error('原任务仍在运行，完成或停止后才能按原配置重跑。', 409, code='replay_source_running')
        snapshot_path = find_run_snapshot(record)
        if not snapshot_path:
            return _api_error('该任务没有可用的运行快照，无法按原配置重跑。', 409, code='snapshot_missing')
        try:
            snapshot = load_run_snapshot(snapshot_path)
        except (OSError, ValueError, ReproducibilityError) as exc:
            return _api_error(f'无法读取运行快照：{exc}', 409, code='snapshot_read_failed')

        replay_errors = validate_replay_snapshot(snapshot, app_version=app_version)
        if replay_errors:
            return _api_error(
                replay_errors[0],
                409,
                code='snapshot_read_failed',
                message_en='The source videos or software version no longer match the run snapshot.',
                replay_errors=replay_errors,
            )

        experiment_type = str(snapshot.get('experiment_type') or 'tst').strip().lower()
        if is_experiment_release_hidden(experiment_type):
            return _api_error(
                '三箱社交模块未开放；内部验证可设置 MOUSESCOPE_ENABLE_3CST=1。',
                403,
                code='three_chamber_hidden',
            )

        session_id = secrets.token_hex(16)
        video_paths = [str(item.get('path')) for item in (snapshot.get('input_files') or []) if item.get('path')]
        video_infos = copy.deepcopy(snapshot.get('video_infos') or [])
        for index, video_path in enumerate(video_paths):
            while len(video_infos) <= index:
                video_infos.append({})
            if not isinstance(video_infos[index], dict):
                video_infos[index] = {}
            video_infos[index]['path'] = video_path
            video_infos[index].setdefault('original_name', Path(video_path).name)
        video_infos = video_infos[:len(video_paths)]

        project_context = copy.deepcopy(snapshot.get('project_context')) if isinstance(snapshot.get('project_context'), dict) else None
        output_root_dir = str(snapshot.get('output_root_dir') or Path(snapshot_path).parent.parent)
        if project_context:
            if project_store is None:
                return _api_error('项目工作台不可用，无法重建该项目任务。', 409, code='replay_project_unavailable')
            project_id = str(project_context.get('project_id') or '')
            try:
                project = project_store.get_project(project_id, include_summary=False)
            except TypeError:
                project = project_store.get_project(project_id)
            if not project:
                return _api_error('原项目不存在，无法按原项目配置重跑。', 409, code='replay_project_missing')
            project_context['project_run_id'] = f"prun_{secrets.token_hex(8)}"
            project_context['subject_candidates'] = project_store.subject_candidates(project_id)
            output_root_dir = str(Path(project['project_dir']) / 'results' / experiment_type)

        session_payload = {
            'experiment_type': experiment_type,
            'video_paths': video_paths,
            'video_infos': video_infos,
            'video_info': copy.deepcopy(video_infos[0] if video_infos else {}),
            'start_seconds': snapshot.get('start_seconds'),
            'end_seconds': snapshot.get('end_seconds'),
            'output_dir': output_root_dir,
            'output_root_dir': output_root_dir,
            'fps_override': snapshot.get('fps_override'),
            'is_batch': bool(snapshot.get('is_batch') or len(video_paths) > 1),
            'draft_payload': copy.deepcopy(snapshot.get('draft_payload') or {}),
            'draft_updated_at': time.time(),
            'selected_video_idx': int(snapshot.get('selected_video_idx') or 0),
            'created_at': time.time(),
            'last_accessed_at': time.time(),
            'reproducibility_seed': int(snapshot.get('random_seed') or DEFAULT_RANDOM_SEED),
            'replay_resolved_config': copy.deepcopy(snapshot.get('resolved_config') or {}),
            'replay_of': {
                'session_id': str(snapshot.get('session_id') or run_id),
                'snapshot_path': str(snapshot_path),
            },
        }
        if project_context:
            session_payload['project_context'] = project_context
        analysis_sessions.create(session_id, session_payload)

        if project_context:
            project_store.upsert_project_run(
                project_id=project_context.get('project_id'),
                project_run_id=project_context.get('project_run_id'),
                run_id=session_id,
                session_id=session_id,
                experiment_type=experiment_type,
                assignment_ids=project_context.get('assignment_ids') or [],
                output_dir=output_root_dir,
                status='configured',
                progress=0,
                message='Reproducibility replay session created',
            )
            project_store.set_assignments_status(
                project_context.get('project_id'),
                project_context.get('assignment_ids') or [],
                'configured',
                latest_run_id=session_id,
            )

        return jsonify({
            'success': True,
            'session_id': session_id,
            'start_url': url_for('start_analysis', session_id=session_id),
            'monitor_url': url_for('monitor', session_id=session_id),
        })

    @app.route('/api/tasks/bulk_delete', methods=['POST'])
    def bulk_delete_task_records():
        payload = request.get_json(silent=True) or {}
        raw_ids = payload.get('run_ids') if isinstance(payload, dict) else None
        if not isinstance(raw_ids, list):
            return jsonify({'success': False, 'error': '缺少要删除的任务记录'}), 400

        seen: set[str] = set()
        run_ids: list[str] = []
        for raw_id in raw_ids:
            run_id = str(raw_id or '').strip()
            if run_id and run_id not in seen:
                seen.add(run_id)
                run_ids.append(run_id)

        if not run_ids:
            return jsonify({'success': False, 'error': '缺少要删除的任务记录'}), 400

        deleted: list[dict] = []
        skipped: list[dict] = []
        for run_id in run_ids:
            record = analysis_history.get_record(run_id)
            session_id = (record or {}).get('session_id') or run_id
            sess = analysis_sessions.get(session_id)
            removed_runtime = False

            if sess:
                if analysis_sessions.has_active_analysis(session_id):
                    skipped.append({
                        'run_id': run_id,
                        'session_id': session_id,
                        'error': '任务仍在运行，请先停止分析。',
                    })
                    continue
                analysis_sessions.delete(session_id)
                removed_runtime = True

            deleted_history = analysis_history.delete_record(run_id)
            if session_id != run_id:
                deleted_history = analysis_history.delete_record(session_id) or deleted_history

            if deleted_history or removed_runtime:
                deleted.append({
                    'run_id': run_id,
                    'session_id': session_id,
                    'deleted_history': bool(deleted_history),
                    'removed_runtime': bool(removed_runtime),
                })
            else:
                skipped.append({
                    'run_id': run_id,
                    'session_id': session_id,
                    'error': '任务记录不存在',
                })

        return jsonify({
            'success': bool(deleted),
            'deleted_count': len(deleted),
            'skipped_count': len(skipped),
            'deleted': deleted,
            'skipped': skipped,
            'error': '' if deleted else '没有可删除的任务记录',
        }), 200 if deleted else 400

    @app.route('/api/history/download_all/<run_id>')
    def history_download_all(run_id):
        record = analysis_history.get_record(run_id)
        if not record:
            return jsonify({'error': '历史记录不存在'}), 404
        files = _history_result_files(record)
        if not files:
            return jsonify({'error': '没有可下载的结果文件'}), 404

        output_dir = _history_output_dir(record)
        zip_path = _temp_zip_path(f'{run_id}_history_results')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                if os.path.exists(path):
                    if output_dir and os.path.isdir(output_dir):
                        arcname = _result_zip_arcname(path, output_dir)
                    else:
                        arcname = os.path.basename(path)
                    zf.write(path, arcname)

        return _send_zip_and_cleanup(zip_path, 'MouseScope_history_results.zip')

    @app.route('/api/history/open_folder/<run_id>', methods=['POST'])
    def history_open_folder(run_id):
        record = analysis_history.get_record(run_id)
        if not record:
            return jsonify({'error': '历史记录不存在'}), 404

        result_dir = _history_output_dir(record)
        if not result_dir or not os.path.isdir(result_dir):
            return jsonify({'error': '结果目录不存在或已移动', 'path': result_dir}), 404

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

    @app.route('/api/get_first_frame/<session_id>')
    def get_first_frame(session_id):
        """获取视频帧（用于标定页面）"""
        sess = analysis_sessions.get(session_id)
        if not sess:
            return _api_error('会话不存在', 404, code='session_not_found')

        try:
            frame_idx = _nonnegative_int(request.args.get('frame', 0), 'frame')
            video_idx = _nonnegative_int(request.args.get('video', 0), 'video')
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

        video_paths = sess.get('video_paths') or []
        if not video_paths:
            return _api_error(
                '会话没有视频文件',
                404,
                code='video_not_found',
                message_en='The analysis session has no video files.',
            )
        video_idx = min(video_idx, len(video_paths) - 1)
        video_info = _session_video_info(sess, video_idx)
        total_frames = int(video_info.get('total_frames') or 0)
        if total_frames > 0:
            frame_idx = min(frame_idx, total_frames - 1)

        video_path = video_paths[video_idx]
        frame = read_video_frame(
            video_path,
            frame_idx,
            fps=video_info.get('detected_fps') or video_info.get('fps'),
            orientation_override=video_info.get('orientation_override'),
        )

        if frame is None:
            return _api_error(
                '无法读取帧',
                400,
                code='video_not_found',
                message_en='The requested video frame could not be read.',
            )

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': img_b64,
            'frame_idx': frame_idx,
            'total_frames': video_info['total_frames'],
            'video_info': video_info,
        })

    @app.route('/api/video_source/<session_id>')
    def video_source(session_id):
        """Serve one session video inline with HTTP Range support."""
        sess = analysis_sessions.get(session_id)
        if not sess:
            return _api_error('会话不存在', 404, code='session_not_found')
        try:
            video_idx = _nonnegative_int(request.args.get('video', 0), 'video')
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        video_paths = sess.get('video_paths') or []
        if video_idx >= len(video_paths):
            return _api_error(
                '会话没有对应视频文件',
                404,
                code='video_not_found',
                message_en='The requested session video does not exist.',
            )
        video_path = Path(video_paths[video_idx])
        if not video_path.is_file():
            return _api_error(
                '视频文件不存在',
                404,
                code='video_not_found',
                message_en='The requested session video file is missing.',
            )
        video_info = _session_video_info(sess, video_idx)
        try:
            browser_path, is_proxy = ensure_browser_video(
                video_path,
                manual_video_cache_dir,
                orientation_degrees=int(video_info.get('orientation_degrees') or 0),
            )
        except BrowserVideoError as exc:
            logger.exception('Failed to prepare browser video: %s', video_path)
            return _api_error(
                f'无法准备人工评分视频：{exc}',
                500,
                code='browser_video_failed',
                message_en='The video could not be prepared for manual scoring playback.',
            )
        response = send_file(
            browser_path,
            conditional=True,
            as_attachment=False,
            download_name=f'{video_path.stem}.mp4' if is_proxy else video_path.name,
            mimetype='video/mp4' if is_proxy else None,
        )
        response.headers['X-MouseScope-Video-Proxy'] = '1' if is_proxy else '0'
        return response


    @app.route('/api/batch_state/<session_id>')
    def api_batch_state(session_id):
        sess = analysis_sessions.get(session_id)
        if not sess:
            return _api_error('会话不存在', 404, code='session_not_found')
        if is_experiment_release_hidden(sess.get('experiment_type')):
            return jsonify({'success': False, 'error': '三箱社交模块未开放；内部验证可设置 MOUSESCOPE_ENABLE_3CST=1。'}), 403
        cards = summarize_batch_state(
            sess.get('experiment_type'),
            sess.get('draft_payload'),
            sess.get('video_infos'),
            project_context=sess.get('project_context'),
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
            return _api_error('会话不存在', 404, code='session_not_found')

        if is_experiment_release_hidden(sess.get('experiment_type')):
            return jsonify({'success': False, 'error': '三箱社交模块未开放；内部验证可设置 MOUSESCOPE_ENABLE_3CST=1。'}), 403

        if request.method == 'GET':
            return jsonify({
                'session_id': session_id,
                'draft_payload': sess.get('draft_payload'),
                'draft_updated_at': sess.get('draft_updated_at'),
                'selected_video_idx': int(sess.get('selected_video_idx') or 0),
            })

        if analysis_sessions.has_active_analysis(session_id):
            return _api_error('分析任务运行期间不能修改配置。', 409, code='analysis_config_locked')

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _api_error('草稿数据格式无效', 400, code='invalid_draft')

        selected_video_idx = payload.pop('_selected_video_idx', None)
        payload = _preserve_per_video_runtime_settings(
            payload,
            copy.deepcopy(sess.get('draft_payload') or {}),
            len(sess.get('video_infos') or []),
        )
        window_errors = validate_analysis_windows(
            payload,
            sess.get('video_infos'),
            session_start=sess.get('start_seconds'),
            session_end=sess.get('end_seconds'),
        )
        if window_errors:
            return _api_error(
                window_errors[0],
                400,
                code='invalid_draft',
                message_en='One or more video or ROI time windows are invalid.',
                window_errors=window_errors,
            )
        _detach_exact_replay_after_edit(sess)
        analysis_sessions.save_draft(session_id, payload, selected_video_idx=selected_video_idx)
        cards = summarize_batch_state(
            sess.get('experiment_type'),
            payload,
            sess.get('video_infos'),
            project_context=sess.get('project_context'),
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
            return _api_error('会话不存在', 404, code='session_not_found')

        if analysis_sessions.has_active_analysis(session_id):
            return _api_error('分析任务运行期间不能修改配置。', 409, code='analysis_config_locked')
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _api_error('草稿操作数据格式无效', 400, code='invalid_draft_action')
        if is_experiment_release_hidden(sess.get('experiment_type')):
            return jsonify({'success': False, 'error': '三箱社交模块未开放；内部验证可设置 MOUSESCOPE_ENABLE_3CST=1。'}), 403
        action = str(payload.get('action') or '').strip().lower()
        draft_payload = copy.deepcopy(sess.get('draft_payload') or {})
        total_videos = len(sess.get('video_infos') or [])
        per_video_configs = draft_payload.get('per_video_configs')
        if not isinstance(per_video_configs, list):
            per_video_configs = []
        while len(per_video_configs) < total_videos:
            per_video_configs.append({})
        workspace_configs = draft_payload.get('workspace_video_configs')
        if not isinstance(workspace_configs, list):
            workspace_configs = []
        while len(workspace_configs) < total_videos:
            workspace_configs.append({})
        video_infos = copy.deepcopy(sess.get('video_infos') or [])
        project_context = sess.get('project_context') if isinstance(sess.get('project_context'), dict) else None

        def _clear_project_roi_identity(config: dict) -> None:
            if not project_context:
                return
            identity_keys = ('project_unit_id', 'subject_id', 'animal_code', 'group_id', 'group_name')
            for roi_index, roi in enumerate(config.get('rois') or []):
                if not isinstance(roi, dict):
                    continue
                previous_code = str(roi.get('animal_code') or '')
                for key in identity_keys:
                    roi.pop(key, None)
                if previous_code and str(roi.get('name') or '') == previous_code:
                    roi['name'] = f'ROI {roi_index + 1}'

        def _refreshed_video_info(index: int, config: dict) -> dict:
            info = copy.deepcopy(video_infos[index]) if isinstance(video_infos[index], dict) else {}
            info['orientation_override'] = orientation_override_degrees(config.get('orientation_override'))
            _update_video_entry_info(info)
            return info

        def _payload_int(value, label: str) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                raise ValueError(f'{label}无效')

        if action == 'copy':
            try:
                source_index = _payload_int(payload.get('source_index', -1), '源视频索引')
            except ValueError as exc:
                return jsonify({'error': str(exc)}), 400
            target_indices = payload.get('target_indices') or []
            if not (0 <= source_index < total_videos):
                return jsonify({'error': '源视频索引无效'}), 400
            if not isinstance(target_indices, list) or not target_indices:
                return jsonify({'error': '目标视频索引无效'}), 400
            source_config = copy.deepcopy(per_video_configs[source_index] or {})
            try:
                source_info = _refreshed_video_info(source_index, source_config)
            except Exception:
                logger.exception('Failed to refresh source video before copying configuration: session=%s', session_id)
                return _api_error('无法读取源视频信息，配置未复制。', 400, code='copy_video_probe_failed')
            parsed_targets: list[int] = []
            for raw_idx in target_indices:
                try:
                    idx = _payload_int(raw_idx, '目标视频索引')
                except ValueError as exc:
                    return jsonify({'error': str(exc)}), 400
                if not (0 <= idx < total_videos):
                    return jsonify({'error': '目标视频索引无效'}), 400
                if idx not in parsed_targets:
                    parsed_targets.append(idx)

            staged: list[tuple[int, dict, dict]] = []
            for idx in parsed_targets:
                copied_config = copy.deepcopy(source_config)
                _clear_project_roi_identity(copied_config)
                try:
                    target_info = _refreshed_video_info(idx, copied_config)
                except Exception:
                    logger.exception('Failed to refresh target video while copying configuration: session=%s video=%s', session_id, idx)
                    return _api_error(
                        f'无法读取第 {idx + 1} 个目标视频的信息，配置未复制。',
                        400,
                        code='copy_video_probe_failed',
                    )
                if copied_config.get('rois'):
                    source_size = (int(source_info.get('width') or 0), int(source_info.get('height') or 0))
                    target_size = (int(target_info.get('width') or 0), int(target_info.get('height') or 0))
                    if source_size != target_size:
                        return _api_error(
                            f'第 {idx + 1} 个目标视频画面尺寸为 {target_size[0]}×{target_size[1]}，'
                            f'与源视频 {source_size[0]}×{source_size[1]} 不一致，不能直接复制像素 ROI。',
                            400,
                            code='copy_geometry_mismatch',
                        )
                window_errors = validate_analysis_windows(
                    {'per_video_configs': [copied_config]},
                    [target_info],
                )
                if window_errors:
                    return _api_error(
                        f'第 {idx + 1} 个目标视频：{window_errors[0]}',
                        400,
                        code='copy_window_invalid',
                        window_errors=window_errors,
                    )
                staged.append((idx, copied_config, target_info))

            for idx, copied_config, target_info in staged:
                per_video_configs[idx] = copied_config
                workspace_configs[idx] = copy.deepcopy(copied_config)
                video_infos[idx] = target_info
        elif action == 'reset':
            try:
                target_index = _payload_int(payload.get('target_index', -1), '目标视频索引')
            except ValueError as exc:
                return jsonify({'error': str(exc)}), 400
            if not (0 <= target_index < total_videos):
                return jsonify({'error': '目标视频索引无效'}), 400
            per_video_configs[target_index] = {}
            workspace_configs[target_index] = {}
            try:
                video_infos[target_index] = _refreshed_video_info(target_index, {})
            except Exception:
                logger.exception('Failed to reset target video orientation metadata: session=%s video=%s', session_id, target_index)
                return _api_error('无法恢复视频方向信息，配置未重置。', 400, code='reset_video_probe_failed')
        else:
            return jsonify({'error': '不支持的草稿操作'}), 400

        draft_payload['per_video_configs'] = per_video_configs
        draft_payload['workspace_video_configs'] = workspace_configs
        _detach_exact_replay_after_edit(sess)
        sess['video_infos'] = video_infos
        analysis_sessions.save_draft(session_id, draft_payload)
        cards = summarize_batch_state(
            sess.get('experiment_type'),
            draft_payload,
            sess.get('video_infos'),
            project_context=sess.get('project_context'),
        )
        return jsonify({
            'success': True,
            'draft_updated_at': sess.get('draft_updated_at'),
            'cards': cards,
        })


    @app.route('/api/video_settings/<session_id>', methods=['POST'])
    @app.route('/api/batch_video_settings/<session_id>', methods=['POST'])
    def api_video_settings(session_id):
        """Save the pre-calibration time, FPS, and orientation for one video."""
        sess = analysis_sessions.get(session_id)
        if not sess:
            return jsonify({'error': '会话不存在'}), 404
        if analysis_sessions.has_active_analysis(session_id):
            return _api_error('分析任务运行期间不能修改配置。', 409, code='analysis_config_locked')
        if is_experiment_release_hidden(sess.get('experiment_type')):
            return jsonify({'success': False, 'error': '三箱社交模块未开放；内部验证可设置 MOUSESCOPE_ENABLE_3CST=1。'}), 403

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _api_error('视频配置数据格式无效', 400, code='invalid_video_settings')
        validate_only = payload.get('validate_only') is True
        try:
            video_index = int(payload.get('video_index'))
        except (TypeError, ValueError):
            return jsonify({'error': '视频索引无效'}), 400
        video_infos = sess.get('video_infos') or []
        if not (0 <= video_index < len(video_infos)):
            return jsonify({'error': '视频索引无效'}), 400

        def optional_number(key: str, label: str) -> float | None:
            value = payload.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f'{label}必须是数字')
            if not math.isfinite(number):
                raise ValueError(f'{label}必须是有限数字')
            return number

        try:
            start_sec = optional_number('start_sec', '起始时间')
            end_sec = optional_number('end_sec', '结束时间')
            fps_override = optional_number('fps_override', 'FPS')
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        if start_sec is not None and start_sec < 0:
            return jsonify({'error': '起始时间不能小于 0'}), 400
        if end_sec is not None and end_sec <= 0:
            return jsonify({'error': '结束时间必须大于 0'}), 400
        if start_sec is not None and end_sec is not None and end_sec <= start_sec:
            return jsonify({'error': '结束时间必须大于起始时间'}), 400
        if fps_override is not None and not (0.001 <= fps_override <= 1000.0):
            return jsonify({'error': 'FPS 必须在 0.001 到 1000 之间'}), 400

        # Work on an isolated copy until every validation and metadata refresh
        # succeeds.  Otherwise a failed request can still leak a partially
        # changed orientation into the in-memory session through the dict
        # reference stored in ``video_infos``.
        info = copy.deepcopy(video_infos[video_index]) if isinstance(video_infos[video_index], dict) else {}
        duration = effective_video_duration_seconds(info, fps_override)
        if duration and start_sec is not None and start_sec >= duration:
            return jsonify({'error': f'起始时间必须小于视频时长（{duration:.3f} 秒）'}), 400
        if duration and end_sec is not None and end_sec > duration + 0.001:
            return jsonify({'error': f'结束时间不能超过视频时长（{duration:.3f} 秒）'}), 400

        raw_orientation = payload.get('orientation_override', 'auto')
        orientation_override = orientation_override_degrees(raw_orientation)
        is_auto_orientation = raw_orientation is None or (
            isinstance(raw_orientation, str) and raw_orientation.strip().lower() in {'', 'auto', 'metadata', 'detected'}
        )
        if orientation_override is None and not is_auto_orientation:
            return jsonify({'error': '方向只能设为自动、0、90、180 或 270 度'}), 400

        draft_payload = copy.deepcopy(sess.get('draft_payload') or {})
        configs = draft_payload.get('per_video_configs')
        if not isinstance(configs, list):
            configs = []
        while len(configs) < len(video_infos):
            configs.append({})
        existing_config = configs[video_index] if isinstance(configs[video_index], dict) else {}
        previous_orientation = orientation_override_degrees(existing_config.get('orientation_override'))
        orientation_changed = previous_orientation != orientation_override
        has_existing_roi = bool(existing_config.get('rois'))
        if orientation_changed and has_existing_roi and not bool(payload.get('reset_configuration')):
            return jsonify({
                'success': False,
                'requires_recalibration': True,
                'error': '改变视频方向会使已有 ROI 坐标失效；请确认后重新标定该视频。',
            }), 409
        if orientation_changed and has_existing_roi:
            existing_config = {}
            configs[video_index] = existing_config
            workspace_configs = draft_payload.get('workspace_video_configs')
            if isinstance(workspace_configs, list) and video_index < len(workspace_configs):
                workspace_configs[video_index] = {}

        existing_config.update({
            'start_sec': max(0.0, float(start_sec)) if start_sec is not None else None,
            'end_sec': float(end_sec) if end_sec is not None else None,
            'fps_override': float(fps_override) if fps_override is not None else None,
            'orientation_override': orientation_override,
        })
        roi_window_errors = validate_roi_windows(
            existing_config.get('rois'),
            video_start=existing_config.get('start_sec'),
            video_end=existing_config.get('end_sec') if existing_config.get('end_sec') is not None else duration,
            video_label=f'第 {video_index + 1} 个视频',
        )
        if roi_window_errors:
            return jsonify({'error': roi_window_errors[0], 'window_errors': roi_window_errors}), 400
        configs[video_index] = existing_config
        draft_payload['per_video_configs'] = configs

        info['orientation_override'] = orientation_override
        try:
            _update_video_entry_info(info)
        except Exception:
            logger.exception('Failed to refresh video metadata after orientation change: session=%s video=%s', session_id, video_index)
            return jsonify({'error': '无法更新视频方向信息'}), 400
        if validate_only:
            return jsonify({
                'success': True,
                'validated': True,
                'video_info': info,
            })
        video_infos[video_index] = info
        sess['video_infos'] = video_infos
        _detach_exact_replay_after_edit(sess)
        analysis_sessions.save_draft(session_id, draft_payload)
        cards = _batch_cards(sess)
        return jsonify({
            'success': True,
            'cards': cards,
            'draft_updated_at': sess.get('draft_updated_at'),
            'video_info': cards[video_index].get('video_info') if video_index < len(cards) else None,
        })


    @app.route('/api/analysis_config/<session_id>', methods=['POST'])
    def api_analysis_config(session_id):
        """Save session-level fields that belong to the unified analysis configuration page."""
        sess = analysis_sessions.get(session_id)
        if not sess:
            return _api_error('会话不存在', 404, code='session_not_found')
        if analysis_sessions.has_active_analysis(session_id):
            return _api_error('分析任务运行期间不能修改配置。', 409, code='analysis_config_locked')
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _api_error('分析配置数据格式无效', 400, code='invalid_analysis_config')
        if sess.get('project_context'):
            return jsonify({
                'success': True,
                'output_dir': sess.get('output_dir'),
                'project_managed': True,
            })
        raw_output_dir = str(payload.get('output_dir') or '').strip().strip('"').strip("'")
        try:
            output_dir = _resolve_result_dir(raw_output_dir) if raw_output_dir else None
            if output_dir:
                output_dir = _ensure_writable_output_directory(output_dir)
        except (TypeError, ValueError, OSError) as exc:
            return _output_directory_error(exc)
        _detach_exact_replay_after_edit(sess)
        sess['output_dir'] = output_dir
        analysis_sessions.persist(session_id)
        return jsonify({'success': True, 'output_dir': output_dir, 'project_managed': False})


    @app.route('/api/debug_preview/<session_id>', methods=['POST'])
    def debug_preview(session_id):
        """参数调试预览"""
        sess = analysis_sessions.get(session_id)
        if not sess:
            return _api_error('会话不存在', 404, code='session_not_found')

        if is_experiment_release_hidden(sess.get('experiment_type')):
            return jsonify({'success': False, 'error': '三箱社交模块未开放；内部验证可设置 MOUSESCOPE_ENABLE_3CST=1。'}), 403

        data = request.get_json()
        if not isinstance(data, dict):
            data = {}
        return build_debug_preview_response(session_id, sess, data)


    @app.route('/api/visual_style', methods=['GET'])
    def get_visual_style():
        if visual_style_store is None:
            return jsonify({'error': 'Visual style store is not configured.'}), 500
        payload = visual_style_store.get_payload()
        payload['preview_experiments'] = preview_experiment_payload()
        payload['success'] = True
        return jsonify(payload)


    @app.route('/api/visual_style', methods=['POST'])
    def save_visual_style():
        if visual_style_store is None:
            return jsonify({'error': 'Visual style store is not configured.'}), 500
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            data = {}
        style = data.get('style', data)
        saved = visual_style_store.save_style(style)
        return jsonify({'success': True, 'style': saved})


    @app.route('/api/visual_style/reset', methods=['POST'])
    def reset_visual_style():
        if visual_style_store is None:
            return jsonify({'error': 'Visual style store is not configured.'}), 500
        saved = visual_style_store.reset()
        return jsonify({'success': True, 'style': saved})


    @app.route('/api/visual_style/preview', methods=['POST'])
    def preview_visual_style():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            data = {}
        style = data.get('style')
        if style is None and visual_style_store is not None:
            style = visual_style_store.get_style()
        experiment = data.get('experiment') or data.get('experiment_type') or 'open_field'
        try:
            payload = render_visual_style_preview(style, experiment)
        except Exception as exc:
            logger.exception("Failed to render visual style preview.")
            return jsonify({'success': False, 'error': str(exc)}), 500
        payload['success'] = True
        return jsonify(payload)


    @app.route('/api/experiment_defaults', methods=['GET'])
    def get_experiment_defaults():
        if experiment_defaults_store is None:
            return jsonify({'error': 'Experiment defaults store is not configured.'}), 500
        payload = experiment_defaults_store.get_payload()
        payload['success'] = True
        return jsonify(payload)


    @app.route('/api/experiment_defaults', methods=['POST'])
    def save_experiment_defaults():
        if experiment_defaults_store is None:
            return jsonify({'error': 'Experiment defaults store is not configured.'}), 500
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': '默认配置格式无效'}), 400
        defaults = data.get('defaults', data)
        try:
            saved = experiment_defaults_store.save_defaults(defaults)
        except ExperimentDefaultsValidationError as exc:
            return jsonify({
                'success': False,
                'error': str(exc),
                'error_en': exc.message_en,
                'validation_errors': exc.errors,
            }), 400
        return jsonify({'success': True, 'defaults': saved})


    @app.route('/api/experiment_defaults/reset', methods=['POST'])
    def reset_experiment_defaults():
        if experiment_defaults_store is None:
            return jsonify({'error': 'Experiment defaults store is not configured.'}), 500
        saved = experiment_defaults_store.reset()
        return jsonify({'success': True, 'defaults': saved})


    @app.route('/api/start_analysis/<session_id>', methods=['POST'])
    def start_analysis(session_id):
        """启动分析"""
        sess = analysis_sessions.get(session_id)
        if not sess:
            return _api_error('会话不存在', 404, code='session_not_found')
        if not analysis_sessions.reserve_start(session_id):
            return _api_error('分析任务已在运行，请勿重复启动。', 409, code='analysis_already_running')

        @after_this_request
        def _release_start_reservation(response):
            analysis_sessions.release_start(session_id)
            return response

        saved_draft = copy.deepcopy(sess.get('draft_payload') or {})
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            data = {}
        use_saved_draft = bool(data.get('use_saved_draft'))
        if use_saved_draft:
            draft_payload = copy.deepcopy(saved_draft)
            if not draft_payload:
                return _api_error('当前没有可用于启动分析的已保存配置', 400, code='missing_saved_config')
            data = draft_payload
        elif not data and saved_draft:
            data = copy.deepcopy(saved_draft)

        # Single-video calibration pages start analysis directly and serialize
        # assay/ROI fields only. Preserve the video-level time window, FPS and
        # orientation saved on the unified analysis configuration page before
        # validating or replacing the stored draft.
        data = _preserve_per_video_runtime_settings(
            data,
            saved_draft,
            len(sess.get('video_infos') or []),
        )

        window_errors = validate_analysis_windows(
            data,
            sess.get('video_infos'),
            session_start=sess.get('start_seconds'),
            session_end=sess.get('end_seconds'),
        )
        if window_errors:
            return _api_error(
                window_errors[0],
                400,
                code='invalid_draft',
                message_en='One or more video or ROI time windows are invalid.',
                window_errors=window_errors,
            )

        binding_error = _project_subject_binding_error(sess, data)
        if binding_error:
            return _api_error(binding_error, 400, code='project_subject_required')

        output_root = (
            sess.get('output_dir')
            if isinstance(sess.get('project_context'), dict)
            else (sess.get('output_root_dir') or sess.get('output_dir') or RESULT_DIR)
        )
        try:
            _ensure_writable_output_directory(_resolve_result_dir(output_root))
        except (TypeError, ValueError, OSError) as exc:
            return _output_directory_error(exc)

        if isinstance(data, dict) and data:
            selected_video_idx = data.get('video', sess.get('selected_video_idx'))
            analysis_sessions.save_draft(session_id, data, selected_video_idx=selected_video_idx)

        prepare_session_run_output(
            sess,
            session_id=session_id,
            default_output_root=RESULT_DIR,
            resolve_output_dir=_resolve_result_dir,
            force_new=sess.get('analysis') is not None,
        )
        analysis_sessions.persist(session_id)

        experiment_type = str(sess.get('experiment_type') or 'tst').strip().lower()
        if is_experiment_release_hidden(experiment_type):
            return _api_error(
                '三箱社交模块未开放；内部验证可设置 MOUSESCOPE_ENABLE_3CST=1。',
                403,
                code='three_chamber_hidden',
            )
        replay_config = sess.get('replay_resolved_config') if isinstance(sess.get('replay_resolved_config'), dict) else None
        if replay_config:
            try:
                vision_config = resolved_config_for_replay({
                    'resolved_config': replay_config,
                    'random_seed': sess.get('reproducibility_seed'),
                }, output_dir=sess.get('output_dir') or RESULT_DIR)
            except ReproducibilityError as exc:
                return _api_error(
                    str(exc),
                    409,
                    code='snapshot_read_failed',
                    message_en='The saved resolved configuration is missing or invalid.',
                )
        else:
            global_defaults = experiment_defaults_store.get_defaults() if experiment_defaults_store is not None else None
            vision_config = build_analysis_config(experiment_type, sess, data, global_defaults=global_defaults)
            if visual_style_store is not None:
                vision_config['visual_style'] = visual_style_store.get_style()

        reproducibility_seed = apply_deterministic_seed(
            sess.get('reproducibility_seed') or vision_config.get('reproducibility_seed') or DEFAULT_RANDOM_SEED
        )
        sess['reproducibility_seed'] = reproducibility_seed
        vision_config['reproducibility_seed'] = reproducibility_seed
        vision_config.setdefault(
            'resource_policy',
            resolved_run_resource_policy(
                len(sess.get('video_paths') or []),
                requested_workers=vision_config.get('batch_workers') or os.environ.get('MOUSESCOPE_BATCH_WORKERS'),
            ),
        )
        try:
            run_snapshot = build_run_snapshot(
                session_id=session_id,
                app_version=app_version,
                sess=sess,
                resolved_config=vision_config,
            )
            snapshot_path = write_run_snapshot(
                vision_config.get('output_dir') or sess.get('output_dir') or RESULT_DIR,
                run_snapshot,
            )
        except (OSError, TypeError, ValueError) as exc:
            logger.exception("Failed to persist run snapshot for session %s", session_id)
            return _api_error(
                f'无法写入运行快照，分析未启动：{exc}',
                500,
                code='snapshot_write_failed',
            )
        analysis_class_map = {
            'tst': VisionAnalysisSession,
            'open_field': OpenFieldMultiArenaAnalysisSession,
            'novel_object': NovelObjectMultiArenaAnalysisSession,
            'elevated_plus': ElevatedPlusMultiArenaAnalysisSession,
            'y_maze': YMazeMultiArenaAnalysisSession,
            'forced_swim': ForcedSwimMultiCylinderAnalysisSession,
            'light_dark_box': LightDarkBoxMultiArenaAnalysisSession,
            'three_chamber': ThreeChamberSocialAnalysisSession,
            'water_maze': WaterMazeMultiPoolAnalysisSession,
        }
        analysis_class = analysis_class_map.get(experiment_type, VisionAnalysisSession)
        analysis = analysis_class(
            session_id=session_id,
            video_paths=sess['video_paths'],
            video_infos=sess['video_infos'],
            config=vision_config
        )
        append_result = getattr(analysis, '_append_result_file', None)
        if callable(append_result):
            append_result(snapshot_path, group='metadata')
        elif snapshot_path not in analysis.result_files:
            analysis.result_files.append(snapshot_path)

        if not analysis_sessions.commit_reserved_analysis(session_id, analysis):
            return _api_error('分析任务已在运行，请勿重复启动。', 409, code='analysis_already_running')
        try:
            analysis.start()
        except RuntimeError as exc:
            analysis.is_running = False
            analysis.status = 'error'
            analysis.message = str(exc)
            analysis.end_time = time.time()
            analysis_history.upsert_runtime(session_id, sess, analysis=analysis)
            return _api_error(str(exc), 409, message_en=str(exc))
        analysis_history.upsert_runtime(session_id, sess, analysis=analysis)
        project_context = sess.get('project_context') if isinstance(sess.get('project_context'), dict) else None
        if project_store is not None and project_context:
            project_store.upsert_project_run(
                project_id=project_context.get('project_id'),
                project_run_id=project_context.get('project_run_id'),
                run_id=session_id,
                session_id=session_id,
                experiment_type=experiment_type,
                assignment_ids=project_context.get('assignment_ids') or [],
                output_dir=vision_config.get('output_dir') or sess.get('output_dir') or '',
                status='analyzing',
                progress=0,
                message='Analysis started from project workspace',
            )
            project_store.set_assignments_status(
                project_context.get('project_id'),
                project_context.get('assignment_ids') or [],
                'analyzing',
                latest_run_id=session_id,
            )

        return jsonify({'success': True, 'session_id': session_id})


    @app.route('/api/status/<session_id>')
    def get_status(session_id):
        """获取分析状态"""
        sess = analysis_sessions.get(session_id)
        if not sess or 'analysis' not in sess:
            return jsonify({'error': '会话不存在'}), 404
        status_payload = sess['analysis'].get_status()
        analysis_history.upsert_runtime(session_id, sess, analysis=sess['analysis'], status_payload=status_payload)
        project_context = sess.get('project_context') if isinstance(sess.get('project_context'), dict) else None
        if project_store is not None and project_context:
            status = str(status_payload.get('status') or '').lower()
            finished_at = None
            if status in {'completed', 'error', 'stopped', 'interrupted'}:
                finished_at = getattr(sess['analysis'], 'end_time', None)
            progress = int(status_payload.get('progress') or 0)
            if status == 'completed':
                progress = 100
            project_store.upsert_project_run(
                project_id=project_context.get('project_id'),
                project_run_id=project_context.get('project_run_id'),
                run_id=session_id,
                session_id=session_id,
                experiment_type=status_payload.get('experiment_type') or sess.get('experiment_type'),
                assignment_ids=project_context.get('assignment_ids') or [],
                output_dir=sess.get('output_dir') or '',
                status=status or 'unknown',
                progress=progress,
                message=str(status_payload.get('message') or ''),
                result_files=status_payload.get('result_files') or [],
                finished_at=finished_at,
            )
            if status in {'completed', 'error', 'stopped', 'interrupted'}:
                project_store.set_assignments_status(
                    project_context.get('project_id'),
                    project_context.get('assignment_ids') or [],
                    status,
                    latest_run_id=session_id,
                )
                if status == 'completed':
                    project_store.cleanup_completed_upload_caches(
                        project_context.get('project_id'),
                        project_context.get('assignment_ids') or [],
                    )
        return jsonify(status_payload)


    @app.route('/api/stop/<session_id>', methods=['POST'])
    def stop_analysis(session_id):
        """停止分析"""
        sess = analysis_sessions.get(session_id)
        if sess and sess.get('analysis') is not None:
            analysis = sess['analysis']
            stopped_now = bool(analysis.stop())
            analysis_history.upsert_runtime(session_id, sess, analysis=analysis)
            project_context = sess.get('project_context') if isinstance(sess.get('project_context'), dict) else None
            if stopped_now and project_store is not None and project_context:
                project_store.set_assignments_status(
                    project_context.get('project_id'),
                    project_context.get('assignment_ids') or [],
                    'stopped',
                    latest_run_id=session_id,
                )
            if stopped_now:
                return jsonify({'success': True, 'message': '已发送停止命令'})
            return jsonify({
                'success': True,
                'message': '分析任务已经结束，无需重复停止。',
                'status': str(getattr(analysis, 'status', '') or ''),
            })
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
        zip_path = _temp_zip_path(f'{session_id}_results')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in analysis.result_files:
                if os.path.exists(f):
                    zf.write(f, _result_zip_arcname(f, result_dir))

        return _send_zip_and_cleanup(zip_path, 'MouseScope_results.zip')


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
        zip_path = _temp_zip_path(f'{session_id}_{file_type}')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                zf.write(path, _result_zip_arcname(path, result_dir))
        return _send_zip_and_cleanup(zip_path, f'MouseScope_{file_type}.zip')


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
