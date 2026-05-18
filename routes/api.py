from __future__ import annotations

import base64
import copy
import logging
import os
import platform
import re
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
from flask import jsonify, request, send_file, url_for

from services.analysis_builders import build_analysis_config
from services.analysis_sessions import (
    ElevatedPlusMultiArenaAnalysisSession,
    ForcedSwimMultiCylinderAnalysisSession,
    NovelObjectMultiArenaAnalysisSession,
    OpenFieldMultiArenaAnalysisSession,
    VisionAnalysisSession,
    YMazeMultiArenaAnalysisSession,
)
from services.debug_preview import build_debug_preview_response
from services.experiment_registry import get_experiment_spec, summarize_batch_state
from services.video_frame_reader import read_video_frame


logger = logging.getLogger(__name__)


def register_api_routes(app, ctx) -> None:
    analysis_sessions = ctx.analysis_sessions
    analysis_history = ctx.analysis_history
    project_store = getattr(ctx, "project_store", None)
    RESULT_DIR = ctx.RESULT_DIR
    _resolve_result_dir = ctx._resolve_result_dir
    _result_zip_arcname = ctx._result_zip_arcname
    _session_video_info = ctx._session_video_info

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

    def _decorate_task_urls(task: dict) -> dict:
        session_id = task.get('session_id')
        run_id = task.get('run_id')
        if task.get('can_monitor') and session_id:
            task['monitor_url'] = url_for('monitor', session_id=session_id)
        if task.get('can_configure') and session_id:
            if task.get('is_batch'):
                task['configure_url'] = url_for('batch_workspace', session_id=session_id)
            else:
                task['configure_url'] = url_for('calibrate', session_id=session_id)
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

        def video_name(index: int) -> str:
            if 0 <= index < len(video_infos):
                info = video_infos[index] or {}
                return str(info.get('original_name') or info.get('name') or f'Video {index + 1}')
            return f'Video {index + 1}'

        def check_rois(rois, video_index: int) -> str:
            if not isinstance(rois, list) or not rois:
                return ''
            for roi_index, roi in enumerate(rois, start=1):
                subject_id = str((roi or {}).get('subject_id') or '').strip() if isinstance(roi, dict) else ''
                if not subject_id or subject_id not in valid_subject_ids:
                    return f"{video_index + 1}. {video_name(video_index)}: ROI {roi_index} 未绑定项目小鼠，请从已有小鼠中选择。"
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
        response = send_file(zip_path, as_attachment=True, download_name=download_name)

        @response.call_on_close
        def _cleanup_zip():
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            except Exception:
                pass

        return response

    @app.route('/api/tasks')
    def api_tasks():
        query = str(request.args.get('q') or '').strip().lower()
        status_filter = str(request.args.get('status') or '').strip().lower()
        experiment_filter = str(request.args.get('experiment') or '').strip().lower()
        try:
            limit = int(request.args.get('limit') or 200)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': '任务数量参数无效'}), 400
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
            analysis = sess.get('analysis')
            if analysis is not None:
                status = str(getattr(analysis, 'status', '') or '').lower()
                is_running = bool(getattr(analysis, 'is_running', False)) or status in {'waiting', 'analyzing'}
                if is_running:
                    return jsonify({'success': False, 'error': '任务仍在运行，请先停止分析。'}), 400
            analysis_sessions.delete(session_id)
            removed_runtime = True

        deleted_history = analysis_history.delete_record(run_id)
        if session_id != run_id:
            deleted_history = analysis_history.delete_record(session_id) or deleted_history

        if not deleted_history and not removed_runtime:
            return jsonify({'success': False, 'error': '任务记录不存在'}), 404

        return jsonify({
            'success': True,
            'run_id': run_id,
            'session_id': session_id,
            'deleted_history': bool(deleted_history),
            'removed_runtime': bool(removed_runtime),
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
                analysis = sess.get('analysis')
                if analysis is not None:
                    status = str(getattr(analysis, 'status', '') or '').lower()
                    is_running = bool(getattr(analysis, 'is_running', False)) or status in {'waiting', 'analyzing'}
                    if is_running:
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
            return jsonify({'error': '会话不存在'}), 404

        try:
            frame_idx = _nonnegative_int(request.args.get('frame', 0), 'frame')
            video_idx = _nonnegative_int(request.args.get('video', 0), 'video')
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

        video_paths = sess.get('video_paths') or []
        if not video_paths:
            return jsonify({'error': '会话没有视频文件'}), 404
        video_idx = min(video_idx, len(video_paths) - 1)
        video_info = _session_video_info(sess, video_idx)
        total_frames = int(video_info.get('total_frames') or 0)
        if total_frames > 0:
            frame_idx = min(frame_idx, total_frames - 1)

        video_path = video_paths[video_idx]
        frame = read_video_frame(video_path, frame_idx, fps=video_info.get('detected_fps') or video_info.get('fps'))

        if frame is None:
            return jsonify({'error': '无法读取帧'}), 400

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': img_b64,
            'frame_idx': frame_idx,
            'total_frames': video_info['total_frames'],
            'video_info': video_info,
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
            for raw_idx in target_indices:
                try:
                    idx = _payload_int(raw_idx, '目标视频索引')
                except ValueError as exc:
                    return jsonify({'error': str(exc)}), 400
                if 0 <= idx < total_videos:
                    per_video_configs[idx] = copy.deepcopy(source_config)
        elif action == 'reset':
            try:
                target_index = _payload_int(payload.get('target_index', -1), '目标视频索引')
            except ValueError as exc:
                return jsonify({'error': str(exc)}), 400
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
            project_context=sess.get('project_context'),
        )
        return jsonify({
            'success': True,
            'draft_updated_at': sess.get('draft_updated_at'),
            'cards': cards,
        })


    @app.route('/api/debug_preview/<session_id>', methods=['POST'])
    def debug_preview(session_id):
        """参数调试预览"""
        sess = analysis_sessions.get(session_id)
        if not sess:
            return jsonify({'error': '会话不存在'}), 404

        data = request.get_json()
        if not isinstance(data, dict):
            data = {}
        return build_debug_preview_response(session_id, sess, data)


    @app.route('/api/start_analysis/<session_id>', methods=['POST'])
    def start_analysis(session_id):
        """启动分析"""
        sess = analysis_sessions.get(session_id)
        if not sess:
            return jsonify({'error': '会话不存在'}), 404
        if analysis_sessions.has_active_analysis(session_id):
            return jsonify({'success': False, 'error': '分析任务已在运行，请勿重复启动。'}), 409

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

        binding_error = _project_subject_binding_error(sess, data)
        if binding_error:
            return jsonify({'success': False, 'error': binding_error}), 400

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
            'forced_swim': ForcedSwimMultiCylinderAnalysisSession,
        }
        analysis_class = analysis_class_map.get(experiment_type, VisionAnalysisSession)
        analysis = analysis_class(
            session_id=session_id,
            video_paths=sess['video_paths'],
            video_infos=sess['video_infos'],
            config=vision_config
        )

        if not analysis_sessions.set_analysis_if_idle(session_id, analysis):
            return jsonify({'success': False, 'error': '分析任务已在运行，请勿重复启动。'}), 409
        try:
            analysis.start()
        except RuntimeError as exc:
            analysis.is_running = False
            analysis.status = 'error'
            analysis.message = str(exc)
            analysis.end_time = time.time()
            analysis_history.upsert_runtime(session_id, sess, analysis=analysis)
            return jsonify({'success': False, 'error': str(exc)}), 409
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
        if sess and 'analysis' in sess:
            sess['analysis'].stop()
            analysis_history.upsert_runtime(session_id, sess, analysis=sess['analysis'])
            project_context = sess.get('project_context') if isinstance(sess.get('project_context'), dict) else None
            if project_store is not None and project_context:
                project_store.set_assignments_status(
                    project_context.get('project_id'),
                    project_context.get('assignment_ids') or [],
                    'stopped',
                    latest_run_id=session_id,
                )
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
