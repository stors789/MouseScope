from __future__ import annotations

import base64
import copy
import os
import platform
import re
import subprocess
import zipfile
from pathlib import Path

import cv2
from flask import jsonify, request, send_file, url_for

from services.analysis_builders import build_analysis_config
from services.analysis_sessions import (
    ElevatedPlusMultiArenaAnalysisSession,
    NovelObjectMultiArenaAnalysisSession,
    OpenFieldMultiArenaAnalysisSession,
    VisionAnalysisSession,
    YMazeMultiArenaAnalysisSession,
)
from services.debug_preview import build_debug_preview_response
from services.experiment_registry import summarize_batch_state


def register_api_routes(app, ctx) -> None:
    analysis_sessions = ctx.analysis_sessions
    analysis_history = ctx.analysis_history
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
            task.get('primary_video_name'),
            task.get('message'),
            task.get('output_dir'),
        ]
        fields.extend(task.get('video_names') or [])
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
        if experiment_filter and str(task.get('experiment_type') or '') != experiment_filter:
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
        if task.get('can_delete') and run_id:
            task['delete_url'] = url_for('delete_task_record', run_id=run_id)
        return task

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
        collected = []
        for root, _dirs, names in os.walk(output_dir):
            for name in names:
                path = os.path.join(root, name)
                if name.endswith('_history_results.zip'):
                    continue
                if os.path.isfile(path):
                    collected.append(path)
        return collected

    @app.route('/api/tasks')
    def api_tasks():
        query = str(request.args.get('q') or '').strip().lower()
        status_filter = str(request.args.get('status') or '').strip().lower()
        experiment_filter = str(request.args.get('experiment') or '').strip().lower()

        merged: dict[str, dict] = {}
        for task in analysis_history.list_records():
            merged[task['run_id']] = task

        for session_id, sess in analysis_sessions.items():
            task = analysis_history.build_runtime_task(session_id, sess)
            if sess.get('analysis') is not None:
                analysis_history.upsert_runtime(session_id, sess, analysis=sess.get('analysis'))
            previous = merged.get(task['run_id']) or {}
            merged[task['run_id']] = {**previous, **task}

        tasks = [_decorate_task_urls(task) for task in merged.values()]
        tasks.sort(key=lambda item: float(item.get('updated_at') or item.get('created_at') or 0), reverse=True)

        all_count = len(tasks)
        active_count = sum(1 for task in tasks if str(task.get('status') or '').lower() in {'waiting', 'analyzing'})
        completed_count = sum(1 for task in tasks if str(task.get('status') or '').lower() == 'completed')
        interrupted_count = sum(1 for task in tasks if str(task.get('status') or '').lower() == 'interrupted')
        error_count = sum(1 for task in tasks if str(task.get('status') or '').lower() == 'error')

        filtered = [
            task for task in tasks
            if _task_matches_filters(task, query, status_filter, experiment_filter)
        ]
        limit = max(1, min(int(request.args.get('limit') or 200), 1000))
        filtered = filtered[:limit]

        return jsonify({
            'success': True,
            'tasks': [] if request.args.get('summary') else filtered,
            'counts': {
                'all': all_count,
                'active': active_count,
                'completed': completed_count,
                'interrupted': interrupted_count,
                'error': error_count,
                'filtered': len(filtered),
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
            analysis_sessions.pop(session_id, None)
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

    @app.route('/api/history/download_all/<run_id>')
    def history_download_all(run_id):
        record = analysis_history.get_record(run_id)
        if not record:
            return jsonify({'error': '历史记录不存在'}), 404
        files = _history_result_files(record)
        if not files:
            return jsonify({'error': '没有可下载的结果文件'}), 404

        output_dir = _history_output_dir(record)
        safe_run_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(run_id))[:80] or 'history'
        zip_path = str(RESULT_DIR / f'{safe_run_id}_history_results.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for path in files:
                if os.path.exists(path):
                    if output_dir and os.path.isdir(output_dir):
                        arcname = _result_zip_arcname(path, output_dir)
                    else:
                        arcname = os.path.basename(path)
                    zf.write(path, arcname)

        return send_file(zip_path, as_attachment=True, download_name='MouseScope_history_results.zip')

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
        analysis_history.upsert_runtime(session_id, sess, analysis=analysis)

        return jsonify({'success': True, 'session_id': session_id})


    @app.route('/api/status/<session_id>')
    def get_status(session_id):
        """获取分析状态"""
        sess = analysis_sessions.get(session_id)
        if not sess or 'analysis' not in sess:
            return jsonify({'error': '会话不存在'}), 404
        status_payload = sess['analysis'].get_status()
        analysis_history.upsert_runtime(session_id, sess, analysis=sess['analysis'], status_payload=status_payload)
        return jsonify(status_payload)


    @app.route('/api/stop/<session_id>', methods=['POST'])
    def stop_analysis(session_id):
        """停止分析"""
        sess = analysis_sessions.get(session_id)
        if sess and 'analysis' in sess:
            sess['analysis'].stop()
            analysis_history.upsert_runtime(session_id, sess, analysis=sess['analysis'])
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
