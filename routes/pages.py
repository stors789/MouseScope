from __future__ import annotations

import secrets
import os
import math
from pathlib import Path

from flask import flash, jsonify, redirect, render_template, request, session, url_for

from services.experiment_registry import (
    get_experiment_spec,
    is_experiment_release_hidden,
    is_three_chamber_enabled,
    normalize_experiment_type,
    summarize_batch_state,
)


def register_page_routes(app, ctx) -> None:
    UploadStorageError = ctx.UploadStorageError
    analysis_sessions = ctx.analysis_sessions
    project_store = getattr(ctx, "project_store", None)
    BUNDLE_DIR = ctx.BUNDLE_DIR
    VALID_VIDEO_EXTS = ctx.VALID_VIDEO_EXTS
    _cleanup_session_artifacts = ctx._cleanup_session_artifacts
    _extract_video_info_from_entry = ctx._extract_video_info_from_entry
    _get_original_name_for_path = ctx._get_original_name_for_path
    _now_ts = ctx._now_ts
    _resolve_result_dir = ctx._resolve_result_dir
    _save_uploaded_video = ctx._save_uploaded_video
    _session_video_info = ctx._session_video_info
    _stage_local_video_if_needed = ctx._stage_local_video_if_needed
    _update_video_entry_info = ctx._update_video_entry_info

    def _clean_local_path_line(raw_path: str) -> str:
        return str(raw_path or '').strip().strip('"').strip("'").strip('“”').strip('‘’').strip()

    def _parse_optional_float(raw_value: str, label: str, *, min_value: float | None = None, max_value: float | None = None) -> float | None:
        text = str(raw_value or '').strip()
        if not text:
            return None
        try:
            value = float(text)
        except (TypeError, ValueError):
            raise ValueError(f'{label}必须是数字')
        if not math.isfinite(value):
            raise ValueError(f'{label}必须是有效数字')
        if min_value is not None and value < min_value:
            raise ValueError(f'{label}不能小于 {min_value:g}')
        if max_value is not None and value > max_value:
            raise ValueError(f'{label}不能大于 {max_value:g}')
        return value

    def _validate_time_window(start_seconds: float | None, end_seconds: float | None) -> None:
        if end_seconds is not None and end_seconds <= (start_seconds or 0):
            raise ValueError('结束秒数必须大于开始秒数')

    def _scan_local_video_paths(local_paths_raw: str) -> tuple[list[dict], list[dict], str]:
        cleaned_paths = []
        valid_entries = []
        invalid_entries = []

        for line in str(local_paths_raw or '').splitlines():
            cleaned = _clean_local_path_line(line)
            if not cleaned:
                continue
            cleaned_paths.append(cleaned)
            path = Path(cleaned).expanduser()
            try:
                if path.is_file():
                    if path.suffix.lower() in VALID_VIDEO_EXTS:
                        valid_entries.append({'path': str(path), 'type': 'file', 'count': 1})
                    else:
                        invalid_entries.append({'path': cleaned, 'reason': '不支持的文件格式'})
                elif path.is_dir():
                    videos = sorted(
                        vf for vf in path.iterdir()
                        if vf.is_file() and vf.suffix.lower() in VALID_VIDEO_EXTS
                    )
                    if videos:
                        valid_entries.append({
                            'path': str(path),
                            'type': 'directory',
                            'count': len(videos),
                            'files': [vf.name for vf in videos[:20]],
                            'has_more': len(videos) > 20,
                        })
                    else:
                        invalid_entries.append({'path': cleaned, 'reason': '文件夹内没有支持的视频文件'})
                else:
                    invalid_entries.append({'path': cleaned, 'reason': '路径不存在'})
            except OSError as exc:
                invalid_entries.append({'path': cleaned, 'reason': f'无法访问路径：{exc}'})

        return valid_entries, invalid_entries, '\n'.join(cleaned_paths)

    @app.route('/')
    def index():
        """首页"""
        return render_template('home.html')


    @app.route('/upload')
    def upload_page():
        """上传视频页面"""
        return render_template('upload.html', enable_three_chamber=is_three_chamber_enabled())


    @app.route('/tasks')
    def tasks_page():
        """Task center."""
        return render_template('tasks.html', enable_three_chamber=is_three_chamber_enabled())


    @app.route('/manual')
    def manual():
        """使用手册"""
        manual_type = request.args.get('module', 'tst').strip().lower()
        manual_modules = {'tst', 'fst', 'oft', 'nor', 'epm', 'ymaze', 'ldb', 'mwm'}
        if is_three_chamber_enabled():
            manual_modules.add('three_chamber')
        if manual_type not in manual_modules:
            manual_type = 'tst'
        return render_template('manual.html', manual_type=manual_type, enable_three_chamber=is_three_chamber_enabled())


    @app.route('/manual/open_field')
    def manual_open_field():
        """Open field manual."""
        return redirect(url_for('manual', module='oft'))


    @app.route('/manual/novel_object')
    def manual_novel_object():
        """Novel object recognition manual."""
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

        try:
            for f in files:
                if f and f.filename:
                    ext = os.path.splitext(f.filename)[1].lower()
                    if ext not in VALID_VIDEO_EXTS:
                        continue
                    save_path, original_name = _save_uploaded_video(f, upload_id)
                    saved_paths.append(save_path)
                    saved_names.append(original_name)
        except UploadStorageError as exc:
            _cleanup_session_artifacts(upload_id)
            app.logger.warning("Upload failed because storage is full: %s", exc)
            return jsonify({'success': False, 'error': str(exc)}), 507

        if not saved_paths:
            return jsonify({'success': False, 'error': '没有有效的视频文件'}), 400

        return jsonify({'success': True, 'paths': saved_paths, 'names': saved_names})


    @app.route('/api/validate_local_paths', methods=['POST'])
    def api_validate_local_paths():
        """Validate pasted local video paths before starting an analysis."""
        payload = request.get_json(silent=True) or {}
        local_paths_raw = payload.get('local_paths') if isinstance(payload, dict) else ''
        valid_entries, invalid_entries, normalized_paths = _scan_local_video_paths(local_paths_raw)
        if not normalized_paths:
            return jsonify({'success': False, 'error': '请输入至少一个路径'}), 400

        valid_count = sum(int(entry.get('count') or 0) for entry in valid_entries)
        response = {
            'success': valid_count > 0,
            'normalized_paths': normalized_paths,
            'valid_count': valid_count,
            'invalid_count': len(invalid_entries),
            'valid_entries': valid_entries,
            'invalid_entries': invalid_entries,
        }
        if valid_count <= 0:
            response['error'] = '没有检测到支持的视频文件'
            return jsonify(response), 400
        return jsonify(response)


    @app.route('/do_upload', methods=['POST'])
    def do_upload():
        """处理视频上传"""
        is_ajax_upload = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        def upload_error(message, status_code=400):
            if is_ajax_upload:
                return jsonify({'success': False, 'error': message}), status_code
            flash(message, 'error')
            return redirect(url_for('upload_page'))

        session_id = secrets.token_hex(16)
        uploaded_files = request.files.getlist('videos')
        local_paths_raw = request.form.get('local_paths', '').strip()
        requested_experiment_type = request.form.get('experiment_type', 'tst')
        if is_experiment_release_hidden(requested_experiment_type):
            return upload_error('三箱社交模块当前版本暂未开放；内部测试可设置 MOUSESCOPE_ENABLE_3CST=1。', 403)
        experiment_type = normalize_experiment_type(requested_experiment_type)
        output_dir = request.form.get('output_dir', '').strip().strip('"').strip("'")
        fps_override_raw = request.form.get('fps_override', '').strip()

        try:
            start_seconds = _parse_optional_float(
                request.form.get('start_seconds', ''),
                '开始秒数',
                min_value=0,
            )
            end_seconds = _parse_optional_float(
                request.form.get('end_seconds', ''),
                '结束秒数',
                min_value=0,
            )
            _validate_time_window(start_seconds, end_seconds)
            fps_override_value = _parse_optional_float(
                fps_override_raw,
                'FPS 覆盖值',
                min_value=1,
                max_value=240,
            )
        except ValueError as exc:
            return upload_error(str(exc), 400)

        resolved_output_dir = _resolve_result_dir(output_dir) if output_dir else None

        video_paths = []
        video_infos = []
        invalid_local_paths = []

        try:
            # 处理上传的文件
            for f in uploaded_files:
                if f and f.filename:
                    save_path, original_name = _save_uploaded_video(f, session_id)
                    video_paths.append(save_path)
                    video_infos.append({'path': save_path, 'original_name': original_name})

            # 处理本地路径
            if local_paths_raw:
                for line in local_paths_raw.split('\n'):
                    p = _clean_local_path_line(line)
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
                    elif p.exists():
                        invalid_local_paths.append(f'{p}: 不支持的文件格式或文件夹内没有支持的视频文件')
                    else:
                        invalid_local_paths.append(f'{p}: 路径不存在')
        except UploadStorageError as exc:
            _cleanup_session_artifacts(session_id)
            app.logger.warning("Upload failed because storage is full: %s", exc)
            return upload_error(str(exc), 507)

        if not video_paths:
            if invalid_local_paths:
                return upload_error(invalid_local_paths[0])
            return upload_error('请选择或输入至少一个视频文件')

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
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
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
            redirect_url = url_for('batch_workspace', session_id=session_id)
        else:
            redirect_url = url_for('calibrate', session_id=session_id)
        if is_ajax_upload:
            return jsonify({'success': True, 'redirect_url': redirect_url})
        return redirect(redirect_url)


    def _render_calibrate_page(session_id, sess, video_idx=0, launch_mode='direct'):
        sess = analysis_sessions.get(session_id)
        if not sess:
            flash('会话不存在或已过期，请重新上传', 'error')
            return redirect(url_for('index'))
        if is_experiment_release_hidden(sess.get('experiment_type')):
            flash('三箱社交模块当前版本暂未开放；内部测试可设置 MOUSESCOPE_ENABLE_3CST=1。', 'warning')
            return redirect(url_for('upload_page'))

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
        project_context = sess.get('project_context') if isinstance(sess.get('project_context'), dict) else None
        project_subjects = []
        if project_context and project_store is not None:
            project_subjects = project_context.get('subject_candidates') or project_store.subject_candidates(project_context.get('project_id'))
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
            project_context=project_context,
            project_subjects=project_subjects,
        )


    @app.route('/batch/<session_id>')
    def batch_workspace(session_id):
        sess = analysis_sessions.get(session_id)
        if not sess:
            flash('会话不存在或已过期，请重新上传', 'error')
            return redirect(url_for('index'))
        if not sess.get('is_batch'):
            return redirect(url_for('calibrate', session_id=session_id))
        if is_experiment_release_hidden(sess.get('experiment_type')):
            flash('三箱社交模块当前版本暂未开放；内部测试可设置 MOUSESCOPE_ENABLE_3CST=1。', 'warning')
            return redirect(url_for('upload_page'))
        spec = get_experiment_spec(sess.get('experiment_type'))
        cards = summarize_batch_state(
            sess.get('experiment_type'),
            sess.get('draft_payload'),
            sess.get('video_infos'),
            project_context=sess.get('project_context'),
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
        if is_experiment_release_hidden(sess.get('experiment_type')):
            flash('三箱社交模块当前版本暂未开放；内部测试可设置 MOUSESCOPE_ENABLE_3CST=1。', 'warning')
            return redirect(url_for('upload_page'))
        spec = get_experiment_spec(sess.get('experiment_type'))
        return render_template(
            spec.monitor_template,
            session_id=session_id,
            maze_module=spec.maze_module,
            experiment_spec=spec,
            is_batch=sess.get('is_batch', False),
        )


    # ==================== API 路由 ====================
