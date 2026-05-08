from __future__ import annotations

import secrets
import os
from pathlib import Path

from flask import flash, jsonify, redirect, render_template, request, session, url_for

from services.experiment_registry import get_experiment_spec, normalize_experiment_type, summarize_batch_state


def register_page_routes(app, ctx) -> None:
    UploadStorageError = ctx.UploadStorageError
    analysis_sessions = ctx.analysis_sessions
    BUNDLE_DIR = ctx.BUNDLE_DIR
    VALID_VIDEO_EXTS = ctx.VALID_VIDEO_EXTS
    _cleanup_session_artifacts = ctx._cleanup_session_artifacts
    _extract_video_info_from_entry = ctx._extract_video_info_from_entry
    _get_original_name_for_path = ctx._get_original_name_for_path
    _now_ts = ctx._now_ts
    _resolve_result_dir = ctx._resolve_result_dir
    _safe_internal_return_to = ctx._safe_internal_return_to
    _save_uploaded_video = ctx._save_uploaded_video
    _session_video_info = ctx._session_video_info
    _stage_local_video_if_needed = ctx._stage_local_video_if_needed
    _update_video_entry_info = ctx._update_video_entry_info

    @app.route('/')
    def index():
        """首页"""
        return render_template('home.html')


    @app.route('/upload')
    def upload_page():
        """上传视频页面"""
        return render_template('upload.html')


    @app.route('/tasks')
    def tasks_page():
        """Task center."""
        return render_template('tasks.html')


    @app.route('/manual')
    def manual():
        """使用手册"""
        manual_type = request.args.get('module', 'tst').strip().lower()
        if manual_type not in {'tst', 'fst', 'oft', 'nor', 'epm', 'ymaze'}:
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


    @app.route('/do_upload', methods=['POST'])
    def do_upload():
        """处理视频上传"""
        session_id = secrets.token_hex(16)
        uploaded_files = request.files.getlist('videos')
        local_paths_raw = request.form.get('local_paths', '').strip()
        experiment_type = normalize_experiment_type(request.form.get('experiment_type', 'tst'))

        video_paths = []
        video_infos = []

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
        except UploadStorageError as exc:
            _cleanup_session_artifacts(session_id)
            app.logger.warning("Upload failed because storage is full: %s", exc)
            flash(str(exc), 'error')
            return redirect(url_for('upload_page'))

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
