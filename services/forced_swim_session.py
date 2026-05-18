from services.analysis_runtime import *


class ForcedSwimAnalysisSession(AnalysisSessionBase):
    """Multi-cylinder Forced Swim Test analysis session."""
    batch_progress_template = "Processing FST video {index}/{total}: {name}"
    batch_complete_template = "All {total} FST videos finished."

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths
        self.video_infos = video_infos
        self.config = config
        self.experiment_type = 'forced_swim'

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
            'ethogram': [],
            'activity_curve': [],
        }

    @staticmethod
    def _parse_points(points, min_points=3):
        if not isinstance(points, list) or len(points) < min_points:
            raise ValueError(f'requires at least {min_points} points')
        parsed = []
        for point in points:
            if isinstance(point, dict):
                parsed.append([float(point['x']), float(point['y'])])
            else:
                parsed.append([float(point[0]), float(point[1])])
        return parsed

    @staticmethod
    def _parse_line(points):
        if not isinstance(points, list) or len(points) != 2:
            raise ValueError('requires exactly 2 points')
        return ForcedSwimAnalysisSession._parse_points(points, min_points=2)[:2]

    def _parse_rois(self, config=None):
        active_config = config or self.config
        rois = active_config.get('rois') or []
        if isinstance(rois, str):
            rois = json.loads(rois)

        default_diameter_cm = float(active_config.get('cylinder_diameter_cm', 10.0))
        parsed = []
        for idx, roi in enumerate(rois):
            if not isinstance(roi, dict):
                raise ValueError(f'FST ROI {idx + 1} must be an object.')
            points = self._parse_points(roi.get('points') or [], min_points=3)
            diameter_points = self._parse_line(roi.get('diameter_points') or [])
            detection_points = None
            if roi.get('detection_points'):
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)
            waterline_points = None
            if roi.get('waterline_points'):
                waterline_points = self._parse_line(roi.get('waterline_points'))
            roi_name = str(roi.get('name') or f'Cylinder {idx + 1}').strip() or f'Cylinder {idx + 1}'
            time_config = roi.get('time_config') or {'mode': 'global', 'start': None, 'end': None}
            parsed.append({
                'roi_id': idx + 1,
                'name': roi_name,
                'points': points,
                'diameter_points': diameter_points,
                'detection_points': detection_points,
                'waterline_points': waterline_points,
                'cylinder_diameter_cm': float(roi.get('cylinder_diameter_cm') or default_diameter_cm),
                'time_config': time_config,
            })

        if not parsed:
            raise ValueError('Forced swim analysis requires at least one cylinder ROI.')
        return parsed

    @staticmethod
    def _build_live_metrics(rows):
        if not rows:
            return {}
        valid_percent = [float(row.get('immobility_percent') or 0.0) for row in rows]
        return {
            'roi_count': len(rows),
            'detected_roi_count': sum(1 for row in rows if row.get('detected')),
            'total_immobility_s': round(sum(float(row.get('immobility_time_s', 0.0) or 0.0) for row in rows), 3),
            'mean_immobility_percent': round(sum(valid_percent) / max(len(valid_percent), 1), 3),
        }

    def _build_processors(self, video_path, vision_config, sampling=None):
        if sampling is None:
            sampling = {
                'analysis_width': 0,
                'analysis_height': 0,
                'scale_x': 1.0,
                'scale_y': 1.0,
            }
        return self._build_processors_for_sampling(
            video_path,
            vision_config,
            sampling,
            static_frames=15,
            erosion_default=1,
            dilation_default=2,
            min_area_default=80,
        )

    def _set_video_phase_progress(self, batch_index, phase_fraction, message):
        self._set_video_progress(batch_index, phase_fraction, message)

    def _open_video_writer(self, output_video_path, fps, width, height, label='FST video', source_video_path=None):
        return create_annotated_video_writer(
            output_video_path,
            fps,
            width,
            height,
            label=label,
            source_video_path=source_video_path,
            logger=self._log,
            async_write=True,
        )

    def _render_final_annotated_video(
        self,
        video_path,
        output_video_path,
        roi_configs,
        vision_config,
        fps,
        width,
        height,
        start_frame,
        end_frame,
        frames_to_process,
        source_trackers,
        global_start_time_s,
        sampling,
        timebase,
        batch_index=None,
    ):
        self._log("Rendering FST annotated video with final scored states...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to reopen video for FST annotated rendering: {video_path}")
            return None

        writer = self._open_video_writer(output_video_path, fps, width, height, source_video_path=video_path)
        if writer is None:
            cap.release()
            self._log("Unable to create FST annotated video writer.")
            return None

        render_trackers = self._build_trackers(roi_configs, vision_config, fps, global_start_time_s)
        for render_tracker, source_tracker in zip(render_trackers, source_trackers):
            render_tracker.immobility_threshold = source_tracker.immobility_threshold
            render_tracker.immobility_high_threshold = source_tracker.immobility_high_threshold
            render_tracker.auto_threshold = source_tracker.auto_threshold

        bg_subtractor, morph_processor, detector = self._build_processors(video_path, vision_config, sampling)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        render_idx = 0
        source_frames_done = 0
        previous_analyzed_frame = None
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + source_frames_done
            if current_frame_no > end_frame:
                break

            current_time = timebase.time_at(current_frame_no)
            current_dt = timebase.duration_between(
                previous_analyzed_frame,
                current_frame_no,
                sample_fps=fps,
            )
            analysis_frame = self._resize_for_sampling(frame, sampling)
            gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
            begin_frame_cache(bg_subtractor, current_frame_no)
            results = []
            for render_tracker, source_tracker in zip(render_trackers, source_trackers):
                result = render_tracker.update(
                    analysis_frame,
                    current_time,
                    gray,
                    bg_subtractor,
                    morph_processor,
                    detector,
                    record=False,
                    frame_index=current_frame_no,
                    dt_s=current_dt,
                )
                final_row = source_tracker.frame_records[render_idx] if render_idx < len(source_tracker.frame_records) else None
                if final_row:
                    display_score = final_row.get('smoothed_motion_score')
                    if display_score is None:
                        display_score = final_row.get('motion_score')
                    result.update({
                        'state': final_row.get('state', result.get('state')),
                        'motion_score': display_score,
                        'threshold_low': final_row.get('threshold_low', render_tracker.low_threshold),
                        'threshold_high': final_row.get('threshold_high', render_tracker.high_threshold),
                        'in_stats_window': final_row.get('in_stats_window', result.get('in_stats_window')),
                        'detected': final_row.get('detected', result.get('detected')),
                    })
                    if result.get('centroid') is None and final_row.get('centroid_x') is not None and final_row.get('centroid_y') is not None:
                        result['centroid'] = (float(final_row['centroid_x']), float(final_row['centroid_y']))
                results.append(result)
            previous_analyzed_frame = current_frame_no

            result_frame = draw_forced_swim_results(analysis_frame, list(zip(render_trackers, results)))
            writer.write(result_frame)
            if render_idx % 10 == 0:
                self.current_frame = result_frame
            render_idx += 1
            skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
            covered = 1 + skipped
            source_frames_done += covered
            self._advance_frame_counters(covered, processed=False)

            if render_idx % 30 == 0:
                fraction = 0.78 + min(source_frames_done / max(frames_to_process, 1), 1.0) * 0.20
                self._set_video_phase_progress(
                    batch_index,
                    fraction,
                    f'Rendering FST video {source_frames_done}/{frames_to_process}',
                )

        cap.release()
        writer.release()
        if self.is_running and os.path.exists(output_video_path):
            self._set_video_phase_progress(batch_index, 0.98, 'FST annotated video rendered.')
            return output_video_path
        return None

    def _build_trackers(self, roi_configs, vision_config, fps, global_start_time_s):
        trackers = []
        for roi in roi_configs:
            trackers.append(
                ForcedSwimTracker(
                    roi_points=roi['points'],
                    diameter_points=roi['diameter_points'],
                    cylinder_diameter_cm=roi['cylinder_diameter_cm'],
                    fps=fps,
                    roi_id=roi['roi_id'],
                    roi_name=roi['name'],
                    time_config=roi.get('time_config'),
                    global_start_time_s=global_start_time_s,
                    detection_points=roi.get('detection_points'),
                    waterline_points=roi.get('waterline_points'),
                    auto_threshold=bool(vision_config.get('auto_threshold', True)),
                    immobility_threshold=vision_config.get('immobility_threshold'),
                    immobility_high_threshold=vision_config.get('immobility_high_threshold'),
                    threshold_sensitivity=float(vision_config.get('threshold_sensitivity', 0.0)),
                    smoothing_window_s=float(vision_config.get('smoothing_window_s', 0.5)),
                    min_immobility_duration_s=float(vision_config.get('min_immobility_duration_s', 1.0)),
                    min_active_duration_s=float(vision_config.get('min_active_duration_s', 0.3)),
                    gap_merge_s=float(vision_config.get('gap_merge_s', 0.5)),
                )
            )
        return trackers

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
        result_folder = vision_config.get('output_dir') or str(RESULT_DIR)
        output_info = _prepare_output_paths(
            self.session_id,
            video_info.get('original_name', 'video'),
            suffix='fst',
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
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps_override = vision_config.get('fps_override')
        video_meta = probe_video_info(video_path, fps_override=fps_override, cap=cap)
        detected_fps = float(video_meta.get('detected_fps') or 30.0)
        fps = float(video_meta.get('fps') or detected_fps or 30.0)
        width = int(video_meta.get('width') or 0)
        height = int(video_meta.get('height') or 0)
        total_frames = int(video_meta.get('total_frames') or 0)
        self._log(f"FST video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")
        timebase, timebase_summary, total_frames = self._build_timebase_for_video(
            video_path,
            fps,
            total_frames,
            label='FST',
        )
        sampling = self._resolve_video_sampling(
            width,
            height,
            fps,
            vision_config,
            label='FST',
            temporal_default=False,
            max_dim_default=960,
            target_fps_default=30.0,
        )
        self._log_video_sampling(sampling)

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        start_frame, end_frame = self._time_range_to_frames(timebase, total_frames, start_sec, end_sec)
        frames_to_process = max(0, end_frame - start_frame + 1)

        source_roi_configs = self._parse_rois(vision_config)
        roi_configs = self._scale_roi_configs_for_sampling(source_roi_configs, sampling)
        trackers = self._build_trackers(roi_configs, vision_config, sampling['effective_fps'], global_start_time_s)
        self._log(f"Configured {len(trackers)} FST cylinders")

        needs_auto = any(tracker.auto_threshold and tracker.immobility_threshold is None for tracker in trackers)
        if needs_auto and self.is_running:
            self._log("Estimating FST immobility thresholds from this video...")
            self._set_video_phase_progress(batch_index, 0.0, 'Estimating FST immobility thresholds...')
            bg_subtractor, morph_processor, detector = self._build_processors(video_path, vision_config, sampling)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            prepass_idx = 0
            prepass_source_frames_done = 0
            previous_analyzed_frame = None
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                except Exception:
                    current_frame_no = start_frame + prepass_source_frames_done
                if current_frame_no > end_frame:
                    break
                current_time = timebase.time_at(current_frame_no)
                current_dt = timebase.duration_between(
                    previous_analyzed_frame,
                    current_frame_no,
                    sample_fps=sampling['effective_fps'],
                )
                analysis_frame = self._resize_for_sampling(frame, sampling)
                gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
                begin_frame_cache(bg_subtractor, current_frame_no)
                for tracker in trackers:
                    tracker.update(
                        analysis_frame,
                        current_time,
                        gray,
                        bg_subtractor,
                        morph_processor,
                        detector,
                        record=False,
                        frame_index=current_frame_no,
                        dt_s=current_dt,
                )
                previous_analyzed_frame = current_frame_no
                prepass_idx += 1
                skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
                covered = 1 + skipped
                prepass_source_frames_done += covered
                self._advance_frame_counters(covered, processed=False)
                if prepass_idx % 90 == 0:
                    fraction = min(prepass_source_frames_done / max(frames_to_process, 1), 1.0) * 0.18
                    self._set_video_phase_progress(
                        batch_index,
                        fraction,
                        f'FST threshold estimation {prepass_source_frames_done}/{frames_to_process}',
                    )

            for tracker in trackers:
                tracker.auto_configure_thresholds(fallback_threshold=float(vision_config.get('immobility_threshold') or 0.08))
                self._log(
                    f"ROI {tracker.roi_id} ({tracker.roi_name}) threshold: "
                    f"low={tracker.low_threshold:.4f}, high={tracker.high_threshold:.4f}"
                )
                tracker.reset_runtime(clear_threshold=False)
            self._set_video_phase_progress(batch_index, 0.18, 'FST immobility thresholds estimated.')

        bg_subtractor, morph_processor, detector = self._build_processors(video_path, vision_config, sampling)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = 0
        source_frames_done = 0
        previous_analyzed_frame = None
        analysis_start_fraction = 0.18 if needs_auto else 0.0
        analysis_end_fraction = 0.78 if generate_video else 0.94
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + source_frames_done
            if current_frame_no > end_frame:
                break

            current_time = timebase.time_at(current_frame_no)
            current_dt = timebase.duration_between(
                previous_analyzed_frame,
                current_frame_no,
                sample_fps=sampling['effective_fps'],
            )
            analysis_frame = self._resize_for_sampling(frame, sampling)
            gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
            begin_frame_cache(bg_subtractor, current_frame_no)
            results = [
                tracker.update(
                    analysis_frame,
                    current_time,
                    gray,
                    bg_subtractor,
                    morph_processor,
                    detector,
                    record=True,
                    frame_index=current_frame_no,
                    dt_s=current_dt,
                )
                for tracker in trackers
            ]
            previous_analyzed_frame = current_frame_no

            if frame_idx % 15 == 0:
                rows = []
                for tracker, result in zip(trackers, results):
                    summary = tracker.build_live_summary()
                    rows.append({
                        'roi_id': tracker.roi_id,
                        'roi_name': tracker.roi_name,
                        'detected': bool(result.get('detected')),
                        'state': result.get('state', 'uncertain'),
                        'motion_score': result.get('motion_score'),
                        'immobility_time_s': summary.get('immobility_time_s'),
                        'active_time_s': summary.get('active_time_s'),
                        'immobility_percent': summary.get('immobility_percent'),
                        'immobility_bouts': summary.get('immobility_bouts'),
                        'latency_to_immobility_s': summary.get('latency_to_immobility_s'),
                        'threshold_low': tracker.low_threshold,
                        'threshold_high': tracker.high_threshold,
                        'in_stats_window': bool(result.get('in_stats_window', True)),
                    })
                self.realtime_data = rows
                self.live_metrics = self._build_live_metrics(rows)

            frame_idx += 1
            skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
            covered = 1 + skipped
            source_frames_done += covered
            self._advance_frame_counters(covered)
            if frame_idx % 20 == 0:
                local = min(source_frames_done / max(frames_to_process, 1), 1.0)
                fraction = analysis_start_fraction + local * (analysis_end_fraction - analysis_start_fraction)
                self._set_video_phase_progress(batch_index, fraction, f'FST scoring {source_frames_done}/{frames_to_process}')

        cap.release()
        if not self.is_running:
            return
        self._ensure_analysis_frames_processed(
            video_path=video_path,
            label='FST',
            frame_idx=frame_idx,
            source_frames_done=source_frames_done,
        )

        for tracker in trackers:
            tracker.finalize_states()

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
            self._add_sampling_summary_fields(summary, sampling)
            self._add_timebase_summary_fields(summary, timebase_summary)
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
                self._append_result_file(_export_result_file(track_csv_work_path, f"{final_base}_track.csv"), 'track_csv')

        if HAS_PANDAS and summary_rows:
            summary_csv_work_path = f"{output_base}_summary.csv"
            if _write_dataframe_csv(summary_rows, summary_csv_work_path):
                self._append_result_file(_export_result_file(summary_csv_work_path, f"{final_base}_summary.csv"), 'summary_csv')

        if HAS_PANDAS and bins_rows:
            bins_csv_work_path = f"{output_base}_bins.csv"
            if _write_dataframe_csv(bins_rows, bins_csv_work_path):
                self._append_result_file(_export_result_file(bins_csv_work_path, f"{final_base}_bins.csv"), 'bins_csv')

        report_txt_work_path = f"{output_base}_report.txt"
        self._generate_txt_report(summary_rows, report_txt_work_path)
        self._append_result_file(_export_result_file(report_txt_work_path, f"{final_base}_report.txt"), 'report')

        ethogram_work_path = f"{output_base}_ethogram.png"
        if render_forced_swim_ethogram(trackers, ethogram_work_path):
            self._append_result_file(_export_result_file(ethogram_work_path, f"{final_base}_ethogram.png"), 'ethogram')
        else:
            self._log(f"Failed to save FST ethogram: {final_base}_ethogram.png")

        activity_work_path = f"{output_base}_activity_curve.png"
        if render_forced_swim_activity_curve(trackers, activity_work_path):
            self._append_result_file(_export_result_file(activity_work_path, f"{final_base}_activity_curve.png"), 'activity_curve')
        else:
            self._log(f"Failed to save FST activity curve: {final_base}_activity_curve.png")

        if generate_video and self.is_running:
            rendered_video = self._render_final_annotated_video(
                video_path,
                output_video_path,
                roi_configs,
                vision_config,
                sampling['effective_fps'],
                sampling['analysis_width'],
                sampling['analysis_height'],
                start_frame,
                end_frame,
                frames_to_process,
                trackers,
                global_start_time_s,
                sampling,
                timebase,
                batch_index=batch_index,
            )
            if rendered_video and os.path.exists(rendered_video):
                exported_video = _export_result_file(rendered_video, final_output_video_path)
                self._append_result_file(exported_video, 'video')
                if exported_video:
                    self._log(f"FST annotated video saved: {exported_video}")

        self.realtime_data = [
            {
                'roi_id': row['roi_id'],
                'roi_name': row['roi_name'],
                'detected': True,
                'state': 'completed',
                'motion_score': row.get('mean_motion_score'),
                'immobility_time_s': row.get('immobility_time_s'),
                'active_time_s': row.get('active_time_s'),
                'immobility_percent': row.get('immobility_percent'),
                'immobility_bouts': row.get('immobility_bouts'),
                'latency_to_immobility_s': row.get('latency_to_immobility_s'),
                'threshold_low': row.get('immobility_threshold'),
                'threshold_high': row.get('active_threshold'),
                'in_stats_window': True,
            }
            for row in summary_rows
        ]
        self.live_metrics = self._build_live_metrics(self.realtime_data)

        if batch_index is None and self.is_running:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = 'Forced swim analysis completed.'

        elapsed = time.time() - video_start_time
        self._log(f"FST video processed in {elapsed:.1f}s, {source_frames_done} source frames ({frame_idx} analyzed samples)")

        import gc
        gc.collect()

    def _generate_txt_report(self, summaries, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 72 + "\n")
                f.write("MouseScope 强迫游泳实验报告 (Forced Swim Test Report)\n")
                f.write("=" * 72 + "\n\n")
                f.write(f"{_bi_label('分析时间', 'Analysis Time')}: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if summaries:
                    f.write(f"{_bi_label('视频', 'Video')}: {summaries[0].get('video_name', '')}\n")
                f.write(f"{_bi_label('圆筒数量', 'Cylinder Count')}: {len(summaries)}\n")
                f.write(
                    f"{_bi_label('统计规则', 'Scoring Rule')}: "
                    "FST 默认使用 active / immobile 二分类；运动评分由光流、质心速度与轮廓形变组成，并按圆筒直径归一化。 "
                    "(FST uses active / immobile scoring; motion is normalized by cylinder diameter.)\n\n"
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
                        ('不动时间', 'Immobility Time', 'immobility_time_s', ' s'),
                        ('活动时间', 'Active Time', 'active_time_s', ' s'),
                        ('不确定时间', 'Uncertain Time', 'uncertain_time_s', ' s'),
                        ('不动比例', 'Immobility Percent', 'immobility_percent', '%'),
                        ('检测帧内不动比例', 'Immobility Percent Detected', 'immobility_percent_detected', '%'),
                        ('不动片段数', 'Immobility Bouts', 'immobility_bouts', ''),
                        ('最长不动片段', 'Longest Immobility Bout', 'longest_immobility_bout_s', ' s'),
                        ('首次不动潜伏期', 'Latency to First Immobility', 'latency_to_immobility_s', ' s'),
                        ('平均活动评分', 'Mean Motion Score', 'mean_motion_score', ''),
                        ('不动阈值', 'Immobility Threshold', 'immobility_threshold', ''),
                        ('圆筒直径', 'Cylinder Diameter', 'cylinder_diameter_cm', ' cm'),
                    ])
                    f.write("\n")
        except Exception as e:
            print(f"Failed to write FST report: {e}")


ForcedSwimMultiCylinderAnalysisSession = ForcedSwimAnalysisSession
