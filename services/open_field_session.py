from services.analysis_runtime import *
from services.pose_tracking import PoseInferenceCoordinator


class OpenFieldMultiArenaAnalysisSession(AnalysisSessionBase):
    """Multi-arena Open Field Test analysis session."""
    batch_progress_template = "Processing OFT video {index}/{total}: {name}"
    batch_complete_template = "All {total} OFT videos finished."

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

    @staticmethod
    def _live_row(tracker, result):
        total_time_s = getattr(tracker, 'time_in_window_s', 0.0) or (tracker.frames_in_window / tracker.fps if tracker.fps > 0 else 0.0)
        center_time_s = getattr(tracker, 'center_time_s', 0.0) or (tracker.center_frames / tracker.fps if tracker.fps > 0 else 0.0)
        wall_time_s = getattr(tracker, 'wall_time_s', 0.0) or (tracker.wall_frames / tracker.fps if tracker.fps > 0 else 0.0)
        return {
            'roi_id': tracker.roi_id,
            'roi_name': tracker.roi_name,
            'detected': bool(result.get('detected')),
            'zone': result.get('zone_label', 'Unknown'),
            'speed_cm_s': round(float(result.get('speed_cm_s', 0.0)), 3),
            'distance_cm': round(float(tracker.total_distance_cm), 3),
            'center_percent': round(center_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            'center_entries': int(tracker.center_entries),
            'wall_time_s': round(wall_time_s, 3),
            'in_stats_window': bool(result.get('in_stats_window', True)),
        }

    def _create_trackers(self, roi_configs, vision_config, fps, global_start_time_s):
        arena_side_cm = float(vision_config.get('arena_side_cm', 50.0))
        return [
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
            for roi in roi_configs
        ]

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
        if vision_config.get('tracking_backend') != vision_config.get('tracking_backend_effective', 'classical'):
            self._log(
                "OFT tracking backend fallback: "
                f"{vision_config.get('tracking_backend')} -> classical "
                f"({vision_config.get('tracking_fallback_reason') or 'unavailable'})."
            )
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

        cap = open_video_capture(video_path, orientation_override=vision_config.get('orientation_override'))
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
        self._log(f"OFT video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")
        timebase, timebase_summary, total_frames = self._build_timebase_for_video(
            video_path,
            fps,
            total_frames,
            label='OFT',
            prefer_pts=not bool(fps_override),
        )

        sampling = self._resolve_video_sampling(
            width,
            height,
            fps,
            vision_config,
            label='OFT',
            temporal_default=True,
            max_dim_default=960,
            target_fps_default=30.0,
        )
        self._log_video_sampling(sampling)

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        start_frame, end_frame = self._time_range_to_frames(timebase, total_frames, start_sec, end_sec)
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0

        bg_subtractor, morph_processor, detector = self._build_processors_for_sampling(
            video_path,
            vision_config,
            sampling,
            static_frames=15,
            erosion_default=1,
            dilation_default=2,
            min_area_default=120,
            temporal_start_frame=start_frame,
            temporal_end_frame=end_frame,
        )

        source_roi_configs = self._parse_rois(vision_config)
        roi_configs = self._scale_roi_configs_for_sampling(source_roi_configs, sampling)

        trackers = self._create_trackers(roi_configs, vision_config, sampling['effective_fps'], global_start_time_s)
        pose_coordinator = PoseInferenceCoordinator(vision_config, trackers)
        if pose_coordinator.enabled:
            self._log("OFT DeepLabCut tracking enabled: one inference per sampled frame; ROI-level classical fallback remains active.")
        self._log(f"Configured {len(trackers)} OFT arenas")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start OFT analysis from frame {start_frame} ({timebase.time_at(start_frame):.1f}s)")

        frames_to_process = max(0, end_frame - start_frame + 1)
        frame_idx = 0
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
                sample_fps=sampling['effective_fps'],
            )
            analysis_frame = self._resize_for_sampling(frame, sampling)
            gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
            begin_frame_cache(bg_subtractor, current_frame_no)
            pose_frame = pose_coordinator.infer(analysis_frame)
            fallback_active = (
                str(vision_config.get('tracking_backend') or 'classical') != 'classical'
            )
            results = []
            for tracker in trackers:
                pose_observation = pose_frame.by_roi_id.get(int(tracker.roi_id))
                fallback_reason = (
                    pose_frame.error
                    or vision_config.get('tracking_fallback_reason')
                    or ('pose_low_confidence_or_unassigned' if fallback_active and pose_observation is None else None)
                )
                results.append(
                    tracker.update(
                        analysis_frame,
                        current_time,
                        gray,
                        bg_subtractor,
                        morph_processor,
                        detector,
                        dt_s=current_dt,
                        pose_observation=pose_observation,
                        tracking_source='classical_fallback' if fallback_active else 'classical',
                        fallback_reason=fallback_reason if pose_observation is None else None,
                    )
                )
            previous_analyzed_frame = current_frame_no
            for result in results:
                self._annotate_sample_record(result, current_frame_no, frame_idx, sampling)

            if frame_idx % 15 == 0:
                rows = [self._live_row(tracker, result) for tracker, result in zip(trackers, results)]
                self.realtime_data = rows
                self.live_metrics = self._build_live_metrics(rows)

            if frame_idx % 30 == 0:
                self.current_frame = analysis_frame

            frame_idx += 1
            skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
            covered = 1 + skipped
            source_frames_done += covered
            self._advance_frame_counters(covered)

            if frame_idx % 20 == 0:
                analysis_fraction = source_frames_done / max(frames_to_process, 1)
                analysis_phase = 0.82 if generate_video else 0.94
                self._set_video_progress(
                    batch_index,
                    analysis_fraction * analysis_phase,
                    f'Frame {source_frames_done}/{frames_to_process} ({{progress}}%)',
                )

        pose_coordinator.close()
        cap.release()
        if not self.is_running:
            return
        self._ensure_analysis_frames_processed(
            video_path=video_path,
            label='OFT',
            frame_idx=frame_idx,
            source_frames_done=source_frames_done,
        )

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
            summary['tracking_backend_requested'] = vision_config.get('tracking_backend', 'classical')
            summary['tracking_backend_effective'] = vision_config.get('tracking_backend_effective', 'classical')
            summary['tracking_fallback_reason'] = vision_config.get('tracking_fallback_reason')
            dlc_frames = int(summary.get('deeplabcut_frames') or 0)
            fallback_frames = int(summary.get('classical_fallback_frames') or 0)
            summary['tracking_backend_observed'] = (
                'mixed' if dlc_frames and fallback_frames
                else ('deeplabcut' if dlc_frames else 'classical')
            )
            self._add_sampling_summary_fields(summary, sampling)
            self._add_timebase_summary_fields(summary, timebase_summary)
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
        visual_style = vision_config.get('visual_style')
        for tracker in trackers:
            roi_slug = sanitize_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"arena_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"

            if render_open_field_heatmap(tracker, heatmap_work_path, style=visual_style):
                exported_heatmap = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported_heatmap)
                self._append_result_file(exported_heatmap)
            else:
                self._log(f"Failed to save OFT heatmap: {heatmap_final_path}")

            if render_open_field_trackmap(tracker, trackmap_work_path, style=visual_style):
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
                if render_open_field_image_grid(heatmap_work_paths, heatmap_grid_work_path, title=f"{video_name} | Heatmaps", style=visual_style):
                    self._append_result_file(_export_result_file(heatmap_grid_work_path, heatmap_grid_final_path), 'heatmap')
                else:
                    self._log(f"Failed to save OFT heatmap grid: {heatmap_grid_final_path}")

        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                trackmap_grid_work_path = f"{output_base}_trackmaps.png"
                trackmap_grid_final_path = f"{final_base}_trackmaps.png"
                if render_open_field_image_grid(trackmap_work_paths, trackmap_grid_work_path, title=f"{video_name} | Trajectory Maps", style=visual_style):
                    self._append_result_file(_export_result_file(trackmap_grid_work_path, trackmap_grid_final_path), 'trackmap')
                else:
                    self._log(f"Failed to save OFT trackmap grid: {trackmap_grid_final_path}")

        if generate_video and self.is_running:
            self._render_annotated_video(
                video_path,
                output_video_path,
                vision_config,
                roi_configs,
                sampling['effective_fps'],
                sampling['analysis_width'],
                sampling['analysis_height'],
                start_frame,
                end_frame,
                frames_to_process,
                global_start_time_s,
                sampling,
                timebase,
                batch_index=batch_index,
            )

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
        self._log(f"OFT video processed in {elapsed:.1f}s, {source_frames_done} source frames ({frame_idx} analyzed samples)")

        import gc
        gc.collect()

    def _render_annotated_video(
        self,
        video_path,
        output_video_path,
        vision_config,
        roi_configs,
        fps,
        width,
        height,
        start_frame,
        end_frame,
        frames_to_process,
        global_start_time_s,
        sampling,
        timebase,
        batch_index=None,
    ):
        cap = open_video_capture(video_path, orientation_override=vision_config.get('orientation_override'))
        if not cap.isOpened():
            self._log(f"Unable to reopen OFT video for rendering: {video_path}")
            return False

        writer = create_annotated_video_writer(
            output_video_path,
            fps,
            width,
            height,
            label='OFT video',
            source_video_path=video_path,
            logger=self._log,
            async_write=True,
        )
        if writer is None:
            cap.release()
            self._log("OFT annotated video disabled because no writer is available.")
            return False

        bg_subtractor, morph_processor, detector = self._build_processors_for_sampling(
            video_path,
            vision_config,
            sampling,
            static_frames=15,
            erosion_default=1,
            dilation_default=2,
            min_area_default=120,
            temporal_start_frame=start_frame,
            temporal_end_frame=end_frame,
        )
        render_trackers = self._create_trackers(roi_configs, vision_config, fps, global_start_time_s)
        pose_coordinator = PoseInferenceCoordinator(vision_config, render_trackers)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = 0
        source_frames_done = 0
        previous_analyzed_frame = None
        try:
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
                pose_frame = pose_coordinator.infer(analysis_frame)
                fallback_active = str(vision_config.get('tracking_backend') or 'classical') != 'classical'
                results = []
                for tracker in render_trackers:
                    pose_observation = pose_frame.by_roi_id.get(int(tracker.roi_id))
                    results.append(
                        tracker.update(
                            analysis_frame,
                            current_time,
                            gray,
                            bg_subtractor,
                            morph_processor,
                            detector,
                            dt_s=current_dt,
                            pose_observation=pose_observation,
                            tracking_source='classical_fallback' if fallback_active else 'classical',
                            fallback_reason=(
                                pose_frame.error
                                or vision_config.get('tracking_fallback_reason')
                                or ('pose_low_confidence_or_unassigned' if fallback_active and pose_observation is None else None)
                            ) if pose_observation is None else None,
                        )
                    )
                previous_analyzed_frame = current_frame_no
                result_frame = draw_open_field_results(analysis_frame, list(zip(render_trackers, results)))
                writer.write(result_frame)
                if frame_idx % 30 == 0:
                    self.current_frame = result_frame
                frame_idx += 1
                skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
                covered = 1 + skipped
                source_frames_done += covered
                self._advance_frame_counters(covered, processed=False)
                if frame_idx % 60 == 0:
                    self._set_video_progress(
                        batch_index,
                        0.82 + 0.16 * source_frames_done / max(frames_to_process, 1),
                        f'Rendering OFT video {source_frames_done}/{frames_to_process} ({{progress}}%)',
                    )
        finally:
            pose_coordinator.close()
            cap.release()
            writer.release()
        return os.path.exists(output_video_path)

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
                        ('中心区距离', 'Center Distance', 'center_distance_cm', ' cm'),
                        ('外周区距离', 'Outer Distance', 'outer_distance_cm', ' cm'),
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


# ==================== Forced Swim ====================
