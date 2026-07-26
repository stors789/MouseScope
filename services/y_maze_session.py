from services.analysis_runtime import *
from y_maze import build_manual_y_maze_summary, normalize_manual_arm_entries


class YMazeMultiArenaAnalysisSession(AnalysisSessionBase):
    """Multi-ROI Y-maze spontaneous alternation analysis session."""
    batch_progress_template = "Processing Y-maze video {index}/{total}: {name}"
    batch_complete_template = "All {total} Y-maze videos finished."

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
            'events_csv': [],
        }

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
            scoring_mode = str(roi.get('scoring_mode') or 'automatic').strip().lower()
            if scoring_mode not in {'automatic', 'manual'}:
                raise ValueError(f'Y-maze ROI {idx + 1} has an invalid scoring mode.')
            manual_events = copy.deepcopy(roi.get('manual_events') or [])
            if scoring_mode == 'manual' and not manual_events:
                raise ValueError(f'Y-maze manual ROI {idx + 1} has no arm-entry events.')
            if scoring_mode == 'manual':
                normalize_manual_arm_entries(manual_events)
            parsed.append({
                'roi_id': idx + 1,
                'name': roi_name,
                'zones': zone_polygons,
                'time_config': time_config,
                'detection_points': detection_points,
                'scoring_mode': scoring_mode,
                'manual_events': manual_events,
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

    @staticmethod
    def _live_row(tracker, result):
        opportunities = max(tracker.total_arm_entries - 2, 0)
        alternation_percent = None
        if opportunities > 0:
            alternation_percent = round(tracker.alternation_count / opportunities * 100.0, 2)
        return {
            'roi_id': tracker.roi_id,
            'roi_name': tracker.roi_name,
            'detected': bool(result.get('detected')),
            'zone': result.get('zone_label', 'Unknown'),
            'speed_cm_s': round(float(result.get('speed_cm_s', 0.0)), 3),
            'distance_cm': round(float(tracker.total_distance_cm), 3),
            'total_arm_entries': int(tracker.total_arm_entries),
            'alternation_count': int(tracker.alternation_count),
            'alternation_percent': alternation_percent,
            'in_stats_window': bool(result.get('in_stats_window', True)),
        }

    def _create_trackers(self, roi_configs, vision_config, fps, global_start_time_s):
        return [
            YMazeTracker(
                zone_polygons=roi['zones'],
                detection_polygon=roi.get('detection_points'),
                arm_length_cm=float(vision_config.get('arm_length_cm', 30.0)),
                arm_width_cm=float(vision_config.get('arm_width_cm', 10.0)),
                fps=fps,
                movement_threshold_cm_s=float(vision_config.get('movement_threshold_cm_s', 2.0)),
                entry_depth_ratio=float(vision_config.get('entry_depth_ratio', 0.2)),
                tracking_max_jump_cm=float(vision_config.get('tracking_max_jump_cm', 12.0)),
                occlusion_hold_s=float(vision_config.get('occlusion_hold_s', 0.4)),
                zone_confirm_s=float(vision_config.get('zone_confirm_s', 0.12)),
                center_reset_s=float(vision_config.get('center_reset_s', 0.2)),
                require_center_passage=bool(vision_config.get('require_center_passage', True)),
                roi_id=roi['roi_id'],
                roi_name=roi['name'],
                time_config=roi.get('time_config'),
                global_start_time_s=global_start_time_s,
            )
            for roi in roi_configs
            if roi.get('scoring_mode', 'automatic') == 'automatic'
        ]

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
        self._log(f"Y-maze video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")
        timebase, timebase_summary, total_frames = self._build_timebase_for_video(
            video_path,
            fps,
            total_frames,
            label='Y-maze',
            prefer_pts=not bool(fps_override),
        )

        sampling = self._resolve_video_sampling(
            width,
            height,
            fps,
            vision_config,
            label='Y-maze',
            temporal_default=True,
            max_dim_default=960,
            target_fps_default=30.0,
        )
        self._log_video_sampling(sampling)

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        start_frame, end_frame = self._time_range_to_frames(timebase, total_frames, start_sec, end_sec)
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        source_roi_configs = self._parse_rois(vision_config)
        roi_configs = self._scale_roi_configs_for_sampling(source_roi_configs, sampling)

        trackers = self._create_trackers(roi_configs, vision_config, sampling['effective_fps'], global_start_time_s)
        bg_subtractor = morph_processor = detector = None
        if trackers:
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
        self._log(
            f"Configured {len(trackers)} automatic and "
            f"{sum(roi.get('scoring_mode') == 'manual' for roi in roi_configs)} manual Y-maze arenas"
        )

        annotated_writer = None
        if generate_video:
            annotated_writer = create_annotated_video_writer(
                output_video_path,
                sampling['effective_fps'],
                sampling['analysis_width'],
                sampling['analysis_height'],
                label='Y-maze video',
                source_video_path=video_path,
                logger=self._log,
                async_write=True,
            )
            if annotated_writer is None:
                self._log("Y-maze annotated video disabled because no writer is available.")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start Y-maze analysis from frame {start_frame} ({timebase.time_at(start_frame):.1f}s)")

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
            if bg_subtractor is not None:
                begin_frame_cache(bg_subtractor, current_frame_no)
            results = [
                tracker.update(analysis_frame, current_time, gray, bg_subtractor, morph_processor, detector, dt_s=current_dt)
                for tracker in trackers
            ]
            previous_analyzed_frame = current_frame_no
            for result in results:
                self._annotate_sample_record(result, current_frame_no, frame_idx, sampling)

            if annotated_writer is not None:
                result_frame = draw_y_maze_results(analysis_frame, list(zip(trackers, results)))
                self._draw_manual_overlays(result_frame, roi_configs, current_time)
                annotated_writer.write(result_frame)
            else:
                result_frame = analysis_frame

            if frame_idx % 15 == 0:
                rows = [self._live_row(tracker, result) for tracker, result in zip(trackers, results)]
                self.realtime_data = rows
                self.live_metrics = self._build_live_metrics(rows)

            if frame_idx % 30 == 0:
                self.current_frame = result_frame

            frame_idx += 1
            skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
            covered = 1 + skipped
            source_frames_done += covered
            self._advance_frame_counters(covered)

            if frame_idx % 20 == 0:
                analysis_fraction = source_frames_done / max(frames_to_process, 1)
                self._set_video_progress(
                    batch_index,
                    analysis_fraction * 0.94,
                    f'Frame {source_frames_done}/{frames_to_process} ({{progress}}%)',
                )

        cap.release()
        if annotated_writer is not None:
            annotated_writer.release()
        if not self.is_running:
            return
        self._ensure_analysis_frames_processed(
            video_path=video_path,
            label='Y-maze',
            frame_idx=frame_idx,
            source_frames_done=source_frames_done,
        )

        summary_rows = []
        track_rows = []
        bins_rows = []
        manual_event_rows = []
        tracker_by_roi = {tracker.roi_id: tracker for tracker in trackers}
        analysis_end_time_s = float(timebase.time_at(end_frame))
        for roi in roi_configs:
            tracker = tracker_by_roi.get(roi['roi_id'])
            if tracker is None:
                time_config = roi.get('time_config') or {}
                manual_start = global_start_time_s
                manual_end = analysis_end_time_s
                if time_config.get('mode') == 'precise':
                    if time_config.get('start') is not None:
                        manual_start = max(manual_start, float(time_config['start']))
                    if time_config.get('end') is not None:
                        manual_end = min(manual_end, float(time_config['end']))
                summary, event_rows = build_manual_y_maze_summary(
                    roi_id=roi['roi_id'],
                    roi_name=roi['name'],
                    events=roi.get('manual_events') or [],
                    window_start_s=manual_start,
                    window_end_s=manual_end,
                    time_mode=time_config.get('mode', 'global'),
                )
                if not event_rows:
                    raise ValueError(
                        f"Y-maze manual ROI {roi['roi_id']} has no events inside its analysis window."
                    )
                for event_row in event_rows:
                    event_row['video_name'] = video_info.get('original_name', video_name)
                    manual_event_rows.append(event_row)
            else:
                summary = tracker.build_summary()
                for row in tracker.frame_records:
                    record = dict(row)
                    record['video_name'] = video_info.get('original_name', video_name)
                    record['scoring_method'] = 'automatic'
                    track_rows.append(record)

                for row in tracker.build_time_bins(bin_seconds=float(vision_config.get('bin_seconds', 60.0))):
                    bucket = dict(row)
                    bucket['video_name'] = video_info.get('original_name', video_name)
                    bucket['scoring_method'] = 'automatic'
                    bins_rows.append(bucket)
            self._add_sampling_summary_fields(summary, sampling)
            self._add_timebase_summary_fields(summary, timebase_summary)
            summary['video_name'] = video_info.get('original_name', video_name)
            summary_rows.append(summary)

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

        if HAS_PANDAS and manual_event_rows:
            events_csv_work_path = f"{output_base}_manual_events.csv"
            if _write_dataframe_csv(manual_event_rows, events_csv_work_path):
                events_csv_path = _export_result_file(events_csv_work_path, f"{final_base}_manual_events.csv")
                self._append_result_file(events_csv_path, 'events_csv')

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
            roi_slug = sanitize_y_maze_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"ymaze_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"

            if render_y_maze_heatmap(tracker, heatmap_work_path, style=visual_style):
                exported_heatmap = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported_heatmap)
                self._append_result_file(exported_heatmap)
            elif not tracker.trajectory_template_cm_stats:
                self._log(f"Skipped Y-maze heatmap, no valid trajectory points: {heatmap_final_path}")
            else:
                self._log(f"Failed to save Y-maze heatmap: {heatmap_final_path}")

            if render_y_maze_trackmap(tracker, trackmap_work_path, style=visual_style):
                exported_trackmap = _export_result_file(trackmap_work_path, trackmap_final_path)
                trackmap_work_paths.append(trackmap_work_path)
                trackmap_final_paths.append(exported_trackmap)
                self._append_result_file(exported_trackmap)
            elif not tracker.trajectory_template_cm_stats:
                self._log(f"Skipped Y-maze trackmap, no valid trajectory points: {trackmap_final_path}")
            else:
                self._log(f"Failed to save Y-maze trackmap: {trackmap_final_path}")

        if heatmap_work_paths:
            if len(heatmap_final_paths) == 1:
                self._append_result_file(heatmap_final_paths[0], 'heatmap')
            else:
                heatmap_grid_work_path = f"{output_base}_heatmaps.png"
                heatmap_grid_final_path = f"{final_base}_heatmaps.png"
                if render_y_maze_image_grid(heatmap_work_paths, heatmap_grid_work_path, title=f"{video_name} | Y-maze Heatmaps", style=visual_style):
                    self._append_result_file(_export_result_file(heatmap_grid_work_path, heatmap_grid_final_path), 'heatmap')
                else:
                    self._log(f"Failed to save Y-maze heatmap grid: {heatmap_grid_final_path}")

        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                trackmap_grid_work_path = f"{output_base}_trackmaps.png"
                trackmap_grid_final_path = f"{final_base}_trackmaps.png"
                if render_y_maze_image_grid(trackmap_work_paths, trackmap_grid_work_path, title=f"{video_name} | Y-maze Trajectory Maps", style=visual_style):
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
        self._log(f"Y-maze video processed in {elapsed:.1f}s, {source_frames_done} source frames ({frame_idx} analyzed samples)")

        import gc
        gc.collect()

    @staticmethod
    def _draw_manual_overlays(frame, roi_configs, current_time):
        for roi in roi_configs:
            if roi.get('scoring_mode') != 'manual':
                continue
            events, _ = normalize_manual_arm_entries(roi.get('manual_events') or [])
            current_arm = None
            for event in events:
                if event['timestamp_s'] > float(current_time):
                    break
                current_arm = event['arm']
            center = np.asarray(roi['zones']['center'], dtype=np.float32).mean(axis=0)
            label = f"Manual: {YMazeTracker.ARM_LABELS.get(current_arm, '-')}"
            cv2.putText(
                frame,
                label,
                (max(4, int(center[0]) - 45), max(18, int(center[1]))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        return frame

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
                        ('评分方式', 'Scoring Method', 'scoring_method', ''),
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
