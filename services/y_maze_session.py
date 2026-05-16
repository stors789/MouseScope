from services.analysis_runtime import *


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
            raise RuntimeError(f"Unable to open video: {video_path}")

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log(f"Y-maze video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

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

        bg_subtractor, morph_processor, detector = self._build_processors_for_sampling(
            video_path,
            vision_config,
            sampling,
            static_frames=15,
            erosion_default=1,
            dilation_default=2,
            min_area_default=120,
        )

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        source_roi_configs = self._parse_rois(vision_config)
        roi_configs = self._scale_roi_configs_for_sampling(source_roi_configs, sampling)

        trackers = self._create_trackers(roi_configs, vision_config, sampling['effective_fps'], global_start_time_s)
        self._log(f"Configured {len(trackers)} Y-maze arenas")

        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start Y-maze analysis from frame {start_frame} ({start_frame / max(fps, 1e-6):.1f}s)")

        frames_to_process = max(0, end_frame - start_frame + 1)
        frame_idx = 0
        source_frames_done = 0

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

            current_time = current_frame_no / max(fps, 1e-6)
            analysis_frame = self._resize_for_sampling(frame, sampling)
            gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
            begin_frame_cache(bg_subtractor, current_frame_no)
            results = [tracker.update(analysis_frame, current_time, gray, bg_subtractor, morph_processor, detector) for tracker in trackers]
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
                if batch_index is not None:
                    per_video = 100.0 / max(len(self.video_paths), 1)
                    self.progress = int(batch_index * per_video + analysis_fraction * analysis_phase * per_video)
                else:
                    self.progress = int(analysis_fraction * analysis_phase * 100)
                self.message = f'Frame {source_frames_done}/{frames_to_process} ({self.progress}%)'

        cap.release()

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
            self._add_sampling_summary_fields(summary, sampling)
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
            elif not tracker.trajectory_template_cm_stats:
                self._log(f"Skipped Y-maze heatmap, no valid trajectory points: {heatmap_final_path}")
            else:
                self._log(f"Failed to save Y-maze heatmap: {heatmap_final_path}")

            if render_y_maze_trackmap(tracker, trackmap_work_path):
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
                batch_index=batch_index,
            )

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
        batch_index=None,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to reopen Y-maze video for rendering: {video_path}")
            return False

        writer = create_annotated_video_writer(
            output_video_path,
            fps,
            width,
            height,
            label='Y-maze video',
            source_video_path=video_path,
            logger=self._log,
            async_write=True,
        )
        if writer is None:
            cap.release()
            self._log("Y-maze annotated video disabled because no writer is available.")
            return False

        bg_subtractor, morph_processor, detector = self._build_processors_for_sampling(
            video_path,
            vision_config,
            sampling,
            static_frames=15,
            erosion_default=1,
            dilation_default=2,
            min_area_default=120,
        )
        render_trackers = self._create_trackers(roi_configs, vision_config, fps, global_start_time_s)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = 0
        source_frames_done = 0
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

                current_time = current_frame_no / max(float(sampling.get('source_fps', fps)), 1e-6)
                analysis_frame = self._resize_for_sampling(frame, sampling)
                gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
                begin_frame_cache(bg_subtractor, current_frame_no)
                results = [
                    tracker.update(analysis_frame, current_time, gray, bg_subtractor, morph_processor, detector)
                    for tracker in render_trackers
                ]
                result_frame = draw_y_maze_results(analysis_frame, list(zip(render_trackers, results)))
                writer.write(result_frame)
                if frame_idx % 30 == 0:
                    self.current_frame = result_frame
                frame_idx += 1
                skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
                covered = 1 + skipped
                source_frames_done += covered
                self._advance_frame_counters(covered, processed=False)
                if frame_idx % 60 == 0:
                    if batch_index is not None:
                        per_video = 100.0 / max(len(self.video_paths), 1)
                        self.progress = int(batch_index * per_video + (0.82 + 0.16 * source_frames_done / max(frames_to_process, 1)) * per_video)
                    else:
                        self.progress = max(self.progress, int(82 + 16 * source_frames_done / max(frames_to_process, 1)))
                    self.message = f'Rendering Y-maze video {source_frames_done}/{frames_to_process}'
        finally:
            cap.release()
            writer.release()
        return os.path.exists(output_video_path)

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
