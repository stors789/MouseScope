from services.analysis_runtime import *


def _subject_fields_from_roi(roi):
    fields = {}
    for key in ('project_unit_id', 'subject_id', 'animal_code', 'group_id', 'group_name'):
        value = (roi or {}).get(key) if isinstance(roi, dict) else None
        if value not in (None, ''):
            fields[key] = value
    return fields


def _coerce_polygon_points(points, min_points=3):
    return _coerce_point_list(points, min_points=min_points)


class LightDarkBoxMultiArenaAnalysisSession(AnalysisSessionBase):
    """Multi-ROI light-dark box analysis session."""
    batch_progress_template = "Processing LDB video {index}/{total}: {name}"
    batch_complete_template = "All {total} LDB videos finished."

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths
        self.video_infos = video_infos
        self.config = config
        self.experiment_type = 'light_dark_box'

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
        dark_box_visible = self._truthy(active_config.get('dark_box_visible'), True)
        if isinstance(rois, str):
            rois = json.loads(rois)
        parsed = []
        for idx, roi in enumerate(rois):
            if not isinstance(roi, dict):
                raise ValueError(f'LDB ROI {idx + 1} must be an object.')
            zones = roi.get('zones') or {}
            zone_polygons = {
                'light': _coerce_polygon_points(zones.get('light'), min_points=3),
            }
            if dark_box_visible:
                zone_polygons['dark'] = _coerce_polygon_points(zones.get('dark'), min_points=3)
            if zones.get('doorway'):
                zone_polygons['doorway'] = _coerce_polygon_points(zones.get('doorway'), min_points=3)
            detection_points = None
            if roi.get('detection_points'):
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)
            parsed.append({
                'roi_id': idx + 1,
                'name': str(roi.get('name') or f'LDB {idx + 1}').strip() or f'LDB {idx + 1}',
                'zones': zone_polygons,
                'detection_points': detection_points,
                'time_config': roi.get('time_config') or {'mode': 'global', 'start': None, 'end': None},
                'subject_fields': _subject_fields_from_roi(roi),
            })
        if not parsed:
            raise ValueError('Light-dark box analysis requires at least one ROI.')
        return parsed

    @staticmethod
    def _build_live_metrics(rows):
        if not rows:
            return {}
        reliable_rows = [row for row in rows if row.get('tracking_reliable') is not False]
        return {
            'roi_count': len(rows),
            'detected_roi_count': sum(1 for row in rows if row.get('detected')),
            'total_distance_cm': round(sum(float(row.get('distance_cm', 0.0) or 0.0) for row in reliable_rows), 3) if reliable_rows else None,
            'total_transitions': sum(int(row.get('total_transitions') or 0) for row in rows),
            'mean_light_percent': round(
                sum(float(row.get('light_percent', 0.0) or 0.0) for row in rows) / max(len(rows), 1),
                3,
            ),
        }

    @staticmethod
    def _live_row(tracker, result):
        total_time_s = getattr(tracker, 'time_in_window_s', 0.0) or (tracker.frames_in_window / tracker.fps if tracker.fps > 0 else 0.0)
        light_time_s = tracker.zone_time_s.get('light', 0.0)
        return {
            'roi_id': tracker.roi_id,
            'roi_name': tracker.roi_name,
            'detected': bool(result.get('detected')),
            'dark_box_visible': bool(getattr(tracker, 'dark_box_visible', True)),
            'tracking_reliable': bool(getattr(tracker, 'locomotion_reliable', True)),
            'zone': result.get('zone_label', 'Unknown'),
            'speed_cm_s': round(float(result.get('speed_cm_s') or 0.0), 3) if getattr(tracker, 'locomotion_reliable', True) else None,
            'distance_cm': round(float(tracker.total_distance_cm), 3) if getattr(tracker, 'locomotion_reliable', True) else None,
            'light_percent': round(light_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            'light_entries': int(tracker.zone_entries.get('light', 0)),
            'total_transitions': int(tracker.transition_count),
            'in_stats_window': bool(result.get('in_stats_window', True)),
        }

    def _create_trackers(self, roi_configs, vision_config, fps, global_start_time_s):
        return [
            LightDarkBoxTracker(
                zone_polygons=roi['zones'],
                detection_polygon=roi.get('detection_points'),
                arena_width_cm=float(vision_config.get('arena_width_cm', 40.0)),
                arena_height_cm=float(vision_config.get('arena_height_cm', 40.0)),
                light_box_length_cm=float(vision_config.get('light_box_length_cm', 40.0)),
                light_box_width_cm=float(vision_config.get('light_box_width_cm', 20.0)),
                dark_box_length_cm=float(vision_config.get('dark_box_length_cm', 40.0)),
                dark_box_width_cm=float(vision_config.get('dark_box_width_cm', 20.0)),
                shared_edge_length_cm=float(vision_config.get('shared_edge_length_cm', 40.0)),
                fps=fps,
                door_open_delay_s=float(vision_config.get('door_open_delay_s', 3.0)),
                transition_min_dwell_s=float(vision_config.get('transition_min_dwell_s', 0.12)),
                doorway_reset_s=float(vision_config.get('doorway_reset_s', 0.2)),
                movement_threshold_cm_s=float(vision_config.get('movement_threshold_cm_s', 2.0)),
                dark_box_visible=self._truthy(vision_config.get('dark_box_visible'), True),
                roi_id=roi['roi_id'],
                roi_name=roi['name'],
                time_config=roi.get('time_config'),
                global_start_time_s=global_start_time_s,
                subject_fields=roi.get('subject_fields'),
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
            suffix='ldb',
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
        self._log(f"LDB video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")
        timebase, timebase_summary, total_frames = self._build_timebase_for_video(video_path, fps, total_frames, label='LDB')

        sampling = self._resolve_video_sampling(
            width,
            height,
            fps,
            vision_config,
            label='LDB',
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
        self._log(f"Configured {len(trackers)} LDB boxes")

        start_frame, end_frame = self._time_range_to_frames(timebase, total_frames, start_sec, end_sec)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start LDB analysis from frame {start_frame} ({timebase.time_at(start_frame):.1f}s)")

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
            current_dt = timebase.duration_between(previous_analyzed_frame, current_frame_no, sample_fps=sampling['effective_fps'])
            analysis_frame = self._resize_for_sampling(frame, sampling)
            gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
            begin_frame_cache(bg_subtractor, current_frame_no)
            results = [
                tracker.update(analysis_frame, current_time, gray, bg_subtractor, morph_processor, detector, dt_s=current_dt)
                for tracker in trackers
            ]
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
                self._set_video_progress(batch_index, analysis_fraction * analysis_phase, f'Frame {source_frames_done}/{frames_to_process} ({{progress}}%)')

        cap.release()
        if not self.is_running:
            return
        self._ensure_analysis_frames_processed(video_path=video_path, label='LDB', frame_idx=frame_idx, source_frames_done=source_frames_done)

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
            self._add_sampling_summary_fields(summary, sampling)
            self._add_timebase_summary_fields(summary, timebase_summary)
            summary['video_name'] = video_info.get('original_name', video_name)
            summary_rows.append(summary)
            if getattr(tracker, 'locomotion_reliable', True):
                for row in tracker.frame_records:
                    record = dict(row)
                    record['video_name'] = video_info.get('original_name', video_name)
                    track_rows.append(record)
            for row in tracker.build_time_bins(bin_seconds=float(vision_config.get('bin_seconds', 60.0))):
                bucket = dict(row)
                bucket['video_name'] = video_info.get('original_name', video_name)
                bins_rows.append(bucket)

        if HAS_PANDAS and track_rows:
            work_path = f"{output_base}_track.csv"
            if _write_dataframe_csv(track_rows, work_path):
                self._append_result_file(_export_result_file(work_path, f"{final_base}_track.csv"), 'track_csv')
        if HAS_PANDAS and summary_rows:
            work_path = f"{output_base}_summary.csv"
            if _write_dataframe_csv(summary_rows, work_path):
                self._append_result_file(_export_result_file(work_path, f"{final_base}_summary.csv"), 'summary_csv')
        if HAS_PANDAS and bins_rows:
            work_path = f"{output_base}_bins.csv"
            if _write_dataframe_csv(bins_rows, work_path):
                self._append_result_file(_export_result_file(work_path, f"{final_base}_bins.csv"), 'bins_csv')

        report_work_path = f"{output_base}_report.txt"
        self._generate_txt_report(summary_rows, report_work_path)
        self._append_result_file(_export_result_file(report_work_path, f"{final_base}_report.txt"), 'report')

        heatmap_work_paths = []
        heatmap_final_paths = []
        trackmap_work_paths = []
        trackmap_final_paths = []
        for tracker in trackers:
            roi_slug = sanitize_light_dark_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"ldb_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"
            if render_light_dark_box_heatmap(tracker, heatmap_work_path):
                exported = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported)
                self._append_result_file(exported)
            if render_light_dark_box_trackmap(tracker, trackmap_work_path):
                exported = _export_result_file(trackmap_work_path, trackmap_final_path)
                trackmap_work_paths.append(trackmap_work_path)
                trackmap_final_paths.append(exported)
                self._append_result_file(exported)
        if heatmap_work_paths:
            if len(heatmap_final_paths) == 1:
                self._append_result_file(heatmap_final_paths[0], 'heatmap')
            else:
                grid_work = f"{output_base}_heatmaps.png"
                if render_light_dark_box_image_grid(heatmap_work_paths, grid_work, title=f"{video_name} | LDB Heatmaps"):
                    self._append_result_file(_export_result_file(grid_work, f"{final_base}_heatmaps.png"), 'heatmap')
        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                grid_work = f"{output_base}_trackmaps.png"
                if render_light_dark_box_image_grid(trackmap_work_paths, grid_work, title=f"{video_name} | LDB Trajectory Maps"):
                    self._append_result_file(_export_result_file(grid_work, f"{final_base}_trackmaps.png"), 'trackmap')

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
                self._log(f"LDB annotated video saved: {exported_video}")

        self.realtime_data = [
            {
                'roi_id': row['roi_id'],
                'roi_name': row['roi_name'],
                'detected': True,
                'dark_box_visible': row.get('dark_box_visible', True),
                'tracking_reliable': row.get('tracking_reliable', True),
                'zone': 'Completed',
                'speed_cm_s': row.get('mean_speed_cm_s'),
                'distance_cm': row.get('total_distance_cm'),
                'light_percent': row['light_time_percent'],
                'light_entries': row['light_entries'],
                'total_transitions': row['total_transitions'],
                'in_stats_window': True,
            }
            for row in summary_rows
        ]
        self.live_metrics = self._build_live_metrics(self.realtime_data)
        if batch_index is None and self.is_running:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = 'Light-dark box analysis completed.'
        elapsed = time.time() - video_start_time
        self._log(f"LDB video processed in {elapsed:.1f}s, {source_frames_done} source frames ({frame_idx} analyzed samples)")
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
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to reopen LDB video for rendering: {video_path}")
            return False
        writer = create_annotated_video_writer(output_video_path, fps, width, height, label='LDB video', source_video_path=video_path, logger=self._log, async_write=True)
        if writer is None:
            cap.release()
            self._log("LDB annotated video disabled because no writer is available.")
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
                current_dt = timebase.duration_between(previous_analyzed_frame, current_frame_no, sample_fps=fps)
                analysis_frame = self._resize_for_sampling(frame, sampling)
                gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
                begin_frame_cache(bg_subtractor, current_frame_no)
                results = [
                    tracker.update(analysis_frame, current_time, gray, bg_subtractor, morph_processor, detector, dt_s=current_dt)
                    for tracker in render_trackers
                ]
                previous_analyzed_frame = current_frame_no
                result_frame = draw_light_dark_box_results(analysis_frame, list(zip(render_trackers, results)))
                writer.write(result_frame)
                if frame_idx % 30 == 0:
                    self.current_frame = result_frame
                frame_idx += 1
                skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
                covered = 1 + skipped
                source_frames_done += covered
                self._advance_frame_counters(covered, processed=False)
                if frame_idx % 60 == 0:
                    self._set_video_progress(batch_index, 0.82 + 0.16 * source_frames_done / max(frames_to_process, 1), f'Rendering LDB video {source_frames_done}/{frames_to_process} ({{progress}}%)')
        finally:
            cap.release()
            writer.release()
        return os.path.exists(output_video_path)

    def _generate_txt_report(self, summaries, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 72 + "\n")
                f.write("MouseScope 明暗箱报告 (Light-Dark Box Report)\n")
                f.write("=" * 72 + "\n\n")
                f.write(f"{_bi_label('分析时间', 'Analysis Time')}: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if summaries:
                    f.write(f"{_bi_label('视频', 'Video')}: {summaries[0].get('video_name', '')}\n")
                f.write(f"{_bi_label('ROI 数量', 'ROI Count')}: {len(summaries)}\n\n")
                for summary in summaries:
                    f.write(f"[ROI {summary.get('roi_id')}] {summary.get('roi_name', '')}\n")
                    f.write(f"  {_bi_label('核心行为指标', 'Core Behavioral Metrics')}\n")
                    _write_report_metric_lines(f, summary, [
                        ('总时长', 'Total Time', 'total_time_s', ' s'),
                        ('检测率', 'Detected Rate', 'detected_rate_percent', '%'),
                        ('明室可见检测率', 'Visible-Light Detection Rate', 'visible_light_detection_rate_percent', '%'),
                        ('明室时间', 'Light Time', 'light_time_s', ' s'),
                        ('暗室时间', 'Dark Time', 'dark_time_s', ' s'),
                        ('门区时间', 'Doorway Time', 'doorway_time_s', ' s'),
                        ('明室比例', 'Light Percent', 'light_time_percent', '%'),
                        ('明室进入次数', 'Light Entries', 'light_entries', ''),
                        ('总转换次数', 'Total Transitions', 'total_transitions', ''),
                        ('首次入明室潜伏期', 'First Light Latency', 'first_light_latency_s', ' s'),
                    ])
                    if summary.get('tracking_reliable', True):
                        f.write(f"  {_bi_label('辅助运动学 / QC 指标', 'Auxiliary Locomotion / QC Metrics')}\n")
                        _write_report_metric_lines(f, summary, [
                            ('明箱长', 'Light Box Length', 'light_box_length_cm', ' cm'),
                            ('明箱宽', 'Light Box Width', 'light_box_width_cm', ' cm'),
                            ('暗箱长', 'Dark Box Length', 'dark_box_length_cm', ' cm'),
                            ('暗箱宽', 'Dark Box Width', 'dark_box_width_cm', ' cm'),
                            ('两箱重合边长度', 'Shared Edge Length', 'shared_edge_length_cm', ' cm'),
                            ('总距离', 'Total Distance', 'total_distance_cm', ' cm'),
                            ('平均速度', 'Mean Speed', 'mean_speed_cm_s', ' cm/s'),
                            ('最大速度', 'Max Speed', 'max_speed_cm_s', ' cm/s'),
                            ('明室距离', 'Light Distance', 'light_distance_cm', ' cm'),
                            ('暗室距离', 'Dark Distance', 'dark_distance_cm', ' cm'),
                            ('移动时间', 'Moving Time', 'moving_time_s', ' s'),
                            ('低活动时间', 'Low-Motion Time', 'immobility_time_s', ' s'),
                        ])
                    else:
                        f.write(f"  {_bi_label('辅助运动学 / QC 指标', 'Auxiliary Locomotion / QC Metrics')}: {_bi_label('暗箱不可见，未输出轨迹、速度和距离', 'Dark chamber invisible; trajectory, speed, and distance were not exported')}\n")
                    if summary.get('qc_flags'):
                        f.write(f"  {_bi_label('质量标记', 'QC Flags')}: {summary.get('qc_flags')}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Failed to write LDB report: {e}")


LightDarkBoxAnalysisSession = LightDarkBoxMultiArenaAnalysisSession
