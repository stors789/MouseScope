from services.analysis_runtime import *
from services.payload_geometry import (
    circle_boundary_points as _circle_boundary_points,
    coerce_point_xy as _point_xy,
    ellipse_boundary_points as _ellipse_boundary_points,
    normalize_water_maze_pool_payload as _normalize_pool_payload,
)


def _point_list(points, min_points=1):
    if not isinstance(points, list) or len(points) < int(min_points):
        raise ValueError(f'requires at least {min_points} points')
    return [_point_xy(point) for point in points]


def _first_value(*values):
    for value in values:
        if value not in (None, ''):
            return value
    return None


def _float_value(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


class WaterMazeMultiPoolAnalysisSession(AnalysisSessionBase):
    """Multi-pool Morris water maze analysis session."""

    batch_progress_template = "Processing MWM video {index}/{total}: {name}"
    batch_complete_template = "All {total} MWM videos finished."

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths
        self.video_infos = video_infos
        self.config = config
        self.experiment_type = 'water_maze'

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

        if not rois and active_config.get('pool_boundary_points') and active_config.get('platform_point'):
            rois = [{
                'name': 'Pool 1',
                'pool_boundary_points': active_config.get('pool_boundary_points'),
                'platform_point': active_config.get('platform_point'),
                'time_config': {'mode': 'global', 'start': None, 'end': None},
            }]

        parsed = []
        for idx, roi in enumerate(rois):
            if not isinstance(roi, dict):
                raise ValueError(f'MWM ROI {idx + 1} must be an object.')
            pool = dict(roi.get('pool') or {})
            boundary = (
                roi.get('pool_boundary_points')
                or roi.get('boundary_points')
                or roi.get('points')
                or pool.get('boundary_points')
                or pool.get('pool_boundary_points')
                or pool.get('points')
            )
            if not boundary:
                center = pool.get('center') or roi.get('pool_center')
                radius_px = pool.get('radius_px') or pool.get('radius') or roi.get('pool_radius_px')
                if center is not None and radius_px is not None:
                    boundary = _circle_boundary_points(center, radius_px)
            if not boundary:
                boundary = _ellipse_boundary_points(pool)
            if not boundary:
                raise ValueError(f'MWM ROI {idx + 1} requires a pool boundary.')
            boundary_points = _point_list(boundary, min_points=5)

            platform = dict(roi.get('platform') or {})
            platform_point = (
                roi.get('platform_point')
                or platform.get('point')
                or platform.get('center')
            )
            if platform_point is None and platform.get('x') is not None and platform.get('y') is not None:
                platform_point = platform
            if platform_point is None:
                raise ValueError(f'MWM ROI {idx + 1} requires a platform point.')

            detection_points = None
            if roi.get('detection_points'):
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)

            platform_exclusion_points = None
            if roi.get('platform_exclusion_points'):
                platform_exclusion_points = _coerce_point_list(roi.get('platform_exclusion_points'), min_points=DETECTION_MASK_MIN_POINTS)

            roi_name = str(roi.get('name') or f'Pool {idx + 1}').strip() or f'Pool {idx + 1}'
            pool_diameter_cm = float(roi.get('pool_diameter_cm') or pool.get('diameter_cm') or active_config.get('pool_diameter_cm') or 120.0)
            platform_payload = {
                'center': _point_xy(platform_point),
                'radius_cm': float(platform.get('radius_cm') or roi.get('platform_radius_cm') or active_config.get('platform_radius_cm') or 5.0),
            }
            quadrant_rotation_deg = _float_value(
                _first_value(
                    roi.get('quadrant_rotation_deg'),
                    roi.get('coordinate_rotation_deg'),
                    active_config.get('quadrant_rotation_deg'),
                    active_config.get('coordinate_rotation_deg'),
                ),
                0.0,
            )
            parsed.append({
                'roi_id': idx + 1,
                'name': roi_name,
                'pool': _normalize_pool_payload(pool, pool_diameter_cm),
                'pool_boundary_points': boundary_points,
                'platform': platform_payload,
                'detection_points': detection_points,
                'platform_exclusion_points': platform_exclusion_points,
                'time_config': roi.get('time_config') or {'mode': 'global', 'start': None, 'end': None},
                'pool_diameter_cm': pool_diameter_cm,
                'target_quadrant': _first_value(roi.get('target_quadrant'), active_config.get('target_quadrant'), ''),
                'start_position': _first_value(roi.get('start_position'), active_config.get('start_position'), 'N'),
                'coordinate_rotation_deg': quadrant_rotation_deg,
                'quadrant_rotation_deg': quadrant_rotation_deg,
                'subject_id': roi.get('subject_id'),
                'group_id': roi.get('group_id'),
                'group_name': roi.get('group_name'),
                'animal_code': roi.get('animal_code'),
            })

        if not parsed:
            raise ValueError('Water maze analysis requires at least one pool ROI.')
        return parsed

    @staticmethod
    def _build_live_metrics(rows):
        if not rows:
            return {}
        return {
            'roi_count': len(rows),
            'detected_roi_count': sum(1 for row in rows if row.get('detected')),
            'success_count': sum(1 for row in rows if row.get('success')),
            'total_distance_cm': round(sum(float(row.get('distance_cm', 0.0) or 0.0) for row in rows), 3),
            'mean_target_quadrant_percent': round(
                sum(float(row.get('target_quadrant_percent', 0.0) or 0.0) for row in rows) / max(len(rows), 1),
                3,
            ),
            'mean_thigmotaxis_percent': round(
                sum(float(row.get('thigmotaxis_percent', 0.0) or 0.0) for row in rows) / max(len(rows), 1),
                3,
            ),
        }

    @staticmethod
    def _live_row(tracker, result):
        total_time_s = getattr(tracker, 'time_in_window_s', 0.0) or (tracker.frames_in_window / tracker.fps if tracker.fps > 0 else 0.0)
        target_time_s = tracker.quadrant_time_s.get(tracker.target_quadrant, 0.0)
        return {
            'roi_id': tracker.roi_id,
            'roi_name': tracker.roi_name,
            'detected': bool(result.get('detected')),
            'quadrant': result.get('quadrant', 'Unknown'),
            'quadrant_label': result.get('quadrant_label', result.get('quadrant', 'Unknown')),
            'target_quadrant': tracker.target_quadrant,
            'target_quadrant_label': tracker.quadrant_label(tracker.target_quadrant),
            'speed_cm_s': round(float(result.get('speed_cm_s', 0.0) or 0.0), 3),
            'distance_cm': round(float(tracker.total_distance_cm), 3),
            'success': bool(tracker.success),
            'escape_latency_s': round(float(tracker.escape_latency_s), 3) if tracker.escape_latency_s is not None else None,
            'platform_entries': int(tracker.platform_entries),
            'platform_crossings': int(tracker.platform_crossings),
            'target_quadrant_percent': round(target_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            'thigmotaxis_percent': round(tracker.thigmotaxis_time_s / max(total_time_s, 1e-6) * 100.0, 2),
            'in_platform_zone': bool(result.get('in_platform_zone')),
            'in_stats_window': bool(result.get('in_stats_window', True)),
        }

    def _create_trackers(self, roi_configs, vision_config, fps, global_start_time_s):
        return [
            WaterMazeTracker(
                pool_boundary_points=roi['pool_boundary_points'],
                pool=roi.get('pool'),
                platform=roi['platform'],
                detection_points=roi.get('detection_points'),
                platform_exclusion_points=roi.get('platform_exclusion_points'),
                pool_diameter_cm=float(roi.get('pool_diameter_cm') or vision_config.get('pool_diameter_cm', 120.0)),
                platform_radius_cm=float(roi.get('platform', {}).get('radius_cm') or vision_config.get('platform_radius_cm', 5.0)),
                trial_type=vision_config.get('trial_type', 'acquisition_hidden'),
                target_quadrant=_first_value(roi.get('target_quadrant'), vision_config.get('target_quadrant')),
                start_position=_first_value(roi.get('start_position'), vision_config.get('start_position'), 'N'),
                day_label=vision_config.get('day_label', ''),
                trial_no=int(vision_config.get('trial_no', 1)),
                fps=fps,
                movement_threshold_cm_s=float(vision_config.get('movement_threshold_cm_s', 2.0)),
                thigmotaxis_band_cm=float(vision_config.get('thigmotaxis_band_cm', 10.0)),
                platform_dwell_s=float(vision_config.get('platform_dwell_s', 1.0)),
                track_smoothing_max_jump_cm=float(vision_config.get('track_smoothing_max_jump_cm', 25.0)),
                max_gap_s=float(vision_config.get('max_gap_s', 0.5)),
                quadrant_rotation_deg=float(
                    _first_value(
                        roi.get('quadrant_rotation_deg'),
                        roi.get('coordinate_rotation_deg'),
                        vision_config.get('quadrant_rotation_deg'),
                        vision_config.get('coordinate_rotation_deg'),
                        0.0,
                    )
                ),
                whishaw_corridor_width_cm=vision_config.get('whishaw_corridor_width_cm'),
                exclude_platform_from_detection=bool(vision_config.get('exclude_platform_from_detection', False)),
                correction_mode=vision_config.get('correction_mode', 'ellipse_affine'),
                distortion_k1=float(vision_config.get('distortion_k1', 0.0)),
                distortion_k2=float(vision_config.get('distortion_k2', 0.0)),
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
            suffix='mwm',
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
        self._log(f"MWM video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")
        timebase, timebase_summary, total_frames = self._build_timebase_for_video(
            video_path,
            fps,
            total_frames,
            label='MWM',
        )

        sampling = self._resolve_video_sampling(
            width,
            height,
            fps,
            vision_config,
            label='MWM',
            temporal_default=True,
            max_dim_default=960,
            target_fps_default=30.0,
        )
        self._log_video_sampling(sampling)

        bg_subtractor, morph_processor, detector = self._build_processors_for_sampling(
            video_path,
            vision_config,
            sampling,
            static_frames=20,
            erosion_default=1,
            dilation_default=2,
            min_area_default=80,
        )

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        source_roi_configs = self._parse_rois(vision_config)
        roi_configs = self._scale_roi_configs_for_sampling(source_roi_configs, sampling)

        trackers = self._create_trackers(roi_configs, vision_config, sampling['effective_fps'], global_start_time_s)
        self._log(f"Configured {len(trackers)} MWM pools")

        start_frame, end_frame = self._time_range_to_frames(timebase, total_frames, start_sec, end_sec)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start MWM analysis from frame {start_frame} ({timebase.time_at(start_frame):.1f}s)")

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
                self._set_video_progress(
                    batch_index,
                    analysis_fraction * analysis_phase,
                    f'Frame {source_frames_done}/{frames_to_process} ({{progress}}%)',
                )

        cap.release()
        if not self.is_running:
            return
        self._ensure_analysis_frames_processed(
            video_path=video_path,
            label='MWM',
            frame_idx=frame_idx,
            source_frames_done=source_frames_done,
        )

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

            for row in tracker.build_time_bins(bin_seconds=float(vision_config.get('bin_seconds', 10.0))):
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
            roi_slug = sanitize_water_maze_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"pool_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"

            if not tracker.trajectory_pool_cm_stats:
                self._log(f"Skipped MWM heatmap, no valid trajectory points: {heatmap_final_path}")
            elif render_water_maze_heatmap(tracker, heatmap_work_path):
                exported_heatmap = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported_heatmap)
                self._append_result_file(exported_heatmap)
            else:
                self._log(f"Failed to save MWM heatmap: {heatmap_final_path}")

            if not tracker.trajectory_pool_cm_stats:
                self._log(f"Skipped MWM trackmap, no valid trajectory points: {trackmap_final_path}")
            elif render_water_maze_trackmap(tracker, trackmap_work_path):
                exported_trackmap = _export_result_file(trackmap_work_path, trackmap_final_path)
                trackmap_work_paths.append(trackmap_work_path)
                trackmap_final_paths.append(exported_trackmap)
                self._append_result_file(exported_trackmap)
            else:
                self._log(f"Failed to save MWM trackmap: {trackmap_final_path}")

        if heatmap_work_paths:
            if len(heatmap_final_paths) == 1:
                self._append_result_file(heatmap_final_paths[0], 'heatmap')
            else:
                heatmap_grid_work_path = f"{output_base}_heatmaps.png"
                heatmap_grid_final_path = f"{final_base}_heatmaps.png"
                if render_water_maze_image_grid(heatmap_work_paths, heatmap_grid_work_path, title=f"{video_name} | Heatmaps"):
                    self._append_result_file(_export_result_file(heatmap_grid_work_path, heatmap_grid_final_path), 'heatmap')
                else:
                    self._log(f"Failed to save MWM heatmap grid: {heatmap_grid_final_path}")

        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                trackmap_grid_work_path = f"{output_base}_trackmaps.png"
                trackmap_grid_final_path = f"{final_base}_trackmaps.png"
                if render_water_maze_image_grid(trackmap_work_paths, trackmap_grid_work_path, title=f"{video_name} | Trajectory Maps"):
                    self._append_result_file(_export_result_file(trackmap_grid_work_path, trackmap_grid_final_path), 'trackmap')
                else:
                    self._log(f"Failed to save MWM trackmap grid: {trackmap_grid_final_path}")

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
                self._log(f"MWM annotated video saved: {exported_video}")

        self.realtime_data = [
            {
                'roi_id': row['roi_id'],
                'roi_name': row['roi_name'],
                'detected': True,
                'quadrant': 'Completed',
                'target_quadrant': row['target_quadrant'],
                'target_quadrant_label': row.get('target_quadrant_label', row['target_quadrant']),
                'speed_cm_s': row['mean_swim_speed_cm_s'],
                'distance_cm': row['total_distance_cm'],
                'success': row['success'],
                'escape_latency_s': row['escape_latency_s'],
                'platform_entries': row['platform_entries'],
                'platform_crossings': row['platform_crossings'],
                'target_quadrant_percent': row['target_quadrant_percent'],
                'thigmotaxis_percent': row['thigmotaxis_percent'],
                'search_strategy': row.get('search_strategy'),
                'search_strategy_label': row.get('search_strategy_label'),
                'search_strategy_label_zh': row.get('search_strategy_label_zh'),
                'search_strategy_display': row.get('search_strategy_display'),
                'search_strategy_score': row.get('search_strategy_score'),
                'in_platform_zone': False,
                'in_stats_window': True,
            }
            for row in summary_rows
        ]
        self.live_metrics = self._build_live_metrics(self.realtime_data)

        if batch_index is None and self.is_running:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = 'Water maze analysis completed.'

        elapsed = time.time() - video_start_time
        self._log(f"MWM video processed in {elapsed:.1f}s, {source_frames_done} source frames ({frame_idx} analyzed samples)")

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
            self._log(f"Unable to reopen MWM video for rendering: {video_path}")
            return False

        writer = create_annotated_video_writer(
            output_video_path,
            fps,
            width,
            height,
            label='MWM video',
            source_video_path=video_path,
            logger=self._log,
            async_write=True,
        )
        if writer is None:
            cap.release()
            self._log("MWM annotated video disabled because no writer is available.")
            return False

        bg_subtractor, morph_processor, detector = self._build_processors_for_sampling(
            video_path,
            vision_config,
            sampling,
            static_frames=20,
            erosion_default=1,
            dilation_default=2,
            min_area_default=80,
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
                current_dt = timebase.duration_between(
                    previous_analyzed_frame,
                    current_frame_no,
                    sample_fps=fps,
                )
                analysis_frame = self._resize_for_sampling(frame, sampling)
                gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
                begin_frame_cache(bg_subtractor, current_frame_no)
                results = [
                    tracker.update(analysis_frame, current_time, gray, bg_subtractor, morph_processor, detector, dt_s=current_dt)
                    for tracker in render_trackers
                ]
                previous_analyzed_frame = current_frame_no
                result_frame = draw_water_maze_results(analysis_frame, list(zip(render_trackers, results)))
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
                        f'Rendering MWM video {source_frames_done}/{frames_to_process} ({{progress}}%)',
                    )
        finally:
            cap.release()
            writer.release()
        return os.path.exists(output_video_path)

    @staticmethod
    def _write_report_section(handle, zh_title, en_title, summary, metric_specs):
        handle.write(f"  {_bi_label(zh_title, en_title)}\n")
        _write_report_metric_lines(handle, summary, metric_specs, indent='    ')

    @staticmethod
    def _trial_type_label(trial_type):
        value = str(trial_type or 'acquisition_hidden').strip().lower()
        labels = {
            'acquisition_hidden': ('隐藏平台训练期', 'Hidden Platform Training'),
            'probe': ('探测试次', 'Probe Trial'),
            'visible_platform': ('可见平台期', 'Visible Platform'),
            'reversal': ('反转学习期', 'Reversal Learning'),
        }
        zh_text, en_text = labels.get(value, labels['acquisition_hidden'])
        return _bi_label(zh_text, en_text)

    @staticmethod
    def _write_period_specific_metrics(handle, summary):
        trial_type = str(summary.get('trial_type') or 'acquisition_hidden').strip().lower()
        if trial_type == 'probe':
            WaterMazeMultiPoolAnalysisSession._write_report_section(handle, '探测试次记忆指标', 'Probe Trial Memory Metrics', summary, [
                ('搜索策略', 'Search Strategy', 'search_strategy_display', ''),
                ('搜索策略评分', 'Search Strategy Score', 'search_strategy_score', ''),
                ('搜索策略置信度', 'Search Strategy Confidence', 'search_strategy_confidence', ''),
                ('目标象限时间', 'Target Quadrant Time', 'target_quadrant_time_s', ' s'),
                ('目标象限比例', 'Target Quadrant Percent', 'target_quadrant_percent', '%'),
                ('目标象限路程', 'Target Quadrant Distance', 'target_quadrant_distance_cm', ' cm'),
                ('原平台穿越次数', 'Former Platform Crossings', 'platform_crossings', ''),
                ('原平台区进入次数', 'Former Platform Entries', 'platform_entries', ''),
                ('首次穿越原平台区潜伏期', 'First Former Platform Crossing Latency', 'latency_to_first_platform_crossing_s', ' s'),
                ('首次进入原平台区潜伏期', 'First Former Platform Zone Entry Latency', 'time_to_first_platform_entry_s', ' s'),
                ('原平台区停留时间', 'Former Platform Zone Time', 'time_in_platform_zone_s', ' s'),
                ('平均距原平台距离', 'Mean Distance to Former Platform', 'mean_distance_to_platform_cm', ' cm'),
                ('Gallagher 接近度', 'Gallagher Proximity', 'gallagher_proximity_score', ' cm'),
                ('CIPL', 'CIPL', 'cipl', ''),
            ])
            WaterMazeMultiPoolAnalysisSession._write_report_section(handle, '四象限占用', 'Quadrant Occupancy', summary, [
                ('象限 I (NE) 时间', 'Quadrant I (NE) Time', 'quadrant_ne_time_s', ' s'),
                ('象限 I (NE) 比例', 'Quadrant I (NE) Percent', 'quadrant_ne_percent', '%'),
                ('象限 II (NW) 时间', 'Quadrant II (NW) Time', 'quadrant_nw_time_s', ' s'),
                ('象限 II (NW) 比例', 'Quadrant II (NW) Percent', 'quadrant_nw_percent', '%'),
                ('象限 III (SW) 时间', 'Quadrant III (SW) Time', 'quadrant_sw_time_s', ' s'),
                ('象限 III (SW) 比例', 'Quadrant III (SW) Percent', 'quadrant_sw_percent', '%'),
                ('象限 IV (SE) 时间', 'Quadrant IV (SE) Time', 'quadrant_se_time_s', ' s'),
                ('象限 IV (SE) 比例', 'Quadrant IV (SE) Percent', 'quadrant_se_percent', '%'),
            ])
        elif trial_type == 'visible_platform':
            WaterMazeMultiPoolAnalysisSession._write_report_section(handle, '可见平台控制指标', 'Visible Platform Control Metrics', summary, [
                ('搜索策略', 'Search Strategy', 'search_strategy_display', ''),
                ('搜索策略评分', 'Search Strategy Score', 'search_strategy_score', ''),
                ('搜索策略置信度', 'Search Strategy Confidence', 'search_strategy_confidence', ''),
                ('是否找到平台', 'Platform Found', 'success', ''),
                ('逃逸潜伏期', 'Escape Latency', 'escape_latency_s', ' s'),
                ('到达平台路径长', 'Path to Platform', 'path_length_to_platform_cm', ' cm'),
                ('首次进入平台区潜伏期', 'First Platform Entry Latency', 'time_to_first_platform_entry_s', ' s'),
                ('平台区进入次数', 'Platform Entries', 'platform_entries', ''),
                ('平台区停留时间', 'Platform Zone Time', 'time_in_platform_zone_s', ' s'),
                ('平均距平台距离', 'Mean Distance to Platform', 'mean_distance_to_platform_cm', ' cm'),
                ('路径效率', 'Path Efficiency', 'path_efficiency', ''),
            ])
        else:
            title_zh = '反转学习指标' if trial_type == 'reversal' else '隐藏平台学习指标'
            title_en = 'Reversal Learning Metrics' if trial_type == 'reversal' else 'Hidden Platform Learning Metrics'
            WaterMazeMultiPoolAnalysisSession._write_report_section(handle, title_zh, title_en, summary, [
                ('搜索策略', 'Search Strategy', 'search_strategy_display', ''),
                ('搜索策略评分', 'Search Strategy Score', 'search_strategy_score', ''),
                ('搜索策略置信度', 'Search Strategy Confidence', 'search_strategy_confidence', ''),
                ('是否找到平台', 'Platform Found', 'success', ''),
                ('逃逸潜伏期', 'Escape Latency', 'escape_latency_s', ' s'),
                ('到达平台路径长', 'Path to Platform', 'path_length_to_platform_cm', ' cm'),
                ('首次进入平台区潜伏期', 'First Platform Entry Latency', 'time_to_first_platform_entry_s', ' s'),
                ('平台区进入次数', 'Platform Entries', 'platform_entries', ''),
                ('平台穿越次数', 'Platform Crossings', 'platform_crossings', ''),
                ('平台区停留时间', 'Platform Zone Time', 'time_in_platform_zone_s', ' s'),
                ('路径效率', 'Path Efficiency', 'path_efficiency', ''),
                ('平均距平台距离', 'Mean Distance to Platform', 'mean_distance_to_platform_cm', ' cm'),
                ('Gallagher 接近度', 'Gallagher Proximity', 'gallagher_proximity_score', ' cm'),
                ('CIPL', 'CIPL', 'cipl', ''),
                ('初始航向误差', 'Initial Heading Error', 'initial_heading_error_deg', ' deg'),
                ('平均航向误差', 'Heading Error', 'heading_error_deg', ' deg'),
            ])

    @staticmethod
    def _write_common_control_metrics(handle, summary):
        WaterMazeMultiPoolAnalysisSession._write_report_section(handle, '通用运动 / QC 指标', 'Common Locomotion / QC Metrics', summary, [
            ('总时长', 'Total Time', 'total_time_s', ' s'),
            ('检测率', 'Detected Rate', 'detected_rate_percent', '%'),
            ('总路程', 'Total Distance', 'total_distance_cm', ' cm'),
            ('平均游泳速度', 'Mean Swim Speed', 'mean_swim_speed_cm_s', ' cm/s'),
            ('最大游泳速度', 'Max Swim Speed', 'max_swim_speed_cm_s', ' cm/s'),
            ('趋边时间', 'Thigmotaxis Time', 'thigmotaxis_time_s', ' s'),
            ('趋边比例', 'Thigmotaxis Percent', 'thigmotaxis_percent', '%'),
            ('Whishaw 通道时间', 'Whishaw Corridor Time', 'whishaw_corridor_time_s', ' s'),
            ('Whishaw 通道比例', 'Whishaw Corridor Percent', 'whishaw_corridor_percent', '%'),
            ('运动时间', 'Moving Time', 'moving_time_s', ' s'),
            ('漂浮/低速时间', 'Floating Time', 'floating_time_s', ' s'),
            ('搜索策略判定依据', 'Search Strategy Basis', 'search_strategy_basis', ''),
        ])

    def _generate_txt_report(self, summaries, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 72 + "\n")
                f.write("MouseScope 水迷宫实验报告 (Morris Water Maze Report)\n")
                f.write("=" * 72 + "\n\n")
                f.write(f"{_bi_label('分析时间', 'Analysis Time')}: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if summaries:
                    f.write(f"{_bi_label('视频', 'Video')}: {summaries[0].get('video_name', '')}\n")
                f.write(f"{_bi_label('泳池数量', 'Pool Count')}: {len(summaries)}\n\n")

                for summary in summaries:
                    f.write(f"[ROI {summary.get('roi_id')}] {summary.get('roi_name', '')}\n")
                    trial_type = summary.get('trial_type', 'acquisition_hidden')
                    f.write(f"  {_bi_label('试次类型', 'Trial Type')}: {self._trial_type_label(trial_type)} [{trial_type}]\n")
                    target_label = summary.get('target_quadrant_label') or summary.get('target_quadrant', '')
                    f.write(f"  {_bi_label('目标象限', 'Target Quadrant')}: {target_label} ({summary.get('target_quadrant', '')})\n")
                    f.write(f"  {_bi_label('坐标轴旋转', 'Axis Rotation')}: {_report_value(summary.get('coordinate_rotation_deg'), ' deg')}\n")
                    f.write(f"  {_bi_label('统计模式', 'Time Mode')}: {_report_time_mode_label(summary.get('time_mode', 'global'))}\n")
                    if summary.get('time_start_s') is not None:
                        f.write(f"  {_bi_label('起始时间', 'Window Start')}: {_report_value(summary.get('time_start_s'), ' s')}\n")
                    if summary.get('time_end_s') is not None:
                        f.write(f"  {_bi_label('结束时间', 'Window End')}: {_report_value(summary.get('time_end_s'), ' s')}\n")
                    self._write_period_specific_metrics(f, summary)
                    self._write_common_control_metrics(f, summary)
                    f.write(f"  {_bi_label('质量标记', 'QC Flags')}: {summary.get('qc_flags', '')}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Failed to write MWM report: {e}")


WaterMazeAnalysisSession = WaterMazeMultiPoolAnalysisSession
