from services.analysis_runtime import *
from services.pose_tracking import PoseFrameResult, PoseInferenceCoordinator


class NovelObjectMultiArenaAnalysisSession(AnalysisSessionBase):
    """Multi-ROI Novel Object Recognition analysis session."""
    batch_progress_template = "Processing NOR video {index}/{total}: {name}"
    batch_complete_template = "All {total} NOR videos finished."

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths
        self.video_infos = video_infos
        self.config = config
        self.experiment_type = 'novel_object'

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
            'exploration_cumulative': [],
        }

    def _parse_rois(self, config=None):
        active_config = config or self.config
        rois = active_config.get('rois') or []
        if isinstance(rois, str):
            rois = json.loads(rois)

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

            objects = roi.get('objects') if isinstance(roi, dict) else None
            if not objects or len(objects) != 2:
                raise ValueError(f'ROI {idx + 1} must contain exactly 2 object points.')

            parsed_objects = []
            for obj_idx, obj in enumerate(objects):
                point = obj.get('point', obj) if isinstance(obj, dict) else obj
                if isinstance(point, dict):
                    point_xy = [float(point['x']), float(point['y'])]
                else:
                    point_xy = [float(point[0]), float(point[1])]

                parsed_objects.append({
                    'point': point_xy,
                    'name': (obj.get('name') if isinstance(obj, dict) else None) or f'Object {obj_idx + 1}',
                    'role': (obj.get('role') if isinstance(obj, dict) else None) or ('novel' if obj_idx == 1 else 'familiar'),
                })

            if isinstance(roi, dict):
                roi_name = str(roi.get('name') or f'ROI {idx + 1}').strip() or f'ROI {idx + 1}'
                time_config = roi.get('time_config') or {'mode': 'global', 'start': None, 'end': None}
            else:
                roi_name = f'ROI {idx + 1}'
                time_config = {'mode': 'global', 'start': None, 'end': None}

            parsed.append({
                'roi_id': idx + 1,
                'name': roi_name,
                'points': parsed_points,
                'detection_points': detection_points,
                'objects': parsed_objects,
                'time_config': time_config,
            })

        if not parsed:
            raise ValueError('Novel object recognition requires at least one ROI.')
        return parsed

    @staticmethod
    def _build_live_metrics(rows):
        if not rows:
            return {}
        valid_dis = [float(row['discrimination_index']) for row in rows if row.get('discrimination_index') is not None]
        return {
            'roi_count': len(rows),
            'detected_roi_count': sum(1 for row in rows if row.get('detected')),
            'total_distance_cm': round(sum(float(row.get('distance_cm', 0.0) or 0.0) for row in rows), 3),
            'total_exploration_s': round(sum(float(row.get('total_exploration_s', 0.0) or 0.0) for row in rows), 3),
            'mean_discrimination_index': round(sum(valid_dis) / len(valid_dis), 4) if valid_dis else None,
        }

    @staticmethod
    def _live_row(tracker, result):
        object_exploration_time_s = getattr(tracker, 'object_exploration_time_s', None)
        if object_exploration_time_s is not None:
            total_exploration_s = sum(float(v) for v in object_exploration_time_s)
        else:
            total_exploration_s = sum(tracker.object_exploration_frames) / tracker.fps if tracker.fps > 0 else 0.0
        role_times = {}
        for idx, obj in enumerate(tracker.objects):
            if object_exploration_time_s is not None:
                value = float(object_exploration_time_s[idx])
            else:
                value = tracker.object_exploration_frames[idx] / max(tracker.fps, 1e-6)
            role_times[obj['role']] = role_times.get(obj['role'], 0.0) + value
        novel_s = role_times.get('novel')
        familiar_s = role_times.get('familiar')
        discrimination = None
        preference = None
        if novel_s is not None and familiar_s is not None and (novel_s + familiar_s) > 0:
            discrimination = (novel_s - familiar_s) / (novel_s + familiar_s)
            preference = novel_s / (novel_s + familiar_s) * 100.0
        return {
            'roi_id': tracker.roi_id,
            'roi_name': tracker.roi_name,
            'detected': bool(result.get('detected')),
            'current_object': result.get('current_object_name') or '-',
            'current_role': result.get('current_object_role') or '-',
            'distance_cm': round(float(tracker.total_distance_cm), 3),
            'total_exploration_s': round(total_exploration_s, 3),
            'novel_exploration_s': round(novel_s, 3) if novel_s is not None else None,
            'familiar_exploration_s': round(familiar_s, 3) if familiar_s is not None else None,
            'discrimination_index': round(discrimination, 4) if discrimination is not None else None,
            'preference_index_percent': round(preference, 2) if preference is not None else None,
            'meets_min_exploration': bool(total_exploration_s >= tracker.min_valid_exploration_s),
            'in_stats_window': bool(result.get('in_stats_window', True)),
        }

    def _create_trackers(self, roi_configs, vision_config, fps, global_start_time_s):
        phase = vision_config.get('phase', 'recognition')
        arena_width_cm = float(vision_config.get('arena_width_cm', vision_config.get('arena_side_cm', 50.0)))
        arena_height_cm = float(vision_config.get('arena_height_cm', vision_config.get('arena_side_cm', 50.0)))
        return [
            NovelObjectTracker(
                arena_points=roi['points'],
                object_points=roi['objects'],
                detection_points=roi.get('detection_points'),
                arena_width_cm=arena_width_cm,
                arena_height_cm=arena_height_cm,
                fps=fps,
                roi_id=roi['roi_id'],
                roi_name=roi['name'],
                phase=phase,
                time_config=roi.get('time_config'),
                global_start_time_s=global_start_time_s,
                center_ratio_x=float(vision_config.get('center_ratio_x', 0.5)),
                center_ratio_y=float(vision_config.get('center_ratio_y', 0.5)),
                interaction_radius_cm=float(vision_config.get('interaction_radius_cm', 4.0)),
                exploration_max_speed_cm_s=float(vision_config.get('exploration_max_speed_cm_s', 12.0)),
                use_head_direction=bool(vision_config.get('use_head_direction', True)),
                head_angle_threshold_deg=float(vision_config.get('head_angle_threshold_deg', 50.0)),
                head_confidence_threshold=float(vision_config.get('head_confidence_threshold', 0.35)),
                pose_head_hold_s=float(vision_config.get('pose_head_hold_s', 0.12)),
                pose_head_smoothing=float(vision_config.get('pose_head_smoothing', 0.35)),
                body_gray_threshold=int(vision_config.get('body_gray_threshold', 95)),
                mouse_is_light=bool(vision_config.get('invert', False)),
                movement_threshold_cm_s=float(vision_config.get('movement_threshold_cm_s', 2.0)),
                min_valid_exploration_s=float(vision_config.get('min_valid_exploration_s', 20.0)),
            )
            for roi in roi_configs
        ]

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
        if vision_config.get('tracking_backend') != vision_config.get('tracking_backend_effective', 'classical'):
            self._log(
                "NOR tracking backend fallback: "
                f"{vision_config.get('tracking_backend')} -> classical "
                f"({vision_config.get('tracking_fallback_reason') or 'unavailable'})."
            )
        result_folder = vision_config.get('output_dir') or str(RESULT_DIR)
        output_info = _prepare_output_paths(
            self.session_id,
            video_info.get('original_name', 'video'),
            suffix='nor',
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
        self._log(f"NOR video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")
        timebase, timebase_summary, total_frames = self._build_timebase_for_video(
            video_path,
            fps,
            total_frames,
            label='NOR',
            prefer_pts=not bool(fps_override),
        )
        sampling = self._resolve_video_sampling(
            width,
            height,
            fps,
            vision_config,
            label='NOR',
            temporal_default=True,
            max_dim_default=960,
            target_fps_default=30.0,
        )
        self._log_video_sampling(sampling)
        if width * height >= 1920 * 1080 and not sampling.get('active'):
            self._log("High-resolution NOR input detected; segmentation, contours, and video encoding will be slower.")

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        start_frame, end_frame = self._time_range_to_frames(timebase, total_frames, start_sec, end_sec)
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0

        requested_tracking_backend = str(
            vision_config.get('tracking_backend') or 'classical'
        ).strip().lower()
        pose_backend_active = (
            str(vision_config.get('tracking_backend_effective') or 'classical')
            == 'deeplabcut'
        )
        strict_pose_only = (
            pose_backend_active and requested_tracking_backend == 'deeplabcut'
        )
        allow_classical_fallback = (
            pose_backend_active and requested_tracking_backend == 'auto'
        )
        missing_tracking_source = (
            'classical_fallback'
            if allow_classical_fallback
            else ('deeplabcut' if pose_backend_active else 'classical')
        )
        if strict_pose_only:
            bg_subtractor = morph_processor = detector = None
        else:
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
        requested_pose_batch_size = pose_coordinator.batch_size
        pose_cache = {}
        if pose_coordinator.enabled:
            self._log(
                "NOR DeepLabCut tracking enabled: "
                f"batch size {requested_pose_batch_size}, two-batch prefetch; "
                "body/nose/ears drive position, head direction, and exploration scoring."
            )
            if strict_pose_only:
                self._log(
                    "NOR strict DeepLabCut mode: grayscale segmentation and "
                    "PCA head-direction fallback are disabled."
                )
        self._log(f"Configured {len(trackers)} NOR arenas")

        if generate_video:
            self._log("NOR annotated video is rendered after scoring; CSV and statistics export independently of video encoding.")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start NOR analysis from frame {start_frame} ({timebase.time_at(start_frame):.1f}s)")

        frames_to_process = max(0, end_frame - start_frame + 1)
        frame_idx = 0
        source_frames_done = 0
        previous_analyzed_frame = None

        logged_effective_batch_size = None
        for batch in self._iter_prefetched_frame_batches(
            cap,
            start_frame=start_frame,
            end_frame=end_frame,
            sampling=sampling,
            batch_size=requested_pose_batch_size,
            prepare_gray=not strict_pose_only,
        ):
            if not self.is_running:
                break
            pose_frames = pose_coordinator.infer_many(
                [item['frame'] for item in batch]
            )
            if (
                pose_coordinator.enabled
                and pose_coordinator.effective_batch_size != logged_effective_batch_size
            ):
                logged_effective_batch_size = pose_coordinator.effective_batch_size
                self._log(
                    "NOR DeepLabCut effective batch size: "
                    f"{logged_effective_batch_size} "
                    f"(requested {requested_pose_batch_size})."
                )
            for item, pose_frame in zip(batch, pose_frames):
                current_frame_no = int(item['source_frame_no'])
                analysis_frame = item['frame']
                gray = item['gray']
                current_time = timebase.time_at(current_frame_no)
                current_dt = timebase.duration_between(
                    previous_analyzed_frame,
                    current_frame_no,
                    sample_fps=sampling['effective_fps'],
                )
                if bg_subtractor is not None:
                    begin_frame_cache(bg_subtractor, current_frame_no)
                pose_cache[current_frame_no] = pose_frame
                results = []
                for tracker in trackers:
                    pose_observation = pose_frame.by_roi_id.get(int(tracker.roi_id))
                    results.append(
                        tracker.update(
                            analysis_frame,
                            current_time,
                            gray,
                            bg_subtractor,
                            morph_processor,
                            detector,
                            include_trajectory=False,
                            dt_s=current_dt,
                            pose_observation=pose_observation,
                            tracking_source=missing_tracking_source,
                            fallback_reason=(
                                pose_frame.error
                                or vision_config.get('tracking_fallback_reason')
                                or ('pose_low_confidence_or_unassigned' if pose_backend_active and pose_observation is None else None)
                            ) if pose_observation is None else None,
                            allow_classical_fallback=allow_classical_fallback or not pose_backend_active,
                            strict_pose_only=strict_pose_only,
                        )
                    )
                previous_analyzed_frame = current_frame_no
                for result in results:
                    self._annotate_sample_record(
                        result,
                        current_frame_no,
                        frame_idx,
                        sampling,
                    )

                if frame_idx % 15 == 0:
                    rows = [self._live_row(tracker, result) for tracker, result in zip(trackers, results)]
                    self.realtime_data = rows
                    self.live_metrics = self._build_live_metrics(rows)

                if frame_idx % 30 == 0:
                    self.current_frame = analysis_frame

                frame_idx += 1
                covered = int(item['covered_source_frames'])
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
            label='NOR',
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
            summary['pose_batch_size_requested'] = requested_pose_batch_size
            summary['pose_batch_size_effective'] = pose_coordinator.effective_batch_size
            summary['tracking_fallback_reason'] = vision_config.get('tracking_fallback_reason')
            dlc_frames = int(summary.get('deeplabcut_frames') or 0)
            fallback_frames = int(summary.get('classical_fallback_frames') or 0)
            missing_dlc_frames = int(summary.get('deeplabcut_missing_frames') or 0)
            summary['tracking_backend_observed'] = (
                'mixed' if dlc_frames and fallback_frames
                else ('deeplabcut' if dlc_frames or missing_dlc_frames else 'classical')
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
        exploration_cumulative_work_paths = []
        exploration_cumulative_final_paths = []
        visual_style = vision_config.get('visual_style')
        for tracker in trackers:
            roi_slug = sanitize_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"roi_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            exploration_cumulative_work_path = f"{output_base}_{roi_slug}_exploration_cumulative.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"
            exploration_cumulative_final_path = f"{final_base}_{roi_slug}_exploration_cumulative.png"

            if render_novel_object_heatmap(tracker, heatmap_work_path, style=visual_style):
                exported_heatmap = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported_heatmap)
                self._append_result_file(exported_heatmap)
            else:
                self._log(f"Failed to save NOR heatmap: {heatmap_final_path}")

            if render_novel_object_trackmap(tracker, trackmap_work_path, style=visual_style):
                exported_trackmap = _export_result_file(trackmap_work_path, trackmap_final_path)
                trackmap_work_paths.append(trackmap_work_path)
                trackmap_final_paths.append(exported_trackmap)
                self._append_result_file(exported_trackmap)
            else:
                self._log(f"Failed to save NOR trackmap: {trackmap_final_path}")

            if render_novel_object_exploration_cumulative(
                tracker,
                exploration_cumulative_work_path,
                style=visual_style,
            ):
                exported_cumulative = _export_result_file(
                    exploration_cumulative_work_path,
                    exploration_cumulative_final_path,
                )
                exploration_cumulative_work_paths.append(exploration_cumulative_work_path)
                exploration_cumulative_final_paths.append(exported_cumulative)
                self._append_result_file(exported_cumulative)
            else:
                self._log(f"Failed to save NOR cumulative exploration chart: {exploration_cumulative_final_path}")

        if heatmap_work_paths:
            if len(heatmap_final_paths) == 1:
                self._append_result_file(heatmap_final_paths[0], 'heatmap')
            else:
                heatmap_grid_work_path = f"{output_base}_heatmaps.png"
                heatmap_grid_final_path = f"{final_base}_heatmaps.png"
                if render_open_field_image_grid(heatmap_work_paths, heatmap_grid_work_path, title=f"{video_name} | NOR Heatmaps", style=visual_style):
                    self._append_result_file(_export_result_file(heatmap_grid_work_path, heatmap_grid_final_path), 'heatmap')
                else:
                    self._log(f"Failed to save NOR heatmap grid: {heatmap_grid_final_path}")

        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                trackmap_grid_work_path = f"{output_base}_trackmaps.png"
                trackmap_grid_final_path = f"{final_base}_trackmaps.png"
                if render_open_field_image_grid(trackmap_work_paths, trackmap_grid_work_path, title=f"{video_name} | NOR Trajectory Maps", style=visual_style):
                    self._append_result_file(_export_result_file(trackmap_grid_work_path, trackmap_grid_final_path), 'trackmap')
                else:
                    self._log(f"Failed to save NOR trackmap grid: {trackmap_grid_final_path}")

        if exploration_cumulative_work_paths:
            if len(exploration_cumulative_final_paths) == 1:
                self._append_result_file(
                    exploration_cumulative_final_paths[0],
                    'exploration_cumulative',
                )
            else:
                cumulative_grid_work_path = f"{output_base}_exploration_cumulative_grid.png"
                cumulative_grid_final_path = f"{final_base}_exploration_cumulative_grid.png"
                if render_open_field_image_grid(
                    exploration_cumulative_work_paths,
                    cumulative_grid_work_path,
                    title=f"{video_name} | NOR Cumulative Object Exploration",
                    style=visual_style,
                ):
                    self._append_result_file(
                        _export_result_file(cumulative_grid_work_path, cumulative_grid_final_path),
                        'exploration_cumulative',
                    )
                else:
                    self._log(f"Failed to save NOR cumulative exploration grid: {cumulative_grid_final_path}")

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
                pose_cache,
                batch_index=batch_index,
            )
        pose_cache.clear()

        if generate_video and os.path.exists(output_video_path):
            exported_video = _export_result_file(output_video_path, final_output_video_path)
            self._append_result_file(exported_video, 'video')
            if exported_video:
                self._log(f"NOR annotated video saved: {exported_video}")

        self.realtime_data = [
            {
                'roi_id': row['roi_id'],
                'roi_name': row['roi_name'],
                'detected': True,
                'current_object': 'Completed',
                'current_role': '-',
                'distance_cm': row['total_distance_cm'],
                'total_exploration_s': row['total_object_exploration_s'],
                'novel_exploration_s': row.get('novel_exploration_s'),
                'familiar_exploration_s': row.get('familiar_exploration_s'),
                'discrimination_index': row.get('discrimination_index'),
                'preference_index_percent': row.get('preference_index_percent'),
                'meets_min_exploration': bool(row.get('meets_min_exploration', False)),
                'in_stats_window': True,
            }
            for row in summary_rows
        ]
        self.live_metrics = self._build_live_metrics(self.realtime_data)

        if batch_index is None and self.is_running:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = 'Novel object recognition analysis completed.'

        elapsed = time.time() - video_start_time
        self._log(f"NOR video processed in {elapsed:.1f}s, {source_frames_done} source frames ({frame_idx} analyzed samples)")

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
        pose_cache,
        batch_index=None,
    ):
        cap = open_video_capture(video_path, orientation_override=vision_config.get('orientation_override'))
        if not cap.isOpened():
            self._log(f"Unable to reopen NOR video for rendering: {video_path}")
            return False

        writer = create_annotated_video_writer(
            output_video_path,
            fps,
            width,
            height,
            label='NOR video',
            source_video_path=video_path,
            logger=self._log,
            async_write=True,
        )
        if writer is None:
            cap.release()
            self._log("NOR annotated video disabled because no writer is available.")
            return False

        requested_tracking_backend = str(
            vision_config.get('tracking_backend') or 'classical'
        ).strip().lower()
        pose_backend_active = (
            str(vision_config.get('tracking_backend_effective') or 'classical')
            == 'deeplabcut'
        )
        strict_pose_only = (
            pose_backend_active and requested_tracking_backend == 'deeplabcut'
        )
        allow_classical_fallback = (
            pose_backend_active and requested_tracking_backend == 'auto'
        )
        missing_tracking_source = (
            'classical_fallback'
            if allow_classical_fallback
            else ('deeplabcut' if pose_backend_active else 'classical')
        )
        if strict_pose_only:
            bg_subtractor = morph_processor = detector = None
        else:
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
                gray = (
                    None
                    if strict_pose_only
                    else cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
                )
                if bg_subtractor is not None:
                    begin_frame_cache(bg_subtractor, current_frame_no)
                pose_frame = pose_cache.get(
                    current_frame_no,
                    PoseFrameResult({}, error='pose_cache_miss'),
                )
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
                            include_trajectory=True,
                            dt_s=current_dt,
                            pose_observation=pose_observation,
                            tracking_source=missing_tracking_source,
                            fallback_reason=(
                                pose_frame.error
                                or vision_config.get('tracking_fallback_reason')
                                or ('pose_low_confidence_or_unassigned' if pose_backend_active and pose_observation is None else None)
                            ) if pose_observation is None else None,
                            allow_classical_fallback=allow_classical_fallback or not pose_backend_active,
                            strict_pose_only=strict_pose_only,
                        )
                    )
                previous_analyzed_frame = current_frame_no
                result_frame = draw_novel_object_results(analysis_frame, list(zip(render_trackers, results)))
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
                        f'Rendering NOR video {source_frames_done}/{frames_to_process} ({{progress}}%)',
                    )
        finally:
            cap.release()
            writer.release()
        return os.path.exists(output_video_path)

    def _generate_txt_report(self, summaries, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 72 + "\n")
                f.write("MouseScope 新奇物体识别报告 (Novel Object Recognition Report)\n")
                f.write("=" * 72 + "\n\n")
                f.write(f"{_bi_label('分析时间', 'Analysis Time')}: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if summaries:
                    f.write(f"{_bi_label('视频', 'Video')}: {summaries[0].get('video_name', '')}\n")
                    f.write(f"{_bi_label('实验阶段', 'Phase')}: {_report_phase_label(summaries[0].get('phase', 'recognition'))}\n")
                f.write(f"{_bi_label('ROI 数量', 'ROI Count')}: {len(summaries)}\n")
                f.write(
                    f"{_bi_label('统计规则', 'Scoring Rule')}: "
                    "探索判定优先使用轮廓接近与头部朝向门控；当鼻尖估计置信度不足时回退到接近规则 "
                    "(Exploration uses contour proximity with head-direction gating when the nose estimate is confident; "
                    "low-confidence frames fall back to proximity.)\n\n"
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
                        ('总距离', 'Total Distance', 'total_distance_cm', ' cm'),
                        ('平均速度', 'Mean Speed', 'mean_speed_cm_s', ' cm/s'),
                        ('中心区时间', 'Center Time', 'center_time_s', ' s'),
                        ('中心区比例', 'Center Percent', 'center_percent', '%'),
                        ('物体 1 探索时间', 'Object 1 Exploration', 'object_1_exploration_s', ' s'),
                        ('物体 2 探索时间', 'Object 2 Exploration', 'object_2_exploration_s', ' s'),
                        ('总探索时间', 'Total Exploration', 'total_object_exploration_s', ' s'),
                        ('头部支持探索时间', 'Head-Supported Exploration', 'head_supported_exploration_s', ' s'),
                        ('回退探索时间', 'Fallback Exploration', 'fallback_exploration_s', ' s'),
                        ('新奇物体探索时间', 'Novel Exploration', 'novel_exploration_s', ' s'),
                        ('熟悉物体探索时间', 'Familiar Exploration', 'familiar_exploration_s', ' s'),
                        ('辨别指数', 'Discrimination Index', 'discrimination_index', ''),
                        ('偏好指数', 'Preference Index', 'preference_index_percent', '%'),
                        ('头部估计率', 'Head Estimated Rate', 'head_estimated_rate_percent', '%'),
                        ('头部高置信率', 'Head Confident Rate', 'head_confident_rate_percent', '%'),
                        ('头部角度阈值', 'Head Angle Threshold', 'head_angle_threshold_deg', ' deg'),
                        ('第二灰度阈值', 'Body/Tail Threshold', 'body_gray_threshold', ''),
                    ])
                    f.write("\n")
        except Exception as e:
            print(f"Failed to write NOR report: {e}")
