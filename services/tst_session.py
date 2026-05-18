import math

from services.analysis_runtime import *
from services.video_timebase import build_video_timebase
from tst_legacy import TSTLegacyMouseTracker, TSTLegacyOpticalFlowAnalyzer
from tst_lis import TSTLISConfig, TSTLISFeatureTracker, apply_tst_lis_labels


class VisionAnalysisSession(AnalysisSessionBase):
    """视觉分析会话（无数据库、无YOLO）"""
    batch_progress_template = "正在处理视频 {index}/{total}: {name}"
    batch_complete_template = "全部 {total} 个视频分析完成！"
    error_template = "分析出错: {error}"
    error_log_prefix = "❌ "

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths  # list of paths
        self.video_infos = video_infos  # list of {path, original_name, ...}
        self.config = config
        self.experiment_type = 'tst'

        self.status = 'waiting'
        self.progress = 0
        self.message = '等待开始...'
        self.logs = []
        self.realtime_data = []
        self.result_files = []
        self.processed_frames = 0
        self.is_running = False
        self.start_time = None
        self.end_time = None
        self.current_frame = None
        self.csv_path = None
        self.immobility_csv_path = None
        self.immobility_txt_path = None
        self.activity_curve_path = None
        self.ethogram_swing_path = None
        self.ethogram_merged_path = None

    @staticmethod
    def _point_xy(point):
        if isinstance(point, dict):
            return [float(point['x']), float(point['y'])]
        return [float(point[0]), float(point[1])]

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
            static_frames=10,
            erosion_default=2,
            dilation_default=3,
            min_area_default=100,
        )

    def _parse_rois(self, vision_config):
        rois_config = vision_config.get('rois', [])
        if isinstance(rois_config, str):
            try:
                rois_config = json.loads(rois_config)
            except Exception:
                rois_config = []

        roi_configs = []
        for roi_idx, roi_item in enumerate(rois_config):
            if isinstance(roi_item, str):
                try:
                    roi_item = json.loads(roi_item)
                except Exception:
                    continue

            if isinstance(roi_item, dict) and 'points' in roi_item:
                roi_points = roi_item['points']
                roi_name = roi_item.get('name', f"ROI {roi_idx + 1}")
                time_config = roi_item.get('time_config')
            else:
                roi_points = roi_item
                roi_name = f"ROI {roi_idx + 1}"
                time_config = None

            if isinstance(roi_points, str):
                try:
                    roi_points = json.loads(roi_points)
                except Exception:
                    continue
            if not isinstance(roi_points, list) or len(roi_points) < 3:
                continue
            try:
                roi_configs.append({
                    'roi_id': roi_idx + 1,
                    'name': roi_name,
                    'points': [self._point_xy(point) for point in roi_points],
                    'time_config': time_config,
                })
            except Exception:
                continue
        return roi_configs

    def _build_legacy_trackers(self, roi_configs, vision_config, fps, pixel_to_mm, processors, threshold_overrides=None):
        bg_subtractor, morph_processor, detector = processors
        threshold_overrides = threshold_overrides or {}
        trackers = []
        for roi in roi_configs:
            thresholds = threshold_overrides.get(roi['roi_id'], {})
            flow_analyzer = TSTLegacyOpticalFlowAnalyzer(fps=fps, pixel_to_mm=pixel_to_mm)
            tracker = TSTLegacyMouseTracker(
                roi_polygon=roi['points'],
                roi_id=roi['roi_id'],
                bg_subtractor=bg_subtractor,
                morphology_processor=morph_processor,
                detector=detector,
                flow_analyzer=flow_analyzer,
                fps=fps,
                pixel_to_mm=pixel_to_mm,
                roi_name=roi['name'],
                time_config=roi.get('time_config'),
                win_seconds=0.3,
                flow_thr_hi=float(thresholds.get('flow_thr_hi', vision_config.get('flow_thr_hi', 50.0))),
                flow_thr_lo=float(thresholds.get('flow_thr_lo', vision_config.get('flow_thr_lo', 35.0))),
                speed_std_thr_hi=float(thresholds.get('speed_std_thr_hi', vision_config.get('speed_std_thr_hi', 40.0))),
                speed_std_thr_lo=float(thresholds.get('speed_std_thr_lo', vision_config.get('speed_std_thr_lo', 20.0))),
                min_active_seconds=0.2,
                calm_seconds=0.4,
                coherence_thr=0.8,
                coherence_calm_seconds=0.5,
                auto_behavior_threshold=bool(vision_config.get('auto_behavior_threshold', False)),
                behavior_threshold_sensitivity=float(vision_config.get('behavior_threshold_sensitivity', 0.0)),
            )
            if thresholds:
                tracker.collect_behavior_threshold_samples = False
            trackers.append(tracker)
        return trackers

    def _build_tst_lis_trackers(self, roi_configs, vision_config, fps, pixel_to_mm, processors):
        bg_subtractor, morph_processor, detector = processors
        lis_cfg = self._build_tst_lis_config(vision_config)
        trackers = []
        for roi in roi_configs:
            trackers.append(TSTLISFeatureTracker(
                roi_polygon=roi['points'],
                roi_id=roi['roi_id'],
                bg_subtractor=bg_subtractor,
                morphology_processor=morph_processor,
                detector=detector,
                fps=fps,
                pixel_to_mm=pixel_to_mm,
                roi_name=roi['name'],
                time_config=roi.get('time_config'),
                config=lis_cfg,
            ))
        return trackers

    def _estimate_behavior_thresholds(
        self,
        cap,
        video_path,
        roi_configs,
        vision_config,
        fps,
        pixel_to_mm,
        start_frame,
        end_frame,
        frames_to_process,
        sampling,
        timebase,
    ):
        self._log("TST auto behavior threshold: scanning the selected video window...")
        processors = self._build_processors(video_path, vision_config, sampling)
        trackers = self._build_legacy_trackers(roi_configs, vision_config, fps, pixel_to_mm, processors)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        sampled_frames = 0
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
            begin_frame_cache(processors[0], current_frame_no)
            for tracker in trackers:
                tracker.update(analysis_frame, current_time=current_time, gray=gray, dt_s=current_dt)
            previous_analyzed_frame = current_frame_no

            sampled_frames += 1
            skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
            covered = 1 + skipped
            source_frames_done += covered
            self._advance_frame_counters(covered, processed=False)
            if sampled_frames % 90 == 0:
                self.message = f'TST auto threshold scan {source_frames_done}/{frames_to_process}'

        threshold_overrides = {}
        for tracker in trackers:
            configured = tracker.auto_configure_behavior_thresholds()
            thresholds = tracker.get_behavior_thresholds()
            threshold_overrides[tracker.roi_id] = thresholds
            source = "auto" if configured else "fixed fallback"
            self._log(
                f"ROI {tracker.roi_id} ({tracker.roi_name}) TST thresholds [{source}]: "
                f"flow {thresholds['flow_thr_lo']:.2f}/{thresholds['flow_thr_hi']:.2f}, "
                f"speed-std {thresholds['speed_std_thr_lo']:.2f}/{thresholds['speed_std_thr_hi']:.2f}"
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        return threshold_overrides

    @staticmethod
    def _build_tst_lis_config(vision_config):
        return TSTLISConfig(
            use_rotation_alignment=bool(vision_config.get('tst_lis_use_rotation_alignment', False)),
            immobile_confirm_s=float(vision_config.get('tst_lis_immobile_confirm_s', 0.75)),
            active_confirm_s=float(vision_config.get('tst_lis_active_confirm_s', 0.35)),
            min_immobile_bout_s=float(vision_config.get('tst_lis_min_immobile_bout_s', 0.50)),
            min_struggle_bout_s=float(vision_config.get('tst_lis_min_struggle_bout_s', 0.20)),
            merge_gap_s=float(vision_config.get('tst_lis_merge_gap_s', 0.30)),
            active_entry_hit_ratio=float(vision_config.get('tst_lis_active_entry_hit_ratio', 0.45)),
            active_release_hit_ratio=float(vision_config.get('tst_lis_active_release_hit_ratio', 0.20)),
            immobile_entry_hit_ratio=float(vision_config.get('tst_lis_immobile_entry_hit_ratio', 0.55)),
            score_mode_prior_min=float(vision_config.get('tst_lis_score_mode_prior_min', 0.35)),
            score_mode_entry=float(vision_config.get('tst_lis_score_mode_entry', 0.50)),
            score_mode_low_percentile=float(vision_config.get('tst_lis_score_mode_low_percentile', 25.0)),
            score_mode_high_percentile=float(vision_config.get('tst_lis_score_mode_high_percentile', 90.0)),
            score_mode_fill_gap_s=float(vision_config.get('tst_lis_score_mode_fill_gap_s', 2.0)),
            appendage_rescue_enabled=bool(vision_config.get('tst_lis_appendage_rescue_enabled', True)),
            appendage_rescue_prior_min=float(vision_config.get('tst_lis_appendage_rescue_prior_min', 0.28)),
            appendage_rescue_prior_max=float(vision_config.get('tst_lis_appendage_rescue_prior_max', 0.35)),
            appendage_rescue_aspect_min=float(vision_config.get('tst_lis_appendage_rescue_aspect_min', 1.55)),
            appendage_rescue_score_entry=float(vision_config.get('tst_lis_appendage_rescue_score_entry', 0.45)),
            appendage_rescue_flow_min=float(vision_config.get('tst_lis_appendage_rescue_flow_min', 20.0)),
            appendage_rescue_speed_std_max=float(vision_config.get('tst_lis_appendage_rescue_speed_std_max', 6.0)),
            appendage_rescue_front_score_entry=float(vision_config.get('tst_lis_appendage_rescue_front_score_entry', 0.32)),
            appendage_rescue_front_residual_min=float(vision_config.get('tst_lis_appendage_rescue_front_residual_min', 0.026)),
            appendage_rescue_front_peripheral_min=float(vision_config.get('tst_lis_appendage_rescue_front_peripheral_min', 0.010)),
            appendage_rescue_front_highfreq_min=float(vision_config.get('tst_lis_appendage_rescue_front_highfreq_min', 0.0060)),
            appendage_rescue_front_flow_min=float(vision_config.get('tst_lis_appendage_rescue_front_flow_min', 15.0)),
            appendage_rescue_front_periodicity_min=float(vision_config.get('tst_lis_appendage_rescue_front_periodicity_min', 0.30)),
            appendage_rescue_front_global_max=float(vision_config.get('tst_lis_appendage_rescue_front_global_max', 0.012)),
            appendage_rescue_front_aspect_min=float(vision_config.get('tst_lis_appendage_rescue_front_aspect_min', 1.10)),
            appendage_rescue_front_speed_std_max=float(vision_config.get('tst_lis_appendage_rescue_front_speed_std_max', 6.5)),
            appendage_rescue_periodic_aspect_min=float(vision_config.get('tst_lis_appendage_rescue_periodic_aspect_min', 1.70)),
            appendage_rescue_periodic_score_entry=float(vision_config.get('tst_lis_appendage_rescue_periodic_score_entry', 0.45)),
            appendage_rescue_periodic_residual_min=float(vision_config.get('tst_lis_appendage_rescue_periodic_residual_min', 0.025)),
            appendage_rescue_periodic_peripheral_min=float(vision_config.get('tst_lis_appendage_rescue_periodic_peripheral_min', 0.016)),
            appendage_rescue_periodic_highfreq_min=float(vision_config.get('tst_lis_appendage_rescue_periodic_highfreq_min', 0.025)),
            appendage_rescue_periodic_flow_min=float(vision_config.get('tst_lis_appendage_rescue_periodic_flow_min', 14.0)),
            appendage_rescue_periodic_periodicity_min=float(vision_config.get('tst_lis_appendage_rescue_periodic_periodicity_min', 0.48)),
            appendage_rescue_periodic_global_max=float(vision_config.get('tst_lis_appendage_rescue_periodic_global_max', 0.010)),
            appendage_rescue_periodic_speed_std_max=float(vision_config.get('tst_lis_appendage_rescue_periodic_speed_std_max', 12.0)),
            appendage_rescue_fill_gap_s=float(vision_config.get('tst_lis_appendage_rescue_fill_gap_s', 1.0)),
            appendage_rescue_retro_s=float(vision_config.get('tst_lis_appendage_rescue_retro_s', 0.7)),
            appendage_rescue_hold_s=float(vision_config.get('tst_lis_appendage_rescue_hold_s', 0.3)),
        )

    @staticmethod
    def _copy_tst_lis_log_fields(log_entry, result):
        tst_lis = result.get('tst_lis') or {}
        for key in (
            'global_motion_norm',
            'aligned_mask_change',
            'residual_gray_motion',
            'residual_motion',
            'core_motion',
            'peripheral_motion',
            'peripheral_core_ratio',
            'shape_change',
            'highfreq_motion',
            'periodicity',
            'area_ratio',
            'body_length',
            'bbox_w',
            'bbox_h',
            'aspect_ratio',
            'major_axis_angle',
            'solidity',
            'extent',
            'centroid_x',
            'centroid_y',
        ):
            log_entry[f'lis_{key}'] = round(float(tst_lis.get(key, 0.0) or 0.0), 6)
        log_entry['lis_segmentation_bad'] = bool(tst_lis.get('segmentation_bad', False))
        log_entry['lis_climbing_suspected'] = bool(tst_lis.get('climbing_suspected', False))
        log_entry['lis_available'] = bool(tst_lis.get('tst_lis_available', False))
        return log_entry

    def _render_tst_lis_final_video(
        self,
        *,
        video_path,
        output_path,
        logs,
        roi_configs,
        vision_config,
        sampling,
        analysis_fps,
        analysis_pixel_to_mm,
        start_frame,
        end_frame,
        frames_to_process,
        timebase,
        batch_index=None,
    ):
        row_by_key = {}
        for row in logs or []:
            try:
                row_by_key[(int(row.get('frame')), int(row.get('roi_id')))] = row
            except Exception:
                continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to reopen video for optimized TST rendering: {video_path}")
            return False
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        writer = create_annotated_video_writer(
            output_path,
            analysis_fps,
            sampling['analysis_width'],
            sampling['analysis_height'],
            label='optimized TST video',
            source_video_path=video_path,
            logger=self._log,
        )
        if writer is None:
            cap.release()
            self._log("Annotated video disabled because no writer is available.")
            return False

        render_config = dict(vision_config)
        render_config['enable_tst_lis'] = True
        processors = self._build_processors(video_path, render_config, sampling)
        trackers = self._build_tst_lis_trackers(
            roi_configs,
            render_config,
            analysis_fps,
            analysis_pixel_to_mm,
            processors,
        )

        source_frames_done = 0
        frame_idx = 0
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
                    sample_fps=analysis_fps,
                )
                analysis_frame = self._resize_for_sampling(frame, sampling)
                gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
                begin_frame_cache(processors[0], current_frame_no)
                results = []
                for tracker in trackers:
                    result = tracker.update(analysis_frame, current_time=current_time, gray=gray, dt_s=current_dt)
                    row = row_by_key.get((current_frame_no, int(result.get('roi_id', 0))))
                    if row is not None:
                        result['is_struggling'] = self._bool_log_value(row.get('is_struggling'), False)
                        result['lis_is_struggling'] = self._bool_log_value(row.get('lis_is_struggling'), False)
                        result['primary_behavior_algorithm'] = 'tst_lis'
                        result['behavior_algorithm'] = 'tst_lis'
                        result['is_swinging'] = False
                        result['lis_reason'] = row.get('lis_reason')
                        result['lis_confidence'] = row.get('lis_confidence')
                        result['lis_passive_swing'] = self._bool_log_value(row.get('lis_passive_swing'), False)
                    results.append(result)
                previous_analyzed_frame = current_frame_no

                writer.write(draw_results(analysis_frame, results))
                frame_idx += 1
                skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
                covered = 1 + skipped
                source_frames_done += covered
                self._advance_frame_counters(covered, processed=False)
                if frame_idx % 30 == 0:
                    fraction = source_frames_done / max(frames_to_process, 1)
                    self._set_video_progress(
                        batch_index,
                        0.65 + 0.25 * fraction,
                        f'微小挣扎优化逻辑渲染最终视频 {source_frames_done}/{frames_to_process} ({{progress}}%)',
                    )
                if frame_idx % 10 == 0:
                    self.current_frame = analysis_frame
        finally:
            cap.release()
            writer.release()
        return True

    def _analyze_single_video_lis(self, video_path, video_info, batch_index=None, video_start_time=None):
        """Analyze TST with the optimized logic as the primary algorithm."""
        video_start_time = time.time() if video_start_time is None else video_start_time
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
        vision_config['use_tst_lis_primary'] = True
        vision_config['use_tst_legacy_primary'] = False
        vision_config['enable_tst_lis'] = True
        vision_config['auto_behavior_threshold'] = False

        result_folder = vision_config.get('output_dir') or str(RESULT_DIR)
        output_info = _prepare_output_paths(
            self.session_id,
            video_info.get('original_name', 'video'),
            suffix='tst',
            result_folder=result_folder,
            fallback='video',
            batch_index=video_index,
            batch_count=len(self.video_paths),
        )

        output_path = f"{output_info['work_base']}_analyzed.mp4"
        final_output_path = f"{output_info['final_base']}_analyzed.mp4"
        generate_video = bool(vision_config.get('generate_video', True))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"❌ 无法打开视频: {video_path}")
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps_override = vision_config.get('fps_override')
        video_meta = probe_video_info(video_path, fps_override=fps_override, cap=cap)
        detected_fps = float(video_meta.get('detected_fps') or 30.0)
        fps = float(video_meta.get('fps') or detected_fps or 30.0)
        width = int(video_meta.get('width') or 0)
        height = int(video_meta.get('height') or 0)
        total_frames = int(video_meta.get('total_frames') or 0)
        if fps_override:
            self._log(f"视频信息: {width}x{height}, {fps:.1f}fps (手动指定, 自动检测={detected_fps:.1f}), {total_frames}帧")
        else:
            self._log(f"视频信息: {width}x{height}, {fps:.1f}fps, {total_frames}帧")

        timebase = build_video_timebase(video_path, fps=fps, total_frames=total_frames, logger=self._log)
        if total_frames <= 0 and timebase.frame_count > 0:
            total_frames = timebase.frame_count
        timebase_summary = timebase.summary()
        self._log(
            "TST timebase: "
            f"{timebase_summary['source']}, "
            f"median dt {float(timebase_summary['median_dt']):.4f}s, "
            f"long gaps >50ms {int(timebase_summary['long_gap_gt_50ms'])}"
        )

        sampling = self._resolve_video_sampling(
            width,
            height,
            fps,
            vision_config,
            label='微小挣扎优化逻辑',
            temporal_default=False,
            max_dim_default=960,
            target_fps_default=30.0,
        )
        self._log_video_sampling(sampling)
        analysis_fps = sampling['effective_fps']
        pixel_to_mm = vision_config.get('pixel_to_mm', 1.0)
        analysis_pixel_to_mm = float(pixel_to_mm) / max(float(sampling.get('scale', 1.0)), 1e-6)

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        start_frame = timebase.frame_at_or_after(float(start_sec)) if start_sec is not None else 0
        end_frame = timebase.frame_at_or_before(float(end_sec)) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if end_frame < start_frame:
            end_frame = start_frame
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"从第 {start_frame} 帧开始分析 ({timebase.time_at(start_frame):.1f}秒)")
        frames_to_process = end_frame - start_frame + 1

        source_roi_configs = self._parse_rois(vision_config)
        roi_configs = self._scale_roi_configs_for_sampling(source_roi_configs, sampling)
        if not roi_configs:
            cap.release()
            self._log("❌ 没有有效的 ROI 配置")
            raise ValueError("No valid ROI configuration.")

        if bool(vision_config.get('auto_behavior_threshold', False)):
            self._log("当前默认逻辑使用按 ROI 自适应阈值；原本逻辑的自动阈值扫描已跳过。")
        processors = self._build_processors(video_path, vision_config, sampling)
        trackers = self._build_tst_lis_trackers(
            roi_configs,
            vision_config,
            analysis_fps,
            analysis_pixel_to_mm,
            processors,
        )
        if not trackers:
            cap.release()
            self._log("❌ 没有有效的 ROI 配置")
            raise ValueError("No valid ROI configuration.")
        self._log(f"微小挣扎优化逻辑跟踪器已就绪: {len(trackers)}")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        frame_idx = 0
        source_frames_done = 0
        previous_analyzed_frame = None
        logs = []
        self._log("第一遍：提取残差/外周/形变特征...")

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
                sample_fps=analysis_fps,
            )
            analysis_frame = self._resize_for_sampling(frame, sampling)
            gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
            begin_frame_cache(processors[0], current_frame_no)
            results = []
            for tracker in trackers:
                result = tracker.update(analysis_frame, current_time=current_time, gray=gray, dt_s=current_dt)
                results.append(result)
            previous_analyzed_frame = current_frame_no

            for result in results:
                if result.get('detected'):
                    log_entry = {
                        'frame': current_frame_no,
                        'time': round(current_time, 3),
                        'roi_id': result['roi_id'],
                        'roi_name': result.get('roi_name', ''),
                        'is_struggling': bool(result.get('is_struggling', False)),
                        'is_swinging': False,
                        'behavior_algorithm': 'tst_lis',
                        'is_in_stats_window': bool(result.get('is_in_stats_window', True)),
                        'dt_s': round(float(result.get('dt_s', current_dt) or current_dt), 6),
                    }
                    if result.get('flow'):
                        log_entry['flow_p90_mm_s'] = round(result['flow'].get('flow_p90_mm_s', 0), 2)
                    log_entry['decision_flow_p90_mm_s'] = round(float(result.get('flow_p90', 0.0) or 0.0), 2)
                    log_entry['speed_std_mm_s'] = round(float(result.get('speed_std', 0.0) or 0.0), 2)
                    log_entry['flow_entropy'] = round(float(result.get('flow_entropy', 0.0) or 0.0), 3)
                    log_entry['flow_over_speed'] = round(float(result.get('flow_over_speed', 0.0) or 0.0), 3)
                    log_entry['flow_over_speed_std'] = round(float(result.get('flow_over_speed_std', 0.0) or 0.0), 3)
                    log_entry['acf_peak'] = round(float(result.get('acf_peak', 0.0) or 0.0), 3)
                    swing_state = result.get('swing_state') or {}
                    log_entry['swing_score'] = round(float(swing_state.get('swing_score', 0.0) or 0.0), 3)
                    log_entry['struggle_score'] = round(float(swing_state.get('struggle_score', 0.0) or 0.0), 3)
                    log_entry['swing_margin'] = round(float(swing_state.get('swing_margin', 0.0) or 0.0), 3)
                    self._copy_tst_lis_log_fields(log_entry, result)
                    logs.append(log_entry)

            if frame_idx % 50 == 0:
                self.realtime_data = []
                for result in results:
                    if result.get('detected'):
                        self.realtime_data.append({
                            'roi_id': result['roi_id'],
                            'roi_name': result.get('roi_name', ''),
                            'is_struggling': bool(result.get('is_struggling', False)),
                            'behavior_algorithm': 'tst_lis',
                            'flow_value': result.get('flow', {}).get('flow_p90_mm_s', 0) if result.get('flow') else 0,
                        })

            frame_idx += 1
            skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
            covered = 1 + skipped
            source_frames_done += covered
            self._advance_frame_counters(covered)
            if frame_idx % 30 == 0:
                fraction = source_frames_done / max(frames_to_process, 1)
                self._set_video_progress(
                    batch_index,
                    0.65 * fraction,
                    f'微小挣扎优化逻辑特征提取 {source_frames_done}/{frames_to_process} ({{progress}}%)',
                )
            if frame_idx % 10 == 0:
                self.current_frame = analysis_frame

        cap.release()
        if not self.is_running:
            return
        self._ensure_analysis_frames_processed(
            video_path=video_path,
            label='TST',
            frame_idx=frame_idx,
            source_frames_done=source_frames_done,
        )

        valid_trackers = []
        for tracker in trackers:
            detection_rate = tracker.detected_count / max(frame_idx, 1)
            if detection_rate >= 0.01 or tracker.detected_count > 30:
                valid_trackers.append(tracker)
                self._log(f"ROI {tracker.roi_id} ({tracker.roi_name}): 检测 {tracker.detected_count}/{frame_idx} 帧 ({detection_rate*100:.1f}%) ✓")
        if not valid_trackers and trackers:
            valid_trackers = trackers

        lis_cfg = self._build_tst_lis_config(vision_config)
        thresholds_by_roi = apply_tst_lis_labels(logs, fps=analysis_fps, config=lis_cfg)
        for roi_id, thresholds in thresholds_by_roi.items():
            self._log(
                f"ROI {roi_id} 默认逻辑阈值: "
                f"residual {thresholds.get('residual_low', 0):.4f}/{thresholds.get('residual_high', 0):.4f}, "
                f"peripheral high {thresholds.get('peripheral_high', 0):.4f}, "
                f"shape high {thresholds.get('shape_high', 0):.4f}"
            )

        if generate_video and len(logs) > 0:
            self._log("第二遍：用最终标签渲染结果视频...")
            rendered = self._render_tst_lis_final_video(
                video_path=video_path,
                output_path=output_path,
                logs=logs,
                roi_configs=roi_configs,
                vision_config=vision_config,
                sampling=sampling,
                analysis_fps=analysis_fps,
                analysis_pixel_to_mm=analysis_pixel_to_mm,
                start_frame=start_frame,
                end_frame=end_frame,
                frames_to_process=frames_to_process,
                timebase=timebase,
                batch_index=batch_index,
            )
            if not rendered:
                generate_video = False

        if len(logs) > 0:
            activity_work_path = output_path.replace('.mp4', '_activity_curve.png')
            if render_tst_activity_curve(logs, valid_trackers, activity_work_path):
                activity_path = _export_result_file(activity_work_path, f"{output_info['final_base']}_activity_curve.png")
                if activity_path:
                    self.result_files.append(activity_path)
                    self.activity_curve_path = activity_path
                    self._log(f"TST activity curve saved: {activity_path}")
            else:
                self._log("Failed to save TST activity curve")

            ethogram_swing_work_path = output_path.replace('.mp4', '_ethogram_swing.png')
            if render_tst_ethogram(logs, valid_trackers, ethogram_swing_work_path, distinguish_swing=True):
                ethogram_swing_path = _export_result_file(
                    ethogram_swing_work_path,
                    f"{output_info['final_base']}_ethogram_swing.png",
                )
                if ethogram_swing_path:
                    self.result_files.append(ethogram_swing_path)
                    self.ethogram_swing_path = ethogram_swing_path
                    self._log(f"TST ethogram with swing saved: {ethogram_swing_path}")
            else:
                self._log("Failed to save TST ethogram with swing")

            ethogram_merged_work_path = output_path.replace('.mp4', '_ethogram_merged.png')
            if render_tst_ethogram(logs, valid_trackers, ethogram_merged_work_path, distinguish_swing=False):
                ethogram_merged_path = _export_result_file(
                    ethogram_merged_work_path,
                    f"{output_info['final_base']}_ethogram_merged.png",
                )
                if ethogram_merged_path:
                    self.result_files.append(ethogram_merged_path)
                    self.ethogram_merged_path = ethogram_merged_path
                    self._log(f"TST ethogram merged saved: {ethogram_merged_path}")
            else:
                self._log("Failed to save TST ethogram merged")

        if len(logs) > 0 and HAS_PANDAS:
            csv_work_path = output_path.replace('.mp4', '_logs.csv')
            if _write_dataframe_csv(logs, csv_work_path):
                csv_path = _export_result_file(csv_work_path, f"{output_info['final_base']}_logs.csv")
                if csv_path:
                    self.result_files.append(csv_path)
                    self.csv_path = csv_path
                    self._log(f"日志已保存: {csv_path}")

        immobility_summaries = []
        for tracker in valid_trackers:
            summary = tracker.get_immobility_summary(analysis_fps)
            if summary:
                self._apply_tst_lis_primary_summary(summary, logs)
                summary['timebase_source'] = timebase_summary['source']
                summary['timebase_median_dt_s'] = round(float(timebase_summary['median_dt']), 6)
                summary['timebase_long_gap_gt_50ms'] = int(timebase_summary['long_gap_gt_50ms'])
                self._add_sampling_summary_fields(summary, sampling)
                immobility_summaries.append(summary)

        if len(immobility_summaries) > 0 and HAS_PANDAS:
            immobility_csv_work = output_path.replace('.mp4', '_immobility.csv')
            if _write_dataframe_csv(immobility_summaries, immobility_csv_work):
                immobility_csv = _export_result_file(
                    immobility_csv_work,
                    f"{output_info['final_base']}_immobility.csv",
                )
                if immobility_csv:
                    self.result_files.append(immobility_csv)
                    self.immobility_csv_path = immobility_csv

            txt_work_path = output_path.replace('.mp4', '_report.txt')
            self._generate_txt_report(immobility_summaries, txt_work_path)
            txt_path = _export_result_file(txt_work_path, f"{output_info['final_base']}_report.txt")
            if txt_path:
                self.result_files.append(txt_path)
                self.immobility_txt_path = txt_path

        if generate_video and os.path.exists(output_path):
            exported_video = _export_result_file(output_path, final_output_path)
            if exported_video:
                file_size = os.path.getsize(exported_video) / (1024 * 1024)
                self._log(f"分析视频已保存: {exported_video} ({file_size:.1f} MB)")
                self.result_files.append(exported_video)

        if batch_index is None and self.is_running:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = '分析完成！'

        elapsed = time.time() - video_start_time
        self._log(f"微小挣扎优化逻辑视频处理完成，耗时: {elapsed:.1f}秒，处理 {source_frames_done} source frames ({frame_idx} analyzed samples)")

        import gc
        gc.collect()

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        """分析单个视频"""
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
        use_tst_lis_primary = (
            not bool(vision_config.get('use_tst_legacy_primary', False))
            and bool(vision_config.get('use_tst_lis_primary', True))
        )
        if use_tst_lis_primary:
            return self._analyze_single_video_lis(video_path, video_info, batch_index=batch_index, video_start_time=video_start_time)

        # 结果文件夹（优先使用自定义输出目录）
        result_folder = vision_config.get('output_dir') or str(RESULT_DIR)
        output_info = _prepare_output_paths(
            self.session_id,
            video_info.get('original_name', 'video'),
            suffix='tst',
            result_folder=result_folder,
            fallback='video',
            batch_index=video_index,
            batch_count=len(self.video_paths),
        )

        video_name = output_info['display_stem']
        output_path = f"{output_info['work_base']}_analyzed.mp4"
        final_output_path = f"{output_info['final_base']}_analyzed.mp4"

        generate_video = vision_config.get('generate_video', True)

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"❌ 无法打开视频: {video_path}")
            raise RuntimeError(f"Unable to open video: {video_path}")

        fps_override = vision_config.get('fps_override')
        video_meta = probe_video_info(video_path, fps_override=fps_override, cap=cap)
        detected_fps = float(video_meta.get('detected_fps') or 30.0)
        fps = float(video_meta.get('fps') or detected_fps or 30.0)
        width = int(video_meta.get('width') or 0)
        height = int(video_meta.get('height') or 0)
        total_frames = int(video_meta.get('total_frames') or 0)

        if fps_override:
            self._log(f"视频信息: {width}x{height}, {fps:.1f}fps (手动指定, 自动检测={detected_fps:.1f}), {total_frames}帧")
        else:
            self._log(f"视频信息: {width}x{height}, {fps:.1f}fps, {total_frames}帧")

        timebase = build_video_timebase(video_path, fps=fps, total_frames=total_frames, logger=self._log)
        if total_frames <= 0 and timebase.frame_count > 0:
            total_frames = timebase.frame_count
        timebase_summary = timebase.summary()
        self._log(
            "TST timebase: "
            f"{timebase_summary['source']}, "
            f"median dt {float(timebase_summary['median_dt']):.4f}s, "
            f"long gaps >50ms {int(timebase_summary['long_gap_gt_50ms'])}"
        )

        sampling = self._resolve_video_sampling(
            width,
            height,
            fps,
            vision_config,
            label='TST',
            temporal_default=False,
            max_dim_default=960,
            target_fps_default=30.0,
        )
        self._log_video_sampling(sampling)
        analysis_fps = sampling['effective_fps']

        pixel_to_mm = vision_config.get('pixel_to_mm', 1.0)
        analysis_pixel_to_mm = float(pixel_to_mm) / max(float(sampling.get('scale', 1.0)), 1e-6)

        # 视频编码
        writer = None
        if generate_video:
            writer = create_annotated_video_writer(
                output_path,
                analysis_fps,
                sampling['analysis_width'],
                sampling['analysis_height'],
                label='TST video',
                source_video_path=video_path,
                logger=self._log,
            )
            if writer is None:
                generate_video = False
                self._log("TST annotated video disabled because no writer is available.")

        # 时间范围
        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        start_frame = timebase.frame_at_or_after(float(start_sec)) if start_sec is not None else 0
        end_frame = timebase.frame_at_or_before(float(end_sec)) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if end_frame < start_frame:
            end_frame = start_frame

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"从第 {start_frame} 帧开始分析 ({timebase.time_at(start_frame):.1f}秒)")

        frames_to_process = end_frame - start_frame + 1

        source_roi_configs = self._parse_rois(vision_config)
        roi_configs = self._scale_roi_configs_for_sampling(source_roi_configs, sampling)
        if not roi_configs:
            self._log("❌ 没有有效的 ROI 配置")
            cap.release()
            if writer is not None:
                writer.release()
            raise ValueError("No valid ROI configuration.")

        threshold_overrides = {}
        if bool(vision_config.get('auto_behavior_threshold', False)):
            threshold_overrides = self._estimate_behavior_thresholds(
                cap,
                video_path,
                roi_configs,
                vision_config,
                analysis_fps,
                analysis_pixel_to_mm,
                start_frame,
                end_frame,
                frames_to_process,
                sampling,
                timebase,
            )
            if not self.is_running:
                cap.release()
                if writer is not None:
                    writer.release()
                return
            cap.release()
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self._log(f"Unable to reopen video after threshold scan: {video_path}")
                if writer is not None:
                    writer.release()
                raise RuntimeError(f"Unable to reopen video after threshold scan: {video_path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            self._log("TST auto behavior threshold disabled; using fixed thresholds.")

        processors = self._build_processors(video_path, vision_config, sampling)
        trackers = self._build_legacy_trackers(
            roi_configs,
            vision_config,
            analysis_fps,
            analysis_pixel_to_mm,
            processors,
            threshold_overrides,
        )
        if not trackers:
            self._log("❌ 没有有效的 ROI 配置")
            cap.release()
            if writer is not None:
                writer.release()
            raise ValueError("No valid ROI configuration.")
        self._log(f"TST final trackers ready: {len(trackers)}")

        # 优化视频读取缓冲区大小
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

        # 异步视频写入（独立线程，避免编码阻塞分析）
        from queue import Queue
        write_queue = Queue(maxsize=120)  # 最多缓存120帧
        write_done = threading.Event()

        def _video_writer_thread():
            """后台线程：从队列取帧并写入视频文件"""
            while True:
                item = write_queue.get()
                if item is None:  # 结束信号
                    break
                if writer is not None:
                    if isinstance(item, tuple):
                        frame_to_draw, results_to_draw = item
                        item = draw_results(frame_to_draw, results_to_draw)
                    writer.write(item)
            write_done.set()

        writer_thread = None
        if generate_video and writer is not None:
            writer_thread = threading.Thread(target=_video_writer_thread, daemon=True)
            writer_thread.start()

        # 视频写入缓冲区（用于摆动状态回溯修正）
        buffer_seconds = 2.0
        buffer_size = int(analysis_fps * buffer_seconds)
        video_buffer = deque(maxlen=buffer_size + 10)
        prev_swing_states = {}

        frame_idx = 0
        source_frames_done = 0
        previous_analyzed_frame = None
        logs = []

        self._log(f"开始分析... (CPU核心: {os.cpu_count()}, OpenCV线程: {cv2.getNumThreads()})")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except:
                current_frame_no = start_frame + source_frames_done

            if end_frame is not None and current_frame_no > end_frame:
                break

            current_time = timebase.time_at(current_frame_no)
            current_dt = timebase.duration_between(
                previous_analyzed_frame,
                current_frame_no,
                sample_fps=analysis_fps,
            )
            analysis_frame = self._resize_for_sampling(frame, sampling)
            gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)

            # 追踪每个 ROI
            begin_frame_cache(processors[0], current_frame_no)
            results = []
            for tracker in trackers:
                result = tracker.update(analysis_frame, current_time=current_time, gray=gray, dt_s=current_dt)
                results.append(result)
            previous_analyzed_frame = current_frame_no

            # 记录日志
            for result in results:
                if result.get('detected'):
                    log_entry = {
                        'frame': current_frame_no,
                        'time': round(current_time, 3),
                        'roi_id': result['roi_id'],
                        'roi_name': result.get('roi_name', ''),
                        'is_struggling': result.get('is_struggling', False),
                        'legacy_is_struggling': result.get('is_struggling', False),
                        'is_swinging': result.get('is_swinging', False),
                        'behavior_algorithm': 'legacy',
                        'is_in_stats_window': bool(result.get('is_in_stats_window', True)),
                        'dt_s': round(float(result.get('dt_s', current_dt) or current_dt), 6),
                    }
                    if result.get('flow'):
                        log_entry['flow_p90_mm_s'] = round(result['flow'].get('flow_p90_mm_s', 0), 2)
                    log_entry['decision_flow_p90_mm_s'] = round(float(result.get('flow_p90', 0.0) or 0.0), 2)
                    log_entry['speed_std_mm_s'] = round(float(result.get('speed_std', 0.0) or 0.0), 2)
                    log_entry['flow_entropy'] = round(float(result.get('flow_entropy', 0.0) or 0.0), 3)
                    log_entry['flow_over_speed'] = round(float(result.get('flow_over_speed', 0.0) or 0.0), 3)
                    log_entry['flow_over_speed_std'] = round(float(result.get('flow_over_speed_std', 0.0) or 0.0), 3)
                    log_entry['acf_peak'] = round(float(result.get('acf_peak', 0.0) or 0.0), 3)
                    swing_state = result.get('swing_state') or {}
                    log_entry['swing_score'] = round(float(swing_state.get('swing_score', 0.0) or 0.0), 3)
                    log_entry['struggle_score'] = round(float(swing_state.get('struggle_score', 0.0) or 0.0), 3)
                    log_entry['swing_margin'] = round(float(swing_state.get('swing_margin', 0.0) or 0.0), 3)
                    log_entry['internal_flow_ok'] = bool(swing_state.get('internal_flow_ok', result.get('internal_flow_ok', False)))
                    log_entry['rigid_body_like'] = bool(swing_state.get('rigid_body_like', result.get('rigid_body_like', False)))
                    log_entry['aspect_range_ratio'] = round(float(swing_state.get('aspect_range_ratio', result.get('aspect_range_ratio', 0.0)) or 0.0), 3)
                    log_entry['y_range_body'] = round(float(swing_state.get('y_range_body', result.get('y_range_body', 0.0)) or 0.0), 3)
                    log_entry['area_range_ratio'] = round(float(swing_state.get('area_range_ratio', result.get('area_range_ratio', 0.0)) or 0.0), 3)
                    logs.append(log_entry)

            # 实时数据更新（每50帧）
            if frame_idx % 50 == 0:
                self.realtime_data = []
                for result in results:
                    if result.get('detected'):
                        self.realtime_data.append({
                            'roi_id': result['roi_id'],
                            'roi_name': result.get('roi_name', ''),
                            'is_struggling': bool(result.get('is_struggling', False)),
                            'legacy_is_struggling': bool(result.get('is_struggling', False)),
                            'behavior_algorithm': 'legacy',
                            'flow_value': result.get('flow', {}).get('flow_p90_mm_s', 0) if result.get('flow') else 0,
                        })

            # 处理摆动状态
            for result in results:
                roi_id = result['roi_id']
                swing_state = result.get('swing_state') or {}
                current_swing = bool(swing_state.get('is_confirmed_swing', result.get('is_swinging', False)))
                prev_swing = prev_swing_states.get(roi_id, False)

                if current_swing and not prev_swing:
                    retroactive_count = int(swing_state.get('retroactive_frame_count') or 0)
                    previous_count = max(0, retroactive_count - 1)  # current frame is not buffered yet
                    buffered_items = list(video_buffer)
                    if previous_count > 0:
                        target_items = buffered_items[-previous_count:]
                    else:
                        # Legacy fallback for tracker results without explicit window size.
                        target_items = [
                            item for item in buffered_items
                            if any(
                                r.get('roi_id') == roi_id
                                and (r.get('swing_state') or {}).get('is_candidate', r.get('is_swinging', False))
                                for r in item.get('results', [])
                            )
                        ]

                    for buf_item in target_items:
                        for r in buf_item['results']:
                            if r['roi_id'] == roi_id:
                                r['is_swinging'] = True
                                r['is_struggling'] = False
                                buffered_swing_state = r.get('swing_state')
                                if isinstance(buffered_swing_state, dict):
                                    buffered_swing_state['is_effective_swing'] = True
                                    buffered_swing_state['retroactively_displayed_swing'] = True

                    remaining_log_rows = retroactive_count
                    for log_entry in reversed(logs):
                        if remaining_log_rows <= 0:
                            break
                        if log_entry.get('roi_id') == roi_id:
                            log_entry['is_swinging'] = True
                            log_entry['is_struggling'] = False
                            log_entry['legacy_is_struggling'] = False
                            remaining_log_rows -= 1

                prev_swing_states[roi_id] = current_swing

            # 绘制结果 & 异步写入
            if generate_video:
                video_buffer.append({'frame': analysis_frame.copy(), 'results': results})

                if len(video_buffer) >= buffer_size:
                    oldest = video_buffer.popleft()
                    # 非阻塞写入：放入队列由后台线程处理
                    write_queue.put((oldest['frame'].copy(), [dict(result) for result in oldest['results']]))

            # 更新进度
            frame_idx += 1
            skipped = self._grab_skipped_frames(cap, current_frame_no, end_frame, sampling['frame_stride'])
            covered = 1 + skipped
            source_frames_done += covered
            self._advance_frame_counters(covered)

            if frame_idx % 30 == 0:
                self._set_video_progress(
                    batch_index,
                    source_frames_done / max(frames_to_process, 1),
                    f'帧 {source_frames_done}/{frames_to_process} ({{progress}}%)',
                )

            # 保存当前帧（用于实时预览）
            if frame_idx % 10 == 0:
                self.current_frame = analysis_frame

        # 刷新缓冲区中剩余的帧
        if generate_video:
            while len(video_buffer) > 0:
                item = video_buffer.popleft()
                write_queue.put((item['frame'].copy(), [dict(result) for result in item['results']]))

        # 通知写入线程结束并等待完成
        if writer_thread is not None:
            write_queue.put(None)  # 发送结束信号
            writer_thread.join()

        cap.release()
        if writer is not None:
            writer.release()
        if not self.is_running:
            return
        self._ensure_analysis_frames_processed(
            video_path=video_path,
            label='TST',
            frame_idx=frame_idx,
            source_frames_done=source_frames_done,
        )

        # 过滤空ROI
        valid_trackers = []
        for tracker in trackers:
            detection_rate = tracker.detected_count / max(frame_idx, 1)
            if detection_rate >= 0.01 or tracker.detected_count > 30:
                valid_trackers.append(tracker)
                self._log(f"ROI {tracker.roi_id} ({tracker.roi_name}): 检测 {tracker.detected_count}/{frame_idx} 帧 ({detection_rate*100:.1f}%) ✓")

        if not valid_trackers and trackers:
            valid_trackers = trackers

        if len(logs) > 0:
            activity_work_path = output_path.replace('.mp4', '_activity_curve.png')
            if render_tst_activity_curve(logs, valid_trackers, activity_work_path):
                activity_path = _export_result_file(activity_work_path, f"{output_info['final_base']}_activity_curve.png")
                if activity_path:
                    self.result_files.append(activity_path)
                    self.activity_curve_path = activity_path
                    self._log(f"TST activity curve saved: {activity_path}")
            else:
                self._log("Failed to save TST activity curve")

            ethogram_swing_work_path = output_path.replace('.mp4', '_ethogram_swing.png')
            if render_tst_ethogram(logs, valid_trackers, ethogram_swing_work_path, distinguish_swing=True):
                ethogram_swing_path = _export_result_file(
                    ethogram_swing_work_path,
                    f"{output_info['final_base']}_ethogram_swing.png",
                )
                if ethogram_swing_path:
                    self.result_files.append(ethogram_swing_path)
                    self.ethogram_swing_path = ethogram_swing_path
                    self._log(f"TST ethogram with swing saved: {ethogram_swing_path}")
            else:
                self._log("Failed to save TST ethogram with swing")

            ethogram_merged_work_path = output_path.replace('.mp4', '_ethogram_merged.png')
            if render_tst_ethogram(logs, valid_trackers, ethogram_merged_work_path, distinguish_swing=False):
                ethogram_merged_path = _export_result_file(
                    ethogram_merged_work_path,
                    f"{output_info['final_base']}_ethogram_merged.png",
                )
                if ethogram_merged_path:
                    self.result_files.append(ethogram_merged_path)
                    self.ethogram_merged_path = ethogram_merged_path
                    self._log(f"TST ethogram merged saved: {ethogram_merged_path}")
            else:
                self._log("Failed to save TST ethogram merged")

        # 保存 CSV
        if len(logs) > 0 and HAS_PANDAS:
            csv_work_path = output_path.replace('.mp4', '_logs.csv')
            if _write_dataframe_csv(logs, csv_work_path):
                csv_path = _export_result_file(csv_work_path, f"{output_info['final_base']}_logs.csv")
                if csv_path:
                    self.result_files.append(csv_path)
                    self.csv_path = csv_path
                    self._log(f"日志已保存: {csv_path}")

        # 保存不动时间统计
        immobility_summaries = []
        for tracker in valid_trackers:
            summary = tracker.get_immobility_summary(analysis_fps)
            if summary:
                summary['timebase_source'] = timebase_summary['source']
                summary['timebase_median_dt_s'] = round(float(timebase_summary['median_dt']), 6)
                summary['timebase_long_gap_gt_50ms'] = int(timebase_summary['long_gap_gt_50ms'])
                self._add_sampling_summary_fields(summary, sampling)
                immobility_summaries.append(summary)

        if len(immobility_summaries) > 0 and HAS_PANDAS:
            immobility_csv_work = output_path.replace('.mp4', '_immobility.csv')
            if _write_dataframe_csv(immobility_summaries, immobility_csv_work):
                immobility_csv = _export_result_file(
                    immobility_csv_work,
                    f"{output_info['final_base']}_immobility.csv",
                )
                if immobility_csv:
                    self.result_files.append(immobility_csv)
                    self.immobility_csv_path = immobility_csv

            txt_work_path = output_path.replace('.mp4', '_report.txt')
            self._generate_txt_report(immobility_summaries, txt_work_path)
            txt_path = _export_result_file(txt_work_path, f"{output_info['final_base']}_report.txt")
            if txt_path:
                self.result_files.append(txt_path)
                self.immobility_txt_path = txt_path

        # 保存分析视频
        if generate_video and os.path.exists(output_path):
            exported_video = _export_result_file(output_path, final_output_path)
            if exported_video:
                file_size = os.path.getsize(exported_video) / (1024 * 1024)
                self._log(f"分析视频已保存: {exported_video} ({file_size:.1f} MB)")
                self.result_files.append(exported_video)

        if batch_index is None and self.is_running:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = '分析完成！'

        elapsed = time.time() - video_start_time
        self._log(f"视频处理完成，耗时: {elapsed:.1f}秒，处理 {source_frames_done} source frames ({frame_idx} analyzed samples)")

        import gc
        gc.collect()

    @staticmethod
    def _bool_log_value(value, default=False):
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y'}
        if value is None:
            return bool(default)
        return bool(value)

    @staticmethod
    def _float_log_value(value, default=0.0):
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(out):
            return float(default)
        return out

    def _apply_tst_lis_primary_summary(self, summary, logs):
        roi_id = summary.get('roi_id')
        roi_rows = [
            row for row in logs
            if row.get('roi_id') == roi_id and self._bool_log_value(row.get('is_in_stats_window'), True)
        ]
        if not roi_rows:
            return

        summary['behavior_algorithm'] = 'tst_lis'
        summary['tst_lis_enabled'] = True
        detected_frames = len(roi_rows)
        struggling_frames = sum(
            1 for row in roi_rows
            if self._bool_log_value(row.get('is_struggling'), False)
        )
        total_frames = int(self._float_log_value(summary.get('total_frames'), detected_frames))
        if total_frames <= 0:
            total_frames = detected_frames
        total_time = self._float_log_value(summary.get('total_time_seconds'), 0.0)
        if total_time <= 0:
            total_time = sum(max(self._float_log_value(row.get('dt_s'), 0.0), 0.0) for row in roi_rows)
        frame_duration = total_time / max(float(total_frames), 1.0)

        uncertain_frames = max(total_frames - detected_frames, 0)
        immobility_frames = max(detected_frames - struggling_frames, 0)

        detected_time = detected_frames * frame_duration
        struggling_time = struggling_frames * frame_duration
        uncertain_time = uncertain_frames * frame_duration
        immobility_time = immobility_frames * frame_duration

        summary['detected_time_seconds'] = round(detected_time, 2)
        summary['uncertain_time_seconds'] = round(uncertain_time, 2)
        summary['immobility_time_seconds'] = round(immobility_time, 2)
        summary['struggling_time_seconds'] = round(struggling_time, 2)
        summary['detected_rate_percent'] = round((detected_frames / max(float(total_frames), 1e-9)) * 100, 1)
        summary['immobility_percent'] = round((immobility_frames / max(float(total_frames), 1e-9)) * 100, 1)
        summary['struggling_percent'] = round((struggling_frames / max(float(total_frames), 1e-9)) * 100, 1)
        summary['uncertain_percent'] = round((uncertain_frames / max(float(total_frames), 1e-9)) * 100, 1)
        summary['lis_struggling_time_seconds'] = round(struggling_time, 2)
        summary['lis_immobility_time_seconds'] = round(immobility_time, 2)
        summary['lis_struggling_percent'] = round((struggling_frames / max(float(total_frames), 1e-9)) * 100, 1)
        summary['lis_immobility_percent'] = round((immobility_frames / max(float(total_frames), 1e-9)) * 100, 1)
        summary['lis_passive_swing_frames'] = sum(
            1 for row in roi_rows if self._bool_log_value(row.get('lis_passive_swing'), False)
        )
        summary['lis_low_confidence_frames'] = sum(
            1 for row in roi_rows if str(row.get('lis_confidence') or '').lower() == 'low'
        )

    def get_download_file_map(self):
        return {
            'video': [f for f in self.result_files if f.endswith('.mp4')],
            'csv': [self.csv_path] if self.csv_path else [],
            'immobility_csv': [self.immobility_csv_path] if self.immobility_csv_path else [],
            'report': [self.immobility_txt_path] if self.immobility_txt_path else [],
            'activity_curve': [self.activity_curve_path] if self.activity_curve_path else [],
            'ethogram_swing': [self.ethogram_swing_path] if self.ethogram_swing_path else [],
            'ethogram_merged': [self.ethogram_merged_path] if self.ethogram_merged_path else [],
        }

    def _generate_txt_report(self, immobility_summaries, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 72 + "\n")
                f.write("MouseScope 悬尾实验报告 (Tail Suspension Test Report)\n")
                f.write("=" * 72 + "\n\n")
                f.write(f"{_bi_label('分析时间', 'Analysis Time')}: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{_bi_label('ROI 数量', 'ROI Count')}: {len(immobility_summaries)}\n\n")
                for i, summary in enumerate(immobility_summaries, 1):
                    roi_name = summary.get('roi_name', f"ROI {summary['roi_id']}")
                    f.write(f"{_bi_label('对象', 'Subject')} {i}: {roi_name}\n")
                    if summary.get('behavior_algorithm') == 'tst_lis':
                        f.write(f"  {_bi_label('主判定算法', 'Primary Algorithm')}: {_bi_label('微小挣扎优化逻辑', 'Subtle-Struggle Optimized Logic')}\n")
                    _write_report_metric_lines(f, summary, [
                        ('总时长', 'Total Time', 'total_time_seconds', ' s'),
                        ('不动时间', 'Immobility Time', 'immobility_time_seconds', ' s'),
                        ('不动比例', 'Immobility Percent', 'immobility_percent', '%'),
                        ('挣扎时间', 'Struggling Time', 'struggling_time_seconds', ' s'),
                        ('挣扎比例', 'Struggling Percent', 'struggling_percent', '%'),
                        ('检测率', 'Detected Rate', 'detected_rate_percent', '%'),
                        ('不确定时间', 'Uncertain Time', 'uncertain_time_seconds', ' s'),
                        ('不确定比例', 'Uncertain Percent', 'uncertain_percent', '%'),
                        ('自动行为阈值', 'Auto Behavior Threshold', 'auto_behavior_threshold', ''),
                    ])
                    f.write(f"  {_bi_label('光流阈值低/高', 'Flow Threshold Lo/Hi')}: {_report_value(summary.get('flow_thr_lo'))} / {_report_value(summary.get('flow_thr_hi'))}\n")
                    f.write(f"  {_bi_label('速度标准差阈值低/高', 'Speed-Std Threshold Lo/Hi')}: {_report_value(summary.get('speed_std_thr_lo'))} / {_report_value(summary.get('speed_std_thr_hi'))}\n")
                    if summary.get('tst_lis_enabled'):
                        f.write(
                            f"  {_bi_label('优化逻辑挣扎时间', 'Optimized Logic Struggling Time')}: "
                            f"{_report_value(summary.get('lis_struggling_time_seconds'), ' s')}\n"
                        )
                        f.write(
                            f"  {_bi_label('优化逻辑挣扎比例', 'Optimized Logic Struggling Percent')}: "
                            f"{_report_value(summary.get('lis_struggling_percent'), '%')}\n"
                        )
                    if summary.get('latency_to_immobility_seconds') is not None:
                        f.write(
                            f"  {_bi_label('首次不动潜伏期', 'Latency to First Immobility')}: "
                            f"{_report_value(summary.get('latency_to_immobility_seconds'), ' s')}\n"
                        )
                    f.write("\n")
        except Exception as e:
            print(f"生成TXT报告失败: {e}")


# ==================== Open Field ====================
