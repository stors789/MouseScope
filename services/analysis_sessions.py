from __future__ import annotations

import copy
import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from vision_tracker import (
    BackgroundSubtractor,
    MorphologyProcessor,
    MouseDetector,
    OpticalFlowAnalyzer,
    MouseTracker,
    draw_results,
    create_static_background,
)
from open_field import (
    OpenFieldTracker,
    draw_open_field_results,
    render_open_field_heatmap,
    render_open_field_image_grid,
    render_open_field_trackmap,
    sanitize_name,
)
from novel_object import (
    NovelObjectTracker,
    draw_novel_object_results,
    render_novel_object_heatmap,
    render_novel_object_trackmap,
)
from elevated_plus import (
    ElevatedPlusTracker,
    draw_elevated_plus_results,
    render_elevated_plus_heatmap,
    render_elevated_plus_trackmap,
    render_elevated_plus_image_grid,
    sanitize_name as sanitize_epm_name,
)
from y_maze import (
    YMazeTracker,
    draw_y_maze_results,
    render_y_maze_heatmap,
    render_y_maze_trackmap,
    render_y_maze_image_grid,
    sanitize_name as sanitize_y_maze_name,
)
from forced_swim import (
    ForcedSwimTracker,
    draw_forced_swim_results,
    render_forced_swim_activity_curve,
    render_forced_swim_ethogram,
)
from services.payload_geometry import (
    coerce_point_list as _coerce_point_list,
    coerce_zone_polygons as _coerce_zone_polygons,
    coerce_zone_polygons_with_schema as _coerce_zone_polygons_with_schema,
)


RESULT_DIR: Path | str | None = None
HAS_PANDAS = False

def _missing_runtime(*_args, **_kwargs):
    raise RuntimeError('Analysis runtime has not been configured.')

china_now: Callable[[], Any] = _missing_runtime
_prepare_output_paths: Callable[..., dict[str, Any]] = _missing_runtime
_export_result_file: Callable[..., str | None] = _missing_runtime
_write_dataframe_csv: Callable[..., bool] = _missing_runtime
_bi_label: Callable[[str, str], str] = _missing_runtime
_report_value: Callable[..., str] = _missing_runtime
_report_time_mode_label: Callable[[Any], str] = _missing_runtime
_report_phase_label: Callable[[Any], str] = _missing_runtime
_write_report_metric_lines: Callable[..., None] = _missing_runtime
_analysis_status_observer: Callable[[Any], None] | None = None

DETECTION_MASK_MIN_POINTS = 3
EPM_ZONE_KEYS = ('center', 'open_1', 'open_2', 'closed_1', 'closed_2')
YMAZE_ZONE_POINT_COUNTS = {'center': 3, 'arm_a': 4, 'arm_b': 4, 'arm_c': 4}


def configure_analysis_runtime(**kwargs: Any) -> None:
    global RESULT_DIR, HAS_PANDAS
    global china_now, _prepare_output_paths, _export_result_file, _write_dataframe_csv
    global _bi_label, _report_value, _report_time_mode_label, _report_phase_label, _write_report_metric_lines
    global _analysis_status_observer

    RESULT_DIR = kwargs['RESULT_DIR']
    HAS_PANDAS = bool(kwargs.get('HAS_PANDAS', False))
    china_now = kwargs['china_now']
    _prepare_output_paths = kwargs['_prepare_output_paths']
    _export_result_file = kwargs['_export_result_file']
    _write_dataframe_csv = kwargs['_write_dataframe_csv']
    _bi_label = kwargs['_bi_label']
    _report_value = kwargs['_report_value']
    _report_time_mode_label = kwargs['_report_time_mode_label']
    _report_phase_label = kwargs['_report_phase_label']
    _write_report_metric_lines = kwargs['_write_report_metric_lines']
    _analysis_status_observer = kwargs.get('_analysis_status_observer')


def _notify_analysis_status(analysis_session: Any) -> None:
    if _analysis_status_observer is None:
        return
    try:
        _analysis_status_observer(analysis_session)
    except Exception:
        pass


def _run_with_status_observer(analysis_session: Any) -> None:
    try:
        analysis_session._run()
    finally:
        _notify_analysis_status(analysis_session)


# ==================== 分析引擎 ====================
class VisionAnalysisSession:
    """视觉分析会话（无数据库、无YOLO）"""

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

    def _config_for_video(self, video_index):
        vision_config = copy.deepcopy(self.config)
        per_video_configs = vision_config.pop('per_video_configs', None) or []
        video_index = int(video_index or 0)
        if 0 <= video_index < len(per_video_configs):
            per_video_config = per_video_configs[video_index] or {}
            for key, value in per_video_config.items():
                vision_config[key] = copy.deepcopy(value)
        return vision_config

    def start(self):
        """在后台线程中启动分析"""
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        _notify_analysis_status(self)
        thread = threading.Thread(target=_run_with_status_observer, args=(self,), daemon=True)
        thread.start()

    def _run(self):
        """分析主循环"""
        try:
            if len(self.video_paths) == 1:
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                # 批量处理
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(f"正在处理视频 {i+1}/{len(self.video_paths)}: {info.get('original_name', '')}")
                    self._analyze_single_video(path, info, batch_index=i)
                    base_progress = int((i + 1) / len(self.video_paths) * 100)
                    self.progress = base_progress

                self.status = 'completed'
                self.end_time = time.time()
                self.message = f'全部 {len(self.video_paths)} 个视频分析完成！'
                self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = f'分析出错: {str(e)}'
            self._log(f"❌ {self.message}")
            import traceback
            self._log(traceback.format_exc())

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        """分析单个视频"""
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)

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
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps_override:
            self._log(f"视频信息: {width}x{height}, {fps:.1f}fps (手动指定, OpenCV检测={detected_fps:.1f}), {total_frames}帧")
        else:
            self._log(f"视频信息: {width}x{height}, {fps:.1f}fps, {total_frames}帧")

        # 创建处理器
        bg_method = vision_config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=vision_config.get('gray_threshold', 127),
                invert=vision_config.get('invert', False)
            )
        elif bg_method == 'static':
            try:
                bg = create_static_background(video_path, num_frames=10)
                bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
                bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
            except:
                bg_subtractor = BackgroundSubtractor(method='threshold', threshold=127)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=vision_config.get('erosion_kernel', 3),
            erosion_iterations=vision_config.get('erosion_iter', 2),
            dilation_kernel=vision_config.get('dilation_kernel', 5),
            dilation_iterations=vision_config.get('dilation_iter', 3)
        )

        detector = MouseDetector(
            min_area=vision_config.get('min_area', 100),
            max_area=50000
        )

        pixel_to_mm = vision_config.get('pixel_to_mm', 1.0)

        # 创建追踪器
        trackers = []
        rois_config = vision_config.get('rois', [])
        if isinstance(rois_config, str):
            try:
                rois_config = json.loads(rois_config)
            except:
                rois_config = []

        for roi_idx, roi_item in enumerate(rois_config):
            if isinstance(roi_item, str):
                try:
                    roi_item = json.loads(roi_item)
                except:
                    continue

            if isinstance(roi_item, dict) and 'points' in roi_item:
                roi_points = roi_item['points']
                roi_name = roi_item.get('name', f"ROI {roi_idx+1}")
                time_config = roi_item.get('time_config')
            else:
                roi_points = roi_item
                roi_name = f"ROI {roi_idx+1}"
                time_config = None

            if isinstance(roi_points, str):
                try:
                    roi_points = json.loads(roi_points)
                except:
                    continue

            roi_polygon = [[p['x'], p['y']] for p in roi_points]

            flow_analyzer = OpticalFlowAnalyzer(fps=fps, pixel_to_mm=pixel_to_mm)

            tracker = MouseTracker(
                roi_polygon=roi_polygon,
                roi_id=roi_idx + 1,
                bg_subtractor=bg_subtractor,
                morphology_processor=morph_processor,
                detector=detector,
                flow_analyzer=flow_analyzer,
                use_yolo_seg=False,
                fps=fps,
                pixel_to_mm=pixel_to_mm,
                roi_name=roi_name,
                time_config=time_config,
                win_seconds=0.3,
                flow_thr_hi=50.0,
                flow_thr_lo=35.0,
                speed_std_thr_hi=40.0,
                speed_std_thr_lo=20.0,
                min_active_seconds=0.2,
                calm_seconds=0.4,
                coherence_thr=0.8,
                coherence_calm_seconds=0.5,
            )
            trackers.append(tracker)

        if not trackers:
            self._log("❌ 没有有效的 ROI 配置")
            return

        self._log(f"创建了 {len(trackers)} 个追踪器")

        # 视频编码
        writer = None
        if generate_video:
            codecs_to_try = [
                ('avc1', 'H264'), ('H264', 'H264'), ('X264', 'X264'), ('mp4v', 'MPEG-4'),
            ]
            for codec, codec_name in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if test_writer.isOpened():
                        writer = test_writer
                        self._log(f"视频编码器: {codec_name}")
                        break
                    else:
                        test_writer.release()
                except:
                    pass

        # 时间范围
        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"从第 {start_frame} 帧开始分析 ({start_frame/fps:.1f}秒)")

        frames_to_process = end_frame - start_frame + 1

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
                    writer.write(item)
            write_done.set()

        writer_thread = None
        if generate_video and writer is not None:
            writer_thread = threading.Thread(target=_video_writer_thread, daemon=True)
            writer_thread.start()

        # 视频写入缓冲区（用于摆动状态回溯修正）
        buffer_seconds = 2.0
        buffer_size = int(fps * buffer_seconds)
        video_buffer = deque(maxlen=buffer_size + 10)
        prev_swing_states = {}

        frame_idx = 0
        logs = []

        self._log(f"开始分析... (CPU核心: {os.cpu_count()}, OpenCV线程: {cv2.getNumThreads()})")

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except:
                current_frame_no = start_frame + frame_idx

            if end_frame is not None and current_frame_no > end_frame:
                break

            current_time = current_frame_no / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 追踪每个 ROI
            results = []
            for tracker in trackers:
                result = tracker.update(frame, current_time=current_time, gray=gray)
                results.append(result)

            # 记录日志
            for result in results:
                if result.get('detected'):
                    log_entry = {
                        'frame': current_frame_no,
                        'time': round(current_time, 3),
                        'roi_id': result['roi_id'],
                        'roi_name': result.get('roi_name', ''),
                        'is_struggling': result.get('is_struggling', False),
                    }
                    if result.get('flow'):
                        log_entry['flow_p90_mm_s'] = round(result['flow'].get('flow_p90_mm_s', 0), 2)
                    logs.append(log_entry)

            # 实时数据更新（每50帧）
            if frame_idx % 50 == 0:
                self.realtime_data = []
                for result in results:
                    if result.get('detected'):
                        self.realtime_data.append({
                            'roi_id': result['roi_id'],
                            'roi_name': result.get('roi_name', ''),
                            'is_struggling': result.get('is_struggling', False),
                            'flow_value': result.get('flow', {}).get('flow_p90_mm_s', 0) if result.get('flow') else 0,
                        })

            # 处理摆动状态
            for result in results:
                roi_id = result['roi_id']
                current_swing = result.get('swing_state', {}).get('is_confirmed_swing', False)
                prev_swing = prev_swing_states.get(roi_id, False)

                if current_swing and not prev_swing:
                    # 摆动开始 — 回溯修正缓冲区
                    for buf_item in video_buffer:
                        for r in buf_item['results']:
                            if r['roi_id'] == roi_id:
                                r['is_swinging'] = True
                                r['is_struggling'] = False

                prev_swing_states[roi_id] = current_swing

            # 绘制结果 & 异步写入
            if generate_video:
                result_frame = draw_results(frame.copy(), results)
                video_buffer.append({'frame': result_frame, 'results': results})

                if len(video_buffer) >= buffer_size:
                    oldest = video_buffer.popleft()
                    # 非阻塞写入：放入队列由后台线程处理
                    write_queue.put(oldest['frame'])

            # 更新进度
            frame_idx += 1
            self.processed_frames = frame_idx

            if frame_idx % 30 == 0:
                if batch_index is not None:
                    batch_progress = int(batch_index / len(self.video_paths) * 100)
                    video_progress = int(frame_idx / max(frames_to_process, 1) * 100 / len(self.video_paths))
                    self.progress = batch_progress + video_progress
                else:
                    self.progress = int(frame_idx / max(frames_to_process, 1) * 100)
                self.message = f'帧 {frame_idx}/{frames_to_process} ({self.progress}%)'

            # 保存当前帧（用于实时预览）
            if frame_idx % 10 == 0:
                self.current_frame = frame

        # 刷新缓冲区中剩余的帧
        if generate_video:
            while len(video_buffer) > 0:
                item = video_buffer.popleft()
                write_queue.put(item['frame'])

        # 通知写入线程结束并等待完成
        if writer_thread is not None:
            write_queue.put(None)  # 发送结束信号
            writer_thread.join(timeout=30)

        cap.release()
        if writer is not None:
            writer.release()

        # 过滤空ROI
        valid_trackers = []
        for tracker in trackers:
            detection_rate = tracker.detected_count / max(frame_idx, 1)
            if detection_rate >= 0.01 or tracker.detected_count > 30:
                valid_trackers.append(tracker)
                self._log(f"ROI {tracker.roi_id} ({tracker.roi_name}): 检测 {tracker.detected_count}/{frame_idx} 帧 ({detection_rate*100:.1f}%) ✓")

        if not valid_trackers and trackers:
            valid_trackers = trackers

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
            summary = tracker.get_immobility_summary(fps)
            if summary:
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

        if batch_index is None:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = '分析完成！'

        elapsed = time.time() - video_start_time
        self._log(f"视频处理完成，耗时: {elapsed:.1f}秒，处理 {frame_idx} 帧")

        import gc
        gc.collect()

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()
        _notify_analysis_status(self)

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        return {
            'session_id': self.session_id,
            'experiment_type': self.experiment_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'logs': self.logs[-10:],
            'realtime_data': self.realtime_data,
            'result_files': self.result_files,
            'stats': {
                'processed_frames': self.processed_frames,
                'elapsed_time': elapsed,
                'fps': processing_fps,
            }
        }

    def get_download_file_map(self):
        return {
            'video': [f for f in self.result_files if f.endswith('.mp4')],
            'csv': [self.csv_path] if self.csv_path else [],
            'immobility_csv': [self.immobility_csv_path] if self.immobility_csv_path else [],
            'report': [self.immobility_txt_path] if self.immobility_txt_path else [],
        }

    def _log(self, message):
        timestamp = china_now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

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
                    _write_report_metric_lines(f, summary, [
                        ('总时长', 'Total Time', 'total_time_seconds', ' s'),
                        ('不动时间', 'Immobility Time', 'immobility_time_seconds', ' s'),
                        ('不动比例', 'Immobility Percent', 'immobility_percent', '%'),
                        ('挣扎时间', 'Struggling Time', 'struggling_time_seconds', ' s'),
                        ('挣扎比例', 'Struggling Percent', 'struggling_percent', '%'),
                    ])
                    if summary.get('latency_to_immobility_seconds') is not None:
                        f.write(
                            f"  {_bi_label('首次不动潜伏期', 'Latency to First Immobility')}: "
                            f"{_report_value(summary.get('latency_to_immobility_seconds'), ' s')}\n"
                        )
                    f.write("\n")
        except Exception as e:
            print(f"生成TXT报告失败: {e}")


# ==================== Open Field ====================

class OpenFieldMultiArenaAnalysisSession:
    """Multi-arena Open Field Test analysis session."""

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

    def start(self):
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        _notify_analysis_status(self)
        thread = threading.Thread(target=_run_with_status_observer, args=(self,), daemon=True)
        thread.start()

    def _log(self, message):
        timestamp = china_now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

    def _append_result_file(self, path, group=None):
        if not path or not os.path.exists(path):
            return
        if path not in self.result_files:
            self.result_files.append(path)
        if group and path not in self.output_groups[group]:
            self.output_groups[group].append(path)

    def _config_for_video(self, video_index):
        vision_config = copy.deepcopy(self.config)
        per_video_configs = vision_config.pop('per_video_configs', None) or []
        video_index = int(video_index or 0)
        if 0 <= video_index < len(per_video_configs):
            per_video_config = per_video_configs[video_index] or {}
            for key, value in per_video_config.items():
                vision_config[key] = copy.deepcopy(value)
        return vision_config

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

    def _run(self):
        try:
            if len(self.video_paths) == 1:
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(f"Processing OFT video {i + 1}/{len(self.video_paths)}: {info.get('original_name', '')}")
                    self._analyze_single_video(path, info, batch_index=i)
                    self.progress = int((i + 1) / len(self.video_paths) * 100)

                if self.is_running:
                    self.status = 'completed'
                    self.end_time = time.time()
                    self.message = f'All {len(self.video_paths)} OFT videos finished.'
                    self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = f'Analysis error: {str(e)}'
            self._log(self.message)
            import traceback
            self._log(traceback.format_exc())

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
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

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to open video: {video_path}")
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log(f"OFT video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

        bg_method = vision_config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=vision_config.get('gray_threshold', 127),
                invert=vision_config.get('invert', False),
            )
        elif bg_method == 'static':
            bg = create_static_background(video_path, num_frames=15)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=vision_config.get('erosion_kernel', 3),
            erosion_iterations=vision_config.get('erosion_iter', 1),
            dilation_kernel=vision_config.get('dilation_kernel', 5),
            dilation_iterations=vision_config.get('dilation_iter', 2),
        )
        detector = MouseDetector(
            min_area=vision_config.get('min_area', 120),
            max_area=vision_config.get('max_area', 50000),
        )

        arena_side_cm = float(vision_config.get('arena_side_cm', 50.0))
        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        roi_configs = self._parse_rois(vision_config)

        trackers = []
        for roi in roi_configs:
            trackers.append(
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
            )
        self._log(f"Configured {len(trackers)} OFT arenas")

        writer = None
        if generate_video:
            for codec in ('avc1', 'H264', 'mp4v'):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    if test_writer.isOpened():
                        writer = test_writer
                        self._log(f"OFT video codec: {codec}")
                        break
                    test_writer.release()
                except Exception:
                    pass

        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start OFT analysis from frame {start_frame} ({start_frame / max(fps, 1e-6):.1f}s)")

        frames_to_process = max(0, end_frame - start_frame + 1)
        frame_idx = 0

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + frame_idx

            if current_frame_no > end_frame:
                break

            current_time = current_frame_no / max(fps, 1e-6)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = [tracker.update(frame, current_time, gray, bg_subtractor, morph_processor, detector) for tracker in trackers]

            if frame_idx % 15 == 0:
                rows = []
                for tracker, result in zip(trackers, results):
                    summary = tracker.build_summary()
                    rows.append({
                        'roi_id': tracker.roi_id,
                        'roi_name': tracker.roi_name,
                        'detected': bool(result.get('detected')),
                        'zone': result.get('zone_label', 'Unknown'),
                        'speed_cm_s': round(float(result.get('speed_cm_s', 0.0)), 3),
                        'distance_cm': summary['total_distance_cm'],
                        'center_percent': summary['center_percent'],
                        'center_entries': summary['center_entries'],
                        'wall_time_s': summary['wall_time_s'],
                        'in_stats_window': bool(result.get('in_stats_window', True)),
                    })
                self.realtime_data = rows
                self.live_metrics = self._build_live_metrics(rows)

            if generate_video and writer is not None:
                result_frame = draw_open_field_results(frame, list(zip(trackers, results)))
                writer.write(result_frame)
                if frame_idx % 10 == 0:
                    self.current_frame = result_frame

            frame_idx += 1
            self.processed_frames += 1

            if frame_idx % 20 == 0:
                if batch_index is not None:
                    batch_progress = int(batch_index / len(self.video_paths) * 100)
                    video_progress = int(frame_idx / max(frames_to_process, 1) * 100 / len(self.video_paths))
                    self.progress = batch_progress + video_progress
                else:
                    self.progress = int(frame_idx / max(frames_to_process, 1) * 100)
                self.message = f'Frame {frame_idx}/{frames_to_process} ({self.progress}%)'

        cap.release()
        if writer is not None:
            writer.release()

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
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
        for tracker in trackers:
            roi_slug = sanitize_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"arena_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"

            if render_open_field_heatmap(tracker, heatmap_work_path):
                exported_heatmap = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported_heatmap)
                self._append_result_file(exported_heatmap)
            else:
                self._log(f"Failed to save OFT heatmap: {heatmap_final_path}")

            if render_open_field_trackmap(tracker, trackmap_work_path):
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
                if render_open_field_image_grid(heatmap_work_paths, heatmap_grid_work_path, title=f"{video_name} | Heatmaps"):
                    self._append_result_file(_export_result_file(heatmap_grid_work_path, heatmap_grid_final_path), 'heatmap')
                else:
                    self._log(f"Failed to save OFT heatmap grid: {heatmap_grid_final_path}")

        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                trackmap_grid_work_path = f"{output_base}_trackmaps.png"
                trackmap_grid_final_path = f"{final_base}_trackmaps.png"
                if render_open_field_image_grid(trackmap_work_paths, trackmap_grid_work_path, title=f"{video_name} | Trajectory Maps"):
                    self._append_result_file(_export_result_file(trackmap_grid_work_path, trackmap_grid_final_path), 'trackmap')
                else:
                    self._log(f"Failed to save OFT trackmap grid: {trackmap_grid_final_path}")

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
        self._log(f"OFT video processed in {elapsed:.1f}s, {frame_idx} frames")

        import gc
        gc.collect()

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()
        _notify_analysis_status(self)

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        return {
            'session_id': self.session_id,
            'experiment_type': self.experiment_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'logs': self.logs[-10:],
            'realtime_data': self.realtime_data,
            'result_files': self.result_files,
            'experiment_stats': self.live_metrics,
            'stats': {
                'processed_frames': self.processed_frames,
                'elapsed_time': elapsed,
                'fps': processing_fps,
            }
        }

    def get_download_file_map(self):
        return self.output_groups

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

class ForcedSwimAnalysisSession:
    """Multi-cylinder Forced Swim Test analysis session."""

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

    def start(self):
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        _notify_analysis_status(self)
        thread = threading.Thread(target=_run_with_status_observer, args=(self,), daemon=True)
        thread.start()

    def _log(self, message):
        timestamp = china_now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

    def _append_result_file(self, path, group=None):
        if not path or not os.path.exists(path):
            return
        if path not in self.result_files:
            self.result_files.append(path)
        if group and path not in self.output_groups[group]:
            self.output_groups[group].append(path)

    def _config_for_video(self, video_index):
        vision_config = copy.deepcopy(self.config)
        per_video_configs = vision_config.pop('per_video_configs', None) or []
        video_index = int(video_index or 0)
        if 0 <= video_index < len(per_video_configs):
            per_video_config = per_video_configs[video_index] or {}
            for key, value in per_video_config.items():
                vision_config[key] = copy.deepcopy(value)
        return vision_config

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

    def _build_processors(self, video_path, vision_config):
        bg_method = vision_config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=vision_config.get('gray_threshold', 127),
                invert=vision_config.get('invert', False),
            )
        elif bg_method == 'static':
            bg = create_static_background(video_path, num_frames=15)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR) if len(bg.shape) == 2 else bg
            bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=vision_config.get('erosion_kernel', 3),
            erosion_iterations=vision_config.get('erosion_iter', 1),
            dilation_kernel=vision_config.get('dilation_kernel', 5),
            dilation_iterations=vision_config.get('dilation_iter', 2),
        )
        detector = MouseDetector(
            min_area=vision_config.get('min_area', 80),
            max_area=vision_config.get('max_area', 50000),
        )
        return bg_subtractor, morph_processor, detector

    def _set_video_phase_progress(self, batch_index, phase_fraction, message):
        phase_fraction = max(0.0, min(1.0, float(phase_fraction or 0.0)))
        if batch_index is not None:
            per_video = 100.0 / max(len(self.video_paths), 1)
            self.progress = int(batch_index * per_video + phase_fraction * per_video)
        else:
            self.progress = int(phase_fraction * 100)
        self.message = message

    def _open_video_writer(self, output_video_path, fps, width, height, label='FST video'):
        for codec in ('avc1', 'H264', 'mp4v'):
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    self._log(f"{label} codec: {codec}")
                    return writer
                writer.release()
            except Exception:
                pass
        return None

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
        batch_index=None,
    ):
        self._log("Rendering FST annotated video with final scored states...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to reopen video for FST annotated rendering: {video_path}")
            return None

        writer = self._open_video_writer(output_video_path, fps, width, height)
        if writer is None:
            cap.release()
            self._log("Unable to create FST annotated video writer.")
            return None

        render_trackers = self._build_trackers(roi_configs, vision_config, fps, global_start_time_s)
        for render_tracker, source_tracker in zip(render_trackers, source_trackers):
            render_tracker.immobility_threshold = source_tracker.immobility_threshold
            render_tracker.immobility_high_threshold = source_tracker.immobility_high_threshold
            render_tracker.auto_threshold = source_tracker.auto_threshold

        bg_subtractor, morph_processor, detector = self._build_processors(video_path, vision_config)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        render_idx = 0
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + render_idx
            if current_frame_no > end_frame:
                break

            current_time = current_frame_no / max(fps, 1e-6)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = []
            for render_tracker, source_tracker in zip(render_trackers, source_trackers):
                result = render_tracker.update(
                    frame,
                    current_time,
                    gray,
                    bg_subtractor,
                    morph_processor,
                    detector,
                    record=False,
                    frame_index=current_frame_no,
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

            result_frame = draw_forced_swim_results(frame, list(zip(render_trackers, results)))
            writer.write(result_frame)
            if render_idx % 10 == 0:
                self.current_frame = result_frame
            render_idx += 1

            if render_idx % 30 == 0:
                fraction = 0.78 + min(render_idx / max(frames_to_process, 1), 1.0) * 0.20
                self._set_video_phase_progress(
                    batch_index,
                    fraction,
                    f'Rendering FST video {render_idx}/{frames_to_process}',
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

    def _run(self):
        try:
            if len(self.video_paths) == 1:
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(f"Processing FST video {i + 1}/{len(self.video_paths)}: {info.get('original_name', '')}")
                    self._analyze_single_video(path, info, batch_index=i)
                    self.progress = int((i + 1) / len(self.video_paths) * 100)

                if self.is_running:
                    self.status = 'completed'
                    self.end_time = time.time()
                    self.message = f'All {len(self.video_paths)} FST videos finished.'
                    self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = f'Analysis error: {str(e)}'
            self._log(self.message)
            import traceback
            self._log(traceback.format_exc())

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
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log(f"FST video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        frames_to_process = max(0, end_frame - start_frame + 1)

        roi_configs = self._parse_rois(vision_config)
        trackers = self._build_trackers(roi_configs, vision_config, fps, global_start_time_s)
        self._log(f"Configured {len(trackers)} FST cylinders")

        needs_auto = any(tracker.auto_threshold and tracker.immobility_threshold is None for tracker in trackers)
        if needs_auto and self.is_running:
            self._log("Estimating FST immobility thresholds from this video...")
            self._set_video_phase_progress(batch_index, 0.0, 'Estimating FST immobility thresholds...')
            bg_subtractor, morph_processor, detector = self._build_processors(video_path, vision_config)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            prepass_idx = 0
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                except Exception:
                    current_frame_no = start_frame + prepass_idx
                if current_frame_no > end_frame:
                    break
                current_time = current_frame_no / max(fps, 1e-6)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for tracker in trackers:
                    tracker.update(
                        frame,
                        current_time,
                        gray,
                        bg_subtractor,
                        morph_processor,
                        detector,
                        record=False,
                        frame_index=current_frame_no,
                    )
                prepass_idx += 1
                if prepass_idx % 90 == 0:
                    fraction = min(prepass_idx / max(frames_to_process, 1), 1.0) * 0.18
                    self._set_video_phase_progress(
                        batch_index,
                        fraction,
                        f'FST threshold estimation {prepass_idx}/{frames_to_process}',
                    )

            for tracker in trackers:
                tracker.auto_configure_thresholds(fallback_threshold=float(vision_config.get('immobility_threshold') or 0.08))
                self._log(
                    f"ROI {tracker.roi_id} ({tracker.roi_name}) threshold: "
                    f"low={tracker.low_threshold:.4f}, high={tracker.high_threshold:.4f}"
                )
                tracker.reset_runtime(clear_threshold=False)
            self._set_video_phase_progress(batch_index, 0.18, 'FST immobility thresholds estimated.')

        bg_subtractor, morph_processor, detector = self._build_processors(video_path, vision_config)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = 0
        analysis_start_fraction = 0.18 if needs_auto else 0.0
        analysis_end_fraction = 0.78 if generate_video else 0.94
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + frame_idx
            if current_frame_no > end_frame:
                break

            current_time = current_frame_no / max(fps, 1e-6)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = [
                tracker.update(
                    frame,
                    current_time,
                    gray,
                    bg_subtractor,
                    morph_processor,
                    detector,
                    record=True,
                    frame_index=current_frame_no,
                )
                for tracker in trackers
            ]

            if frame_idx % 15 == 0:
                rows = []
                for tracker, result in zip(trackers, results):
                    summary = tracker.build_summary()
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
            self.processed_frames += 1
            if frame_idx % 20 == 0:
                local = min(frame_idx / max(frames_to_process, 1), 1.0)
                fraction = analysis_start_fraction + local * (analysis_end_fraction - analysis_start_fraction)
                self._set_video_phase_progress(batch_index, fraction, f'FST scoring {frame_idx}/{frames_to_process}')

        cap.release()

        for tracker in trackers:
            tracker.finalize_states()

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
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
                fps,
                width,
                height,
                start_frame,
                end_frame,
                frames_to_process,
                trackers,
                global_start_time_s,
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
        self._log(f"FST video processed in {elapsed:.1f}s, {frame_idx} frames")

        import gc
        gc.collect()

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()
        _notify_analysis_status(self)

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        return {
            'session_id': self.session_id,
            'experiment_type': self.experiment_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'logs': self.logs[-10:],
            'realtime_data': self.realtime_data,
            'result_files': self.result_files,
            'experiment_stats': self.live_metrics,
            'stats': {
                'processed_frames': self.processed_frames,
                'elapsed_time': elapsed,
                'fps': processing_fps,
            }
        }

    def get_download_file_map(self):
        return self.output_groups

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


class NovelObjectMultiArenaAnalysisSession:
    """Multi-ROI Novel Object Recognition analysis session."""

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
        }

    def start(self):
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        _notify_analysis_status(self)
        thread = threading.Thread(target=_run_with_status_observer, args=(self,), daemon=True)
        thread.start()

    def _log(self, message):
        timestamp = china_now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

    def _append_result_file(self, path, group=None):
        if not path or not os.path.exists(path):
            return
        if path not in self.result_files:
            self.result_files.append(path)
        if group and path not in self.output_groups[group]:
            self.output_groups[group].append(path)

    def _config_for_video(self, video_index):
        vision_config = copy.deepcopy(self.config)
        per_video_configs = vision_config.pop('per_video_configs', None) or []
        video_index = int(video_index or 0)
        if 0 <= video_index < len(per_video_configs):
            per_video_config = per_video_configs[video_index] or {}
            for key, value in per_video_config.items():
                vision_config[key] = copy.deepcopy(value)
        return vision_config

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

    def _run(self):
        try:
            if len(self.video_paths) == 1:
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(f"Processing NOR video {i + 1}/{len(self.video_paths)}: {info.get('original_name', '')}")
                    self._analyze_single_video(path, info, batch_index=i)
                    self.progress = int((i + 1) / len(self.video_paths) * 100)

                if self.is_running:
                    self.status = 'completed'
                    self.end_time = time.time()
                    self.message = f'All {len(self.video_paths)} NOR videos finished.'
                    self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = f'Analysis error: {str(e)}'
            self._log(self.message)
            import traceback
            self._log(traceback.format_exc())

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
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

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to open video: {video_path}")
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log(f"NOR video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

        bg_method = vision_config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=vision_config.get('gray_threshold', 127),
                invert=vision_config.get('invert', False),
            )
        elif bg_method == 'static':
            bg = create_static_background(video_path, num_frames=15)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=vision_config.get('erosion_kernel', 3),
            erosion_iterations=vision_config.get('erosion_iter', 1),
            dilation_kernel=vision_config.get('dilation_kernel', 5),
            dilation_iterations=vision_config.get('dilation_iter', 2),
        )
        detector = MouseDetector(
            min_area=vision_config.get('min_area', 120),
            max_area=vision_config.get('max_area', 50000),
        )

        phase = vision_config.get('phase', 'recognition')
        arena_width_cm = float(vision_config.get('arena_width_cm', vision_config.get('arena_side_cm', 50.0)))
        arena_height_cm = float(vision_config.get('arena_height_cm', vision_config.get('arena_side_cm', 50.0)))
        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        roi_configs = self._parse_rois(vision_config)

        trackers = []
        for roi in roi_configs:
            trackers.append(
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
                    movement_threshold_cm_s=float(vision_config.get('movement_threshold_cm_s', 2.0)),
                    min_valid_exploration_s=float(vision_config.get('min_valid_exploration_s', 20.0)),
                )
            )
        self._log(f"Configured {len(trackers)} NOR arenas")

        writer = None
        if generate_video:
            for codec in ('avc1', 'H264', 'mp4v'):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    if test_writer.isOpened():
                        writer = test_writer
                        self._log(f"NOR video codec: {codec}")
                        break
                    test_writer.release()
                except Exception:
                    pass

        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start NOR analysis from frame {start_frame} ({start_frame / max(fps, 1e-6):.1f}s)")

        frames_to_process = max(0, end_frame - start_frame + 1)
        frame_idx = 0

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + frame_idx

            if current_frame_no > end_frame:
                break

            current_time = current_frame_no / max(fps, 1e-6)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = [tracker.update(frame, current_time, gray, bg_subtractor, morph_processor, detector) for tracker in trackers]

            if frame_idx % 15 == 0:
                rows = []
                for tracker, result in zip(trackers, results):
                    summary = tracker.build_summary()
                    rows.append({
                        'roi_id': tracker.roi_id,
                        'roi_name': tracker.roi_name,
                        'detected': bool(result.get('detected')),
                        'current_object': result.get('current_object_name') or '-',
                        'current_role': result.get('current_object_role') or '-',
                        'distance_cm': summary['total_distance_cm'],
                        'total_exploration_s': summary['total_object_exploration_s'],
                        'novel_exploration_s': summary.get('novel_exploration_s'),
                        'familiar_exploration_s': summary.get('familiar_exploration_s'),
                        'discrimination_index': summary.get('discrimination_index'),
                        'preference_index_percent': summary.get('preference_index_percent'),
                        'meets_min_exploration': bool(summary.get('meets_min_exploration', False)),
                        'in_stats_window': bool(result.get('in_stats_window', True)),
                    })
                self.realtime_data = rows
                self.live_metrics = self._build_live_metrics(rows)

            if generate_video and writer is not None:
                result_frame = draw_novel_object_results(frame, list(zip(trackers, results)))
                writer.write(result_frame)
                if frame_idx % 10 == 0:
                    self.current_frame = result_frame

            frame_idx += 1
            self.processed_frames += 1

            if frame_idx % 20 == 0:
                if batch_index is not None:
                    batch_progress = int(batch_index / len(self.video_paths) * 100)
                    video_progress = int(frame_idx / max(frames_to_process, 1) * 100 / len(self.video_paths))
                    self.progress = batch_progress + video_progress
                else:
                    self.progress = int(frame_idx / max(frames_to_process, 1) * 100)
                self.message = f'Frame {frame_idx}/{frames_to_process} ({self.progress}%)'

        cap.release()
        if writer is not None:
            writer.release()

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
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
        for tracker in trackers:
            roi_slug = sanitize_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"roi_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"

            if render_novel_object_heatmap(tracker, heatmap_work_path):
                exported_heatmap = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported_heatmap)
                self._append_result_file(exported_heatmap)
            else:
                self._log(f"Failed to save NOR heatmap: {heatmap_final_path}")

            if render_novel_object_trackmap(tracker, trackmap_work_path):
                exported_trackmap = _export_result_file(trackmap_work_path, trackmap_final_path)
                trackmap_work_paths.append(trackmap_work_path)
                trackmap_final_paths.append(exported_trackmap)
                self._append_result_file(exported_trackmap)
            else:
                self._log(f"Failed to save NOR trackmap: {trackmap_final_path}")

        if heatmap_work_paths:
            if len(heatmap_final_paths) == 1:
                self._append_result_file(heatmap_final_paths[0], 'heatmap')
            else:
                heatmap_grid_work_path = f"{output_base}_heatmaps.png"
                heatmap_grid_final_path = f"{final_base}_heatmaps.png"
                if render_open_field_image_grid(heatmap_work_paths, heatmap_grid_work_path, title=f"{video_name} | NOR Heatmaps"):
                    self._append_result_file(_export_result_file(heatmap_grid_work_path, heatmap_grid_final_path), 'heatmap')
                else:
                    self._log(f"Failed to save NOR heatmap grid: {heatmap_grid_final_path}")

        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                trackmap_grid_work_path = f"{output_base}_trackmaps.png"
                trackmap_grid_final_path = f"{final_base}_trackmaps.png"
                if render_open_field_image_grid(trackmap_work_paths, trackmap_grid_work_path, title=f"{video_name} | NOR Trajectory Maps"):
                    self._append_result_file(_export_result_file(trackmap_grid_work_path, trackmap_grid_final_path), 'trackmap')
                else:
                    self._log(f"Failed to save NOR trackmap grid: {trackmap_grid_final_path}")

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
        self._log(f"NOR video processed in {elapsed:.1f}s, {frame_idx} frames")

        import gc
        gc.collect()

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()
        _notify_analysis_status(self)

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        return {
            'session_id': self.session_id,
            'experiment_type': self.experiment_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'logs': self.logs[-10:],
            'realtime_data': self.realtime_data,
            'result_files': self.result_files,
            'experiment_stats': self.live_metrics,
            'stats': {
                'processed_frames': self.processed_frames,
                'elapsed_time': elapsed,
                'fps': processing_fps,
            }
        }

    def get_download_file_map(self):
        return self.output_groups

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
                    ])
                    f.write("\n")
        except Exception as e:
            print(f"Failed to write NOR report: {e}")


class ElevatedPlusMultiArenaAnalysisSession:
    """Multi-ROI Elevated Plus Maze analysis session."""

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths
        self.video_infos = video_infos
        self.config = config
        self.experiment_type = 'elevated_plus'

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

    def start(self):
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        _notify_analysis_status(self)
        thread = threading.Thread(target=_run_with_status_observer, args=(self,), daemon=True)
        thread.start()

    def _log(self, message):
        timestamp = china_now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

    def _append_result_file(self, path, group=None):
        if not path or not os.path.exists(path):
            return
        if path not in self.result_files:
            self.result_files.append(path)
        if group and path not in self.output_groups[group]:
            self.output_groups[group].append(path)

    def _config_for_video(self, video_index):
        vision_config = copy.deepcopy(self.config)
        per_video_configs = vision_config.pop('per_video_configs', None) or []
        video_index = int(video_index or 0)
        if 0 <= video_index < len(per_video_configs):
            per_video_config = per_video_configs[video_index] or {}
            for key, value in per_video_config.items():
                vision_config[key] = copy.deepcopy(value)
        return vision_config

    def _parse_rois(self, config=None):
        active_config = config or self.config
        rois = active_config.get('rois') or []
        if isinstance(rois, str):
            rois = json.loads(rois)

        parsed = []
        for idx, roi in enumerate(rois):
            if not isinstance(roi, dict):
                raise ValueError(f'EPM ROI {idx + 1} must be an object.')
            zone_polygons = _coerce_zone_polygons(roi.get('zones'), EPM_ZONE_KEYS, 'EPM')

            roi_name = str(roi.get('name') or f'EPM {idx + 1}').strip() or f'EPM {idx + 1}'
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
            raise ValueError('Elevated plus maze analysis requires at least one ROI.')
        return parsed

    @staticmethod
    def _build_live_metrics(rows):
        if not rows:
            return {}
        valid_open = [float(row['open_time_percent_total']) for row in rows if row.get('open_time_percent_total') is not None]
        return {
            'roi_count': len(rows),
            'detected_roi_count': sum(1 for row in rows if row.get('detected')),
            'total_distance_cm': round(sum(float(row.get('distance_cm', 0.0) or 0.0) for row in rows), 3),
            'mean_open_time_percent_total': round(sum(valid_open) / len(valid_open), 3) if valid_open else None,
            'total_open_entries': sum(int(row.get('open_entries') or 0) for row in rows),
        }

    def _run(self):
        try:
            if len(self.video_paths) == 1:
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(f"Processing EPM video {i + 1}/{len(self.video_paths)}: {info.get('original_name', '')}")
                    self._analyze_single_video(path, info, batch_index=i)
                    self.progress = int((i + 1) / len(self.video_paths) * 100)

                if self.is_running:
                    self.status = 'completed'
                    self.end_time = time.time()
                    self.message = f'All {len(self.video_paths)} EPM videos finished.'
                    self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = f'Analysis error: {str(e)}'
            self._log(self.message)
            import traceback
            self._log(traceback.format_exc())

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        video_start_time = time.time()
        video_index = 0 if batch_index is None else int(batch_index)
        vision_config = self._config_for_video(video_index)
        result_folder = vision_config.get('output_dir') or str(RESULT_DIR)
        output_info = _prepare_output_paths(
            self.session_id,
            video_info.get('original_name', 'video'),
            suffix='epm',
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
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log(f"EPM video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

        bg_method = vision_config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=vision_config.get('gray_threshold', 127),
                invert=vision_config.get('invert', False),
            )
        elif bg_method == 'static':
            bg = create_static_background(video_path, num_frames=15)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=vision_config.get('erosion_kernel', 3),
            erosion_iterations=vision_config.get('erosion_iter', 1),
            dilation_kernel=vision_config.get('dilation_kernel', 5),
            dilation_iterations=vision_config.get('dilation_iter', 2),
        )
        detector = MouseDetector(
            min_area=vision_config.get('min_area', 120),
            max_area=vision_config.get('max_area', 50000),
        )

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        roi_configs = self._parse_rois(vision_config)

        trackers = []
        for roi in roi_configs:
            trackers.append(
                ElevatedPlusTracker(
                    zone_polygons=roi['zones'],
                    detection_polygon=roi.get('detection_points'),
                    center_size_cm=float(vision_config.get('center_size_cm', 5.0)),
                    open_arm_length_cm=float(vision_config.get('open_arm_length_cm', 30.0)),
                    closed_arm_length_cm=float(vision_config.get('closed_arm_length_cm', 30.0)),
                    fps=fps,
                    movement_threshold_cm_s=float(vision_config.get('movement_threshold_cm_s', 2.0)),
                    entry_depth_ratio=float(vision_config.get('entry_depth_ratio', 0.2)),
                    roi_id=roi['roi_id'],
                    roi_name=roi['name'],
                    time_config=roi.get('time_config'),
                    global_start_time_s=global_start_time_s,
                )
            )
        self._log(f"Configured {len(trackers)} EPM mazes")

        writer = None
        if generate_video:
            for codec in ('avc1', 'H264', 'mp4v'):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    if test_writer.isOpened():
                        writer = test_writer
                        self._log(f"EPM video codec: {codec}")
                        break
                    test_writer.release()
                except Exception:
                    pass

        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start EPM analysis from frame {start_frame} ({start_frame / max(fps, 1e-6):.1f}s)")

        frames_to_process = max(0, end_frame - start_frame + 1)
        frame_idx = 0

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + frame_idx

            if current_frame_no > end_frame:
                break

            current_time = current_frame_no / max(fps, 1e-6)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = [tracker.update(frame, current_time, gray, bg_subtractor, morph_processor, detector) for tracker in trackers]

            if frame_idx % 15 == 0:
                rows = []
                for tracker, result in zip(trackers, results):
                    summary = tracker.build_summary()
                    rows.append({
                        'roi_id': tracker.roi_id,
                        'roi_name': tracker.roi_name,
                        'detected': bool(result.get('detected')),
                        'zone': result.get('zone_label', 'Unknown'),
                        'speed_cm_s': round(float(result.get('speed_cm_s', 0.0)), 3),
                        'distance_cm': summary['total_distance_cm'],
                        'open_time_percent_total': summary.get('open_time_percent_total'),
                        'open_entries': summary.get('open_entries'),
                        'closed_entries': summary.get('closed_entries'),
                        'open_entry_percent': summary.get('open_entry_percent'),
                        'in_stats_window': bool(result.get('in_stats_window', True)),
                    })
                self.realtime_data = rows
                self.live_metrics = self._build_live_metrics(rows)

            if generate_video and writer is not None:
                result_frame = draw_elevated_plus_results(frame, list(zip(trackers, results)))
                writer.write(result_frame)
                if frame_idx % 10 == 0:
                    self.current_frame = result_frame

            frame_idx += 1
            self.processed_frames += 1

            if frame_idx % 20 == 0:
                if batch_index is not None:
                    batch_progress = int(batch_index / len(self.video_paths) * 100)
                    video_progress = int(frame_idx / max(frames_to_process, 1) * 100 / len(self.video_paths))
                    self.progress = batch_progress + video_progress
                else:
                    self.progress = int(frame_idx / max(frames_to_process, 1) * 100)
                self.message = f'Frame {frame_idx}/{frames_to_process} ({self.progress}%)'

        cap.release()
        if writer is not None:
            writer.release()

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
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
            roi_slug = sanitize_epm_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"epm_{tracker.roi_id:02d}")
            heatmap_work_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_work_path = f"{output_base}_{roi_slug}_trackmap.png"
            heatmap_final_path = f"{final_base}_{roi_slug}_heatmap.png"
            trackmap_final_path = f"{final_base}_{roi_slug}_trackmap.png"

            if render_elevated_plus_heatmap(tracker, heatmap_work_path):
                exported_heatmap = _export_result_file(heatmap_work_path, heatmap_final_path)
                heatmap_work_paths.append(heatmap_work_path)
                heatmap_final_paths.append(exported_heatmap)
                self._append_result_file(exported_heatmap)
            else:
                self._log(f"Failed to save EPM heatmap: {heatmap_final_path}")

            if render_elevated_plus_trackmap(tracker, trackmap_work_path):
                exported_trackmap = _export_result_file(trackmap_work_path, trackmap_final_path)
                trackmap_work_paths.append(trackmap_work_path)
                trackmap_final_paths.append(exported_trackmap)
                self._append_result_file(exported_trackmap)
            else:
                self._log(f"Failed to save EPM trackmap: {trackmap_final_path}")

        if heatmap_work_paths:
            if len(heatmap_final_paths) == 1:
                self._append_result_file(heatmap_final_paths[0], 'heatmap')
            else:
                heatmap_grid_work_path = f"{output_base}_heatmaps.png"
                heatmap_grid_final_path = f"{final_base}_heatmaps.png"
                if render_elevated_plus_image_grid(heatmap_work_paths, heatmap_grid_work_path, title=f"{video_name} | EPM Heatmaps"):
                    self._append_result_file(_export_result_file(heatmap_grid_work_path, heatmap_grid_final_path), 'heatmap')
                else:
                    self._log(f"Failed to save EPM heatmap grid: {heatmap_grid_final_path}")

        if trackmap_work_paths:
            if len(trackmap_final_paths) == 1:
                self._append_result_file(trackmap_final_paths[0], 'trackmap')
            else:
                trackmap_grid_work_path = f"{output_base}_trackmaps.png"
                trackmap_grid_final_path = f"{final_base}_trackmaps.png"
                if render_elevated_plus_image_grid(trackmap_work_paths, trackmap_grid_work_path, title=f"{video_name} | EPM Trajectory Maps"):
                    self._append_result_file(_export_result_file(trackmap_grid_work_path, trackmap_grid_final_path), 'trackmap')
                else:
                    self._log(f"Failed to save EPM trackmap grid: {trackmap_grid_final_path}")

        if generate_video and os.path.exists(output_video_path):
            exported_video = _export_result_file(output_video_path, final_output_video_path)
            self._append_result_file(exported_video, 'video')
            if exported_video:
                self._log(f"EPM annotated video saved: {exported_video}")

        self.realtime_data = [
            {
                'roi_id': row['roi_id'],
                'roi_name': row['roi_name'],
                'detected': True,
                'zone': 'Completed',
                'speed_cm_s': row['mean_speed_cm_s'],
                'distance_cm': row['total_distance_cm'],
                'open_time_percent_total': row.get('open_time_percent_total'),
                'open_entries': row.get('open_entries'),
                'closed_entries': row.get('closed_entries'),
                'open_entry_percent': row.get('open_entry_percent'),
                'in_stats_window': True,
            }
            for row in summary_rows
        ]
        self.live_metrics = self._build_live_metrics(self.realtime_data)

        if batch_index is None and self.is_running:
            self.status = 'completed'
            self.end_time = time.time()
            self.progress = 100
            self.message = 'Elevated plus maze analysis completed.'

        elapsed = time.time() - video_start_time
        self._log(f"EPM video processed in {elapsed:.1f}s, {frame_idx} frames")

        import gc
        gc.collect()

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()
        _notify_analysis_status(self)

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        return {
            'session_id': self.session_id,
            'experiment_type': self.experiment_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'logs': self.logs[-10:],
            'realtime_data': self.realtime_data,
            'result_files': self.result_files,
            'experiment_stats': self.live_metrics,
            'stats': {
                'processed_frames': self.processed_frames,
                'elapsed_time': elapsed,
                'fps': processing_fps,
            }
        }

    def get_download_file_map(self):
        return self.output_groups

    def _generate_txt_report(self, summaries, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 72 + "\n")
                f.write("MouseScope 高架十字迷宫报告 (Elevated Plus Maze Report)\n")
                f.write("=" * 72 + "\n\n")
                f.write(f"{_bi_label('分析时间', 'Analysis Time')}: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if summaries:
                    f.write(f"{_bi_label('视频', 'Video')}: {summaries[0].get('video_name', '')}\n")
                f.write(f"{_bi_label('ROI 数量', 'ROI Count')}: {len(summaries)}\n")
                f.write(
                    f"{_bi_label('统计规则', 'Scoring Rule')}: "
                    "当质心越过各臂内部入口阈值时记为一次入臂 "
                    "(Arm entries are scored when the centroid crosses the inner entry threshold inside each arm.)\n\n"
                )

                for summary in summaries:
                    f.write(f"[ROI {summary.get('roi_id')}] {summary.get('roi_name', '')}\n")
                    _write_report_metric_lines(f, summary, [
                        ('总时长', 'Total Time', 'total_time_s', ' s'),
                        ('检测率', 'Detected Rate', 'detected_rate_percent', '%'),
                        ('总距离', 'Total Distance', 'total_distance_cm', ' cm'),
                        ('平均速度', 'Mean Speed', 'mean_speed_cm_s', ' cm/s'),
                        ('开放臂时间', 'Open Time', 'open_time_s', ' s'),
                        ('封闭臂时间', 'Closed Time', 'closed_time_s', ' s'),
                        ('中心区时间', 'Center Time', 'center_time_s', ' s'),
                        ('开放臂时间占比（总时长）', 'Open Time % (Total)', 'open_time_percent_total', '%'),
                        ('开放臂时间占比（仅双臂）', 'Open Time % (Arms Only)', 'open_time_percent_arms_only', '%'),
                        ('开放臂进入次数', 'Open Entries', 'open_entries', ''),
                        ('封闭臂进入次数', 'Closed Entries', 'closed_entries', ''),
                        ('开放臂进入占比', 'Open Entry %', 'open_entry_percent', '%'),
                        ('首次进入开放臂潜伏期', 'First Open-Entry Latency', 'first_open_entry_latency_s', ' s'),
                    ])
                    f.write("\n")
        except Exception as e:
            print(f"Failed to write EPM report: {e}")


class YMazeMultiArenaAnalysisSession:
    """Multi-ROI Y-maze spontaneous alternation analysis session."""

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

    def start(self):
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        _notify_analysis_status(self)
        thread = threading.Thread(target=_run_with_status_observer, args=(self,), daemon=True)
        thread.start()

    def _log(self, message):
        timestamp = china_now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

    def _append_result_file(self, path, group=None):
        if not path or not os.path.exists(path):
            return
        if path not in self.result_files:
            self.result_files.append(path)
        if group and path not in self.output_groups[group]:
            self.output_groups[group].append(path)

    def _config_for_video(self, video_index):
        vision_config = copy.deepcopy(self.config)
        per_video_configs = vision_config.pop('per_video_configs', None) or []
        video_index = int(video_index or 0)
        if 0 <= video_index < len(per_video_configs):
            per_video_config = per_video_configs[video_index] or {}
            for key, value in per_video_config.items():
                vision_config[key] = copy.deepcopy(value)
        return vision_config

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

    def _run(self):
        try:
            if len(self.video_paths) == 1:
                self._analyze_single_video(self.video_paths[0], self.video_infos[0])
            else:
                for i, (path, info) in enumerate(zip(self.video_paths, self.video_infos)):
                    if not self.is_running:
                        break
                    self._log(f"Processing Y-maze video {i + 1}/{len(self.video_paths)}: {info.get('original_name', '')}")
                    self._analyze_single_video(path, info, batch_index=i)
                    self.progress = int((i + 1) / len(self.video_paths) * 100)

                if self.is_running:
                    self.status = 'completed'
                    self.end_time = time.time()
                    self.message = f'All {len(self.video_paths)} Y-maze videos finished.'
                    self._log(self.message)
        except Exception as e:
            self.status = 'error'
            self.message = f'Analysis error: {str(e)}'
            self._log(self.message)
            import traceback
            self._log(traceback.format_exc())

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
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = vision_config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log(f"Y-maze video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

        bg_method = vision_config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=vision_config.get('gray_threshold', 127),
                invert=vision_config.get('invert', False),
            )
        elif bg_method == 'static':
            bg = create_static_background(video_path, num_frames=15)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=vision_config.get('erosion_kernel', 3),
            erosion_iterations=vision_config.get('erosion_iter', 1),
            dilation_kernel=vision_config.get('dilation_kernel', 5),
            dilation_iterations=vision_config.get('dilation_iter', 2),
        )
        detector = MouseDetector(
            min_area=vision_config.get('min_area', 120),
            max_area=vision_config.get('max_area', 50000),
        )

        start_sec = vision_config.get('start_sec')
        end_sec = vision_config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        roi_configs = self._parse_rois(vision_config)

        trackers = []
        for roi in roi_configs:
            trackers.append(
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
            )
        self._log(f"Configured {len(trackers)} Y-maze arenas")

        writer = None
        if generate_video:
            for codec in ('avc1', 'H264', 'mp4v'):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    if test_writer.isOpened():
                        writer = test_writer
                        self._log(f"Y-maze video codec: {codec}")
                        break
                    test_writer.release()
                except Exception:
                    pass

        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start Y-maze analysis from frame {start_frame} ({start_frame / max(fps, 1e-6):.1f}s)")

        frames_to_process = max(0, end_frame - start_frame + 1)
        frame_idx = 0

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            except Exception:
                current_frame_no = start_frame + frame_idx

            if current_frame_no > end_frame:
                break

            current_time = current_frame_no / max(fps, 1e-6)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = [tracker.update(frame, current_time, gray, bg_subtractor, morph_processor, detector) for tracker in trackers]

            if frame_idx % 15 == 0:
                rows = []
                for tracker, result in zip(trackers, results):
                    summary = tracker.build_summary()
                    rows.append({
                        'roi_id': tracker.roi_id,
                        'roi_name': tracker.roi_name,
                        'detected': bool(result.get('detected')),
                        'zone': result.get('zone_label', 'Unknown'),
                        'speed_cm_s': round(float(result.get('speed_cm_s', 0.0)), 3),
                        'distance_cm': summary['total_distance_cm'],
                        'total_arm_entries': summary.get('total_arm_entries'),
                        'alternation_count': summary.get('alternation_count'),
                        'alternation_percent': summary.get('alternation_percent'),
                        'in_stats_window': bool(result.get('in_stats_window', True)),
                    })
                self.realtime_data = rows
                self.live_metrics = self._build_live_metrics(rows)

            if generate_video and writer is not None:
                result_frame = draw_y_maze_results(frame, list(zip(trackers, results)))
                writer.write(result_frame)
                if frame_idx % 10 == 0:
                    self.current_frame = result_frame

            frame_idx += 1
            self.processed_frames += 1

            if frame_idx % 20 == 0:
                if batch_index is not None:
                    batch_progress = int(batch_index / len(self.video_paths) * 100)
                    video_progress = int(frame_idx / max(frames_to_process, 1) * 100 / len(self.video_paths))
                    self.progress = batch_progress + video_progress
                else:
                    self.progress = int(frame_idx / max(frames_to_process, 1) * 100)
                self.message = f'Frame {frame_idx}/{frames_to_process} ({self.progress}%)'

        cap.release()
        if writer is not None:
            writer.release()

        summary_rows = []
        track_rows = []
        bins_rows = []
        for tracker in trackers:
            summary = tracker.build_summary()
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
            else:
                self._log(f"Failed to save Y-maze heatmap: {heatmap_final_path}")

            if render_y_maze_trackmap(tracker, trackmap_work_path):
                exported_trackmap = _export_result_file(trackmap_work_path, trackmap_final_path)
                trackmap_work_paths.append(trackmap_work_path)
                trackmap_final_paths.append(exported_trackmap)
                self._append_result_file(exported_trackmap)
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
        self._log(f"Y-maze video processed in {elapsed:.1f}s, {frame_idx} frames")

        import gc
        gc.collect()

    def stop(self):
        self.is_running = False
        self.status = 'stopped'
        self.end_time = time.time()
        _notify_analysis_status(self)

    def get_status(self):
        elapsed = 0
        if self.start_time:
            elapsed = (self.end_time or time.time()) - self.start_time
        processing_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        return {
            'session_id': self.session_id,
            'experiment_type': self.experiment_type,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'logs': self.logs[-10:],
            'realtime_data': self.realtime_data,
            'result_files': self.result_files,
            'experiment_stats': self.live_metrics,
            'stats': {
                'processed_frames': self.processed_frames,
                'elapsed_time': elapsed,
                'fps': processing_fps,
            }
        }

    def get_download_file_map(self):
        return self.output_groups

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
