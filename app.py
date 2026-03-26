#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MouseScope Behavior Analysis Platform — 简化版 Flask 应用
去除：用户认证、数据库、YOLO、多语言
保留：视频上传、ROI标定、传统CV分析、结果下载
"""

import os
import sys

def _configure_console_streams():
    """Avoid startup crashes when Windows console encoding cannot represent log symbols."""
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, 'reconfigure'):
            continue
        try:
            stream.reconfigure(errors='replace')
        except Exception:
            pass


_configure_console_streams()

# ==================== 性能环境变量（必须在 import numpy/cv2 之前） ====================
# 让底层 BLAS/OpenMP 库充分利用所有 CPU 核心
_cpu_count = str(os.cpu_count() or 4)
os.environ.setdefault('OMP_NUM_THREADS', _cpu_count)
os.environ.setdefault('OPENBLAS_NUM_THREADS', _cpu_count)
os.environ.setdefault('MKL_NUM_THREADS', _cpu_count)

import secrets
import json
import cv2
import numpy as np
import base64
import threading
import time
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import deque
import subprocess
import platform
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, jsonify, send_file, session
)

# 将 core/ 加入路径（兼容 PyInstaller 打包环境）
if getattr(sys, 'frozen', False):
    # 打包后：core/ 在 sys._MEIPASS 解压目录下
    _base = sys._MEIPASS
    sys.path.insert(0, os.path.join(_base, 'core'))
    sys.path.insert(0, _base)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

from vision_tracker import (
    BackgroundSubtractor,
    MorphologyProcessor,
    MouseDetector,
    OpticalFlowAnalyzer,
    MouseTracker,
    draw_results,
    create_static_background
)
from open_field import (
    OpenFieldTracker,
    draw_open_field_result,
    draw_open_field_results,
    render_open_field_heatmap,
    render_open_field_image_grid,
    render_open_field_trackmap,
    sanitize_name,
)
from performance_utils import enable_opencv_optimizations

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠ pandas 未安装，将无法生成 CSV 报告")

# ==================== 配置 ====================
# PyInstaller 打包兼容：资源文件路径
if getattr(sys, 'frozen', False):
    # 打包后的 exe 运行环境
    BUNDLE_DIR = Path(sys._MEIPASS)          # 资源解压目录（templates, static 等）
    BASE_DIR = Path(sys.executable).parent   # exe 所在目录（用于存放用户数据）
else:
    BUNDLE_DIR = Path(__file__).parent
    BASE_DIR = BUNDLE_DIR

DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = DATA_DIR / 'uploads'
RESULT_DIR = DATA_DIR / 'results'

for d in [DATA_DIR, UPLOAD_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CHINA_TZ = timezone(timedelta(hours=8))

def china_now():
    return datetime.now(CHINA_TZ)

# ==================== 性能优化 ====================
enable_opencv_optimizations()

# ==================== Flask 应用 ====================
app = Flask(__name__,
            template_folder=str(BUNDLE_DIR / 'templates'),
            static_folder=str(BUNDLE_DIR / 'static'))
app.secret_key = secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB

# 会话存储（内存中，无数据库）
analysis_sessions = {}

# ==================== Jinja2 过滤器 ====================
@app.template_filter('filesizeformat')
def filesizeformat(value):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(value) < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"

@app.template_filter('basename')
def basename_filter(path):
    """获取文件名"""
    return os.path.basename(path)


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

    def start(self):
        """在后台线程中启动分析"""
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        thread = threading.Thread(target=self._run, daemon=True)
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
        vision_config = self.config

        # 结果文件夹（优先使用自定义输出目录）
        result_folder = self.config.get('output_dir') or str(RESULT_DIR)
        os.makedirs(result_folder, exist_ok=True)

        video_name = Path(video_info.get('original_name', 'video')).stem
        output_name = f"{video_name}_analyzed.mp4"
        output_path = os.path.join(result_folder, output_name)

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
            csv_path = output_path.replace('.mp4', '_logs.csv')
            pd.DataFrame(logs).to_csv(csv_path, index=False)
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
            immobility_csv = output_path.replace('.mp4', '_immobility.csv')
            pd.DataFrame(immobility_summaries).to_csv(immobility_csv, index=False)
            self.result_files.append(immobility_csv)
            self.immobility_csv_path = immobility_csv

            txt_path = output_path.replace('.mp4', '_report.txt')
            self._generate_txt_report(immobility_summaries, txt_path)
            self.result_files.append(txt_path)
            self.immobility_txt_path = txt_path

        # 保存分析视频
        if generate_video and os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024*1024)
            self._log(f"分析视频已保存: {output_path} ({file_size:.1f} MB)")
            self.result_files.append(output_path)

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
                f.write("=" * 60 + "\n")
                f.write("MouseScope 动物行为分析报告\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"分析时间: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"分析对象数量: {len(immobility_summaries)}\n\n")
                for i, summary in enumerate(immobility_summaries, 1):
                    roi_name = summary.get('roi_name', f"ROI {summary['roi_id']}")
                    f.write(f"对象 {i} ({roi_name}):\n")
                    f.write(f"  总时长: {summary['total_time_seconds']} 秒\n")
                    f.write(f"  不动时间: {summary['immobility_time_seconds']} 秒\n")
                    f.write(f"  不动比例: {summary['immobility_percent']}%\n")
                    f.write(f"  挣扎时间: {summary['struggling_time_seconds']} 秒\n")
                    f.write(f"  挣扎比例: {summary['struggling_percent']}%\n")
                    if summary.get('latency_to_immobility_seconds'):
                        f.write(f"  首次不动潜伏期: {summary['latency_to_immobility_seconds']} 秒\n")
                    f.write("\n")
        except Exception as e:
            print(f"生成TXT报告失败: {e}")


# ==================== Open Field ====================

class OpenFieldAnalysisSession:
    """Open Field Test analysis session."""

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

        self.video_path = None
        self.track_csv_path = None
        self.summary_csv_path = None
        self.bins_csv_path = None
        self.report_txt_path = None
        self.heatmap_path = None
        self.trackmap_path = None
        self.live_metrics = {}

    def start(self):
        self.is_running = True
        self.status = 'analyzing'
        self.start_time = time.time()
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

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

    def _parse_arena_points(self):
        points = self.config.get('arena_points', [])
        if isinstance(points, str):
            points = json.loads(points)
        parsed = []
        for point in points:
            if isinstance(point, dict):
                parsed.append([float(point['x']), float(point['y'])])
            else:
                parsed.append([float(point[0]), float(point[1])])
        if len(parsed) != 4:
            raise ValueError('Open field calibration requires 4 arena corner points.')
        return parsed

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        video_start_time = time.time()
        result_folder = self.config.get('output_dir') or str(RESULT_DIR)
        os.makedirs(result_folder, exist_ok=True)

        video_name = Path(video_info.get('original_name', 'video')).stem
        output_name = f"{video_name}_oft_analyzed.mp4"
        output_path = os.path.join(result_folder, output_name)
        generate_video = self.config.get('generate_video', True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to open video: {video_path}")
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = self.config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_path = output_path

        self._log(f"OFT video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

        bg_method = self.config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=self.config.get('gray_threshold', 127),
                invert=self.config.get('invert', False)
            )
        elif bg_method == 'static':
            bg = create_static_background(video_path, num_frames=15)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=self.config.get('erosion_kernel', 3),
            erosion_iterations=self.config.get('erosion_iter', 1),
            dilation_kernel=self.config.get('dilation_kernel', 5),
            dilation_iterations=self.config.get('dilation_iter', 2)
        )
        detector = MouseDetector(
            min_area=self.config.get('min_area', 120),
            max_area=self.config.get('max_area', 50000)
        )

        tracker = OpenFieldTracker(
            arena_points=self._parse_arena_points(),
            arena_width_cm=float(self.config.get('arena_width_cm', 50.0)),
            arena_height_cm=float(self.config.get('arena_height_cm', 50.0)),
            fps=fps,
            center_ratio_x=float(self.config.get('center_ratio_x', 0.5)),
            center_ratio_y=float(self.config.get('center_ratio_y', 0.5)),
            wall_margin_cm=float(self.config.get('wall_margin_cm', 8.0)),
            movement_threshold_cm_s=float(self.config.get('movement_threshold_cm_s', 2.0)),
        )

        writer = None
        if generate_video:
            for codec in ('mp4v', 'avc1', 'H264'):
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    test_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    if test_writer.isOpened():
                        writer = test_writer
                        self._log(f"OFT video codec: {codec}")
                        break
                    test_writer.release()
                except Exception:
                    pass

        start_sec = self.config.get('start_sec')
        end_sec = self.config.get('end_sec')
        start_frame = int(float(start_sec) * fps) if start_sec is not None else 0
        end_frame = int(float(end_sec) * fps) if end_sec is not None else (total_frames - 1)
        start_frame = max(0, min(start_frame, max(total_frames - 1, 0)))
        end_frame = max(0, min(end_frame, max(total_frames - 1, 0)))
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._log(f"Start OFT analysis from frame {start_frame} ({start_frame / max(fps, 1e-6):.1f}s)")

        frames_to_process = end_frame - start_frame + 1
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
            result = tracker.update(frame, current_time, gray, bg_subtractor, morph_processor, detector)

            if frame_idx % 15 == 0:
                summary = tracker.build_summary()
                self.live_metrics = {
                    'distance_cm': summary['total_distance_cm'],
                    'mean_speed_cm_s': summary['mean_speed_cm_s'],
                    'center_time_s': summary['center_time_s'],
                    'center_percent': summary['center_percent'],
                    'center_entries': summary['center_entries'],
                    'wall_time_s': summary['wall_time_s'],
                    'zone': result.get('zone_label', 'Unknown'),
                    'speed_cm_s': round(float(result.get('speed_cm_s', 0.0)), 3),
                }
                self.realtime_data = [self.live_metrics]

            if generate_video and writer is not None:
                writer.write(draw_open_field_result(frame, tracker, result))

            frame_idx += 1
            self.processed_frames = frame_idx

            if frame_idx % 20 == 0:
                if batch_index is not None:
                    batch_progress = int(batch_index / len(self.video_paths) * 100)
                    video_progress = int(frame_idx / max(frames_to_process, 1) * 100 / len(self.video_paths))
                    self.progress = batch_progress + video_progress
                else:
                    self.progress = int(frame_idx / max(frames_to_process, 1) * 100)
                self.message = f'Frame {frame_idx}/{frames_to_process} ({self.progress}%)'

            if frame_idx % 10 == 0:
                self.current_frame = frame

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

            for row in tracker.build_time_bins(bin_seconds=float(self.config.get('bin_seconds', 300.0))):
                bucket = dict(row)
                bucket['video_name'] = video_info.get('original_name', video_name)
                bins_rows.append(bucket)

        if HAS_PANDAS and track_rows:
            track_csv_path = f"{output_base}_track.csv"
            pd.DataFrame(track_rows).to_csv(track_csv_path, index=False)
            self._append_result_file(track_csv_path, 'track_csv')

        if HAS_PANDAS and summary_rows:
            summary_csv_path = f"{output_base}_summary.csv"
            pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
            self._append_result_file(summary_csv_path, 'summary_csv')

        if HAS_PANDAS and bins_rows:
            bins_csv_path = f"{output_base}_bins.csv"
            pd.DataFrame(bins_rows).to_csv(bins_csv_path, index=False)
            self._append_result_file(bins_csv_path, 'bins_csv')

        report_txt_path = f"{output_base}_report.txt"
        self._generate_txt_report(summary_rows, report_txt_path)
        self._append_result_file(report_txt_path, 'report')

        heatmap_paths = []
        trackmap_paths = []
        for tracker in trackers:
            roi_slug = sanitize_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"arena_{tracker.roi_id:02d}")
            heatmap_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_path = f"{output_base}_{roi_slug}_trackmap.png"

            if render_open_field_heatmap(tracker, heatmap_path):
                heatmap_paths.append(heatmap_path)
                self._append_result_file(heatmap_path)

            if render_open_field_trackmap(tracker, trackmap_path):
                trackmap_paths.append(trackmap_path)
                self._append_result_file(trackmap_path)

        if heatmap_paths:
            if len(heatmap_paths) == 1:
                self._append_result_file(heatmap_paths[0], 'heatmap')
            else:
                heatmap_grid_path = f"{output_base}_heatmaps.png"
                if render_open_field_image_grid(heatmap_paths, heatmap_grid_path, title=f"{video_name} | Heatmaps"):
                    self._append_result_file(heatmap_grid_path, 'heatmap')

        if trackmap_paths:
            if len(trackmap_paths) == 1:
                self._append_result_file(trackmap_paths[0], 'trackmap')
            else:
                trackmap_grid_path = f"{output_base}_trackmaps.png"
                if render_open_field_image_grid(trackmap_paths, trackmap_grid_path, title=f"{video_name} | Trajectory Maps"):
                    self._append_result_file(trackmap_grid_path, 'trackmap')

        if generate_video and os.path.exists(output_video_path):
            self._append_result_file(output_video_path, 'video')
            self._log(f"OFT annotated video saved: {output_video_path}")

        self.realtime_data = [
            {
                'roi_id': row['roi_id'],
                'roi_name': row['roi_name'],
                'detected': True,
                'zone': 'Ready',
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

        summary = tracker.build_summary()
        summary['video_name'] = video_info.get('original_name', video_name)

        if HAS_PANDAS and tracker.frame_records:
            self.track_csv_path = output_path.replace('.mp4', '_track.csv')
            pd.DataFrame(tracker.frame_records).to_csv(self.track_csv_path, index=False)
            self.result_files.append(self.track_csv_path)

            self.summary_csv_path = output_path.replace('.mp4', '_summary.csv')
            pd.DataFrame([summary]).to_csv(self.summary_csv_path, index=False)
            self.result_files.append(self.summary_csv_path)

            bin_rows = tracker.build_time_bins(bin_seconds=300.0)
            if bin_rows:
                self.bins_csv_path = output_path.replace('.mp4', '_bins.csv')
                pd.DataFrame(bin_rows).to_csv(self.bins_csv_path, index=False)
                self.result_files.append(self.bins_csv_path)

        self.report_txt_path = output_path.replace('.mp4', '_report.txt')
        self._generate_txt_report(summary, self.report_txt_path)
        if os.path.exists(self.report_txt_path):
            self.result_files.append(self.report_txt_path)

        self.heatmap_path = output_path.replace('.mp4', '_heatmap.png')
        if render_open_field_heatmap(tracker, self.heatmap_path):
            self.result_files.append(self.heatmap_path)

        self.trackmap_path = output_path.replace('.mp4', '_trackmap.png')
        if render_open_field_trackmap(tracker, self.trackmap_path):
            self.result_files.append(self.trackmap_path)

        if generate_video and os.path.exists(output_path):
            self.result_files.append(output_path)
            self._log(f"OFT annotated video saved: {output_path}")

        self.live_metrics = {
            'distance_cm': summary['total_distance_cm'],
            'mean_speed_cm_s': summary['mean_speed_cm_s'],
            'center_time_s': summary['center_time_s'],
            'center_percent': summary['center_percent'],
            'center_entries': summary['center_entries'],
            'wall_time_s': summary['wall_time_s'],
            'zone': self.live_metrics.get('zone', 'Unknown'),
            'speed_cm_s': self.live_metrics.get('speed_cm_s', 0.0),
        }

        if batch_index is None:
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
        return {
            'video': [self.video_path] if self.video_path else [],
            'track_csv': [self.track_csv_path] if self.track_csv_path else [],
            'summary_csv': [self.summary_csv_path] if self.summary_csv_path else [],
            'bins_csv': [self.bins_csv_path] if self.bins_csv_path else [],
            'report': [self.report_txt_path] if self.report_txt_path else [],
            'heatmap': [self.heatmap_path] if self.heatmap_path else [],
            'trackmap': [self.trackmap_path] if self.trackmap_path else [],
        }

    def _log(self, message):
        timestamp = china_now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(log_entry)

    def _generate_txt_report(self, summary, txt_path):
        try:
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("MouseScope Open Field Test Report\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Analysis time: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Video: {summary.get('video_name', '')}\n\n")
                for label, key, suffix in [
                    ('Total time', 'total_time_s', 's'),
                    ('Detected rate', 'detected_rate_percent', '%'),
                    ('Total distance', 'total_distance_cm', 'cm'),
                    ('Mean speed', 'mean_speed_cm_s', 'cm/s'),
                    ('Max speed', 'max_speed_cm_s', 'cm/s'),
                    ('Center time', 'center_time_s', 's'),
                    ('Center percent', 'center_percent', '%'),
                    ('Center entries', 'center_entries', ''),
                    ('Center latency', 'center_latency_s', 's'),
                    ('Wall time', 'wall_time_s', 's'),
                    ('Wall percent', 'wall_percent', '%'),
                    ('Outer time', 'outer_time_s', 's'),
                    ('Moving time', 'moving_time_s', 's'),
                    ('Immobility time', 'immobility_time_s', 's'),
                ]:
                    value = summary.get(key)
                    if value is None:
                        value = '-'
                        suffix = ''
                    f.write(f"{label}: {value}{suffix}\n")
        except Exception as e:
            print(f"Failed to write OFT report: {e}")


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
        thread = threading.Thread(target=self._run, daemon=True)
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

    def _parse_rois(self):
        rois = self.config.get('rois') or []
        if isinstance(rois, str):
            rois = json.loads(rois)

        if not rois and self.config.get('arena_points'):
            rois = [{
                'points': self.config.get('arena_points'),
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
        result_folder = self.config.get('output_dir') or str(RESULT_DIR)
        os.makedirs(result_folder, exist_ok=True)

        video_name = Path(video_info.get('original_name', 'video')).stem
        output_base = os.path.join(result_folder, f"{video_name}_oft")
        output_video_path = f"{output_base}_analyzed.mp4"
        generate_video = bool(self.config.get('generate_video', True))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"Unable to open video: {video_path}")
            return

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_override = self.config.get('fps_override')
        fps = fps_override if fps_override else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._log(f"OFT video info: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

        bg_method = self.config.get('bg_method', 'threshold')
        if bg_method == 'threshold':
            bg_subtractor = BackgroundSubtractor(
                method='threshold',
                threshold=self.config.get('gray_threshold', 127),
                invert=self.config.get('invert', False),
            )
        elif bg_method == 'static':
            bg = create_static_background(video_path, num_frames=15)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
            bg_subtractor = BackgroundSubtractor(method='static', static_bg=bg_rgb)
        else:
            bg_subtractor = BackgroundSubtractor(method=bg_method)

        morph_processor = MorphologyProcessor(
            erosion_kernel=self.config.get('erosion_kernel', 3),
            erosion_iterations=self.config.get('erosion_iter', 1),
            dilation_kernel=self.config.get('dilation_kernel', 5),
            dilation_iterations=self.config.get('dilation_iter', 2),
        )
        detector = MouseDetector(
            min_area=self.config.get('min_area', 120),
            max_area=self.config.get('max_area', 50000),
        )

        arena_side_cm = float(self.config.get('arena_side_cm', 50.0))
        start_sec = self.config.get('start_sec')
        end_sec = self.config.get('end_sec')
        global_start_time_s = float(start_sec) if start_sec is not None else 0.0
        roi_configs = self._parse_rois()

        trackers = []
        for roi in roi_configs:
            trackers.append(
                OpenFieldTracker(
                    arena_points=roi['points'],
                    arena_side_cm=arena_side_cm,
                    fps=fps,
                    center_ratio_x=float(self.config.get('center_ratio_x', 0.5)),
                    center_ratio_y=float(self.config.get('center_ratio_y', 0.5)),
                    wall_margin_cm=float(self.config.get('wall_margin_cm', 8.0)),
                    movement_threshold_cm_s=float(self.config.get('movement_threshold_cm_s', 2.0)),
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

            for row in tracker.build_time_bins(bin_seconds=float(self.config.get('bin_seconds', 300.0))):
                bucket = dict(row)
                bucket['video_name'] = video_info.get('original_name', video_name)
                bins_rows.append(bucket)

        if HAS_PANDAS and track_rows:
            track_csv_path = f"{output_base}_track.csv"
            pd.DataFrame(track_rows).to_csv(track_csv_path, index=False)
            self._append_result_file(track_csv_path, 'track_csv')

        if HAS_PANDAS and summary_rows:
            summary_csv_path = f"{output_base}_summary.csv"
            pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
            self._append_result_file(summary_csv_path, 'summary_csv')

        if HAS_PANDAS and bins_rows:
            bins_csv_path = f"{output_base}_bins.csv"
            pd.DataFrame(bins_rows).to_csv(bins_csv_path, index=False)
            self._append_result_file(bins_csv_path, 'bins_csv')

        report_txt_path = f"{output_base}_report.txt"
        self._generate_txt_report(summary_rows, report_txt_path)
        self._append_result_file(report_txt_path, 'report')

        heatmap_paths = []
        trackmap_paths = []
        for tracker in trackers:
            roi_slug = sanitize_name(f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback=f"arena_{tracker.roi_id:02d}")
            heatmap_path = f"{output_base}_{roi_slug}_heatmap.png"
            trackmap_path = f"{output_base}_{roi_slug}_trackmap.png"

            if render_open_field_heatmap(tracker, heatmap_path):
                heatmap_paths.append(heatmap_path)
                self._append_result_file(heatmap_path)

            if render_open_field_trackmap(tracker, trackmap_path):
                trackmap_paths.append(trackmap_path)
                self._append_result_file(trackmap_path)

        if heatmap_paths:
            if len(heatmap_paths) == 1:
                self._append_result_file(heatmap_paths[0], 'heatmap')
            else:
                heatmap_grid_path = f"{output_base}_heatmaps.png"
                if render_open_field_image_grid(heatmap_paths, heatmap_grid_path, title=f"{video_name} | Heatmaps"):
                    self._append_result_file(heatmap_grid_path, 'heatmap')

        if trackmap_paths:
            if len(trackmap_paths) == 1:
                self._append_result_file(trackmap_paths[0], 'trackmap')
            else:
                trackmap_grid_path = f"{output_base}_trackmaps.png"
                if render_open_field_image_grid(trackmap_paths, trackmap_grid_path, title=f"{video_name} | Trajectory Maps"):
                    self._append_result_file(trackmap_grid_path, 'trackmap')

        if generate_video and os.path.exists(output_video_path):
            self._append_result_file(output_video_path, 'video')
            self._log(f"OFT annotated video saved: {output_video_path}")

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
                f.write("MouseScope Open Field Test Report\n")
                f.write("=" * 68 + "\n\n")
                f.write(f"Analysis time: {china_now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if summaries:
                    f.write(f"Video: {summaries[0].get('video_name', '')}\n")
                f.write(f"Arena count: {len(summaries)}\n\n")

                for summary in summaries:
                    f.write(f"[ROI {summary.get('roi_id')}] {summary.get('roi_name', '')}\n")
                    f.write(f"  Time mode: {summary.get('time_mode', 'global')}\n")
                    if summary.get('time_start_s') is not None:
                        f.write(f"  Window start: {summary.get('time_start_s')} s\n")
                    if summary.get('time_end_s') is not None:
                        f.write(f"  Window end: {summary.get('time_end_s')} s\n")
                    for label, key, suffix in [
                        ('Total time', 'total_time_s', 's'),
                        ('Detected rate', 'detected_rate_percent', '%'),
                        ('Total distance', 'total_distance_cm', 'cm'),
                        ('Mean speed', 'mean_speed_cm_s', 'cm/s'),
                        ('Max speed', 'max_speed_cm_s', 'cm/s'),
                        ('Center time', 'center_time_s', 's'),
                        ('Center percent', 'center_percent', '%'),
                        ('Center entries', 'center_entries', ''),
                        ('Center latency', 'center_latency_s', 's'),
                        ('Wall time', 'wall_time_s', 's'),
                        ('Wall percent', 'wall_percent', '%'),
                        ('Outer time', 'outer_time_s', 's'),
                        ('Moving time', 'moving_time_s', 's'),
                        ('Immobility time', 'immobility_time_s', 's'),
                    ]:
                        value = summary.get(key)
                        if value is None:
                            value = '-'
                            suffix = ''
                        f.write(f"  {label}: {value}{suffix}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Failed to write OFT report: {e}")


# ==================== 路由 ====================

@app.route('/')
def index():
    """首页"""
    return render_template('home.html')


@app.route('/upload')
def upload_page():
    """上传视频页面"""
    return render_template('upload.html')


@app.route('/manual')
def manual():
    """使用手册"""
    manual_type = request.args.get('module', 'tst').strip().lower()
    if manual_type not in {'tst', 'oft'}:
        manual_type = 'tst'
    return render_template('manual.html', manual_type=manual_type)


@app.route('/manual/open_field')
def manual_open_field():
    """Open field manual."""
    return redirect(url_for('manual', module='oft'))


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

    valid_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    saved_paths = []
    upload_id = secrets.token_hex(8)

    for f in files:
        if f and f.filename:
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in valid_exts:
                continue
            safe_name = f.filename
            save_path = str(UPLOAD_DIR / f"{upload_id}_{safe_name}")
            f.save(save_path)
            saved_paths.append(save_path)

    if not saved_paths:
        return jsonify({'success': False, 'error': '没有有效的视频文件'}), 400

    return jsonify({'success': True, 'paths': saved_paths})


@app.route('/do_upload', methods=['POST'])
def do_upload():
    """处理视频上传"""
    session_id = secrets.token_hex(16)
    uploaded_files = request.files.getlist('videos')
    local_paths_raw = request.form.get('local_paths', '').strip()
    experiment_type = request.form.get('experiment_type', 'tst').strip() or 'tst'
    experiment_type = 'open_field' if experiment_type == 'open_field' else 'tst'

    video_paths = []
    video_infos = []

    # 处理上传的文件
    for f in uploaded_files:
        if f and f.filename:
            safe_name = f.filename
            save_path = str(UPLOAD_DIR / f"{session_id}_{safe_name}")
            f.save(save_path)
            video_paths.append(save_path)
            video_infos.append({'path': save_path, 'original_name': safe_name})

    # 处理本地路径
    if local_paths_raw:
        valid_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        for line in local_paths_raw.split('\n'):
            p = line.strip().strip('"').strip("'").strip('“”').strip('‘’').strip()
            if not p:
                continue
            p = Path(p).expanduser()
            if p.is_file() and p.suffix.lower() in valid_exts:
                video_paths.append(str(p))
                video_infos.append({'path': str(p), 'original_name': p.name})
            elif p.is_dir():
                for vf in sorted(p.iterdir()):
                    if vf.is_file() and vf.suffix.lower() in valid_exts:
                        video_paths.append(str(vf))
                        video_infos.append({'path': str(vf), 'original_name': vf.name})

    if not video_paths:
        flash('请选择或输入至少一个视频文件', 'error')
        return redirect(url_for('upload_page'))

    # 获取时间配置
    start_seconds = request.form.get('start_seconds', '').strip()
    end_seconds = request.form.get('end_seconds', '').strip()
    output_dir = request.form.get('output_dir', '').strip().strip('"').strip("'")
    fps_override = request.form.get('fps_override', '').strip()

    # 读取第一个视频的基本信息
    cap = cv2.VideoCapture(video_paths[0])
    detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_info_data = {
        'fps': float(fps_override) if fps_override else detected_fps,
        'detected_fps': detected_fps,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()

    # 保存到会话
    analysis_sessions[session_id] = {
        'experiment_type': experiment_type,
        'video_paths': video_paths,
        'video_infos': video_infos,
        'video_info': video_info_data,
        'start_seconds': start_seconds if start_seconds else None,
        'end_seconds': end_seconds if end_seconds else None,
        'output_dir': output_dir if output_dir else None,
        'fps_override': float(fps_override) if fps_override else None,
        'is_batch': len(video_paths) > 1,
    }

    return redirect(url_for('calibrate', session_id=session_id))


@app.route('/calibrate/<session_id>')
def calibrate(session_id):
    """标定页面（ROI + 参数调试）"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        flash('会话不存在或已过期，请重新上传', 'error')
        return redirect(url_for('index'))

    video_list = []
    if sess['is_batch']:
        for i, info in enumerate(sess['video_infos']):
            video_list.append({'index': i, 'name': info['original_name']})

    template_name = 'calibrate_openfield.html' if sess.get('experiment_type') == 'open_field' else 'calibrate.html'
    return render_template(
        template_name,
        session_id=session_id,
        video_info=sess['video_info'],
        video_path=sess['video_paths'][0],
        is_batch=sess['is_batch'],
        video_list=video_list,
        experiment_type=sess.get('experiment_type', 'tst'),
    )


@app.route('/monitor/<session_id>')
def monitor(session_id):
    """监控分析进度"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        flash('Session expired. Please start again.', 'error')
        return redirect(url_for('index'))
    template_name = 'monitor_openfield.html' if sess.get('experiment_type') == 'open_field' else 'monitor.html'
    return render_template(template_name, session_id=session_id)


# ==================== API 路由 ====================

@app.route('/api/get_first_frame/<session_id>')
def get_first_frame(session_id):
    """获取视频帧（用于标定页面）"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    frame_idx = int(request.args.get('frame', 0))
    video_idx = int(request.args.get('video', 0))

    video_path = sess['video_paths'][min(video_idx, len(sess['video_paths'])-1)]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': '无法读取帧'}), 500

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image': img_b64,
        'frame_idx': frame_idx,
        'total_frames': sess['video_info']['total_frames'],
    })


def _open_field_debug_preview(frame, data):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 120))
    max_area = int(data.get('max_area', 50000))
    rois = data.get('rois') or []
    if not rois and data.get('arena_points'):
        rois = [{
            'points': data.get('arena_points'),
            'name': 'Arena 1',
            'time_config': {'mode': 'global', 'start': None, 'end': None},
        }]
    if not rois:
        return jsonify({'error': 'Open field preview requires at least one ROI.'}), 400

    if bg_method == 'threshold':
        bg_sub = BackgroundSubtractor(method='threshold', threshold=gray_threshold, invert=invert)
    else:
        bg_sub = BackgroundSubtractor(method=bg_method)

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter
    )
    det = MouseDetector(min_area=min_area, max_area=max_area)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    arena_side_cm = float(data.get('arena_side_cm', data.get('arena_width_cm', 50.0)))

    tracker_results = []
    detected_count = 0
    for idx, roi in enumerate(rois):
        points = roi.get('points', roi) if isinstance(roi, dict) else roi
        if len(points) != 4:
            return jsonify({'error': f'ROI {idx + 1} must contain exactly 4 corner points.'}), 400

        tracker = OpenFieldTracker(
            arena_points=[[p['x'], p['y']] if isinstance(p, dict) else p for p in points],
            arena_side_cm=arena_side_cm,
            fps=30.0,
            center_ratio_x=float(data.get('center_ratio_x', 0.5)),
            center_ratio_y=float(data.get('center_ratio_y', 0.5)),
            wall_margin_cm=float(data.get('wall_margin_cm', 8.0)),
            movement_threshold_cm_s=float(data.get('movement_threshold_cm_s', 2.0)),
            roi_id=idx + 1,
            roi_name=(roi.get('name') if isinstance(roi, dict) else None) or f'Arena {idx + 1}',
            time_config=(roi.get('time_config') if isinstance(roi, dict) else None),
        )
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))

    result_frame = draw_open_field_results(frame, tracker_results)

    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detected': detected_count > 0,
        'detections': detected_count,
        'arena_count': len(tracker_results),
    })


@app.route('/api/debug_preview/<session_id>', methods=['POST'])
def debug_preview(session_id):
    """参数调试预览"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    data = request.get_json()
    frame_idx = int(data.get('frame', 0))
    video_idx = int(data.get('video', 0))

    video_path = sess['video_paths'][min(video_idx, len(sess['video_paths'])-1)]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': '无法读取帧'}), 500

    # 使用传统检测
    if sess.get('experiment_type') == 'open_field':
        return _open_field_debug_preview(frame, data)

    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')

    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 2))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 3))
    min_area = int(data.get('min_area', 100))

    # ROI 掩码
    rois = data.get('rois', [])

    if bg_method == 'threshold':
        bg_sub = BackgroundSubtractor(method='threshold', threshold=gray_threshold, invert=invert)
    else:
        bg_sub = BackgroundSubtractor(method=bg_method)

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter
    )
    det = MouseDetector(min_area=min_area, max_area=50000)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = bg_sub.apply(frame, gray=gray)
    fg_mask = morph.process(fg_mask)

    # 应用 ROI 掩码
    if rois:
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for roi in rois:
            pts = roi if isinstance(roi, list) else roi.get('points', [])
            if pts:
                polygon = np.array([[p['x'], p['y']] for p in pts], dtype=np.int32)
                cv2.fillPoly(roi_mask, [polygon], 255)
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

    detections = det.detect(fg_mask)

    # 绘制结果
    result_frame = frame.copy()
    for d in detections:
        cx, cy = d['centroid']
        cv2.circle(result_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        if d.get('contour') is not None:
            cv2.drawContours(result_frame, [d['contour']], -1, (0, 255, 0), 2)

    # 绘制 ROI
    if rois:
        for roi in rois:
            pts = roi if isinstance(roi, list) else roi.get('points', [])
            if pts:
                polygon = np.array([[p['x'], p['y']] for p in pts], dtype=np.int32)
                cv2.polylines(result_frame, [polygon], True, (255, 255, 0), 2)

    # 返回两张图
    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detections': len(detections),
    })


@app.route('/api/start_analysis/<session_id>', methods=['POST'])
def start_analysis(session_id):
    """启动分析"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    data = request.get_json()

    if sess.get('experiment_type') == 'open_field':
        vision_config = {
            'bg_method': data.get('bg_method', 'threshold'),
            'gray_threshold': int(data.get('gray_threshold', 127)),
            'invert': data.get('bg_type', 'light') == 'dark',
            'erosion_kernel': int(data.get('erosion_kernel', 3)),
            'erosion_iter': int(data.get('erosion_iter', 1)),
            'dilation_kernel': int(data.get('dilation_kernel', 5)),
            'dilation_iter': int(data.get('dilation_iter', 2)),
            'min_area': int(data.get('min_area', 120)),
            'max_area': int(data.get('max_area', 50000)),
            'rois': data.get('rois', []),
            'arena_points': data.get('arena_points', []),
            'arena_side_cm': float(data.get('arena_side_cm', data.get('arena_width_cm', 50.0))),
            'center_ratio_x': float(data.get('center_ratio_x', 0.5)),
            'center_ratio_y': float(data.get('center_ratio_y', 0.5)),
            'wall_margin_cm': float(data.get('wall_margin_cm', 8.0)),
            'movement_threshold_cm_s': float(data.get('movement_threshold_cm_s', 2.0)),
            'bin_seconds': float(data.get('bin_seconds', 300.0)),
            'generate_video': data.get('generate_video', True),
            'output_dir': sess.get('output_dir'),
            'start_sec': float(sess['start_seconds']) if sess.get('start_seconds') else None,
            'end_sec': float(sess['end_seconds']) if sess.get('end_seconds') else None,
            'fps_override': sess.get('fps_override'),
        }

        analysis = OpenFieldMultiArenaAnalysisSession(
            session_id=session_id,
            video_paths=sess['video_paths'],
            video_infos=sess['video_infos'],
            config=vision_config
        )

        analysis_sessions[session_id]['analysis'] = analysis
        analysis.start()
        return jsonify({'success': True, 'session_id': session_id})

    vision_config = {
        'bg_method': data.get('bg_method', 'threshold'),
        'gray_threshold': int(data.get('gray_threshold', 127)),
        'invert': data.get('bg_type', 'light') == 'dark',
        'erosion_kernel': int(data.get('erosion_kernel', 3)),
        'erosion_iter': int(data.get('erosion_iter', 2)),
        'dilation_kernel': int(data.get('dilation_kernel', 5)),
        'dilation_iter': int(data.get('dilation_iter', 3)),
        'min_area': int(data.get('min_area', 100)),
        'pixel_to_mm': float(data.get('pixel_to_mm', 1.0)),
        'rois': data.get('rois', []),
        'generate_video': data.get('generate_video', True),
        'output_dir': sess.get('output_dir'),
        'start_sec': float(sess['start_seconds']) if sess.get('start_seconds') else None,
        'end_sec': float(sess['end_seconds']) if sess.get('end_seconds') else None,
        'fps_override': sess.get('fps_override'),
    }

    # 创建分析会话
    analysis = VisionAnalysisSession(
        session_id=session_id,
        video_paths=sess['video_paths'],
        video_infos=sess['video_infos'],
        config=vision_config
    )

    analysis_sessions[session_id]['analysis'] = analysis
    analysis.start()

    return jsonify({'success': True, 'session_id': session_id})


@app.route('/api/status/<session_id>')
def get_status(session_id):
    """获取分析状态"""
    sess = analysis_sessions.get(session_id)
    if not sess or 'analysis' not in sess:
        return jsonify({'error': '会话不存在'}), 404
    return jsonify(sess['analysis'].get_status())


@app.route('/api/stop/<session_id>', methods=['POST'])
def stop_analysis(session_id):
    """停止分析"""
    sess = analysis_sessions.get(session_id)
    if sess and 'analysis' in sess:
        sess['analysis'].stop()
        return jsonify({'message': '已发送停止命令'})
    return jsonify({'error': '会话不存在'}), 404


@app.route('/api/download_all/<session_id>')
def download_all(session_id):
    """下载所有结果（ZIP）"""
    sess = analysis_sessions.get(session_id)
    if not sess or 'analysis' not in sess:
        return jsonify({'error': '会话不存在'}), 404

    analysis = sess['analysis']
    if not analysis.result_files:
        return jsonify({'error': '没有结果文件'}), 404

    zip_path = str(RESULT_DIR / f'{session_id}_results.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in analysis.result_files:
            if os.path.exists(f):
                zf.write(f, os.path.basename(f))

    return send_file(zip_path, as_attachment=True, download_name='MouseScope_results.zip')


@app.route('/api/download/<session_id>/<file_type>')
def download_file(session_id, file_type):
    """下载单个结果文件"""
    sess = analysis_sessions.get(session_id)
    if not sess or 'analysis' not in sess:
        return jsonify({'error': '会话不存在'}), 404

    analysis = sess['analysis']
    if hasattr(analysis, 'get_download_file_map'):
        file_map = analysis.get_download_file_map()
    else:
        file_map = {
            'video': [f for f in analysis.result_files if f.endswith('.mp4')],
            'csv': [analysis.csv_path] if analysis.csv_path else [],
            'immobility_csv': [analysis.immobility_csv_path] if analysis.immobility_csv_path else [],
            'report': [analysis.immobility_txt_path] if analysis.immobility_txt_path else [],
        }

    files = file_map.get(file_type, [])
    files = [f for f in files if f and os.path.exists(f)]

    if not files:
        return jsonify({'error': '文件不存在'}), 404

    if len(files) == 1:
        return send_file(files[0], as_attachment=True)

    result_dir = sess.get('output_dir') or str(RESULT_DIR)
    os.makedirs(result_dir, exist_ok=True)
    zip_path = os.path.join(result_dir, f'{session_id}_{file_type}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, os.path.basename(path))
    return send_file(zip_path, as_attachment=True)


@app.route('/api/open_folder/<session_id>', methods=['POST'])
def open_folder(session_id):
    """用系统文件管理器打开结果文件夹（桌面端专属功能）"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    # 确定结果文件夹
    result_dir = sess.get('output_dir') or str(RESULT_DIR)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    try:
        system = platform.system()
        if system == 'Linux':
            subprocess.Popen(['xdg-open', result_dir])
        elif system == 'Darwin':
            subprocess.Popen(['open', result_dir])
        elif system == 'Windows':
            os.startfile(result_dir)
        else:
            return jsonify({'error': f'不支持的操作系统: {system}'}), 500

        return jsonify({'success': True, 'path': result_dir})
    except Exception as e:
        return jsonify({'error': f'打开文件夹失败: {str(e)}', 'path': result_dir}), 500


@app.route('/api/result_dir/<session_id>')
def get_result_dir(session_id):
    """获取结果文件夹路径"""
    sess = analysis_sessions.get(session_id)
    if not sess:
        return jsonify({'error': '会话不存在'}), 404

    result_dir = sess.get('output_dir') or str(RESULT_DIR)
    return jsonify({'path': result_dir})





# ==================== 启动 ====================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MouseScope Behavior Analysis Platform')
    parser.add_argument('--port', type=int, default=8080, help='端口号')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--no-browser', action='store_true', help='不自动打开浏览器')
    args = parser.parse_args()

    # 获取本机局域网IP
    def get_local_ip():
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return '127.0.0.1'

    local_ip = get_local_ip()
    browse_url = f'http://{local_ip}:{args.port}'

    print(f"\n{'='*60}")
    print(f"  MouseScope Behavior Analysis Platform")
    print(f"  访问地址: {browse_url}")
    print(f"{'='*60}\n")

    if not args.no_browser:
        import webbrowser
        import socket as _socket

        def _wait_and_open(url, host, port, timeout=30):
            """等待 Flask 端口真正就绪后再打开浏览器，避免竞态条件"""
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    with _socket.create_connection((host if host != '0.0.0.0' else '127.0.0.1', port), timeout=0.5):
                        pass
                    webbrowser.open(url)
                    return
                except OSError:
                    time.sleep(0.2)
            # 超时仍未就绪，兜底直接打开
            webbrowser.open(url)

        threading.Thread(
            target=_wait_and_open,
            args=(browse_url, args.host, args.port),
            daemon=True
        ).start()

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
