from __future__ import annotations

import base64
import os
import time
from typing import Any

import cv2
import numpy as np
from flask import jsonify

from vision_tracker import BackgroundSubtractor, MorphologyProcessor, MouseDetector, create_static_background
from open_field import OpenFieldTracker, draw_open_field_results
from novel_object import NovelObjectTracker, draw_novel_object_results
from elevated_plus import ElevatedPlusTracker, draw_elevated_plus_results
from y_maze import YMazeTracker, draw_y_maze_results
from services.payload_geometry import (
    coerce_point_list as _coerce_point_list,
    coerce_zone_polygons as _coerce_zone_polygons,
    coerce_zone_polygons_with_schema as _coerce_zone_polygons_with_schema,
)


DETECTION_MASK_MIN_POINTS = 3
EPM_ZONE_KEYS = ('center', 'open_1', 'open_2', 'closed_1', 'closed_2')
YMAZE_ZONE_POINT_COUNTS = {'center': 3, 'arm_a': 4, 'arm_b': 4, 'arm_c': 4}

debug_static_bg_cache: dict[str, Any] = {}


def _now_ts() -> float:
    return time.time()


def cleanup_debug_preview_cache(now: float, ttl_seconds: float, max_items: int) -> None:
    for cache_key, entry in list(debug_static_bg_cache.items()):
        if isinstance(entry, dict):
            last_access = float(entry.get('last_accessed_at') or now)
        else:
            last_access = now
        if now - last_access > float(ttl_seconds):
            debug_static_bg_cache.pop(cache_key, None)

    if len(debug_static_bg_cache) > int(max_items):
        ordered = sorted(
            debug_static_bg_cache.items(),
            key=lambda item: float(item[1].get('last_accessed_at', 0.0)) if isinstance(item[1], dict) else 0.0,
        )
        overflow = len(debug_static_bg_cache) - int(max_items)
        for cache_key, _ in ordered[:overflow]:
            debug_static_bg_cache.pop(cache_key, None)


def _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert):
    if bg_method == 'threshold':
        return BackgroundSubtractor(method='threshold', threshold=gray_threshold, invert=invert)
    if bg_method == 'static':
        cache_key = os.path.abspath(video_path)
        cache_entry = debug_static_bg_cache.get(cache_key)
        if isinstance(cache_entry, dict):
            static_bg = cache_entry.get('bg')
        else:
            static_bg = cache_entry
        if static_bg is None:
            static_bg = create_static_background(video_path, num_frames=15)
        debug_static_bg_cache[cache_key] = {
            'bg': static_bg,
            'last_accessed_at': _now_ts(),
        }
        bg_rgb = cv2.cvtColor(static_bg, cv2.COLOR_GRAY2BGR) if len(static_bg.shape) == 2 else static_bg
        return BackgroundSubtractor(method='static', static_bg=bg_rgb)
    return BackgroundSubtractor(method=bg_method)

def _open_field_debug_preview(frame, data, video_path):
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

    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400

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
            detection_points=(([[p['x'], p['y']] if isinstance(p, dict) else p for p in roi.get('detection_points', [])]) if isinstance(roi, dict) and roi.get('detection_points') else None),
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


def _novel_object_debug_preview(frame, data, video_path):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 120))
    rois = data.get('rois') or []
    if not rois:
        return jsonify({'error': 'Novel object preview requires at least one ROI.'}), 400

    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter
    )
    det = MouseDetector(min_area=min_area, max_area=50000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tracker_results = []
    detected_count = 0
    phase = data.get('phase', 'recognition')
    arena_width_cm = float(data.get('arena_width_cm', data.get('arena_side_cm', 50.0)))
    arena_height_cm = float(data.get('arena_height_cm', data.get('arena_side_cm', 50.0)))

    for idx, roi in enumerate(rois):
        points = roi.get('points', roi) if isinstance(roi, dict) else roi
        if len(points) != 4:
            return jsonify({'error': f'ROI {idx + 1} must contain exactly 4 arena corner points.'}), 400

        objects = roi.get('objects') if isinstance(roi, dict) else None
        if not objects or len(objects) != 2:
            return jsonify({'error': f'ROI {idx + 1} must contain exactly 2 object points.'}), 400

        tracker = NovelObjectTracker(
            arena_points=[[p['x'], p['y']] if isinstance(p, dict) else p for p in points],
            object_points=objects,
            detection_points=(([[p['x'], p['y']] if isinstance(p, dict) else p for p in roi.get('detection_points', [])]) if isinstance(roi, dict) and roi.get('detection_points') else None),
            arena_width_cm=arena_width_cm,
            arena_height_cm=arena_height_cm,
            fps=30.0,
            roi_id=idx + 1,
            roi_name=(roi.get('name') if isinstance(roi, dict) else None) or f'ROI {idx + 1}',
            phase=phase,
            time_config=(roi.get('time_config') if isinstance(roi, dict) else None),
            center_ratio_x=float(data.get('center_ratio_x', 0.5)),
            center_ratio_y=float(data.get('center_ratio_y', 0.5)),
            interaction_radius_cm=float(data.get('interaction_radius_cm', 4.0)),
            exploration_max_speed_cm_s=float(data.get('exploration_max_speed_cm_s', 12.0)),
            use_head_direction=bool(data.get('use_head_direction', True)),
            head_angle_threshold_deg=float(data.get('head_angle_threshold_deg', 50.0)),
            head_confidence_threshold=float(data.get('head_confidence_threshold', 0.35)),
            movement_threshold_cm_s=float(data.get('movement_threshold_cm_s', 2.0)),
            min_valid_exploration_s=float(data.get('min_valid_exploration_s', 20.0)),
        )
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))

    result_frame = draw_novel_object_results(frame, tracker_results)
    exploring_count = sum(1 for _, result in tracker_results if result.get('current_object_name'))

    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detected': detected_count > 0,
        'detections': detected_count,
        'roi_count': len(tracker_results),
        'exploring': exploring_count,
    })


def _elevated_plus_debug_preview(frame, data, video_path):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 120))
    rois = data.get('rois') or []
    if not rois:
        return jsonify({'error': 'Elevated plus maze preview requires at least one ROI.'}), 400

    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter
    )
    det = MouseDetector(min_area=min_area, max_area=50000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tracker_results = []
    detected_count = 0
    for idx, roi in enumerate(rois):
        if not isinstance(roi, dict):
            return jsonify({'error': f'EPM ROI {idx + 1} must be an object.'}), 400
        try:
            zone_polygons = _coerce_zone_polygons(roi.get('zones'), EPM_ZONE_KEYS, 'EPM')
        except Exception as exc:
            return jsonify({'error': f'EPM ROI {idx + 1}: {exc}'}), 400
        detection_points = None
        if isinstance(roi, dict) and roi.get('detection_points'):
            try:
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)
            except Exception as exc:
                return jsonify({'error': f'EPM ROI {idx + 1}: detection mask {exc}'}), 400

        tracker = ElevatedPlusTracker(
            zone_polygons=zone_polygons,
            detection_polygon=detection_points,
            center_size_cm=float(data.get('center_size_cm', 5.0)),
            open_arm_length_cm=float(data.get('open_arm_length_cm', 30.0)),
            closed_arm_length_cm=float(data.get('closed_arm_length_cm', 30.0)),
            fps=30.0,
            movement_threshold_cm_s=float(data.get('movement_threshold_cm_s', 2.0)),
            entry_depth_ratio=float(data.get('entry_depth_ratio', 0.2)),
            roi_id=idx + 1,
            roi_name=(roi.get('name') if isinstance(roi, dict) else None) or f'EPM {idx + 1}',
            time_config=(roi.get('time_config') if isinstance(roi, dict) else None),
        )
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))

    result_frame = draw_elevated_plus_results(frame, tracker_results)
    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detected': detected_count > 0,
        'detections': detected_count,
        'roi_count': len(tracker_results),
    })


def _y_maze_debug_preview(frame, data, video_path):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 120))
    rois = data.get('rois') or []
    if not rois:
        return jsonify({'error': 'Y-maze preview requires at least one ROI.'}), 400

    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter
    )
    det = MouseDetector(min_area=min_area, max_area=50000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tracker_results = []
    detected_count = 0
    for idx, roi in enumerate(rois):
        if not isinstance(roi, dict):
            return jsonify({'error': f'Y-maze ROI {idx + 1} must be an object.'}), 400
        try:
            zone_polygons = _coerce_zone_polygons_with_schema(roi.get('zones'), YMAZE_ZONE_POINT_COUNTS, 'Y-maze')
        except Exception as exc:
            return jsonify({'error': f'Y-maze ROI {idx + 1}: {exc}'}), 400
        detection_points = None
        if isinstance(roi, dict) and roi.get('detection_points'):
            try:
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)
            except Exception as exc:
                return jsonify({'error': f'Y-maze ROI {idx + 1}: detection mask {exc}'}), 400

        tracker = YMazeTracker(
            zone_polygons=zone_polygons,
            detection_polygon=detection_points,
            arm_length_cm=float(data.get('arm_length_cm', 30.0)),
            arm_width_cm=float(data.get('arm_width_cm', 10.0)),
            fps=30.0,
            movement_threshold_cm_s=float(data.get('movement_threshold_cm_s', 2.0)),
            entry_depth_ratio=float(data.get('entry_depth_ratio', 0.2)),
            roi_id=idx + 1,
            roi_name=(roi.get('name') if isinstance(roi, dict) else None) or f'Y-Maze {idx + 1}',
            time_config=(roi.get('time_config') if isinstance(roi, dict) else None),
        )
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))

    result_frame = draw_y_maze_results(frame, tracker_results)
    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detected': detected_count > 0,
        'detections': detected_count,
        'roi_count': len(tracker_results),
    })

def build_debug_preview_response(session_id: str, sess: dict[str, Any], data: dict[str, Any]):
    frame_idx = int(data.get('frame', 0))
    video_idx = int(data.get('video', 0))

    video_path = sess['video_paths'][min(video_idx, len(sess['video_paths']) - 1)]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': '无法读取帧'}), 500

    experiment_type = sess.get('experiment_type')
    if experiment_type == 'open_field':
        return _open_field_debug_preview(frame, data, video_path)
    if experiment_type == 'novel_object':
        return _novel_object_debug_preview(frame, data, video_path)
    if experiment_type == 'elevated_plus':
        return _elevated_plus_debug_preview(frame, data, video_path)
    if experiment_type == 'y_maze':
        return _y_maze_debug_preview(frame, data, video_path)

    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')

    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 2))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 3))
    min_area = int(data.get('min_area', 100))

    rois = data.get('rois', [])

    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter,
    )
    det = MouseDetector(min_area=min_area, max_area=50000)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = bg_sub.apply(frame, gray=gray)
    fg_mask = morph.process(fg_mask)

    if rois:
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for roi in rois:
            pts = roi if isinstance(roi, list) else roi.get('points', [])
            if pts:
                polygon = np.array([[p['x'], p['y']] for p in pts], dtype=np.int32)
                cv2.fillPoly(roi_mask, [polygon], 255)
        fg_mask = cv2.bitwise_and(fg_mask, roi_mask)

    detections = det.detect(fg_mask)

    result_frame = frame.copy()
    for detection in detections:
        cx, cy = detection['centroid']
        cv2.circle(result_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        if detection.get('contour') is not None:
            cv2.drawContours(result_frame, [detection['contour']], -1, (0, 255, 0), 2)

    if rois:
        for roi in rois:
            pts = roi if isinstance(roi, list) else roi.get('points', [])
            if pts:
                polygon = np.array([[p['x'], p['y']] for p in pts], dtype=np.int32)
                cv2.polylines(result_frame, [polygon], True, (255, 255, 0), 2)

    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detections': len(detections),
    })
