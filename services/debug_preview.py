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
from light_dark_box import LightDarkBoxTracker, draw_light_dark_box_results
from three_chamber import ThreeChamberTracker, draw_three_chamber_results
from water_maze import WaterMazeTracker, draw_water_maze_results
from forced_swim import ForcedSwimTracker, draw_forced_swim_results
from services.payload_geometry import (
    circle_boundary_points as _circle_boundary_points,
    coerce_point_list as _coerce_point_list,
    coerce_point_xy as _point_xy,
    coerce_zone_polygons as _coerce_zone_polygons,
    coerce_zone_polygons_with_schema as _coerce_zone_polygons_with_schema,
    ellipse_boundary_points as _ellipse_boundary_points,
    normalize_water_maze_pool_payload as _normalize_pool_payload,
)
from services.video_frame_reader import read_video_frame


DETECTION_MASK_MIN_POINTS = 3
EPM_ZONE_KEYS = ('center', 'open_1', 'open_2', 'closed_1', 'closed_2')
YMAZE_ZONE_POINT_COUNTS = {'center': 3, 'arm_a': 4, 'arm_b': 4, 'arm_c': 4}

debug_static_bg_cache: dict[str, Any] = {}


def _now_ts() -> float:
    return time.time()


def _nonnegative_int(value, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label} 参数必须是非负整数')
    return max(0, parsed)


def _session_total_frames(sess: dict[str, Any], video_idx: int) -> int:
    infos = sess.get('video_infos') or []
    if 0 <= video_idx < len(infos) and isinstance(infos[video_idx], dict):
        try:
            return int(infos[video_idx].get('total_frames') or 0)
        except Exception:
            return 0
    return 0


def _session_video_fps(sess: dict[str, Any], video_idx: int) -> float | None:
    infos = sess.get('video_infos') or []
    if 0 <= video_idx < len(infos) and isinstance(infos[video_idx], dict):
        info = infos[video_idx]
        try:
            return float(info.get('detected_fps') or info.get('fps') or 0.0) or None
        except Exception:
            return None
    return None


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


def _debug_first_value(*values):
    for value in values:
        if value not in (None, ''):
            return value
    return None

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
            body_gray_threshold=int(data.get('body_gray_threshold', 95)),
            mouse_is_light=invert,
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


def _light_dark_debug_preview(frame, data, video_path):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    invert = data.get('bg_type', 'light') == 'dark'
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 120))
    rois = data.get('rois') or []
    dark_box_visible = str(data.get('dark_box_visible', True)).strip().lower() not in {'0', 'false', 'no', 'off', 'hidden', 'invisible'}
    if not rois:
        return jsonify({'error': 'Light-dark box preview requires at least one ROI.'}), 400
    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400
    morph = MorphologyProcessor(erosion_kernel=erosion_kernel, erosion_iterations=erosion_iter, dilation_kernel=dilation_kernel, dilation_iterations=dilation_iter)
    det = MouseDetector(min_area=min_area, max_area=int(data.get('max_area', 50000)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker_results = []
    detected_count = 0
    for idx, roi in enumerate(rois):
        if not isinstance(roi, dict):
            return jsonify({'error': f'LDB ROI {idx + 1} must be an object.'}), 400
        zones = roi.get('zones') or {}
        if not zones.get('light') or (dark_box_visible and not zones.get('dark')):
            required = 'light and dark zones' if dark_box_visible else 'a light zone'
            return jsonify({'error': f'LDB ROI {idx + 1} needs {required}.'}), 400
        detection_points = None
        if roi.get('detection_points'):
            try:
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)
            except Exception as exc:
                return jsonify({'error': f'LDB ROI {idx + 1}: detection mask {exc}'}), 400
        tracker = LightDarkBoxTracker(
            zone_polygons=zones,
            detection_polygon=detection_points,
            arena_width_cm=float(data.get('arena_width_cm', 40.0)),
            arena_height_cm=float(data.get('arena_height_cm', 40.0)),
            light_box_length_cm=float(data.get('light_box_length_cm', 40.0)),
            light_box_width_cm=float(data.get('light_box_width_cm', 20.0)),
            dark_box_length_cm=float(data.get('dark_box_length_cm', 40.0)),
            dark_box_width_cm=float(data.get('dark_box_width_cm', 20.0)),
            shared_edge_length_cm=float(data.get('shared_edge_length_cm', 40.0)),
            fps=30.0,
            door_open_delay_s=float(data.get('door_open_delay_s', 3.0)),
            transition_min_dwell_s=float(data.get('transition_min_dwell_s', 0.12)),
            doorway_reset_s=float(data.get('doorway_reset_s', 0.2)),
            movement_threshold_cm_s=float(data.get('movement_threshold_cm_s', 2.0)),
            dark_box_visible=dark_box_visible,
            roi_id=idx + 1,
            roi_name=roi.get('name') or f'LDB {idx + 1}',
            time_config=roi.get('time_config'),
        )
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))
    result_frame = draw_light_dark_box_results(frame, tracker_results)
    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detected': detected_count > 0,
        'detections': detected_count,
        'roi_count': len(tracker_results),
    })


def _three_chamber_debug_preview(frame, data, video_path):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    invert = data.get('bg_type', 'light') == 'dark'
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 120))
    rois = data.get('rois') or []
    if not rois:
        return jsonify({'error': 'Three-chamber preview requires at least one ROI.'}), 400
    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'无法创建背景模型: {e}'}), 400
    morph = MorphologyProcessor(erosion_kernel=erosion_kernel, erosion_iterations=erosion_iter, dilation_kernel=dilation_kernel, dilation_iterations=dilation_iter)
    det = MouseDetector(min_area=min_area, max_area=int(data.get('max_area', 50000)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker_results = []
    detected_count = 0
    for idx, roi in enumerate(rois):
        if not isinstance(roi, dict):
            return jsonify({'error': f'3CST ROI {idx + 1} must be an object.'}), 400
        zones = roi.get('zones') or {}
        if not zones.get('left') or not zones.get('center') or not zones.get('right'):
            return jsonify({'error': f'3CST ROI {idx + 1} needs left, center, and right chambers.'}), 400
        objects = roi.get('objects') or []
        if len(objects) < 2:
            return jsonify({'error': f'3CST ROI {idx + 1} needs left and right cup points.'}), 400
        detection_points = None
        if roi.get('detection_points'):
            try:
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)
            except Exception as exc:
                return jsonify({'error': f'3CST ROI {idx + 1}: detection mask {exc}'}), 400
        tracker = ThreeChamberTracker(
            zone_polygons=zones,
            objects=objects,
            detection_polygon=detection_points,
            stimulus_exclusion_polygons=[],
            role_schedule=roi.get('role_schedule') or data.get('role_schedule') or {},
            protocol_segments=roi.get('protocol_segments') or data.get('protocol_segments') or [],
            analysis_mode=data.get('analysis_mode', 'single_phase'),
            selected_phase=data.get('selected_phase') or data.get('phase') or 'sociability',
            arena_width_cm=float(data.get('arena_width_cm', 60.0)),
            arena_height_cm=float(data.get('arena_height_cm', 40.0)),
            fps=30.0,
            interaction_radius_cm=float(data.get('interaction_radius_cm', 2.0)),
            cup_exclusion_radius_cm=float(data.get('cup_exclusion_radius_cm', 1.5)),
            use_head_direction=bool(data.get('use_head_direction', False)),
            head_angle_threshold_deg=float(data.get('head_angle_threshold_deg', 60.0)),
            head_confidence_threshold=float(data.get('head_confidence_threshold', 0.35)),
            transition_min_dwell_s=float(data.get('transition_min_dwell_s', 0.12)),
            center_reset_s=float(data.get('center_reset_s', 0.2)),
            movement_threshold_cm_s=float(data.get('movement_threshold_cm_s', 2.0)),
            roi_id=idx + 1,
            roi_name=roi.get('name') or f'3CST {idx + 1}',
            time_config=roi.get('time_config'),
        )
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))
    result_frame = draw_three_chamber_results(frame, tracker_results)
    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detected': detected_count > 0,
        'detections': detected_count,
        'roi_count': len(tracker_results),
    })

def _forced_swim_debug_preview(frame, data, video_path):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 80))
    max_area = int(data.get('max_area', 50000))
    rois = data.get('rois') or []
    if not rois:
        return jsonify({'error': 'FST preview requires at least one cylinder ROI.'}), 400

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

    tracker_results = []
    detected_count = 0
    default_diameter_cm = float(data.get('cylinder_diameter_cm', 10.0))
    for idx, roi in enumerate(rois):
        if not isinstance(roi, dict):
            return jsonify({'error': f'FST ROI {idx + 1} must be an object.'}), 400
        points = roi.get('points') or []
        if len(points) < 3:
            return jsonify({'error': f'FST ROI {idx + 1} needs at least 3 polygon points.'}), 400
        diameter_points = roi.get('diameter_points') or []
        if len(diameter_points) != 2:
            return jsonify({'error': f'FST ROI {idx + 1} needs a cylinder diameter line.'}), 400
        detection_points = None
        if roi.get('detection_points'):
            try:
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)
            except Exception as exc:
                return jsonify({'error': f'FST ROI {idx + 1}: detection mask {exc}'}), 400

        try:
            tracker = ForcedSwimTracker(
                roi_points=[[p['x'], p['y']] if isinstance(p, dict) else p for p in points],
                diameter_points=[[p['x'], p['y']] if isinstance(p, dict) else p for p in diameter_points],
                cylinder_diameter_cm=float(roi.get('cylinder_diameter_cm') or default_diameter_cm),
                detection_points=detection_points,
                waterline_points=([[p['x'], p['y']] if isinstance(p, dict) else p for p in roi.get('waterline_points', [])] if roi.get('waterline_points') else None),
                fps=30.0,
                roi_id=idx + 1,
                roi_name=roi.get('name') or f'Cylinder {idx + 1}',
                time_config=roi.get('time_config'),
                auto_threshold=bool(data.get('auto_threshold', True)),
                immobility_threshold=data.get('immobility_threshold'),
                threshold_sensitivity=float(data.get('threshold_sensitivity', 0.0)),
                smoothing_window_s=float(data.get('smoothing_window_s', 0.5)),
                min_immobility_duration_s=float(data.get('min_immobility_duration_s', 1.0)),
                min_active_duration_s=float(data.get('min_active_duration_s', 0.3)),
                gap_merge_s=float(data.get('gap_merge_s', 0.5)),
            )
        except Exception as exc:
            return jsonify({'error': f'FST ROI {idx + 1}: {exc}'}), 400
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))

    result_frame = draw_forced_swim_results(frame, tracker_results, preview_detection=True)

    _, buf_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    _, buf_result = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return jsonify({
        'original': base64.b64encode(buf_orig).decode('utf-8'),
        'result': base64.b64encode(buf_result).decode('utf-8'),
        'detected': detected_count > 0,
        'detections': detected_count,
        'roi_count': len(tracker_results),
    })


def _water_maze_debug_preview(frame, data, video_path):
    bg_method = data.get('bg_method', 'threshold')
    gray_threshold = int(data.get('gray_threshold', 127))
    bg_type = data.get('bg_type', 'light')
    invert = (bg_type == 'dark')
    erosion_kernel = int(data.get('erosion_kernel', 3))
    erosion_iter = int(data.get('erosion_iter', 1))
    dilation_kernel = int(data.get('dilation_kernel', 5))
    dilation_iter = int(data.get('dilation_iter', 2))
    min_area = int(data.get('min_area', 80))
    max_area = int(data.get('max_area', 50000))
    rois = data.get('rois') or []
    if not rois:
        return jsonify({'error': 'MWM preview requires at least one pool ROI.'}), 400

    try:
        bg_sub = _build_debug_bg_subtractor(video_path, bg_method, gray_threshold, invert)
    except Exception as e:
        return jsonify({'error': f'Failed to build MWM background model: {e}'}), 400

    morph = MorphologyProcessor(
        erosion_kernel=erosion_kernel,
        erosion_iterations=erosion_iter,
        dilation_kernel=dilation_kernel,
        dilation_iterations=dilation_iter,
    )
    det = MouseDetector(min_area=min_area, max_area=max_area)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tracker_results = []
    detected_count = 0
    for idx, roi in enumerate(rois):
        if not isinstance(roi, dict):
            return jsonify({'error': f'MWM ROI {idx + 1} must be an object.'}), 400
        pool = roi.get('pool') if isinstance(roi.get('pool'), dict) else {}
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
        if not boundary or len(boundary) < 5:
            return jsonify({'error': f'MWM ROI {idx + 1} needs a pool ellipse or at least 5 pool boundary points.'}), 400

        platform = roi.get('platform') if isinstance(roi.get('platform'), dict) else {}
        platform_point = roi.get('platform_point') or platform.get('point') or platform.get('center')
        if platform_point is None and platform.get('x') is not None and platform.get('y') is not None:
            platform_point = platform
        if platform_point is None:
            return jsonify({'error': f'MWM ROI {idx + 1} needs a platform point.'}), 400

        detection_points = None
        if roi.get('detection_points'):
            try:
                detection_points = _coerce_point_list(roi.get('detection_points'), min_points=DETECTION_MASK_MIN_POINTS)
            except Exception as exc:
                return jsonify({'error': f'MWM ROI {idx + 1}: detection mask {exc}'}), 400
        platform_exclusion_points = None
        if roi.get('platform_exclusion_points'):
            try:
                platform_exclusion_points = _coerce_point_list(roi.get('platform_exclusion_points'), min_points=DETECTION_MASK_MIN_POINTS)
            except Exception as exc:
                return jsonify({'error': f'MWM ROI {idx + 1}: platform exclusion mask {exc}'}), 400

        try:
            pool_diameter_cm = float(roi.get('pool_diameter_cm') or pool.get('diameter_cm') or data.get('pool_diameter_cm', 120.0))
            quadrant_rotation_deg = float(
                _debug_first_value(
                    roi.get('quadrant_rotation_deg'),
                    roi.get('coordinate_rotation_deg'),
                    data.get('quadrant_rotation_deg'),
                    data.get('coordinate_rotation_deg'),
                    0.0,
                )
            )
            platform_radius_cm = float(platform.get('radius_cm') or roi.get('platform_radius_cm') or data.get('platform_radius_cm', 5.0))
            tracker = WaterMazeTracker(
                pool_boundary_points=[_point_xy(point) for point in boundary],
                pool=_normalize_pool_payload(pool, pool_diameter_cm),
                platform={
                    'center': _point_xy(platform_point),
                    'radius_cm': platform_radius_cm,
                },
                detection_points=detection_points,
                platform_exclusion_points=platform_exclusion_points,
                pool_diameter_cm=pool_diameter_cm,
                platform_radius_cm=platform_radius_cm,
                trial_type=data.get('trial_type', 'acquisition_hidden'),
                target_quadrant=_debug_first_value(roi.get('target_quadrant'), data.get('target_quadrant')),
                start_position=_debug_first_value(roi.get('start_position'), data.get('start_position'), 'N'),
                day_label=data.get('day_label', ''),
                trial_no=int(data.get('trial_no', 1)),
                fps=30.0,
                movement_threshold_cm_s=float(data.get('movement_threshold_cm_s', 2.0)),
                thigmotaxis_band_cm=float(data.get('thigmotaxis_band_cm', 10.0)),
                platform_dwell_s=float(data.get('platform_dwell_s', 1.0)),
                track_smoothing_max_jump_cm=float(data.get('track_smoothing_max_jump_cm', data.get('max_jump_cm', 25.0))),
                max_gap_s=float(data.get('max_gap_s', 0.5)),
                quadrant_rotation_deg=quadrant_rotation_deg,
                exclude_platform_from_detection=bool(data.get('exclude_platform_from_detection', False)),
                correction_mode=data.get('correction_mode', 'ellipse_affine'),
                distortion_k1=float(data.get('distortion_k1', 0.0)),
                distortion_k2=float(data.get('distortion_k2', 0.0)),
                roi_id=idx + 1,
                roi_name=roi.get('name') or f'Pool {idx + 1}',
                time_config=roi.get('time_config'),
            )
        except Exception as exc:
            return jsonify({'error': f'MWM ROI {idx + 1}: {exc}'}), 400
        result = tracker.update(frame, 0.0, gray, bg_sub, morph, det)
        tracker_results.append((tracker, result))
        detected_count += int(bool(result.get('detected')))

    result_frame = draw_water_maze_results(frame, tracker_results)

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
    try:
        frame_idx = _nonnegative_int(data.get('frame', 0), 'frame')
        video_idx = _nonnegative_int(data.get('video', 0), 'video')
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    video_paths = sess.get('video_paths') or []
    if not video_paths:
        return jsonify({'error': '会话没有视频文件'}), 404

    video_idx = min(video_idx, len(video_paths) - 1)
    total_frames = _session_total_frames(sess, video_idx)
    if total_frames > 0:
        frame_idx = min(frame_idx, total_frames - 1)

    video_path = video_paths[video_idx]
    frame = read_video_frame(video_path, frame_idx, fps=_session_video_fps(sess, video_idx))

    if frame is None:
        return jsonify({'error': '无法读取帧'}), 400

    experiment_type = sess.get('experiment_type')
    if experiment_type == 'open_field':
        return _open_field_debug_preview(frame, data, video_path)
    if experiment_type == 'novel_object':
        return _novel_object_debug_preview(frame, data, video_path)
    if experiment_type == 'elevated_plus':
        return _elevated_plus_debug_preview(frame, data, video_path)
    if experiment_type == 'y_maze':
        return _y_maze_debug_preview(frame, data, video_path)
    if experiment_type == 'light_dark_box':
        return _light_dark_debug_preview(frame, data, video_path)
    if experiment_type == 'three_chamber':
        return _three_chamber_debug_preview(frame, data, video_path)
    if experiment_type == 'forced_swim':
        return _forced_swim_debug_preview(frame, data, video_path)
    if experiment_type == 'water_maze':
        return _water_maze_debug_preview(frame, data, video_path)

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
