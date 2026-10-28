"""Single-frame Barnes preview uses the same geometry and detectors as a run."""

import base64

import cv2
from flask import jsonify

from barnes_maze import BarnesArena, BarnesTracker, draw_barnes_results
from vision_tracker import MorphologyProcessor, MouseDetector
from services.barnes_config import normalise_barnes_roi
from services.pose_tracking import PoseInferenceCoordinator


def build_barnes_preview(frame, data, video_path):
    from services.debug_preview import _build_debug_bg_subtractor

    coordinator = None
    try:
        rois = [
            normalise_barnes_roi(r, data, i)
            for i, r in enumerate(data.get("rois") or [])
        ]
        if not rois:
            raise ValueError("请先完成圆盘和全部洞位标定。")
        backend = data.get("tracking_backend", "classical")
        trackers = [
            BarnesTracker(
                BarnesArena(
                    r["arena_points"],
                    r["holes"],
                    r["target_hole_id"],
                    r["arena_diameter_cm"],
                    r["hole_radius_cm"],
                ),
                detection_points=r.get("detection_points"),
                roi_id=i + 1,
                roi_name=r["name"],
                tracking_backend=backend,
                head_confidence_threshold=(
                    data.get("pose_confidence_threshold", 0.8)
                    if backend == "deeplabcut"
                    else data.get("head_confidence_threshold", 0.6)
                ),
            )
            for i, r in enumerate(rois)
        ]
        bg = morph = detector = None
        if backend == "classical":
            bg = _build_debug_bg_subtractor(
                video_path,
                data.get("bg_method", "temporal"),
                data.get("gray_threshold", 127),
                data.get("bg_type") == "dark",
                data.get("_orientation_override"),
                temporal_bg_samples=data.get("temporal_bg_samples", 90),
                temporal_diff_threshold=data.get("temporal_diff_threshold", 25),
                temporal_start_frame=data.get("_temporal_start_frame", 0),
                temporal_end_frame=data.get("_temporal_end_frame"),
            )
            morph = MorphologyProcessor(
                erosion_kernel=data.get("erosion_kernel", 3),
                erosion_iterations=data.get("erosion_iter", 1),
                dilation_kernel=data.get("dilation_kernel", 5),
                dilation_iterations=data.get("dilation_iter", 2),
            )
            detector = MouseDetector(
                min_area=data.get("min_area", 120), max_area=data.get("max_area", 50000)
            )
        coordinator = (
            PoseInferenceCoordinator(data, trackers)
            if backend == "deeplabcut"
            else None
        )
        if backend == "deeplabcut" and not coordinator.enabled:
            raise ValueError("DLC未启用：请检查worker和模型。")
        pose = coordinator.infer(frame) if coordinator is not None else None
        if pose is not None and pose.error:
            raise ValueError(f"DLC预览失败：{pose.error}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = [
            t.update(
                frame,
                0,
                gray,
                bg,
                morph,
                detector,
                dt_s=1 / 30,
                pose_observation=(
                    pose.by_roi_id.get(t.roi_id) if pose is not None else None
                ),
            )
            for t in trackers
        ]
        annotated = draw_barnes_results(frame, trackers, results)

        def encode(im):
            return base64.b64encode(cv2.imencode(".jpg", im)[1]).decode("ascii")

        return jsonify(
            original=encode(frame),
            result=encode(annotated),
            detections=sum(bool(r["detected"]) for r in results),
            backend=backend,
            landmarks=[t.last_keypoints for t in trackers],
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 400
    finally:
        if coordinator is not None:
            coordinator.close()
