"""Single-frame Barnes preview uses the same geometry and detectors as a run."""

import base64

import cv2
from flask import jsonify

from barnes_maze import BarnesArena, BarnesTracker, draw_barnes_results
from services.barnes_config import normalise_barnes_roi
from services.pose_tracking import PoseInferenceCoordinator
from classical_mouse import create_classical_background


def build_barnes_preview(frame, data, video_path):
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
                classical_config=data,
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
            bg = create_classical_background(
                video_path,
                data,
                start_frame=data.get("_temporal_start_frame", 0),
                end_frame=data.get("_temporal_end_frame"),
                orientation_override=data.get("_orientation_override"),
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
