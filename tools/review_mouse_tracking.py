"""Render reproducible classical tracking, optionally alongside saved DLC points.

All classical processing is imported from core.classical_mouse, as used by the
Barnes session and preview. A manifest supplies video and single-mouse ROIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "core")]

import cv2
import numpy as np

from classical_mouse import ClassicalMouseDetector, create_classical_background
from file_utils import write_image_unicode
from services.video_export import create_annotated_video_writer
from services.video_metadata import probe_video_info
from services.video_orientation import open_video_capture
from tools.validate_landmarks import read_json, resolve_prediction_config, write_json

KEYS = ("nose", "body_center", "tail_base")
CODES = dict(nose="N", body_center="B", tail_base="T")
COLORS = dict(nose=(0, 230, 255), body_center=(40, 60, 255), tail_base=(70, 230, 70))


def inside(point, width, height):
    return bool(
        point
        and point.get("visible", True)
        and 0 <= point["x"] < width
        and 0 <= point["y"] < height
    )


def render_panel(canvas, image, points, box, offset, row_y, cutoff):
    x, y, w, h = box
    scale = min(640 / w, 420 / h)
    dw, dh = round(w * scale), round(h * scale)
    px = offset + 20 + (640 - dw) // 2
    py = row_y + (420 - dh) // 2
    canvas[py : py + dh, px : px + dw] = cv2.resize(
        image[y : y + h, x : x + w], (dw, dh)
    )
    for key in KEYS:
        point = points.get(key)
        if not inside(point, image.shape[1], image.shape[0]) or not (
            x <= point["x"] < x + w and y <= point["y"] < y + h
        ):
            continue
        qx = round(px + (point["x"] - x) * dw / w)
        qy = round(py + (point["y"] - y) * dh / h)
        good = point["confidence"] >= cutoff
        color = COLORS[key] if good else (165, 165, 165)
        cv2.circle(canvas, (qx, qy), 5, (10, 10, 10), -1 if good else 2, cv2.LINE_AA)
        cv2.circle(canvas, (qx, qy), 4, color, -1 if good else 1, cv2.LINE_AA)
        position = (
            min(offset + 635, max(offset + 20, qx + 8)),
            min(row_y + 416, max(row_y + 20, qy + 19)),
        )
        cv2.putText(
            canvas,
            CODES[key],
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (10, 10, 10),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            CODES[key],
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
    return scale, px - offset, py - row_y


def review(args):
    manifest = read_json(args.manifest)
    config, cutoff = resolve_prediction_config(args)
    config["head_confidence_threshold"] = cutoff
    if args.confidence is not None:
        config["pose_confidence_threshold"] = cutoff
    pose_cutoff = float(config["pose_confidence_threshold"])
    video_path = manifest["video_path"]
    oriented = open_video_capture(
        video_path, orientation_override=config.get("orientation_override")
    )
    try:
        info = probe_video_info(
            video_path, fps_override=config.get("fps_override"), cap=oriented
        )
    finally:
        oriented.release()
    fps = float(info["fps"])
    rois = []
    for record in manifest["records"]:
        roi = record["crop"]
        existing = next((r for r in rois if r["name"] == roi["name"]), None)
        if existing is not None and existing != roi:
            raise ValueError("ROI names must identify one fixed rectangle.")
        if existing is None:
            rois.append(dict(roi))
    if not 1 <= len(rois) <= 2:
        raise ValueError("Review supports one or two single-mouse ROIs.")
    for roi in rois:
        if (
            min(roi["x"], roi["y"]) < 0
            or min(roi["width"], roi["height"]) <= 0
            or roi["x"] + roi["width"] > info["width"]
            or roi["y"] + roi["height"] > info["height"]
        ):
            raise ValueError("A review ROI is outside the video.")
    segments = []
    for value in args.segment or [
        f'{config.get("start_sec") or 0}:{config.get("end_sec") or info["duration_seconds"]}'
    ]:
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError("Segment format is START:END seconds.")
        start, end = map(float, parts)
        if not (
            np.isfinite([start, end]).all()
            and 0 <= start < end <= info["duration_seconds"] + 0.01
        ):
            raise ValueError("Invalid video segment.")
        segments.append((start, end))
    predictions = {}
    if args.compare_predictions:
        metadata = read_json(args.prediction_metadata)
        if Path(metadata["source_video"]).resolve() != Path(video_path).resolve():
            raise ValueError("DLC predictions belong to another video.")
        for line in (
            Path(args.compare_predictions).read_text(encoding="utf-8-sig").splitlines()
        ):
            record = json.loads(line)
            key = (int(record["frame"]), record["roi"])
            if key in predictions:
                raise ValueError("Duplicate prediction frame/ROI.")
            predictions[key] = record["keypoints"]
    methods = ["classical"] + (["deeplabcut"] if args.compare_predictions else [])
    fixed = {}
    if args.viewports:
        fixed = {
            (int(r["clip"]) - 1, r["roi"]): tuple(r["box"])
            for r in read_json(args.viewports)["windows"]
        }
    for ci, _ in enumerate(segments):
        for roi in rois:
            if args.viewports and (ci, roi["name"]) not in fixed:
                raise ValueError("A requested fixed viewport is missing.")
            fixed.setdefault((ci, roi["name"]), (0, 0, roi["width"], roi["height"]))
            x, y, w, h = fixed[(ci, roi["name"])]
            if (
                min(x, y) < 0
                or min(w, h) <= 0
                or x + w > roi["width"]
                or y + h > roi["height"]
            ):
                raise ValueError("Viewport outside ROI.")
    if args.compare_predictions:
        for start, end in segments:
            for frame in range(round(start * fps), round(end * fps)):
                for roi in rois:
                    if (frame, roi["name"]) not in predictions:
                        raise ValueError(
                            f'Missing saved DLC input for frame {frame}, ROI {roi["name"]}.'
                        )
    out = Path(args.output)
    if (out / "review.mp4").exists():
        raise ValueError("Choose a new output directory; review.mp4 already exists.")
    out.mkdir(parents=True, exist_ok=True)
    start_frame = int(round(float(config.get("start_sec") or 0) * fps))
    end_frame = (
        None
        if config.get("end_sec") is None
        else int(round(config["end_sec"] * fps)) - 1
    )
    background = create_classical_background(
        video_path, config, start_frame=start_frame, end_frame=end_frame
    )
    detectors = [ClassicalMouseDetector(config) for _ in rois]
    width = 960 * len(methods)
    template = np.full((1080, width, 3), (34, 26, 20), np.uint8)
    cv2.putText(
        template,
        f'PROJECT METHOD {detectors[0].version} | body {config["body_erosion_kernel"]}x{config["body_erosion_kernel"]} x{config["body_erosion_iter"]}',
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (235, 240, 246),
        2,
        cv2.LINE_AA,
    )
    for mi, method in enumerate(methods):
        q = cutoff if method == "classical" else pose_cutoff
        cv2.putText(
            template,
            f"{method.upper()} cutoff {q:g}",
            (mi * 960 + 20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (235, 240, 246),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        template,
        "Cyan: body mask | Gray: preserved landmark evidence | N: nose B: body T: tail",
        (20, 1040),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 219, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        template,
        "Fixed view. Raw per-frame landmarks; no interpolation. Scores are not accuracy.",
        (20, 1068),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 219, 230),
        1,
        cv2.LINE_AA,
    )
    os.environ.setdefault("MOUSESCOPE_VIDEO_CRF", "20")
    writer = create_annotated_video_writer(
        out / "review.mp4",
        fps,
        width,
        1080,
        label="Project method review",
        async_write=True,
        max_queue=8,
        logger=lambda m: print(m, flush=True),
    )
    if writer is None:
        raise RuntimeError("Video writer unavailable.")
    cap = open_video_capture(
        video_path, orientation_override=config.get("orientation_override")
    )
    frames = 0
    transforms = {}
    stats = {
        r["name"]: dict(
            frames=0, body=0, nose=0, tail_base=0, head_pixels=0, body_pixels=0
        )
        for r in rois
    }
    fields = [
        "frame",
        "time_s",
        "clip",
        "roi",
        "method",
        "point",
        "x_roi_px",
        "y_roi_px",
        "score",
        "cutoff",
        "available",
        "method_version",
        "landmark_evidence",
    ]
    with (out / "points.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        csvout = csv.DictWriter(stream, fields)
        csvout.writeheader()
        try:
            for ci, (start, end) in enumerate(segments):
                first, last = round(start * fps), round(end * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                for frame_no in range(first, last):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"Cannot read source frame {frame_no}.")
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    fg = background.apply(frame, gray=gray)
                    bg = getattr(background, "static_bg", None)
                    canvas = template.copy()
                    cv2.putText(
                        canvas,
                        f"SOURCE {frame_no/fps:.2f}s | CLIP {ci+1}/{len(segments)} | {fps:g}FPS",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (235, 240, 246),
                        2,
                        cv2.LINE_AA,
                    )
                    for ri, roi in enumerate(rois):
                        x, y, w, h = [roi[k] for k in ["x", "y", "width", "height"]]
                        observation = detectors[ri].detect(
                            gray[y : y + h, x : x + w],
                            fg[y : y + h, x : x + w],
                            background=None if bg is None else bg[y : y + h, x : x + w],
                        )
                        points = observation["keypoints"]
                        state = stats[roi["name"]]
                        state["frames"] += 1
                        for key, field in [
                            ("nose", "nose"),
                            ("body_center", "body"),
                            ("tail_base", "tail_base"),
                        ]:
                            state[field] += int(
                                bool(
                                    points.get(key)
                                    and points[key]["confidence"] >= cutoff
                                )
                            )
                        state["head_pixels"] += observation["landmark_area_px"]
                        state["body_pixels"] += observation["body_area_px"]
                        image = frame[y : y + h, x : x + w].copy()
                        for key, color in [
                            ("landmark_contour", (150, 150, 150)),
                            ("body_contour", (255, 220, 0)),
                        ]:
                            if observation[key] is not None:
                                cv2.drawContours(
                                    image, [observation[key]], -1, color, 1
                                )
                        box = fixed[(ci, roi["name"])]
                        row_y = 128 if ri == 0 else 594
                        for mi, method in enumerate(methods):
                            q = cutoff if method == "classical" else pose_cutoff
                            values = (
                                points
                                if method == "classical"
                                else predictions[(frame_no, roi["name"])]
                            )
                            transform = render_panel(
                                canvas, image, values, box, mi * 960, row_y, q
                            )
                            identity = (ci, roi["name"])
                            if identity in transforms:
                                assert transforms[identity] == transform
                            transforms[identity] = transform
                            cv2.putText(
                                canvas,
                                roi["name"],
                                (mi * 960 + 690, row_y + 32),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (235, 240, 246),
                                2,
                                cv2.LINE_AA,
                            )
                            for ki, key in enumerate(KEYS):
                                p = values.get(key)
                                good = bool(inside(p, w, h) and p["confidence"] >= q)
                                visible = (
                                    inside(p, w, h)
                                    and box[0] <= p["x"] < box[0] + box[2]
                                    and box[1] <= p["y"] < box[1] + box[3]
                                )
                                label = (
                                    f'{CODES[key]} {p["confidence"]:.2f}'
                                    if p
                                    else f"{CODES[key]} MISSING"
                                )
                                if p and not visible:
                                    label += " OUT"
                                cv2.putText(
                                    canvas,
                                    label,
                                    (mi * 960 + 690, row_y + 90 + ki * 70),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    COLORS[key] if good else (165, 165, 165),
                                    2,
                                    cv2.LINE_AA,
                                )
                                csvout.writerow(
                                    dict(
                                        frame=frame_no,
                                        time_s=frame_no / fps,
                                        clip=ci + 1,
                                        roi=roi["name"],
                                        method=method,
                                        point=key,
                                        x_roi_px=p["x"] if p else "",
                                        y_roi_px=p["y"] if p else "",
                                        score=p["confidence"] if p else "",
                                        cutoff=q,
                                        available=good,
                                        method_version=(
                                            observation["classical_method_version"]
                                            if method == "classical"
                                            else metadata.get("model_reference", "")
                                        ),
                                        landmark_evidence=(
                                            observation["landmark_evidence"]
                                            if method == "classical"
                                            else "model_prediction"
                                        ),
                                    )
                                )
                    if frame_no < first + 12:
                        cv2.putText(
                            canvas,
                            "CUT",
                            (width - 110, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (90, 210, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    writer.write(canvas)
                    frames += 1
                    if frames == 1:
                        write_image_unicode(str(out / "preview.jpg"), canvas)
        finally:
            cap.release()
            writer.release()
    expected = sum(round(b * fps) - round(a * fps) for a, b in segments)
    video = probe_video_info(out / "review.mp4")
    if frames != expected or video["total_frames"] != expected:
        raise RuntimeError("Output frame count mismatch.")
    verify = cv2.VideoCapture(str(out / "review.mp4"))
    try:
        for index in [0, frames // 2, frames - 1]:
            verify.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, _ = verify.read()
            if not ok:
                raise RuntimeError(f"Output frame {index} cannot be decoded.")
    finally:
        verify.release()
    write_json(
        out / "review_config.json",
        dict(
            video_path=video_path,
            rois=rois,
            segments=segments,
            resolved_config=config,
            classic_cutoff=cutoff,
            dlc_cutoff=pose_cutoff,
            dlc_predictions=args.compare_predictions,
            prediction_metadata=args.prediction_metadata,
            viewports=[
                dict(clip=ci + 1, roi=name, box=box)
                for (ci, name), box in fixed.items()
            ],
        ),
    )
    write_json(
        out / "summary.json",
        dict(
            video=video,
            method_version=detectors[0].version,
            landmark_evidence=(
                "pre_body_erosion"
                if config["classical_method"] == "local_otsu_v1"
                else "legacy_processed_contour"
            ),
            stats=stats,
        ),
    )
    print(
        json.dumps(
            dict(
                output=str(out),
                method_version=detectors[0].version,
                frames=frames,
                stats=stats,
            ),
            indent=2,
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--segment",
        action="append",
        help="START:END seconds; repeat for multiple clips",
    )
    p.add_argument("--run-snapshot")
    p.add_argument("--video-index", type=int, default=0)
    p.add_argument("--confidence", type=float)
    p.add_argument("--classical-method", choices=["local_otsu_v1", "legacy"])
    p.add_argument("--segmentation-kernel", type=int)
    p.add_argument("--body-erosion-kernel", type=int)
    p.add_argument("--body-erosion-iter", type=int)
    p.add_argument("--diff-threshold", type=int)
    p.add_argument("--min-area", type=int)
    p.add_argument("--background-samples", type=int)
    p.add_argument("--compare-predictions")
    p.add_argument("--prediction-metadata")
    p.add_argument("--viewports")
    args = p.parse_args()
    args.backend = "classical"
    if args.compare_predictions and not args.prediction_metadata:
        p.error("--prediction-metadata is required with saved DLC predictions")
    try:
        review(args)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
