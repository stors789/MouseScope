"""Prepare independent mouse landmark labels, predict, and score held-out frames.

Run with the same Python as MouseScope. No network upload is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "core")]

import cv2
import numpy as np

from barnes_maze import estimate_silhouette_landmarks
from file_utils import write_image_unicode
from services.pose_model_profiles import DEFAULT_KEYPOINT_SCHEMA, SUPERANIMAL_MODEL_REF
from services.pose_tracking import normalise_pose_output
from services.video_frame_reader import read_video_frame
from services.video_metadata import probe_video_info
from vision_tracker import (
    BackgroundSubtractor,
    MorphologyProcessor,
    MouseDetector,
    create_temporal_background,
)

KEYPOINTS = ("nose", "left_ear", "right_ear", "body_center", "tail_base")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare(args):
    video = Path(args.video).resolve()
    info = probe_video_info(video)
    count = int(info["total_frames"])
    if count <= 0 or args.samples < 1:
        raise ValueError("视频无有效帧，或抽样帧数小于1。")
    start = max(0, int(round(args.start * info["fps"])))
    stop = min(
        count - 1,
        int(round(args.end * info["fps"])) if args.end is not None else count - 1,
    )
    if stop < start:
        raise ValueError("抽样时间窗不在视频内。")
    rois = []
    for index, raw in enumerate(args.roi or []):
        parts = raw.split(":")
        if len(parts) != 5:
            raise ValueError("--roi 格式为 名称:x:y:宽:高，可重复指定。")
        x, y, width, height = map(int, parts[1:])
        if (
            min(x, y) < 0
            or min(width, height) <= 0
            or x + width > info["width"]
            or y + height > info["height"]
        ):
            raise ValueError("验证裁剪框必须位于原始画面内。")
        rois.append(dict(name=parts[0], x=x, y=y, width=width, height=height))
    if not rois:
        rois = [
            dict(name="mouse", x=0, y=0, width=info["width"], height=info["height"])
        ]
    output = Path(args.output)
    if (output / "manifest.json").exists():
        raise ValueError(
            "输出目录已有验证集；请选择新目录，避免覆盖人工标注对应的画面。"
        )
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)
    records = []
    for frame_index in np.unique(
        np.linspace(start, stop, min(args.samples, stop - start + 1)).astype(int)
    ):
        frame = read_video_frame(str(video), int(frame_index), fps=info["fps"])
        if frame is None:
            raise ValueError(f"不能读取验证帧 {frame_index}。")
        for index, roi in enumerate(rois):
            x, y, w, h = [roi[k] for k in ["x", "y", "width", "height"]]
            crop = frame[y : y + h, x : x + w]
            record_id = f"f{frame_index:07d}_r{index:02d}"
            image_path = frames_dir / f"{record_id}.png"
            if not write_image_unicode(str(image_path), crop):
                raise ValueError(f"不能写入 {image_path}")
            records.append(
                dict(
                    id=record_id,
                    frame_index=int(frame_index),
                    time_s=float(frame_index / info["fps"]),
                    crop=roi,
                    image=f"frames/{image_path.name}",
                    image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    keypoints={},
                )
            )
    dataset = dict(
        schema_version=1,
        dataset_id=uuid.uuid4().hex,
        video_path=str(video),
        video_info=info,
        annotation_origin="unlabelled",
        coordinate_space="crop_pixels",
        intended_split="validation",
        records=records,
    )
    write_json(output / "manifest.json", dataset)
    html = (Path(__file__).with_name("landmark_validation.html")).read_text(
        encoding="utf-8"
    )
    html = html.replace(
        "__DATASET_JSON__",
        json.dumps(dataset, ensure_ascii=False).replace("<", "\\u003c"),
    )
    (output / "标注.html").write_text(html, encoding="utf-8")
    print(
        f'Prepared {len(records)} crops. Open {output / "标注.html"}; mark independent labels before viewing predictions.'
    )


def predict(args):
    if (
        not 0 <= args.confidence <= 1
        or args.min_area <= 0
        or not 0 <= args.diff_threshold <= 255
        or args.background_samples < 5
    ):
        raise ValueError("请检查关键点阈值、最小面积、差分阈值及背景抽样帧数。")
    manifest = read_json(args.manifest)
    config = {}
    if args.run_snapshot:
        snapshot = read_json(args.run_snapshot)
        config = snapshot.get("resolved_config") or snapshot
    reference = config.get("pose_model_reference") or SUPERANIMAL_MODEL_REF
    schema = config.get("pose_keypoint_schema") or DEFAULT_KEYPOINT_SCHEMA
    bg = morph = detector = runtime = None
    if args.backend == "classical":
        background, noise, _ = create_temporal_background(
            manifest["video_path"], num_samples=args.background_samples
        )
        bg = BackgroundSubtractor(
            method="temporal",
            static_bg=background,
            temporal_noise_map=noise,
            threshold=args.diff_threshold,
        )
        morph = MorphologyProcessor(
            erosion_kernel=3,
            erosion_iterations=1,
            dilation_kernel=5,
            dilation_iterations=2,
        )
        detector = MouseDetector(min_area=args.min_area, max_area=50000)
    else:
        from services.deeplabcut_runtime import DeepLabCutRuntime

        runtime = DeepLabCutRuntime()
        status = runtime.status()
        if not status.get("analysis_available"):
            raise ValueError(f'DLC不可用：{status.get("code")}')
    records = []
    try:
        for source in manifest["records"]:
            frame = read_video_frame(
                manifest["video_path"],
                source["frame_index"],
                fps=manifest["video_info"]["fps"],
            )
            if frame is None:
                raise ValueError(f'不能读取验证帧 {source["frame_index"]}')
            x, y, w, h = [int(source["crop"][k]) for k in ["x", "y", "width", "height"]]
            crop = frame[y : y + h, x : x + w]
            keypoints = {}
            image_sha256 = hashlib.sha256(
                cv2.imencode(".png", crop)[1].tobytes()
            ).hexdigest()
            if image_sha256 != source["image_sha256"]:
                raise ValueError(f'视频画面已改变，请重新导出验证集：{source["id"]}')
            reason = ""
            if runtime:
                payload = runtime.infer_frame(
                    crop,
                    model_reference=reference,
                    keypoint_schema=schema,
                    max_individuals=1,
                )
                individuals = normalise_pose_output(payload)
                if individuals:
                    selected = max(
                        individuals,
                        key=lambda obs: obs["keypoints"]
                        .get("body_center", {})
                        .get("confidence", 0),
                    )
                    keypoints = selected["keypoints"]
            else:
                foreground = bg.apply(
                    frame, gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                )
                detections = detector.detect(
                    morph.process(foreground[y : y + h, x : x + w])
                )
                if detections:
                    observation = estimate_silhouette_landmarks(
                        max(detections, key=lambda d: d["area"])["contour"]
                    )
                    keypoints = observation["keypoints"]
                    reason = observation["reason"]
            filtered = {
                key: dict(
                    p,
                    visible=bool(p.get("visible", True))
                    and 0 <= p["x"] < w
                    and 0 <= p["y"] < h
                    and float(p.get("confidence", 0)) >= args.confidence,
                )
                for key, p in keypoints.items()
            }
            records.append(
                dict(
                    id=source["id"],
                    frame_index=source["frame_index"],
                    image_sha256=source["image_sha256"],
                    keypoints=filtered,
                    reason=reason,
                )
            )
    finally:
        if runtime:
            runtime.close()
    write_json(
        args.output,
        dict(
            schema_version=1,
            dataset_id=manifest["dataset_id"],
            backend=args.backend,
            confidence_threshold=args.confidence,
            model_reference=reference if runtime else None,
            coordinate_space="crop_pixels",
            settings=dict(
                diff_threshold=args.diff_threshold,
                min_area=args.min_area,
                background_samples=args.background_samples,
            ),
            records=records,
        ),
    )
    print(
        f"Predicted {len(records)} crops with {args.backend}; this is not an accuracy score."
    )


def evaluate_labels(labels, predictions, tolerance_px):
    if labels.get("annotation_origin") != "human":
        raise ValueError("评估需要独立人工标注；不能将模型预测当作真实值。")
    if (
        labels.get("dataset_id") != predictions.get("dataset_id")
        or labels.get("coordinate_space") != "crop_pixels"
        or predictions.get("coordinate_space") != "crop_pixels"
    ):
        raise ValueError("标注与预测必须对应同一数据集和裁剪坐标。")
    if not math.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("误差容差必须为正有限像素数。")
    by_id = {r["id"]: r for r in predictions["records"]}
    details = []
    for record in labels["records"]:
        predicted = by_id.get(record["id"])
        if predicted is not None and predicted.get("image_sha256") != record.get(
            "image_sha256"
        ):
            raise ValueError(f'验证画面不匹配：{record["id"]}')
        for key in KEYPOINTS:
            label = record.get("keypoints", {}).get(key)
            if not label or not label.get("visible"):
                continue
            if not all(math.isfinite(float(label[k])) for k in ["x", "y"]):
                raise ValueError("可见人工关键点必须有有限坐标。")
            point = (predicted or {}).get("keypoints", {}).get(key)
            observed = bool(point and point.get("visible"))
            error = (
                math.dist(
                    [float(label[k]) for k in ["x", "y"]],
                    [float(point[k]) for k in ["x", "y"]],
                )
                if observed
                else None
            )
            details.append(
                dict(
                    id=record["id"],
                    frame_index=record["frame_index"],
                    keypoint=key,
                    predicted=observed,
                    error_px=error,
                    within_tolerance=error is not None and error <= tolerance_px,
                )
            )
    if not details:
        raise ValueError("没有可见的人工关键点可供评估。")
    summary = []
    for key in KEYPOINTS:
        rows = [r for r in details if r["keypoint"] == key]
        if not rows:
            continue
        errors = [r["error_px"] for r in rows if r["predicted"]]
        summary.append(
            dict(
                backend=predictions["backend"],
                keypoint=key,
                human_visible_points=len(rows),
                predicted_points=len(errors),
                coverage_percent=100 * len(errors) / len(rows),
                within_tolerance_percent=100
                * sum(r["within_tolerance"] for r in rows)
                / len(rows),
                tolerance_px=tolerance_px,
                median_error_px=float(np.median(errors)) if errors else None,
                p90_error_px=float(np.percentile(errors, 90)) if errors else None,
            )
        )
    return summary, details


def evaluate(args):
    labels, predictions = read_json(args.labels), read_json(args.predictions)
    summary, details = evaluate_labels(labels, predictions, args.tolerance_px)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in [("summary", summary), ("point_errors", details)]:
        with (directory / f"{name}.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(
        directory / "evaluation.json",
        dict(
            dataset_id=labels["dataset_id"],
            interpretation="Localization on independently labelled visible keypoints; not Barnes event validation.",
            summary=summary,
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "prepare",
        help="Export blank independent annotation frames and an offline HTML page",
    )
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--end", type=float)
    p.add_argument(
        "--roi", action="append", help="name:x:y:width:height; one mouse per crop"
    )
    p.set_defaults(func=prepare)
    p = sub.add_parser(
        "predict", help="Produce predictions without using manual labels"
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=["classical", "deeplabcut"], required=True)
    p.add_argument(
        "--run-snapshot",
        help="Reuse the resolved DLC reference and canonical schema from a saved run",
    )
    p.add_argument("--confidence", type=float, default=0.6)
    p.add_argument("--diff-threshold", type=int, default=25)
    p.add_argument("--min-area", type=int, default=120)
    p.add_argument("--background-samples", type=int, default=90)
    p.set_defaults(func=predict)
    p = sub.add_parser(
        "evaluate", help="Compare predictions with independent human labels"
    )
    p.add_argument("--labels", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--tolerance-px", type=float, default=5)
    p.add_argument("--output", required=True)
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
