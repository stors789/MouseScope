"""Barnes video analysis using source timestamps and explicit landmark backends."""

import csv
import json

from services.analysis_runtime import *
from services.pose_tracking import PoseInferenceCoordinator
from services.barnes_config import normalise_barnes_roi
from barnes_maze import (
    BarnesArena,
    BarnesTracker,
    draw_barnes_results,
    render_barnes_map,
)
from file_utils import ascii_slug
from classical_mouse import create_classical_background


class BarnesMultiArenaAnalysisSession(AnalysisSessionBase):
    batch_progress_template = "Processing Barnes video {index}/{total}: {name}"
    batch_complete_template = "All {total} Barnes videos finished."

    def __init__(self, session_id, video_paths, video_infos, config):
        self.session_id = session_id
        self.video_paths = video_paths
        self.video_infos = video_infos
        self.config = config
        self.experiment_type = "barnes_maze"
        self.status = "waiting"
        self.progress = 0
        self.message = "Waiting to start..."
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
            k: []
            for k in [
                "video",
                "track_csv",
                "summary_csv",
                "events_csv",
                "bins_csv",
                "report",
                "heatmap",
                "trackmap",
            ]
        }

    def _parse_rois(self, config=None):
        active = config or self.config
        rois = active.get("rois") or []
        if not rois:
            raise ValueError("请先标定巴恩斯圆盘和全部洞位。")
        return [normalise_barnes_roi(r, active, i) for i, r in enumerate(rois)]

    @staticmethod
    def _live_row(tracker, result):
        s = tracker.build_summary()
        return dict(
            roi_id=tracker.roi_id,
            roi_name=tracker.roi_name,
            detected=result["detected"],
            nose_observed=result["nose_observed"],
            hole_id=result["hole_id"],
            hole_visit_count=len(tracker.scorer.events),
            primary_errors=s["primary_errors"],
            primary_latency_s=s["primary_latency_s"],
            total_distance_cm=s["total_distance_cm"],
            nose_observed_percent=s["nose_observed_percent"],
        )

    def _analyze_single_video(self, video_path, video_info, batch_index=None):
        index = 0 if batch_index is None else int(batch_index)
        cfg = self._config_for_video(index)
        backend = cfg.get("tracking_backend", "classical")
        if (
            backend == "deeplabcut"
            and cfg.get("tracking_backend_effective") != "deeplabcut"
        ):
            raise RuntimeError(
                "DLC已开启，但模型或worker不可用。请修复DLC配置或明确关闭开关后再运行。"
            )
        output = _prepare_output_paths(
            self.session_id,
            video_info.get("original_name", "video"),
            suffix="barnes",
            result_folder=cfg.get("output_dir") or str(RESULT_DIR),
            fallback="barnes",
            batch_index=index,
            batch_count=len(self.video_paths),
        )
        work = output["work_base"]
        final = output["final_base"]
        name = output["display_stem"]
        cap = open_video_capture(
            video_path, orientation_override=cfg.get("orientation_override")
        )
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open Barnes video: {video_path}")
        writer = None
        coordinator = None
        try:
            meta = probe_video_info(
                video_path, fps_override=cfg.get("fps_override"), cap=cap
            )
            fps = float(meta["fps"])
            width = int(meta["width"])
            height = int(meta["height"])
            count = int(meta["total_frames"])
            clock, clock_summary, count = self._build_timebase_for_video(
                video_path,
                fps,
                count,
                label="Barnes",
                prefer_pts=not bool(cfg.get("fps_override")),
            )
            sampling = self._resolve_video_sampling(
                width,
                height,
                fps,
                dict(
                    cfg,
                    fast_scoring=False,
                    spatial_downsample=False,
                    temporal_downsample=False,
                ),
                label="Barnes",
                temporal_default=False,
            )
            self._log_video_sampling(sampling)
            start_frame, end_frame = self._time_range_to_frames(
                clock, count, cfg.get("start_sec"), cfg.get("end_sec")
            )
            start_s = float(cfg.get("start_sec") or 0.0)
            video_end_s = clock.time_at(count - 1) + clock.median_dt
            roi_configs = self._parse_rois(cfg)
            for r in roi_configs:
                points = r["arena_points"] + [h["point"] for h in r["holes"]]
                if any(not (0 <= p[0] < width and 0 <= p[1] < height) for p in points):
                    raise ValueError(
                        "圆盘边界和全部洞位必须位于当前画面内；请检查标定、方向及所用模板。"
                    )
            trackers = []
            for r in roi_configs:
                timing = r["time_config"]
                rs = start_s
                re = min(float(cfg.get("end_sec") or video_end_s), video_end_s)
                if timing["mode"] == "precise":
                    if timing.get("start") is not None:
                        rs = max(rs, float(timing["start"]))
                    if timing.get("end") is not None:
                        re = (
                            min(float(re), float(timing["end"]))
                            if re is not None
                            else float(timing["end"])
                        )
                arena = BarnesArena(
                    r["arena_points"],
                    r["holes"],
                    r["target_hole_id"],
                    r["arena_diameter_cm"],
                    r["hole_radius_cm"],
                )
                trackers.append(
                    BarnesTracker(
                        arena,
                        detection_points=r.get("detection_points"),
                        fps=fps,
                        classical_config=cfg,
                        roi_id=r["roi_id"],
                        roi_name=r["name"],
                        start_s=rs,
                        end_s=re,
                        trial_type=cfg.get("trial_type", "acquisition"),
                        tracking_backend=backend,
                        visit_confirm_s=cfg.get("visit_confirm_s", 0.1),
                        visit_reset_s=cfg.get("visit_reset_s", 0.1),
                        head_confidence_threshold=(
                            cfg.get("pose_confidence_threshold", 0.8)
                            if backend == "deeplabcut"
                            else cfg.get("head_confidence_threshold", 0.6)
                        ),
                        tracking_max_jump_cm=cfg.get("tracking_max_jump_cm", 12.0),
                        manual_escape_time_s=r.get("manual_escape_time_s"),
                        manual_events=r.get("manual_hole_events"),
                        escape_absence_s=cfg.get("escape_absence_s", 0.5),
                        escape_zone_radius_cm=cfg.get("escape_zone_radius_cm", 6.0),
                    )
                )
            bg = morph = detector = None
            if backend == "classical":
                bg = enable_frame_cache(
                    create_classical_background(
                        video_path, cfg, start_frame=start_frame, end_frame=end_frame
                    )
                )

            coordinator = (
                PoseInferenceCoordinator(cfg, trackers)
                if backend == "deeplabcut"
                else None
            )
            if backend == "deeplabcut" and not coordinator.enabled:
                raise RuntimeError("DLC开关已开启，但正式推理后端未启用。")
            if cfg.get("generate_video", True):
                writer = create_annotated_video_writer(
                    f"{work}_analyzed.mp4",
                    fps,
                    width,
                    height,
                    label="Barnes annotated video",
                    source_video_path=video_path,
                    logger=self._log,
                )
                if writer is None:
                    raise RuntimeError("Cannot create the Barnes annotated video.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = 0
            covered = 0
            total = max(1, end_frame - start_frame + 1)
            for batch in self._iter_prefetched_frame_batches(
                cap,
                start_frame=start_frame,
                end_frame=end_frame,
                sampling=sampling,
                batch_size=coordinator.batch_size if coordinator is not None else 1,
                prepare_gray=backend == "classical",
            ):
                if not self.is_running:
                    break
                poses = (
                    coordinator.infer_many([x["frame"] for x in batch])
                    if coordinator is not None
                    else [None] * len(batch)
                )
                for item, pose in zip(batch, poses):
                    if pose is not None and pose.error:
                        raise RuntimeError(f"DLC推理失败，未回退经典视觉：{pose.error}")
                    source_frame = int(item["source_frame_no"])
                    frame = item["frame"]
                    t = clock.time_at(source_frame)
                    dt = (
                        clock.time_at(source_frame + 1) - t
                        if source_frame + 1 < count
                        else clock.median_dt
                    )
                    if bg is not None:
                        begin_frame_cache(bg, source_frame)
                    results = []
                    for tracker in trackers:
                        rec = tracker.update(
                            frame,
                            t,
                            item["gray"],
                            bg,
                            morph,
                            detector,
                            dt_s=dt,
                            pose_observation=(
                                pose.by_roi_id.get(tracker.roi_id)
                                if pose is not None
                                else None
                            ),
                        )
                        rec.update(
                            frame=source_frame,
                            analysis_sample=frames,
                            video_name=video_info.get("original_name", name),
                            analysis_frame_stride=1,
                            analysis_spatial_scale=1.0,
                        )
                        results.append(rec)
                    annotated = draw_barnes_results(frame, trackers, results)
                    if writer is not None:
                        writer.write(annotated)
                    if frames % 15 == 0:
                        self.realtime_data = [
                            self._live_row(tracker, r)
                            for tracker, r in zip(trackers, results)
                        ]
                        self.live_metrics = {
                            "roi_count": len(trackers),
                            "hole_visit_count": sum(
                                x["hole_visit_count"] for x in self.realtime_data
                            ),
                        }
                        self.current_frame = annotated
                    frames += 1
                    covered += int(item["covered_source_frames"])
                    self._advance_frame_counters(int(item["covered_source_frames"]))
                    if frames % 20 == 0:
                        self._set_video_progress(
                            batch_index,
                            covered / total * 0.94,
                            f"Barnes frame {covered}/{total} ({{progress}}%)",
                        )
            if not self.is_running:
                return
            self._ensure_analysis_frames_processed(
                video_path=video_path,
                label="Barnes",
                frame_idx=frames,
                source_frames_done=covered,
            )
            summaries = []
            tracks = []
            events = []
            holes = []
            bins = []
            for tracker, r in zip(trackers, roi_configs):
                summary = tracker.scorer.finish()
                summary.update(
                    video_name=video_info.get("original_name", name),
                    tracking_backend_requested=backend,
                    tracking_backend_effective=backend,
                    pose_model_reference=(
                        cfg.get("pose_model_reference")
                        if backend == "deeplabcut"
                        else None
                    ),
                    pose_model_profile_id=(
                        cfg.get("pose_model_profile_id")
                        if backend == "deeplabcut"
                        else None
                    ),
                    pose_model_version_id=(
                        cfg.get("pose_model_version_id")
                        if backend == "deeplabcut"
                        else None
                    ),
                    pose_batch_size_effective=(
                        coordinator.effective_batch_size
                        if backend == "deeplabcut"
                        else None
                    ),
                    arena_diameter_cm=tracker.arena.diameter_cm,
                    hole_radius_cm=tracker.arena.hole_radius_cm,
                    day_label=cfg.get("day_label"),
                    trial_no=cfg.get("trial_no"),
                    classical_method=(
                        tracker.classical_detector.method
                        if backend == "classical"
                        else None
                    ),
                    classical_method_version=(
                        tracker.classical_detector.version
                        if backend == "classical"
                        else None
                    ),
                    landmark_evidence=(
                        "pre_body_erosion"
                        if backend == "classical"
                        and cfg.get("classical_method") == "local_otsu_v1"
                        else None
                    ),
                )
                for key in ["subject_id", "group_id", "group_name", "animal_code"]:
                    summary[key] = r.get(key)
                self._add_sampling_summary_fields(summary, sampling)
                self._add_timebase_summary_fields(summary, clock_summary)
                summaries.append(summary)
                tracks.extend(tracker.frame_records)
                for rows, out in [
                    (tracker.scorer.events, events),
                    (tracker.scorer.hole_summary(), holes),
                    (tracker.build_time_bins(cfg.get("bin_seconds", 60.0)), bins),
                ]:
                    out.extend(
                        [
                            dict(x, video_name=video_info.get("original_name", name))
                            for x in rows
                        ]
                    )
                slug = ascii_slug(
                    f"{tracker.roi_id:02d}_{tracker.roi_name}", fallback="barnes"
                )
                for heat, kind in [(False, "trackmap"), (True, "heatmap")]:
                    wp = f"{work}_{slug}_{kind}.png"
                    fp = f"{final}_{slug}_{kind}.png"
                    if not render_barnes_map(
                        tracker, wp, heatmap=heat, style=cfg.get("visual_style")
                    ):
                        raise RuntimeError(f"Cannot write Barnes {kind}.")
                    self._append_result_file(_export_result_file(wp, fp), kind)
            for suffix, rows, group in [
                ("track", tracks, "track_csv"),
                ("summary", summaries, "summary_csv"),
                ("events", events, "events_csv"),
                ("holes", holes, "summary_csv"),
                ("bins", bins, "bins_csv"),
            ]:
                wp = f"{work}_{suffix}.csv"
                fp = f"{final}_{suffix}.csv"
                fields = list(dict.fromkeys(k for row in rows for k in row)) or [
                    "roi_id",
                    "roi_name",
                    "time_s",
                    "hole_id",
                    "source",
                ]
                with open(wp, "w", encoding="utf-8-sig", newline="") as output_file:
                    writer_csv = csv.DictWriter(output_file, fields)
                    writer_csv.writeheader()
                    writer_csv.writerows(rows)
                self._append_result_file(_export_result_file(wp, fp), group)
            report = f"{work}_report.txt"
            with open(report, "w", encoding="utf-8-sig") as output_file:
                output_file.write(
                    "巴恩斯迷宫 / Barnes maze\n\n到达目标洞与确认入箱分开计时。自动事件采用鼻尖进入洞ROI的操作定义。\n"
                )
                output_file.write(
                    "检测丢失只生成待复核入箱候选；未确认的逃生潜伏期留空。搜索模式为探索性规则候选。\n\n"
                )
                for s in summaries:
                    output_file.write(
                        json.dumps(s, ensure_ascii=False, indent=2) + "\n\n"
                    )
            self._append_result_file(
                _export_result_file(report, f"{final}_report.txt"), "report"
            )
        finally:
            cap.release()
            if coordinator is not None:
                coordinator.close()
            if writer is not None:
                writer.release()
        if cfg.get("generate_video", True):
            self._append_result_file(
                _export_result_file(f"{work}_analyzed.mp4", f"{final}_analyzed.mp4"),
                "video",
            )
        self._set_video_progress(batch_index, 1.0, "Barnes analysis completed.")
