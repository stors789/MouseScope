"""Video/export tests use synthetic observations with known event times."""

import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "core")]
import app as application
from services.barnes_config import build_barnes_config
from services.barnes_session import BarnesMultiArenaAnalysisSession
from services.pose_tracking import resolve_pose_runtime_config
from services.video_metadata import probe_video_info


def payload():
    points = [
        [160 + 135 * math.cos(t), 160 + 120 * math.sin(t)]
        for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)
    ]
    return dict(
        hole_count=4,
        arena_diameter_cm=90,
        bg_method="threshold",
        generate_video=False,
        rois=[
            dict(
                name="Mouse A",
                arena_points=points,
                holes=[
                    dict(id=str(i + 1), point=p)
                    for i, p in enumerate(
                        [(265, 160), (160, 255), (55, 160), (160, 65)]
                    )
                ],
                target_hole_id="1",
                manual_escape_time_s=2.8,
                manual_hole_events=[
                    dict(time_s=0.4, hole_id="2"),
                    dict(time_s=1.0, hole_id="2"),
                    dict(time_s=2.0, hole_id="1"),
                ],
            )
        ],
    )


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "mouse.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 320))
    assert writer.isOpened()
    for i in range(40):
        frame = np.full((320, 320, 3), 235, np.uint8)
        cv2.ellipse(frame, (140 + i, 160), (18, 10), 0, 0, 360, (15, 15, 15), -1)
        cv2.line(frame, (122 + i, 160), (103 + i, 165), (15, 15, 15), 3)
        writer.write(frame)
    writer.release()
    info = dict(original_name=path.name, **probe_video_info(path))
    return path, info


def read_csv(files, suffix):
    path = next(Path(p) for p in files if str(p).endswith(suffix))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def run_clip(clip, tmp_path, data):
    path, info = clip
    config = build_barnes_config(
        dict(video_infos=[info], output_dir=str(tmp_path)), data
    )
    session = BarnesMultiArenaAnalysisSession(
        "barnes_test_video", [str(path)], [info], config
    )
    session.is_running = True
    session._analyze_single_video(str(path), info)
    return session


def test_full_video_exports_preserve_manual_repeats_and_separate_latencies(
    clip, tmp_path
):
    session = run_clip(clip, tmp_path, payload())
    rows = read_csv(session.result_files, "_summary.csv")
    assert len(rows) == 1
    summary = rows[0]
    assert summary["hole_visit_sequence"] == "2-2-1"
    assert float(summary["primary_latency_s"]) == 2
    assert float(summary["total_latency_s"]) == 2.8
    assert float(summary["primary_errors"]) == 2
    assert summary["event_source"] == "manual"
    assert float(summary["body_observed_percent"]) > 90
    assert len(read_csv(session.result_files, "_holes.csv")) == 4
    assert len(read_csv(session.result_files, "_track.csv")) == 40
    assert len(read_csv(session.result_files, "_events.csv")) == 3
    assert sum(str(p).endswith(".png") for p in session.result_files) == 2


def test_no_manual_escape_keeps_total_latency_empty(clip, tmp_path):
    data = payload()
    data["rois"][0].pop("manual_escape_time_s")
    session = run_clip(clip, tmp_path, data)
    summary = read_csv(session.result_files, "_summary.csv")[0]
    assert summary["total_latency_s"] == ""
    assert summary["total_latency_censored"] == "True"


def test_holes_all_required_and_invalid_backend_rejected():
    data = payload()
    data["hole_count"] = 20
    with pytest.raises(ValueError, match="全部洞位"):
        build_barnes_config({}, data)
    data = payload()
    data["tracking_backend"] = "auto"
    with pytest.raises(ValueError):
        build_barnes_config({}, data)


def test_checked_dlc_switch_never_silently_falls_back():
    config = build_barnes_config({}, dict(payload(), tracking_backend="deeplabcut"))
    with pytest.raises(ValueError, match="DLC已开启"):
        resolve_pose_runtime_config(
            config,
            "barnes_maze",
            runtime_status=dict(available=False, code="missing_python"),
        )


def test_unchecked_dlc_does_not_require_a_worker():
    config = build_barnes_config({}, payload())
    resolve_pose_runtime_config(
        config,
        "barnes_maze",
        runtime_status=dict(available=False, code="missing_python"),
    )
    assert config["tracking_backend_effective"] == "classical"


class DeterministicPoseRuntime:
    """A contract fixture, not evidence of a real model's accuracy."""

    def __init__(self):
        self.index = 0

    def infer_frames(self, frames, **kwargs):
        outputs = []
        for frame in frames:
            nose = (
                (160, 255)
                if self.index < 10
                else (160, 160) if self.index < 20 else (265, 160)
            )
            self.index += 1
            outputs.append(
                dict(
                    keypoints=dict(
                        body_center=dict(x=160, y=160, confidence=1),
                        nose=dict(x=nose[0], y=nose[1], confidence=1),
                    )
                )
            )
        return outputs


def test_dlc_batches_produce_the_same_automatic_sequence(clip, tmp_path, monkeypatch):
    summaries = []
    for batch_size in [1, 4]:
        runtime = DeterministicPoseRuntime()
        monkeypatch.setattr(
            "services.pose_tracking.shared_deeplabcut_runtime", lambda: runtime
        )
        data = payload()
        data.update(
            tracking_backend="deeplabcut",
            pose_batch_size=batch_size,
            visit_confirm_s=0.15,
        )
        data["rois"][0]["manual_hole_events"] = None
        output = tmp_path / str(batch_size)
        output.mkdir()
        session = run_clip(clip, output, data)
        summary = read_csv(session.result_files, "_summary.csv")[0]
        assert summary["tracking_backend_effective"] == "deeplabcut"
        assert summary["hole_visit_sequence"] == "2-1"
        assert float(summary["primary_latency_s"]) == 2
        summaries.append(summary["hole_visit_sequence"])
    assert summaries[0] == summaries[1]


def test_dlc_inference_failure_stops_without_classical_fallback(
    clip, tmp_path, monkeypatch
):
    from services.deeplabcut_runtime import DeepLabCutRuntimeError

    class BrokenRuntime:
        def infer_frames(self, *args, **kwargs):
            raise DeepLabCutRuntimeError("test worker failure")

    monkeypatch.setattr(
        "services.pose_tracking.shared_deeplabcut_runtime", lambda: BrokenRuntime()
    )
    with pytest.raises(RuntimeError, match="DLC推理失败"):
        run_clip(clip, tmp_path, dict(payload(), tracking_backend="deeplabcut"))
    assert not list(tmp_path.glob("*_summary.csv"))
