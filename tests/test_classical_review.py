import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "core")]
from classical_mouse import CLASSICAL_METHOD


def mouse_scene():
    gray = np.full((160, 200), 210, np.uint8)
    cv2.ellipse(gray, (95, 82), (25, 17), 0, 0, 360, 20, -1)
    cv2.line(gray, (118, 82), (163, 82), 20, 4)
    return gray, None, None


def test_old_snapshot_keeps_its_legacy_method(tmp_path):
    import json
    from types import SimpleNamespace
    from tools.validate_landmarks import resolve_prediction_config

    p = tmp_path / "run_snapshot.json"
    p.write_text(
        json.dumps(
            dict(
                resolved_config=dict(
                    invert=True,
                    gray_threshold=64,
                    tracking_backend="auto",
                    per_video_configs=[dict(min_area=90)],
                )
            )
        )
    )
    config, q = resolve_prediction_config(
        SimpleNamespace(
            run_snapshot=str(p), video_index=0, backend="classical", confidence=None
        )
    )
    assert config["classical_method"] == "legacy"
    assert config["invert"] is True
    assert config["gray_threshold"] == 64 and config["min_area"] == 90


def test_old_barnes_replay_marks_legacy_without_mutating_snapshot(tmp_path):
    from services.reproducibility import resolved_config_for_replay

    snapshot = dict(
        experiment_type="barnes_maze",
        resolved_config=dict(
            tracking_backend="classical", per_video_configs=[dict(min_area=120)]
        ),
    )
    replay = resolved_config_for_replay(snapshot, output_dir=tmp_path)
    assert replay["classical_method"] == "legacy"
    assert replay["per_video_configs"][0]["classical_method"] == "legacy"
    assert "classical_method" not in snapshot["resolved_config"]


def test_review_exports_project_method_from_a_real_clip(tmp_path):
    import json
    from types import SimpleNamespace
    from tools.review_mouse_tracking import review
    from services.video_metadata import probe_video_info

    gray, _, _ = mouse_scene()
    source = tmp_path / "mouse.avi"
    video = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (200, 160)
    )
    assert video.isOpened()
    for _ in range(6):
        video.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    video.release()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                video_path=str(source),
                records=[
                    dict(crop=dict(name="mouse", x=0, y=0, width=200, height=160))
                ],
            )
        )
    )
    snapshot = tmp_path / "config.json"
    snapshot.write_text(
        json.dumps(
            dict(classical_method=CLASSICAL_METHOD, bg_method="threshold", min_area=60)
        )
    )
    out = tmp_path / "review"
    args = SimpleNamespace(
        manifest=str(manifest),
        output=str(out),
        segment=["0:.6"],
        run_snapshot=str(snapshot),
        video_index=0,
        confidence=0.4,
        backend="classical",
        compare_predictions=None,
        prediction_metadata=None,
        viewports=None,
    )
    review(args)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["method_version"] == CLASSICAL_METHOD
    assert summary["landmark_evidence"] == "pre_body_erosion"
    assert summary["video"]["total_frames"] == 6
    assert (out / "points.csv").is_file()
