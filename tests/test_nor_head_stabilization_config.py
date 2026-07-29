from pathlib import Path

from services.analysis_builders import build_novel_object_config
from services.experiment_defaults import (
    BUILTIN_EXPERIMENT_DEFAULTS,
    experiment_defaults_schema,
)
from services.reproducibility import build_run_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_nor_head_stabilization_defaults_and_schema() -> None:
    defaults = BUILTIN_EXPERIMENT_DEFAULTS["novel_object"]
    assert defaults["pose_head_hold_s"] == 0.12
    assert defaults["pose_head_smoothing"] == 0.35

    fields = {
        field["key"]: field
        for field in experiment_defaults_schema()["modules"]["novel_object"]
    }
    assert fields["pose_head_hold_s"]["group"] == "analysis"
    assert fields["pose_head_hold_s"]["min"] == 0.0
    assert fields["pose_head_hold_s"]["max"] == 1.0
    assert fields["pose_head_hold_s"]["step"] == 0.01
    assert fields["pose_head_smoothing"]["group"] == "analysis"
    assert fields["pose_head_smoothing"]["min"] == 0.0
    assert fields["pose_head_smoothing"]["max"] == 1.0
    assert fields["pose_head_smoothing"]["step"] == 0.05


def test_nor_head_stabilization_reaches_global_and_per_video_configs() -> None:
    sess = {"video_infos": [{}], "start_seconds": None, "end_seconds": None}
    config = build_novel_object_config(
        sess,
        {
            "pose_head_hold_s": 0.18,
            "pose_head_smoothing": 0.4,
            "per_video_configs": [
                {
                    "pose_head_hold_s": 0.25,
                    "pose_head_smoothing": 0.6,
                }
            ],
        },
    )

    assert config["pose_head_hold_s"] == 0.18
    assert config["pose_head_smoothing"] == 0.4
    assert config["per_video_configs"][0]["pose_head_hold_s"] == 0.25
    assert config["per_video_configs"][0]["pose_head_smoothing"] == 0.6


def test_nor_head_stabilization_is_preserved_in_run_snapshot() -> None:
    resolved_config = {
        "pose_head_hold_s": 0.12,
        "pose_head_smoothing": 0.35,
    }
    draft_payload = {
        "pose_head_hold_s": 0.12,
        "pose_head_smoothing": 0.35,
    }
    snapshot = build_run_snapshot(
        session_id="nor-head-config",
        app_version="test",
        sess={
            "experiment_type": "novel_object",
            "video_paths": [],
            "draft_payload": draft_payload,
        },
        resolved_config=resolved_config,
    )

    assert snapshot["draft_payload"]["pose_head_hold_s"] == 0.12
    assert snapshot["draft_payload"]["pose_head_smoothing"] == 0.35
    assert snapshot["resolved_config"]["pose_head_hold_s"] == 0.12
    assert snapshot["resolved_config"]["pose_head_smoothing"] == 0.35


def test_nor_template_wires_head_stabilization_controls() -> None:
    source = (ROOT / "templates" / "calibrate_nor.html").read_text(encoding="utf-8")
    for token in (
        'id="poseHeadHold"',
        'id="poseHeadSmoothing"',
        "pose_head_hold_s: currentTuningParams.pose_head_hold_s",
        "pose_head_smoothing: currentTuningParams.pose_head_smoothing",
    ):
        assert token in source
