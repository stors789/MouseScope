import pathlib
import sys

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from core.novel_object import NovelObjectTracker


FRAME = np.zeros((120, 120, 3), dtype=np.uint8)
GRAY = np.zeros((120, 120), dtype=np.uint8)


def make_tracker(**kwargs) -> NovelObjectTracker:
    return NovelObjectTracker(
        arena_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        object_points=[
            {"point": [90, 50], "name": "Novel", "role": "novel"},
            {"point": [10, 10], "name": "Familiar", "role": "familiar"},
        ],
        arena_side_cm=50,
        fps=25,
        interaction_radius_cm=4,
        head_angle_threshold_deg=50,
        head_confidence_threshold=0.8,
        **kwargs,
    )


def pose(
    body=(70, 50),
    nose=(88, 50),
    *,
    body_confidence=0.99,
    nose_confidence=0.99,
):
    return {
        "source": "deeplabcut",
        "confidence_threshold": 0.8,
        "keypoints": {
            "body_center": {
                "x": body[0],
                "y": body[1],
                "confidence": body_confidence,
                "visible": True,
            },
            "nose": {
                "x": nose[0],
                "y": nose[1],
                "confidence": nose_confidence,
                "visible": True,
            },
        },
    }


def update(
    tracker: NovelObjectTracker,
    observation,
    time_s: float,
    *,
    dt_s: float = 0.04,
    strict_pose_only: bool = True,
    allow_classical_fallback: bool = False,
):
    return tracker.update(
        FRAME,
        time_s,
        GRAY,
        None,
        None,
        None,
        dt_s=dt_s,
        pose_observation=observation,
        allow_classical_fallback=allow_classical_fallback,
        strict_pose_only=strict_pose_only,
    )


def test_short_pose_gap_holds_head_and_counts_exploration() -> None:
    tracker = make_tracker(pose_head_hold_s=0.12, pose_head_smoothing=1.0)
    update(tracker, pose(), 0.0)

    held = update(
        tracker,
        pose(body=(70, 50), nose_confidence=0.1),
        0.04,
    )

    assert held["head_direction_method"] == "deeplabcut_held"
    assert held["head_direction_confident"] is True
    assert held["exploration_mode"] == "head_held"
    assert held["current_object_idx"] == 0
    assert held["head_point_cm"] == pytest.approx((44.0, 25.0))
    assert tracker.object_exploration_time_s[0] == pytest.approx(0.08)


def test_hybrid_mode_keeps_classical_centroid_and_uses_dlc_head() -> None:
    tracker = make_tracker(pose_head_hold_s=0.0, pose_head_smoothing=1.0)
    classical_detection = {
        "centroid": (20.0, 30.0),
        "bbox": (10, 20, 20, 20),
        "contour": np.array(
            [[[10, 20]], [[30, 20]], [[30, 40]], [[10, 40]]],
            dtype=np.int32,
        ),
    }
    result = tracker.update(
        FRAME,
        0.0,
        GRAY,
        None,
        None,
        None,
        dt_s=0.04,
        pose_observation=pose(),
        allow_classical_fallback=False,
        precomputed_detection=(
            classical_detection,
            np.zeros((100, 100), dtype=np.uint8),
        ),
        prefer_classical_position=True,
        require_pose_for_exploration=True,
    )

    assert result["centroid"] == pytest.approx((20.0, 30.0))
    assert result["body_centroid"] == pytest.approx((70.0, 50.0))
    assert result["tracking_source"] == "hybrid_dlc"
    assert result["head_direction_method"].startswith("deeplabcut")


def test_pose_gap_expires_using_accumulated_dt() -> None:
    tracker = make_tracker(pose_head_hold_s=0.12, pose_head_smoothing=1.0)
    update(tracker, pose(), 0.0)

    for frame_index in range(1, 4):
        result = update(
            tracker,
            pose(body=(72, 50), nose_confidence=0.1),
            frame_index * 0.04,
        )
        assert result["head_direction_method"] == "deeplabcut_held"

    expired = update(
        tracker,
        pose(body=(72, 50), nose_confidence=0.1),
        0.16,
    )

    assert expired["head_direction_method"] is None
    assert expired["head_point"] is None
    assert expired["current_object_idx"] is None


def test_reliable_observation_resets_hold_timer() -> None:
    tracker = make_tracker(pose_head_hold_s=0.08, pose_head_smoothing=1.0)
    update(tracker, pose(), 0.0)
    update(tracker, pose(nose_confidence=0.1), 0.04)
    update(tracker, pose(), 0.08)

    held = update(tracker, pose(nose_confidence=0.1), 0.12)

    assert held["head_direction_method"] == "deeplabcut_held"
    assert tracker.pose_head_missing_time_s == pytest.approx(0.04)


def test_reliable_pose_direction_is_exponentially_smoothed() -> None:
    tracker = make_tracker(pose_head_hold_s=0.12, pose_head_smoothing=0.25)
    update(tracker, pose(body=(50, 50), nose=(70, 50)), 0.0)

    smoothed = update(
        tracker,
        pose(body=(50, 50), nose=(50, 30)),
        0.04,
    )

    expected_angle = np.deg2rad(-22.5)
    expected_unit = np.array(
        [np.cos(expected_angle), np.sin(expected_angle)],
        dtype=np.float32,
    )
    expected_nose_cm = np.array([25.0, 25.0]) + expected_unit * 10.0
    assert smoothed["head_point_cm"] == pytest.approx(expected_nose_cm, abs=1e-5)
    assert tracker.pose_head_unit_vector == pytest.approx(expected_unit, abs=1e-6)


def test_repeated_opposite_observations_do_not_lock_to_old_direction() -> None:
    tracker = make_tracker(pose_head_hold_s=0.12, pose_head_smoothing=0.35)
    update(tracker, pose(body=(50, 50), nose=(70, 50)), 0.0)

    first = update(tracker, pose(body=(50, 50), nose=(30, 50)), 0.04)
    second = update(tracker, pose(body=(50, 50), nose=(30, 50)), 0.08)

    assert first["head_point_cm"][0] > 25.0
    assert second["head_point_cm"][0] < 25.0


def test_head_stabilization_parameters_are_clamped_and_summarized() -> None:
    tracker = make_tracker(pose_head_hold_s=2.0, pose_head_smoothing=-1.0)

    summary = tracker.build_summary()

    assert tracker.pose_head_hold_s == 1.0
    assert tracker.pose_head_smoothing == 0.0
    assert summary["pose_head_hold_s"] == 1.0
    assert summary["pose_head_smoothing"] == 0.0


def test_strict_pose_mode_does_not_use_classical_or_body_fallback(monkeypatch) -> None:
    tracker = make_tracker()
    monkeypatch.setattr(
        tracker,
        "_estimate_head_direction",
        lambda *_args, **_kwargs: pytest.fail("classical head estimator was called"),
    )
    result = update(
        tracker,
        pose(body=(88, 50), nose_confidence=0.1),
        0.0,
        strict_pose_only=True,
        allow_classical_fallback=True,
    )

    assert result["near_flags"][0] is True
    assert result["head_direction_method"] is None
    assert result["current_object_idx"] is None
    assert result["exploration_mode"] is None


def test_non_strict_mode_retains_body_proximity_fallback() -> None:
    tracker = make_tracker()
    result = update(
        tracker,
        pose(body=(88, 50), nose_confidence=0.1),
        0.0,
        strict_pose_only=False,
    )

    assert result["current_object_idx"] == 0
    assert result["exploration_mode"] == "fallback"
