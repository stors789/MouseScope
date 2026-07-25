from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from core.y_maze import YMazeTracker
from services.analysis_builders import build_y_maze_config


ZONES = {
    "center": [(90, 90), (110, 90), (100, 110)],
    "arm_a": [(90, 90), (110, 90), (110, 10), (90, 10)],
    "arm_b": [(110, 90), (200, 90), (200, 110), (110, 110)],
    "arm_c": [(0, 90), (90, 90), (90, 110), (0, 110)],
}
POINTS = {
    "center": (100.0, 98.0),
    "arm_a": (100.0, 20.0),
    "arm_b": (180.0, 100.0),
    "arm_c": (20.0, 100.0),
}


def make_tracker(**kwargs: Any) -> YMazeTracker:
    return YMazeTracker(
        ZONES,
        fps=10.0,
        zone_confirm_s=0.0,
        center_reset_s=0.0,
        entry_depth_ratio=0.2,
        **kwargs,
    )


def detection(point: tuple[float, float], area: float = 100.0, solidity: float = 0.85) -> dict[str, Any]:
    x, y = point
    return {
        "centroid": (x, y),
        "area": area,
        "solidity": solidity,
        "bbox": (int(x - 2), int(y - 2), 4, 4),
        "contour": None,
    }


def feed(tracker: YMazeTracker, sequence: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    pending = iter(sequence)

    def fake_detect(*_args: Any, **_kwargs: Any):
        return next(pending), np.zeros((200, 200), dtype=np.uint8)

    tracker.detect_mouse = fake_detect  # type: ignore[method-assign]
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    gray = np.zeros((200, 200), dtype=np.uint8)
    return [
        tracker.update(frame, idx / 10.0, gray, None, None, None, dt_s=0.1)
        for idx in range(len(sequence))
    ]


def test_candidate_scoring_rejects_larger_distant_interference() -> None:
    tracker = make_tracker(tracking_max_jump_cm=8.0)
    tracker.last_all_centroid_px = POINTS["arm_a"]
    tracker.last_detection_time_abs = 0.0
    tracker.expected_area_px = 100.0

    mouse_fragment = detection((102.0, 22.0), area=65.0, solidity=0.72)
    wall_shadow = detection((180.0, 100.0), area=1800.0, solidity=0.96)

    assert tracker._select_detection([wall_shadow, mouse_fragment], elapsed_s=0.1) is mouse_fragment


def test_max_jump_rejects_only_implausible_candidate() -> None:
    tracker = make_tracker(tracking_max_jump_cm=4.0)
    tracker.last_all_centroid_px = POINTS["arm_a"]
    tracker.last_detection_time_abs = 0.0
    tracker.expected_area_px = 100.0

    assert tracker._select_detection([detection(POINTS["arm_b"])], elapsed_s=0.1) is None


def test_short_occlusion_is_predicted_but_long_loss_becomes_unknown() -> None:
    tracker = make_tracker(occlusion_hold_s=0.2)
    results = feed(
        tracker,
        [detection((100.0, 24.0)), detection((100.0, 20.0)), None, None, None],
    )

    assert results[2]["tracking_state"] == "predicted"
    assert results[2]["centroid"] is not None
    assert results[3]["tracking_state"] == "predicted"
    assert results[4]["tracking_state"] == "unknown"
    assert results[4]["centroid"] is None
    assert results[4]["zone_key"] == "Unknown"


def test_direct_arm_to_arm_jump_is_blocked_until_center_passage() -> None:
    tracker = make_tracker(tracking_max_jump_cm=200.0)
    results = feed(
        tracker,
        [
            detection(POINTS["arm_a"]),
            detection(POINTS["arm_b"]),
            detection(POINTS["center"]),
            detection(POINTS["arm_b"]),
        ],
    )

    assert results[0]["arm_entry_event"] is True
    assert results[1]["arm_entry_event"] is False
    assert results[1]["zone_key"] == "arm_a"
    assert results[3]["arm_entry_event"] is True
    assert tracker.arm_entry_sequence == ["arm_a", "arm_b"]


def test_normal_three_arm_sequence_counts_one_alternation() -> None:
    tracker = make_tracker(tracking_max_jump_cm=200.0)
    feed(
        tracker,
        [
            detection(POINTS["arm_a"]),
            detection(POINTS["center"]),
            detection(POINTS["arm_b"]),
            detection(POINTS["center"]),
            detection(POINTS["arm_c"]),
        ],
    )

    assert tracker.arm_entry_sequence == ["arm_a", "arm_b", "arm_c"]
    assert tracker.total_arm_entries == 3
    assert tracker.alternation_count == 1
    assert tracker.build_summary()["alternation_percent"] == 100.0


def test_builder_adds_safe_defaults_and_preserves_per_video_overrides() -> None:
    session = {"video_infos": [{}, {}]}
    config = build_y_maze_config(
        session,
        {
            "rois": [],
            "per_video_configs": [{}, {"tracking_max_jump_cm": 7, "occlusion_hold_s": 0.8}],
        },
    )

    assert config["tracking_max_jump_cm"] == 12.0
    assert config["occlusion_hold_s"] == 0.4
    assert config["zone_confirm_s"] == 0.12
    assert config["center_reset_s"] == 0.2
    assert config["per_video_configs"][0]["tracking_max_jump_cm"] == 12.0
    assert config["per_video_configs"][1]["tracking_max_jump_cm"] == 7.0
    assert config["per_video_configs"][1]["occlusion_hold_s"] == 0.8
