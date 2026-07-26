from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from core.y_maze import YMazeTracker


ZONES = {
    "center": [(90, 90), (110, 90), (100, 110)],
    "arm_a": [(90, 90), (110, 90), (110, 10), (90, 10)],
    "arm_b": [(110, 90), (200, 90), (200, 110), (110, 110)],
    "arm_c": [(0, 90), (90, 90), (90, 110), (0, 110)],
}


def test_summary_exports_the_complete_chronological_entry_sequence() -> None:
    tracker = YMazeTracker(ZONES, fps=10.0)
    arm_order = [
        "arm_a",
        "arm_b",
        "arm_c",
        "arm_a",
        "arm_c",
        "arm_b",
        "arm_a",
        "arm_b",
        "arm_c",
        "arm_b",
        "arm_a",
        "arm_c",
        "arm_a",
    ]
    tracker.arm_entry_sequence = arm_order
    tracker.total_arm_entries = len(arm_order)
    tracker.arm_entries = {
        name: arm_order.count(name)
        for name in tracker.ARM_NAMES
    }

    expected = "-".join(tracker.ARM_LABELS[name] for name in arm_order)
    assert len(arm_order) > 12
    assert tracker.build_summary()["entry_sequence"] == expected
