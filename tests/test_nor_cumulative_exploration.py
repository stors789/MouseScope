import pathlib
import sys
import tempfile
import unittest

import cv2


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from core.novel_object import (
    NovelObjectTracker,
    _exploration_curve_label,
    build_exploration_cumulative_series,
    render_novel_object_exploration_cumulative,
)
from services.novel_object_session import NovelObjectMultiArenaAnalysisSession


def make_tracker(phase="recognition"):
    return NovelObjectTracker(
        arena_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        object_points=[
            {"point": [25, 50], "name": "Cube", "role": "familiar"},
            {"point": [75, 50], "name": "Bottle", "role": "novel"},
        ],
        arena_side_cm=50,
        fps=10,
        phase=phase,
    )


class NorCumulativeExplorationTests(unittest.TestCase):
    def test_series_uses_dt_and_real_timeline(self):
        tracker = make_tracker()
        tracker.frame_records = [
            {
                "time_s": 10.0,
                "window_time_s": 0.0,
                "dt_s": 0.10,
                "in_stats_window": True,
                "object_1_exploring": True,
                "object_2_exploring": False,
            },
            {
                "time_s": 10.4,
                "window_time_s": 0.4,
                "dt_s": 0.35,
                "in_stats_window": True,
                "object_1_exploring": False,
                "object_2_exploring": True,
            },
            {
                "time_s": 10.9,
                "window_time_s": 0.9,
                "dt_s": 0.20,
                "in_stats_window": False,
                "object_1_exploring": True,
                "object_2_exploring": True,
            },
        ]

        times, object_1, object_2 = build_exploration_cumulative_series(tracker)

        self.assertEqual(times, [0.0, 0.0, 0.1, 0.4, 0.75])
        self.assertEqual(object_1, [0.0, 0.0, 0.1, 0.1, 0.1])
        self.assertEqual(object_2[:4], [0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(object_1[-1], 0.10)
        self.assertAlmostEqual(object_2[-1], 0.35)

        # The 0.1 -> 0.4 gap is a true plateau for both curves.
        self.assertEqual(times[2:4], [0.1, 0.4])
        self.assertEqual(object_1[2:4], [0.1, 0.1])
        self.assertEqual(object_2[2:4], [0.0, 0.0])

    def test_non_monotonic_timestamps_are_clamped_without_losing_dt(self):
        tracker = make_tracker()
        tracker.frame_records = [
            {
                "time_s": 1.0,
                "window_time_s": 0.5,
                "dt_s": 0.2,
                "in_stats_window": True,
                "object_1_exploring": True,
                "object_2_exploring": False,
            },
            {
                "time_s": 0.8,
                "window_time_s": 0.3,
                "dt_s": 0.15,
                "in_stats_window": True,
                "object_1_exploring": False,
                "object_2_exploring": True,
            },
        ]

        times, object_1, object_2 = build_exploration_cumulative_series(tracker)

        self.assertEqual(times, [0.0, 0.5, 0.7, 0.7, 0.85])
        self.assertAlmostEqual(object_1[-1], 0.2)
        self.assertAlmostEqual(object_2[-1], 0.15)

    def test_recognition_labels_include_object_and_role(self):
        tracker = make_tracker("recognition")
        self.assertEqual(_exploration_curve_label(tracker.objects[0], tracker.phase), "Cube (Familiar)")
        self.assertEqual(_exploration_curve_label(tracker.objects[1], tracker.phase), "Bottle (Novel)")

        training_tracker = make_tracker("familiarization")
        self.assertEqual(_exploration_curve_label(training_tracker.objects[0], training_tracker.phase), "Cube")
        self.assertEqual(
            _exploration_curve_label({"name": "Unknown", "role": ""}, "recognition"),
            "Unknown",
        )

    def test_renderer_writes_chart_with_no_exploration_and_empty_records(self):
        for rows in (
            [],
            [
                {
                    "time_s": 0.0,
                    "window_time_s": 0.0,
                    "dt_s": 0.25,
                    "in_stats_window": True,
                    "object_1_exploring": False,
                    "object_2_exploring": False,
                }
            ],
        ):
            tracker = make_tracker()
            tracker.frame_records = rows
            with tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "roi_01_exploration_cumulative.png"
                self.assertEqual(
                    render_novel_object_exploration_cumulative(tracker, str(path)),
                    str(path),
                )
                image = cv2.imread(str(path))
                self.assertIsNotNone(image)
                self.assertGreater(image.shape[0], 100)
                self.assertGreater(image.shape[1], 100)

    def test_session_registers_cumulative_download_category(self):
        session = NovelObjectMultiArenaAnalysisSession("session", [], [], {})
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "video_roi_01_exploration_cumulative.png"
            path.write_bytes(b"png")
            session._append_result_file(str(path), "exploration_cumulative")

            download_map = session.get_download_file_map()
            self.assertIn("exploration_cumulative", download_map)
            self.assertEqual(download_map["exploration_cumulative"], [str(path)])


if __name__ == "__main__":
    unittest.main()
