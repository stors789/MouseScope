from __future__ import annotations

import unittest
from unittest.mock import patch

from services.video_timebase import build_video_timebase


class VideoTimebaseTest(unittest.TestCase):
    def test_rejects_sparse_packet_pts_and_uses_frame_pts(self):
        packet_pts = "0.000000\n0.200000\n"
        frame_pts = "0.200000\n0.266667\n0.333334\n0.400001\n0.466668\n"
        logs: list[str] = []

        def fake_run_text(command, timeout=60.0):
            joined = " ".join(command)
            if "packet=pts_time" in joined:
                return packet_pts
            if "frame=best_effort_timestamp_time" in joined:
                return frame_pts
            return ""

        with (
            patch("services.video_timebase.find_ffprobe_exe", return_value="ffprobe"),
            patch("services.video_timebase._run_text", side_effect=fake_run_text),
        ):
            timebase = build_video_timebase(
                "sample.avi",
                fps=15.0,
                total_frames=6,
                logger=logs.append,
            )

        summary = timebase.summary()
        self.assertEqual(summary["source"], "pts")
        self.assertEqual(summary["pts_frame_count"], 5)
        self.assertAlmostEqual(summary["median_dt"], 1.0 / 15.0, places=5)
        self.assertAlmostEqual(timebase.time_at(5), 5.0 / 15.0, places=5)
        self.assertTrue(any("packet PTS ignored" in message for message in logs))

    def test_sparse_inconsistent_pts_falls_back_to_uniform(self):
        packet_pts = "0.000000\n0.200000\n"
        logs: list[str] = []

        def fake_run_text(command, timeout=60.0):
            joined = " ".join(command)
            if "packet=pts_time" in joined:
                return packet_pts
            if "frame=best_effort_timestamp_time" in joined:
                return ""
            return ""

        with (
            patch("services.video_timebase.find_ffprobe_exe", return_value="ffprobe"),
            patch("services.video_timebase._run_text", side_effect=fake_run_text),
        ):
            timebase = build_video_timebase(
                "sample.avi",
                fps=15.0,
                total_frames=6,
                logger=logs.append,
            )

        summary = timebase.summary()
        self.assertEqual(summary["source"], "uniform")
        self.assertEqual(summary["pts_frame_count"], 0)
        self.assertAlmostEqual(summary["median_dt"], 1.0 / 15.0, places=6)
        self.assertAlmostEqual(timebase.time_at(5), 5.0 / 15.0, places=6)
        self.assertTrue(any("falling back to uniform fps timing" in message for message in logs))

    def test_manual_fps_uses_uniform_timing_and_exclusive_end(self):
        with patch("services.video_timebase._extract_pts_times") as extract_pts:
            timebase = build_video_timebase(
                "sample.mp4",
                fps=15.0,
                total_frames=100,
                prefer_pts=False,
            )

        extract_pts.assert_not_called()
        self.assertEqual(timebase.source, "uniform")
        self.assertEqual(timebase.summary()["long_gap_gt_50ms"], 0)
        self.assertEqual(timebase.frame_before(1.0), 14)
        self.assertEqual(timebase.frame_at_or_after(1.0), 15)


if __name__ == "__main__":
    unittest.main()
