from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from services.analysis_runtime import AnalysisSessionBase


class DummyAnalysisSession(AnalysisSessionBase):
    def __init__(self, total_videos: int):
        self.video_paths = [f"video-{idx}.mp4" for idx in range(total_videos)]
        self.progress = 0
        self.message = ""
        self.status = "analyzing"
        self.logs = []

    def _log(self, message, *, force_console=False):
        self.logs.append(str(message))


class BatchProgressAggregationTest(unittest.TestCase):
    def test_parallel_batch_progress_uses_average_not_video_order_slot(self):
        session = DummyAnalysisSession(total_videos=9)
        session._begin_batch_progress(9, aggregate=True)

        session._set_video_progress(8, 0.5, "Frame 50/100 ({progress}%)")
        self.assertEqual(session.progress, 5)
        self.assertEqual(session.message, "Frame 50/100 (5%)")

        session._set_video_progress(0, 0.5)
        self.assertEqual(session.progress, 11)

        session._set_video_progress(8, 0.2)
        self.assertEqual(session.progress, 11)

        session._mark_batch_video_completed(8)
        self.assertEqual(session.progress, 16)

    def test_sequential_batch_progress_keeps_original_slot_behavior(self):
        session = DummyAnalysisSession(total_videos=4)
        session._begin_batch_progress(4, aggregate=False)

        session._set_video_progress(2, 0.5)

        self.assertEqual(session.progress, 62)

    def test_zero_decoded_frame_guard_raises_clear_error(self):
        session = DummyAnalysisSession(total_videos=1)

        with self.assertRaisesRegex(RuntimeError, "decoded 0 frames"):
            session._ensure_analysis_frames_processed(
                video_path="bad-codec.mp4",
                label="OFT",
                frame_idx=0,
                source_frames_done=0,
            )

    def test_zero_decoded_frame_guard_allows_real_samples(self):
        session = DummyAnalysisSession(total_videos=1)

        session._ensure_analysis_frames_processed(
            video_path="ok.mp4",
            label="OFT",
            frame_idx=1,
            source_frames_done=1,
        )


if __name__ == "__main__":
    unittest.main()
