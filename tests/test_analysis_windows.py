from __future__ import annotations

import unittest

from services.analysis_windows import (
    effective_roi_window,
    effective_video_duration_seconds,
    validate_analysis_windows,
    validate_roi_windows,
)


class AnalysisWindowTest(unittest.TestCase):
    def test_precise_roi_window_is_clipped_to_video_window(self):
        self.assertEqual(
            effective_roi_window(10, 600, {"mode": "precise", "start": 60, "end": 700}),
            (60.0, 600.0),
        )

    def test_global_roi_window_follows_video_window(self):
        self.assertEqual(effective_roi_window(10, 600, {"mode": "global"}), (10.0, 600.0))

    def test_roi_window_without_intersection_is_rejected(self):
        errors = validate_roi_windows(
            [{"time_config": {"mode": "precise", "start": 0, "end": 5}}],
            video_start=10,
            video_end=600,
            video_label="视频 1",
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("没有交集", errors[0])

    def test_analysis_payload_uses_per_video_outer_window(self):
        payload = {
            "per_video_configs": [
                {
                    "start_sec": 20,
                    "end_sec": 100,
                    "rois": [{"time_config": {"mode": "precise", "start": 0, "end": 10}}],
                }
            ]
        }
        errors = validate_analysis_windows(payload, [{"duration_seconds": 120}])
        self.assertEqual(len(errors), 1)
        self.assertIn("第 1 个视频", errors[0])

    def test_relative_roi_window_validates_its_own_duration(self):
        errors = validate_roi_windows(
            [{"time_config": {"mode": "relative", "start": 30, "end": 10}}],
            video_start=0,
            video_end=60,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("结束时间必须大于开始时间", errors[0])

    def test_relative_roi_start_cannot_begin_after_video_window(self):
        errors = validate_roi_windows(
            [{"time_config": {"mode": "relative", "start": 20, "end": 30}}],
            video_start=5,
            video_end=15,
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("相对开始时间超出视频级时间窗", errors[0])

    def test_fps_override_reinterprets_duration_and_rejects_out_of_range_window(self):
        info = {"total_frames": 900, "duration_seconds": 30, "detected_fps": 30}
        self.assertEqual(effective_video_duration_seconds(info, 15), 60.0)
        errors = validate_analysis_windows(
            {"per_video_configs": [{"fps_override": 15, "start_sec": 60}]},
            [info],
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("开始时间必须小于视频时长", errors[0])

    def test_video_level_values_require_valid_order_range_and_finite_numbers(self):
        info = {"total_frames": 900, "duration_seconds": 30, "detected_fps": 30}
        cases = [
            ({"start_sec": 20, "end_sec": 1}, "结束时间必须大于开始时间"),
            ({"start_sec": -1}, "开始时间不能小于 0"),
            ({"end_sec": 0}, "结束时间必须大于 0"),
            ({"fps_override": 0}, "FPS 必须在"),
            ({"start_sec": "not-a-number"}, "开始时间必须是有限数字"),
            ({"end_sec": float("nan")}, "结束时间必须是有限数字"),
            ({"fps_override": float("inf")}, "FPS 必须是有限数字"),
        ]
        for config, expected in cases:
            with self.subTest(config=config):
                errors = validate_analysis_windows({"per_video_configs": [config]}, [info])
                self.assertTrue(any(expected in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
