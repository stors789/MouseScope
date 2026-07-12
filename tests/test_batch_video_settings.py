from __future__ import annotations

import unittest

from services.analysis_builders import build_open_field_config


class BatchVideoSettingsConfigTest(unittest.TestCase):
    def test_per_video_timing_fps_and_orientation_override_reach_runtime_config(self):
        sess = {
            "video_infos": [{"path": "one.mp4"}, {"path": "two.mp4"}],
            "output_dir": "results",
            "start_seconds": None,
            "end_seconds": None,
            "fps_override": None,
        }
        payload = {
            "per_video_configs": [
                {"start_sec": 5, "end_sec": 35, "fps_override": 25, "orientation_override": 90},
                {"start_sec": 0, "end_sec": None, "fps_override": None, "orientation_override": 0},
            ]
        }

        config = build_open_field_config(sess, payload)

        self.assertEqual(config["per_video_configs"][0]["start_sec"], 5.0)
        self.assertEqual(config["per_video_configs"][0]["end_sec"], 35.0)
        self.assertEqual(config["per_video_configs"][0]["fps_override"], 25.0)
        self.assertEqual(config["per_video_configs"][0]["orientation_override"], 90)
        self.assertEqual(config["per_video_configs"][1]["start_sec"], 0.0)
        self.assertIsNone(config["per_video_configs"][1]["end_sec"])
        self.assertIsNone(config["per_video_configs"][1]["fps_override"])
        self.assertEqual(config["per_video_configs"][1]["orientation_override"], 0)


if __name__ == "__main__":
    unittest.main()
