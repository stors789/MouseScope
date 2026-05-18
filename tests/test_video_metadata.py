from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from services.video_metadata import probe_video_info


class VideoMetadataTest(unittest.TestCase):
    def test_prefers_actual_average_fps_over_nominal_opencv_fps(self):
        expected_fps = 36720000 / 1468111
        with TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "bandicam-vfr.mp4"
            video_path.write_bytes(b"")
            with (
                patch(
                    "services.video_metadata._capture_video_props",
                    return_value={
                        "opencv_fps": 30.0,
                        "width": 1920,
                        "height": 1080,
                        "total_frames": 11016,
                    },
                ),
                patch(
                    "services.video_metadata._probe_with_ffprobe",
                    return_value={
                        "width": 1920,
                        "height": 1080,
                        "duration_seconds": 440.433,
                        "nb_frames": 11016,
                        "avg_frame_rate": 36720000 / 1468111,
                        "nominal_fps": 30.0,
                    },
                ),
                patch("services.video_metadata._probe_with_ffmpeg_banner", return_value={}),
            ):
                info = probe_video_info(video_path)

        self.assertAlmostEqual(info["detected_fps"], expected_fps, places=6)
        self.assertAlmostEqual(info["fps"], info["detected_fps"], places=6)
        self.assertEqual(info["fps_source"], "ffprobe_avg_frame_rate")
        self.assertEqual(info["total_frames"], 11016)
        self.assertEqual(info["nominal_fps"], 30.0)
        self.assertEqual(info["opencv_fps"], 30.0)

    def test_manual_override_keeps_detected_fps_for_display(self):
        expected_fps = 36720000 / 1468111
        with TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "bandicam-vfr.mp4"
            video_path.write_bytes(b"")
            with (
                patch(
                    "services.video_metadata._capture_video_props",
                    return_value={
                        "opencv_fps": 30.0,
                        "width": 1920,
                        "height": 1080,
                        "total_frames": 11016,
                    },
                ),
                patch(
                    "services.video_metadata._probe_with_ffprobe",
                    return_value={
                        "duration_seconds": 440.433,
                        "nb_frames": 11016,
                        "avg_frame_rate": 36720000 / 1468111,
                        "nominal_fps": 30.0,
                    },
                ),
                patch("services.video_metadata._probe_with_ffmpeg_banner", return_value={}),
            ):
                info = probe_video_info(video_path, fps_override="20")

        self.assertAlmostEqual(info["detected_fps"], expected_fps, places=6)
        self.assertEqual(info["fps"], 20.0)

    def test_ffmpeg_duration_fallback_handles_missing_ffprobe(self):
        expected_fps = 11016 / 440.433
        with TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "bandicam-vfr.mp4"
            video_path.write_bytes(b"")
            with (
                patch(
                    "services.video_metadata._capture_video_props",
                    return_value={
                        "opencv_fps": 30.0,
                        "width": 1920,
                        "height": 1080,
                        "total_frames": 11016,
                    },
                ),
                patch("services.video_metadata._probe_with_ffprobe", return_value={}),
                patch(
                    "services.video_metadata._probe_with_ffmpeg_banner",
                    return_value={"duration_seconds": 440.433, "nominal_fps": 30.0},
                ),
            ):
                info = probe_video_info(video_path)

        self.assertAlmostEqual(info["detected_fps"], expected_fps, places=6)
        self.assertEqual(info["fps_source"], "frame_count_over_duration")


if __name__ == "__main__":
    unittest.main()
