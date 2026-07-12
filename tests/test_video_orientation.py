from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from services.video_orientation import (
    OrientedVideoCapture,
    _orientation_from_ffprobe_stream,
    oriented_dimensions,
)


class _FakeCapture:
    def __init__(self, frame):
        self.frame = frame
        self.released = False
        self.set_calls = []

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame.copy()

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frame.shape[1])
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frame.shape[0])
        return 0.0

    def set(self, prop_id, value):
        self.set_calls.append((prop_id, value))
        return True

    def release(self):
        self.released = True


class VideoOrientationTest(unittest.TestCase):
    def test_ffprobe_display_matrix_negative_90_means_clockwise_90(self):
        self.assertEqual(
            _orientation_from_ffprobe_stream({"side_data_list": [{"rotation": -90}]}),
            90,
        )
        self.assertEqual(
            _orientation_from_ffprobe_stream({"tags": {"rotate": "270"}}),
            270,
        )

    def test_oriented_dimensions_swap_only_for_quarter_turns(self):
        self.assertEqual(oriented_dimensions(434, 432, 90), (432, 434))
        self.assertEqual(oriented_dimensions(434, 432, 270), (432, 434))
        self.assertEqual(oriented_dimensions(434, 432, 180), (434, 432))

    def test_capture_disables_backend_auto_rotation_and_rotates_once(self):
        raw = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))
        fake = _FakeCapture(raw)
        with (
            patch("services.video_orientation.cv2.VideoCapture", return_value=fake),
            patch("services.video_orientation.resolve_video_orientation", return_value=(90, "ffprobe")),
        ):
            cap = OrientedVideoCapture("phone.mov")
            ok, frame = cap.read()

        self.assertTrue(ok)
        np.testing.assert_array_equal(frame, cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE))
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 2.0)
        self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 3.0)
        self.assertIn((cv2.CAP_PROP_ORIENTATION_AUTO, 0), fake.set_calls)


if __name__ == "__main__":
    unittest.main()
