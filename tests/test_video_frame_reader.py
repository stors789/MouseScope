from __future__ import annotations

import unittest

import cv2
import numpy as np

from services.video_frame_reader import _apply_orientation_meta, _normalise_orientation_degrees


class VideoFrameReaderTest(unittest.TestCase):
    def test_orientation_normalisation_accepts_right_angles(self):
        self.assertEqual(_normalise_orientation_degrees(90), 90)
        self.assertEqual(_normalise_orientation_degrees(-90), 270)
        self.assertEqual(_normalise_orientation_degrees("180.0"), 180)
        self.assertEqual(_normalise_orientation_degrees(45), 0)
        self.assertEqual(_normalise_orientation_degrees(None), 0)

    def test_apply_orientation_meta_rotates_like_opencv_auto_orientation(self):
        frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((2, 3, 3))

        np.testing.assert_array_equal(
            _apply_orientation_meta(frame, 90),
            cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
        )
        np.testing.assert_array_equal(
            _apply_orientation_meta(frame, 180),
            cv2.rotate(frame, cv2.ROTATE_180),
        )
        np.testing.assert_array_equal(
            _apply_orientation_meta(frame, 270),
            cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE),
        )
        self.assertIs(_apply_orientation_meta(frame, 0), frame)


if __name__ == "__main__":
    unittest.main()
