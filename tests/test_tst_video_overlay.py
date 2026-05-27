import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import vision_tracker


class TSTVideoOverlayTest(unittest.TestCase):
    def _base_lis_result(self):
        return {
            "detected": True,
            "roi_id": 1,
            "roi_name": "ROI 1",
            "centroid": (100.0, 120.0),
            "contour": None,
            "bbox": (90, 100, 20, 40),
            "roi_bbox": (60, 60, 160, 120),
            "is_struggling": True,
            "is_swinging": False,
            "primary_behavior_algorithm": "tst_lis",
            "behavior_algorithm": "tst_lis",
            "lis_reason": "motion_score_active_long_debug_reason",
            "lis_passive_swing": False,
            "is_in_stats_window": True,
            "flow": {"flow_p90_mm_s": 42.0},
            "flow_p90": 42.0,
            "aspect_ratio": 2.0,
            "trajectory": [],
        }

    def _draw_and_capture_text(self, result):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        captured = []

        def fake_put_text(image, text, *args, **kwargs):
            captured.append(str(text))
            return image

        with patch.object(vision_tracker.cv2, "putText", side_effect=fake_put_text):
            vision_tracker.draw_results(frame, [result], show_trajectory=False)
        return captured

    def test_lis_overlay_uses_legacy_style_labels_without_debug_reason(self):
        captured = self._draw_and_capture_text(self._base_lis_result())

        self.assertIn("[REC] R1 | Struggle", captured)
        self.assertIn("F:42.0 AR:2.00", captured)
        self.assertFalse(any("LIS" in text for text in captured))
        self.assertFalse(any("motion_score" in text for text in captured))

    def test_lis_passive_swing_is_rendered_like_legacy_swing(self):
        result = copy.deepcopy(self._base_lis_result())
        result["is_struggling"] = False
        result["lis_passive_swing"] = True

        captured = self._draw_and_capture_text(result)

        self.assertIn("[REC] R1 | Swing", captured)
        self.assertFalse(any("LIS passive" in text for text in captured))


if __name__ == "__main__":
    unittest.main()
