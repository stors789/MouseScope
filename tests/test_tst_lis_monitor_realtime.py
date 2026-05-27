import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from services.tst_session import VisionAnalysisSession


class TSTLISMonitorRealtimeTest(unittest.TestCase):
    def test_feature_extraction_realtime_row_is_pending_not_immobile(self):
        row = VisionAnalysisSession._tst_lis_realtime_row_from_result(
            {
                "roi_id": 1,
                "roi_name": "ROI 1",
                "is_struggling": False,
                "behavior_algorithm": "tst_lis",
                "flow": {"flow_p90_mm_s": 12.3},
            },
            pending=True,
        )

        self.assertIsNone(row["is_struggling"])
        self.assertTrue(row["state_pending"])
        self.assertEqual(row["state_phase"], "feature_extraction")
        self.assertEqual(row["flow_value"], 12.3)

    def test_final_lis_logs_refresh_realtime_rows_with_labels(self):
        session = VisionAnalysisSession.__new__(VisionAnalysisSession)
        session.realtime_data = []

        session._set_tst_lis_realtime_from_logs(
            [
                {"frame": 1, "roi_id": 2, "roi_name": "ROI 2", "is_struggling": False, "flow_p90_mm_s": 5.0},
                {"frame": 1, "roi_id": 1, "roi_name": "ROI 1", "is_struggling": False, "flow_p90_mm_s": 2.0},
                {"frame": 2, "roi_id": 1, "roi_name": "ROI 1", "is_struggling": True, "flow_p90_mm_s": 9.5},
            ]
        )

        self.assertEqual([row["roi_id"] for row in session.realtime_data], [1, 2])
        self.assertTrue(session.realtime_data[0]["is_struggling"])
        self.assertFalse(session.realtime_data[0]["state_pending"])
        self.assertEqual(session.realtime_data[0]["flow_value"], 9.5)


if __name__ == "__main__":
    unittest.main()
