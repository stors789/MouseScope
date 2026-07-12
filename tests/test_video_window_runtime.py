from __future__ import annotations

import csv
import sys
import time
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as mousescope_app
from services.video_metadata import probe_video_info


class VideoWindowRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.client = mousescope_app.app.test_client()
        with self.client.session_transaction() as flask_session:
            flask_session["license_accepted"] = True

    @staticmethod
    def _tst_payload() -> dict:
        roi = {
            "name": "ROI 1",
            "points": [
                {"x": 100, "y": 80},
                {"x": 300, "y": 80},
                {"x": 300, "y": 420},
                {"x": 100, "y": 420},
            ],
            "time_config": {"mode": "global", "start": None, "end": None},
        }
        return {
            "generate_video": True,
            "per_video_configs": [{
                "pixel_to_mm": 1.0,
                "bg_method": "threshold",
                "bg_type": "light",
                "gray_threshold": 127,
                "erosion_kernel": 3,
                "erosion_iter": 2,
                "dilation_kernel": 5,
                "dilation_iter": 3,
                "min_area": 100,
                "auto_behavior_threshold": False,
                "use_tst_legacy_primary": True,
                "rois": [roi],
            }],
        }

    def test_saved_per_video_end_and_fps_limit_csv_and_annotated_video(self):
        video_path = ROOT / "test.mp4"
        video_info = {"path": str(video_path), "original_name": video_path.name, **probe_video_info(video_path)}
        session_id = f"test_video_window_{uuid.uuid4().hex}"
        with TemporaryDirectory() as temp_dir:
            now = time.time()
            mousescope_app.analysis_sessions[session_id] = {
                "experiment_type": "tst",
                "video_paths": [str(video_path)],
                "video_infos": [video_info],
                "video_info": video_info,
                "start_seconds": None,
                "end_seconds": None,
                "output_dir": temp_dir,
                "fps_override": None,
                "is_batch": False,
                "draft_payload": None,
                "draft_updated_at": None,
                "selected_video_idx": 0,
                "created_at": now,
                "last_accessed_at": now,
            }
            try:
                settings = self.client.post(
                    f"/api/video_settings/{session_id}",
                    json={
                        "video_index": 0,
                        "start_sec": 0,
                        "end_sec": 1,
                        "fps_override": 15,
                        "orientation_override": "auto",
                    },
                )
                self.assertEqual(settings.status_code, 200, settings.get_data(as_text=True))

                # A single-video calibration page starts analysis directly with
                # assay fields only. Starting must not discard the video-level
                # window that was saved on the analysis configuration page.
                started = self.client.post(
                    f"/api/start_analysis/{session_id}",
                    json=self._tst_payload(),
                )
                self.assertEqual(started.status_code, 200, started.get_data(as_text=True))

                analysis = mousescope_app.analysis_sessions.get(session_id)["analysis"]
                deadline = time.time() + 60
                while time.time() < deadline and analysis.status not in {"completed", "error", "stopped"}:
                    time.sleep(0.05)
                self.assertEqual(analysis.status, "completed", analysis.message)
                worker = getattr(analysis, "_thread", None)
                if worker is not None:
                    worker.join(timeout=10)
                self.assertEqual(analysis.config["per_video_configs"][0]["end_sec"], 1.0)
                self.assertEqual(analysis.config["per_video_configs"][0]["fps_override"], 15.0)

                log_path = next(Path(path) for path in analysis.result_files if str(path).endswith("_logs.csv"))
                with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 15)
                self.assertLessEqual(max(float(row["time"]) for row in rows), 1.0)

                video_result = next(Path(path) for path in analysis.result_files if str(path).endswith("_analyzed.mp4"))
                result_info = probe_video_info(video_result)
                self.assertGreater(float(result_info["duration_seconds"]), 0.8)
                self.assertLess(float(result_info["duration_seconds"]), 1.2)
            finally:
                mousescope_app.analysis_history.delete_record(session_id)
                mousescope_app.analysis_sessions.delete(session_id)


if __name__ == "__main__":
    unittest.main()
