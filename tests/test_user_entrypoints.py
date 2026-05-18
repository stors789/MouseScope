from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as mousescope_app


class UserEntrypointGuardTest(unittest.TestCase):
    def setUp(self):
        self.app = mousescope_app.app
        self.client = self.app.test_client()
        with self.client.session_transaction() as flask_session:
            flask_session["license_accepted"] = True

    def test_license_redirect_preserves_original_get_path(self):
        fresh_client = self.app.test_client()
        response = fresh_client.get("/upload", follow_redirects=False)

        self.assertEqual(response.status_code, 302)
        self.assertIn("/license?next=/upload", response.headers["Location"])

    def test_upload_rejects_invalid_fps_before_creating_analysis(self):
        response = self.client.post(
            "/do_upload",
            data={"local_paths": str(ROOT / "test.mp4"), "fps_override": "abc"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("FPS", response.get_json()["error"])

    def test_upload_rejects_reversed_time_window(self):
        response = self.client.post(
            "/do_upload",
            data={"local_paths": str(ROOT / "test.mp4"), "start_seconds": "10", "end_seconds": "1"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("结束秒数", response.get_json()["error"])

    def test_upload_rejects_zero_length_end_time(self):
        response = self.client.post(
            "/do_upload",
            data={"local_paths": str(ROOT / "test.mp4"), "end_seconds": "0"},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("结束秒数", response.get_json()["error"])

    def test_local_path_validation_checks_server_filesystem(self):
        response = self.client.post(
            "/api/validate_local_paths",
            json={"local_paths": str(ROOT / "test.mp4")},
        )

        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertGreaterEqual(payload["valid_count"], 1)

    def test_local_path_validation_rejects_missing_path(self):
        with TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.mp4"
            response = self.client.post(
                "/api/validate_local_paths",
                json={"local_paths": str(missing)},
            )

        payload = response.get_json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["invalid_count"], 1)

    def test_tasks_rejects_invalid_limit(self):
        response = self.client.get("/api/tasks?limit=abc")

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])


if __name__ == "__main__":
    unittest.main()
