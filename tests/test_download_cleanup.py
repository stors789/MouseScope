from __future__ import annotations

import time
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import app as mousescope_app


class _DownloadAnalysis:
    def __init__(self, files):
        self.result_files = [str(path) for path in files]

    def get_download_file_map(self):
        return {"csv": list(self.result_files)}


class DownloadCleanupTest(unittest.TestCase):
    def setUp(self):
        self.client = mousescope_app.app.test_client()
        with self.client.session_transaction() as flask_session:
            flask_session["license_accepted"] = True

    def test_multi_file_download_uses_a_cleaned_temporary_zip(self):
        session_id = f"test_download_{uuid.uuid4().hex}"
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = [root / "first.csv", root / "second.csv"]
            for index, path in enumerate(files):
                path.write_text(f"value\n{index}\n", encoding="utf-8")
            now = time.time()
            mousescope_app.analysis_sessions[session_id] = {
                "analysis": _DownloadAnalysis(files),
                "output_dir": str(root),
                "video_paths": [],
                "video_infos": [],
                "created_at": now,
                "last_accessed_at": now,
            }
            prefix = f"{session_id}_csv_"
            try:
                response = self.client.get(f"/api/download/{session_id}/csv", buffered=True)
                self.assertEqual(response.status_code, 200)
                self.assertIn("MouseScope_csv.zip", response.headers.get("Content-Disposition", ""))
                response.close()

                self.assertFalse(list(root.glob("*.zip")))
                self.assertFalse([
                    path for path in Path(mousescope_app.RESULT_DIR).glob(f"{prefix}*.zip")
                    if path.exists()
                ])
            finally:
                mousescope_app.analysis_sessions.delete(session_id)


if __name__ == "__main__":
    unittest.main()
