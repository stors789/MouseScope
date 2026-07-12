from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as mousescope_app
from services.api_errors import build_api_error


class BilingualErrorTest(unittest.TestCase):
    def setUp(self):
        self.client = mousescope_app.app.test_client()
        with self.client.session_transaction() as flask_session:
            flask_session["license_accepted"] = True

    def test_common_error_payload_has_stable_bilingual_fields(self):
        payload = build_api_error("项目不存在")
        self.assertEqual(payload["error_code"], "project_not_found")
        self.assertEqual(payload["error"], "项目不存在")
        self.assertEqual(payload["error_en"], "The project does not exist.")

    def test_project_missing_response_is_bilingual(self):
        response = self.client.get("/api/projects/does-not-exist")
        payload = response.get_json()
        self.assertEqual(response.status_code, 404)
        self.assertEqual(payload["error_code"], "project_not_found")
        self.assertIn("project", payload["error_en"].lower())

    def test_missing_analysis_session_response_is_bilingual(self):
        response = self.client.get("/api/batch_state/does-not-exist")
        payload = response.get_json()
        self.assertEqual(response.status_code, 404)
        self.assertEqual(payload["error_code"], "session_not_found")
        self.assertNotRegex(payload["error_en"], r"[\u3400-\u9fff]")

    def test_task_validation_response_is_bilingual(self):
        response = self.client.get("/api/tasks?limit=invalid")
        payload = response.get_json()
        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error_code"], "invalid_task_limit")
        self.assertNotRegex(payload["error_en"], r"[\u3400-\u9fff]")


class ProjectRoiI18nStaticTest(unittest.TestCase):
    SUBJECT_TEMPLATES = (
        "calibrate.html",
        "calibrate_openfield.html",
        "calibrate_forced_swim.html",
        "calibrate_nor.html",
        "calibrate_maze.html",
        "calibrate_water_maze.html",
        "calibrate_light_dark.html",
        "calibrate_three_chamber.html",
    )

    def test_project_subject_controls_have_english_copy(self):
        for filename in self.SUBJECT_TEMPLATES:
            source = (ROOT / "templates" / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertIn("Select a registered animal", source)
                self.assertRegex(source, r"Animal\s*/\s*(?:Experimental Unit|Animal)")

    def test_project_subject_placeholder_is_not_unconditionally_chinese(self):
        forbidden = '<option value="">请选择已有小鼠</option>'
        for filename in self.SUBJECT_TEMPLATES:
            source = (ROOT / "templates" / filename).read_text(encoding="utf-8")
            self.assertNotIn(forbidden, source, filename)

    def test_shared_client_error_selector_prefers_error_en(self):
        source = (ROOT / "static" / "js" / "i18n.js").read_text(encoding="utf-8")
        self.assertIn("function apiErrorText", source)
        self.assertIn("data.error_en", source)

    def test_tst_page_i18n_does_not_shadow_shared_translator(self):
        source = (ROOT / "templates" / "calibrate.html").read_text(encoding="utf-8")
        self.assertIn("function applyTstPageI18n()", source)
        self.assertNotIn("function applyPageI18n()", source)
        self.assertIn("'header.title': 'Tail Suspension Test Setup'", source)


if __name__ == "__main__":
    unittest.main()
