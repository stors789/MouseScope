from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import app as mousescope_app
from services.desktop_preferences import DesktopPreferencesStore, license_sha256


class DesktopPreferencesStoreTest(unittest.TestCase):
    def test_consent_and_tour_completion_survive_new_store_instance(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            license_path = root / "LICENSE"
            license_path.write_text("test license", encoding="utf-8")
            preferences_path = root / "desktop_preferences.json"
            digest = license_sha256(license_path)
            key = "mousescope_tour_home_v2"

            first_run = DesktopPreferencesStore(preferences_path)
            self.assertFalse(first_run.is_license_accepted(digest))
            first_run.accept_license(digest)
            first_run.complete_tour(key)

            second_run = DesktopPreferencesStore(preferences_path)
            self.assertTrue(second_run.is_license_accepted(digest))
            self.assertEqual(second_run.completed_tour_keys(), {key})

            license_path.write_text("changed license", encoding="utf-8")
            self.assertFalse(second_run.is_license_accepted(license_sha256(license_path)))

    def test_invalid_tour_key_is_not_written(self):
        with TemporaryDirectory() as temp_dir:
            store = DesktopPreferencesStore(Path(temp_dir) / "desktop_preferences.json")
            with self.assertRaises(ValueError):
                store.complete_tour("../../not-a-tour")
            self.assertFalse(store.path.exists())


class DesktopOnboardingRestartTest(unittest.TestCase):
    def setUp(self):
        self.app = mousescope_app.app
        self.original_config = {
            key: self.app.config.get(key)
            for key in (
                "TESTING",
                "MOUSESCOPE_DESKTOP_MODE",
                "MOUSESCOPE_DESKTOP_TOKEN",
                "MOUSESCOPE_DESKTOP_ORIGIN",
                "MOUSESCOPE_DESKTOP_PREFERENCES_STORE",
                "MOUSESCOPE_LICENSE_SHA256",
            )
        }

    def tearDown(self):
        self.app.config.update(self.original_config)

    def test_consent_and_tour_state_survive_new_client_and_random_port(self):
        with TemporaryDirectory() as temp_dir:
            store = DesktopPreferencesStore(Path(temp_dir) / "desktop_preferences.json")
            digest = license_sha256(Path(mousescope_app.BUNDLE_DIR) / "LICENSE")
            first_origin = "http://127.0.0.1:49152"
            self.app.config.update(
                TESTING=True,
                MOUSESCOPE_DESKTOP_MODE=True,
                MOUSESCOPE_DESKTOP_TOKEN="first-run-token",
                MOUSESCOPE_DESKTOP_ORIGIN=first_origin,
                MOUSESCOPE_DESKTOP_PREFERENCES_STORE=store,
                MOUSESCOPE_LICENSE_SHA256=digest,
            )

            first_client = self.app.test_client()
            bootstrap = first_client.get(
                "/__desktop_bootstrap?token=first-run-token",
                base_url=first_origin,
            )
            self.assertEqual(bootstrap.status_code, 302)
            first_page = first_client.get("/", base_url=first_origin, follow_redirects=True)
            self.assertEqual(first_page.request.path, "/license")
            self.assertFalse(store.path.exists())

            acceptance = first_client.post(
                "/api/accept_license",
                base_url=first_origin,
                headers={"Origin": first_origin},
            )
            self.assertEqual(acceptance.status_code, 200)
            tour_key = "mousescope_tour_home_v2"
            completion = first_client.post(
                "/api/desktop/preferences/tours",
                base_url=first_origin,
                headers={"Origin": first_origin},
                json={"key": tour_key},
            )
            self.assertEqual(completion.status_code, 200)

            saved = json.loads(store.path.read_text(encoding="utf-8"))
            self.assertEqual(saved["license"]["sha256"], digest)
            self.assertIn(tour_key, saved["completed_tours"])

            second_origin = "http://127.0.0.1:58341"
            self.app.config.update(
                MOUSESCOPE_DESKTOP_TOKEN="second-run-token",
                MOUSESCOPE_DESKTOP_ORIGIN=second_origin,
            )
            second_client = self.app.test_client()
            restarted = second_client.get(
                "/__desktop_bootstrap?token=second-run-token",
                base_url=second_origin,
                follow_redirects=True,
            )
            self.assertEqual(restarted.status_code, 200)
            self.assertNotEqual(restarted.request.path, "/license")
            body = restarted.get_data(as_text=True)
            self.assertIn("licenseAccepted: true", body)
            self.assertIn(f'"{tour_key}"', body)

    def test_web_mode_acceptance_remains_session_only(self):
        with TemporaryDirectory() as temp_dir:
            store = DesktopPreferencesStore(Path(temp_dir) / "desktop_preferences.json")
            self.app.config.update(
                TESTING=True,
                MOUSESCOPE_DESKTOP_MODE=False,
                MOUSESCOPE_DESKTOP_TOKEN="",
                MOUSESCOPE_DESKTOP_ORIGIN="",
                MOUSESCOPE_DESKTOP_PREFERENCES_STORE=store,
            )
            client = self.app.test_client()
            response = client.post("/api/accept_license")
            self.assertEqual(response.status_code, 200)
            self.assertFalse(store.path.exists())


if __name__ == "__main__":
    unittest.main()
