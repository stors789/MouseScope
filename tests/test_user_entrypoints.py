from __future__ import annotations

import sys
import copy
import time
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as mousescope_app
from services.video_metadata import probe_video_info
from services.reproducibility import (
    RUN_SNAPSHOT_FILENAME,
    build_run_snapshot,
    load_run_snapshot,
    write_run_snapshot,
)


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

    def _seed_single_session(self):
        video_path = ROOT / "test.mp4"
        video_info = {"path": str(video_path), "original_name": video_path.name, **probe_video_info(video_path)}
        session_id = f"test_single_settings_{uuid.uuid4().hex}"
        now = time.time()
        mousescope_app.analysis_sessions[session_id] = {
            "experiment_type": "open_field",
            "video_paths": [str(video_path)],
            "video_infos": [video_info],
            "start_seconds": None,
            "end_seconds": None,
            "output_dir": None,
            "fps_override": None,
            "is_batch": False,
            "draft_payload": None,
            "draft_updated_at": None,
            "selected_video_idx": 0,
            "created_at": now,
            "last_accessed_at": now,
        }
        return session_id

    def test_upload_redirects_to_unified_analysis_config(self):
        response = self.client.post(
            "/do_upload",
            data={
                "local_paths": str(ROOT / "test.mp4"),
                "fps_override": "abc",
                "start_seconds": "10",
                "end_seconds": "1",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("/analysis-config/", payload["redirect_url"])
        session_id = payload["redirect_url"].rsplit("/", 1)[-1]
        try:
            session = mousescope_app.analysis_sessions.get(session_id)
            self.assertIsNone(session["start_seconds"])
            self.assertIsNone(session["end_seconds"])
            self.assertIsNone(session["fps_override"])
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_mpeg_extensions_are_accepted_by_local_path_validation(self):
        with TemporaryDirectory() as temp_dir:
            paths = []
            for filename in ("recording.mpg", "recording.mpeg"):
                path = Path(temp_dir) / filename
                path.write_bytes(b"mpeg")
                paths.append(str(path))

            response = self.client.post(
                "/api/validate_local_paths",
                json={"local_paths": "\n".join(paths)},
            )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["valid_count"], 2)

    def test_single_video_settings_are_validated_on_analysis_config_api(self):
        session_id = self._seed_single_session()
        try:
            cases = [
                ({"video_index": 0, "fps_override": "abc"}, "FPS"),
                ({"video_index": 0, "start_sec": 10, "end_sec": 1}, "结束时间"),
                ({"video_index": 0, "end_sec": 0}, "结束时间"),
                ({"video_index": 0, "start_sec": 40}, "起始时间"),
            ]
            for payload, expected in cases:
                with self.subTest(payload=payload):
                    response = self.client.post(f"/api/video_settings/{session_id}", json=payload)
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(expected, response.get_json()["error"])
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_fps_override_updates_effective_duration_and_validation(self):
        session_id = self._seed_single_session()
        try:
            response = self.client.post(
                f"/api/video_settings/{session_id}",
                json={"video_index": 0, "fps_override": 15, "end_sec": 50},
            )
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            video_info = response.get_json()["cards"][0]["video_info"]
            self.assertEqual(video_info["fps"], 15.0)
            self.assertAlmostEqual(video_info["duration_seconds"], 902 / 15.0, places=3)

            response = self.client.post(
                f"/api/video_settings/{session_id}",
                json={"video_index": 0, "fps_override": 15, "end_sec": 61},
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("视频时长", response.get_json()["error"])
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_video_settings_validation_only_does_not_mutate_session(self):
        session_id = self._seed_single_session()
        try:
            before = copy.deepcopy(mousescope_app.analysis_sessions.get(session_id))
            response = self.client.post(
                f"/api/video_settings/{session_id}",
                json={
                    "video_index": 0,
                    "start_sec": 1,
                    "end_sec": 20,
                    "fps_override": 15,
                    "orientation_override": 90,
                    "validate_only": True,
                },
            )
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            self.assertTrue(response.get_json()["validated"])
            after = mousescope_app.analysis_sessions.get(session_id)
            self.assertEqual(after.get("draft_payload"), before.get("draft_payload"))
            self.assertEqual(after.get("draft_updated_at"), before.get("draft_updated_at"))
            self.assertEqual(after.get("video_infos"), before.get("video_infos"))
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_draft_and_start_reject_invalid_video_level_windows(self):
        session_id = self._seed_single_session()
        invalid = {"per_video_configs": [{"start_sec": 20, "end_sec": 1, "rois": []}]}
        try:
            response = self.client.post(f"/api/draft/{session_id}", json=invalid)
            self.assertEqual(response.status_code, 400, response.get_data(as_text=True))
            self.assertEqual(response.get_json()["error_code"], "invalid_draft")

            session = mousescope_app.analysis_sessions.get(session_id)
            session["draft_payload"] = copy.deepcopy(invalid)
            mousescope_app.analysis_sessions.persist(session_id)
            response = self.client.post(
                f"/api/start_analysis/{session_id}",
                json={"use_saved_draft": True},
            )
            self.assertEqual(response.status_code, 400, response.get_data(as_text=True))
            self.assertEqual(response.get_json()["error_code"], "invalid_draft")
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_configuration_writes_reject_non_object_json_and_active_runs(self):
        session_id = self._seed_single_session()
        try:
            endpoints = [
                f"/api/video_settings/{session_id}",
                f"/api/analysis_config/{session_id}",
                f"/api/draft_action/{session_id}",
            ]
            for endpoint in endpoints:
                with self.subTest(endpoint=endpoint):
                    response = self.client.post(endpoint, json=["invalid"])
                    self.assertEqual(response.status_code, 400, response.get_data(as_text=True))

            class ActiveAnalysis:
                status = "analyzing"
                is_running = True

            session = mousescope_app.analysis_sessions.get(session_id)
            session["analysis"] = ActiveAnalysis()
            for endpoint, payload in [
                (f"/api/video_settings/{session_id}", {"video_index": 0}),
                (f"/api/analysis_config/{session_id}", {"output_dir": ""}),
                (f"/api/draft_action/{session_id}", {"action": "reset", "target_index": 0}),
                (f"/api/draft/{session_id}", {"per_video_configs": [{}]}),
            ]:
                with self.subTest(active_endpoint=endpoint):
                    response = self.client.post(endpoint, json=payload)
                    self.assertEqual(response.status_code, 409, response.get_data(as_text=True))
                    self.assertEqual(response.get_json()["error_code"], "analysis_config_locked")
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_single_video_analysis_config_page_saves_output_folder(self):
        session_id = self._seed_single_session()
        try:
            page = self.client.get(f"/analysis-config/{session_id}")
            self.assertEqual(page.status_code, 200)
            self.assertIn('data-tour-page="analysis-config"', page.get_data(as_text=True))

            with TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir) / "custom-results"
                response = self.client.post(
                    f"/api/analysis_config/{session_id}",
                    json={"output_dir": str(output_dir)},
                )
                self.assertEqual(response.status_code, 200)
                expected = str(output_dir)
                self.assertTrue(output_dir.is_dir())
                self.assertEqual(response.get_json()["output_dir"], expected)
                self.assertEqual(mousescope_app.analysis_sessions.get(session_id)["output_dir"], expected)
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_output_location_must_be_a_writable_directory(self):
        session_id = self._seed_single_session()
        try:
            with TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / "not-a-folder.mp4"
                file_path.write_bytes(b"not a directory")
                response = self.client.post(
                    f"/api/analysis_config/{session_id}",
                    json={"output_dir": str(file_path)},
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["error_code"], "output_dir_invalid")
                self.assertIsNone(mousescope_app.analysis_sessions.get(session_id)["output_dir"])

                session = mousescope_app.analysis_sessions.get(session_id)
                session["output_dir"] = str(file_path)
                mousescope_app.analysis_sessions.persist(session_id)
                started = self.client.post(
                    f"/api/start_analysis/{session_id}",
                    json={"per_video_configs": [{}]},
                )
                self.assertEqual(started.status_code, 400)
                self.assertEqual(started.get_json()["error_code"], "output_dir_invalid")
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_project_analysis_config_keeps_project_managed_output_folder(self):
        session_id = self._seed_single_session()
        session = mousescope_app.analysis_sessions.get(session_id)
        project_output = str(ROOT / "data" / "projects" / "example" / "results")
        session["output_dir"] = project_output
        session["project_context"] = {"project_id": "example"}
        mousescope_app.analysis_sessions.persist(session_id)
        try:
            page = self.client.get(f"/analysis-config/{session_id}")
            self.assertEqual(page.status_code, 200)
            self.assertIn('id="analysisOutputDir"', page.get_data(as_text=True))
            self.assertIn("readonly", page.get_data(as_text=True))

            response = self.client.post(
                f"/api/analysis_config/{session_id}",
                json={"output_dir": "data/should-not-replace-project-output"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.get_json()["project_managed"])
            self.assertEqual(mousescope_app.analysis_sessions.get(session_id)["output_dir"], project_output)
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_editing_replay_configuration_detaches_exact_replay_snapshot(self):
        session_id = self._seed_single_session()
        session = mousescope_app.analysis_sessions.get(session_id)
        session["replay_resolved_config"] = {"gray_threshold": 99}
        session["replay_of"] = {"session_id": "source", "snapshot_path": "source/run_snapshot.json"}
        mousescope_app.analysis_sessions.persist(session_id)
        try:
            with TemporaryDirectory() as temp_dir:
                response = self.client.post(
                    f"/api/analysis_config/{session_id}",
                    json={"output_dir": temp_dir},
                )
            self.assertEqual(response.status_code, 200)
            updated = mousescope_app.analysis_sessions.get(session_id)
            self.assertNotIn("replay_resolved_config", updated)
            self.assertNotIn("replay_of", updated)
            self.assertTrue(updated["replay_detached_by_edit"])
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_roi_window_must_intersect_video_level_window(self):
        session_id = self._seed_single_session()
        try:
            response = self.client.post(
                f"/api/video_settings/{session_id}",
                json={"video_index": 0, "start_sec": 20, "end_sec": 30, "orientation_override": "auto"},
            )
            self.assertEqual(response.status_code, 200)

            response = self.client.post(
                f"/api/draft/{session_id}",
                json={
                    "per_video_configs": [
                        {"rois": [{"time_config": {"mode": "precise", "start": 0, "end": 10}}]}
                    ]
                },
            )
            self.assertEqual(response.status_code, 400)
            self.assertIn("没有交集", response.get_json()["error"])
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

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

    def test_experiment_defaults_api_rejects_unsafe_kernel_size(self):
        response = self.client.post(
            "/api/experiment_defaults",
            json={"defaults": {"open_field": {"erosion_kernel": 0}}},
        )

        payload = response.get_json()
        self.assertEqual(response.status_code, 400)
        self.assertFalse(payload["success"])
        self.assertEqual(payload["validation_errors"][0]["field"], "erosion_kernel")

    def test_batch_video_settings_are_saved_per_video_and_survive_calibration_draft(self):
        video_path = ROOT / "test.mp4"
        video_info = {
            "path": str(video_path),
            "original_name": video_path.name,
            **probe_video_info(video_path),
        }
        session_id = f"test_batch_settings_{uuid.uuid4().hex}"
        now = time.time()
        mousescope_app.analysis_sessions[session_id] = {
            "experiment_type": "open_field",
            "video_paths": [str(video_path), str(video_path)],
            "video_infos": [dict(video_info), dict(video_info)],
            "start_seconds": None,
            "end_seconds": None,
            "output_dir": str(ROOT / "data" / "results"),
            "fps_override": None,
            "is_batch": True,
            "draft_payload": None,
            "draft_updated_at": None,
            "selected_video_idx": 0,
            "created_at": now,
            "last_accessed_at": now,
        }
        try:
            response = self.client.post(
                f"/api/batch_video_settings/{session_id}",
                json={
                    "video_index": 0,
                    "start_sec": 1.25,
                    "end_sec": 3.0,
                    "fps_override": 20,
                    "orientation_override": "0",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.get_json()["success"])

            response = self.client.post(
                f"/api/draft/{session_id}",
                json={"per_video_configs": [{"rois": []}, {}]},
            )
            self.assertEqual(response.status_code, 200)
            config = mousescope_app.analysis_sessions.get(session_id)["draft_payload"]["per_video_configs"][0]
            self.assertEqual(config["start_sec"], 1.25)
            self.assertEqual(config["end_sec"], 3.0)
            self.assertEqual(config["fps_override"], 20.0)
            self.assertEqual(config["orientation_override"], 0)
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_copy_and_reset_keep_workspace_orientation_and_project_identity_consistent(self):
        session_id = self._seed_single_session()
        try:
            session = mousescope_app.analysis_sessions.get(session_id)
            session["is_batch"] = True
            session["video_paths"] = session["video_paths"] * 2
            session["video_infos"] = [
                copy.deepcopy(session["video_infos"][0]),
                copy.deepcopy(session["video_infos"][0]),
            ]
            source_config = {
                "orientation_override": 90,
                "start_sec": 0,
                "end_sec": 20,
                "rois": [{
                    "name": "Mouse-A",
                    "points": [[0, 0], [20, 0], [20, 20], [0, 20]],
                    "subject_id": "sub-a",
                    "project_unit_id": "unit-a",
                    "animal_code": "Mouse-A",
                }],
            }
            session["draft_payload"] = {
                "per_video_configs": [copy.deepcopy(source_config), {}],
                "workspace_video_configs": [copy.deepcopy(source_config), {"stale": True}],
            }
            session["project_context"] = {
                "project_id": "project-a",
                "video_assignment_map": {"0": "assignment-a", "1": "assignment-b"},
                "unit_map": {
                    "assignment-a": [{"unit_id": "unit-a", "subject_id": "sub-a"}],
                    "assignment-b": [{"unit_id": "unit-b", "subject_id": "sub-b"}],
                },
                "subject_candidates": [
                    {"subject_id": "sub-a", "animal_code": "Mouse-A"},
                    {"subject_id": "sub-b", "animal_code": "Mouse-B"},
                ],
            }
            mousescope_app.analysis_sessions.persist(session_id)

            response = self.client.post(
                f"/api/draft_action/{session_id}",
                json={"action": "copy", "source_index": 0, "target_indices": [1]},
            )
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            updated = mousescope_app.analysis_sessions.get(session_id)
            target = updated["draft_payload"]["per_video_configs"][1]
            workspace = updated["draft_payload"]["workspace_video_configs"][1]
            self.assertEqual(target, workspace)
            self.assertEqual(target["orientation_override"], 90)
            self.assertNotIn("subject_id", target["rois"][0])
            self.assertNotIn("project_unit_id", target["rois"][0])
            self.assertEqual(updated["video_infos"][1]["orientation_override"], 90)

            response = self.client.post(
                f"/api/draft_action/{session_id}",
                json={"action": "reset", "target_index": 1},
            )
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            updated = mousescope_app.analysis_sessions.get(session_id)
            self.assertEqual(updated["draft_payload"]["per_video_configs"][1], {})
            self.assertEqual(updated["draft_payload"]["workspace_video_configs"][1], {})
            self.assertIsNone(updated["video_infos"][1]["orientation_override"])
        finally:
            mousescope_app.analysis_sessions.delete(session_id)

    def test_completed_task_can_replay_exact_resolved_config_into_new_output(self):
        class DummyAnalysis:
            def __init__(self, session_id, video_paths, video_infos, config):
                self.session_id = session_id
                self.video_paths = video_paths
                self.video_infos = video_infos
                self.config = config
                self.experiment_type = "open_field"
                self.status = "waiting"
                self.progress = 0
                self.message = "waiting"
                self.result_files = []
                self.is_running = False
                self.start_time = None
                self.end_time = None
                self.processed_frames = 0

            def start(self):
                self.status = "analyzing"
                self.is_running = True
                self.start_time = time.time()

        video_path = ROOT / "test.mp4"
        video_info = {"path": str(video_path), "original_name": video_path.name, **probe_video_info(video_path)}
        original_id = f"test_replay_original_{uuid.uuid4().hex}"
        replay_id = None
        with TemporaryDirectory() as temp_dir, patch(
            "routes.api.OpenFieldMultiArenaAnalysisSession",
            DummyAnalysis,
        ):
            now = time.time()
            mousescope_app.analysis_sessions[original_id] = {
                "experiment_type": "open_field",
                "video_paths": [str(video_path)],
                "video_infos": [video_info],
                "start_seconds": None,
                "end_seconds": None,
                "output_dir": temp_dir,
                "fps_override": None,
                "is_batch": False,
                "draft_payload": {"video": 0, "per_video_configs": [{"gray_threshold": 113}]},
                "draft_updated_at": now,
                "selected_video_idx": 0,
                "created_at": now,
                "last_accessed_at": now,
            }
            try:
                start = self.client.post(
                    f"/api/start_analysis/{original_id}",
                    json={"use_saved_draft": True},
                )
                self.assertEqual(start.status_code, 200, start.get_data(as_text=True))
                original_session = mousescope_app.analysis_sessions.get(original_id)
                original_analysis = original_session["analysis"]
                original_output = Path(original_analysis.config["output_dir"])
                snapshot_path = original_output / RUN_SNAPSHOT_FILENAME
                self.assertTrue(snapshot_path.is_file())
                snapshot = load_run_snapshot(snapshot_path)
                self.assertEqual(snapshot["resolved_config"]["per_video_configs"][0]["gray_threshold"], 113)
                self.assertIn("resource_policy", snapshot["resolved_config"])

                original_analysis.status = "completed"
                original_analysis.progress = 100
                original_analysis.is_running = False
                original_analysis.end_time = time.time()
                mousescope_app._analysis_status_observer(original_analysis)

                replay = self.client.post(f"/api/tasks/{original_id}/replay")
                self.assertEqual(replay.status_code, 200, replay.get_data(as_text=True))
                replay_payload = replay.get_json()
                replay_id = replay_payload["session_id"]
                replay_session = mousescope_app.analysis_sessions.get(replay_id)
                self.assertEqual(replay_session["replay_resolved_config"]["per_video_configs"][0]["gray_threshold"], 113)

                replay_start = self.client.post(
                    replay_payload["start_url"],
                    json={"use_saved_draft": True},
                )
                self.assertEqual(replay_start.status_code, 200, replay_start.get_data(as_text=True))
                replay_analysis = mousescope_app.analysis_sessions.get(replay_id)["analysis"]
                self.assertEqual(replay_analysis.config["per_video_configs"][0]["gray_threshold"], 113)
                self.assertNotEqual(Path(replay_analysis.config["output_dir"]), original_output)
            finally:
                mousescope_app.analysis_history.delete_record(original_id)
                mousescope_app.analysis_sessions.delete(original_id)
                if replay_id:
                    mousescope_app.analysis_history.delete_record(replay_id)
                    mousescope_app.analysis_sessions.delete(replay_id)

    def test_project_task_replay_creates_new_project_run_context(self):
        class CompletedAnalysis:
            def __init__(self, run_id, config, result_files):
                self.session_id = run_id
                self.experiment_type = "open_field"
                self.config = config
                self.result_files = result_files
                self.status = "completed"
                self.progress = 100
                self.message = "completed"
                self.is_running = False
                self.start_time = time.time() - 1
                self.end_time = time.time()
                self.processed_frames = 30

        source_id = f"test_project_replay_{uuid.uuid4().hex}"
        replay_id = None
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = ROOT / "test.mp4"
            video_info = {"path": str(video_path), "original_name": video_path.name, **probe_video_info(video_path)}
            project_id = "project-replay-test"
            project_context = {
                "project_id": project_id,
                "project_name": "Replay Test",
                "project_run_id": "prun_source",
                "assignment_ids": [],
                "video_assignment_map": {},
                "unit_map": {},
                "subject_candidates": [],
            }
            source_output = root / "source-output"
            source_session = {
                "experiment_type": "open_field",
                "video_paths": [str(video_path)],
                "video_infos": [video_info],
                "output_dir": str(source_output),
                "output_root_dir": str(root / "project-dir" / "results" / "open_field"),
                "draft_payload": {"video": 0, "per_video_configs": [{"gray_threshold": 111}]},
                "project_context": project_context,
                "created_at": time.time(),
            }
            resolved = {
                "output_dir": str(source_output),
                "per_video_configs": [{"gray_threshold": 111}],
            }
            snapshot = build_run_snapshot(
                session_id=source_id,
                app_version="3.5.0",
                sess=source_session,
                resolved_config=resolved,
            )
            snapshot_path = write_run_snapshot(source_output, snapshot)
            analysis = CompletedAnalysis(source_id, resolved, [snapshot_path])
            mousescope_app.analysis_history.upsert_runtime(source_id, source_session, analysis=analysis)
            try:
                with patch.object(
                    mousescope_app.project_store,
                    "get_project",
                    return_value={"project_id": project_id, "project_dir": str(root / "project-dir"), "name": "Replay Test"},
                ), patch.object(
                    mousescope_app.project_store,
                    "subject_candidates",
                    return_value=[],
                ), patch.object(
                    mousescope_app.project_store,
                    "upsert_project_run",
                    return_value={},
                ) as upsert_run:
                    response = self.client.post(f"/api/tasks/{source_id}/replay")

                self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
                replay_id = response.get_json()["session_id"]
                replay_session = mousescope_app.analysis_sessions.get(replay_id)
                self.assertEqual(replay_session["project_context"]["project_id"], project_id)
                self.assertNotEqual(replay_session["project_context"]["project_run_id"], "prun_source")
                self.assertEqual(
                    Path(replay_session["output_root_dir"]),
                    root / "project-dir" / "results" / "open_field",
                )
                upsert_run.assert_called_once()
            finally:
                mousescope_app.analysis_history.delete_record(source_id)
                if replay_id:
                    mousescope_app.analysis_sessions.delete(replay_id)


if __name__ == "__main__":
    unittest.main()
