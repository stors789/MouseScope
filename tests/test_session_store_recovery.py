from __future__ import annotations

import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from services.atomic_json import atomic_write_json
from services.session_store import SessionStore


def _store(root: Path, now: float = 10_000.0) -> SessionStore:
    return SessionStore(
        upload_dir=root / "uploads",
        work_dir=root / "work",
        pending_ttl_seconds=7200,
        finished_ttl_seconds=43200,
        error_ttl_seconds=86400,
        now_fn=lambda: now,
    )


class _DummyAnalysis:
    status = "analyzing"
    is_running = True


class _DummyThread:
    def __init__(self, alive: bool):
        self._alive = alive

    def is_alive(self):
        return self._alive


class SessionStoreRecoveryTest(unittest.TestCase):
    def test_pending_session_is_discovered_after_restart(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            first = _store(root)
            first.create("session-a", {
                "video_paths": [str(video)],
                "video_infos": [{"path": str(video)}],
                "created_at": 9_000.0,
                "last_accessed_at": 9_500.0,
                "draft_payload": {"rois": []},
            })

            restored = _store(root)

            self.assertIn("session-a", restored.restored_session_ids)
            self.assertIsNotNone(restored.get("session-a"))

    def test_started_session_is_not_restored_as_configurable(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            first = _store(root)
            first.create("session-b", {
                "video_paths": [str(video)],
                "created_at": 9_000.0,
                "last_accessed_at": 9_500.0,
            })
            first.set_analysis("session-b", _DummyAnalysis())

            restored = _store(root)

            self.assertNotIn("session-b", restored.restored_session_ids)
            self.assertIsNone(restored.get("session-b"))

    def test_corrupt_snapshot_is_quarantined_without_blocking_startup(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot_dir = root / "work" / "broken"
            snapshot_dir.mkdir(parents=True)
            (snapshot_dir / "session_state.json").write_text("{broken", encoding="utf-8")

            restored = _store(root)

            self.assertTrue(restored.restore_errors)
            self.assertFalse((snapshot_dir / "session_state.json").exists())
            self.assertTrue(list(snapshot_dir.glob("session_state.json.corrupt.*")))

    def test_atomic_replace_failure_keeps_previous_json(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "settings.json"
            atomic_write_json(path, {"value": 1})
            with patch("services.atomic_json.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    atomic_write_json(path, {"value": 2})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 1})
            self.assertFalse(list(path.parent.glob(".settings.json.*.tmp")))

    def test_expired_pending_snapshot_is_removed(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            first = _store(root, now=1_000.0)
            first.create("expired", {
                "video_paths": [str(video)],
                "created_at": 100.0,
                "last_accessed_at": 100.0,
            })

            restored = _store(root, now=10_000.0)

            self.assertNotIn("expired", restored.restored_session_ids)
            self.assertFalse((root / "work" / "expired").exists())

    def test_start_reservation_is_atomic_and_commits_only_for_owner(self):
        with TemporaryDirectory() as temp_dir:
            store = _store(Path(temp_dir))
            store.create("reserved", {"video_paths": [], "created_at": 1.0})

            self.assertTrue(store.reserve_start("reserved"))
            self.assertFalse(store.reserve_start("reserved"))
            self.assertTrue(store.has_active_analysis("reserved"))
            analysis = _DummyAnalysis()
            self.assertTrue(store.commit_reserved_analysis("reserved", analysis))
            self.assertIs(store.get("reserved")["analysis"], analysis)
            self.assertFalse(store.commit_reserved_analysis("reserved", _DummyAnalysis()))

    def test_worker_thread_remains_active_after_stop_flag_changes(self):
        with TemporaryDirectory() as temp_dir:
            store = _store(Path(temp_dir))
            analysis = _DummyAnalysis()
            analysis.status = "stopped"
            analysis.is_running = False
            analysis._thread = _DummyThread(True)
            store.create("stopping", {"video_paths": [], "created_at": 1.0})
            store.set_analysis("stopping", analysis)

            self.assertTrue(store.has_active_analysis("stopping"))
            self.assertFalse(store.reserve_start("stopping"))
            analysis._thread._alive = False
            self.assertFalse(store.has_active_analysis("stopping"))


if __name__ == "__main__":
    unittest.main()
