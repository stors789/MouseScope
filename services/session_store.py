from __future__ import annotations

import copy
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from services.atomic_json import atomic_write_json, quarantine_corrupt_json, read_json


def _safe_timestamp(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            text = value.strip()
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            try:
                return datetime.fromisoformat(text).timestamp()
            except ValueError:
                return default
        return default


def _analysis_is_active(analysis: Any) -> bool:
    if analysis is None:
        return False
    thread = getattr(analysis, "_thread", None)
    try:
        if thread is not None and thread.is_alive():
            return True
    except (AttributeError, RuntimeError):
        pass
    status = str(getattr(analysis, "status", "") or "").lower()
    return bool(getattr(analysis, "is_running", False)) or status in {"waiting", "analyzing"}


class SessionStore:
    """Runtime + snapshot-backed session store.

    The live analysis object only exists in memory. Serializable session
    metadata and draft payloads are mirrored to disk so batch configuration
    work can be restored after an unexpected app shutdown.
    """

    SNAPSHOT_NAME = "session_state.json"
    SNAPSHOT_VERSION = 2

    def __init__(
        self,
        upload_dir: str | Path,
        work_dir: str | Path,
        pending_ttl_seconds: float,
        finished_ttl_seconds: float,
        error_ttl_seconds: float,
        now_fn,
    ) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._starting_sessions: set[str] = set()
        self._upload_dir = Path(upload_dir)
        self._work_dir = Path(work_dir)
        self._pending_ttl = float(pending_ttl_seconds)
        self._finished_ttl = float(finished_ttl_seconds)
        self._error_ttl = float(error_ttl_seconds)
        self._now_fn = now_fn
        self.restored_session_ids: list[str] = []
        self.restore_errors: list[str] = []
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._restore_pending_snapshots()

    def _session_dir(self, session_id: str) -> Path:
        return self._work_dir / str(session_id)

    def _snapshot_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / self.SNAPSHOT_NAME

    def _serialize_session(self, sess: dict[str, Any]) -> dict[str, Any]:
        data = {}
        analysis = sess.get("analysis")
        for key, value in sess.items():
            if key == "analysis":
                continue
            data[key] = copy.deepcopy(value)
        analysis_started = bool(sess.get("analysis_started")) or analysis is not None
        data["snapshot_version"] = self.SNAPSHOT_VERSION
        data["analysis_started"] = analysis_started
        data["analysis_status"] = str(getattr(analysis, "status", "") or data.get("analysis_status") or "")
        data["restorable"] = not analysis_started
        return data

    def _load_snapshot(self, session_id: str, *, quarantine_invalid: bool = False) -> dict[str, Any] | None:
        snapshot_path = self._snapshot_path(session_id)
        if not snapshot_path.exists():
            return None
        try:
            data = read_json(snapshot_path)
        except Exception as exc:
            if quarantine_invalid:
                quarantined = quarantine_corrupt_json(snapshot_path)
                self.restore_errors.append(f"{session_id}: invalid snapshot ({exc}); quarantined={quarantined}")
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _snapshot_ttl(self, data: dict[str, Any]) -> float:
        if not data.get("analysis_started"):
            return self._pending_ttl
        status = str(data.get("analysis_status") or "").lower()
        return self._finished_ttl if status in {"completed", "stopped"} else self._error_ttl

    @staticmethod
    def _video_paths_available(data: dict[str, Any]) -> bool:
        paths = [Path(str(path)) for path in data.get("video_paths") or [] if str(path or "").strip()]
        return bool(paths) and all(path.is_file() for path in paths)

    def _restore_pending_snapshots(self) -> None:
        now = self._now_fn()
        for session_dir in self._work_dir.iterdir():
            if not session_dir.is_dir():
                continue
            session_id = session_dir.name
            data = self._load_snapshot(session_id, quarantine_invalid=True)
            if not data:
                continue
            last_access = _safe_timestamp(data.get("last_accessed_at") or data.get("created_at"), now)
            if now - last_access > self._snapshot_ttl(data):
                self.cleanup_artifacts(session_id)
                continue
            if int(data.get("snapshot_version") or 0) < self.SNAPSHOT_VERSION:
                continue
            if data.get("analysis_started") or not data.get("restorable"):
                continue
            if not self._video_paths_available(data):
                continue
            data["restored_at"] = now
            self._sessions[session_id] = data
            self.restored_session_ids.append(session_id)

    def persist(self, session_id: str) -> None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return
            snapshot_path = self._snapshot_path(session_id)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._serialize_session(sess)
            atomic_write_json(snapshot_path, payload)

    def create(self, session_id: str, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            sess = copy.deepcopy(data)
            self._starting_sessions.discard(session_id)
            self._sessions[session_id] = sess
            self.persist(session_id)
            return sess

    def __setitem__(self, session_id: str, data: dict[str, Any]) -> None:
        self.create(session_id, data)

    def __getitem__(self, session_id: str) -> dict[str, Any]:
        sess = self.get(session_id)
        if sess is None:
            raise KeyError(session_id)
        return sess

    def __contains__(self, session_id: str) -> bool:
        return self.get(session_id) is not None

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = self._load_snapshot(session_id)
                if (
                    isinstance(sess, dict)
                    and int(sess.get("snapshot_version") or 0) >= self.SNAPSHOT_VERSION
                    and (sess.get("analysis_started") or not sess.get("restorable"))
                ):
                    sess = None
                if sess is not None:
                    self._sessions[session_id] = sess
            return sess

    def pop(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def items(self):
        with self._lock:
            return list(self._sessions.items())

    def mark_accessed(self, session_id: str) -> None:
        with self._lock:
            sess = self.get(session_id)
            if not sess:
                return
            sess["last_accessed_at"] = self._now_fn()
            self.persist(session_id)

    def set_analysis(self, session_id: str, analysis: Any) -> None:
        with self._lock:
            sess = self.get(session_id)
            if not sess:
                return
            sess["analysis"] = analysis
            sess["analysis_started"] = True
            sess["last_accessed_at"] = self._now_fn()
            self.persist(session_id)

    def has_active_analysis(self, session_id: str) -> bool:
        with self._lock:
            sess = self.get(session_id)
            if not sess:
                return False
            return session_id in self._starting_sessions or _analysis_is_active(sess.get("analysis"))

    def reserve_start(self, session_id: str) -> bool:
        """Atomically reserve all start-up side effects for one request."""
        with self._lock:
            sess = self.get(session_id)
            if (
                not sess
                or session_id in self._starting_sessions
                or _analysis_is_active(sess.get("analysis"))
            ):
                return False
            self._starting_sessions.add(session_id)
            return True

    def release_start(self, session_id: str) -> None:
        with self._lock:
            self._starting_sessions.discard(session_id)

    def commit_reserved_analysis(self, session_id: str, analysis: Any) -> bool:
        """Attach an analysis only to the request holding the start reservation."""
        with self._lock:
            sess = self.get(session_id)
            if not sess or session_id not in self._starting_sessions:
                return False
            if _analysis_is_active(sess.get("analysis")):
                self._starting_sessions.discard(session_id)
                return False
            sess["analysis"] = analysis
            sess["analysis_started"] = True
            sess["last_accessed_at"] = self._now_fn()
            self._starting_sessions.discard(session_id)
            self.persist(session_id)
            return True

    def set_analysis_if_idle(self, session_id: str, analysis: Any) -> bool:
        with self._lock:
            sess = self.get(session_id)
            if not sess or session_id in self._starting_sessions or _analysis_is_active(sess.get("analysis")):
                return False
            sess["analysis"] = analysis
            sess["analysis_started"] = True
            sess["last_accessed_at"] = self._now_fn()
            self.persist(session_id)
            return True

    def save_draft(self, session_id: str, draft_payload: dict[str, Any], selected_video_idx: int | None = None) -> dict[str, Any] | None:
        with self._lock:
            sess = self.get(session_id)
            if not sess:
                return None
            sess["draft_payload"] = copy.deepcopy(draft_payload)
            sess["draft_updated_at"] = self._now_fn()
            if selected_video_idx is not None:
                sess["selected_video_idx"] = int(selected_video_idx)
            sess["last_accessed_at"] = self._now_fn()
            self.persist(session_id)
            return sess

    def cleanup_artifacts(self, session_id: str) -> None:
        for target in (
            self._upload_dir / "_staged" / str(session_id),
            self._session_dir(session_id),
        ):
            try:
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
            except Exception:
                pass

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._starting_sessions.discard(session_id)
            self._sessions.pop(session_id, None)
        self.cleanup_artifacts(session_id)

    def cleanup_expired(self) -> list[str]:
        expired_session_ids: list[str] = []
        now = self._now_fn()

        for session_id, sess in self.items():
            last_access = _safe_timestamp(sess.get("last_accessed_at") or sess.get("created_at"), now)
            analysis = sess.get("analysis")
            if analysis is None:
                ttl = self._pending_ttl
            else:
                status = str(getattr(analysis, "status", "") or "").lower()
                if _analysis_is_active(analysis):
                    continue
                ttl = self._finished_ttl if status in {"completed", "stopped"} else self._error_ttl

            if now - last_access > ttl:
                expired_session_ids.append(session_id)

        for session_id in expired_session_ids:
            self.delete(session_id)

        return expired_session_ids
