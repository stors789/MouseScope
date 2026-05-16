from __future__ import annotations

import copy
import json
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


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
    status = str(getattr(analysis, "status", "") or "").lower()
    return bool(getattr(analysis, "is_running", False)) or status in {"waiting", "analyzing"}


class SessionStore:
    """Runtime + snapshot-backed session store.

    The live analysis object only exists in memory. Serializable session
    metadata and draft payloads are mirrored to disk so batch configuration
    work can be restored after an unexpected app shutdown.
    """

    SNAPSHOT_NAME = "session_state.json"

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
        self._upload_dir = Path(upload_dir)
        self._work_dir = Path(work_dir)
        self._pending_ttl = float(pending_ttl_seconds)
        self._finished_ttl = float(finished_ttl_seconds)
        self._error_ttl = float(error_ttl_seconds)
        self._now_fn = now_fn

    def _session_dir(self, session_id: str) -> Path:
        return self._work_dir / str(session_id)

    def _snapshot_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / self.SNAPSHOT_NAME

    def _serialize_session(self, sess: dict[str, Any]) -> dict[str, Any]:
        data = {}
        for key, value in sess.items():
            if key == "analysis":
                continue
            data[key] = copy.deepcopy(value)
        return data

    def _load_snapshot(self, session_id: str) -> dict[str, Any] | None:
        snapshot_path = self._snapshot_path(session_id)
        if not snapshot_path.exists():
            return None
        try:
            data = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        data["last_accessed_at"] = self._now_fn()
        return data

    def persist(self, session_id: str) -> None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return
            snapshot_path = self._snapshot_path(session_id)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._serialize_session(sess)
            snapshot_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def create(self, session_id: str, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            sess = copy.deepcopy(data)
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
            sess["last_accessed_at"] = self._now_fn()
            self.persist(session_id)

    def has_active_analysis(self, session_id: str) -> bool:
        with self._lock:
            sess = self.get(session_id)
            if not sess:
                return False
            return _analysis_is_active(sess.get("analysis"))

    def set_analysis_if_idle(self, session_id: str, analysis: Any) -> bool:
        with self._lock:
            sess = self.get(session_id)
            if not sess or _analysis_is_active(sess.get("analysis")):
                return False
            sess["analysis"] = analysis
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
                is_running = bool(getattr(analysis, "is_running", False))
                if is_running or status in {"waiting", "analyzing"}:
                    continue
                ttl = self._finished_ttl if status in {"completed", "stopped"} else self._error_ttl

            if now - last_access > ttl:
                expired_session_ids.append(session_id)

        for session_id in expired_session_ids:
            self.delete(session_id)

        return expired_session_ids
