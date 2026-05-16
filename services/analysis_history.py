from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from services.experiment_registry import get_experiment_spec


RUNNING_STATUSES = {"waiting", "analyzing"}
TERMINAL_STATUSES = {"completed", "error", "stopped", "interrupted"}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _video_names(video_infos: list[dict[str, Any]] | None, video_paths: list[str] | None = None) -> list[str]:
    names: list[str] = []
    for info in video_infos or []:
        if isinstance(info, dict):
            name = info.get("original_name") or info.get("name") or info.get("path")
            if name:
                names.append(str(name))
    if not names:
        for path in video_paths or []:
            names.append(Path(str(path)).name)
    return names


def _result_files_from_analysis(analysis: Any) -> list[str]:
    files = getattr(analysis, "result_files", None)
    if isinstance(files, list):
        return [str(path) for path in files if path]
    return []


def _status_payload_from_analysis(analysis: Any) -> dict[str, Any]:
    elapsed = 0.0
    start_time = _safe_float(getattr(analysis, "start_time", None), None)
    end_time = _safe_float(getattr(analysis, "end_time", None), None)
    if start_time:
        import time

        elapsed = (end_time or time.time()) - start_time
    processed_frames = _safe_int(getattr(analysis, "processed_frames", 0))
    fps = processed_frames / elapsed if elapsed > 0 else 0
    return {
        "session_id": getattr(analysis, "session_id", None),
        "experiment_type": getattr(analysis, "experiment_type", None),
        "status": getattr(analysis, "status", "waiting"),
        "progress": _safe_int(getattr(analysis, "progress", 0)),
        "message": str(getattr(analysis, "message", "") or ""),
        "result_files": _result_files_from_analysis(analysis),
        "stats": {
            "processed_frames": processed_frames,
            "elapsed_time": elapsed,
            "fps": fps,
        },
    }


def _derive_output_dir_from_files(result_files: list[str]) -> str:
    for path in result_files or []:
        if path and os.path.exists(path):
            return str(Path(path).parent)
    return ""


class AnalysisHistoryStore:
    """SQLite-backed index of analysis runs.

    Runtime analysis objects are still owned by SessionStore. This store keeps
    durable metadata, so the task center can show completed or interrupted work
    after MouseScope is restarted.
    """

    def __init__(self, db_path: str | Path, now_fn, default_result_dir: str | Path | None = None) -> None:
        self.db_path = Path(db_path)
        self._now_fn = now_fn
        self._default_result_dir = str(default_result_dir) if default_result_dir else ""
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    experiment_type TEXT NOT NULL,
                    experiment_label TEXT NOT NULL,
                    experiment_short_label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT,
                    video_names TEXT NOT NULL DEFAULT '[]',
                    video_paths TEXT NOT NULL DEFAULT '[]',
                    output_dir TEXT,
                    result_files TEXT NOT NULL DEFAULT '[]',
                    config_summary TEXT NOT NULL DEFAULT '{}',
                    project_id TEXT,
                    project_name TEXT,
                    project_run_id TEXT,
                    is_batch INTEGER NOT NULL DEFAULT 0,
                    created_at REAL,
                    started_at REAL,
                    finished_at REAL,
                    updated_at REAL NOT NULL,
                    duration_seconds REAL,
                    error_message TEXT
                )
                """
            )
            existing_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(analysis_runs)").fetchall()
            }
            for col_name in ("project_id", "project_name", "project_run_id"):
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE analysis_runs ADD COLUMN {col_name} TEXT")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_updated ON analysis_runs(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_status ON analysis_runs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_experiment ON analysis_runs(experiment_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_project ON analysis_runs(project_id)")

    def mark_interrupted_running(self) -> int:
        now = self._now_fn()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE analysis_runs
                   SET status = 'interrupted',
                       progress = CASE WHEN progress >= 100 THEN 99 ELSE progress END,
                       message = CASE
                           WHEN message IS NULL OR message = '' THEN ?
                           ELSE message
                       END,
                       finished_at = COALESCE(finished_at, ?),
                       updated_at = ?,
                       error_message = CASE
                           WHEN error_message IS NULL OR error_message = '' THEN ?
                           ELSE error_message
                       END
                 WHERE status IN ('waiting', 'analyzing')
                """,
                (
                    "软件关闭时分析未完成。",
                    now,
                    now,
                    "软件关闭时分析未完成。",
                ),
            )
            return int(cur.rowcount or 0)

    def _record_from_runtime(
        self,
        session_id: str,
        sess: dict[str, Any] | None,
        analysis: Any | None = None,
        status_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sess = sess or {}
        status_payload = status_payload or (_status_payload_from_analysis(analysis) if analysis is not None else {})
        experiment_type = str(
            status_payload.get("experiment_type")
            or getattr(analysis, "experiment_type", None)
            or sess.get("experiment_type")
            or "tst"
        )
        spec = get_experiment_spec(experiment_type)
        video_infos = list(sess.get("video_infos") or getattr(analysis, "video_infos", []) or [])
        video_paths = list(sess.get("video_paths") or getattr(analysis, "video_paths", []) or [])
        config = dict(getattr(analysis, "config", {}) or {})
        output_dir = sess.get("output_dir") or config.get("output_dir") or self._default_result_dir
        status = str(status_payload.get("status") or getattr(analysis, "status", None) or "configured")
        progress = _safe_int(status_payload.get("progress", getattr(analysis, "progress", 0) if analysis is not None else 0))
        if analysis is not None:
            message = str(status_payload.get("message") or getattr(analysis, "message", "") or "")
            result_files = list(status_payload.get("result_files") or _result_files_from_analysis(analysis))
        else:
            message = str(status_payload.get("message") or "")
            result_files = list(status_payload.get("result_files") or [])
        if not output_dir:
            output_dir = _derive_output_dir_from_files(result_files)
        start_time = _safe_float(getattr(analysis, "start_time", None), None) if analysis is not None else None
        end_time = _safe_float(getattr(analysis, "end_time", None), None) if analysis is not None else None
        stats = status_payload.get("stats") if isinstance(status_payload.get("stats"), dict) else {}
        duration = _safe_float(stats.get("elapsed_time"), None)
        if duration is None and start_time:
            duration = (end_time or self._now_fn()) - start_time
        error_message = message if status == "error" else None
        now = self._now_fn()
        created_at = _safe_float(sess.get("created_at"), None) or start_time or now
        project_context = sess.get("project_context") if isinstance(sess.get("project_context"), dict) else {}

        return {
            "run_id": str(session_id),
            "session_id": str(session_id),
            "experiment_type": spec.key,
            "experiment_label": spec.label,
            "experiment_short_label": spec.short_label,
            "status": status,
            "progress": max(0, min(100, progress)),
            "message": message,
            "video_names": _video_names(video_infos, video_paths),
            "video_paths": [str(path) for path in video_paths if path],
            "output_dir": str(output_dir) if output_dir else "",
            "result_files": [str(path) for path in result_files if path],
            "config_summary": {
                "video_count": len(video_infos) or len(video_paths),
                "generate_video": bool(config.get("generate_video", True)),
                "start_sec": config.get("start_sec"),
                "end_sec": config.get("end_sec"),
                "fps_override": config.get("fps_override"),
            },
            "project_id": str(project_context.get("project_id") or ""),
            "project_name": str(project_context.get("project_name") or ""),
            "project_run_id": str(project_context.get("project_run_id") or ""),
            "is_batch": 1 if bool(sess.get("is_batch") or len(video_infos) > 1 or len(video_paths) > 1) else 0,
            "created_at": created_at,
            "started_at": start_time,
            "finished_at": end_time if status in TERMINAL_STATUSES else None,
            "updated_at": now,
            "duration_seconds": duration,
            "error_message": error_message,
        }

    def upsert_runtime(
        self,
        session_id: str,
        sess: dict[str, Any] | None,
        analysis: Any | None = None,
        status_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = self._record_from_runtime(session_id, sess, analysis=analysis, status_payload=status_payload)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_runs (
                    run_id, session_id, experiment_type, experiment_label, experiment_short_label,
                    status, progress, message, video_names, video_paths, output_dir, result_files,
                    config_summary, project_id, project_name, project_run_id,
                    is_batch, created_at, started_at, finished_at, updated_at,
                    duration_seconds, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    experiment_type = excluded.experiment_type,
                    experiment_label = excluded.experiment_label,
                    experiment_short_label = excluded.experiment_short_label,
                    status = excluded.status,
                    progress = excluded.progress,
                    message = excluded.message,
                    video_names = excluded.video_names,
                    video_paths = excluded.video_paths,
                    output_dir = excluded.output_dir,
                    result_files = excluded.result_files,
                    config_summary = excluded.config_summary,
                    project_id = COALESCE(NULLIF(excluded.project_id, ''), analysis_runs.project_id),
                    project_name = COALESCE(NULLIF(excluded.project_name, ''), analysis_runs.project_name),
                    project_run_id = COALESCE(NULLIF(excluded.project_run_id, ''), analysis_runs.project_run_id),
                    is_batch = excluded.is_batch,
                    created_at = COALESCE(analysis_runs.created_at, excluded.created_at),
                    started_at = COALESCE(analysis_runs.started_at, excluded.started_at),
                    finished_at = excluded.finished_at,
                    updated_at = excluded.updated_at,
                    duration_seconds = excluded.duration_seconds,
                    error_message = excluded.error_message
                """,
                (
                    record["run_id"],
                    record["session_id"],
                    record["experiment_type"],
                    record["experiment_label"],
                    record["experiment_short_label"],
                    record["status"],
                    record["progress"],
                    record["message"],
                    _json_dumps(record["video_names"]),
                    _json_dumps(record["video_paths"]),
                    record["output_dir"],
                    _json_dumps(record["result_files"]),
                    _json_dumps(record["config_summary"]),
                    record["project_id"],
                    record["project_name"],
                    record["project_run_id"],
                    record["is_batch"],
                    record["created_at"],
                    record["started_at"],
                    record["finished_at"],
                    record["updated_at"],
                    record["duration_seconds"],
                    record["error_message"],
                ),
            )
        return record

    def get_record(self, run_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM analysis_runs WHERE run_id = ?", (str(run_id),)).fetchone()
        return self._row_to_task(row) if row else None

    def delete_record(self, run_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM analysis_runs WHERE run_id = ?", (str(run_id),))
            return bool(cur.rowcount)

    def list_records(self, limit: int = 500) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 500), 2000))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analysis_runs ORDER BY updated_at DESC, created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def _row_to_task(self, row: sqlite3.Row) -> dict[str, Any]:
        video_names = _json_loads(row["video_names"], [])
        result_files = _json_loads(row["result_files"], [])
        output_dir = row["output_dir"] or ""
        existing_result_files = [path for path in result_files if path and os.path.exists(path)]
        if not output_dir:
            output_dir = _derive_output_dir_from_files(existing_result_files)
        output_exists = bool(output_dir and os.path.isdir(output_dir))
        has_result_files = bool(existing_result_files)
        return {
            "run_id": row["run_id"],
            "session_id": row["session_id"],
            "experiment_type": row["experiment_type"],
            "experiment_label": row["experiment_label"],
            "experiment_short_label": row["experiment_short_label"],
            "status": row["status"],
            "progress": _safe_int(row["progress"]),
            "message": row["message"] or "",
            "video_names": video_names,
            "video_paths": _json_loads(row["video_paths"], []),
            "primary_video_name": video_names[0] if video_names else row["session_id"],
            "video_count": len(video_names),
            "output_dir": output_dir,
            "result_files": result_files,
            "result_file_count": len(existing_result_files),
            "has_results": has_result_files,
            "has_output_dir": output_exists,
            "config_summary": _json_loads(row["config_summary"], {}),
            "project_id": row["project_id"] or "",
            "project_name": row["project_name"] or "",
            "project_run_id": row["project_run_id"] or "",
            "task_scope": "project" if (row["project_id"] or row["project_run_id"]) else "quick",
            "is_batch": bool(row["is_batch"]),
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "updated_at": row["updated_at"],
            "duration_seconds": row["duration_seconds"],
            "error_message": row["error_message"] or "",
            "source": "history",
            "is_live": False,
            "can_monitor": False,
            "can_stop": False,
            "can_download": has_result_files,
            "can_open_folder": output_exists,
            "can_delete": row["status"] not in RUNNING_STATUSES,
        }

    def build_runtime_task(self, session_id: str, sess: dict[str, Any]) -> dict[str, Any]:
        analysis = sess.get("analysis")
        status_payload = _status_payload_from_analysis(analysis) if analysis is not None else {}
        record = self._record_from_runtime(session_id, sess, analysis=analysis, status_payload=status_payload)
        result_files = record["result_files"]
        output_dir = record["output_dir"] or _derive_output_dir_from_files(record["result_files"])
        existing_result_files = [path for path in result_files if path and os.path.exists(path)]
        output_exists = bool(output_dir and os.path.isdir(output_dir))
        has_analysis = analysis is not None
        status = record["status"] if has_analysis else ("configured" if sess.get("draft_payload") else "uploaded")
        video_names = record["video_names"]
        has_result_files = bool(existing_result_files)
        has_openable_output = bool(output_exists and (has_analysis or has_result_files))
        return {
            **record,
            "status": status,
            "primary_video_name": video_names[0] if video_names else str(session_id),
            "video_count": len(video_names),
            "result_file_count": len(existing_result_files),
            "has_results": has_result_files,
            "has_output_dir": has_openable_output,
            "task_scope": "project" if (record.get("project_id") or record.get("project_run_id")) else "quick",
            "source": "runtime",
            "is_live": True,
            "can_monitor": has_analysis,
            "can_stop": has_analysis and status in RUNNING_STATUSES,
            "can_download": has_result_files,
            "can_open_folder": has_openable_output,
            "can_configure": not has_analysis,
            "can_delete": status not in RUNNING_STATUSES,
        }
