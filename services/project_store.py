from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


PROJECT_SCHEMA_VERSION = "3.3.0"
GROUP_COLOR_PALETTE = (
    "#c96442",
    "#2f8f65",
    "#4f6fb8",
    "#b07a2f",
    "#8a5fb7",
    "#3f8f9a",
    "#c84f6d",
    "#6f7d3b",
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def _safe_slug(value: str, fallback: str = "Project") -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]+", "_", text).strip("._-")
    return (text or fallback)[:64]


def _subject_code_prefix(value: str, fallback: str = "M") -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", text).strip("_-")
    return (text or fallback)[:48]


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _path_is_within(path: str | Path, parent: str | Path) -> bool:
    try:
        path_abs = _path_key(path)
        parent_abs = _path_key(parent)
        return os.path.commonpath([path_abs, parent_abs]) == parent_abs
    except Exception:
        return False


def _cleanup_empty_parent_dirs(start_dir: str | Path, root: str | Path) -> None:
    current = Path(start_dir)
    root = Path(root)
    while _path_is_within(current, root) and _path_key(current) != _path_key(root):
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _merge_note_marker(notes: str | None, marker: str) -> str:
    text = str(notes or "").strip()
    if marker in text:
        return text
    return f"{text}; {marker}".strip("; ")


def _project_id() -> str:
    return f"PRJ_{time.strftime('%Y%m%d')}_{secrets.token_hex(3)}"


def _entity_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


class ProjectStore:
    """Local SQLite project workspace store.

    The store deliberately stays below the analysis runtime. It owns project
    metadata, subjects, videos, assignments, and project run indexes, while the
    existing SessionStore and AnalysisHistoryStore remain authoritative for
    live analysis execution and the global task center.
    """

    def __init__(self, projects_root: str | Path, now_fn=None) -> None:
        self.projects_root = Path(projects_root)
        self.index_db_path = self.projects_root / "project_index.sqlite"
        self._now_fn = now_fn or time.time
        self._lock = threading.RLock()
        self.projects_root.mkdir(parents=True, exist_ok=True)
        self._initialize_index()

    def _now(self) -> float:
        return float(self._now_fn())

    def _connect_index(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.index_db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _connect_project(self, project: dict[str, Any] | str) -> sqlite3.Connection:
        if isinstance(project, dict):
            project_dir = project.get("project_dir")
        else:
            item = self.get_project(project, include_summary=False)
            project_dir = item.get("project_dir") if item else None
        if not project_dir:
            raise KeyError("Project does not exist")
        db_path = Path(project_dir) / "project.sqlite"
        conn = sqlite3.connect(str(db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_index(self) -> None:
        with self._lock, self._connect_index() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_index (
                    project_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    project_dir TEXT NOT NULL,
                    description TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_opened_at REAL,
                    archived INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_project_index_updated ON project_index(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_project_index_archived ON project_index(archived)")

    def _initialize_project_db(self, project_dir: str | Path, meta: dict[str, Any]) -> None:
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        for child in ("videos", "results", "exports", "thumbnails"):
            (project_dir / child).mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(project_dir / "project.sqlite"), timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_groups (
                    group_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    color TEXT,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    notes TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_subjects (
                    subject_id TEXT PRIMARY KEY,
                    animal_code TEXT NOT NULL UNIQUE,
                    group_id TEXT,
                    sex TEXT,
                    strain TEXT,
                    genotype TEXT,
                    cage_id TEXT,
                    birth_date TEXT,
                    age_label TEXT,
                    weight_g REAL,
                    notes TEXT,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(group_id) REFERENCES project_groups(group_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_videos (
                    video_id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    project_path TEXT,
                    storage_mode TEXT NOT NULL DEFAULT 'linked',
                    path_exists INTEGER NOT NULL DEFAULT 1,
                    sha1 TEXT,
                    fps REAL,
                    width INTEGER,
                    height INTEGER,
                    total_frames INTEGER,
                    duration_s REAL,
                    imported_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_assignments (
                    assignment_id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL,
                    experiment_type TEXT NOT NULL,
                    session_label TEXT,
                    day_label TEXT,
                    trial_no INTEGER,
                    planned_at REAL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    latest_run_id TEXT,
                    notes TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(video_id) REFERENCES project_videos(video_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assignment_units (
                    unit_id TEXT PRIMARY KEY,
                    assignment_id TEXT NOT NULL,
                    subject_id TEXT,
                    group_id TEXT,
                    unit_label TEXT,
                    roi_index INTEGER,
                    roi_name TEXT,
                    role TEXT,
                    include_in_export INTEGER NOT NULL DEFAULT 1,
                    manual_qc_status TEXT,
                    notes TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(assignment_id) REFERENCES experiment_assignments(assignment_id),
                    FOREIGN KEY(subject_id) REFERENCES project_subjects(subject_id),
                    FOREIGN KEY(group_id) REFERENCES project_groups(group_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS project_runs (
                    project_run_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    experiment_type TEXT NOT NULL,
                    assignment_ids TEXT NOT NULL DEFAULT '[]',
                    output_dir TEXT,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    finished_at REAL,
                    result_files TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS result_metrics (
                    metric_id TEXT PRIMARY KEY,
                    project_run_id TEXT NOT NULL,
                    assignment_id TEXT,
                    unit_id TEXT,
                    subject_id TEXT,
                    group_id TEXT,
                    experiment_type TEXT NOT NULL,
                    source_file TEXT,
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    qc_json TEXT NOT NULL DEFAULT '{}',
                    include_in_export INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(project_run_id) REFERENCES project_runs(project_run_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_subjects_group ON project_subjects(group_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_path ON project_videos(source_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assignments_video ON experiment_assignments(video_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assignments_exp ON experiment_assignments(experiment_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_units_assignment ON assignment_units(assignment_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_updated ON project_runs(updated_at)")

            for key, value in meta.items():
                conn.execute(
                    "INSERT OR REPLACE INTO project_meta(key, value) VALUES(?, ?)",
                    (str(key), str(value) if value is not None else ""),
                )
            conn.execute(
                "INSERT OR REPLACE INTO project_meta(key, value) VALUES('schema_version', ?)",
                (PROJECT_SCHEMA_VERSION,),
            )

    def _project_summary(self, project: dict[str, Any]) -> dict[str, Any]:
        project_dir = Path(project.get("project_dir") or "")
        db_path = project_dir / "project.sqlite"
        summary = {
            "group_count": 0,
            "subject_count": 0,
            "video_count": 0,
            "assignment_count": 0,
            "completed_assignment_count": 0,
            "run_count": 0,
            "project_dir_exists": project_dir.exists(),
        }
        if not db_path.exists():
            return summary
        try:
            with sqlite3.connect(str(db_path), timeout=10) as conn:
                conn.row_factory = sqlite3.Row
                summary["group_count"] = int(conn.execute("SELECT COUNT(*) FROM project_groups").fetchone()[0])
                summary["subject_count"] = int(conn.execute("SELECT COUNT(*) FROM project_subjects WHERE active = 1").fetchone()[0])
                summary["video_count"] = int(conn.execute("SELECT COUNT(*) FROM project_videos").fetchone()[0])
                summary["assignment_count"] = int(conn.execute("SELECT COUNT(*) FROM experiment_assignments").fetchone()[0])
                summary["completed_assignment_count"] = int(
                    conn.execute("SELECT COUNT(*) FROM experiment_assignments WHERE status = 'completed'").fetchone()[0]
                )
                summary["run_count"] = int(conn.execute("SELECT COUNT(*) FROM project_runs").fetchone()[0])
        except Exception:
            summary["project_db_error"] = True
        return summary

    def list_projects(self, include_archived: bool = False) -> list[dict[str, Any]]:
        with self._lock, self._connect_index() as conn:
            if include_archived:
                rows = conn.execute("SELECT * FROM project_index ORDER BY COALESCE(last_opened_at, updated_at) DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM project_index WHERE archived = 0 ORDER BY COALESCE(last_opened_at, updated_at) DESC"
                ).fetchall()
        projects = []
        for row in rows:
            item = _row_to_dict(row) or {}
            item.update(self._project_summary(item))
            projects.append(item)
        return projects

    def get_project(self, project_id: str, include_summary: bool = True) -> dict[str, Any] | None:
        with self._lock, self._connect_index() as conn:
            row = conn.execute("SELECT * FROM project_index WHERE project_id = ?", (str(project_id),)).fetchone()
        item = _row_to_dict(row)
        if item and include_summary:
            item.update(self._project_summary(item))
        return item

    def create_project(self, name: str, description: str = "", project_dir: str | None = None) -> dict[str, Any]:
        name = str(name or "").strip()
        if not name:
            raise ValueError("Project name is required")
        now = self._now()
        project_id = _project_id()
        if project_dir:
            target_dir = Path(project_dir).expanduser()
        else:
            target_dir = self.projects_root / f"{project_id}_{_safe_slug(name)}"
        target_dir = target_dir.resolve()
        meta = {
            "project_id": project_id,
            "name": name,
            "description": description or "",
            "default_output_dir": str(target_dir / "results"),
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._initialize_project_db(target_dir, meta)
            with self._connect_index() as conn:
                conn.execute(
                    """
                    INSERT INTO project_index(
                        project_id, name, project_dir, description,
                        created_at, updated_at, last_opened_at, archived
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (project_id, name, str(target_dir), description or "", now, now, now),
                )
        return self.get_project(project_id) or {}

    def update_project(self, project_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return None
        allowed = {key: fields.get(key) for key in ("name", "description") if key in fields}
        now = self._now()
        with self._lock:
            if allowed:
                assignments = ", ".join(f"{key} = ?" for key in allowed)
                values = [str(value or "").strip() for value in allowed.values()]
                values.extend([now, project_id])
                with self._connect_index() as conn:
                    conn.execute(
                        f"UPDATE project_index SET {assignments}, updated_at = ? WHERE project_id = ?",
                        values,
                    )
                with self._connect_project(project) as conn:
                    for key, value in allowed.items():
                        conn.execute(
                            "INSERT OR REPLACE INTO project_meta(key, value) VALUES(?, ?)",
                            (key, str(value or "").strip()),
                        )
                    conn.execute("INSERT OR REPLACE INTO project_meta(key, value) VALUES('updated_at', ?)", (str(now),))
        return self.get_project(project_id)

    def touch_project(self, project_id: str) -> None:
        now = self._now()
        with self._lock, self._connect_index() as conn:
            conn.execute("UPDATE project_index SET last_opened_at = ?, updated_at = ? WHERE project_id = ?", (now, now, project_id))

    def archive_project(self, project_id: str, archived: bool = True) -> bool:
        with self._lock, self._connect_index() as conn:
            cur = conn.execute(
                "UPDATE project_index SET archived = ?, updated_at = ? WHERE project_id = ?",
                (1 if archived else 0, self._now(), project_id),
            )
            return bool(cur.rowcount)

    def open_project_folder(self, project_id: str) -> str:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            raise KeyError("Project does not exist")
        project_dir = Path(project["project_dir"])
        project_dir.mkdir(parents=True, exist_ok=True)
        return str(project_dir)

    def list_groups(self, project_id: str) -> list[dict[str, Any]]:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return []
        with self._lock, self._connect_project(project) as conn:
            rows = conn.execute("SELECT * FROM project_groups ORDER BY sort_order, name").fetchall()
        return [_row_to_dict(row) or {} for row in rows]

    def add_group(self, project_id: str, name: str, color: str = "", notes: str = "") -> dict[str, Any]:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            raise KeyError("Project does not exist")
        name = str(name or "").strip()
        if not name:
            raise ValueError("Group name is required")
        now = self._now()
        group_id = _entity_id("grp")
        color = str(color or "").strip()
        with self._lock, self._connect_project(project) as conn:
            existing = conn.execute("SELECT group_id FROM project_groups WHERE name = ?", (name,)).fetchone()
            if existing:
                raise ValueError(f"组别名称“{name}”已存在，请使用不同名称。")
            next_order = int(conn.execute("SELECT COALESCE(MAX(sort_order), 0) + 1 FROM project_groups").fetchone()[0])
            if not color:
                color = GROUP_COLOR_PALETTE[(next_order - 1) % len(GROUP_COLOR_PALETTE)]
            try:
                conn.execute(
                    """
                    INSERT INTO project_groups(group_id, name, color, sort_order, notes, created_at, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (group_id, name, color, next_order, notes or "", now, now),
                )
            except sqlite3.IntegrityError as exc:
                if "project_groups.name" in str(exc):
                    raise ValueError(f"组别名称“{name}”已存在，请使用不同名称。") from exc
                raise
        return self.get_group(project_id, group_id) or {}

    def get_group(self, project_id: str, group_id: str) -> dict[str, Any] | None:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return None
        with self._lock, self._connect_project(project) as conn:
            row = conn.execute("SELECT * FROM project_groups WHERE group_id = ?", (group_id,)).fetchone()
        return _row_to_dict(row)

    def delete_group(self, project_id: str, group_id: str) -> bool:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return False
        with self._lock, self._connect_project(project) as conn:
            conn.execute("UPDATE project_subjects SET group_id = NULL, updated_at = ? WHERE group_id = ?", (self._now(), group_id))
            cur = conn.execute("DELETE FROM project_groups WHERE group_id = ?", (group_id,))
            return bool(cur.rowcount)

    def list_subjects(self, project_id: str, include_inactive: bool = False) -> list[dict[str, Any]]:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return []
        where = "" if include_inactive else "WHERE s.active = 1"
        with self._lock, self._connect_project(project) as conn:
            rows = conn.execute(
                f"""
                SELECT s.*, g.name AS group_name, g.color AS group_color
                  FROM project_subjects s
             LEFT JOIN project_groups g ON g.group_id = s.group_id
                  {where}
              ORDER BY COALESCE(g.sort_order, 9999), s.created_at, s.animal_code
                """
            ).fetchall()
        return [_row_to_dict(row) or {} for row in rows]

    def add_subject(self, project_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            raise KeyError("Project does not exist")
        animal_code = str(fields.get("animal_code") or "").strip()
        if not animal_code:
            raise ValueError("Animal code is required")
        now = self._now()
        subject_id = _entity_id("sub")
        weight_g = _safe_float(fields.get("weight_g"), None)
        subject_values = (
            str(fields.get("group_id") or "").strip() or None,
            str(fields.get("sex") or "").strip(),
            str(fields.get("strain") or "").strip(),
            str(fields.get("genotype") or "").strip(),
            str(fields.get("cage_id") or "").strip(),
            str(fields.get("birth_date") or "").strip(),
            str(fields.get("age_label") or "").strip(),
            weight_g,
            str(fields.get("notes") or "").strip(),
        )
        with self._lock, self._connect_project(project) as conn:
            existing = conn.execute(
                "SELECT subject_id, active FROM project_subjects WHERE animal_code = ?",
                (animal_code,),
            ).fetchone()
            if existing:
                if int(existing["active"] or 0) == 1:
                    raise ValueError(f"小鼠编号“{animal_code}”已存在，请使用不同编号。")
                subject_id = str(existing["subject_id"])
                conn.execute(
                    """
                    UPDATE project_subjects
                       SET group_id = ?, sex = ?, strain = ?, genotype = ?,
                           cage_id = ?, birth_date = ?, age_label = ?, weight_g = ?,
                           notes = ?, active = 1, updated_at = ?
                     WHERE subject_id = ?
                    """,
                    (*subject_values, now, subject_id),
                )
            else:
                try:
                    conn.execute(
                        """
                        INSERT INTO project_subjects(
                            subject_id, animal_code, group_id, sex, strain, genotype,
                            cage_id, birth_date, age_label, weight_g, notes,
                            active, created_at, updated_at
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                        """,
                        (subject_id, animal_code, *subject_values, now, now),
                    )
                except sqlite3.IntegrityError as exc:
                    if "project_subjects.animal_code" in str(exc):
                        raise ValueError(f"小鼠编号“{animal_code}”已存在，请使用不同编号。") from exc
                    raise
        return self.get_subject(project_id, subject_id) or {}

    def add_subjects_to_group(
        self,
        project_id: str,
        group_id: str,
        count: int,
        prefix: str = "",
    ) -> list[dict[str, Any]]:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            raise KeyError("Project does not exist")
        group_id = str(group_id or "").strip()
        if not group_id:
            raise ValueError("Group is required")
        count = _safe_int(count, 0)
        if count < 1:
            raise ValueError("Mouse count must be at least 1")
        if count > 500:
            raise ValueError("Mouse count is too large")

        now = self._now()
        created_ids: list[str] = []
        with self._lock, self._connect_project(project) as conn:
            group = conn.execute("SELECT * FROM project_groups WHERE group_id = ?", (group_id,)).fetchone()
            if not group:
                raise ValueError("组别不存在")
            code_prefix = _subject_code_prefix(prefix or group["name"], "M")
            existing_codes = {
                str(row["animal_code"])
                for row in conn.execute("SELECT animal_code FROM project_subjects").fetchall()
            }
            next_no = 1
            while len(created_ids) < count:
                animal_code = f"{code_prefix}{next_no}"
                next_no += 1
                if animal_code in existing_codes:
                    continue
                existing_codes.add(animal_code)
                subject_id = _entity_id("sub")
                row_now = now + (len(created_ids) * 0.000001)
                conn.execute(
                    """
                    INSERT INTO project_subjects(
                        subject_id, animal_code, group_id, sex, strain, genotype,
                        cage_id, birth_date, age_label, weight_g, notes,
                        active, created_at, updated_at
                    ) VALUES(?, ?, ?, '', '', '', '', '', '', NULL, '', 1, ?, ?)
                    """,
                    (subject_id, animal_code, group_id, row_now, row_now),
                )
                created_ids.append(subject_id)

        return [subject for subject_id in created_ids if (subject := self.get_subject(project_id, subject_id))]

    def get_subject(self, project_id: str, subject_id: str) -> dict[str, Any] | None:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return None
        with self._lock, self._connect_project(project) as conn:
            row = conn.execute(
                """
                SELECT s.*, g.name AS group_name, g.color AS group_color
                  FROM project_subjects s
             LEFT JOIN project_groups g ON g.group_id = s.group_id
                 WHERE s.subject_id = ?
                """,
                (subject_id,),
            ).fetchone()
        return _row_to_dict(row)

    def update_subject(self, project_id: str, subject_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return None
        allowed = [
            "animal_code", "group_id", "sex", "strain", "genotype", "cage_id",
            "birth_date", "age_label", "weight_g", "notes", "active",
        ]
        updates = {key: fields[key] for key in allowed if key in fields}
        if not updates:
            return self.get_subject(project_id, subject_id)
        if "animal_code" in updates:
            updates["animal_code"] = str(updates["animal_code"] or "").strip()
            if not updates["animal_code"]:
                raise ValueError("Animal code is required")
        if "group_id" in updates:
            updates["group_id"] = str(updates["group_id"] or "").strip() or None
        for key in ("sex", "strain", "genotype", "cage_id", "birth_date", "age_label", "notes"):
            if key in updates:
                updates[key] = str(updates[key] or "").strip()
        if "weight_g" in updates:
            updates["weight_g"] = _safe_float(updates["weight_g"], None)
        if "active" in updates:
            updates["active"] = 1 if bool(updates["active"]) else 0
        now = self._now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = [updates[key] for key in updates]
        values.extend([now, subject_id])
        with self._lock, self._connect_project(project) as conn:
            if "animal_code" in updates:
                existing = conn.execute(
                    "SELECT subject_id FROM project_subjects WHERE animal_code = ? AND subject_id <> ?",
                    (updates["animal_code"], subject_id),
                ).fetchone()
                if existing:
                    raise ValueError(f"小鼠编号“{updates['animal_code']}”已存在，请使用不同编号。")
            try:
                conn.execute(
                    f"UPDATE project_subjects SET {assignments}, updated_at = ? WHERE subject_id = ?",
                    values,
                )
            except sqlite3.IntegrityError as exc:
                if "project_subjects.animal_code" in str(exc) and "animal_code" in updates:
                    raise ValueError(f"小鼠编号“{updates['animal_code']}”已存在，请使用不同编号。") from exc
                raise
        return self.get_subject(project_id, subject_id)

    def delete_subject(self, project_id: str, subject_id: str) -> bool:
        updated = self.update_subject(project_id, subject_id, {"active": 0})
        return bool(updated)

    def list_videos(self, project_id: str) -> list[dict[str, Any]]:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return []
        with self._lock, self._connect_project(project) as conn:
            rows = conn.execute("SELECT * FROM project_videos ORDER BY imported_at DESC").fetchall()
        videos = []
        for row in rows:
            item = _row_to_dict(row) or {}
            exists = os.path.isfile(item.get("project_path") or item.get("source_path") or "")
            item["path_exists"] = 1 if exists else 0
            videos.append(item)
        return videos

    def add_video(
        self,
        project_id: str,
        source_path: str,
        original_name: str | None = None,
        video_info: dict[str, Any] | None = None,
        storage_mode: str = "linked",
        notes: str = "",
    ) -> dict[str, Any]:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            raise KeyError("Project does not exist")
        source = Path(str(source_path)).expanduser()
        if not source.is_file():
            raise FileNotFoundError(str(source))
        info = dict(video_info or {})
        now = self._now()
        video_id = _entity_id("vid")
        display_name = str(original_name or source.name)
        project_path = None
        actual_source = str(source)
        if storage_mode == "copied":
            project_dir = Path(project["project_dir"])
            safe_stem = _safe_slug(Path(display_name).stem, fallback=source.stem)
            safe_name = f"{video_id}_{safe_stem}{source.suffix.lower()}"
            target = project_dir / "videos" / safe_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            project_path = str(target)
            actual_source = str(source)
        duration_s = None
        fps = _safe_float(info.get("fps"), None)
        total_frames = _safe_int(info.get("total_frames"), 0)
        if fps and total_frames:
            duration_s = total_frames / fps
        with self._lock, self._connect_project(project) as conn:
            conn.execute(
                """
                INSERT INTO project_videos(
                    video_id, original_name, source_path, project_path, storage_mode,
                    path_exists, sha1, fps, width, height, total_frames, duration_s,
                    imported_at, updated_at, notes
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    display_name,
                    actual_source,
                    project_path,
                    storage_mode,
                    1,
                    info.get("sha1"),
                    fps,
                    _safe_int(info.get("width"), 0),
                    _safe_int(info.get("height"), 0),
                    total_frames,
                    duration_s,
                    now,
                    now,
                    notes or "",
                ),
            )
        return self.get_video(project_id, video_id) or {}

    def get_video(self, project_id: str, video_id: str) -> dict[str, Any] | None:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return None
        with self._lock, self._connect_project(project) as conn:
            row = conn.execute("SELECT * FROM project_videos WHERE video_id = ?", (video_id,)).fetchone()
        item = _row_to_dict(row)
        if item:
            item["path_exists"] = 1 if os.path.isfile(item.get("project_path") or item.get("source_path") or "") else 0
        return item

    def delete_video(self, project_id: str, video_id: str) -> dict[str, Any] | None:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return None
        video_id = str(video_id or "").strip()
        if not video_id:
            return None

        project_dir = Path(project["project_dir"])
        videos_dir = project_dir / "videos"
        staged_root = self._uploaded_staging_root()
        assignment_ids: list[str] = []
        removed_paths: list[str] = []

        with self._lock, self._connect_project(project) as conn:
            row = conn.execute("SELECT * FROM project_videos WHERE video_id = ?", (video_id,)).fetchone()
            video = _row_to_dict(row)
            if not video:
                return None

            assignment_rows = conn.execute(
                "SELECT assignment_id, status FROM experiment_assignments WHERE video_id = ?",
                (video_id,),
            ).fetchall()
            active_statuses = {"waiting", "analyzing", "running"}
            if any(str(row["status"] or "").lower() in active_statuses for row in assignment_rows):
                raise ValueError("视频正在分析中，请先停止分析后再删除。")

            assignment_ids = [str(row["assignment_id"]) for row in assignment_rows if row["assignment_id"]]
            if assignment_ids:
                placeholders = ",".join("?" for _ in assignment_ids)
                conn.execute(f"DELETE FROM assignment_units WHERE assignment_id IN ({placeholders})", assignment_ids)
            conn.execute("DELETE FROM experiment_assignments WHERE video_id = ?", (video_id,))
            conn.execute("DELETE FROM project_videos WHERE video_id = ?", (video_id,))

        seen_paths: set[str] = set()
        removable_paths = [video.get("project_path")]
        if self._is_auto_cleanable_upload_cache(video):
            removable_paths.append(video.get("source_path"))
        for raw_path in removable_paths:
            if not raw_path:
                continue
            target = Path(str(raw_path))
            key = _path_key(target)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            safe_root = videos_dir if _path_is_within(target, videos_dir) else staged_root
            if not (_path_is_within(target, videos_dir) or _path_is_within(target, staged_root)):
                continue
            try:
                if target.is_file():
                    target.unlink()
                    removed_paths.append(str(target))
                    _cleanup_empty_parent_dirs(target.parent, safe_root)
            except Exception:
                continue

        return {
            "video": video,
            "assignment_ids": assignment_ids,
            "removed_paths": removed_paths,
        }

    def _uploaded_staging_root(self) -> Path:
        return self.projects_root.parent / "uploads" / "_staged"

    def _is_auto_cleanable_upload_cache(self, video: dict[str, Any]) -> bool:
        notes = str(video.get("notes") or "")
        if "upload_cache:auto_cleanup" in notes or "upload_cache:cleaned" in notes:
            return True
        storage_mode = str(video.get("storage_mode") or "").lower()
        source_path = str(video.get("source_path") or "")
        return storage_mode in {"copied", "uploaded_cache"} and bool(source_path) and _path_is_within(
            source_path,
            self._uploaded_staging_root(),
        )

    def cleanup_completed_upload_caches(self, project_id: str, assignment_ids: list[str]) -> list[dict[str, Any]]:
        assignment_ids = [str(item or "").strip() for item in assignment_ids if str(item or "").strip()]
        if not assignment_ids:
            return []
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return []
        project_dir = Path(project["project_dir"])
        videos_dir = project_dir / "videos"
        staged_root = self._uploaded_staging_root()
        placeholders = ",".join("?" for _ in assignment_ids)
        cleaned: list[dict[str, Any]] = []
        now = self._now()
        with self._lock, self._connect_project(project) as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT v.*
                  FROM project_videos v
                  JOIN experiment_assignments a ON a.video_id = v.video_id
                 WHERE a.assignment_id IN ({placeholders})
                """,
                assignment_ids,
            ).fetchall()
            for row in rows:
                video = _row_to_dict(row) or {}
                if not self._is_auto_cleanable_upload_cache(video):
                    continue
                removed_paths: list[str] = []
                seen_paths: set[str] = set()
                for raw_path, safe_root in (
                    (video.get("project_path"), videos_dir),
                    (video.get("source_path"), staged_root),
                ):
                    if not raw_path:
                        continue
                    target = Path(str(raw_path))
                    key = _path_key(target)
                    if key in seen_paths:
                        continue
                    seen_paths.add(key)
                    if not _path_is_within(target, safe_root):
                        continue
                    try:
                        if target.is_file():
                            target.unlink()
                            removed_paths.append(str(target))
                            _cleanup_empty_parent_dirs(target.parent, safe_root)
                    except Exception:
                        continue

                path_still_exists = os.path.isfile(video.get("project_path") or "") or os.path.isfile(video.get("source_path") or "")
                if removed_paths or not path_still_exists:
                    conn.execute(
                        """
                        UPDATE project_videos
                           SET storage_mode = 'cleaned',
                               path_exists = 0,
                               notes = ?,
                               updated_at = ?
                         WHERE video_id = ?
                        """,
                        (_merge_note_marker(video.get("notes"), "upload_cache:cleaned"), now, video.get("video_id")),
                    )
                    cleaned.append({
                        "video_id": video.get("video_id"),
                        "original_name": video.get("original_name"),
                        "removed_paths": removed_paths,
                    })
        return cleaned

    def create_assignment(self, project_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            raise KeyError("Project does not exist")
        video_id = str(fields.get("video_id") or "").strip()
        experiment_type = str(fields.get("experiment_type") or "").strip()
        if not video_id or not experiment_type:
            raise ValueError("Video and experiment type are required")
        now = self._now()
        assignment_id = _entity_id("asg")
        subject_id = str(fields.get("subject_id") or "").strip() or None
        subject = self.get_subject(project_id, subject_id) if subject_id else None
        group_id = subject.get("group_id") if subject else None
        animal_code = subject.get("animal_code") if subject else None
        with self._lock, self._connect_project(project) as conn:
            video = conn.execute("SELECT video_id FROM project_videos WHERE video_id = ?", (video_id,)).fetchone()
            if not video:
                raise ValueError("Video does not exist")
            existing_assignment = conn.execute(
                "SELECT assignment_id FROM experiment_assignments WHERE video_id = ? ORDER BY created_at LIMIT 1",
                (video_id,),
            ).fetchone()
            if existing_assignment:
                assignment_id = str(existing_assignment["assignment_id"])
                conn.execute(
                    """
                    UPDATE experiment_assignments
                       SET experiment_type = ?, session_label = ?, day_label = ?,
                           trial_no = ?, planned_at = ?, status = 'pending',
                           latest_run_id = NULL, notes = ?, updated_at = ?
                     WHERE assignment_id = ?
                    """,
                    (
                        experiment_type,
                        str(fields.get("session_label") or "").strip(),
                        str(fields.get("day_label") or "").strip(),
                        _safe_int(fields.get("trial_no"), 0) or None,
                        _safe_float(fields.get("planned_at"), None),
                        str(fields.get("notes") or "").strip(),
                        now,
                        assignment_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO experiment_assignments(
                        assignment_id, video_id, experiment_type, session_label,
                        day_label, trial_no, planned_at, status, latest_run_id,
                        notes, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, ?, ?)
                    """,
                    (
                        assignment_id,
                        video_id,
                        experiment_type,
                        str(fields.get("session_label") or "").strip(),
                        str(fields.get("day_label") or "").strip(),
                        _safe_int(fields.get("trial_no"), 0) or None,
                        _safe_float(fields.get("planned_at"), None),
                        str(fields.get("notes") or "").strip(),
                        now,
                        now,
                    ),
                )
                if subject:
                    conn.execute(
                        """
                        INSERT INTO assignment_units(
                            unit_id, assignment_id, subject_id, group_id, unit_label,
                            roi_index, roi_name, role, include_in_export,
                            manual_qc_status, notes, created_at, updated_at
                        ) VALUES(?, ?, ?, ?, ?, 0, ?, NULL, 1, NULL, '', ?, ?)
                        """,
                        (
                            _entity_id("unt"),
                            assignment_id,
                            subject_id,
                            group_id,
                            animal_code,
                            animal_code,
                            now,
                            now,
                        ),
                    )
        return self.get_assignment(project_id, assignment_id) or {}

    def list_assignments(self, project_id: str, experiment_type: str | None = None) -> list[dict[str, Any]]:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return []
        where = ""
        params: list[Any] = []
        if experiment_type:
            where = "WHERE a.experiment_type = ?"
            params.append(experiment_type)
        with self._lock, self._connect_project(project) as conn:
            rows = conn.execute(
                f"""
                SELECT a.*, v.original_name, v.source_path, v.project_path, v.path_exists,
                       v.storage_mode AS video_storage_mode, v.notes AS video_notes,
                       v.fps, v.width, v.height, v.total_frames, v.duration_s,
                       u.unit_id, u.subject_id, u.group_id, u.roi_name,
                       s.animal_code, g.name AS group_name, g.color AS group_color
                  FROM experiment_assignments a
                  JOIN project_videos v ON v.video_id = a.video_id
             LEFT JOIN assignment_units u ON u.assignment_id = a.assignment_id AND COALESCE(u.roi_index, 0) = 0
             LEFT JOIN project_subjects s ON s.subject_id = u.subject_id
             LEFT JOIN project_groups g ON g.group_id = u.group_id
                  {where}
              ORDER BY a.updated_at DESC
                """,
                params,
            ).fetchall()
        assignments = []
        for row in rows:
            item = _row_to_dict(row) or {}
            item["path_exists"] = 1 if os.path.isfile(item.get("project_path") or item.get("source_path") or "") else 0
            assignments.append(item)
        return assignments

    def get_assignment(self, project_id: str, assignment_id: str) -> dict[str, Any] | None:
        matches = [item for item in self.list_assignments(project_id) if item.get("assignment_id") == assignment_id]
        return matches[0] if matches else None

    def get_assignments(self, project_id: str, assignment_ids: list[str]) -> list[dict[str, Any]]:
        wanted = {str(item) for item in assignment_ids}
        return [item for item in self.list_assignments(project_id) if item.get("assignment_id") in wanted]

    def list_units(self, project_id: str, assignment_id: str) -> list[dict[str, Any]]:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return []
        with self._lock, self._connect_project(project) as conn:
            rows = conn.execute(
                """
                SELECT u.*, s.animal_code, g.name AS group_name, g.color AS group_color
                  FROM assignment_units u
             LEFT JOIN project_subjects s ON s.subject_id = u.subject_id
             LEFT JOIN project_groups g ON g.group_id = u.group_id
                 WHERE u.assignment_id = ?
              ORDER BY COALESCE(u.roi_index, 9999), u.created_at
                """,
                (assignment_id,),
            ).fetchall()
        return [_row_to_dict(row) or {} for row in rows]

    def update_assignment_experiment(self, project_id: str, assignment_id: str, experiment_type: str) -> dict[str, Any] | None:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return None
        experiment_type = str(experiment_type or "").strip()
        if not experiment_type:
            raise ValueError("Experiment type is required")
        with self._lock, self._connect_project(project) as conn:
            cur = conn.execute(
                """
                UPDATE experiment_assignments
                   SET experiment_type = ?, status = 'pending',
                       latest_run_id = NULL, updated_at = ?
                 WHERE assignment_id = ?
                """,
                (experiment_type, self._now(), assignment_id),
            )
            if not cur.rowcount:
                return None
        return self.get_assignment(project_id, assignment_id)

    def delete_assignment(self, project_id: str, assignment_id: str) -> bool:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return False
        with self._lock, self._connect_project(project) as conn:
            conn.execute("DELETE FROM assignment_units WHERE assignment_id = ?", (assignment_id,))
            cur = conn.execute("DELETE FROM experiment_assignments WHERE assignment_id = ?", (assignment_id,))
            return bool(cur.rowcount)

    def update_assignment_subject(self, project_id: str, assignment_id: str, subject_id: str | None) -> dict[str, Any] | None:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return None
        subject = self.get_subject(project_id, subject_id) if subject_id else None
        now = self._now()
        with self._lock, self._connect_project(project) as conn:
            existing = conn.execute(
                "SELECT unit_id FROM assignment_units WHERE assignment_id = ? AND COALESCE(roi_index, 0) = 0",
                (assignment_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE assignment_units
                       SET subject_id = ?, group_id = ?, unit_label = ?, roi_name = ?, updated_at = ?
                     WHERE unit_id = ?
                    """,
                    (
                        subject.get("subject_id") if subject else None,
                        subject.get("group_id") if subject else None,
                        subject.get("animal_code") if subject else None,
                        subject.get("animal_code") if subject else None,
                        now,
                        existing["unit_id"],
                    ),
                )
            elif subject:
                conn.execute(
                    """
                    INSERT INTO assignment_units(
                        unit_id, assignment_id, subject_id, group_id, unit_label,
                        roi_index, roi_name, include_in_export, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, 0, ?, 1, ?, ?)
                    """,
                    (
                        _entity_id("unt"),
                        assignment_id,
                        subject.get("subject_id"),
                        subject.get("group_id"),
                        subject.get("animal_code"),
                        subject.get("animal_code"),
                        now,
                        now,
                    ),
                )
        return self.get_assignment(project_id, assignment_id)

    def set_assignments_status(
        self,
        project_id: str,
        assignment_ids: list[str],
        status: str,
        latest_run_id: str | None = None,
    ) -> None:
        if not assignment_ids:
            return
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return
        placeholders = ",".join("?" for _ in assignment_ids)
        params: list[Any] = [status, self._now()]
        if latest_run_id is not None:
            latest_sql = ", latest_run_id = ?"
            params.append(latest_run_id)
        else:
            latest_sql = ""
        params.extend(assignment_ids)
        with self._lock, self._connect_project(project) as conn:
            conn.execute(
                f"""
                UPDATE experiment_assignments
                   SET status = ?, updated_at = ?{latest_sql}
                 WHERE assignment_id IN ({placeholders})
                """,
                params,
            )

    def upsert_project_run(
        self,
        project_id: str,
        project_run_id: str,
        run_id: str,
        session_id: str,
        experiment_type: str,
        assignment_ids: list[str],
        output_dir: str = "",
        status: str = "configured",
        progress: int = 0,
        message: str = "",
        result_files: list[str] | None = None,
        finished_at: float | None = None,
    ) -> dict[str, Any] | None:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return None
        now = self._now()
        with self._lock, self._connect_project(project) as conn:
            existing = conn.execute("SELECT created_at FROM project_runs WHERE project_run_id = ?", (project_run_id,)).fetchone()
            created_at = float(existing["created_at"]) if existing else now
            conn.execute(
                """
                INSERT OR REPLACE INTO project_runs(
                    project_run_id, run_id, session_id, experiment_type, assignment_ids,
                    output_dir, status, progress, message, created_at, updated_at,
                    finished_at, result_files
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_run_id,
                    run_id,
                    session_id,
                    experiment_type,
                    _json_dumps(assignment_ids),
                    output_dir or "",
                    status,
                    max(0, min(100, int(progress or 0))),
                    message or "",
                    created_at,
                    now,
                    finished_at,
                    _json_dumps(result_files or []),
                ),
            )
        return self.get_project_run(project_id, project_run_id)

    def get_project_run(self, project_id: str, project_run_id: str) -> dict[str, Any] | None:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return None
        with self._lock, self._connect_project(project) as conn:
            row = conn.execute("SELECT * FROM project_runs WHERE project_run_id = ?", (project_run_id,)).fetchone()
        item = _row_to_dict(row)
        if item:
            item["assignment_ids"] = _json_loads(item.get("assignment_ids"), [])
            item["result_files"] = _json_loads(item.get("result_files"), [])
        return item

    def list_project_runs(self, project_id: str) -> list[dict[str, Any]]:
        project = self.get_project(project_id, include_summary=False)
        if not project:
            return []
        with self._lock, self._connect_project(project) as conn:
            rows = conn.execute("SELECT * FROM project_runs ORDER BY updated_at DESC").fetchall()
        runs = []
        for row in rows:
            item = _row_to_dict(row) or {}
            item["assignment_ids"] = _json_loads(item.get("assignment_ids"), [])
            item["result_files"] = _json_loads(item.get("result_files"), [])
            runs.append(item)
        return runs

    def subject_candidates(self, project_id: str) -> list[dict[str, Any]]:
        subjects = self.list_subjects(project_id)
        return [
            {
                "subject_id": item.get("subject_id"),
                "animal_code": item.get("animal_code"),
                "group_id": item.get("group_id"),
                "group_name": item.get("group_name"),
                "group_color": item.get("group_color"),
                "label": f"{item.get('animal_code') or ''}"
                + (f" · {item.get('group_name')}" if item.get("group_name") else ""),
            }
            for item in subjects
        ]
