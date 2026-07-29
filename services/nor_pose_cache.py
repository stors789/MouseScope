from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any


CACHE_FORMAT_VERSION = 1


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def video_identity(video_path: str | Path) -> dict[str, Any]:
    path = Path(video_path)
    stat = path.stat()
    size = int(stat.st_size)
    sample_size = 1024 * 1024
    offsets = sorted(
        {
            0,
            max(0, (size - sample_size) // 2),
            max(0, size - sample_size),
        }
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(offset.to_bytes(8, "little", signed=False))
            digest.update(handle.read(sample_size))
    return {
        "size": size,
        "sample_sha256": digest.hexdigest(),
    }


def cache_fingerprint(payload: dict[str, Any]) -> str:
    envelope = {
        "format_version": CACHE_FORMAT_VERSION,
        "payload": payload,
    }
    return hashlib.sha256(_stable_json(envelope).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedPose:
    observation: dict[str, Any] | None
    error: str | None = None


class PersistentNorPoseCache:
    """Small SQLite cache for crop-local NOR pose inference results."""

    def __init__(
        self,
        cache_root: str | Path,
        fingerprint_payload: dict[str, Any],
        *,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self.fingerprint = cache_fingerprint(fingerprint_payload)
        self.path = Path(cache_root) / f"{self.fingerprint}.sqlite3"
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pose_results (
                frame_no INTEGER NOT NULL,
                roi_id INTEGER NOT NULL,
                crop_box TEXT NOT NULL,
                observation TEXT,
                error TEXT,
                PRIMARY KEY (frame_no, roi_id)
            )
            """
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('fingerprint_payload', ?)",
            (_stable_json(fingerprint_payload),),
        )
        connection.commit()
        self._connection = connection

    @staticmethod
    def _crop_json(crop_box: tuple[int, int, int, int]) -> str:
        return _stable_json([int(value) for value in crop_box])

    def get(
        self,
        frame_no: int,
        roi_id: int,
        crop_box: tuple[int, int, int, int],
    ) -> CachedPose | None:
        if not self.enabled:
            return None
        if self._connection is None:
            self.misses += 1
            return None
        try:
            with self._lock:
                row = self._connection.execute(
                    """
                    SELECT crop_box, observation, error
                    FROM pose_results
                    WHERE frame_no = ? AND roi_id = ?
                    """,
                    (int(frame_no), int(roi_id)),
                ).fetchone()
        except sqlite3.Error:
            self.misses += 1
            return None
        if row is None or row[0] != self._crop_json(crop_box):
            self.misses += 1
            return None
        try:
            observation = json.loads(row[1]) if row[1] else None
        except (TypeError, json.JSONDecodeError):
            self.misses += 1
            return None
        self.hits += 1
        return CachedPose(observation=observation, error=row[2])

    def put(
        self,
        frame_no: int,
        roi_id: int,
        crop_box: tuple[int, int, int, int],
        observation: dict[str, Any] | None,
        *,
        error: str | None = None,
    ) -> None:
        if self._connection is None:
            return
        observation_json = _stable_json(observation) if observation is not None else None
        try:
            with self._lock:
                self._connection.execute(
                    """
                    INSERT OR REPLACE INTO pose_results(
                        frame_no, roi_id, crop_box, observation, error
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        int(frame_no),
                        int(roi_id),
                        self._crop_json(crop_box),
                        observation_json,
                        error,
                    ),
                )
                self._connection.commit()
        except sqlite3.Error:
            return
        self.writes += 1

    def close(self) -> None:
        with self._lock:
            connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def __enter__(self) -> "PersistentNorPoseCache":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()
