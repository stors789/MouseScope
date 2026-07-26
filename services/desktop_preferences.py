from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.atomic_json import atomic_write_json, read_json


DESKTOP_PREFERENCES_SCHEMA_VERSION = 1
MAX_TOUR_KEY_LENGTH = 160


def license_sha256(path: str | Path) -> str:
    """Return the digest used to invalidate acceptance when the EULA changes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_preferences() -> dict[str, Any]:
    return {
        "schema_version": DESKTOP_PREFERENCES_SCHEMA_VERSION,
        "license": {},
        "completed_tours": {},
    }


def _normalize_preferences(raw: Any) -> dict[str, Any]:
    normalized = _empty_preferences()
    if not isinstance(raw, dict):
        return normalized

    license_state = raw.get("license")
    if isinstance(license_state, dict):
        digest = str(license_state.get("sha256") or "").strip().lower()
        accepted_at = str(license_state.get("accepted_at") or "").strip()
        if len(digest) == 64 and all(char in "0123456789abcdef" for char in digest):
            normalized["license"] = {
                "sha256": digest,
                "accepted_at": accepted_at,
            }

    completed_tours = raw.get("completed_tours")
    if isinstance(completed_tours, dict):
        normalized["completed_tours"] = {
            str(key): str(value or "")
            for key, value in completed_tours.items()
            if DesktopPreferencesStore.is_valid_tour_key(key)
        }
    return normalized


class DesktopPreferencesStore:
    """Small disk-backed store for consent and WebView onboarding state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_valid_tour_key(value: Any) -> bool:
        key = str(value or "")
        return (
            key.startswith("mousescope_tour_")
            and 0 < len(key) <= MAX_TOUR_KEY_LENGTH
            and all(char.isalnum() or char in "._-" for char in key)
        )

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return _empty_preferences()
        try:
            return _normalize_preferences(read_json(self.path))
        except (OSError, ValueError, TypeError):
            return _empty_preferences()

    def is_license_accepted(self, digest: str) -> bool:
        expected = str(digest or "").strip().lower()
        with self._lock:
            return self._read()["license"].get("sha256") == expected

    def accept_license(self, digest: str) -> None:
        expected = str(digest or "").strip().lower()
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError("A valid license SHA-256 digest is required.")
        with self._lock:
            preferences = self._read()
            preferences["license"] = {
                "sha256": expected,
                "accepted_at": _timestamp(),
            }
            atomic_write_json(self.path, preferences)

    def completed_tour_keys(self) -> set[str]:
        with self._lock:
            return set(self._read()["completed_tours"])

    def complete_tour(self, key: str) -> None:
        if not self.is_valid_tour_key(key):
            raise ValueError("Invalid guided-tour preference key.")
        with self._lock:
            preferences = self._read()
            preferences["completed_tours"][key] = _timestamp()
            atomic_write_json(self.path, preferences)
