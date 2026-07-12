from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

from services.atomic_json import atomic_write_json, read_json

from services.experiment_defaults import (
    ensure_valid_experiment_defaults,
    experiment_defaults_schema,
    get_experiment_defaults,
    normalize_experiment_defaults,
    safe_experiment_defaults,
)


class ExperimentDefaultsStore:
    """Disk-backed global defaults for all assay configuration builders."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get_defaults(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            if not self.path.is_file():
                return normalize_experiment_defaults(None)
            try:
                data = read_json(self.path)
            except Exception:
                data = None
            return safe_experiment_defaults(data)

    def get_experiment_defaults(self, experiment_type: str | None) -> dict[str, Any]:
        return get_experiment_defaults(self.get_defaults(), experiment_type)

    def get_payload(self) -> dict[str, Any]:
        return {"defaults": self.get_defaults(), "schema": experiment_defaults_schema()}

    def save_defaults(self, payload: Any) -> dict[str, dict[str, Any]]:
        ensure_valid_experiment_defaults(payload)
        normalized = normalize_experiment_defaults(payload)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.path, normalized)
        return copy.deepcopy(normalized)

    def reset(self) -> dict[str, dict[str, Any]]:
        return self.save_defaults(None)
