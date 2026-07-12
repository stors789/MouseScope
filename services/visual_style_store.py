from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

from services.atomic_json import atomic_write_json, read_json

from core.visual_style import DEFAULT_VISUAL_STYLE, normalize_visual_style, style_payload


class VisualStyleStore:
    """Disk-backed application-level visual style settings."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get_style(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                return normalize_visual_style(None)
            try:
                data = read_json(self.path)
            except Exception:
                return normalize_visual_style(None)
            return normalize_visual_style(data)

    def get_payload(self) -> dict[str, Any]:
        return style_payload(self.get_style())

    def save_style(self, style: Any) -> dict[str, Any]:
        normalized = normalize_visual_style(style)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.path, normalized)
        return normalized

    def reset(self) -> dict[str, Any]:
        return self.save_style(copy.deepcopy(DEFAULT_VISUAL_STYLE))
