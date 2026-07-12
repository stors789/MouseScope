from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    """Durably replace a JSON file without exposing a partially written file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quarantine_corrupt_json(path: str | Path) -> Path | None:
    source = Path(path)
    if not source.is_file():
        return None
    suffix = time.strftime("%Y%m%d_%H%M%S")
    target = source.with_name(f"{source.name}.corrupt.{suffix}")
    counter = 2
    while target.exists():
        target = source.with_name(f"{source.name}.corrupt.{suffix}_{counter}")
        counter += 1
    try:
        os.replace(source, target)
        return target
    except OSError:
        return None
