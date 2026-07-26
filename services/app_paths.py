from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping


logger = logging.getLogger(__name__)


def _ensure_writable_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(str(path))
    fd = None
    probe = None
    try:
        fd, probe = tempfile.mkstemp(prefix='.mousescope-write-test-', dir=str(path))
    finally:
        if fd is not None:
            os.close(fd)
        if probe:
            try:
                os.unlink(probe)
            except FileNotFoundError:
                pass
    return path


def _user_data_directory(environ: Mapping[str, str]) -> Path:
    local_root = environ.get('LOCALAPPDATA') or environ.get('APPDATA')
    if local_root:
        return Path(local_root).expanduser() / 'MouseScope' / 'data'
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'MouseScope' / 'data'
    return Path.home() / '.mousescope' / 'data'


def _macos_data_directory(environ: Mapping[str, str]) -> Path:
    home = Path(str(environ.get('HOME') or Path.home())).expanduser()
    return home / 'Library' / 'Application Support' / 'MouseScope' / 'data'


def _copy_missing_tree(source: Path, destination: Path) -> None:
    """Migrate legacy data without replacing files already in the stable store."""
    if not source.is_dir():
        return
    for item in source.rglob('*'):
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def resolve_application_data_directory(
    base_dir: str | os.PathLike,
    *,
    frozen: bool,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
) -> Path:
    """Choose a writable data directory for source and packaged executions."""
    env = os.environ if environ is None else environ
    override = str(env.get('MOUSESCOPE_DATA_DIR') or '').strip().strip('"').strip("'")
    if override:
        return _ensure_writable_directory(Path(os.path.expandvars(override)).expanduser())

    if not frozen:
        return _ensure_writable_directory(Path(base_dir) / 'data')

    portable = Path(base_dir) / 'data'
    if (platform_name or sys.platform) == 'darwin':
        stable = _ensure_writable_directory(_macos_data_directory(env))
        legacy_candidates = (
            portable,
            _user_data_directory(env),
        )
        for legacy in legacy_candidates:
            try:
                if legacy.resolve() == stable.resolve():
                    continue
                _copy_missing_tree(legacy, stable)
            except OSError as exc:
                logger.warning('Unable to migrate legacy MouseScope data from %s: %s', legacy, exc)
        return stable

    try:
        return _ensure_writable_directory(portable)
    except OSError as exc:
        fallback = _user_data_directory(env)
        logger.warning(
            'Portable data directory is not writable (%s); using %s instead.',
            portable,
            fallback,
        )
        try:
            return _ensure_writable_directory(fallback)
        except OSError as fallback_exc:
            raise RuntimeError(
                f'MouseScope cannot create a writable data directory at {portable} or {fallback}.'
            ) from fallback_exc
