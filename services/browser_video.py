from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from services.video_export import find_ffmpeg_exe


class BrowserVideoError(RuntimeError):
    pass


_NATIVE_SUFFIXES = {".mp4", ".m4v"}
_proxy_locks: dict[str, threading.Lock] = {}
_proxy_locks_guard = threading.Lock()


def _proxy_lock(key: str) -> threading.Lock:
    with _proxy_locks_guard:
        return _proxy_locks.setdefault(key, threading.Lock())


def _orientation_filter(degrees: int) -> str | None:
    normalized = int(degrees or 0) % 360
    if normalized == 90:
        return "transpose=clock"
    if normalized == 180:
        return "hflip,vflip"
    if normalized == 270:
        return "transpose=cclock"
    return None


def ensure_browser_video(
    source_path: str | os.PathLike,
    cache_dir: str | os.PathLike,
    *,
    orientation_degrees: int = 0,
) -> tuple[Path, bool]:
    """Return an HTML5-compatible source, transcoding legacy containers once."""
    source = Path(source_path)
    if not source.is_file():
        raise BrowserVideoError(f"Video file does not exist: {source}")
    if source.suffix.lower() in _NATIVE_SUFFIXES and int(orientation_degrees or 0) % 360 == 0:
        return source, False

    ffmpeg_exe = find_ffmpeg_exe()
    if not ffmpeg_exe:
        raise BrowserVideoError("FFmpeg is unavailable, so this video cannot be prepared for browser playback.")

    stat = source.stat()
    identity = "\0".join(
        (
            str(source.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(int(orientation_degrees or 0) % 360),
        )
    )
    cache_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / f"{cache_key}.mp4"
    if target.is_file() and target.stat().st_size > 0:
        return target, True

    with _proxy_lock(cache_key):
        if target.is_file() and target.stat().st_size > 0:
            return target, True
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_key}-",
            suffix=".mp4",
            dir=str(cache_root),
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            cmd = [
                ffmpeg_exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-noautorotate",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
            ]
            orientation_filter = _orientation_filter(orientation_degrees)
            if orientation_filter:
                cmd.extend(["-vf", orientation_filter])
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "24",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ]
            )
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=1800,
                check=False,
            )
            if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
                detail = result.stderr.decode("utf-8", errors="replace").strip()
                raise BrowserVideoError(detail or "Failed to prepare the video for browser playback.")
            os.replace(temporary, target)
        except subprocess.TimeoutExpired as exc:
            raise BrowserVideoError("Timed out while preparing the video for browser playback.") from exc
        finally:
            temporary.unlink(missing_ok=True)
    return target, True
