from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from services.resource_policy import video_queue_capacity


LogFn = Callable[[str], None]

DEFAULT_CRF = 26
DEFAULT_PRESET = "veryfast"
DEFAULT_MAXRATE_MBPS = 18.0
DEFAULT_SOURCE_RATIO_LIMIT = 1.8

_ACTIVE_FFMPEG_LOCK = threading.RLock()
_ACTIVE_FFMPEG_PROCESSES: set[subprocess.Popen] = set()


def _noop_log(_message: str) -> None:
    return


def _register_ffmpeg_process(process: subprocess.Popen) -> None:
    with _ACTIVE_FFMPEG_LOCK:
        _ACTIVE_FFMPEG_PROCESSES.add(process)


def _unregister_ffmpeg_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    with _ACTIVE_FFMPEG_LOCK:
        _ACTIVE_FFMPEG_PROCESSES.discard(process)


def terminate_active_ffmpeg_processes(*, timeout: float = 3.0) -> int:
    """Terminate only FFmpeg children created by MouseScope video writers."""
    with _ACTIVE_FFMPEG_LOCK:
        processes = list(_ACTIVE_FFMPEG_PROCESSES)

    active = [process for process in processes if process.poll() is None]
    for process in active:
        try:
            process.terminate()
        except OSError:
            pass

    deadline = time.monotonic() + max(0.0, float(timeout))
    for process in active:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        finally:
            _unregister_ffmpeg_process(process)
    return len(active)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _candidate_ffmpeg_paths() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("MOUSESCOPE_FFMPEG_EXE")
    if env_path:
        candidates.append(Path(env_path))

    try:
        import imageio_ffmpeg

        candidates.append(Path(imageio_ffmpeg.get_ffmpeg_exe()))
    except Exception:
        pass

    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))

    executable_names = ["ffmpeg.exe", "ffmpeg"] if os.name == "nt" else ["ffmpeg", "ffmpeg.exe"]
    for root in roots:
        for exe_name in executable_names:
            candidates.append(root / exe_name)
            candidates.append(root / "bin" / exe_name)
        for pattern in (
            "imageio_ffmpeg/binaries/ffmpeg*",
            "_internal/imageio_ffmpeg/binaries/ffmpeg*",
        ):
            candidates.extend(root.glob(pattern))

    which_path = shutil.which("ffmpeg")
    if which_path:
        candidates.append(Path(which_path))

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except Exception:
            resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(candidate)
    return unique_candidates


def find_ffmpeg_exe() -> str | None:
    for candidate in _candidate_ffmpeg_paths():
        if candidate.is_file():
            return str(candidate)
    return None


def _source_bitrate_mbps(source_video_path: str | os.PathLike | None) -> float | None:
    if not source_video_path:
        return None
    path = Path(source_video_path)
    if not path.is_file():
        return None

    duration_s = 0.0
    cap = cv2.VideoCapture(str(path))
    try:
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            if fps > 0.0 and frames > 0.0:
                duration_s = frames / fps
    finally:
        cap.release()

    if duration_s <= 0.0:
        return None
    try:
        return (path.stat().st_size * 8.0) / duration_s / 1_000_000.0
    except OSError:
        return None


def _min_rate_for_size(width: int, height: int) -> float:
    pixels = int(width) * int(height)
    if pixels >= 1280 * 720:
        return 2.5
    if pixels >= 640 * 480:
        return 1.5
    return 0.8


def _target_maxrate_mbps(width: int, height: int, fps: float, source_video_path: str | os.PathLike | None) -> float:
    env_value = os.environ.get("MOUSESCOPE_VIDEO_MAXRATE_MBPS")
    if env_value:
        return max(0.5, min(80.0, _env_float("MOUSESCOPE_VIDEO_MAXRATE_MBPS", DEFAULT_MAXRATE_MBPS)))

    fps = max(1.0, float(fps or 30.0))
    min_rate = _min_rate_for_size(width, height)
    pixel_rate = max(1, int(width) * int(height)) * fps
    base_mbps = pixel_rate / (1920.0 * 1080.0 * 30.0) * 8.0
    target_mbps = max(min_rate, min(DEFAULT_MAXRATE_MBPS, base_mbps))

    source_mbps = _source_bitrate_mbps(source_video_path)
    if source_mbps:
        ratio_limit = max(0.25, source_mbps * _env_float("MOUSESCOPE_VIDEO_SOURCE_RATIO_LIMIT", DEFAULT_SOURCE_RATIO_LIMIT))
        target_mbps = min(target_mbps, ratio_limit)

    return round(target_mbps, 2)


class FfmpegVideoWriter:
    def __init__(
        self,
        output_path: str | os.PathLike,
        fps: float,
        width: int,
        height: int,
        *,
        ffmpeg_exe: str,
        label: str,
        source_video_path: str | os.PathLike | None = None,
        logger: LogFn | None = None,
    ) -> None:
        self.output_path = str(output_path)
        self.fps = max(1.0, float(fps or 30.0))
        self.width = int(width)
        self.height = int(height)
        self.ffmpeg_exe = ffmpeg_exe
        self.label = label
        self.logger = logger or _noop_log
        self.frame_count = 0
        self.closed = False
        self.failed = False
        self.error_message = ""
        self.process: subprocess.Popen | None = None

        self.crf = max(18, min(36, _env_int("MOUSESCOPE_VIDEO_CRF", DEFAULT_CRF)))
        self.preset = os.environ.get("MOUSESCOPE_VIDEO_PRESET", DEFAULT_PRESET).strip() or DEFAULT_PRESET
        self.maxrate_mbps = _target_maxrate_mbps(self.width, self.height, self.fps, source_video_path)
        self._open()

    def _open(self) -> None:
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        maxrate_kbps = max(500, int(self.maxrate_mbps * 1000))
        bufsize_kbps = maxrate_kbps * 2
        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostats",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-maxrate",
            f"{maxrate_kbps}k",
            "-bufsize",
            f"{bufsize_kbps}k",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            self.output_path,
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        _register_ffmpeg_process(self.process)
        self.logger(
            f"{self.label} codec: FFmpeg H.264 "
            f"(CRF {self.crf}, maxrate {self.maxrate_mbps:.2f} Mbps)"
        )

    def isOpened(self) -> bool:
        return self.process is not None and self.process.stdin is not None and self.process.poll() is None

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("Cannot write an empty video frame.")
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        elif frame.shape[2] != 3:
            raise ValueError(f"Unsupported video frame shape: {frame.shape!r}")
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        return frame

    def write(self, frame: np.ndarray) -> None:
        if self.closed or self.failed:
            return
        if not self.isOpened():
            self.failed = True
            self.error_message = "FFmpeg process is not running."
            return
        try:
            normalized = self._normalize_frame(frame)
            assert self.process is not None and self.process.stdin is not None
            self.process.stdin.write(normalized.tobytes())
            self.frame_count += 1
        except (BrokenPipeError, OSError, ValueError) as exc:
            self.failed = True
            self.error_message = str(exc)

    def release(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process is None:
            return

        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass

        try:
            try:
                return_code = self.process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                self.process.kill()
                return_code = self.process.wait(timeout=10)
                self.failed = True
                self.error_message = "FFmpeg did not finish within the timeout."
        finally:
            _unregister_ffmpeg_process(self.process)

        stderr_text = ""
        if self.process.stderr is not None:
            try:
                stderr_text = self.process.stderr.read().decode("utf-8", errors="replace").strip()
            except Exception:
                stderr_text = ""
            finally:
                try:
                    self.process.stderr.close()
                except OSError:
                    pass

        if return_code != 0:
            self.failed = True
            if stderr_text:
                self.error_message = stderr_text[-500:]

        if self.failed:
            detail = f": {self.error_message}" if self.error_message else ""
            self.logger(f"{self.label} FFmpeg writer failed{detail}")
            return

        try:
            size_mb = Path(self.output_path).stat().st_size / (1024 * 1024)
            self.logger(f"{self.label} encoded: {size_mb:.1f} MB, {self.frame_count} frames")
        except OSError:
            self.logger(f"{self.label} encoded: {self.frame_count} frames")


class AsyncVideoWriter:
    """Queue video frames so analysis is not blocked by encoder I/O."""

    def __init__(
        self,
        writer,
        *,
        label: str = "Annotated video",
        logger: LogFn | None = None,
        max_queue: int = 120,
    ) -> None:
        self.writer = writer
        self.label = label
        self.logger = logger or _noop_log
        self.queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max(1, int(max_queue or 1)))
        self.closed = False
        self.failed = False
        self.error_message = ""
        self.frame_count = 0
        self._dropped_after_failure = 0
        self._thread = threading.Thread(target=self._run, name=f"{label} writer", daemon=True)
        self._thread.start()

    def isOpened(self) -> bool:
        try:
            return bool(self.writer is not None and self.writer.isOpened())
        except Exception:
            return self.writer is not None and not self.closed

    def _run(self) -> None:
        while True:
            frame = self.queue.get()
            try:
                if frame is None:
                    return
                if self.writer is None:
                    self.failed = True
                    self.error_message = "No underlying video writer."
                    continue
                self.writer.write(frame)
                self.frame_count += 1
                if getattr(self.writer, "failed", False):
                    self.failed = True
                    self.error_message = getattr(self.writer, "error_message", "")
            except Exception as exc:
                self.failed = True
                self.error_message = str(exc)
            finally:
                self.queue.task_done()

    def write(self, frame: np.ndarray) -> None:
        if self.closed or self.failed or frame is None:
            if self.failed:
                self._dropped_after_failure += 1
            return
        try:
            self.queue.put(frame.copy())
        except Exception as exc:
            self.failed = True
            self.error_message = str(exc)

    def release(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            self.queue.put(None)
            self._thread.join(timeout=300)
        finally:
            if self.writer is not None:
                try:
                    self.writer.release()
                except Exception as exc:
                    self.failed = True
                    self.error_message = str(exc)
            if getattr(self.writer, "failed", False):
                self.failed = True
                self.error_message = getattr(self.writer, "error_message", self.error_message)
            if self.failed and self.error_message:
                self.logger(f"{self.label} async writer failed: {self.error_message}")


def _open_opencv_writer(
    output_path: str | os.PathLike,
    fps: float,
    width: int,
    height: int,
    *,
    label: str,
    logger: LogFn,
):
    for codec in ("avc1", "H264", "mp4v"):
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (int(width), int(height)))
            if writer.isOpened():
                logger(f"{label} codec: OpenCV {codec} (FFmpeg unavailable; size control is limited)")
                return writer
            writer.release()
        except Exception:
            pass
    return None


def create_annotated_video_writer(
    output_path: str | os.PathLike,
    fps: float,
    width: int,
    height: int,
    *,
    label: str = "Annotated video",
    source_video_path: str | os.PathLike | None = None,
    logger: LogFn | None = None,
    async_write: bool = False,
    max_queue: int = 120,
):
    logger = logger or _noop_log
    queue_capacity = video_queue_capacity(width, height, requested=max_queue)
    if async_write and queue_capacity < int(max_queue or 0):
        logger(f"{label} async queue limited to {queue_capacity} frames for {int(width)}x{int(height)} output")
    writer = None
    if os.environ.get("MOUSESCOPE_DISABLE_FFMPEG_WRITER", "").lower() not in {"1", "true", "yes"}:
        ffmpeg_exe = find_ffmpeg_exe()
        if ffmpeg_exe:
            try:
                writer = FfmpegVideoWriter(
                    output_path,
                    fps,
                    width,
                    height,
                    ffmpeg_exe=ffmpeg_exe,
                    label=label,
                    source_video_path=source_video_path,
                    logger=logger,
                )
                if writer.isOpened():
                    if async_write:
                        return AsyncVideoWriter(writer, label=label, logger=logger, max_queue=queue_capacity)
                    return writer
            except Exception as exc:
                logger(f"{label} FFmpeg startup failed: {exc}")
        else:
            logger(f"{label} FFmpeg executable not found.")

    writer = _open_opencv_writer(output_path, fps, width, height, label=label, logger=logger)
    if writer is not None and async_write:
        return AsyncVideoWriter(writer, label=label, logger=logger, max_queue=queue_capacity)
    return writer
