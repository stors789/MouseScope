from __future__ import annotations

import hmac
import ipaddress
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from werkzeug.serving import make_server


logger = logging.getLogger(__name__)

DESKTOP_TOKEN_HEADER = "X-MouseScope-Desktop-Token"
DESKTOP_TOKEN_COOKIE = "mousescope_desktop_token"

_shutdown_lock = threading.RLock()
_shutdown_handler: Any | None = None
_shutdown_requested = False


def is_loopback_address(value: str | None) -> bool:
    """Return whether a request peer is a numeric loopback address."""
    try:
        return ipaddress.ip_address(str(value or "").split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def desktop_token_matches(expected: str, supplied: str | None) -> bool:
    return bool(expected and supplied and hmac.compare_digest(expected, str(supplied)))


def configure_desktop_shutdown(handler: Any | None) -> None:
    global _shutdown_handler, _shutdown_requested
    with _shutdown_lock:
        _shutdown_handler = handler
        _shutdown_requested = False


def request_desktop_shutdown() -> bool:
    """Request that the native window close without blocking the HTTP response."""
    global _shutdown_requested
    with _shutdown_lock:
        if _shutdown_requested or _shutdown_handler is None:
            return False
        _shutdown_requested = True
        handler = _shutdown_handler

    def _invoke() -> None:
        try:
            handler()
        except Exception:
            logger.exception("MouseScope desktop shutdown handler failed.")

    threading.Thread(target=_invoke, name="MouseScopeDesktopQuit", daemon=True).start()
    return True


def stop_analysis_sessions(session_store: Any, *, timeout: float = 12.0) -> dict[str, int]:
    """Stop active analyses, wait briefly, then terminate owned FFmpeg children."""
    analyses: list[Any] = []
    seen: set[int] = set()
    try:
        session_items = list(session_store.items())
    except Exception:
        logger.exception("Unable to enumerate analysis sessions during shutdown.")
        session_items = []

    for _session_id, session_data in session_items:
        analysis = session_data.get("analysis") if isinstance(session_data, dict) else None
        if analysis is None or id(analysis) in seen:
            continue
        seen.add(id(analysis))
        analyses.append(analysis)
        try:
            analysis.stop()
        except Exception:
            logger.exception("Unable to stop an analysis session during shutdown.")

    deadline = time.monotonic() + max(0.0, float(timeout))
    for analysis in analyses:
        thread = getattr(analysis, "_thread", None)
        if thread is None or not hasattr(thread, "join"):
            continue
        remaining = max(0.0, deadline - time.monotonic())
        if remaining <= 0:
            break
        try:
            thread.join(timeout=remaining)
        except (RuntimeError, TypeError):
            logger.exception("Unable to join an analysis thread during shutdown.")

    from services.video_export import terminate_active_ffmpeg_processes

    terminated = terminate_active_ffmpeg_processes(timeout=3.0)
    alive = sum(
        1
        for analysis in analyses
        if getattr(analysis, "_thread", None) is not None
        and getattr(analysis._thread, "is_alive", lambda: False)()
    )
    if alive:
        logger.warning("%d analysis thread(s) were still finishing when the desktop window closed.", alive)
    return {
        "analyses": len(analyses),
        "analysis_threads_alive": alive,
        "ffmpeg_processes_terminated": terminated,
    }


@dataclass(frozen=True)
class DesktopServerAddress:
    host: str
    port: int

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"


class DesktopServer:
    """Run the Flask application on a random loopback port."""

    def __init__(self, flask_app: Any, *, host: str = "127.0.0.1") -> None:
        if not is_loopback_address(host):
            raise ValueError("The desktop server must bind to a loopback address.")
        self._server = make_server(host, 0, flask_app, threaded=True)
        self.address = DesktopServerAddress(host=host, port=int(self._server.server_port))
        self._thread = threading.Thread(
            target=self._serve,
            name="MouseScopeDesktopHTTP",
            daemon=True,
        )
        self._started = False
        self._stopped = False

    def _serve(self) -> None:
        try:
            self._server.serve_forever()
        except Exception:
            if not self._stopped:
                logger.exception("MouseScope desktop HTTP server stopped unexpectedly.")

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def wait_until_ready(self, token: str, *, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + max(0.1, float(timeout))
        request = urllib.request.Request(
            f"{self.address.origin}/api/desktop/health",
            headers={DESKTOP_TOKEN_HEADER: token},
        )
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if self._started and not self._thread.is_alive():
                raise RuntimeError("MouseScope desktop HTTP server exited during startup.")
            try:
                with urllib.request.urlopen(request, timeout=0.5) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError) as exc:
                last_error = exc
            time.sleep(0.05)
        raise RuntimeError("MouseScope desktop HTTP server did not become ready.") from last_error

    def shutdown(self, *, timeout: float = 5.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            if self._started:
                self._server.shutdown()
        finally:
            self._server.server_close()
            if self._thread.is_alive() and threading.current_thread() is not self._thread:
                self._thread.join(timeout=max(0.0, float(timeout)))
