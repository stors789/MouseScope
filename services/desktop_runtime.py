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


def is_loopback_address(value: str | None) -> bool:
    """Return whether a request peer is a numeric loopback address."""
    try:
        return ipaddress.ip_address(str(value or "").split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def desktop_token_matches(expected: str, supplied: str | None) -> bool:
    return bool(expected and supplied and hmac.compare_digest(expected, str(supplied)))


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
            self._server.shutdown()
        finally:
            self._server.server_close()
            if self._thread.is_alive() and threading.current_thread() is not self._thread:
                self._thread.join(timeout=max(0.0, float(timeout)))
