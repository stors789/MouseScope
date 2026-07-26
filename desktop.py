#!/usr/bin/env python3
from __future__ import annotations

import logging
import secrets
import sys
import urllib.parse
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from services.desktop_runtime import DesktopServer


def _configure_file_logging(data_dir: Path) -> None:
    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "desktop.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)


class DesktopApi:
    """Small, deliberately narrow bridge exposed to the Web UI."""

    def __init__(self, origin: str) -> None:
        self._origin = origin.rstrip("/")

    def open_external(self, raw_url: str) -> bool:
        try:
            parsed = urllib.parse.urlsplit(str(raw_url or ""))
        except ValueError:
            return False
        if parsed.scheme not in {"http", "https"}:
            return False
        target_origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        if target_origin == self._origin:
            return False
        return bool(webbrowser.open(parsed.geturl()))


def run_desktop_app(*, webview_module: Any | None = None) -> int:
    if sys.platform != "darwin":
        raise RuntimeError("The MouseScope desktop window entry point is macOS-only.")

    import app as app_module

    token = secrets.token_urlsafe(32)
    app_module.app.config.update(
        MOUSESCOPE_DESKTOP_MODE=True,
        MOUSESCOPE_DESKTOP_TOKEN=token,
    )
    server = DesktopServer(app_module.app)
    app_module.app.config["MOUSESCOPE_DESKTOP_ORIGIN"] = server.address.origin
    _configure_file_logging(Path(app_module.DATA_DIR))

    if webview_module is None:
        import webview as webview_module

    bootstrap_url = (
        f"{server.address.origin}/__desktop_bootstrap?"
        + urllib.parse.urlencode({"token": token})
    )
    window = webview_module.create_window(
        "MouseScope",
        bootstrap_url,
        js_api=DesktopApi(server.address.origin),
        width=1440,
        height=960,
        min_size=(960, 680),
        resizable=True,
        text_select=True,
        zoomable=True,
        background_color="#f5f1eb",
    )
    if window is None:
        raise RuntimeError("Unable to create the MouseScope desktop window.")

    try:
        server.start()
        server.wait_until_ready(token)
        webview_module.settings["ALLOW_DOWNLOADS"] = True
        webview_module.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False
        webview_module.start(
            gui="cocoa",
            debug=False,
            private_mode=False,
            storage_path=str(Path(app_module.DATA_DIR) / "webview"),
        )
        return 0
    finally:
        server.shutdown()


if __name__ == "__main__":
    raise SystemExit(run_desktop_app())
