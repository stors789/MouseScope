from __future__ import annotations

import logging
import threading
import time
import unittest
import urllib.request
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask, jsonify, request

import app as mousescope_app
import desktop
from services.desktop_runtime import (
    DESKTOP_TOKEN_HEADER,
    DesktopServer,
    configure_desktop_shutdown,
    desktop_token_matches,
    is_loopback_address,
    request_desktop_shutdown,
    stop_analysis_sessions,
)


class _FakeWindow:
    def __init__(self):
        self.destroyed = False
        self.dialog_result = None

    def destroy(self):
        self.destroyed = True

    def create_file_dialog(self, *_args, **_kwargs):
        return self.dialog_result


class _FakeWebview:
    settings = {}
    FileDialog = SimpleNamespace(FOLDER=20)

    def __init__(self):
        self.window = _FakeWindow()
        self.create_args = None
        self.start_args = None

    def create_window(self, *args, **kwargs):
        self.create_args = (args, kwargs)
        return self.window

    def start(self, **kwargs):
        self.start_args = kwargs


class _DummyAnalysis:
    def __init__(self):
        self.is_running = True
        self.status = "analyzing"
        self.stop_calls = 0
        self._finished = threading.Event()
        self._thread = threading.Thread(target=self._work)
        self._thread.start()

    def _work(self):
        self._finished.wait(timeout=2)

    def stop(self):
        self.stop_calls += 1
        self.is_running = False
        self.status = "stopped"
        self._finished.set()
        return True


class DesktopRuntimeTest(unittest.TestCase):
    def tearDown(self):
        configure_desktop_shutdown(None)
        mousescope_app.app.config.update(
            MOUSESCOPE_DESKTOP_MODE=False,
            MOUSESCOPE_DESKTOP_TOKEN="",
            MOUSESCOPE_DESKTOP_ORIGIN="",
        )

    def test_loopback_and_constant_time_token_helpers(self):
        self.assertTrue(is_loopback_address("127.0.0.1"))
        self.assertTrue(is_loopback_address("::1"))
        self.assertFalse(is_loopback_address("0.0.0.0"))
        self.assertFalse(is_loopback_address("192.168.1.10"))
        self.assertTrue(desktop_token_matches("secret", "secret"))
        self.assertFalse(desktop_token_matches("secret", "wrong"))

    def test_random_port_server_readiness_and_shutdown(self):
        flask_app = Flask(__name__)

        @flask_app.route("/api/desktop/health")
        def health():
            if request.headers.get(DESKTOP_TOKEN_HEADER) != "ready-token":
                return jsonify({"success": False}), 403
            return jsonify({"success": True})

        server = DesktopServer(flask_app)
        self.assertEqual(server.address.host, "127.0.0.1")
        self.assertGreater(server.address.port, 0)
        self.assertNotEqual(server.address.port, 8080)
        server.start()
        try:
            server.wait_until_ready("ready-token", timeout=3)
            request_obj = urllib.request.Request(
                f"{server.address.origin}/api/desktop/health",
                headers={DESKTOP_TOKEN_HEADER: "ready-token"},
            )
            with urllib.request.urlopen(request_obj, timeout=2) as response:
                self.assertEqual(response.status, 200)
        finally:
            server.shutdown()
        self.assertFalse(server._thread.is_alive())

    def test_desktop_guard_requires_token_host_and_same_origin(self):
        app = mousescope_app.app
        origin = "http://127.0.0.1:49152"
        token = "desktop-test-token"
        app.config.update(
            TESTING=True,
            MOUSESCOPE_DESKTOP_MODE=True,
            MOUSESCOPE_DESKTOP_TOKEN=token,
            MOUSESCOPE_DESKTOP_ORIGIN=origin,
        )
        client = app.test_client()

        rejected = client.get("/api/desktop/health", base_url=origin)
        self.assertEqual(rejected.status_code, 403)
        wrong_host = client.get(
            "/api/desktop/health",
            base_url="http://localhost:49152",
            headers={DESKTOP_TOKEN_HEADER: token},
        )
        self.assertEqual(wrong_host.status_code, 403)
        accepted = client.get(
            "/api/desktop/health",
            base_url=origin,
            headers={DESKTOP_TOKEN_HEADER: token},
        )
        self.assertEqual(accepted.status_code, 200)

        bootstrap = client.get(
            f"/__desktop_bootstrap?token={token}",
            base_url=origin,
            follow_redirects=False,
        )
        self.assertEqual(bootstrap.status_code, 302)
        self.assertIn("HttpOnly", bootstrap.headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", bootstrap.headers["Set-Cookie"])

        cross_origin = client.post(
            "/api/accept_license",
            base_url=origin,
            headers={"Origin": "https://example.invalid"},
        )
        self.assertEqual(cross_origin.status_code, 403)

    def test_shutdown_callback_runs_only_once(self):
        called = threading.Event()
        calls = []

        def handler():
            calls.append(True)
            called.set()

        configure_desktop_shutdown(handler)
        self.assertTrue(request_desktop_shutdown())
        self.assertFalse(request_desktop_shutdown())
        self.assertTrue(called.wait(timeout=1))
        self.assertEqual(len(calls), 1)

    def test_analysis_shutdown_stops_and_joins_workers(self):
        analysis = _DummyAnalysis()
        store = {"session": {"analysis": analysis}}
        with patch(
            "services.video_export.terminate_active_ffmpeg_processes",
            return_value=2,
        ):
            summary = stop_analysis_sessions(store, timeout=1)
        self.assertEqual(analysis.stop_calls, 1)
        self.assertFalse(analysis._thread.is_alive())
        self.assertEqual(summary["analysis_threads_alive"], 0)
        self.assertEqual(summary["ffmpeg_processes_terminated"], 2)

    def test_launcher_uses_cocoa_without_opening_external_browser(self):
        fake_webview = _FakeWebview()
        werkzeug_logger = logging.getLogger("werkzeug")
        original_level = werkzeug_logger.level
        with patch("desktop._configure_file_logging"), patch(
            "desktop.stop_analysis_sessions",
            return_value={},
        ), patch("desktop.webbrowser.open") as open_browser:
            try:
                result = desktop.run_desktop_app(webview_module=fake_webview)
                self.assertGreaterEqual(werkzeug_logger.level, logging.WARNING)
            finally:
                werkzeug_logger.setLevel(original_level)

        self.assertEqual(result, 0)
        args, kwargs = fake_webview.create_args
        self.assertEqual(args[0], "MouseScope")
        self.assertIn("/__desktop_bootstrap?token=", args[1])
        self.assertEqual(kwargs["min_size"], (960, 680))
        self.assertEqual(fake_webview.start_args["gui"], "cocoa")
        self.assertFalse(fake_webview.start_args["debug"])
        self.assertFalse(fake_webview.start_args["private_mode"])
        open_browser.assert_not_called()
        self.assertFalse(mousescope_app.app.config["MOUSESCOPE_DESKTOP_MODE"])


if __name__ == "__main__":
    unittest.main()
