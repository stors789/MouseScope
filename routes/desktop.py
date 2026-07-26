from __future__ import annotations

from flask import jsonify, make_response, redirect, request, url_for

from services.desktop_runtime import DESKTOP_TOKEN_COOKIE, desktop_token_matches
from services.desktop_runtime import request_desktop_shutdown


def register_desktop_routes(app, _ctx) -> None:
    @app.route("/__desktop_bootstrap")
    def desktop_bootstrap():
        expected = str(app.config.get("MOUSESCOPE_DESKTOP_TOKEN") or "")
        supplied = request.args.get("token")
        if not desktop_token_matches(expected, supplied):
            return jsonify({"success": False, "error": "Invalid desktop session."}), 403

        response = make_response(redirect(url_for("index")))
        response.set_cookie(
            DESKTOP_TOKEN_COOKIE,
            expected,
            httponly=True,
            samesite="Strict",
            path="/",
        )
        return response

    @app.route("/api/desktop/health")
    def desktop_health():
        if not app.config.get("MOUSESCOPE_DESKTOP_MODE"):
            return jsonify({"success": False, "error": "Desktop mode is not active."}), 404
        return jsonify({"success": True, "desktop": True})

    @app.route("/api/desktop/quit", methods=["POST"])
    def desktop_quit():
        if not app.config.get("MOUSESCOPE_DESKTOP_MODE"):
            return jsonify({"success": False, "error": "Desktop mode is not active."}), 404
        accepted = request_desktop_shutdown()
        return jsonify({"success": True, "closing": accepted}), 202

    @app.route("/api/desktop/preferences/tours", methods=["POST"])
    def desktop_complete_tour():
        if not app.config.get("MOUSESCOPE_DESKTOP_MODE"):
            return jsonify({"success": False, "error": "Desktop mode is not active."}), 404
        payload = request.get_json(silent=True) or {}
        key = str(payload.get("key") or "")
        store = app.config.get("MOUSESCOPE_DESKTOP_PREFERENCES_STORE")
        if store is None or not store.is_valid_tour_key(key):
            return jsonify({"success": False, "error": "Invalid guided-tour key."}), 400
        store.complete_tour(key)
        return jsonify({"success": True})
