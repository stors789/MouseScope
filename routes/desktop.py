from __future__ import annotations

from flask import jsonify, make_response, redirect, request, url_for

from services.desktop_runtime import DESKTOP_TOKEN_COOKIE, desktop_token_matches


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
        return jsonify({"success": True, "desktop": True})
