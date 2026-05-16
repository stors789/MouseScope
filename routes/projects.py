from __future__ import annotations

from flask import flash, redirect, render_template, url_for

from services.experiment_registry import list_experiments


def register_project_page_routes(app, ctx) -> None:
    project_store = ctx.project_store

    @app.route("/projects")
    def projects_page():
        return render_template("projects.html")

    @app.route("/projects/<project_id>")
    def project_home(project_id):
        project = project_store.get_project(project_id)
        if not project:
            flash("项目不存在或已被移动。", "error")
            return redirect(url_for("projects_page"))
        project_store.touch_project(project_id)
        return render_template(
            "project_home.html",
            project=project,
            experiments=list_experiments(),
        )
