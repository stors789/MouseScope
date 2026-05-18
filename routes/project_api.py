from __future__ import annotations

import logging
import os
import platform
import secrets
import subprocess
import time
from pathlib import Path

from flask import jsonify, request, url_for

from services.experiment_registry import get_experiment_spec, normalize_experiment_type


logger = logging.getLogger(__name__)


def register_project_api_routes(app, ctx) -> None:
    project_store = ctx.project_store
    analysis_sessions = ctx.analysis_sessions
    analysis_history = ctx.analysis_history
    VALID_VIDEO_EXTS = ctx.VALID_VIDEO_EXTS
    _extract_video_info_from_entry = ctx._extract_video_info_from_entry
    _now_ts = ctx._now_ts
    _stage_local_video_if_needed = ctx._stage_local_video_if_needed
    _update_video_entry_info = ctx._update_video_entry_info
    _discard_uploaded_staging_file = getattr(ctx, "_discard_uploaded_staging_file", None)

    def _json_payload() -> dict:
        payload = request.get_json(silent=True)
        return payload if isinstance(payload, dict) else {}

    def _error(message: str, status: int = 400):
        return jsonify({"success": False, "error": message}), status

    def _sync_project_runs_from_history(project_id: str) -> None:
        terminal_statuses = {"completed", "error", "stopped", "interrupted"}
        try:
            runs = project_store.list_project_runs(project_id)
        except Exception:
            logger.exception("Failed to list project runs while syncing history: project_id=%s", project_id)
            return
        for run in runs:
            run_id = str(run.get("run_id") or "").strip()
            session_id = str(run.get("session_id") or run_id).strip()
            record = None
            sess = analysis_sessions.get(session_id) if session_id else None
            try:
                if sess and sess.get("analysis") is not None:
                    record = analysis_history.upsert_runtime(session_id, sess, analysis=sess.get("analysis"))
            except Exception:
                logger.debug("Failed to upsert runtime project run status: session_id=%s", session_id, exc_info=True)
                record = None
            if record is None:
                for key in (run_id, session_id):
                    if not key:
                        continue
                    try:
                        record = analysis_history.get_record(key)
                    except Exception:
                        logger.debug("Failed to read analysis history record: run_or_session_id=%s", key, exc_info=True)
                        record = None
                    if record:
                        break
            if not record:
                continue
            record_project_run_id = str(record.get("project_run_id") or "").strip()
            run_project_run_id = str(run.get("project_run_id") or "").strip()
            if not run_project_run_id:
                continue
            if record_project_run_id and run_project_run_id and record_project_run_id != run_project_run_id:
                continue
            status = str(record.get("status") or run.get("status") or "unknown").lower()
            current_status = str(run.get("status") or "").lower()
            record_updated = float(record.get("updated_at") or 0)
            run_updated = float(run.get("updated_at") or 0)
            if current_status in terminal_statuses and status not in terminal_statuses:
                continue
            if record_updated and run_updated and record_updated < run_updated:
                if status not in terminal_statuses or current_status in terminal_statuses:
                    continue
            finished_at = record.get("finished_at")
            if status in terminal_statuses and not finished_at:
                finished_at = record.get("updated_at")
            assignment_ids = run.get("assignment_ids") or []
            progress = int(record.get("progress") or run.get("progress") or 0)
            if status == "completed":
                progress = 100
            project_store.upsert_project_run(
                project_id=project_id,
                project_run_id=run_project_run_id,
                run_id=run_id or session_id,
                session_id=session_id or run_id,
                experiment_type=record.get("experiment_type") or run.get("experiment_type"),
                assignment_ids=assignment_ids,
                output_dir=record.get("output_dir") or run.get("output_dir") or "",
                status=status,
                progress=progress,
                message=str(record.get("message") or run.get("message") or ""),
                result_files=record.get("result_files") or run.get("result_files") or [],
                finished_at=finished_at,
            )
            if status in terminal_statuses:
                project_store.set_assignments_status(project_id, assignment_ids, status, latest_run_id=session_id or run_id)
                if status == "completed":
                    project_store.cleanup_completed_upload_caches(project_id, assignment_ids)

    def _iter_video_paths(raw_value) -> list[Path]:
        raw_items: list[str] = []
        if isinstance(raw_value, list):
            raw_items = [str(item or "") for item in raw_value]
        else:
            text = str(raw_value or "")
            raw_items = [line for line in text.replace("\r", "\n").split("\n")]

        paths: list[Path] = []
        seen: set[str] = set()
        for raw in raw_items:
            value = raw.strip().strip('"').strip("'")
            if not value:
                continue
            candidate = Path(value).expanduser()
            if candidate.is_dir():
                candidates = sorted(child for child in candidate.iterdir() if child.is_file())
            else:
                candidates = [candidate]
            for path in candidates:
                if not path.is_file() or path.suffix.lower() not in VALID_VIDEO_EXTS:
                    continue
                key = os.path.abspath(os.fspath(path)).lower()
                if key in seen:
                    continue
                seen.add(key)
                paths.append(path)
        return paths

    def _iter_uploaded_file_items(raw_value) -> list[tuple[Path, str]]:
        if not isinstance(raw_value, list):
            return []
        items: list[tuple[Path, str]] = []
        seen: set[str] = set()
        for raw in raw_value:
            if isinstance(raw, dict):
                path_value = raw.get("path")
                name_value = str(raw.get("name") or "").strip()
            else:
                path_value = raw
                name_value = ""
            value = str(path_value or "").strip().strip('"').strip("'")
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_file() or path.suffix.lower() not in VALID_VIDEO_EXTS:
                continue
            key = os.path.abspath(os.fspath(path)).lower()
            if key in seen:
                continue
            seen.add(key)
            items.append((path, name_value or path.name))
        return items

    def _open_folder(path: str):
        system = platform.system()
        if system == "Linux":
            subprocess.Popen(["xdg-open", path])
        elif system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            raise RuntimeError(f"Unsupported operating system: {system}")

    @app.route("/api/projects")
    def api_projects():
        include_archived = str(request.args.get("include_archived") or "").lower() in {"1", "true", "yes"}
        return jsonify({"success": True, "projects": project_store.list_projects(include_archived=include_archived)})

    @app.route("/api/projects", methods=["POST"])
    def api_create_project():
        payload = _json_payload()
        try:
            project = project_store.create_project(
                name=str(payload.get("name") or "").strip(),
                description=str(payload.get("description") or "").strip(),
                project_dir=str(payload.get("project_dir") or "").strip() or None,
            )
        except Exception as exc:
            return _error(str(exc), 400)
        return jsonify({"success": True, "project": project, "project_url": url_for("project_home", project_id=project["project_id"])})

    @app.route("/api/projects/<project_id>")
    def api_project_detail(project_id):
        project = project_store.get_project(project_id)
        if not project:
            return _error("项目不存在", 404)
        _sync_project_runs_from_history(project_id)
        project = project_store.get_project(project_id)
        return jsonify({
            "success": True,
            "project": project,
            "groups": project_store.list_groups(project_id),
            "subjects": project_store.list_subjects(project_id),
            "videos": project_store.list_videos(project_id),
            "assignments": project_store.list_assignments(project_id),
            "runs": project_store.list_project_runs(project_id),
        })

    @app.route("/api/projects/<project_id>", methods=["PATCH"])
    def api_update_project(project_id):
        project = project_store.update_project(project_id, _json_payload())
        if not project:
            return _error("项目不存在", 404)
        return jsonify({"success": True, "project": project})

    @app.route("/api/projects/<project_id>/archive", methods=["POST"])
    def api_archive_project(project_id):
        payload = _json_payload()
        archived = bool(payload.get("archived", True))
        if not project_store.archive_project(project_id, archived=archived):
            return _error("项目不存在", 404)
        return jsonify({"success": True, "archived": archived})

    @app.route("/api/projects/<project_id>/open_folder", methods=["POST"])
    def api_open_project_folder(project_id):
        try:
            path = project_store.open_project_folder(project_id)
            _open_folder(path)
        except Exception as exc:
            return _error(str(exc), 500)
        return jsonify({"success": True, "path": path})

    @app.route("/api/projects/<project_id>/groups", methods=["POST"])
    def api_add_group(project_id):
        payload = _json_payload()
        try:
            group = project_store.add_group(
                project_id,
                name=str(payload.get("name") or "").strip(),
                color=str(payload.get("color") or "").strip(),
                notes=str(payload.get("notes") or "").strip(),
            )
        except Exception as exc:
            return _error(str(exc), 400)
        return jsonify({"success": True, "group": group, "groups": project_store.list_groups(project_id)})

    @app.route("/api/projects/<project_id>/groups/<group_id>", methods=["DELETE"])
    def api_delete_group(project_id, group_id):
        if not project_store.delete_group(project_id, group_id):
            return _error("组别不存在", 404)
        return jsonify({"success": True, "groups": project_store.list_groups(project_id), "subjects": project_store.list_subjects(project_id)})

    @app.route("/api/projects/<project_id>/groups/<group_id>/subjects/bulk", methods=["POST"])
    def api_add_group_subjects(project_id, group_id):
        payload = _json_payload()
        try:
            created = project_store.add_subjects_to_group(
                project_id,
                group_id,
                count=payload.get("count"),
                prefix=str(payload.get("prefix") or "").strip(),
            )
        except Exception as exc:
            return _error(str(exc), 400)
        return jsonify({
            "success": True,
            "created": created,
            "subjects": project_store.list_subjects(project_id),
        })

    @app.route("/api/projects/<project_id>/subjects", methods=["POST"])
    def api_add_subject(project_id):
        try:
            subject = project_store.add_subject(project_id, _json_payload())
        except Exception as exc:
            return _error(str(exc), 400)
        return jsonify({"success": True, "subject": subject, "subjects": project_store.list_subjects(project_id)})

    @app.route("/api/projects/<project_id>/subjects/<subject_id>", methods=["PATCH"])
    def api_update_subject(project_id, subject_id):
        try:
            subject = project_store.update_subject(project_id, subject_id, _json_payload())
        except Exception as exc:
            return _error(str(exc), 400)
        if not subject:
            return _error("小鼠不存在", 404)
        return jsonify({"success": True, "subject": subject, "subjects": project_store.list_subjects(project_id)})

    @app.route("/api/projects/<project_id>/subjects/<subject_id>", methods=["DELETE"])
    def api_delete_subject(project_id, subject_id):
        if not project_store.delete_subject(project_id, subject_id):
            return _error("小鼠不存在", 404)
        return jsonify({"success": True, "subjects": project_store.list_subjects(project_id)})

    @app.route("/api/projects/<project_id>/videos/import", methods=["POST"])
    def api_import_project_videos(project_id):
        project = project_store.get_project(project_id)
        if not project:
            return _error("项目不存在", 404)
        payload = _json_payload()
        paths = _iter_video_paths(payload.get("paths") or payload.get("local_paths"))
        uploaded_items = _iter_uploaded_file_items(payload.get("uploaded_files"))
        if not paths and not uploaded_items:
            return _error("没有找到可导入的视频文件")

        storage_mode = "copied" if str(payload.get("storage_mode") or "").lower() == "copied" else "linked"

        imported = []
        errors = []
        import_items = [(path, path.name, storage_mode, "") for path in paths]
        import_items.extend((path, original_name, "copied", "upload_cache:auto_cleanup") for path, original_name in uploaded_items)
        for path, original_name, item_storage_mode, item_notes in import_items:
            try:
                info = {"path": str(path), "original_name": original_name, "source_path": str(path)}
                _update_video_entry_info(info)
                video = project_store.add_video(
                    project_id,
                    source_path=str(path),
                    original_name=original_name,
                    video_info=info,
                    storage_mode=item_storage_mode,
                    notes=item_notes,
                )
                if item_notes and callable(_discard_uploaded_staging_file):
                    _discard_uploaded_staging_file(path)
                imported.append(video)
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})

        if not imported:
            return jsonify({"success": False, "imported": imported, "errors": errors, "error": "没有成功导入的视频"}), 400
        return jsonify({
            "success": True,
            "imported": imported,
            "errors": errors,
            "videos": project_store.list_videos(project_id),
            "all_assignments": project_store.list_assignments(project_id),
        })

    @app.route("/api/projects/<project_id>/videos/<video_id>", methods=["DELETE"])
    def api_delete_project_video(project_id, video_id):
        if not project_store.get_project(project_id):
            return _error("项目不存在", 404)
        try:
            deleted = project_store.delete_video(project_id, video_id)
        except Exception as exc:
            return _error(str(exc), 400)
        if not deleted:
            return _error("视频不存在", 404)
        return jsonify({
            "success": True,
            "deleted": deleted,
            "project": project_store.get_project(project_id),
            "videos": project_store.list_videos(project_id),
            "assignments": project_store.list_assignments(project_id),
            "runs": project_store.list_project_runs(project_id),
        })

    @app.route("/api/projects/<project_id>/assignments")
    def api_project_assignments(project_id):
        experiment_type = str(request.args.get("experiment_type") or "").strip()
        return jsonify({
            "success": True,
            "assignments": project_store.list_assignments(project_id, experiment_type=experiment_type or None),
        })

    @app.route("/api/projects/<project_id>/assignments", methods=["POST"])
    def api_create_project_assignment(project_id):
        payload = _json_payload()
        if "experiment_type" in payload:
            payload["experiment_type"] = normalize_experiment_type(payload.get("experiment_type"))
        try:
            assignment = project_store.create_assignment(project_id, payload)
        except Exception as exc:
            return _error(str(exc), 400)
        return jsonify({"success": True, "assignment": assignment, "assignments": project_store.list_assignments(project_id)})

    @app.route("/api/projects/<project_id>/assignments/<assignment_id>", methods=["PATCH"])
    def api_update_project_assignment(project_id, assignment_id):
        payload = _json_payload()
        if "experiment_type" in payload:
            try:
                assignment = project_store.update_assignment_experiment(
                    project_id,
                    assignment_id,
                    normalize_experiment_type(payload.get("experiment_type")),
                )
            except Exception as exc:
                return _error(str(exc), 400)
            if not assignment:
                return _error("视频实验记录不存在", 404)
            return jsonify({"success": True, "assignment": assignment, "assignments": project_store.list_assignments(project_id)})
        if "subject_id" not in payload:
            return _error("当前仅支持更新小鼠绑定")
        assignment = project_store.update_assignment_subject(project_id, assignment_id, payload.get("subject_id"))
        if not assignment:
            return _error("视频实验记录不存在", 404)
        return jsonify({"success": True, "assignment": assignment, "assignments": project_store.list_assignments(project_id)})

    @app.route("/api/projects/<project_id>/assignments/<assignment_id>", methods=["DELETE"])
    def api_delete_project_assignment(project_id, assignment_id):
        if not project_store.delete_assignment(project_id, assignment_id):
            return _error("视频实验记录不存在", 404)
        return jsonify({"success": True, "assignments": project_store.list_assignments(project_id)})

    @app.route("/api/projects/<project_id>/analysis/start", methods=["POST"])
    def api_start_project_analysis(project_id):
        project = project_store.get_project(project_id)
        if not project:
            return _error("项目不存在", 404)
        payload = _json_payload()
        assignment_ids = [str(item or "").strip() for item in payload.get("assignment_ids") or []]
        assignment_ids = [item for item in assignment_ids if item]
        if not assignment_ids:
            return _error("请至少选择一个视频实验记录")
        assignments = project_store.get_assignments(project_id, assignment_ids)
        if len(assignments) != len(set(assignment_ids)):
            return _error("部分视频实验记录不存在", 404)

        experiment_types = {item.get("experiment_type") for item in assignments}
        if len(experiment_types) != 1:
            return _error("一次项目分析只能包含同一种实验类型")
        experiment_type = normalize_experiment_type(assignments[0].get("experiment_type"))
        spec = get_experiment_spec(experiment_type)

        session_id = secrets.token_hex(16)
        video_paths = []
        video_infos = []
        video_assignment_map: dict[str, str] = {}
        unit_map: dict[str, list[dict]] = {}
        try:
            for index, assignment in enumerate(assignments):
                source = assignment.get("project_path") or assignment.get("source_path")
                if not source or not os.path.isfile(source):
                    return _error(f"视频文件不存在: {assignment.get('original_name') or source}", 400)
                staged_path, original_name = _stage_local_video_if_needed(
                    source,
                    session_id,
                    original_name=assignment.get("original_name") or Path(source).name,
                )
                info = {
                    "path": staged_path,
                    "original_name": original_name,
                    "source_path": assignment.get("source_path") or source,
                    "project_video_id": assignment.get("video_id"),
                    "assignment_id": assignment.get("assignment_id"),
                }
                _update_video_entry_info(info)
                video_paths.append(staged_path)
                video_infos.append(info)
                video_assignment_map[str(index)] = assignment["assignment_id"]
                unit_map[assignment["assignment_id"]] = project_store.list_units(project_id, assignment["assignment_id"])
        except Exception as exc:
            return _error(str(exc), 500)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(project["project_dir"]) / "results" / experiment_type / stamp)
        project_run_id = f"prun_{secrets.token_hex(8)}"
        project_context = {
            "project_id": project_id,
            "project_name": project.get("name"),
            "project_run_id": project_run_id,
            "assignment_ids": [item["assignment_id"] for item in assignments],
            "video_assignment_map": video_assignment_map,
            "unit_map": unit_map,
            "subject_candidates": project_store.subject_candidates(project_id),
        }
        analysis_sessions[session_id] = {
            "experiment_type": spec.key,
            "video_paths": video_paths,
            "video_infos": video_infos,
            "video_info": _extract_video_info_from_entry(video_infos[0]),
            "start_seconds": None,
            "end_seconds": None,
            "output_dir": output_dir,
            "fps_override": None,
            "is_batch": len(video_paths) > 1,
            "draft_payload": None,
            "draft_updated_at": None,
            "selected_video_idx": 0,
            "created_at": _now_ts(),
            "last_accessed_at": _now_ts(),
            "project_context": project_context,
        }
        project_store.upsert_project_run(
            project_id=project_id,
            project_run_id=project_run_id,
            run_id=session_id,
            session_id=session_id,
            experiment_type=spec.key,
            assignment_ids=project_context["assignment_ids"],
            output_dir=output_dir,
            status="configured",
            progress=0,
            message="Project analysis session created",
        )
        project_store.set_assignments_status(project_id, project_context["assignment_ids"], "configured", latest_run_id=session_id)

        redirect_url = url_for("batch_workspace", session_id=session_id) if len(video_paths) > 1 else url_for("calibrate", session_id=session_id)
        return jsonify({
            "success": True,
            "session_id": session_id,
            "project_run_id": project_run_id,
            "redirect_url": redirect_url,
        })

    @app.route("/api/projects/<project_id>/runs")
    def api_project_runs(project_id):
        _sync_project_runs_from_history(project_id)
        return jsonify({"success": True, "runs": project_store.list_project_runs(project_id)})

    @app.route("/api/projects/<project_id>/runs/<project_run_id>/open_folder", methods=["POST"])
    def api_open_project_run_folder(project_id, project_run_id):
        run = project_store.get_project_run(project_id, project_run_id)
        if not run:
            return _error("项目分析记录不存在", 404)
        output_dir = str(run.get("output_dir") or "").strip()
        if not output_dir or not os.path.isdir(output_dir):
            return _error("结果目录不存在或已移动", 404)
        try:
            _open_folder(output_dir)
        except Exception as exc:
            return _error(str(exc), 500)
        return jsonify({"success": True, "path": output_dir})
