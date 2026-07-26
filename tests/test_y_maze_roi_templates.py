from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from flask import Flask

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from services.y_maze_roi_templates import (
    RoiTemplateValidationError,
    YMazeRoiTemplateStore,
)
from routes.api import register_api_routes


def _payload() -> dict:
    return {
        "geometry": {"arm_length_cm": 31, "arm_width_cm": 9.5},
        "rois": [
            {
                "name": "Right",
                "zones": {
                    "center": [{"x": 10, "y": 10}, {"x": 20, "y": 10}, {"x": 15, "y": 20}],
                    "arm_a": [{"x": 10, "y": 10}, {"x": 20, "y": 10}, {"x": 20, "y": 0}, {"x": 10, "y": 0}],
                    "arm_b": [{"x": 20, "y": 10}, {"x": 30, "y": 10}, {"x": 30, "y": 20}, {"x": 20, "y": 20}],
                    "arm_c": [{"x": 0, "y": 10}, {"x": 10, "y": 10}, {"x": 10, "y": 20}, {"x": 0, "y": 20}],
                },
                "detection_points": [
                    {"x": 0, "y": 0},
                    {"x": 30, "y": 0},
                    {"x": 30, "y": 20},
                    {"x": 0, "y": 20},
                ],
                # Per-run fields must not leak into reusable device templates.
                "subject_id": "mouse-1",
                "scoring_mode": "manual",
                "manual_events": [{"timestamp_s": 2, "arm": "arm_a"}],
                "time_config": {"mode": "precise", "start": 1, "end": 9},
            }
        ],
    }


def test_store_persists_geometry_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "roi_templates.json"
    saved = YMazeRoiTemplateStore(path).save_template("  Right   device  ", _payload())

    loaded = YMazeRoiTemplateStore(path).list_templates()
    assert len(loaded) == 1
    assert loaded[0]["id"] == saved["id"]
    assert loaded[0]["name"] == "Right device"
    assert loaded[0]["geometry"] == {"arm_length_cm": 31.0, "arm_width_cm": 9.5}
    assert loaded[0]["rois"][0]["detection_points"][-1] == {"x": 0.0, "y": 20.0}
    assert "subject_id" not in loaded[0]["rois"][0]
    assert "manual_events" not in loaded[0]["rois"][0]


def test_same_name_updates_and_delete_is_persistent(tmp_path: Path) -> None:
    path = tmp_path / "roi_templates.json"
    store = YMazeRoiTemplateStore(path)
    first = store.save_template("Device 1", _payload())
    changed = _payload()
    changed["geometry"]["arm_width_cm"] = 12
    second = store.save_template("device 1", changed)

    assert first["id"] == second["id"]
    assert len(store.list_templates()) == 1
    assert store.list_templates()[0]["geometry"]["arm_width_cm"] == 12
    assert store.delete_template(first["id"]) is True
    assert YMazeRoiTemplateStore(path).list_templates() == []
    assert store.delete_template(first["id"]) is False


def test_store_rejects_incomplete_or_invalid_roi(tmp_path: Path) -> None:
    store = YMazeRoiTemplateStore(tmp_path / "roi_templates.json")
    payload = _payload()
    payload["rois"][0]["zones"]["center"] = [{"x": 1, "y": 2}]

    with pytest.raises(RoiTemplateValidationError):
        store.save_template("Incomplete", payload)
    with pytest.raises(RoiTemplateValidationError):
        store.save_template("", _payload())


def test_template_api_lists_saves_and_deletes(tmp_path: Path) -> None:
    app = Flask(__name__)
    store = YMazeRoiTemplateStore(tmp_path / "roi_templates.json")
    noop = lambda *args, **kwargs: None
    register_api_routes(
        app,
        SimpleNamespace(
            analysis_sessions={},
            analysis_history={},
            RESULT_DIR=str(tmp_path),
            DATA_DIR=tmp_path,
            _resolve_result_dir=noop,
            _result_zip_arcname=noop,
            _session_video_info=noop,
            _update_video_entry_info=noop,
            y_maze_roi_template_store=store,
        ),
    )
    client = app.test_client()

    created_response = client.post(
        "/api/y_maze_roi_templates",
        json={"name": "Device", **_payload()},
    )
    assert created_response.status_code == 200
    created = created_response.get_json()["template"]
    assert client.get("/api/y_maze_roi_templates").get_json()["templates"][0]["id"] == created["id"]
    assert client.delete(f"/api/y_maze_roi_templates/{created['id']}").status_code == 200
    assert client.get("/api/y_maze_roi_templates").get_json()["templates"] == []

    invalid = client.post("/api/y_maze_roi_templates", json={"name": "", "rois": []})
    assert invalid.status_code == 400


def test_y_maze_calibration_exposes_persistent_template_controls() -> None:
    template = (
        ROOT / "templates" / "calibrate_maze.html"
    ).read_text(encoding="utf-8")

    assert 'id="roiTemplatePanel"' in template
    assert "fetch('/api/y_maze_roi_templates'" in template
    assert "saveRoiTemplate()" in template
    assert "applyRoiTemplate()" in template
    assert "deleteRoiTemplate()" in template
    assert "applyTemplateToAll" in template
    assert "manual_events:[]" in template
