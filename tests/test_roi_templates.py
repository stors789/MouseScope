from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from flask import Flask
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from services.roi_templates import (
    RoiTemplateStore,
    RoiTemplateValidationError,
    materialize_template,
)
from routes.api import register_api_routes


def _maze_payload(assay: str, width: int = 640, height: int = 480) -> dict:
    if assay == "elevated_plus":
        zones = {
            key: [
                {"x": 64, "y": 48},
                {"x": 128, "y": 48},
                {"x": 128, "y": 96},
                {"x": 64, "y": 96},
            ]
            for key in ("center", "open_1", "open_2", "closed_1", "closed_2")
        }
        geometry = {
            "center_size_cm": 5,
            "open_arm_length_cm": 30,
            "closed_arm_length_cm": 31,
        }
    else:
        zones = {
            "center": [{"x": 64, "y": 48}, {"x": 128, "y": 48}, {"x": 96, "y": 96}],
            **{
                key: [
                    {"x": 64, "y": 48},
                    {"x": 128, "y": 48},
                    {"x": 128, "y": 96},
                    {"x": 64, "y": 96},
                ]
                for key in ("arm_a", "arm_b", "arm_c")
            },
        }
        geometry = {"arm_length_cm": 30, "arm_width_cm": 10}
    return {
        "source_frame": {"width": width, "height": height, "orientation_degrees": 0},
        "geometry": geometry,
        "rois": [
            {
                "name": "Apparatus 1",
                "zones": zones,
                "detection_points": [
                    {"x": 32, "y": 24},
                    {"x": 160, "y": 24},
                    {"x": 160, "y": 120},
                ],
                "subject_id": "mouse-7",
                "animal_code": "M007",
                "time_config": {"mode": "precise", "start": 3, "end": 8},
                "manual_events": [{"timestamp_s": 4, "arm": "arm_a"}],
                "gray_threshold": 77,
            }
        ],
    }


def test_store_isolates_assays_and_strips_runtime_fields(tmp_path: Path) -> None:
    store = RoiTemplateStore(tmp_path / "roi_templates.json")
    store.save_template("y_maze", "Device", _maze_payload("y_maze"))
    store.save_template("elevated_plus", "Device", _maze_payload("elevated_plus"))

    y_templates = store.list_templates("y_maze")
    epm_templates = store.list_templates("elevated_plus")
    assert len(y_templates) == len(epm_templates) == 1
    assert y_templates[0]["id"] != epm_templates[0]["id"]
    roi = y_templates[0]["rois"][0]
    assert roi["zones"]["center"][0] == {"x": pytest.approx(0.1), "y": pytest.approx(0.1)}
    for forbidden in (
        "subject_id",
        "animal_code",
        "time_config",
        "manual_events",
        "gray_threshold",
    ):
        assert forbidden not in roi


def test_materialize_scales_coordinates_and_reports_compatibility(tmp_path: Path) -> None:
    store = RoiTemplateStore(tmp_path / "roi_templates.json")
    saved = store.save_template("y_maze", "Device", _maze_payload("y_maze"))
    scaled = materialize_template(
        saved, {"width": 1280, "height": 960, "orientation_degrees": 0}
    )
    assert scaled["rois"][0]["zones"]["center"][0] == {"x": 128, "y": 96}
    assert scaled["warnings"] == []

    incompatible = materialize_template(
        saved, {"width": 1920, "height": 1080, "orientation_degrees": 90}
    )
    assert incompatible["warnings"] == [
        "orientation_mismatch",
        "aspect_ratio_mismatch",
    ]


def test_legacy_migration_preserves_pixels_without_inventing_frame_size(tmp_path: Path) -> None:
    legacy_path = tmp_path / "y_maze_roi_templates.json"
    generic_path = tmp_path / "roi_templates.json"
    legacy = {
        "version": 1,
        "templates": [
            {
                "id": "old-one",
                "name": "Old rig",
                "created_at": 1,
                "updated_at": 2,
                "geometry": {"arm_length_cm": 30},
                "rois": _maze_payload("y_maze")["rois"],
            }
        ],
    }
    legacy_path.write_text(json.dumps(legacy), encoding="utf-8")

    store = RoiTemplateStore(generic_path, legacy_y_maze_path=legacy_path)
    migrated = store.list_templates("y_maze")[0]
    assert migrated["source_frame"] == {"unknown": True}
    assert migrated["coordinate_space"] == "legacy_pixels"
    applied = materialize_template(
        migrated, {"width": 1920, "height": 1080, "orientation_degrees": 0}
    )
    assert applied["rois"][0]["zones"]["center"][0] == {"x": 64.0, "y": 48.0}
    assert applied["warnings"] == ["legacy_frame_unknown"]
    # Reopening is idempotent and does not alter the legacy source file.
    assert len(RoiTemplateStore(generic_path, legacy_path).list_templates("y_maze")) == 1
    assert json.loads(legacy_path.read_text(encoding="utf-8")) == legacy
    assert store.delete_template("y_maze", "old-one")
    assert RoiTemplateStore(generic_path, legacy_path).list_templates("y_maze") == []


def test_rejects_cross_assay_geometry_and_incomplete_zones(tmp_path: Path) -> None:
    store = RoiTemplateStore(tmp_path / "roi_templates.json")
    payload = _maze_payload("elevated_plus")
    payload["rois"][0]["zones"].pop("closed_2")
    with pytest.raises(RoiTemplateValidationError):
        store.save_template("elevated_plus", "Broken", payload)
    with pytest.raises(RoiTemplateValidationError):
        store.save_template("unknown", "Broken", payload)


def test_generic_api_lists_materializes_and_deletes_by_assay(tmp_path: Path) -> None:
    app = Flask(__name__)
    store = RoiTemplateStore(tmp_path / "roi_templates.json")
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
            roi_template_store=store,
        ),
    )
    client = app.test_client()
    created = client.post(
        "/api/roi_templates",
        json={
            "assay": "y_maze",
            "name": "Rig",
            **_maze_payload("y_maze"),
        },
    )
    assert created.status_code == 200
    template_id = created.get_json()["template"]["id"]
    listed = client.get(
        "/api/roi_templates?assay=y_maze&width=1280&height=960&orientation=0"
    ).get_json()["templates"]
    assert listed[0]["rois"][0]["zones"]["center"][0] == {"x": 128.0, "y": 96.0}
    assert client.get("/api/roi_templates?assay=elevated_plus").get_json()["templates"] == []
    assert client.delete(
        f"/api/roi_templates/{template_id}?assay=elevated_plus"
    ).status_code == 404
    assert client.delete(
        f"/api/roi_templates/{template_id}?assay=y_maze"
    ).status_code == 200


def test_all_calibration_assay_schemas_store_only_device_fields(tmp_path: Path) -> None:
    store = RoiTemplateStore(tmp_path / "roi_templates.json")
    frame = {"width": 640, "height": 480}
    quad = [
        {"x": 10, "y": 10}, {"x": 100, "y": 10},
        {"x": 100, "y": 100}, {"x": 10, "y": 100},
    ]
    polygon = quad[:3]
    cases = {
        "tst": ({"points": polygon}, {}),
        "forced_swim": (
            {"points": polygon, "diameter_points": quad[:2], "cylinder_diameter_cm": 10},
            {"cylinder_diameter_cm": 10},
        ),
        "water_maze": (
            {
                "pool": {"center": {"x": 200, "y": 180}, "radius_x_px": 150, "radius_y_px": 140},
                "platform": {"center": {"x": 240, "y": 200}, "radius_x_px": 12, "radius_y_px": 10},
                "pool_diameter_cm": 120,
                "platform_radius_cm": 5,
            },
            {"pool_diameter_cm": 120, "platform_radius_cm": 5},
        ),
        "open_field": ({"points": quad}, {"arena_side_cm": 50}),
        "novel_object": (
            {
                "points": quad,
                "objects": [
                    {"point": {"x": 30, "y": 30}, "role": "familiar"},
                    {"point": {"x": 80, "y": 80}, "role": "novel"},
                ],
            },
            {"arena_width_cm": 50, "arena_height_cm": 50, "interaction_radius_cm": 4},
        ),
        "light_dark_box": (
            {"zones": {"light": polygon, "dark": polygon, "doorway": polygon}},
            {"light_box_length_cm": 40, "light_box_width_cm": 20},
        ),
        "three_chamber": (
            {
                "zones": {"left": polygon, "center": polygon, "right": polygon},
                "objects": [
                    {"side": "left", "point": {"x": 30, "y": 30}},
                    {"side": "right", "point": {"x": 80, "y": 80}},
                ],
            },
            {"arena_width_cm": 60, "arena_height_cm": 40, "interaction_radius_cm": 2},
        ),
    }
    for assay, (roi, geometry) in cases.items():
        roi.update(
            {
                "name": "Rig",
                "detection_points": polygon,
                "subject_id": "mouse-x",
                "time_config": {"mode": "precise", "start": 1, "end": 2},
                "manual_events": [{"timestamp_s": 1.5}],
                "gray_threshold": 99,
            }
        )
        saved = store.save_template(
            assay,
            f"{assay} rig",
            {"source_frame": frame, "rois": [roi], "geometry": geometry},
        )
        stored_roi = saved["rois"][0]
        assert saved["assay"] == assay
        assert "subject_id" not in stored_roi
        assert "time_config" not in stored_roi
        assert "manual_events" not in stored_roi
        assert "gray_threshold" not in stored_roi


def test_every_roi_calibration_page_registers_the_shared_adapter() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "calibrate.html": "tst",
        "calibrate_forced_swim.html": "forced_swim",
        "calibrate_water_maze.html": "water_maze",
        "calibrate_openfield.html": "open_field",
        "calibrate_nor.html": "novel_object",
        "calibrate_light_dark.html": "light_dark_box",
        "calibrate_three_chamber.html": "three_chamber",
    }
    for filename, assay in expected.items():
        source = (root / "templates" / filename).read_text(encoding="utf-8")
        assert "MouseScopeRoiTemplateAdapter" in source
        assert f"assay:'{assay}'" in source or f"assay: '{assay}'" in source
        assert "getPayload:" in source
        assert "applyPayload:" in source
        assert "getFrameMeta:" in source
    shared = (root / "static/js/roi_template_controls.js").read_text(encoding="utf-8")
    assert "/api/roi_templates" in shared
    assert "Apply to every video in this batch" in shared
    assert "subject" not in shared.lower()
