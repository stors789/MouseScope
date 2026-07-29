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

from routes.api import register_api_routes
from services.analysis_builders import build_novel_object_config, build_open_field_config
from services.deeplabcut_runtime import (
    DeepLabCutRuntime,
    DeepLabCutRuntimeError,
    resolve_tracking_backend,
)
from services.pose_model_profiles import PoseModelProfileStore, PoseModelValidationError
from services.pose_tracking import PoseInferenceCoordinator, resolve_pose_runtime_config
from core.open_field import OpenFieldTracker
from core.novel_object import NovelObjectTracker
import numpy as np


def _annotation(frame: int, *, split: str = "train") -> dict:
    return {
        "video_ref": "session-a:mouse-01.mp4",
        "frame_index": frame,
        "frame_size": {"width": 640, "height": 480},
        "split": split,
        "keypoints": {
            "nose": {"x": 110 + frame, "y": 80, "visible": True},
            "body_center": {"x": 90 + frame, "y": 100, "visible": True},
            "left_ear": {"x": 101 + frame, "y": 90, "visible": True},
        },
    }


def test_pose_versions_are_immutable_and_inherit_history_and_validation(tmp_path: Path) -> None:
    store = PoseModelProfileStore(tmp_path / "pose_model_profiles.json")
    profile = store.create_profile(
        name="Animal room A",
        assays=["open_field", "novel_object"],
        metadata={"mouse_color": "black", "ignored": "not persisted"},
    )
    v1 = profile["versions"][0]

    v2 = store.create_version(
        profile["profile_id"],
        parent_version=v1["version_id"],
        annotations=[_annotation(1), _annotation(2, split="validation")],
    )
    v3 = store.create_version(
        profile["profile_id"],
        parent_version=v2["version_id"],
        annotations=[_annotation(3)],
    )

    stored = store.get_profile(profile["profile_id"], include_annotations=True)
    assert [item["version_number"] for item in stored["versions"]] == [1, 2, 3]
    assert stored["versions"][0] == v1
    assert v2["parent_version"] == v1["version_id"]
    assert v3["parent_version"] == v2["version_id"]
    assert len(v2["annotations"]) == 2
    assert len(v3["annotations"]) == 3
    assert v3["fixed_validation_annotation_ids"] == v2["fixed_validation_annotation_ids"]
    assert stored["metadata"] == {"mouse_color": "black"}
    assert v2["training"]["status"] == "awaiting_external_training"


def test_ready_custom_version_requires_weight_checksum(tmp_path: Path) -> None:
    store = PoseModelProfileStore(tmp_path / "pose_model_profiles.json")
    profile = store.create_profile(name="Lab model", assays=["open_field"])
    with pytest.raises(PoseModelValidationError, match="SHA-256"):
        store.create_version(
            profile["profile_id"],
            parent_version=profile["latest_version_id"],
            annotations=[_annotation(1)],
            training_status="ready",
            weight_reference={"kind": "local_file", "reference": "weights.pt"},
        )

    checksum = "a" * 64
    version = store.create_version(
        profile["profile_id"],
        parent_version=profile["latest_version_id"],
        annotations=[_annotation(2)],
        training_status="ready",
        weight_reference={
            "kind": "external_artifact",
            "reference": "artifact://lab/v2",
            "sha256": checksum,
        },
    )
    assert version["weights"]["checksum"] == {
        "algorithm": "sha256",
        "value": checksum,
        "status": "recorded",
    }

    registered = store.create_version(
        profile["profile_id"],
        parent_version=version["version_id"],
        annotations=[],
        training_status="ready",
        weight_reference={
            "kind": "external_artifact",
            "reference": "artifact://lab/v3",
            "sha256": "b" * 64,
        },
    )
    assert registered["parent_version"] == version["version_id"]
    assert registered["annotations"] == version["annotations"]


def test_tracking_backend_is_separate_from_background_method() -> None:
    sess = {"video_infos": [{}], "start_seconds": None, "end_seconds": None}
    data = {
        "tracking_backend": "auto",
        "pose_model_profile_id": "profile-a",
        "pose_model_version_id": "version-2",
        "pose_confidence_threshold": 0.91,
        "bg_method": "static",
        "per_video_configs": [
            {
                "tracking_backend": "deeplabcut",
                "pose_model_profile_id": "profile-b",
                "pose_model_version_id": "version-3",
                "bg_method": "threshold",
            }
        ],
    }
    oft = build_open_field_config(sess, data)
    nor = build_novel_object_config(sess, data)
    for config in (oft, nor):
        assert config["tracking_backend"] == "auto"
        assert config["bg_method"] == "static"
        assert config["per_video_configs"][0]["tracking_backend"] == "deeplabcut"
        assert config["per_video_configs"][0]["bg_method"] == "threshold"
        assert config["per_video_configs"][0]["pose_model_profile_id"] == "profile-b"


def _api_app(tmp_path: Path, sessions: dict | None = None) -> Flask:
    app = Flask(__name__)
    noop = lambda *args, **kwargs: None
    register_api_routes(
        app,
        SimpleNamespace(
            analysis_sessions=sessions or {},
            analysis_history={},
            RESULT_DIR=str(tmp_path),
            DATA_DIR=tmp_path,
            _resolve_result_dir=noop,
            _result_zip_arcname=noop,
            _session_video_info=noop,
            _update_video_entry_info=noop,
            pose_model_profile_store=PoseModelProfileStore(tmp_path / "profiles.json"),
        ),
    )
    return app


def test_pose_profile_api_creates_archive_and_child_version(tmp_path: Path) -> None:
    client = _api_app(tmp_path).test_client()
    created = client.post(
        "/api/pose_models",
        json={"name": "NOR room A", "assays": ["novel_object"]},
    )
    assert created.status_code == 200
    profile = created.get_json()["profile"]

    version_response = client.post(
        f"/api/pose_models/{profile['profile_id']}/versions",
        json={
            "parent_version": profile["latest_version_id"],
            "annotations": [_annotation(10, split="validation")],
        },
    )
    assert version_response.status_code == 200
    child = version_response.get_json()["version"]
    assert child["version_number"] == 2
    assert child["parent_version"] == profile["latest_version_id"]

    listed = client.get("/api/pose_models?assay=novel_object").get_json()["profiles"]
    assert listed[0]["latest_version_id"] == child["version_id"]
    assert listed[0]["versions"][-1]["validation_count"] == 1
    assert "annotations" not in listed[0]["versions"][-1]


def test_unavailable_dlc_preview_falls_back_without_reading_video(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MOUSESCOPE_DLC_WORKER", raising=False)
    monkeypatch.delenv("MOUSESCOPE_DLC_PYTHON", raising=False)
    monkeypatch.setattr(
        "services.deeplabcut_runtime.DeepLabCutRuntime._python_candidates",
        staticmethod(lambda: []),
    )
    video = tmp_path / "placeholder.mp4"
    video.write_bytes(b"")
    sessions = {
        "session-a": {
            "experiment_type": "open_field",
            "video_paths": [str(video)],
            "draft_payload": {},
        }
    }
    client = _api_app(tmp_path, sessions).test_client()
    payload = client.post(
        "/api/pose_preview/session-a",
        json={"video": 0, "frame": 5},
    ).get_json()
    # CI may have the importable DLC package, but no isolated worker is configured.
    assert payload["success"] is False
    assert payload["fallback"] is True
    assert payload["effective_backend"] == "classical"


def test_runtime_resolution_is_explicit_about_fallback() -> None:
    unavailable = {"available": False, "code": "dependency_missing"}
    assert resolve_tracking_backend("classical", runtime_status=unavailable)["fallback_reason"] is None
    resolved = resolve_tracking_backend("auto", runtime_status=unavailable, model_ready=True)
    assert resolved == {
        "requested": "auto",
        "effective": "classical",
        "fallback_reason": "dependency_missing",
    }
    assert DeepLabCutRuntime(command="").status()["available"] is False


def test_frozen_runtime_discovers_dlc_venv_beside_executable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executable = tmp_path / "MouseScope.exe"
    monkeypatch.delenv("MOUSESCOPE_DLC_PYTHON", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    candidates = DeepLabCutRuntime._python_candidates()
    assert (
        tmp_path / ".venv-dlc" / "Scripts" / "python.exe",
        "application_dlc_venv",
    ) in candidates
    assert all("E:\\MouseScope" not in str(path) for path, _origin in candidates)


def test_oft_and_nor_pages_expose_pose_archive_controls() -> None:
    for filename in ("calibrate_openfield.html", "calibrate_nor.html"):
        text = (ROOT / "templates" / filename).read_text(encoding="utf-8")
        assert '_pose_model_controls.html' in text
        assert "pose_model_controls.js" in text
        assert "tracking_backend" in text


def _pose(body=(50, 50), nose=(65, 50), confidence=0.98) -> dict:
    return {
        "source": "deeplabcut",
        "confidence_threshold": 0.8,
        "keypoints": {
            "body_center": {"x": body[0], "y": body[1], "confidence": confidence, "visible": True},
            "nose": {"x": nose[0], "y": nose[1], "confidence": confidence, "visible": True},
            "left_ear": {"x": nose[0] - 7, "y": nose[1] - 4, "confidence": confidence, "visible": True},
            "right_ear": {"x": nose[0] - 7, "y": nose[1] + 4, "confidence": confidence, "visible": True},
        },
    }


def test_oft_pose_body_center_drives_formal_tracking() -> None:
    tracker = OpenFieldTracker(
        arena_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        arena_side_cm=50,
        fps=10,
    )
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    gray = np.zeros((120, 120), dtype=np.uint8)
    first = tracker.update(
        frame, 0.0, gray, None, None, None, dt_s=0.1, pose_observation=_pose(body=(20, 50)),
    )
    second = tracker.update(
        frame, 0.1, gray, None, None, None, dt_s=0.1, pose_observation=_pose(body=(40, 50)),
    )
    assert first["centroid"] == (20.0, 50.0)
    assert second["tracking_source"] == "deeplabcut"
    assert second["record"]["tracking_source"] == "deeplabcut"
    assert tracker.total_distance_cm == pytest.approx(10.0)
    assert tracker.build_summary()["deeplabcut_frames"] == 2


def test_nor_pose_nose_and_ears_drive_head_gated_exploration() -> None:
    tracker = NovelObjectTracker(
        arena_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        object_points=[
            {"point": [90, 50], "name": "Novel", "role": "novel"},
            {"point": [10, 10], "name": "Familiar", "role": "familiar"},
        ],
        arena_side_cm=50,
        fps=10,
        interaction_radius_cm=4,
        head_angle_threshold_deg=40,
        head_confidence_threshold=0.8,
    )
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    gray = np.zeros((120, 120), dtype=np.uint8)
    result = tracker.update(
        frame,
        0.0,
        gray,
        None,
        None,
        None,
        dt_s=0.1,
        pose_observation=_pose(body=(70, 50), nose=(88, 50)),
    )
    assert result["tracking_source"] == "deeplabcut"
    assert result["head_direction_method"] == "deeplabcut_ears"
    assert result["current_object_idx"] == 0
    assert result["exploration_mode"] == "head"
    assert result["record"]["object_1_nose_distance_cm"] == pytest.approx(1.0)
    assert tracker.object_exploration_time_s[0] == pytest.approx(0.1)


def test_strict_dlc_does_not_run_classical_detection_when_pose_is_missing(
    monkeypatch,
) -> None:
    tracker = OpenFieldTracker(
        arena_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        arena_side_cm=50,
        fps=10,
    )
    monkeypatch.setattr(
        tracker,
        "detect_mouse",
        lambda *_args, **_kwargs: pytest.fail("classical detector was called"),
    )
    result = tracker.update(
        np.zeros((120, 120, 3), dtype=np.uint8),
        0.0,
        np.zeros((120, 120), dtype=np.uint8),
        None,
        None,
        None,
        pose_observation=None,
        tracking_source="deeplabcut",
        allow_classical_fallback=False,
        fallback_reason="pose_low_confidence_or_unassigned",
    )
    assert result["detected"] is False
    assert result["tracking_source"] == "deeplabcut_missing"
    assert tracker.build_summary()["deeplabcut_missing_frames"] == 1
    assert tracker.build_summary()["classical_fallback_frames"] == 0


def test_auto_mode_still_runs_classical_detection_when_pose_is_missing(
    monkeypatch,
) -> None:
    tracker = OpenFieldTracker(
        arena_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        arena_side_cm=50,
        fps=10,
    )
    monkeypatch.setattr(
        tracker,
        "detect_mouse",
        lambda *_args, **_kwargs: (
            {"centroid": (30.0, 40.0), "bbox": None, "contour": None},
            np.zeros((120, 120), dtype=np.uint8),
        ),
    )
    result = tracker.update(
        np.zeros((120, 120, 3), dtype=np.uint8),
        0.0,
        np.zeros((120, 120), dtype=np.uint8),
        None,
        None,
        None,
        pose_observation=None,
        tracking_source="classical_fallback",
        allow_classical_fallback=True,
    )
    assert result["detected"] is True
    assert result["tracking_source"] == "classical_fallback"


def test_strict_nor_does_not_use_grayscale_head_estimation(
    monkeypatch,
) -> None:
    tracker = NovelObjectTracker(
        arena_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        object_points=[
            {"point": [90, 50], "name": "Novel", "role": "novel"},
            {"point": [10, 10], "name": "Familiar", "role": "familiar"},
        ],
        arena_side_cm=50,
        fps=10,
    )
    pose = _pose(body=(50, 50), nose=(65, 50))
    pose["keypoints"]["nose"]["confidence"] = 0.1
    pose["keypoints"]["left_ear"]["confidence"] = 0.1
    pose["keypoints"]["right_ear"]["confidence"] = 0.1
    monkeypatch.setattr(
        tracker,
        "_estimate_head_direction",
        lambda *_args, **_kwargs: pytest.fail("grayscale head estimator was called"),
    )
    result = tracker.update(
        np.zeros((120, 120, 3), dtype=np.uint8),
        0.0,
        np.zeros((120, 120), dtype=np.uint8),
        None,
        None,
        None,
        pose_observation=pose,
        tracking_source="deeplabcut",
        allow_classical_fallback=False,
    )
    assert result["tracking_source"] == "deeplabcut"
    assert result["head_direction_method"] is None
    assert result["head_point"] is None


def test_pose_coordinator_runs_once_and_assigns_individuals_to_rois() -> None:
    trackers = [
        OpenFieldTracker(
            arena_points=[[0, 0], [49, 0], [49, 100], [0, 100]],
            arena_side_cm=50,
            roi_id=1,
        ),
        OpenFieldTracker(
            arena_points=[[51, 0], [100, 0], [100, 100], [51, 100]],
            arena_side_cm=50,
            roi_id=2,
        ),
    ]

    class Runtime:
        calls = 0

        def infer_frame(self, *_args, **_kwargs):
            self.calls += 1
            return {
                "individuals": [
                    {"individual_id": "right", "keypoints": _pose(body=(75, 50))["keypoints"]},
                    {"individual_id": "left", "keypoints": _pose(body=(25, 50))["keypoints"]},
                ]
            }

    runtime = Runtime()
    coordinator = PoseInferenceCoordinator(
        {
            "tracking_backend_effective": "deeplabcut",
            "pose_confidence_threshold": 0.8,
        },
        trackers,
        runtime=runtime,
    )
    result = coordinator.infer(np.zeros((120, 120, 3), dtype=np.uint8))
    assert runtime.calls == 1
    assert result.by_roi_id[1]["individual_id"] == "left"
    assert result.by_roi_id[2]["individual_id"] == "right"
    assert coordinator.frames_succeeded == 1


def test_pose_coordinator_batch_preserves_frame_order_and_roi_assignment() -> None:
    trackers = [
        OpenFieldTracker(
            arena_points=[[0, 0], [49, 0], [49, 100], [0, 100]],
            arena_side_cm=50,
            roi_id=1,
        ),
        OpenFieldTracker(
            arena_points=[[51, 0], [100, 0], [100, 100], [51, 100]],
            arena_side_cm=50,
            roi_id=2,
        ),
    ]

    class Runtime:
        effective_batch_size = 2

        def infer_frames(self, frames, **_kwargs):
            assert len(frames) == 2
            return [
                {
                    "individuals": [
                        {"individual_id": "first", "keypoints": _pose(body=(25, 50))["keypoints"]}
                    ]
                },
                {
                    "individuals": [
                        {"individual_id": "second", "keypoints": _pose(body=(75, 50))["keypoints"]}
                    ]
                },
            ]

        def close(self):
            pass

    coordinator = PoseInferenceCoordinator(
        {
            "tracking_backend_effective": "deeplabcut",
            "pose_confidence_threshold": 0.8,
            "pose_batch_size": 2,
        },
        trackers,
        runtime=Runtime(),
    )
    results = coordinator.infer_many(
        [np.zeros((8, 8, 3), dtype=np.uint8) for _index in range(2)]
    )
    assert len(results) == 2
    assert results[0].by_roi_id[1]["individual_id"] == "first"
    assert results[1].by_roi_id[2]["individual_id"] == "second"
    assert coordinator.effective_batch_size == 2


def test_runtime_config_activates_ready_profile_and_rejects_pending_child(tmp_path: Path) -> None:
    store = PoseModelProfileStore(tmp_path / "profiles.json")
    profile = store.create_profile(name="Runtime model", assays=["open_field"])
    ready_config = {
        "tracking_backend": "deeplabcut",
        "pose_model_profile_id": profile["profile_id"],
        "pose_model_version_id": profile["latest_version_id"],
        "per_video_configs": [],
    }
    available = {"available": True, "code": "external_worker_configured"}
    resolve_pose_runtime_config(
        ready_config,
        "open_field",
        profile_store=store,
        runtime_status=available,
    )
    assert ready_config["tracking_backend_effective"] == "deeplabcut"
    assert ready_config["tracking_fallback_reason"] is None
    assert ready_config["pose_model_reference"].startswith("deeplabcut://")

    child = store.create_version(
        profile["profile_id"],
        parent_version=profile["latest_version_id"],
        annotations=[_annotation(20)],
    )
    pending_config = {
        "tracking_backend": "auto",
        "pose_model_profile_id": profile["profile_id"],
        "pose_model_version_id": child["version_id"],
    }
    resolve_pose_runtime_config(
        pending_config,
        "open_field",
        profile_store=store,
        runtime_status=available,
    )
    assert pending_config["tracking_backend_effective"] == "classical"
    assert pending_config["tracking_fallback_reason"] == "pose_model_training_incomplete"


def test_pose_coordinator_circuit_breaks_repeated_worker_errors() -> None:
    tracker = OpenFieldTracker(
        arena_points=[[0, 0], [100, 0], [100, 100], [0, 100]],
        arena_side_cm=50,
    )

    class BrokenRuntime:
        calls = 0

        def infer_frame(self, *_args, **_kwargs):
            self.calls += 1
            raise DeepLabCutRuntimeError("worker offline")

    runtime = BrokenRuntime()
    coordinator = PoseInferenceCoordinator(
        {"tracking_backend_effective": "deeplabcut"},
        [tracker],
        runtime=runtime,
    )
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    results = [coordinator.infer(frame) for _ in range(4)]
    assert runtime.calls == 3
    assert coordinator.disabled_reason is not None
    assert all(result.by_roi_id == {} for result in results)
    assert "repeated errors" in results[-1].error
