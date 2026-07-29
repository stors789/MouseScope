from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from services.nor_pose_cache import PersistentNorPoseCache, cache_fingerprint
from services.novel_object_session import NovelObjectMultiArenaAnalysisSession
from services.open_field_session import OpenFieldMultiArenaAnalysisSession
from services.pose_tracking import PoseFrameResult


def test_nor_annotated_render_uses_pose_cache_without_dlc_coordinator() -> None:
    source = inspect.getsource(
        NovelObjectMultiArenaAnalysisSession._render_annotated_video
    )
    assert "pose_cache.get" in source
    assert "PoseInferenceCoordinator(" not in source
    assert ".infer(" not in source
    assert ".infer_many(" not in source


def test_oft_annotated_render_uses_pose_cache_without_dlc_coordinator() -> None:
    source = inspect.getsource(
        OpenFieldMultiArenaAnalysisSession._render_annotated_video
    )
    assert "pose_cache.get" in source
    assert "PoseInferenceCoordinator(" not in source
    assert ".infer(" not in source
    assert ".infer_many(" not in source


def test_persistent_nor_pose_cache_reuses_only_matching_crop(tmp_path: Path) -> None:
    payload = {"video": {"path": "video.mp4", "size": 42}, "model": "v1"}
    observation = {
        "source": "deeplabcut",
        "keypoints": {
            "body_center": {
                "x": 12.0,
                "y": 18.0,
                "confidence": 0.99,
                "visible": True,
            }
        },
    }
    first = PersistentNorPoseCache(tmp_path, payload)
    first.put(15, 2, (4, 5, 30, 40), observation)
    first.close()

    second = PersistentNorPoseCache(tmp_path, payload)
    cached = second.get(15, 2, (4, 5, 30, 40))
    assert cached is not None
    assert cached.observation == observation
    assert second.hits == 1
    assert second.get(15, 2, (5, 5, 30, 40)) is None
    assert second.misses == 1
    second.close()


def test_persistent_nor_pose_cache_fingerprint_tracks_configuration() -> None:
    base = {"video": {"path": "video.mp4", "size": 42}, "model": "v1"}
    changed = {**base, "model": "v2"}
    assert cache_fingerprint(base) == cache_fingerprint(dict(base))
    assert cache_fingerprint(base) != cache_fingerprint(changed)


def test_hybrid_batch_skips_dlc_after_persistent_cache_hit(tmp_path: Path) -> None:
    detection = {
        "centroid": (20.0, 20.0),
        "bbox": (10, 10, 20, 20),
        "contour": np.array(
            [[[10, 10]], [[30, 10]], [[30, 30]], [[10, 30]]],
            dtype=np.int32,
        ),
    }
    observation = {
        "source": "deeplabcut",
        "confidence_threshold": 0.8,
        "keypoints": {
            "body_center": {
                "x": 20.0,
                "y": 20.0,
                "confidence": 0.99,
                "visible": True,
            }
        },
    }

    class Tracker:
        roi_id = 1
        interaction_radius_cm = 4.0

        def detect_mouse(self, *_args):
            return detection, np.zeros((40, 40), dtype=np.uint8)

        def project_point_to_cm(self, point):
            return point

        def _object_distances_cm(self, *_args):
            return [1.0, 20.0]

    class Coordinator:
        def __init__(self):
            self.calls = 0

        def infer_crops(self, frames, *, roi_ids, offsets, valid_sizes):
            self.calls += 1
            assert len(frames) == len(roi_ids) == len(offsets) == 1
            assert valid_sizes == [(40, 40)]
            assert frames[0].shape[:2] == (64, 64)
            return [PoseFrameResult({1: observation})]

    session = object.__new__(NovelObjectMultiArenaAnalysisSession)
    batch = [{
        "source_frame_no": 12,
        "frame": np.zeros((60, 60, 3), dtype=np.uint8),
        "gray": np.zeros((60, 60), dtype=np.uint8),
    }]
    config = {
        "pose_trigger_margin_cm": 2.0,
        "pose_crop_padding_ratio": 0.5,
    }
    first_cache = PersistentNorPoseCache(tmp_path, {"video": "same"})
    first_coordinator = Coordinator()
    first_results, _detections, _triggered = session._infer_hybrid_batch(
        batch,
        [Tracker()],
        first_coordinator,
        object(),
        None,
        None,
        config,
        first_cache,
    )
    assert first_coordinator.calls == 1
    assert first_results[0].by_roi_id[1] == observation
    first_cache.close()

    second_cache = PersistentNorPoseCache(tmp_path, {"video": "same"})
    second_coordinator = Coordinator()
    second_results, _detections, _triggered = session._infer_hybrid_batch(
        batch,
        [Tracker()],
        second_coordinator,
        object(),
        None,
        None,
        config,
        second_cache,
    )
    assert second_coordinator.calls == 0
    assert second_results[0].by_roi_id[1] == observation
    assert second_cache.hits == 1
    second_cache.close()


def test_hybrid_crop_batch_is_padded_without_rescaling() -> None:
    requests = [
        {"frame": np.full((17, 11, 3), 32, dtype=np.uint8)},
        {"frame": np.full((29, 23, 3), 196, dtype=np.uint8)},
    ]
    padded, valid_sizes = (
        NovelObjectMultiArenaAnalysisSession._pad_hybrid_crop_batch(requests)
    )

    assert valid_sizes == [(11, 17), (23, 29)]
    assert {frame.shape for frame in padded} == {(32, 32, 3)}
    assert np.array_equal(padded[0][:17, :11], requests[0]["frame"])
    assert np.array_equal(padded[1][:29, :23], requests[1]["frame"])
