"""Head/tail evidence must survive body-only morphology changes."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "core")]
from classical_mouse import CLASSICAL_METHOD, ClassicalMouseDetector
from barnes_maze import BarnesArena, BarnesTracker


def mouse_scene(light=False):
    background = np.full((160, 200), 25 if light else 210, np.uint8)
    gray = background.copy()
    cv2.ellipse(gray, (95, 82), (25, 17), 0, 0, 360, 235 if light else 20, -1)
    cv2.line(gray, (118, 82), (163, 82), 235 if light else 20, 4)
    foreground = cv2.absdiff(gray, background) > 25
    return gray, np.uint8(foreground) * 255, background


def config(**kwargs):
    return dict(
        classical_method=CLASSICAL_METHOD, min_area=60, max_area=10000, **kwargs
    )


def test_body_erosion_does_not_delete_head_tail_evidence():
    gray, fg, bg = mouse_scene()
    results = [
        ClassicalMouseDetector(
            config(body_erosion_kernel=k, body_erosion_iter=n)
        ).detect(gray, fg, background=bg)
        for k, n in [(3, 0), (3, 1), (5, 1)]
    ]
    assert (
        results[0]["body_area_px"]
        > results[1]["body_area_px"]
        > results[2]["body_area_px"]
    )
    for point in ["nose", "tail_base"]:
        assert point in results[0]["keypoints"], results[0]["reason"]
        assert all(
            r["keypoints"][point] == results[0]["keypoints"][point] for r in results
        )
    assert all(
        np.array_equal(r["landmark_mask"], results[0]["landmark_mask"]) for r in results
    )
    assert results[1]["landmark_evidence"] == "pre_body_erosion"


def test_shadow_is_rejected_without_expanding_the_mouse():
    gray, fg, bg = mouse_scene()
    cv2.ellipse(gray, (95, 58), (28, 9), 0, 0, 360, 130, -1)
    fg = np.uint8(cv2.absdiff(gray, bg) > 25) * 255
    result = ClassicalMouseDetector(config(body_erosion_iter=0)).detect(
        gray, fg, background=bg
    )
    assert result["body_mask"][58, 95] == 0
    assert result["body_mask"][82, 95] > 0
    assert result["local_gray_threshold"] < 130


def test_bright_mouse_on_dark_background():
    gray, fg, bg = mouse_scene(light=True)
    result = ClassicalMouseDetector(config(bg_type="dark", body_erosion_iter=0)).detect(
        gray, fg, background=bg
    )
    assert "body_center" in result["keypoints"]
    assert result["body_mask"][82, 95] > 0


def test_roi_mask_is_respected_after_cleanup():
    gray, fg, bg = mouse_scene()
    roi = np.zeros_like(fg)
    roi[:, 85:125] = 255
    result = ClassicalMouseDetector(config(body_erosion_iter=0)).detect(
        gray, fg, background=bg, detection_mask=roi
    )
    assert not np.any(result["body_mask"][roi == 0])
    assert not np.any(result["landmark_mask"][roi == 0])


def test_erosion_that_removes_body_does_not_fall_back():
    gray, fg, bg = mouse_scene()
    result = ClassicalMouseDetector(
        config(body_erosion_kernel=31, body_erosion_iter=2)
    ).detect(gray, fg, background=bg)
    assert result["keypoints"] == {}
    assert result["body_area_px"] == 0
    assert result["landmark_area_px"] > 0
    assert result["reason"] == "body_removed_by_erosion"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(segmentation_kernel=4),
        dict(body_erosion_kernel=0),
        dict(body_erosion_iter=1.5),
        dict(body_erosion_iter=-1),
        dict(classical_method="unknown"),
    ],
)
def test_invalid_method_settings_fail(kwargs):
    base = config()
    base.update(kwargs)
    with pytest.raises(ValueError):
        ClassicalMouseDetector(base)


def test_tracker_uses_the_shared_detector_and_offsets_coordinates():
    gray, fg, bg = mouse_scene()
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    points = [
        [100 + 70 * np.cos(t), 80 + 65 * np.sin(t)]
        for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)
    ]
    arena = BarnesArena(
        points,
        [dict(id="1", point=[145, 80]), dict(id="2", point=[55, 80])],
        "1",
        90,
        2,
    )
    cfg = config(body_erosion_iter=1)
    tracker = BarnesTracker(arena, classical_config=cfg)

    class FixedBackground:
        static_bg = bg

        def apply(self, frame, gray=None):
            return fg.copy()

    result = tracker.detect_landmarks(frame, gray, FixedBackground(), None, None)
    x, y, w, h = tracker.detection_bbox
    direct = ClassicalMouseDetector(cfg).detect(
        gray[y : y + h, x : x + w],
        fg[y : y + h, x : x + w],
        background=bg[y : y + h, x : x + w],
        detection_mask=tracker._cropped_detection_mask,
    )
    assert result["classical_method_version"] == CLASSICAL_METHOD
    assert np.array_equal(result["body_mask"], direct["body_mask"])
    for key, p in direct["keypoints"].items():
        assert result["keypoints"][key]["x"] == pytest.approx(p["x"] + x)
        assert result["keypoints"][key]["y"] == pytest.approx(p["y"] + y)


def test_new_barnes_config_records_version_and_requested_cutoffs():
    from services.barnes_config import build_barnes_config

    result = build_barnes_config({}, {})
    assert result["classical_method"] == CLASSICAL_METHOD
    assert result["classical_method_version"] == CLASSICAL_METHOD
    assert result["head_confidence_threshold"] == 0.4
    assert result["pose_confidence_threshold"] == 0.4
    assert result["body_erosion_kernel"] == 3 and result["body_erosion_iter"] == 1


def test_global_defaults_reject_even_cleanup_kernel():
    from services.experiment_defaults import validate_experiment_defaults

    errors = validate_experiment_defaults(dict(barnes_maze=dict(segmentation_kernel=4)))
    assert errors
