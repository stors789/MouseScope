import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "core")]
from tools.validate_landmarks import evaluate_labels


def fixtures():
    labels = dict(
        dataset_id="held-out",
        annotation_origin="human",
        coordinate_space="crop_pixels",
        records=[
            dict(
                id="a",
                frame_index=10,
                image_sha256="aaa",
                keypoints=dict(nose=dict(x=10, y=20, visible=True)),
            ),
            dict(
                id="b",
                frame_index=20,
                image_sha256="bbb",
                keypoints=dict(nose=dict(x=10, y=20, visible=True)),
            ),
        ],
    )
    predictions = dict(
        dataset_id="held-out",
        backend="classical",
        coordinate_space="crop_pixels",
        records=[
            dict(
                id="a",
                image_sha256="aaa",
                keypoints=dict(nose=dict(x=10, y=20, visible=True)),
            ),
            dict(id="b", image_sha256="bbb", keypoints={}),
        ],
    )
    return labels, predictions


def test_missing_prediction_counts_as_a_miss_in_pck():
    labels, predictions = fixtures()
    summary, details = evaluate_labels(labels, predictions, 5)
    assert summary[0]["coverage_percent"] == 50
    assert summary[0]["within_tolerance_percent"] == 50
    assert summary[0]["median_error_px"] == 0
    assert len(details) == 2 and details[1]["error_px"] is None


def test_model_output_cannot_be_used_as_independent_labels():
    labels, predictions = fixtures()
    labels["annotation_origin"] = "prediction"
    with pytest.raises(ValueError, match="独立人工标注"):
        evaluate_labels(labels, predictions, 5)


def test_different_frames_and_datasets_are_rejected():
    labels, predictions = fixtures()
    predictions["dataset_id"] = "another-dataset"
    with pytest.raises(ValueError, match="同一数据集"):
        evaluate_labels(labels, predictions, 5)
    labels, predictions = fixtures()
    predictions["records"][0]["image_sha256"] = "changed-frame"
    with pytest.raises(ValueError, match="画面不匹配"):
        evaluate_labels(labels, predictions, 5)


def test_tolerance_changes_success_without_changing_coverage():
    labels, predictions = fixtures()
    predictions["records"][0]["keypoints"]["nose"]["x"] = 16
    loose, _ = evaluate_labels(labels, predictions, 7)
    strict, _ = evaluate_labels(labels, predictions, 5)
    assert loose[0]["within_tolerance_percent"] == 50
    assert strict[0]["within_tolerance_percent"] == 0
    assert loose[0]["coverage_percent"] == strict[0]["coverage_percent"] == 50
