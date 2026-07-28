from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.deeplabcut_runtime import DeepLabCutRuntime
from tools.mousescope_dlc_worker import (
    OfficialSuperAnimalPredictor,
    PredictorRegistry,
    format_predictions,
    handle_request,
    main,
    parse_model_reference,
)


BODY_PARTS = [
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "mid_back",
    "mouse_center",
    "tail_base",
]


def _pose(offset: float = 0.0):
    return [
        [10 + offset, 11, 0.99],
        [8 + offset, 12, 0.95],
        [12 + offset, 12, 0.94],
        [10 + offset, 15, 0.91],
        [10 + offset, 20, 0.92],
        [10 + offset, 24, 0.98],
        [10 + offset, 35, 0.93],
    ]


def test_worker_maps_official_topviewmouse_vocabulary() -> None:
    payload = format_predictions(
        {
            "bodyparts": [_pose(), _pose(100)],
            "bboxes": [[1, 2, 20, 30], [101, 2, 20, 30]],
            "bbox_scores": [0.9, 0.8],
        },
        bodyparts=BODY_PARTS,
        keypoint_schema=[
            {"name": "nose"},
            {"name": "left_ear"},
            {"name": "right_ear"},
            {"name": "body_center"},
            {"name": "tail_base"},
        ],
        model_reference="deeplabcut://superanimal_topviewmouse/hrnet_w32",
        api_backend="test",
    )
    assert len(payload["individuals"]) == 2
    first = payload["individuals"][0]["keypoints"]
    assert set(first) == {
        "nose",
        "left_ear",
        "right_ear",
        "body_center",
        "tail_base",
    }
    assert first["body_center"] == {"x": 10.0, "y": 24.0, "likelihood": 0.98}
    assert payload["keypoints"] == first


def test_official_predictor_initializes_runners_once_and_reuses_them() -> None:
    events: list[tuple[str, object]] = []

    class FakeModelZoo:
        @staticmethod
        def load_super_animal_config(**kwargs):
            events.append(("config", kwargs))
            return {
                "metadata": {"bodyparts": BODY_PARTS, "individuals": ["animal"]},
                "detector": {"model": {}},
            }

        @staticmethod
        def get_super_animal_snapshot_path(*, dataset, model_name):
            events.append(("snapshot", model_name))
            return f"/weights/{dataset}/{model_name}.pt"

    class DetectorRunner:
        def inference(self, *, images):
            events.append(("detect", tuple(images)))
            return [{"boxes": [[0, 0, 20, 20]]}]

    class PoseRunner:
        def inference(self, inputs):
            events.append(("pose", tuple(path for path, _boxes in inputs)))
            return [{"bodyparts": [_pose()]}]

    def update_config(config, max_individuals, device):
        events.append(("update", (max_individuals, device)))
        return config

    api = {
        "modelzoo": FakeModelZoo,
        "superanimal_analyze_images": object(),
        "update_config": update_config,
        "get_detector_inference_runner": lambda **kwargs: DetectorRunner(),
        "get_pose_inference_runner": lambda **kwargs: PoseRunner(),
    }
    predictor = OfficialSuperAnimalPredictor(
        parse_model_reference(
            "deeplabcut://superanimal_topviewmouse/hrnet_w32?device=cpu"
        ),
        2,
        api_loader=lambda: api,
    )
    assert predictor.predict("/tmp/frame-1.png")["bodyparts"]
    assert predictor.predict("/tmp/frame-2.png")["bodyparts"]
    assert len([event for event in events if event[0] == "config"]) == 1
    assert len([event for event in events if event[0] == "detect"]) == 2
    assert len([event for event in events if event[0] == "pose"]) == 2


def test_preview_cli_contract_without_dlc_install(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"fake image; predictor is monkeypatched")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "operation": "preview",
                "image_path": str(image_path),
                "model_reference": "deeplabcut://superanimal_topviewmouse/hrnet_w32",
                "keypoint_schema": [{"name": "nose"}, {"name": "body_center"}],
                "max_individuals": 1,
            }
        ),
        encoding="utf-8",
    )

    class FakePredictor:
        bodyparts = BODY_PARTS
        api_backend = "fake-test"

        def predict(self, image_path):
            return {"bodyparts": [_pose()]}

    registry = PredictorRegistry(
        predictor_factory=lambda _spec, _count: FakePredictor()
    )
    stdout = io.StringIO()
    assert main(
        ["preview", str(request_path)],
        stdout=stdout,
        registry=registry,
    ) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["success"] is True
    assert payload["api_backend"] == "fake-test"
    assert payload["keypoints"]["nose"]["likelihood"] == 0.99
    assert payload["keypoints"]["body_center"]["x"] == 10.0


def test_handle_request_rejects_non_superanimal_reference(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"x")
    registry = PredictorRegistry(predictor_factory=lambda *_args: None)
    try:
        handle_request(
            {
                "image_path": str(image_path),
                "model_reference": "artifact://custom/model.pt",
            },
            registry=registry,
        )
    except Exception as exc:
        assert "deeplabcut://" in str(exc)
    else:
        raise AssertionError("custom artifact unexpectedly accepted by bundled worker")


def test_runtime_status_and_persistent_preview_contract(tmp_path: Path) -> None:
    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        """
import json
import sys

if sys.argv[1] == "status":
    print(json.dumps({
        "available": True,
        "supported_operations": ["status", "preview", "serve"],
        "persistent": True,
        "protocol_version": 1
    }))
    raise SystemExit(0)
if sys.argv[1] == "serve":
    for line in sys.stdin:
        request = json.loads(line)
        print(json.dumps({
            "success": True,
            "request_id": request["request_id"],
            "keypoints": {
                "nose": {"x": 6, "y": 7, "likelihood": 0.97},
                "body_center": {"x": 4, "y": 5, "likelihood": 0.99}
            },
            "individuals": [{
                "individual_id": "animal0",
                "keypoints": {
                    "nose": {"x": 6, "y": 7, "likelihood": 0.97},
                    "body_center": {"x": 4, "y": 5, "likelihood": 0.99}
                }
            }]
        }), flush=True)
    raise SystemExit(0)
raise SystemExit(2)
""".lstrip(),
        encoding="utf-8",
    )
    runtime = DeepLabCutRuntime(
        command=f"{sys.executable} {fake_worker}",
        timeout_seconds=5,
    )
    status = runtime.status()
    assert status["available"] is True
    assert status["persistent"] is True
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    first = runtime.preview_frame(
        frame,
        model_reference="deeplabcut://superanimal_topviewmouse/hrnet_w32",
        keypoint_schema=[{"name": "nose"}, {"name": "body_center"}],
    )
    second = runtime.preview_frame(
        frame,
        model_reference="deeplabcut://superanimal_topviewmouse/hrnet_w32",
        keypoint_schema=[{"name": "nose"}, {"name": "body_center"}],
    )
    assert first["keypoints"]["body_center"]["likelihood"] == 0.99
    assert second["individuals"][0]["keypoints"]["nose"]["x"] == 6
    process = runtime._process
    assert process is not None and process.poll() is None
    runtime.close()
    assert process.poll() is not None
