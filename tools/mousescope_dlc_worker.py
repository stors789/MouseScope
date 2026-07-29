#!/usr/bin/env python3
"""Optional DeepLabCut 3 sidecar for MouseScope.

This file intentionally lives outside the desktop application's dependency
graph. Run it with a Python environment that has the PyTorch build of
DeepLabCut installed:

    .venv-dlc/bin/python tools/mousescope_dlc_worker.py status
    .venv-dlc/bin/python tools/mousescope_dlc_worker.py preview request.json
    .venv-dlc/bin/python tools/mousescope_dlc_worker.py serve

``serve`` is a newline-delimited JSON protocol. It keeps the official DLC
detector and pose runners resident so formal video analysis does not reload
hundreds of megabytes of weights for every sampled frame.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata
import importlib.util
import gc
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, TextIO
from urllib.parse import parse_qs, urlparse


PROTOCOL_VERSION = 1
DEFAULT_MODEL_REFERENCE = "deeplabcut://superanimal_topviewmouse/hrnet_w32"
DEFAULT_DETECTOR = "fasterrcnn_resnet50_fpn_v2"
CANONICAL_KEYPOINTS = ("nose", "left_ear", "right_ear", "body_center", "tail_base")
KEYPOINT_ALIASES = {
    "nose": ("nose", "snout"),
    "left_ear": ("left_ear", "leftear"),
    "right_ear": ("right_ear", "rightear"),
    # The official SuperAnimal-TopViewMouse vocabulary calls this mouse_center.
    "body_center": ("mouse_center", "body_center", "centroid", "center", "mid_back"),
    "tail_base": ("tail_base", "tailbase"),
}


class WorkerError(RuntimeError):
    pass


def worker_status() -> dict[str, Any]:
    try:
        available = importlib.util.find_spec("deeplabcut") is not None
    except (ImportError, ValueError):
        available = False
    version = None
    if available:
        try:
            version = importlib.metadata.version("deeplabcut")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return {
        "available": available,
        "protocol_version": PROTOCOL_VERSION,
        "worker": "mousescope-superanimal-topviewmouse",
        "deeplabcut_version": version,
        "supported_operations": ["status", "preview", "serve"],
        "persistent": True,
        "message": (
            "DeepLabCut PyTorch worker is ready."
            if available
            else "DeepLabCut is not installed in this worker environment."
        ),
    }


def parse_model_reference(reference: Any) -> dict[str, Any]:
    value = str(reference or DEFAULT_MODEL_REFERENCE).strip()
    parsed = urlparse(value)
    if parsed.scheme != "deeplabcut":
        raise WorkerError(
            "The bundled worker accepts deeplabcut:// SuperAnimal references. "
            "Use a custom MOUSESCOPE_DLC_WORKER launcher for project-specific artifacts."
        )
    superanimal_name = parsed.netloc.strip()
    model_name = parsed.path.strip("/")
    if not superanimal_name or not model_name:
        raise WorkerError(f"Invalid DeepLabCut model reference: {value}")
    if superanimal_name != "superanimal_topviewmouse":
        raise WorkerError("MouseScope's bundled worker only supports superanimal_topviewmouse.")
    query = parse_qs(parsed.query)
    detector_name = str((query.get("detector") or [DEFAULT_DETECTOR])[0]).strip()
    device = str((query.get("device") or [os.environ.get("MOUSESCOPE_DLC_DEVICE", "")])[0]).strip() or None
    return {
        "reference": value,
        "superanimal_name": superanimal_name,
        "model_name": model_name,
        "detector_name": detector_name,
        "device": device,
    }


def _finite_point(row: Any) -> dict[str, float] | None:
    try:
        x, y, likelihood = float(row[0]), float(row[1]), float(row[2])
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, likelihood)):
        return None
    return {
        "x": x,
        "y": y,
        "likelihood": max(0.0, min(1.0, likelihood)),
    }


def _canonical_points(
    pose: Any,
    bodyparts: list[str],
    requested_names: set[str],
) -> dict[str, dict[str, float]]:
    indexed: dict[str, dict[str, float]] = {}
    for index, name in enumerate(bodyparts):
        try:
            point = _finite_point(pose[index])
        except (IndexError, TypeError):
            point = None
        if point is not None:
            indexed[str(name).strip().lower()] = point

    result: dict[str, dict[str, float]] = {}
    for canonical in CANONICAL_KEYPOINTS:
        if canonical not in requested_names:
            continue
        for alias in KEYPOINT_ALIASES[canonical]:
            point = indexed.get(alias)
            if point is not None:
                result[canonical] = point
                break

    # Older/custom TopViewMouse configs may omit mouse_center. Derive a stable
    # center from paired lateral torso landmarks before falling back to mid-back.
    if "body_center" in requested_names and "body_center" not in result:
        pairs = (
            ("left_midside", "right_midside"),
            ("left_hip", "right_hip"),
            ("left_shoulder", "right_shoulder"),
        )
        for left_name, right_name in pairs:
            left, right = indexed.get(left_name), indexed.get(right_name)
            if left is None or right is None:
                continue
            result["body_center"] = {
                "x": (left["x"] + right["x"]) / 2.0,
                "y": (left["y"] + right["y"]) / 2.0,
                "likelihood": min(left["likelihood"], right["likelihood"]),
            }
            break
    return result


def format_predictions(
    prediction: dict[str, Any],
    *,
    bodyparts: list[str],
    keypoint_schema: Any,
    model_reference: str,
    api_backend: str,
) -> dict[str, Any]:
    schema = keypoint_schema if isinstance(keypoint_schema, list) else []
    requested_names = {
        str(item.get("name") if isinstance(item, dict) else item).strip().lower()
        for item in schema
    }
    requested_names &= set(CANONICAL_KEYPOINTS)
    if not requested_names:
        requested_names = set(CANONICAL_KEYPOINTS)

    raw_poses = prediction.get("bodyparts") if isinstance(prediction, dict) else None
    if raw_poses is None:
        raise WorkerError("DeepLabCut prediction is missing bodyparts.")
    individuals = []
    for index, pose in enumerate(raw_poses):
        points = _canonical_points(pose, bodyparts, requested_names)
        if not points:
            continue
        individual: dict[str, Any] = {
            "individual_id": f"animal{index}",
            "keypoints": points,
        }
        bboxes = prediction.get("bboxes")
        if bboxes is not None:
            try:
                bbox = [float(value) for value in bboxes[index][:4]]
            except (IndexError, TypeError, ValueError):
                bbox = None
            if bbox is not None:
                individual["bbox"] = bbox
        bbox_scores = prediction.get("bbox_scores")
        if bbox_scores is not None:
            try:
                individual["bbox_score"] = float(bbox_scores[index])
            except (IndexError, TypeError, ValueError):
                pass
        individuals.append(individual)

    payload: dict[str, Any] = {
        "success": True,
        "model_reference": model_reference,
        "api_backend": api_backend,
        "individuals": individuals,
    }
    # Keep the original single-animal preview contract as a convenience.
    payload["keypoints"] = individuals[0]["keypoints"] if individuals else {}
    return payload


def _load_official_api() -> dict[str, Any]:
    """Load the same PyTorch runners used by DLC's official image API."""
    try:
        import deeplabcut.pose_estimation_pytorch.modelzoo as modelzoo
        from deeplabcut.pose_estimation_pytorch.apis.utils import (
            get_detector_inference_runner,
            get_pose_inference_runner,
        )
        from deeplabcut.pose_estimation_pytorch.modelzoo.utils import update_config
    except Exception as exc:
        raise WorkerError(f"Unable to import the DeepLabCut PyTorch API: {exc}") from exc
    return {
        "modelzoo": modelzoo,
        "get_detector_inference_runner": get_detector_inference_runner,
        "get_pose_inference_runner": get_pose_inference_runner,
        "update_config": update_config,
    }


class OfficialSuperAnimalPredictor:
    """Resident official DLC detector/pose runners for one model/max-count tuple."""

    def __init__(
        self,
        spec: dict[str, Any],
        max_individuals: int,
        batch_size: int = 1,
        *,
        api_loader: Callable[[], dict[str, Any]] = _load_official_api,
    ) -> None:
        self.spec = dict(spec)
        self.max_individuals = max(1, int(max_individuals))
        self.batch_size = max(1, int(batch_size))
        self.api = api_loader()
        self.api_backend = "resident_official_runners"
        self.config: dict[str, Any] | None = None
        self.bodyparts: list[str] = []
        self.pose_runner = None
        self.detector_runner = None
        self._initialize()

    def _initialize(self) -> None:
        modelzoo = self.api["modelzoo"]
        config = modelzoo.load_super_animal_config(
            super_animal=self.spec["superanimal_name"],
            model_name=self.spec["model_name"],
            detector_name=self.spec["detector_name"],
        )
        config = self.api["update_config"](
            copy.deepcopy(config),
            self.max_individuals,
            self.spec.get("device"),
        )
        config["metadata"]["individuals"] = [
            f"animal{index}" for index in range(self.max_individuals)
        ]
        self.bodyparts = [str(name) for name in config["metadata"]["bodyparts"]]
        pose_path = modelzoo.get_super_animal_snapshot_path(
            dataset=self.spec["superanimal_name"],
            model_name=self.spec["model_name"],
        )
        detector_path = modelzoo.get_super_animal_snapshot_path(
            dataset=self.spec["superanimal_name"],
            model_name=self.spec["detector_name"],
        )
        self.pose_runner = self.api["get_pose_inference_runner"](
            model_config=config,
            snapshot_path=pose_path,
            batch_size=self.batch_size,
            device=self.spec.get("device"),
            max_individuals=self.max_individuals,
            cond_provider=None,
        )
        self.detector_runner = self.api["get_detector_inference_runner"](
            model_config=config,
            snapshot_path=detector_path,
            batch_size=self.batch_size,
            device=self.spec.get("device"),
            max_individuals=self.max_individuals,
        )
        self.config = config

    def predict(self, image_path: str) -> dict[str, Any]:
        return self.predict_many([image_path])[0]

    def predict_many(self, image_paths: list[str]) -> list[dict[str, Any]]:
        if self.pose_runner is None or self.detector_runner is None:
            raise WorkerError("DeepLabCut runners were not initialized.")
        paths = [str(path) for path in image_paths]
        bbox_predictions = list(self.detector_runner.inference(images=paths))
        if len(bbox_predictions) != len(paths):
            raise WorkerError(
                "DeepLabCut detector returned a different number of results than inputs."
            )
        detected_indices = [
            index
            for index, prediction in enumerate(bbox_predictions)
            if isinstance(prediction, dict)
            and len(
                prediction.get("bboxes")
                if prediction.get("bboxes") is not None
                else prediction.get("boxes", [])
            )
            > 0
        ]
        detected_predictions = list(
            self.pose_runner.inference(
                [
                    (paths[index], bbox_predictions[index])
                    for index in detected_indices
                ]
            )
        )
        if len(detected_predictions) != len(detected_indices):
            raise WorkerError(
                "DeepLabCut pose runner returned "
                f"{len(detected_predictions)} results for "
                f"{len(detected_indices)} detected inputs."
            )
        predictions = [{"bodyparts": []} for _path in paths]
        for index, prediction in zip(
            detected_indices,
            detected_predictions,
            strict=True,
        ):
            predictions[index] = prediction
        return [
            prediction if isinstance(prediction, dict) else {"bodyparts": []}
            for prediction in predictions
        ]


class HighLevelOfficialPredictor:
    """Compatibility fallback using DLC's public synchronous image API.

    Current DLC 3 exposes resident runner constructors and uses them internally.
    If a future/older DLC release does not expose those constructors with the
    expected signature, this remains a real (but slower) inference path.
    """

    def __init__(
        self,
        spec: dict[str, Any],
        max_individuals: int,
        batch_size: int = 1,
    ) -> None:
        try:
            import deeplabcut.pose_estimation_pytorch.modelzoo as modelzoo
            from deeplabcut.pose_estimation_pytorch.apis import (
                superanimal_analyze_images,
            )
        except Exception as exc:
            raise WorkerError(f"Unable to import the official SuperAnimal API: {exc}") from exc
        self.spec = dict(spec)
        self.max_individuals = max(1, int(max_individuals))
        self.batch_size = max(1, int(batch_size))
        self.superanimal_analyze_images = superanimal_analyze_images
        config = modelzoo.load_super_animal_config(
            super_animal=self.spec["superanimal_name"],
            model_name=self.spec["model_name"],
            detector_name=self.spec["detector_name"],
        )
        self.bodyparts = [str(name) for name in config["metadata"]["bodyparts"]]
        self.api_backend = "official_superanimal_analyze_images"

    def predict(self, image_path: str) -> dict[str, Any]:
        return self.predict_many([image_path])[0]

    def predict_many(self, image_paths: list[str]) -> list[dict[str, Any]]:
        paths = [str(path) for path in image_paths]
        with tempfile.TemporaryDirectory(prefix="mousescope-dlc-output-") as out_folder:
            predictions = self.superanimal_analyze_images(
                self.spec["superanimal_name"],
                self.spec["model_name"],
                self.spec["detector_name"],
                paths,
                self.max_individuals,
                out_folder=out_folder,
                progress_bar=False,
                device=self.spec.get("device"),
                plot_skeleton=False,
                close_figure_after_save=True,
            )
        if not isinstance(predictions, dict):
            raise WorkerError("The official SuperAnimal API returned an invalid result.")
        return [
            prediction if isinstance(prediction, dict) else {"bodyparts": []}
            for path in paths
            for prediction in [predictions.get(path)]
        ]


def create_official_predictor(
    spec: dict[str, Any],
    max_individuals: int,
    batch_size: int = 1,
):
    try:
        return OfficialSuperAnimalPredictor(spec, max_individuals, batch_size)
    except (WorkerError, ImportError, AttributeError, TypeError) as exc:
        print(
            f"Resident DLC runner API unavailable ({exc}); using slower public image API.",
            file=sys.stderr,
        )
        return HighLevelOfficialPredictor(spec, max_individuals, batch_size)


class PredictorRegistry:
    def __init__(
        self,
        *,
        predictor_factory: Callable[[dict[str, Any], int], Any] = create_official_predictor,
    ) -> None:
        self.predictor_factory = predictor_factory
        self._predictors: dict[tuple[Any, ...], Any] = {}

    def get(
        self,
        spec: dict[str, Any],
        max_individuals: int,
        batch_size: int = 1,
    ):
        normalized_batch_size = max(1, min(16, int(batch_size)))
        key = (
            spec["superanimal_name"],
            spec["model_name"],
            spec["detector_name"],
            spec.get("device"),
            max(1, int(max_individuals)),
            normalized_batch_size,
        )
        if key not in self._predictors:
            model_key = key[:-1]
            stale_keys = [
                cached_key
                for cached_key in self._predictors
                if cached_key[:-1] == model_key
            ]
            for stale_key in stale_keys:
                self._predictors.pop(stale_key, None)
            if stale_keys:
                gc.collect()
                _clear_cuda_cache()
            try:
                inspect.signature(self.predictor_factory).bind(
                    spec,
                    key[-2],
                    normalized_batch_size,
                )
                accepts_batch_size = True
            except TypeError:
                accepts_batch_size = False
            if accepts_batch_size:
                self._predictors[key] = self.predictor_factory(
                    spec,
                    key[-2],
                    normalized_batch_size,
                )
            else:
                self._predictors[key] = self.predictor_factory(spec, key[-2])
        return self._predictors[key]

    def discard(
        self,
        spec: dict[str, Any],
        max_individuals: int,
        batch_size: int,
    ) -> None:
        key = (
            spec["superanimal_name"],
            spec["model_name"],
            spec["detector_name"],
            spec.get("device"),
            max(1, int(max_individuals)),
            max(1, min(16, int(batch_size))),
        )
        removed = self._predictors.pop(key, None)
        if removed is not None:
            del removed
            gc.collect()


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _batch_steps(requested: int) -> list[int]:
    requested = max(1, min(16, int(requested)))
    return [size for size in (16, 8, 4, 2, 1) if size <= requested]


def _format_prediction(
    prediction: dict[str, Any],
    *,
    predictor,
    request: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    return format_predictions(
        prediction,
        bodyparts=predictor.bodyparts,
        keypoint_schema=request.get("keypoint_schema"),
        model_reference=spec["reference"],
        api_backend=predictor.api_backend,
    )


def _predict_batch_with_fallback(
    image_paths: list[Path],
    *,
    request: dict[str, Any],
    registry: PredictorRegistry,
    spec: dict[str, Any],
    max_individuals: int,
) -> tuple[list[dict[str, Any]], int]:
    requested_batch_size = max(
        1,
        min(16, int(request.get("batch_size") or len(image_paths) or 1)),
    )
    last_error: BaseException | None = None
    for batch_size in _batch_steps(requested_batch_size):
        predictor = registry.get(spec, max_individuals, batch_size)
        try:
            predictions = predictor.predict_many([str(path) for path in image_paths])
            return (
                [
                    _format_prediction(
                        prediction,
                        predictor=predictor,
                        request=request,
                        spec=spec,
                    )
                    for prediction in predictions
                ],
                batch_size,
            )
        except Exception as exc:
            if not _is_cuda_oom(exc) or batch_size == 1:
                raise
            last_error = exc
            registry.discard(spec, max_individuals, batch_size)
            _clear_cuda_cache()
    raise WorkerError(f"DeepLabCut batch inference failed: {last_error}")


def handle_request(
    request: Any,
    *,
    registry: PredictorRegistry,
) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise WorkerError("Request must be a JSON object.")
    operation = str(request.get("operation") or "preview").strip().lower()
    if operation not in {"preview", "infer", "infer_batch"}:
        raise WorkerError(f"Unsupported operation: {operation}")
    spec = parse_model_reference(request.get("model_reference"))
    max_individuals = max(1, int(request.get("max_individuals") or 1))
    if operation == "infer_batch":
        raw_paths = request.get("image_paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            raise WorkerError("infer_batch requires a non-empty image_paths array.")
        image_paths = [Path(str(path)).expanduser() for path in raw_paths]
        missing = next((path for path in image_paths if not path.is_file()), None)
        if missing is not None:
            raise WorkerError(f"Image does not exist: {missing}")
        with contextlib.redirect_stdout(sys.stderr):
            items, effective_batch_size = _predict_batch_with_fallback(
                image_paths,
                request=request,
                registry=registry,
                spec=spec,
                max_individuals=max_individuals,
            )
        payload = {
            "success": True,
            "items": items,
            "effective_batch_size": effective_batch_size,
        }
        if request.get("request_id") is not None:
            payload["request_id"] = request["request_id"]
        return payload

    image_path = Path(str(request.get("image_path") or "")).expanduser()
    if not image_path.is_file():
        raise WorkerError(f"Image does not exist: {image_path}")
    # DLC/tqdm may print diagnostics. Protocol JSON must remain the only stdout.
    with contextlib.redirect_stdout(sys.stderr):
        predictor = registry.get(spec, max_individuals, 1)
        prediction = predictor.predict(str(image_path))
    payload = _format_prediction(
        prediction,
        predictor=predictor,
        request=request,
        spec=spec,
    )
    if request.get("request_id") is not None:
        payload["request_id"] = request["request_id"]
    return payload


def _write_json(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def serve(
    stdin: TextIO,
    stdout: TextIO,
    *,
    registry: PredictorRegistry | None = None,
) -> int:
    registry = registry or PredictorRegistry()
    for line in stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            request = json.loads(line)
            request_id = request.get("request_id") if isinstance(request, dict) else None
            payload = handle_request(request, registry=registry)
        except Exception as exc:
            payload = {
                "success": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            if request_id is not None:
                payload["request_id"] = request_id
        _write_json(stdout, payload)
    return 0


def main(
    argv: list[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    registry: PredictorRegistry | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="MouseScope DeepLabCut sidecar")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    preview_parser = subparsers.add_parser("preview")
    preview_parser.add_argument("request_json")
    subparsers.add_parser("serve")
    args = parser.parse_args(argv)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    if args.command == "status":
        _write_json(stdout, worker_status())
        return 0
    if args.command == "serve":
        return serve(stdin, stdout, registry=registry)

    request_path = Path(args.request_json)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        payload = handle_request(request, registry=registry or PredictorRegistry())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _write_json(stdout, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
