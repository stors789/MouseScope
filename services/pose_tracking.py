from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import cv2
import numpy as np

from services.deeplabcut_runtime import (
    DeepLabCutRuntime,
    DeepLabCutRuntimeError,
    resolve_tracking_backend,
)
from services.pose_model_profiles import DEFAULT_KEYPOINT_SCHEMA, SUPERANIMAL_MODEL_REF


def resolve_pose_runtime_config(
    config: dict[str, Any],
    assay: str,
    *,
    profile_store=None,
    runtime_status: dict[str, Any] | None = None,
) -> None:
    """Resolve requested backends and immutable model refs before snapshotting."""
    if assay not in {"open_field", "novel_object"}:
        return
    status = runtime_status or DeepLabCutRuntime().status()

    def resolve_one(target: dict[str, Any]) -> None:
        requested = str(target.get("tracking_backend") or "classical").strip().lower()
        model_ready = True
        model_reference = SUPERANIMAL_MODEL_REF
        keypoint_schema = DEFAULT_KEYPOINT_SCHEMA
        profile_error = None
        profile_id = str(target.get("pose_model_profile_id") or "").strip()
        version_id = str(target.get("pose_model_version_id") or "").strip() or None
        if profile_id:
            if profile_store is None:
                model_ready = False
                profile_error = "pose_store_missing"
            else:
                try:
                    profile = profile_store.get_profile(
                        profile_id,
                        version_id=version_id,
                        include_annotations=False,
                    )
                    if assay not in (profile.get("assays") or []):
                        raise KeyError(profile_id)
                    version = (profile.get("versions") or [])[-1]
                    training_status = str((version.get("training") or {}).get("status") or "")
                    weights = version.get("weights") or {}
                    model_ready = training_status in {"pretrained", "ready"} and weights.get("kind") != "pending"
                    model_reference = str(weights.get("reference") or model_reference)
                    keypoint_schema = version.get("keypoint_schema") or keypoint_schema
                    target["pose_model_version_id"] = version.get("version_id")
                    target["pose_model_version_number"] = version.get("version_number")
                    target["pose_model_weights_checksum"] = copy.deepcopy(weights.get("checksum"))
                    if not model_ready:
                        profile_error = "pose_model_training_incomplete"
                except KeyError:
                    model_ready = False
                    profile_error = "pose_model_version_missing"
        resolution = resolve_tracking_backend(
            requested,
            runtime_status=status,
            model_ready=model_ready,
        )
        target["tracking_backend"] = resolution["requested"]
        target["tracking_backend_effective"] = resolution["effective"]
        target["tracking_fallback_reason"] = profile_error or resolution["fallback_reason"]
        target["pose_model_reference"] = model_reference
        target["pose_keypoint_schema"] = copy.deepcopy(keypoint_schema)
        target["pose_runtime_code"] = status.get("code")

    resolve_one(config)
    for item in config.get("per_video_configs") or []:
        if isinstance(item, dict):
            resolve_one(item)


def _normalise_point(value: Any) -> dict[str, Any] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        value = {
            "x": value[0],
            "y": value[1],
            "confidence": value[2] if len(value) > 2 else 1.0,
        }
    if not isinstance(value, dict):
        return None
    try:
        x, y = float(value["x"]), float(value["y"])
        confidence = float(value.get("confidence", value.get("likelihood", value.get("score", 1.0))))
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(confidence):
        return None
    return {
        "x": x,
        "y": y,
        "confidence": max(0.0, min(1.0, confidence)),
        "visible": bool(value.get("visible", True)),
    }


def normalise_pose_output(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_individuals = payload.get("individuals")
    if not isinstance(raw_individuals, list):
        raw_individuals = [{"keypoints": payload.get("keypoints")}]
    observations: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_individuals):
        if not isinstance(raw, dict) or not isinstance(raw.get("keypoints"), dict):
            continue
        points = {
            str(name): point
            for name, value in raw["keypoints"].items()
            if (point := _normalise_point(value)) is not None
        }
        if not points:
            continue
        observations.append(
            {
                "individual_id": str(raw.get("individual_id") or index),
                "keypoints": points,
                "source": "deeplabcut",
            }
        )
    return observations


def _tracker_polygon(tracker) -> np.ndarray:
    points = getattr(tracker, "detection_points", None)
    if points is None:
        points = getattr(tracker, "arena_points", None)
    return np.asarray(points, dtype=np.float32)


@dataclass
class PoseFrameResult:
    by_roi_id: dict[int, dict[str, Any]]
    error: str | None = None


class PoseInferenceCoordinator:
    """One DLC inference per frame, then assign individuals to calibrated ROIs."""

    def __init__(
        self,
        config: dict[str, Any],
        trackers: list[Any],
        *,
        runtime: DeepLabCutRuntime | None = None,
    ) -> None:
        self.config = config
        self.trackers = list(trackers)
        self.runtime = runtime or DeepLabCutRuntime()
        self.enabled = str(config.get("tracking_backend_effective") or "classical") == "deeplabcut"
        self.confidence_threshold = float(config.get("pose_confidence_threshold", 0.8))
        self.model_reference = str(config.get("pose_model_reference") or SUPERANIMAL_MODEL_REF)
        self.keypoint_schema = config.get("pose_keypoint_schema") or DEFAULT_KEYPOINT_SCHEMA
        self.frames_requested = 0
        self.frames_succeeded = 0
        self.frames_failed = 0
        self.consecutive_worker_errors = 0
        self.disabled_reason: str | None = None

    def infer(self, frame) -> PoseFrameResult:
        if not self.enabled or self.disabled_reason:
            return PoseFrameResult({}, error=self.disabled_reason)
        self.frames_requested += 1
        try:
            payload = self.runtime.infer_frame(
                frame,
                model_reference=self.model_reference,
                keypoint_schema=self.keypoint_schema,
                max_individuals=max(1, len(self.trackers)),
            )
        except DeepLabCutRuntimeError as exc:
            self.frames_failed += 1
            self.consecutive_worker_errors += 1
            if self.consecutive_worker_errors >= 3:
                self.disabled_reason = f"DLC worker disabled after repeated errors: {exc}"
            return PoseFrameResult({}, error=self.disabled_reason or str(exc))
        self.consecutive_worker_errors = 0
        observations = normalise_pose_output(payload)
        assigned: dict[int, dict[str, Any]] = {}
        used: set[int] = set()
        for tracker in self.trackers:
            polygon = _tracker_polygon(tracker)
            candidates: list[tuple[float, int, dict[str, Any]]] = []
            for index, observation in enumerate(observations):
                if index in used:
                    continue
                body = observation["keypoints"].get("body_center")
                if not body or not body.get("visible") or body["confidence"] < self.confidence_threshold:
                    continue
                inside = cv2.pointPolygonTest(
                    polygon,
                    (float(body["x"]), float(body["y"])),
                    False,
                )
                if inside >= 0:
                    candidates.append((float(body["confidence"]), index, observation))
            if candidates:
                _score, index, observation = max(candidates, key=lambda item: item[0])
                used.add(index)
                observation = dict(observation)
                observation["confidence_threshold"] = self.confidence_threshold
                assigned[int(tracker.roi_id)] = observation
        if assigned:
            self.frames_succeeded += 1
        else:
            self.frames_failed += 1
        return PoseFrameResult(assigned)

    def close(self) -> None:
        self.runtime.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
