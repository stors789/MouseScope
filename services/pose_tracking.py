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
    shared_deeplabcut_runtime,
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
        if requested == "hybrid" and assay != "novel_object":
            requested = "classical"
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
        self.runtime = runtime or shared_deeplabcut_runtime()
        self._owns_runtime = runtime is not None
        self.enabled = str(config.get("tracking_backend_effective") or "classical") == "deeplabcut"
        self.confidence_threshold = float(config.get("pose_confidence_threshold", 0.8))
        try:
            configured_batch_size = int(config.get("pose_batch_size") or 4)
        except (TypeError, ValueError):
            configured_batch_size = 4
        self.batch_size = max(1, min(16, configured_batch_size))
        self.effective_batch_size = self.batch_size
        self.model_reference = str(config.get("pose_model_reference") or SUPERANIMAL_MODEL_REF)
        self.keypoint_schema = config.get("pose_keypoint_schema") or DEFAULT_KEYPOINT_SCHEMA
        self.frames_requested = 0
        self.frames_succeeded = 0
        self.frames_failed = 0
        self.consecutive_worker_errors = 0
        self.disabled_reason: str | None = None

    def infer(self, frame) -> PoseFrameResult:
        return self.infer_many([frame])[0]

    def _assign_observations(
        self,
        observations: list[dict[str, Any]],
    ) -> PoseFrameResult:
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

    def _assign_crop_observation(
        self,
        payload: dict[str, Any],
        *,
        roi_id: int,
        offset: tuple[int, int],
        valid_size: tuple[int, int] | None = None,
    ) -> PoseFrameResult:
        observations = normalise_pose_output(payload)
        candidates: list[tuple[float, dict[str, Any]]] = []
        for observation in observations:
            body = observation["keypoints"].get("body_center")
            if (
                not body
                or not body.get("visible")
                or float(body.get("confidence", 0.0)) < self.confidence_threshold
            ):
                continue
            if valid_size is not None:
                valid_width, valid_height = valid_size
                if not (
                    0.0 <= float(body["x"]) < float(valid_width)
                    and 0.0 <= float(body["y"]) < float(valid_height)
                ):
                    continue
            candidates.append((float(body.get("confidence", 0.0)), observation))
        if not candidates:
            self.frames_failed += 1
            return PoseFrameResult({})

        _confidence, selected = max(candidates, key=lambda item: item[0])
        translated = copy.deepcopy(selected)
        offset_x, offset_y = int(offset[0]), int(offset[1])
        for point in (translated.get("keypoints") or {}).values():
            if valid_size is not None:
                valid_width, valid_height = valid_size
                if not (
                    0.0 <= float(point["x"]) < float(valid_width)
                    and 0.0 <= float(point["y"]) < float(valid_height)
                ):
                    point["visible"] = False
            point["x"] = float(point["x"]) + offset_x
            point["y"] = float(point["y"]) + offset_y
        translated["confidence_threshold"] = self.confidence_threshold
        self.frames_succeeded += 1
        return PoseFrameResult({int(roi_id): translated})

    def infer_crops(
        self,
        frames,
        *,
        roi_ids: list[int],
        offsets: list[tuple[int, int]],
        valid_sizes: list[tuple[int, int]] | None = None,
    ) -> list[PoseFrameResult]:
        """Infer one mouse crop per ROI and restore full-frame coordinates."""
        frame_list = list(frames or [])
        if not frame_list:
            return []
        if len(frame_list) != len(roi_ids) or len(frame_list) != len(offsets):
            raise ValueError("Crop frames, ROI ids, and offsets must have equal lengths.")
        if valid_sizes is not None and len(frame_list) != len(valid_sizes):
            raise ValueError("Crop frames and valid sizes must have equal lengths.")
        if not self.enabled or self.disabled_reason:
            return [
                PoseFrameResult({}, error=self.disabled_reason)
                for _frame in frame_list
            ]
        self.frames_requested += len(frame_list)
        try:
            if hasattr(self.runtime, "infer_frames"):
                payloads = self.runtime.infer_frames(
                    frame_list,
                    model_reference=self.model_reference,
                    keypoint_schema=self.keypoint_schema,
                    max_individuals=1,
                    batch_size=self.batch_size,
                )
            else:
                payloads = [
                    self.runtime.infer_frame(
                        frame,
                        model_reference=self.model_reference,
                        keypoint_schema=self.keypoint_schema,
                        max_individuals=1,
                    )
                    for frame in frame_list
                ]
        except DeepLabCutRuntimeError as exc:
            self.frames_failed += len(frame_list)
            self.consecutive_worker_errors += 1
            if self.consecutive_worker_errors >= 3:
                self.disabled_reason = f"DLC worker disabled after repeated errors: {exc}"
            return [
                PoseFrameResult({}, error=self.disabled_reason or str(exc))
                for _frame in frame_list
            ]
        self.consecutive_worker_errors = 0
        effective = getattr(self.runtime, "effective_batch_size", None)
        if effective is not None:
            self.effective_batch_size = max(1, int(effective))
        return [
            self._assign_crop_observation(
                payload,
                roi_id=roi_id,
                offset=offset,
                valid_size=(valid_sizes[index] if valid_sizes is not None else None),
            )
            for index, (payload, roi_id, offset) in enumerate(
                zip(payloads, roi_ids, offsets)
            )
        ]

    def infer_many(self, frames) -> list[PoseFrameResult]:
        frame_list = list(frames or [])
        if not frame_list:
            return []
        if not self.enabled or self.disabled_reason:
            return [
                PoseFrameResult({}, error=self.disabled_reason)
                for _frame in frame_list
            ]
        self.frames_requested += len(frame_list)
        try:
            if hasattr(self.runtime, "infer_frames"):
                payloads = self.runtime.infer_frames(
                    frame_list,
                    model_reference=self.model_reference,
                    keypoint_schema=self.keypoint_schema,
                    max_individuals=max(1, len(self.trackers)),
                    batch_size=self.batch_size,
                )
            else:
                payloads = [
                    self.runtime.infer_frame(
                        frame,
                        model_reference=self.model_reference,
                        keypoint_schema=self.keypoint_schema,
                        max_individuals=max(1, len(self.trackers)),
                    )
                    for frame in frame_list
                ]
        except DeepLabCutRuntimeError as exc:
            self.frames_failed += len(frame_list)
            self.consecutive_worker_errors += 1
            if self.consecutive_worker_errors >= 3:
                self.disabled_reason = f"DLC worker disabled after repeated errors: {exc}"
            return [
                PoseFrameResult({}, error=self.disabled_reason or str(exc))
                for _frame in frame_list
            ]
        self.consecutive_worker_errors = 0
        effective = getattr(self.runtime, "effective_batch_size", None)
        if effective is not None:
            self.effective_batch_size = max(1, int(effective))
        return [
            self._assign_observations(normalise_pose_output(payload))
            for payload in payloads
        ]

    def close(self) -> None:
        if self._owns_runtime:
            self.runtime.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
