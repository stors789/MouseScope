"""Classical mouse segmentation and same-frame silhouette landmarks.

Body erosion changes the body mask only. Head/tail evidence is never eroded
for that purpose; missing landmarks remain missing.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from vision_tracker import (
    BackgroundSubtractor,
    MorphologyProcessor,
    MouseDetector,
    create_static_background,
    create_temporal_background,
)

CLASSICAL_METHOD = "local_otsu_v1"
CLASSICAL_METHODS = {CLASSICAL_METHOD, "legacy"}


def create_classical_background(
    video_path, config, *, start_frame=0, end_frame=None, orientation_override=None
):
    """Build the same background for preview, analysis and review tools."""
    method = str(config.get("bg_method", "temporal")).lower()
    orientation = (
        config.get("orientation_override")
        if orientation_override is None
        else orientation_override
    )
    if method == "threshold":
        return BackgroundSubtractor(
            method="threshold",
            threshold=int(config.get("gray_threshold", 127)),
            invert=bool(config.get("invert", config.get("bg_type") == "dark")),
        )
    if method == "temporal":
        background, noise, _ = create_temporal_background(
            video_path,
            num_samples=int(config.get("temporal_bg_samples", 90)),
            start_frame=start_frame,
            end_frame=end_frame,
            orientation_override=orientation,
        )
        return BackgroundSubtractor(
            method="temporal",
            threshold=int(config.get("temporal_diff_threshold", 25)),
            static_bg=background,
            temporal_noise_map=noise,
        )
    if method == "static":
        return BackgroundSubtractor(
            method="static",
            static_bg=create_static_background(
                video_path, num_frames=15, orientation_override=orientation
            ),
        )
    if method in {"mog2", "knn"}:
        return BackgroundSubtractor(method=method.upper())
    raise ValueError("不支持的背景方法。")


def _odd_kernel(config, name, default):
    value = float(config.get(name, default))
    if (
        not math.isfinite(value)
        or not value.is_integer()
        or not 1 <= value <= 99
        or int(value) % 2 == 0
    ):
        raise ValueError(f"{name}必须是1到99之间的奇数。")
    return int(value)


class ClassicalMouseDetector:
    """One configured method shared by analysis, preview and validation tools."""

    def __init__(self, config):
        self.config = dict(config)
        self.method = str(config.get("classical_method", "legacy"))
        if self.method not in CLASSICAL_METHODS:
            raise ValueError("未知经典分割方法。")
        self.version = (
            CLASSICAL_METHOD
            if self.method == CLASSICAL_METHOD
            else "legacy_morphology_v1"
        )
        self.dark_mouse = not bool(
            config.get("invert", config.get("bg_type") == "dark")
        )
        self.detector = MouseDetector(
            min_area=int(config.get("min_area", 120)),
            max_area=int(config.get("max_area", 50000)),
        )
        if (
            self.detector.min_area <= 0
            or self.detector.max_area < self.detector.min_area
        ):
            raise ValueError("目标面积范围无效。")
        cleanup = _odd_kernel(config, "segmentation_kernel", 3)
        body = _odd_kernel(config, "body_erosion_kernel", 3)
        iterations = float(config.get("body_erosion_iter", 1))
        if (
            not math.isfinite(iterations)
            or not iterations.is_integer()
            or not 0 <= iterations <= 10
        ):
            raise ValueError("身体腐蚀次数必须是0到10之间的整数。")
        self.body_iterations = int(iterations)
        self.cleanup_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cleanup, cleanup)
        )
        self.body_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (body, body))
        self.legacy_morph = MorphologyProcessor(
            erosion_kernel=int(config.get("erosion_kernel", 3)),
            erosion_iterations=int(config.get("erosion_iter", 1)),
            dilation_kernel=int(config.get("dilation_kernel", 5)),
            dilation_iterations=int(config.get("dilation_iter", 2)),
        )

    def _largest(self, mask, detector=None):
        found = (detector or self.detector).detect(mask)
        detection = max(found, key=lambda item: item["area"]) if found else None
        result = np.zeros_like(mask)
        if detection is not None:
            cv2.drawContours(result, [detection["contour"]], -1, 255, -1)
        return result, detection

    def detect(
        self,
        gray,
        foreground,
        *,
        background=None,
        detection_mask=None,
        morphology=None,
        detector=None,
    ):
        if gray.ndim != 2 or foreground.shape != gray.shape:
            raise ValueError("灰度图和前景掩膜尺寸必须一致。")
        if background is not None and background.shape != gray.shape:
            raise ValueError("背景图和灰度图尺寸必须一致。")
        mask = foreground.copy()
        if detection_mask is not None:
            if detection_mask.shape != gray.shape:
                raise ValueError("ROI掩膜尺寸不一致。")
            mask = cv2.bitwise_and(mask, detection_mask)
        threshold = None
        if self.method == "legacy":
            evidence, head_detection = self._largest(
                (morphology or self.legacy_morph).process(mask, inplace=False), detector
            )
            body_mask, body_detection = evidence.copy(), head_detection
        else:
            if background is not None:
                polarity = gray < background if self.dark_mouse else gray > background
                mask = np.where(polarity, mask, 0).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.cleanup_kernel)
            mask, proposal = self._largest(mask)
            if proposal is None:
                evidence, head_detection = mask, None
            else:
                x, y, width, height = proposal["bbox"]
                pad = max(1, round(max(width, height) * 0.15))
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(gray.shape[1], x + width + pad), min(
                    gray.shape[0], y + height + pad
                )
                threshold, _ = cv2.threshold(
                    gray[y0:y1, x0:x1], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                intensity = gray <= threshold if self.dark_mouse else gray > threshold
                mask = np.where(intensity, mask, 0).astype(np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.cleanup_kernel)
                if detection_mask is not None:
                    mask = cv2.bitwise_and(mask, detection_mask)
                evidence, head_detection = self._largest(mask)
            body_mask = evidence.copy()
            if self.body_iterations:
                body_mask = cv2.erode(
                    body_mask, self.body_kernel, iterations=self.body_iterations
                )
            body_mask, body_detection = self._largest(body_mask)
        # The nose and tail are inferred from evidence before BODY erosion.
        observation = (
            estimate_silhouette_landmarks(head_detection["contour"])
            if head_detection is not None
            else dict(keypoints={}, source="classical", reason="no_mouse_detected")
        )
        if body_detection is None:
            observation = dict(
                keypoints={},
                source="classical",
                reason=(
                    "body_removed_by_erosion"
                    if head_detection is not None
                    else "no_mouse_detected"
                ),
            )
        elif self.method != "legacy":
            cx, cy = body_detection["centroid"]
            observation["keypoints"]["body_center"] = dict(
                x=float(cx), y=float(cy), confidence=1.0, visible=True
            )
        observation.update(
            body_mask=body_mask,
            landmark_mask=evidence,
            body_contour=(
                body_detection["contour"] if body_detection is not None else None
            ),
            landmark_contour=(
                head_detection["contour"] if head_detection is not None else None
            ),
            body_area_px=int(cv2.countNonZero(body_mask)),
            landmark_area_px=int(cv2.countNonZero(evidence)),
            classical_method=self.method,
            classical_method_version=self.version,
            landmark_evidence=(
                "pre_body_erosion"
                if self.method != "legacy"
                else "legacy_processed_contour"
            ),
            local_gray_threshold=threshold,
            score_kind="geometric_quality",
        )
        return observation


def estimate_silhouette_landmarks(contour):
    """Conservative tail/body geometry; unknown head stays unknown.

    Quality is a heuristic, not a calibrated probability of correctness.
    This function deliberately has no knowledge of holes or their target.
    """
    if contour is None or len(contour) < 3:
        return {"keypoints": {}, "source": "classical", "reason": "no_contour"}
    x, y, w, h = cv2.boundingRect(contour)
    local = np.zeros((h + 4, w + 4), np.uint8)
    shifted = contour.astype(np.int32) - np.array([[[x - 2, y - 2]]], np.int32)
    cv2.drawContours(local, [shifted], -1, 255, -1)
    m = cv2.moments(local)
    if m["m00"] < 20:
        return {"keypoints": {}, "source": "classical", "reason": "small_contour"}
    radius = math.sqrt(cv2.countNonZero(local) / math.pi)
    size = max(3, int(radius * 0.5) // 2 * 2 + 1)
    opened = cv2.morphologyEx(
        local,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
    )
    body_contours, _ = cv2.findContours(
        opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not body_contours:
        return {"keypoints": {}, "source": "classical", "reason": "body_unresolved"}
    body_contour = max(body_contours, key=cv2.contourArea)
    bm = cv2.moments(body_contour)
    if bm["m00"] <= 0:
        return {"keypoints": {}, "source": "classical", "reason": "body_unresolved"}
    center = np.array([bm["m10"] / bm["m00"], bm["m01"] / bm["m00"]], np.float32)
    body = np.zeros_like(local)
    cv2.drawContours(body, [body_contour], -1, 255, -1)
    tail_mask = cv2.bitwise_and(local, cv2.bitwise_not(body))
    _, labels, sizes, _ = cv2.connectedComponentsWithStats(tail_mask)
    candidates = []
    for label in range(1, len(sizes)):
        ys, xs = np.where(labels == label)
        if len(xs) < max(5, cv2.countNonZero(local) * 0.015):
            continue
        pixels = np.column_stack([xs, ys]).astype(np.float32)
        distances = np.linalg.norm(pixels - center, axis=1)
        eigenvalues = np.linalg.eigvalsh(np.cov(pixels.T))
        elongation = float(eigenvalues[-1] / max(eigenvalues[0], 0.1))
        if elongation < 4 or float(distances.max()) < radius * 1.2:
            continue
        candidates.append(
            (float(distances.max()), pixels[int(distances.argmin())], elongation)
        )
    offset = np.array([x - 2, y - 2], np.float32)

    def keypoint(p, confidence):
        return dict(
            x=float(p[0] + offset[0]),
            y=float(p[1] + offset[1]),
            confidence=float(confidence),
            visible=True,
        )

    points = {"body_center": keypoint(center, 1.0)}
    if not candidates:
        return dict(keypoints=points, source="classical", reason="tail_unresolved")
    _, root, elongation = max(candidates, key=lambda c: c[0])
    direction = center - root
    length = float(np.linalg.norm(direction))
    if length < radius * 0.25:
        return dict(keypoints=points, source="classical", reason="head_ambiguous")
    direction /= length
    vertices = body_contour.reshape(-1, 2).astype(np.float32)
    projection = (vertices - center) @ direction
    nose = vertices[int(projection.argmax())]
    quality = min(
        0.95,
        0.35
        + 0.3 * min(1.0, elongation / 20.0)
        + 0.3 * min(1.0, float(projection.max()) / max(radius, 1.0)),
    )
    points.update(nose=keypoint(nose, quality), tail_base=keypoint(root, quality))
    return dict(keypoints=points, source="classical", reason="silhouette_tail_geometry")
