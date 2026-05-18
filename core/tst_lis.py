from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


STATE_STRUGGLE = "struggle"
STATE_IMMOBILE = "immobile"


@dataclass
class TSTLISConfig:
    eps: float = 1e-6
    min_area_ratio: float = 0.0005
    max_area_ratio: float = 0.70
    min_body_length_px: float = 14.0
    use_rotation_alignment: bool = False
    max_rotation_deg: float = 30.0
    core_erode_ratio: float = 0.08
    temporal_window_s: float = 1.5
    highfreq_window_s: float = 0.5
    immobile_confirm_s: float = 0.75
    active_confirm_s: float = 0.35
    min_immobile_bout_s: float = 0.50
    min_struggle_bout_s: float = 0.20
    merge_gap_s: float = 0.30
    active_entry_hit_ratio: float = 0.45
    active_release_hit_ratio: float = 0.20
    immobile_entry_hit_ratio: float = 0.55
    score_mode_prior_min: float = 0.35
    score_mode_entry: float = 0.50
    score_mode_low_percentile: float = 25.0
    score_mode_high_percentile: float = 90.0
    score_mode_fill_gap_s: float = 2.0
    appendage_rescue_enabled: bool = True
    appendage_rescue_prior_min: float = 0.28
    appendage_rescue_prior_max: float = 0.35
    appendage_rescue_aspect_min: float = 1.55
    appendage_rescue_score_entry: float = 0.45
    appendage_rescue_flow_min: float = 20.0
    appendage_rescue_speed_std_max: float = 6.0
    appendage_rescue_front_score_entry: float = 0.32
    appendage_rescue_front_residual_min: float = 0.026
    appendage_rescue_front_peripheral_min: float = 0.010
    appendage_rescue_front_highfreq_min: float = 0.0060
    appendage_rescue_front_flow_min: float = 15.0
    appendage_rescue_front_periodicity_min: float = 0.30
    appendage_rescue_front_global_max: float = 0.012
    appendage_rescue_front_aspect_min: float = 1.10
    appendage_rescue_front_speed_std_max: float = 6.5
    appendage_rescue_periodic_aspect_min: float = 1.70
    appendage_rescue_periodic_score_entry: float = 0.45
    appendage_rescue_periodic_residual_min: float = 0.025
    appendage_rescue_periodic_peripheral_min: float = 0.016
    appendage_rescue_periodic_highfreq_min: float = 0.025
    appendage_rescue_periodic_flow_min: float = 14.0
    appendage_rescue_periodic_periodicity_min: float = 0.48
    appendage_rescue_periodic_global_max: float = 0.010
    appendage_rescue_periodic_speed_std_max: float = 12.0
    appendage_rescue_fill_gap_s: float = 1.0
    appendage_rescue_retro_s: float = 0.7
    appendage_rescue_hold_s: float = 0.3
    passive_periodicity_threshold: float = 0.40
    low_info_epsilon: float = 1e-4
    top_contact_ratio: float = 0.10
    centroid_high_ratio: float = 0.25
    elongation_threshold: float = 1.8


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _clip01(value: Any) -> float:
    return float(np.clip(_safe_float(value, 0.0), 0.0, 1.0))


def _safe_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _odd_kernel_size(value: float) -> int:
    size = max(3, int(round(float(value))))
    if size % 2 == 0:
        size += 1
    return size


def _mask_from_contour(
    shape: tuple[int, int],
    contour_full: np.ndarray,
    x0: int,
    y0: int,
) -> np.ndarray:
    shifted = contour_full.astype(np.int32, copy=True)
    shifted[:, 0, 0] -= int(x0)
    shifted[:, 0, 1] -= int(y0)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    return mask


def _centroid_from_contour(contour: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if abs(float(moments.get("m00", 0.0))) > 1e-9:
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
    x, y, w, h = bbox
    return float(x + w * 0.5), float(y + h * 0.5)


def _pca_body_axis(mask: np.ndarray) -> tuple[float, float]:
    points = np.column_stack(np.nonzero(mask > 0))
    if points.shape[0] < 5:
        return 0.0, 0.0
    xy = points[:, ::-1].astype(np.float32)
    mean = np.mean(xy, axis=0)
    centered = xy - mean
    cov = centered.T @ centered / max(float(points.shape[0] - 1), 1.0)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    order = np.argsort(eigvals)[::-1]
    primary = eigvecs[:, order[0]]
    projections = centered @ primary
    length = float(np.percentile(projections, 95) - np.percentile(projections, 5))
    angle = float(math.degrees(math.atan2(float(primary[1]), float(primary[0]))))
    return max(length, 0.0), angle


def _angle_delta_deg(curr: float, prev: float) -> float:
    # PCA axes have a 180 degree ambiguity. Fold the delta into [-90, 90].
    delta = (float(curr) - float(prev) + 90.0) % 180.0 - 90.0
    return float(delta)


def _hu_log(mask: np.ndarray, eps: float) -> np.ndarray:
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    return -np.sign(hu) * np.log10(np.abs(hu) + float(eps))


def _acf_periodicity(values: list[float], dts: list[float], min_lag_s: float = 0.2, max_lag_s: float = 1.0) -> float:
    if len(values) < 8:
        return 0.0
    span_s = float(sum(max(float(dt), 1e-6) for dt in dts))
    if span_s <= max_lag_s or span_s <= 1e-6:
        return 0.0
    effective_fps = max((len(values) - 1) / span_s, 1e-6)
    min_lag = max(1, int(round(min_lag_s * effective_fps)))
    max_lag = max(min_lag, int(round(max_lag_s * effective_fps)))
    series = np.asarray(values, dtype=np.float32)
    series = series - float(np.median(series))
    if float(np.std(series)) <= 1e-6:
        return 0.0
    corr = np.correlate(series, series, mode="full")
    corr = corr[corr.size // 2:]
    if corr.size <= min_lag or abs(float(corr[0])) <= 1e-9:
        return 0.0
    max_lag = min(max_lag, corr.size - 1)
    if max_lag < min_lag:
        return 0.0
    return _clip01(float(np.max(corr[min_lag:max_lag + 1]) / max(float(corr[0]), 1e-9)))


class TSTLISFeatureExtractor:
    """Feature extractor for residual-motion TST scoring.

    The extractor consumes the already detected body contour/bbox. It does not
    perform mouse detection; it only measures whether motion remains after the
    previous body is translated to the current body position.
    """

    def __init__(self, fps: float = 30.0, config: TSTLISConfig | None = None) -> None:
        self.fps = max(float(fps or 30.0), 1e-6)
        self.config = config or TSTLISConfig()
        self.prev_gray: np.ndarray | None = None
        self.prev_contour: np.ndarray | None = None
        self.prev_bbox: tuple[int, int, int, int] | None = None
        self.prev_centroid: tuple[float, float] | None = None
        self.prev_geom: dict[str, float] | None = None
        self.history: deque[dict[str, float]] = deque()

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_contour = None
        self.prev_bbox = None
        self.prev_centroid = None
        self.prev_geom = None
        self.history.clear()

    def mark_missed(self, gray_frame: np.ndarray | None = None) -> None:
        self.prev_gray = gray_frame.copy() if gray_frame is not None else None
        self.prev_contour = None
        self.prev_bbox = None
        self.prev_centroid = None
        self.prev_geom = None

    def _trim_history(self) -> None:
        max_s = max(float(self.config.temporal_window_s), float(self.config.highfreq_window_s), 0.1)
        while len(self.history) > 1 and sum(float(item.get("dt_s", 0.0)) for item in self.history) > max_s:
            self.history.popleft()

    def _recent_values(self, key: str, window_s: float) -> tuple[list[float], list[float]]:
        values: list[float] = []
        dts: list[float] = []
        remaining = max(float(window_s), 1e-6)
        for item in reversed(self.history):
            dt = max(_safe_float(item.get("dt_s"), 0.0), 1e-6)
            values.append(_safe_float(item.get(key), 0.0))
            dts.append(dt)
            remaining -= dt
            if remaining <= 0:
                break
        values.reverse()
        dts.reverse()
        return values, dts

    @staticmethod
    def _highfreq(values: list[float]) -> float:
        if len(values) < 3:
            return 0.0
        diffs = np.abs(np.diff(np.asarray(values, dtype=np.float32)))
        if diffs.size == 0:
            return 0.0
        return float(np.median(diffs))

    def update(
        self,
        *,
        frame_idx: int,
        gray_frame: np.ndarray,
        contour: np.ndarray | None,
        bbox: tuple[int, int, int, int] | None,
        centroid: tuple[float, float] | None = None,
        area: float | None = None,
        solidity: float | None = None,
        roi_bbox: tuple[int, int, int, int] | None = None,
        dt_s: float | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        dt = max(_safe_float(dt_s, 1.0 / self.fps), 1e-6)
        h_img, w_img = gray_frame.shape[:2]
        roi_area = float(w_img * h_img)
        if roi_bbox is not None:
            try:
                roi_area = max(float(roi_bbox[2]) * float(roi_bbox[3]), 1.0)
            except Exception:
                roi_area = float(w_img * h_img)

        features: dict[str, Any] = {
            "tst_lis_available": False,
            "segmentation_bad": True,
            "global_motion_norm": 0.0,
            "aligned_mask_change": 0.0,
            "residual_gray_motion": 0.0,
            "residual_motion": 0.0,
            "core_motion": 0.0,
            "peripheral_motion": 0.0,
            "peripheral_core_ratio": 0.0,
            "shape_change": 0.0,
            "highfreq_motion": 0.0,
            "periodicity": 0.0,
            "passive_swing_evidence": False,
            "climbing_suspected": False,
            "dt_s": float(dt),
        }

        if contour is None or bbox is None:
            self.mark_missed(gray_frame)
            return features

        try:
            x, y, w, h = [int(v) for v in bbox]
        except Exception:
            self.mark_missed(gray_frame)
            return features
        if w <= 0 or h <= 0:
            self.mark_missed(gray_frame)
            return features

        contour_full = contour.astype(np.int32, copy=False)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            self.mark_missed(gray_frame)
            return features

        curr_mask_bbox = _mask_from_contour((y2 - y1, x2 - x1), contour_full, x1, y1)
        body_area = float(area) if area is not None else float(cv2.countNonZero(curr_mask_bbox))
        body_area = max(body_area, 0.0)
        perimeter = float(cv2.arcLength(contour_full, True))
        extent = body_area / max(float(w * h), 1.0)
        if solidity is None:
            hull = cv2.convexHull(contour_full)
            hull_area = float(cv2.contourArea(hull))
            solidity_value = body_area / max(hull_area, 1.0)
        else:
            solidity_value = max(_safe_float(solidity, 0.0), 0.0)
        centroid_value = centroid if centroid is not None else _centroid_from_contour(contour_full, (x, y, w, h))
        cx, cy = float(centroid_value[0]), float(centroid_value[1])
        body_length, axis_angle = _pca_body_axis(curr_mask_bbox)
        body_length = max(body_length, float(max(w, h)))
        aspect_ratio = float(h / max(w, 1))
        area_ratio = body_area / max(roi_area, 1.0)
        curr_hu = _hu_log(curr_mask_bbox, cfg.eps)
        segmentation_bad = bool(
            body_area <= 0.0
            or area_ratio < cfg.min_area_ratio
            or area_ratio > cfg.max_area_ratio
            or body_length < cfg.min_body_length_px
        )

        top_contact = (float(y) <= (float(roi_bbox[1]) + float(roi_bbox[3]) * cfg.top_contact_ratio)) if roi_bbox else (y <= h_img * cfg.top_contact_ratio)
        centroid_high = (cy <= (float(roi_bbox[1]) + float(roi_bbox[3]) * cfg.centroid_high_ratio)) if roi_bbox else (cy <= h_img * cfg.centroid_high_ratio)
        climbing_suspected = bool(top_contact and centroid_high and aspect_ratio >= cfg.elongation_threshold)

        geom = {
            "area": float(body_area),
            "perimeter": float(perimeter),
            "aspect_ratio": float(aspect_ratio),
            "solidity": float(solidity_value),
            "extent": float(extent),
            "body_length": float(body_length),
            "axis_angle": float(axis_angle),
            "hu0": float(curr_hu[0]) if curr_hu.size else 0.0,
        }

        global_motion_norm = 0.0
        aligned_mask_change = 0.0
        residual_gray_motion = 0.0
        core_motion = 0.0
        peripheral_motion = 0.0
        peripheral_core_ratio = 0.0
        shape_change = 0.0
        has_prev = (
            self.prev_gray is not None
            and self.prev_contour is not None
            and self.prev_bbox is not None
            and self.prev_centroid is not None
            and self.prev_geom is not None
        )

        if has_prev:
            prev_cx, prev_cy = self.prev_centroid or (cx, cy)
            dx = float(cx - prev_cx)
            dy = float(cy - prev_cy)
            global_motion_norm = float(math.hypot(dx, dy) / max(body_length, cfg.eps))

            prev_x, prev_y, prev_w, prev_h = self.prev_bbox or (x, y, w, h)
            pad = int(max(8, abs(dx) + abs(dy) + body_length * 0.08 + 4))
            ux1 = max(0, int(min(x, prev_x) - pad))
            uy1 = max(0, int(min(y, prev_y) - pad))
            ux2 = min(w_img, int(max(x + w, prev_x + prev_w) + pad))
            uy2 = min(h_img, int(max(y + h, prev_y + prev_h) + pad))
            if ux2 > ux1 and uy2 > uy1:
                curr_crop = gray_frame[uy1:uy2, ux1:ux2]
                prev_crop = self.prev_gray[uy1:uy2, ux1:ux2]
                affine = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
                if cfg.use_rotation_alignment:
                    dtheta = _angle_delta_deg(axis_angle, float(self.prev_geom.get("axis_angle", axis_angle)))
                    if abs(dtheta) <= cfg.max_rotation_deg:
                        center = (float(cx - ux1), float(cy - uy1))
                        rot = cv2.getRotationMatrix2D(center, dtheta, 1.0)
                        affine = np.asarray(rot, dtype=np.float32)
                        affine[:, 2] += np.array([dx, dy], dtype=np.float32)

                prev_aligned = cv2.warpAffine(
                    prev_crop,
                    affine,
                    (curr_crop.shape[1], curr_crop.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                curr_mask = _mask_from_contour(curr_crop.shape, contour_full, ux1, uy1)
                prev_mask = _mask_from_contour(curr_crop.shape, self.prev_contour, ux1, uy1)
                prev_mask_aligned = cv2.warpAffine(
                    prev_mask,
                    affine,
                    (curr_crop.shape[1], curr_crop.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                curr_bool = curr_mask > 0
                prev_bool = prev_mask_aligned > 0
                union = curr_bool | prev_bool
                inter = curr_bool & prev_bool
                if np.any(union):
                    aligned_mask_change = float(1.0 - (float(inter.sum()) / max(float(union.sum()), 1.0)))
                    diff = cv2.absdiff(curr_crop, prev_aligned).astype(np.float32) / 255.0
                    residual_gray_motion = float(np.mean(diff[union])) if np.any(union) else 0.0
                    kernel_size = _odd_kernel_size(body_length * cfg.core_erode_ratio)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    core = cv2.erode(curr_mask, kernel, iterations=1) > 0
                    if not np.any(core):
                        core = curr_bool
                    peripheral = curr_bool & (~core)
                    if not np.any(peripheral):
                        peripheral = curr_bool
                    core_motion = float(np.mean(diff[core])) if np.any(core) else 0.0
                    peripheral_motion = float(np.mean(diff[peripheral])) if np.any(peripheral) else 0.0
                    peripheral_core_ratio = float(peripheral_motion / max(core_motion, cfg.eps))

            prev_geom = self.prev_geom
            d_area = abs(geom["area"] - prev_geom["area"]) / max(prev_geom["area"], 1.0)
            d_perimeter = abs(geom["perimeter"] - prev_geom["perimeter"]) / max(prev_geom["perimeter"], 1.0)
            d_aspect = abs(geom["aspect_ratio"] - prev_geom["aspect_ratio"]) / max(abs(prev_geom["aspect_ratio"]), cfg.eps)
            d_solidity = abs(geom["solidity"] - prev_geom["solidity"])
            d_extent = abs(geom["extent"] - prev_geom["extent"])
            try:
                prev_mask_bbox = _mask_from_contour(
                    (max(prev_h, 1), max(prev_w, 1)),
                    self.prev_contour,
                    prev_x,
                    prev_y,
                )
                prev_hu = _hu_log(prev_mask_bbox, cfg.eps)
                d_hu = float(np.linalg.norm(curr_hu - prev_hu) / 25.0)
            except Exception:
                d_hu = 0.0
            shape_terms = [
                min(float(d_area), 1.0),
                min(float(d_perimeter), 1.0),
                min(float(d_aspect), 1.0),
                min(float(d_solidity), 1.0),
                min(float(d_extent), 1.0),
                min(float(d_hu), 1.0),
                min(float(aligned_mask_change), 1.0),
            ]
            shape_change = float(np.median(shape_terms))

        residual_motion = max(float(aligned_mask_change), float(residual_gray_motion))
        self.history.append({
            "dt_s": float(dt),
            "cx": float(cx),
            "cy": float(cy),
            "global_motion_norm": float(global_motion_norm),
            "residual_motion": float(residual_motion),
            "peripheral_motion": float(peripheral_motion),
            "shape_change": float(shape_change),
        })
        self._trim_history()

        residual_values, _ = self._recent_values("residual_motion", cfg.highfreq_window_s)
        peripheral_values, _ = self._recent_values("peripheral_motion", cfg.highfreq_window_s)
        shape_values, _ = self._recent_values("shape_change", cfg.highfreq_window_s)
        highfreq_motion = max(
            self._highfreq(residual_values),
            self._highfreq(peripheral_values),
            self._highfreq(shape_values),
        )
        cx_values, cx_dts = self._recent_values("cx", cfg.temporal_window_s)
        cy_values, cy_dts = self._recent_values("cy", cfg.temporal_window_s)
        periodicity = max(_acf_periodicity(cx_values, cx_dts), _acf_periodicity(cy_values, cy_dts))

        features.update({
            "tst_lis_available": True,
            "segmentation_bad": bool(segmentation_bad),
            "body_area": float(body_area),
            "area_ratio": float(area_ratio),
            "bbox_w": int(w),
            "bbox_h": int(h),
            "aspect_ratio": float(aspect_ratio),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "body_length": float(body_length),
            "major_axis_angle": float(axis_angle),
            "perimeter": float(perimeter),
            "solidity": float(solidity_value),
            "extent": float(extent),
            "global_motion_norm": float(global_motion_norm),
            "aligned_mask_change": float(aligned_mask_change),
            "residual_gray_motion": float(residual_gray_motion),
            "residual_motion": float(residual_motion),
            "core_motion": float(core_motion),
            "peripheral_motion": float(peripheral_motion),
            "peripheral_core_ratio": float(peripheral_core_ratio),
            "shape_change": float(shape_change),
            "highfreq_motion": float(highfreq_motion),
            "periodicity": float(periodicity),
            "climbing_suspected": bool(climbing_suspected),
        })

        self.prev_gray = gray_frame.copy()
        self.prev_contour = contour_full.copy()
        self.prev_bbox = (x, y, w, h)
        self.prev_centroid = (cx, cy)
        self.prev_geom = geom
        return features


def default_tst_lis_features(*, detected: bool = False) -> dict[str, Any]:
    return {
        "tst_lis_available": bool(detected),
        "segmentation_bad": not bool(detected),
        "global_motion_norm": 0.0,
        "aligned_mask_change": 0.0,
        "residual_gray_motion": 0.0,
        "residual_motion": 0.0,
        "core_motion": 0.0,
        "peripheral_motion": 0.0,
        "peripheral_core_ratio": 0.0,
        "shape_change": 0.0,
        "highfreq_motion": 0.0,
        "periodicity": 0.0,
        "passive_swing_evidence": False,
        "climbing_suspected": False,
        "lis_is_struggling": False,
        "lis_confidence": "low",
        "lis_reason": "not_detected",
    }


class TSTLISOpticalFlowFeatures:
    """TST-LIS optical-flow feature helper.

    This intentionally lives outside the baseline tracker so LIS can evolve as
    a separate behavior path.
    """

    def __init__(self, fps: float = 30.0, pixel_to_mm: float = 1.0) -> None:
        self.fps = max(float(fps or 30.0), 1e-6)
        self.pixel_to_mm = float(pixel_to_mm or 1.0)
        self.prev_gray: np.ndarray | None = None

    def reset(self, gray_frame: np.ndarray | None = None) -> None:
        self.prev_gray = gray_frame

    def update(
        self,
        *,
        gray_frame: np.ndarray,
        fg_mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        mask_offset: tuple[int, int],
    ) -> dict[str, float] | None:
        if self.prev_gray is None:
            self.prev_gray = gray_frame
            return None
        x, y, w, h = [int(v) for v in bbox]
        margin = 10
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(gray_frame.shape[1], x + w + margin)
        y2 = min(gray_frame.shape[0], y + h + margin)
        if x2 <= x1 or y2 <= y1:
            self.prev_gray = gray_frame
            return None

        roi_prev = self.prev_gray[y1:y2, x1:x2]
        roi_curr = gray_frame[y1:y2, x1:x2]
        if roi_prev.size == 0 or roi_curr.size == 0:
            self.prev_gray = gray_frame
            return None

        scale_factor = 1.0
        if roi_prev.shape[0] > 40 or roi_prev.shape[1] > 40:
            scale_factor = 0.35
            roi_prev = cv2.resize(roi_prev, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            roi_curr = cv2.resize(roi_curr, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(
            roi_prev,
            roi_curr,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        if scale_factor != 1.0:
            flow = flow / scale_factor

        off_x, off_y = mask_offset
        local_y1 = max(0, y1 - off_y)
        local_y2 = min(fg_mask.shape[0], y2 - off_y)
        local_x1 = max(0, x1 - off_x)
        local_x2 = min(fg_mask.shape[1], x2 - off_x)
        raw_mask = fg_mask[local_y1:local_y2, local_x1:local_x2]
        if raw_mask.size > 0:
            local_mask = cv2.resize(raw_mask, (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            local_mask = np.zeros(flow.shape[:2], dtype=np.uint8)

        mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        u = flow[:, :, 0][local_mask == 255]
        v = flow[:, :, 1][local_mask == 255]
        mouse_mag = mag[local_mask == 255]
        if mouse_mag.size == 0:
            self.prev_gray = gray_frame
            return None

        valid = mouse_mag > 1e-6
        if int(np.sum(valid)) > 0:
            u_valid = u[valid]
            v_valid = v[valid]
            mag_valid = mouse_mag[valid]
            if mag_valid.size >= 50:
                k = int(0.60 * (mag_valid.size - 1))
                threshold = float(np.partition(mag_valid, k)[k])
                strong = mag_valid >= max(1e-6, threshold)
                if int(np.sum(strong)) < 20:
                    strong = np.ones_like(mag_valid, dtype=bool)
            else:
                strong = np.ones_like(mag_valid, dtype=bool)
            mag_s = mag_valid[strong]
            u_norm = u_valid[strong] / mag_s
            v_norm = v_valid[strong] / mag_s
            r_val = float(np.sqrt(np.mean(u_norm) ** 2 + np.mean(v_norm) ** 2))
            entropy = float(1.0 - r_val)
            mean_u = float(np.mean(u_valid[strong]))
            mean_v = float(np.mean(v_valid[strong]))
            coherence = float(np.sqrt(mean_u ** 2 + mean_v ** 2) / max(float(np.mean(mag_s)), 1e-6))
        else:
            entropy = 1.0
            coherence = 0.0

        n = int(mouse_mag.size)
        if n > 50:
            p90 = float(np.partition(mouse_mag, int(n * 0.9))[int(n * 0.9)])
        else:
            sorted_mag = np.sort(mouse_mag)
            p90 = float(sorted_mag[int(n * 0.9)]) if n else 0.0
        frame_rate = self.fps
        self.prev_gray = gray_frame
        return {
            "flow_p90_mm_s": float(p90 * self.pixel_to_mm * frame_rate),
            "flow_entropy": float(entropy),
            "coherence": float(coherence),
        }


class TSTLISFeatureTracker:
    """Independent LIS tracker/feature pipeline.

    It shares only low-level segmentation processors supplied by the session
    (background subtraction, morphology, contour detection). It does not run
    another behavior state machine.
    """

    def __init__(
        self,
        *,
        roi_polygon,
        roi_id: int,
        bg_subtractor,
        morphology_processor,
        detector,
        fps: float,
        pixel_to_mm: float,
        roi_name: str | None = None,
        time_config: dict[str, Any] | None = None,
        config: TSTLISConfig | None = None,
    ) -> None:
        self.roi_polygon = np.asarray(roi_polygon, dtype=np.int32)
        self.roi_id = int(roi_id)
        self.roi_name = roi_name or f"ROI {roi_id}"
        self.time_config = time_config
        if self.time_config:
            for key in ("start", "end"):
                if self.time_config.get(key) is not None:
                    try:
                        self.time_config[key] = float(self.time_config[key])
                    except Exception:
                        pass
        self.bg_subtractor = bg_subtractor
        self.morphology_processor = morphology_processor
        self.detector = detector
        self.fps = max(float(fps or 30.0), 1e-6)
        self.pixel_to_mm = float(pixel_to_mm or 1.0)
        self.extractor = TSTLISFeatureExtractor(fps=self.fps, config=config)
        self.flow_features = TSTLISOpticalFlowFeatures(fps=self.fps, pixel_to_mm=self.pixel_to_mm)
        self.roi_mask = None
        self.roi_bbox = None
        self._cropped_roi_mask = None
        self.frame_count = 0
        self.detected_count = 0
        self.first_detected_time = None
        self.last_centroid = None
        self.speeds: deque[float] = deque(maxlen=max(1, int(round(0.3 * self.fps))))
        self.trajectory = deque(maxlen=100)
        self.stats = {
            "total_frames": 0,
            "detected_frames": 0,
            "undetected_frames": 0,
        }

    def initialize_roi_mask(self, frame_shape) -> None:
        self.roi_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [self.roi_polygon], 255)
        x, y, w, h = cv2.boundingRect(self.roi_polygon)
        self.roi_bbox = (x, y, w, h)

    def _is_in_stats_window(self, current_time=None) -> bool:
        if not self.time_config:
            return True
        if current_time is None:
            current_time = self.frame_count / self.fps
        mode = self.time_config.get("mode", "global")
        start_cfg = self.time_config.get("start")
        end_cfg = self.time_config.get("end")
        if start_cfg == "":
            start_cfg = None
        if end_cfg == "":
            end_cfg = None
        if mode == "global":
            return True
        if mode == "precise":
            if start_cfg is not None and current_time < start_cfg:
                return False
            if end_cfg is not None and current_time > end_cfg:
                return False
            return True
        if mode == "relative":
            if self.first_detected_time is None:
                return False
            elapsed = current_time - self.first_detected_time
            if start_cfg is not None and elapsed < start_cfg:
                return False
            if end_cfg is not None and elapsed > end_cfg:
                return False
            return True
        return True

    def _mark_stats(self, detected: bool) -> None:
        self.stats["total_frames"] += 1
        if detected:
            self.stats["detected_frames"] += 1
        else:
            self.stats["undetected_frames"] += 1

    def update(self, frame, *, current_time=None, gray=None, dt_s=None) -> dict[str, Any]:
        if self.roi_mask is None:
            self.initialize_roi_mask(frame.shape)
        self.frame_count += 1
        dt = max(_safe_float(dt_s, 1.0 / self.fps), 1e-6)
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x, y, w, h = self.roi_bbox
        roi_frame = frame[y:y + h, x:x + w]
        roi_gray = gray[y:y + h, x:x + w]
        if self._cropped_roi_mask is None:
            self._cropped_roi_mask = self.roi_mask[y:y + h, x:x + w]
        fg_mask_raw = self.bg_subtractor.apply(roi_frame, gray=roi_gray)
        fg_mask = cv2.bitwise_and(fg_mask_raw, fg_mask_raw, mask=self._cropped_roi_mask)
        fg_mask = self.morphology_processor.process(fg_mask, inplace=True)
        detections = self.detector.detect(fg_mask)

        if not detections:
            self._mark_stats(False)
            self.last_centroid = None
            self.extractor.mark_missed(gray)
            self.flow_features.reset(gray)
            return {
                "roi_id": self.roi_id,
                "roi_name": self.roi_name,
                "roi_bbox": self.roi_bbox,
                "detected": False,
                "centroid": None,
                "contour": None,
                "bbox": None,
                "area": 0.0,
                "flow": None,
                "is_struggling": False,
                "is_swinging": False,
                "behavior_algorithm": "tst_lis",
                "is_in_stats_window": self._is_in_stats_window(current_time),
                "dt_s": float(dt),
                "tst_lis": default_tst_lis_features(detected=False),
                "fg_mask": fg_mask,
                "roi_offset": (x, y),
                "trajectory": list(self.trajectory),
                "speed": 0.0,
                "speed_std": 0.0,
                "flow_p90": 0.0,
                "flow_entropy": 1.0,
                "flow_over_speed": 0.0,
                "flow_over_speed_std": 0.0,
                "acf_peak": 0.0,
                "swing_state": {},
            }

        detection = max(detections, key=lambda d: d.get("area", 0))
        local_centroid = detection["centroid"]
        centroid = (float(local_centroid[0] + x), float(local_centroid[1] + y))
        local_bbox = detection["bbox"]
        bbox = (int(local_bbox[0] + x), int(local_bbox[1] + y), int(local_bbox[2]), int(local_bbox[3]))
        local_contour = detection.get("contour")
        contour = local_contour + np.array([[x, y]]) if local_contour is not None else None
        detection["centroid"] = centroid
        detection["bbox"] = bbox
        detection["contour"] = contour
        if self.first_detected_time is None:
            self.first_detected_time = current_time if current_time is not None else self.frame_count / self.fps

        if self.last_centroid is None:
            speed = 0.0
        else:
            distance_px = float(np.linalg.norm(np.asarray(centroid) - np.asarray(self.last_centroid)))
            speed = distance_px * self.pixel_to_mm * self.fps
        self.last_centroid = centroid
        self.speeds.append(float(speed))
        speed_std = float(np.std(np.asarray(self.speeds, dtype=np.float32))) if len(self.speeds) >= 3 else 0.0

        flow = self.flow_features.update(
            gray_frame=gray,
            fg_mask=fg_mask,
            bbox=bbox,
            mask_offset=(x, y),
        )
        flow_p90 = float(flow.get("flow_p90_mm_s", 0.0)) if flow else 0.0
        flow_entropy = float(flow.get("flow_entropy", 1.0)) if flow else 1.0
        flow_over_speed = float(flow_p90 / (abs(speed) + 1e-6))
        flow_over_speed_std = float(flow_p90 / (abs(speed_std) + 1e-6))

        lis_features = self.extractor.update(
            frame_idx=self.frame_count,
            gray_frame=gray,
            contour=contour,
            bbox=bbox,
            centroid=centroid,
            area=detection.get("area"),
            solidity=detection.get("solidity"),
            roi_bbox=self.roi_bbox,
            dt_s=dt,
        )
        self._mark_stats(True)
        self.detected_count += 1
        self.trajectory.append(centroid)
        aspect_ratio = float(bbox[3] / max(bbox[2], 1))

        return {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "roi_bbox": self.roi_bbox,
            "detected": True,
            "centroid": centroid,
            "contour": contour,
            "bbox": bbox,
            "area": detection.get("area", 0.0),
            "flow": flow,
            "is_struggling": False,
            "is_swinging": False,
            "behavior_algorithm": "tst_lis",
            "is_in_stats_window": self._is_in_stats_window(current_time),
            "dt_s": float(dt),
            "tst_lis": lis_features,
            "fg_mask": fg_mask,
            "roi_offset": (0, 0),
            "trajectory": list(self.trajectory),
            "speed": float(speed),
            "speed_std": float(speed_std),
            "flow_p90": float(flow_p90),
            "coherence": float(flow.get("coherence", 0.0)) if flow else 0.0,
            "flow_entropy": float(flow_entropy),
            "flow_over_speed": float(flow_over_speed),
            "flow_over_speed_std": float(flow_over_speed_std),
            "aspect_ratio": float(aspect_ratio),
            "acf_peak": float(lis_features.get("periodicity", 0.0) or 0.0),
            "swing_state": {
                "swing_score": 0.0,
                "struggle_score": 0.0,
                "swing_margin": 0.0,
                "internal_flow_ok": False,
                "rigid_body_like": False,
            },
        }

    def get_behavior_thresholds(self) -> dict[str, float]:
        return {
            "flow_thr_hi": 0.0,
            "flow_thr_lo": 0.0,
            "speed_std_thr_hi": 0.0,
            "speed_std_thr_lo": 0.0,
        }

    def get_immobility_summary(self, fps: float) -> dict[str, Any]:
        total_frames = int(self.stats.get("total_frames", 0) or 0)
        detected_frames = int(self.stats.get("detected_frames", 0) or 0)
        undetected_frames = int(self.stats.get("undetected_frames", 0) or 0)
        total_time = total_frames / max(float(fps or self.fps), 1e-6)
        detected_time = detected_frames / max(float(fps or self.fps), 1e-6)
        uncertain_time = undetected_frames / max(float(fps or self.fps), 1e-6)
        return {
            "roi_id": self.roi_id,
            "roi_name": self.roi_name,
            "time_config": self.time_config,
            "auto_behavior_threshold": False,
            "behavior_threshold_sensitivity": 0.0,
            "flow_thr_hi": 0.0,
            "flow_thr_lo": 0.0,
            "speed_std_thr_hi": 0.0,
            "speed_std_thr_lo": 0.0,
            "total_time_seconds": round(total_time, 2),
            "detected_time_seconds": round(detected_time, 2),
            "uncertain_time_seconds": round(uncertain_time, 2),
            "immobility_time_seconds": round(detected_time, 2),
            "struggling_time_seconds": 0.0,
            "detected_rate_percent": round((detected_time / max(total_time, 1e-9)) * 100, 1) if total_time else 0.0,
            "immobility_percent": 100.0 if detected_frames else 0.0,
            "struggling_percent": 0.0,
            "uncertain_percent": round((uncertain_time / max(total_time, 1e-9)) * 100, 1) if total_time else 0.0,
            "immobility_episodes": 0,
            "struggling_episodes": 0,
            "immobility_frequency_per_min": 0.0,
            "struggling_frequency_per_min": 0.0,
            "latency_to_immobility_seconds": None,
            "detected_frames": detected_frames,
            "undetected_frames": undetected_frames,
            "total_frames": total_frames,
        }


def robust_feature_stats(values: list[float] | np.ndarray, cfg: TSTLISConfig | None = None) -> dict[str, float | bool]:
    cfg = cfg or TSTLISConfig()
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "q20": 0.0,
            "q35": 0.0,
            "q50": 0.0,
            "q75": 0.0,
            "q80": 0.0,
            "q90": 0.0,
            "mad": 0.0,
            "low_information": True,
        }
    q20, q35, q50, q75, q80, q90 = [float(v) for v in np.percentile(arr, [20, 35, 50, 75, 80, 90])]
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    return {
        "q20": q20,
        "q35": q35,
        "q50": q50,
        "q75": q75,
        "q80": q80,
        "q90": q90,
        "mad": mad,
        "low_information": bool((q80 - q20) < cfg.low_info_epsilon),
    }


def estimate_tst_lis_thresholds(rows: list[dict[str, Any]], cfg: TSTLISConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TSTLISConfig()
    valid = [
        row for row in rows
        if row.get("detected", True)
        and not bool(row.get("lis_segmentation_bad", row.get("segmentation_bad", False)))
    ]
    if not valid:
        valid = [row for row in rows if row.get("detected", True)]

    thresholds: dict[str, Any] = {"low_information_features": []}
    feature_stats: dict[str, dict[str, float | bool]] = {}

    def add_motion_thresholds(prefix: str, column: str) -> None:
        stats = robust_feature_stats([_safe_float(row.get(column), 0.0) for row in valid], cfg)
        feature_stats[prefix] = stats
        low = min(float(stats["q50"]), max(0.0, float(stats["q20"]) + 0.5 * float(stats["mad"])))
        thresholds[f"{prefix}_low"] = float(low)
        thresholds[f"{prefix}_mid"] = float(stats["q50"])
        thresholds[f"{prefix}_high"] = float(stats["q80"])
        thresholds[f"{prefix}_very_high"] = float(stats["q90"])
        if stats["low_information"]:
            thresholds["low_information_features"].append(prefix)

    add_motion_thresholds("residual", "lis_residual_motion")
    add_motion_thresholds("peripheral", "lis_peripheral_motion")
    add_motion_thresholds("shape", "lis_shape_change")
    add_motion_thresholds("highfreq", "lis_highfreq_motion")
    add_motion_thresholds("flow", "decision_flow_p90_mm_s")
    add_motion_thresholds("speed_std", "speed_std_mm_s")

    score_columns = {
        "residual": "lis_residual_motion",
        "peripheral": "lis_peripheral_motion",
        "highfreq": "lis_highfreq_motion",
        "shape": "lis_shape_change",
        "flow": "decision_flow_p90_mm_s",
        "speed_std": "speed_std_mm_s",
    }
    low_pct = float(np.clip(cfg.score_mode_low_percentile, 0.0, 95.0))
    high_pct = float(np.clip(cfg.score_mode_high_percentile, low_pct + 1.0, 100.0))
    for prefix, column in score_columns.items():
        arr = np.asarray([_safe_float(row.get(column), 0.0) for row in valid], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            score_low = 0.0
            score_high = 0.0
        else:
            score_low, score_high = [float(v) for v in np.percentile(arr, [low_pct, high_pct])]
        if score_high <= score_low:
            score_high = score_low + cfg.eps
        thresholds[f"{prefix}_score_low"] = float(score_low)
        thresholds[f"{prefix}_score_high"] = float(score_high)

    ratio_stats = robust_feature_stats([_safe_float(row.get("lis_peripheral_core_ratio"), 0.0) for row in valid], cfg)
    thresholds["peripheral_core_ratio_high"] = float(max(1.15, ratio_stats["q80"]))
    thresholds["peripheral_core_ratio_mid"] = float(max(1.0, ratio_stats["q50"]))
    if ratio_stats["low_information"]:
        thresholds["low_information_features"].append("peripheral_core_ratio")

    global_stats = robust_feature_stats([_safe_float(row.get("lis_global_motion_norm"), 0.0) for row in valid], cfg)
    thresholds["global_mid"] = float(global_stats["q50"])
    thresholds["global_high"] = float(global_stats["q80"])

    periodicity_stats = robust_feature_stats([_safe_float(row.get("lis_periodicity"), 0.0) for row in valid], cfg)
    thresholds["periodicity_high"] = float(max(cfg.passive_periodicity_threshold, periodicity_stats["q75"]))
    flow_stats = feature_stats.get("flow") or robust_feature_stats([], cfg)
    speed_stats = feature_stats.get("speed_std") or robust_feature_stats([], cfg)
    median_prior = 0.5 * (
        _clip01(_safe_float(flow_stats.get("q50"), 0.0) / 45.0)
        + _clip01(_safe_float(speed_stats.get("q50"), 0.0) / 25.0)
    )
    high_prior = 0.75 * 0.5 * (
        _clip01(_safe_float(flow_stats.get("q90"), 0.0) / 90.0)
        + _clip01(_safe_float(speed_stats.get("q90"), 0.0) / 40.0)
    )
    motion_prior = float(max(median_prior, high_prior))
    thresholds["motion_prior"] = motion_prior
    thresholds["score_mode_enabled"] = bool(motion_prior >= float(cfg.score_mode_prior_min))
    thresholds["score_mode_entry"] = float(cfg.score_mode_entry)
    thresholds["postprocess_fill_gap_s"] = (
        float(cfg.score_mode_fill_gap_s)
        if thresholds["score_mode_enabled"]
        else float(cfg.min_immobile_bout_s)
    )
    return thresholds


def _reason_join(parts: list[str]) -> str:
    return ";".join(parts) if parts else "state_persistence"


class TSTLISStateMachine:
    def __init__(self, thresholds: dict[str, Any], fps: float = 30.0, config: TSTLISConfig | None = None) -> None:
        self.thresholds = thresholds
        self.fps = max(float(fps or 30.0), 1e-6)
        self.config = config or TSTLISConfig()
        self.state = STATE_IMMOBILE
        self.immobile_duration_s = 0.0
        self.active_duration_s = 0.0
        self._immobile_evidence_window: deque[tuple[bool, float]] = deque()
        self._active_evidence_window: deque[tuple[bool, float]] = deque()

    def _threshold(self, key: str, default: float = 0.0) -> float:
        return _safe_float(self.thresholds.get(key), default)

    def _below_or_equal(self, value: float, key: str, default: float = 0.0) -> bool:
        threshold = self._threshold(key, default)
        margin = max(self.config.eps, abs(threshold) * 0.05)
        return float(value) <= threshold + margin

    def _above(self, value: float, key: str, default: float = 0.0) -> bool:
        threshold = self._threshold(key, default)
        margin = max(self.config.eps, abs(threshold) * 0.02)
        return float(value) > threshold + margin

    def _score_norm(self, value: float, prefix: str) -> float:
        low = self._threshold(f"{prefix}_score_low", self._threshold(f"{prefix}_low", 0.0))
        high = self._threshold(f"{prefix}_score_high", self._threshold(f"{prefix}_very_high", low + self.config.eps))
        return _clip01((float(value) - low) / max(high - low, self.config.eps))

    def _motion_score(
        self,
        *,
        residual: float,
        peripheral: float,
        highfreq: float,
        shape: float,
        flow_p90: float,
        speed_std: float,
    ) -> tuple[float, dict[str, float]]:
        comps = {
            "residual": self._score_norm(residual, "residual"),
            "peripheral": self._score_norm(peripheral, "peripheral"),
            "highfreq": self._score_norm(highfreq, "highfreq"),
            "shape": self._score_norm(shape, "shape"),
            "flow": self._score_norm(flow_p90, "flow"),
            "speed_std": self._score_norm(speed_std, "speed_std"),
        }
        score = (
            0.24 * comps["residual"]
            + 0.14 * comps["peripheral"]
            + 0.22 * comps["highfreq"]
            + 0.08 * comps["shape"]
            + 0.18 * comps["flow"]
            + 0.14 * comps["speed_std"]
        )
        return float(score), comps

    @staticmethod
    def _append_evidence_window(window: deque[tuple[bool, float]], flag: bool, dt: float, max_s: float) -> None:
        window.append((bool(flag), max(float(dt), 1e-6)))
        while len(window) > 1 and sum(item[1] for item in window) > max(float(max_s), 1e-6):
            window.popleft()

    @staticmethod
    def _window_hit_ratio(window: deque[tuple[bool, float]]) -> tuple[float, float, float]:
        total_s = sum(item[1] for item in window)
        hit_s = sum(item[1] for item in window if item[0])
        ratio = hit_s / max(total_s, 1e-9)
        return float(total_s), float(hit_s), float(ratio)

    def update(self, row: dict[str, Any]) -> dict[str, Any]:
        dt = max(_safe_float(row.get("dt_s"), 1.0 / self.fps), 1e-6)
        if bool(row.get("lis_segmentation_bad", False)) or not bool(row.get("detected", True)):
            return {
                "state": self.state,
                "is_struggling": self.state == STATE_STRUGGLE,
                "confidence": "low",
                "reason": "segmentation_bad_keep_previous",
                "passive_swing": False,
                "strong_active": False,
                "immobile_evidence": False,
                "active_evidence": False,
            }

        residual = _safe_float(row.get("lis_residual_motion"), 0.0)
        peripheral = _safe_float(row.get("lis_peripheral_motion"), 0.0)
        shape = _safe_float(row.get("lis_shape_change"), 0.0)
        highfreq = _safe_float(row.get("lis_highfreq_motion"), 0.0)
        ratio = _safe_float(row.get("lis_peripheral_core_ratio"), 0.0)
        global_motion = _safe_float(row.get("lis_global_motion_norm"), 0.0)
        periodicity = _safe_float(row.get("lis_periodicity"), 0.0)
        flow_p90 = _safe_float(row.get("decision_flow_p90_mm_s"), 0.0)
        speed_std = _safe_float(row.get("speed_std_mm_s"), 0.0)

        immobile_reasons: list[str] = []
        if self._below_or_equal(residual, "residual_low"):
            immobile_reasons.append("residual_low")
        if self._below_or_equal(peripheral, "peripheral_low"):
            immobile_reasons.append("peripheral_low")
        if self._below_or_equal(shape, "shape_low"):
            immobile_reasons.append("shape_low")
        if self._below_or_equal(highfreq, "highfreq_low"):
            immobile_reasons.append("highfreq_low")
        immobile_evidence = len(immobile_reasons) == 4

        active_reasons: list[str] = []
        if self._above(residual, "residual_high"):
            active_reasons.append("residual_high")
        if self._above(peripheral, "peripheral_high"):
            active_reasons.append("peripheral_high")
        if self._above(shape, "shape_high"):
            active_reasons.append("shape_high")
        if self._above(highfreq, "highfreq_high"):
            active_reasons.append("highfreq_high")
        if self._above(ratio, "peripheral_core_ratio_high", 1.15):
            active_reasons.append("peripheral_core_ratio_high")

        residual_active = "residual_high" in active_reasons
        peripheral_active = "peripheral_high" in active_reasons
        shape_active = "shape_high" in active_reasons
        highfreq_active = "highfreq_high" in active_reasons
        ratio_active = "peripheral_core_ratio_high" in active_reasons
        local_active_count = int(peripheral_active) + int(shape_active) + int(highfreq_active) + int(ratio_active)

        strong_active = bool(
            (residual_active and local_active_count >= 1)
            or (local_active_count >= 2)
        )
        very_strong_active = bool(
            (residual_active and local_active_count >= 2)
            or (local_active_count >= 3)
            or (peripheral_active and shape_active and (highfreq_active or ratio_active))
        )
        periodic_swing_like = bool(
            _safe_float(row.get("swing_score"), 0.0) >= 3.8
            and _safe_float(row.get("struggle_score"), 0.0) <= 1.6
            and _safe_float(row.get("swing_margin"), 0.0) >= 2.0
            and _safe_float(row.get("acf_peak"), 0.0) >= 0.45
            and _safe_float(row.get("flow_entropy"), 1.0) <= 0.15
        )
        passive_swing = bool(
            (
                self._above(global_motion, "global_mid")
                and self._below_or_equal(residual, "residual_mid")
                and self._below_or_equal(peripheral, "peripheral_mid")
                and self._below_or_equal(shape, "shape_mid")
                and self._above(periodicity, "periodicity_high", self.config.passive_periodicity_threshold)
            )
            or (
                self._above(global_motion, "global_high")
                and self._below_or_equal(residual, "residual_mid")
                and self._below_or_equal(peripheral, "peripheral_mid")
                and self._below_or_equal(shape, "shape_mid")
                and self._below_or_equal(highfreq, "highfreq_high")
            )
            or (
                periodic_swing_like
                and not very_strong_active
            )
        )
        motion_score, motion_components = self._motion_score(
            residual=residual,
            peripheral=peripheral,
            highfreq=highfreq,
            shape=shape,
            flow_p90=flow_p90,
            speed_std=speed_std,
        )
        score_mode_enabled = _safe_bool(self.thresholds.get("score_mode_enabled"), False)
        score_entry = self._threshold("score_mode_entry", self.config.score_mode_entry)
        if score_mode_enabled:
            score_active = bool(motion_score >= score_entry)
            self.state = STATE_STRUGGLE if score_active else STATE_IMMOBILE
            reason = "motion_score_active" if score_active else "motion_score_below_entry"
            confidence = "high" if abs(motion_score - score_entry) >= 0.15 else "medium"
            if bool(row.get("lis_climbing_suspected", False)):
                confidence = "low"
                reason = f"{reason};climbing_suspected"
            return {
                "state": self.state,
                "is_struggling": score_active,
                "confidence": confidence,
                "reason": reason,
                "passive_swing": bool(passive_swing),
                "strong_active": bool(score_active),
                "immobile_evidence": bool(not score_active),
                "active_evidence": bool(score_active),
                "motion_score": float(motion_score),
                "motion_prior": _safe_float(self.thresholds.get("motion_prior"), 0.0),
                "score_mode": True,
                "score_entry": float(score_entry),
                "score_residual": float(motion_components["residual"]),
                "score_peripheral": float(motion_components["peripheral"]),
                "score_highfreq": float(motion_components["highfreq"]),
                "score_shape": float(motion_components["shape"]),
                "score_flow": float(motion_components["flow"]),
                "score_speed_std": float(motion_components["speed_std"]),
            }

        active_evidence = bool(strong_active)
        if passive_swing and not strong_active:
            active_evidence = False
            active_reasons.append("passive_swing_suppressed")
            immobile_evidence = True

        self._append_evidence_window(
            self._immobile_evidence_window,
            immobile_evidence,
            dt,
            self.config.immobile_confirm_s,
        )
        self._append_evidence_window(
            self._active_evidence_window,
            active_evidence,
            dt,
            self.config.active_confirm_s,
        )
        immobile_total_s, immobile_hit_s, immobile_hit_ratio = self._window_hit_ratio(self._immobile_evidence_window)
        active_total_s, active_hit_s, active_hit_ratio = self._window_hit_ratio(self._active_evidence_window)
        self.immobile_duration_s = immobile_hit_s
        self.active_duration_s = active_hit_s
        immobile_confirmed = bool(
            immobile_total_s >= self.config.immobile_confirm_s * 0.80
            and immobile_hit_ratio >= self.config.immobile_entry_hit_ratio
        )
        active_confirmed = bool(
            active_total_s >= self.config.active_confirm_s * 0.80
            and active_hit_ratio >= self.config.active_entry_hit_ratio
        )
        active_released = bool(
            active_total_s >= self.config.active_confirm_s * 0.80
            and active_hit_ratio <= self.config.active_release_hit_ratio
        )

        reason = "state_persistence"
        if self.state == STATE_STRUGGLE:
            if immobile_confirmed or active_released:
                self.state = STATE_IMMOBILE
                reason = _reason_join(immobile_reasons) if immobile_reasons else "active_window_released"
            elif active_reasons:
                reason = _reason_join(active_reasons)
        else:
            if active_confirmed:
                self.state = STATE_STRUGGLE
                reason = _reason_join(active_reasons)
            elif immobile_reasons:
                reason = _reason_join(immobile_reasons)

        confidence = "medium"
        if self.state == STATE_STRUGGLE and (strong_active or len(active_reasons) >= 2):
            confidence = "high"
        elif self.state == STATE_IMMOBILE and immobile_evidence and not active_evidence:
            confidence = "high"
        elif passive_swing:
            confidence = "medium"
        if bool(row.get("lis_climbing_suspected", False)):
            confidence = "low"
            reason = f"{reason};climbing_suspected"

        return {
            "state": self.state,
            "is_struggling": self.state == STATE_STRUGGLE,
            "confidence": confidence,
            "reason": reason,
            "passive_swing": bool(passive_swing),
            "strong_active": bool(strong_active),
            "immobile_evidence": bool(immobile_evidence),
            "active_evidence": bool(active_evidence),
            "motion_score": float(motion_score),
            "motion_prior": _safe_float(self.thresholds.get("motion_prior"), 0.0),
            "score_mode": False,
            "score_entry": float(score_entry),
            "score_residual": float(motion_components["residual"]),
            "score_peripheral": float(motion_components["peripheral"]),
            "score_highfreq": float(motion_components["highfreq"]),
            "score_shape": float(motion_components["shape"]),
            "score_flow": float(motion_components["flow"]),
            "score_speed_std": float(motion_components["speed_std"]),
        }


def _postprocess_bool_runs(
    values: list[bool],
    dts: list[float],
    *,
    min_immobile_s: float,
    min_struggle_s: float,
    merge_gap_s: float,
) -> list[bool]:
    n = len(values)
    if n == 0:
        return []
    out = [bool(v) for v in values]
    durations = [max(float(dt), 0.0) for dt in dts]

    def iter_runs(seq: list[bool]):
        start = 0
        while start < len(seq):
            end = start + 1
            while end < len(seq) and seq[end] == seq[start]:
                end += 1
            yield start, end, seq[start], sum(durations[start:end])
            start = end

    for start, end, value, duration_s in list(iter_runs(out)):
        if (
            (not value)
            and duration_s < float(min_immobile_s)
            and start > 0
            and end < n
            and out[start - 1]
            and out[end]
        ):
            for idx in range(start, end):
                out[idx] = True

    for start, end, value, duration_s in list(iter_runs(out)):
        if value and duration_s < float(min_struggle_s) and start > 0 and end < n and (not out[start - 1]) and (not out[end]):
            for idx in range(start, end):
                out[idx] = False

    for start, end, value, duration_s in list(iter_runs(out)):
        if value and duration_s <= float(merge_gap_s) and start > 0 and end < n and (not out[start - 1]) and (not out[end]):
            for idx in range(start, end):
                out[idx] = False

    return out


def _extend_true_runs(
    values: list[bool],
    dts: list[float],
    *,
    retro_s: float = 0.0,
    hold_s: float = 0.0,
) -> list[bool]:
    n = len(values)
    if n == 0:
        return []
    out = [bool(v) for v in values]
    durations = [max(float(dt), 0.0) for dt in dts]
    runs: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and out[end] == out[start]:
            end += 1
        if out[start]:
            runs.append((start, end))
        start = end

    for start, end in runs:
        remaining = max(float(retro_s), 0.0)
        idx = start - 1
        while idx >= 0 and remaining > 0:
            out[idx] = True
            remaining -= durations[idx]
            idx -= 1

        remaining = max(float(hold_s), 0.0)
        idx = end
        while idx < n and remaining > 0:
            out[idx] = True
            remaining -= durations[idx]
            idx += 1

    return out


def apply_tst_lis_labels(
    logs: list[dict[str, Any]],
    *,
    fps: float = 30.0,
    config: TSTLISConfig | None = None,
) -> dict[int, dict[str, Any]]:
    cfg = config or TSTLISConfig()
    rows_by_roi: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(logs):
        if "lis_residual_motion" not in row:
            continue
        try:
            roi_id = int(row.get("roi_id"))
        except Exception:
            continue
        rows_by_roi[roi_id].append((idx, row))

    thresholds_by_roi: dict[int, dict[str, Any]] = {}
    for roi_id, indexed_rows in rows_by_roi.items():
        indexed_rows.sort(key=lambda item: (_safe_float(item[1].get("time"), 0.0), int(item[1].get("frame", 0) or 0)))
        rows = [row for _, row in indexed_rows if bool(row.get("is_in_stats_window", True))]
        if not rows:
            rows = [row for _, row in indexed_rows]
        thresholds = estimate_tst_lis_thresholds(rows, cfg)
        thresholds_by_roi[roi_id] = thresholds
        machine = TSTLISStateMachine(thresholds, fps=fps, config=cfg)
        raw_values: list[bool] = []
        dts: list[float] = []
        updates: list[dict[str, Any]] = []
        appendage_raw_values: list[bool] = []
        for _, row in indexed_rows:
            if not bool(row.get("is_in_stats_window", True)):
                update = {
                    "is_struggling": False,
                    "state": STATE_IMMOBILE,
                    "confidence": "low",
                    "reason": "outside_stats_window",
                    "passive_swing": False,
                    "strong_active": False,
                    "immobile_evidence": False,
                    "active_evidence": False,
                }
            else:
                update = machine.update(row)
            updates.append(update)
            raw_values.append(bool(update["is_struggling"]))
            dts.append(max(_safe_float(row.get("dt_s"), 1.0 / max(float(fps), 1e-6)), 0.0))
            motion_prior = _safe_float(update.get("motion_prior"), _safe_float(thresholds.get("motion_prior"), 0.0))
            appendage_base = bool(
                cfg.appendage_rescue_enabled
                and not _safe_bool(thresholds.get("score_mode_enabled"), False)
                and motion_prior >= float(cfg.appendage_rescue_prior_min)
                and motion_prior <= float(cfg.appendage_rescue_prior_max)
            )
            motion_score = _safe_float(update.get("motion_score"), 0.0)
            flow_p90 = _safe_float(row.get("decision_flow_p90_mm_s"), 0.0)
            speed_std = _safe_float(row.get("speed_std_mm_s"), 0.0)
            aspect_ratio = _safe_float(row.get("lis_aspect_ratio"), 0.0)
            appendage_strong_raw = bool(
                appendage_base
                and aspect_ratio >= float(cfg.appendage_rescue_aspect_min)
                and motion_score >= float(cfg.appendage_rescue_score_entry)
                and flow_p90 >= float(cfg.appendage_rescue_flow_min)
                and speed_std <= float(cfg.appendage_rescue_speed_std_max)
            )
            appendage_front_raw = bool(
                appendage_base
                and motion_score >= float(cfg.appendage_rescue_front_score_entry)
                and _safe_float(row.get("lis_residual_motion"), 0.0) >= float(cfg.appendage_rescue_front_residual_min)
                and _safe_float(row.get("lis_peripheral_motion"), 0.0) >= float(cfg.appendage_rescue_front_peripheral_min)
                and _safe_float(row.get("lis_highfreq_motion"), 0.0) >= float(cfg.appendage_rescue_front_highfreq_min)
                and flow_p90 >= float(cfg.appendage_rescue_front_flow_min)
                and _safe_float(row.get("lis_periodicity"), 0.0) >= float(cfg.appendage_rescue_front_periodicity_min)
                and _safe_float(row.get("lis_global_motion_norm"), 0.0) <= float(cfg.appendage_rescue_front_global_max)
                and aspect_ratio >= float(cfg.appendage_rescue_front_aspect_min)
                and speed_std <= float(cfg.appendage_rescue_front_speed_std_max)
            )
            appendage_periodic_raw = bool(
                appendage_base
                and aspect_ratio >= float(cfg.appendage_rescue_periodic_aspect_min)
                and motion_score >= float(cfg.appendage_rescue_periodic_score_entry)
                and _safe_float(row.get("lis_residual_motion"), 0.0) >= float(cfg.appendage_rescue_periodic_residual_min)
                and _safe_float(row.get("lis_peripheral_motion"), 0.0) >= float(cfg.appendage_rescue_periodic_peripheral_min)
                and _safe_float(row.get("lis_highfreq_motion"), 0.0) >= float(cfg.appendage_rescue_periodic_highfreq_min)
                and flow_p90 >= float(cfg.appendage_rescue_periodic_flow_min)
                and _safe_float(row.get("lis_periodicity"), 0.0) >= float(cfg.appendage_rescue_periodic_periodicity_min)
                and _safe_float(row.get("lis_global_motion_norm"), 0.0) <= float(cfg.appendage_rescue_periodic_global_max)
                and speed_std <= float(cfg.appendage_rescue_periodic_speed_std_max)
            )
            appendage_raw_values.append(bool(appendage_strong_raw or appendage_front_raw or appendage_periodic_raw))
            update["appendage_rescue_raw"] = bool(appendage_strong_raw or appendage_front_raw or appendage_periodic_raw)
            update["appendage_front_rescue_raw"] = bool(appendage_front_raw)
            update["appendage_periodic_rescue_raw"] = bool(appendage_periodic_raw)

        post_values = _postprocess_bool_runs(
            raw_values,
            dts,
            min_immobile_s=_safe_float(thresholds.get("postprocess_fill_gap_s"), cfg.min_immobile_bout_s),
            min_struggle_s=cfg.min_struggle_bout_s,
            merge_gap_s=cfg.merge_gap_s,
        )
        appendage_values = _postprocess_bool_runs(
            appendage_raw_values,
            dts,
            min_immobile_s=cfg.appendage_rescue_fill_gap_s,
            min_struggle_s=cfg.min_struggle_bout_s,
            merge_gap_s=cfg.merge_gap_s,
        )
        appendage_values = _extend_true_runs(
            appendage_values,
            dts,
            retro_s=cfg.appendage_rescue_retro_s,
            hold_s=cfg.appendage_rescue_hold_s,
        )
        for (idx, row), update, post_value, appendage_value in zip(indexed_rows, updates, post_values, appendage_values):
            model_value = bool(post_value or appendage_value)
            final_value = model_value
            reason = str(update["reason"])
            if appendage_value and not bool(post_value):
                reason = f"{reason};appendage_rescue"
            row["lis_online_is_struggling"] = bool(update["is_struggling"])
            row["lis_model_is_struggling"] = model_value
            row["lis_is_struggling"] = final_value
            row["lis_postprocessed_is_struggling"] = final_value
            row["lis_state"] = STATE_STRUGGLE if final_value else STATE_IMMOBILE
            row["lis_confidence"] = update["confidence"]
            row["lis_reason"] = reason
            row["lis_passive_swing"] = bool(update["passive_swing"])
            row["lis_strong_active"] = bool(update["strong_active"])
            row["lis_immobile_evidence"] = bool(update["immobile_evidence"])
            row["lis_active_evidence"] = bool(update["active_evidence"])
            row["lis_appendage_rescue"] = bool(appendage_value)
            row["lis_appendage_rescue_raw"] = bool(update.get("appendage_rescue_raw", False))
            row["lis_appendage_front_rescue_raw"] = bool(update.get("appendage_front_rescue_raw", False))
            row["lis_appendage_periodic_rescue_raw"] = bool(update.get("appendage_periodic_rescue_raw", False))
            row["lis_motion_score"] = round(_safe_float(update.get("motion_score"), 0.0), 6)
            row["lis_motion_prior"] = round(_safe_float(update.get("motion_prior"), _safe_float(thresholds.get("motion_prior"), 0.0)), 6)
            row["lis_score_mode"] = bool(update.get("score_mode", False))
            row["lis_score_entry"] = round(_safe_float(update.get("score_entry"), _safe_float(thresholds.get("score_mode_entry"), 0.0)), 6)
            row["lis_score_residual"] = round(_safe_float(update.get("score_residual"), 0.0), 6)
            row["lis_score_peripheral"] = round(_safe_float(update.get("score_peripheral"), 0.0), 6)
            row["lis_score_highfreq"] = round(_safe_float(update.get("score_highfreq"), 0.0), 6)
            row["lis_score_shape"] = round(_safe_float(update.get("score_shape"), 0.0), 6)
            row["lis_score_flow"] = round(_safe_float(update.get("score_flow"), 0.0), 6)
            row["lis_score_speed_std"] = round(_safe_float(update.get("score_speed_std"), 0.0), 6)
            row["lis_threshold_residual_low"] = round(_safe_float(thresholds.get("residual_low"), 0.0), 6)
            row["lis_threshold_residual_high"] = round(_safe_float(thresholds.get("residual_high"), 0.0), 6)
            row["lis_threshold_peripheral_high"] = round(_safe_float(thresholds.get("peripheral_high"), 0.0), 6)
            row["lis_threshold_shape_high"] = round(_safe_float(thresholds.get("shape_high"), 0.0), 6)
            row["lis_threshold_score_entry"] = round(_safe_float(thresholds.get("score_mode_entry"), 0.0), 6)
            row["lis_threshold_postprocess_fill_gap_s"] = round(_safe_float(thresholds.get("postprocess_fill_gap_s"), cfg.min_immobile_bout_s), 6)
            row["is_struggling"] = final_value
            row["is_swinging"] = False
            row["behavior_algorithm"] = "tst_lis"
            logs[idx] = row

    return thresholds_by_roi
