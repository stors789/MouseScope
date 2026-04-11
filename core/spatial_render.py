#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


CLASSIC_BG = (255, 255, 255)
CLASSIC_BORDER = (214, 221, 229)
CLASSIC_HEAT_OUTLINE = (248, 248, 248)
CLASSIC_GUIDE_OUTLINE = (86, 152, 245)
CLASSIC_GUIDE_FILL = (223, 236, 255)
CLASSIC_TRACK = (92, 158, 245)
CLASSIC_START = (94, 197, 34)
CLASSIC_END = (246, 130, 59)
CLASSIC_TEXT = (38, 38, 38)
CLASSIC_TEXT_SOFT = (110, 110, 110)


def classic_heatmap(norm: np.ndarray) -> np.ndarray:
    scaled = np.clip(np.asarray(norm, dtype=np.float32), 0.0, 1.0)
    scaled = np.power(scaled, 0.74)
    heat_u8 = np.uint8(np.clip(np.round(scaled * 255.0), 0, 255))
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    color[heat_u8 <= 3] = np.array([118, 0, 0], dtype=np.uint8)
    return color


def classic_border_thickness(pixels_per_cm: int) -> int:
    return max(2, int(round(pixels_per_cm / 6.0)))


def classic_guide_thickness(pixels_per_cm: int) -> int:
    return max(2, classic_border_thickness(pixels_per_cm) - 1)


def classic_path_thickness(pixels_per_cm: int) -> int:
    return max(1, int(round(pixels_per_cm / 9.0)))


def classic_marker_radius(pixels_per_cm: int) -> int:
    return max(4, int(round(pixels_per_cm / 3.5)))


def fill_rect_alpha(
    image: np.ndarray,
    top_left: Tuple[int, int],
    bottom_right: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.82,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=-1)
    cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0.0, dst=image)


def fill_poly_alpha(
    image: np.ndarray,
    polygon: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.82,
) -> None:
    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon.astype(np.int32)], color)
    cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0.0, dst=image)


def draw_start_end_markers(
    image: np.ndarray,
    points: Sequence[Tuple[int, int]],
    radius: int,
) -> None:
    if not points:
        return
    start = tuple(int(v) for v in points[0])
    end = tuple(int(v) for v in points[-1])
    cv2.circle(image, start, radius, CLASSIC_START, -1, cv2.LINE_AA)
    cv2.circle(image, start, radius + 1, CLASSIC_BG, 1, cv2.LINE_AA)
    cv2.circle(image, end, radius, CLASSIC_END, -1, cv2.LINE_AA)
    cv2.circle(image, end, radius + 1, CLASSIC_BG, 1, cv2.LINE_AA)
