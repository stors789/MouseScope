#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Sequence, Tuple

import cv2
import numpy as np

try:
    from core.visual_style import apply_visual_colormap, normalize_visual_style, style_color
except ImportError:  # pragma: no cover - supports direct imports with core/ on sys.path
    from visual_style import apply_visual_colormap, normalize_visual_style, style_color


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


def classic_heatmap(norm: np.ndarray, style: Any | None = None) -> np.ndarray:
    return apply_visual_colormap(norm, style)


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
    style: Any | None = None,
) -> None:
    if not points:
        return
    start = tuple(int(v) for v in points[0])
    end = tuple(int(v) for v in points[-1])
    start_color = style_color(style, "trackmap", "start")
    end_color = style_color(style, "trackmap", "end")
    bg_color = style_color(style, "trackmap", "background")
    cv2.circle(image, start, radius, start_color, -1, cv2.LINE_AA)
    cv2.circle(image, start, radius + 1, bg_color, 1, cv2.LINE_AA)
    cv2.circle(image, end, radius, end_color, -1, cv2.LINE_AA)
    cv2.circle(image, end, radius + 1, bg_color, 1, cv2.LINE_AA)


def append_heatmap_legend(image: np.ndarray, style: Any | None = None) -> np.ndarray:
    normalized = normalize_visual_style(style)
    heat_style = normalized["heatmap"]
    if not heat_style.get("show_legend", True):
        return image

    h, w = image.shape[:2]
    legend_w = 96
    pad = min(14, max(4, h // 8))
    bar_w = 18
    label_gap = 8
    bg = style_color(normalized, "heatmap", "legend_background")
    border = style_color(normalized, "heatmap", "legend_border")
    text = style_color(normalized, "heatmap", "legend_text")

    canvas = np.full((h, w + legend_w, 3), bg, dtype=np.uint8)
    canvas[:, :w] = image
    cv2.rectangle(canvas, (w, 0), (w + legend_w - 1, h - 1), border, 1, cv2.LINE_AA)

    available_h = max(10, h - pad * 2)
    bar_h = min(max(60, h - pad * 2 - 26), available_h)
    x0 = w + pad
    y0 = pad
    gradient = np.linspace(1.0, 0.0, bar_h, dtype=np.float32).reshape(bar_h, 1)
    bar = classic_heatmap(gradient, normalized)
    bar = cv2.resize(bar, (bar_w, bar_h), interpolation=cv2.INTER_NEAREST)
    canvas[y0 : y0 + bar_h, x0 : x0 + bar_w] = bar
    cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y0 + bar_h), border, 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    high = str(heat_style.get("legend_high_label") or "High")[:16]
    low = str(heat_style.get("legend_low_label") or "Low")[:16]
    cv2.putText(canvas, high, (x0 + bar_w + label_gap, y0 + 10), font, 0.38, text, 1, cv2.LINE_AA)
    cv2.putText(canvas, low, (x0 + bar_w + label_gap, y0 + bar_h), font, 0.38, text, 1, cv2.LINE_AA)
    return canvas
