#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import re
from typing import Any, Mapping, Sequence, Tuple

import cv2
import numpy as np


HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

COLORMAP_CHOICES: dict[str, dict[str, Any]] = {
    "jet": {"label": "Jet", "opencv": "COLORMAP_JET"},
    "hot": {"label": "Hot", "opencv": "COLORMAP_HOT"},
    "turbo": {"label": "Turbo", "opencv": "COLORMAP_TURBO"},
    "viridis": {"label": "Viridis", "opencv": "COLORMAP_VIRIDIS"},
    "plasma": {"label": "Plasma", "opencv": "COLORMAP_PLASMA"},
    "inferno": {"label": "Inferno", "opencv": "COLORMAP_INFERNO"},
    "magma": {"label": "Magma", "opencv": "COLORMAP_MAGMA"},
    "cividis": {"label": "Cividis", "opencv": "COLORMAP_CIVIDIS"},
    "custom": {"label": "Custom Low-High", "opencv": None},
}

DEFAULT_VISUAL_STYLE: dict[str, Any] = {
    "version": 1,
    "trackmap": {
        "background": "#ffffff",
        "border": "#d6dde5",
        "region_fill": "#dfecff",
        "region_outline": "#5698f5",
        "special_fill": "#e2f2e3",
        "special_outline": "#5eaa6a",
        "dark_region_fill": "#d8dce1",
        "track": "#5c9ef5",
        "start": "#5ec522",
        "end": "#f6823b",
        "text": "#262626",
        "text_soft": "#6e6e6e",
        "object_familiar": "#a7794e",
        "object_novel": "#2b8ef2",
        "object_neutral": "#94a3b8",
        "platform": "#5eaa6a",
        "thigmotaxis": "#d9aa58",
        "grid_background": "#ffffff",
        "grid_title": "#1e293b",
        "grid_border": "#e2e8f0",
    },
    "heatmap": {
        "colormap": "jet",
        "custom_low": "#000076",
        "custom_high": "#ff0000",
        "zero_color": "#000076",
        "outside": "#ffffff",
        "outline": "#f8f8f8",
        "border": "#d6dde5",
        "legend_background": "#ffffff",
        "legend_border": "#d6dde5",
        "legend_text": "#262626",
        "legend_low_label": "Low",
        "legend_high_label": "High",
        "show_legend": True,
    },
}


def _deep_merge(default: Any, incoming: Any) -> Any:
    if isinstance(default, dict):
        merged = copy.deepcopy(default)
        if isinstance(incoming, Mapping):
            for key, value in incoming.items():
                if key in merged:
                    merged[key] = _deep_merge(merged[key], value)
        return merged
    return copy.deepcopy(incoming) if incoming is not None else copy.deepcopy(default)


def _clean_hex(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text.startswith("#") and re.fullmatch(r"[0-9a-fA-F]{6}", text):
        text = f"#{text}"
    return text.lower() if HEX_RE.match(text) else fallback


def normalize_visual_style(value: Any | None) -> dict[str, Any]:
    style = _deep_merge(DEFAULT_VISUAL_STYLE, value if isinstance(value, Mapping) else {})
    for section_name, section in DEFAULT_VISUAL_STYLE.items():
        if not isinstance(section, dict):
            continue
        current = style.get(section_name) if isinstance(style.get(section_name), dict) else {}
        for key, default_value in section.items():
            if isinstance(default_value, str) and default_value.startswith("#"):
                current[key] = _clean_hex(current.get(key), default_value)
        style[section_name] = current

    heatmap = style["heatmap"]
    cmap = str(heatmap.get("colormap") or "jet").strip().lower()
    if cmap not in COLORMAP_CHOICES:
        cmap = "jet"
    heatmap["colormap"] = cmap
    heatmap["show_legend"] = bool(heatmap.get("show_legend", True))
    for label_key in ("legend_low_label", "legend_high_label"):
        label = str(heatmap.get(label_key) or "").strip()
        heatmap[label_key] = label[:16] or DEFAULT_VISUAL_STYLE["heatmap"][label_key]
    style["version"] = int(DEFAULT_VISUAL_STYLE["version"])
    return style


def style_payload(style: Any | None = None) -> dict[str, Any]:
    return {
        "style": normalize_visual_style(style),
        "defaults": copy.deepcopy(DEFAULT_VISUAL_STYLE),
        "colormaps": [
            {"key": key, "label": value["label"]}
            for key, value in COLORMAP_CHOICES.items()
        ],
    }


def hex_to_bgr(value: str, fallback: str = "#000000") -> Tuple[int, int, int]:
    text = _clean_hex(value, fallback).lstrip("#")
    r = int(text[0:2], 16)
    g = int(text[2:4], 16)
    b = int(text[4:6], 16)
    return b, g, r


def bgr_to_hex(color: Sequence[int]) -> str:
    b, g, r = [int(np.clip(v, 0, 255)) for v in color[:3]]
    return f"#{r:02x}{g:02x}{b:02x}"


def style_color(style: Any | None, section: str, key: str) -> Tuple[int, int, int]:
    normalized = normalize_visual_style(style)
    fallback = DEFAULT_VISUAL_STYLE.get(section, {}).get(key, "#000000")
    value = normalized.get(section, {}).get(key, fallback)
    return hex_to_bgr(value, str(fallback))


def apply_visual_colormap(norm: np.ndarray, style: Any | None = None) -> np.ndarray:
    normalized = normalize_visual_style(style)
    heatmap_style = normalized["heatmap"]
    scaled = np.clip(np.asarray(norm, dtype=np.float32), 0.0, 1.0)
    scaled = np.power(scaled, 0.74)
    heat_u8 = np.uint8(np.clip(np.round(scaled * 255.0), 0, 255))
    cmap = str(heatmap_style.get("colormap") or "jet").lower()

    if cmap == "custom":
        low = np.array(hex_to_bgr(heatmap_style["custom_low"]), dtype=np.float32)
        high = np.array(hex_to_bgr(heatmap_style["custom_high"]), dtype=np.float32)
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for idx in range(256):
            t = idx / 255.0
            lut[idx, 0] = np.clip(np.round(low * (1.0 - t) + high * t), 0, 255)
        color = cv2.applyColorMap(heat_u8, lut)
    else:
        cv_name = COLORMAP_CHOICES.get(cmap, COLORMAP_CHOICES["jet"]).get("opencv") or "COLORMAP_JET"
        cv_id = getattr(cv2, cv_name, cv2.COLORMAP_JET)
        color = cv2.applyColorMap(heat_u8, cv_id)

    color[heat_u8 <= 3] = np.array(style_color(normalized, "heatmap", "zero_color"), dtype=np.uint8)
    return color
