#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Filesystem helpers for Windows/OpenCV Unicode compatibility."""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

_ASCII_RE = re.compile(r"[^0-9A-Za-z._-]+")
_WINDOWS_INVALID_RE = re.compile(r'[<>:"/\\\\|?*\x00-\x1f]+')


def contains_non_ascii(value: object) -> bool:
    try:
        str(value).encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def clean_display_name(text: object, fallback: str = "file", max_length: int = 120) -> str:
    cleaned = _WINDOWS_INVALID_RE.sub("_", str(text or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(". ")
    return cleaned[:max_length] or fallback


def clean_display_stem(text: object, fallback: str = "file", max_length: int = 96) -> str:
    stem = Path(str(text or "")).stem
    return clean_display_name(stem, fallback=fallback, max_length=max_length)


def ascii_slug(text: object, fallback: str = "file", max_length: int = 80) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = _ASCII_RE.sub("_", ascii_text).strip("._-")
    return cleaned[:max_length] or fallback


def build_ascii_file_name(
    original_name: object,
    prefix: Optional[str] = None,
    fallback: str = "file",
    max_stem_length: int = 48,
) -> str:
    original_name = str(original_name or "")
    path = Path(original_name)
    ext = path.suffix.lower()
    safe_stem = ascii_slug(path.stem, fallback=fallback, max_length=max_stem_length)
    digest = hashlib.sha1(original_name.encode("utf-8", "ignore")).hexdigest()[:10]

    parts = []
    if prefix:
        parts.append(ascii_slug(prefix, fallback="session", max_length=24))
    parts.append(safe_stem)
    parts.append(digest)
    return "_".join(parts) + ext


def write_image_unicode(path: object, image: np.ndarray, params: Optional[Sequence[int]] = None) -> bool:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = output_path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image, list(params or []))
    if not ok:
        return False

    try:
        encoded.tofile(str(output_path))
        return output_path.exists()
    except Exception:
        return False


def read_image_unicode(path: object, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except Exception:
        return None

    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def normalized_path_key(path: object) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))
