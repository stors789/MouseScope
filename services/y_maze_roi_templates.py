from __future__ import annotations

import copy
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from services.atomic_json import atomic_write_json, read_json


class RoiTemplateValidationError(ValueError):
    pass


_NAME_WHITESPACE = re.compile(r"\s+")
_ZONE_KEYS = ("center", "arm_a", "arm_b", "arm_c")
_GEOMETRY_KEYS = ("arm_length_cm", "arm_width_cm")


def _clean_name(value: Any) -> str:
    name = _NAME_WHITESPACE.sub(" ", str(value or "").strip())
    if not name:
        raise RoiTemplateValidationError("模板名称不能为空。")
    if len(name) > 80:
        raise RoiTemplateValidationError("模板名称不能超过 80 个字符。")
    return name


def _point_list(value: Any, *, minimum: int, maximum: int) -> list[dict[str, float]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise RoiTemplateValidationError(f"ROI 点数必须在 {minimum} 到 {maximum} 之间。")
    points: list[dict[str, float]] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise RoiTemplateValidationError("ROI 坐标格式无效。")
        try:
            x, y = float(raw["x"]), float(raw["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RoiTemplateValidationError("ROI 坐标必须是数字。") from exc
        if not (-1_000_000 <= x <= 1_000_000 and -1_000_000 <= y <= 1_000_000):
            raise RoiTemplateValidationError("ROI 坐标超出允许范围。")
        points.append({"x": x, "y": y})
    return points


def normalize_template_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RoiTemplateValidationError("ROI 模板格式无效。")
    raw_rois = payload.get("rois")
    if not isinstance(raw_rois, list) or not raw_rois:
        raise RoiTemplateValidationError("请先绘制至少一个完整 ROI。")
    if len(raw_rois) > 32:
        raise RoiTemplateValidationError("单个模板最多保存 32 个 ROI。")

    rois: list[dict[str, Any]] = []
    for index, raw_roi in enumerate(raw_rois):
        if not isinstance(raw_roi, dict):
            raise RoiTemplateValidationError("ROI 格式无效。")
        raw_zones = raw_roi.get("zones")
        if not isinstance(raw_zones, dict):
            raise RoiTemplateValidationError(f"ROI {index + 1} 缺少区域。")
        zones: dict[str, list[dict[str, float]]] = {}
        for zone_key in _ZONE_KEYS:
            expected = 3 if zone_key == "center" else 4
            points = _point_list(raw_zones.get(zone_key), minimum=expected, maximum=expected)
            if points is None:
                raise RoiTemplateValidationError(f"ROI {index + 1} 的 {zone_key} 尚未完成。")
            zones[zone_key] = points
        detection_points = _point_list(
            raw_roi.get("detection_points"), minimum=3, maximum=200
        )
        rois.append(
            {
                "name": str(raw_roi.get("name") or f"Y-Maze {index + 1}")[:120],
                "zones": zones,
                "detection_points": detection_points,
            }
        )

    geometry: dict[str, float] = {}
    raw_geometry = payload.get("geometry")
    if isinstance(raw_geometry, dict):
        for key in _GEOMETRY_KEYS:
            if raw_geometry.get(key) is None:
                continue
            try:
                number = float(raw_geometry[key])
            except (TypeError, ValueError) as exc:
                raise RoiTemplateValidationError("迷宫尺寸必须是数字。") from exc
            if not 0 < number <= 10_000:
                raise RoiTemplateValidationError("迷宫尺寸超出允许范围。")
            geometry[key] = number
    return {"rois": rois, "geometry": geometry}


class YMazeRoiTemplateStore:
    """Application-level, disk-backed Y-maze device ROI templates."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = read_json(self.path)
        except Exception:
            return []
        templates = data.get("templates", []) if isinstance(data, dict) else []
        return templates if isinstance(templates, list) else []

    def list_templates(self) -> list[dict[str, Any]]:
        with self._lock:
            templates = copy.deepcopy(self._read())
        return sorted(templates, key=lambda item: float(item.get("updated_at") or 0), reverse=True)

    def save_template(self, name: Any, payload: Any) -> dict[str, Any]:
        clean_name = _clean_name(name)
        normalized = normalize_template_payload(payload)
        now = time.time()
        with self._lock:
            templates = self._read()
            existing = next(
                (item for item in templates if str(item.get("name", "")).casefold() == clean_name.casefold()),
                None,
            )
            if existing is None:
                existing = {
                    "id": uuid.uuid4().hex,
                    "created_at": now,
                }
                templates.append(existing)
            existing.update(
                {
                    "name": clean_name,
                    "updated_at": now,
                    **normalized,
                }
            )
            atomic_write_json(self.path, {"version": 1, "templates": templates})
            return copy.deepcopy(existing)

    def delete_template(self, template_id: str) -> bool:
        template_id = str(template_id or "").strip()
        if not template_id:
            return False
        with self._lock:
            templates = self._read()
            kept = [item for item in templates if str(item.get("id")) != template_id]
            if len(kept) == len(templates):
                return False
            atomic_write_json(self.path, {"version": 1, "templates": kept})
        return True
