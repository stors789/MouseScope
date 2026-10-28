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
_ASSAY_ALIASES = {
    "tst": "tst",
    "forced_swim": "forced_swim",
    "fst": "forced_swim",
    "water_maze": "water_maze",
    "mwm": "water_maze",
    "barnes_maze": "barnes_maze",
    "barnes": "barnes_maze",
    "open_field": "open_field",
    "oft": "open_field",
    "novel_object": "novel_object",
    "nor": "novel_object",
    "ymaze": "y_maze",
    "y_maze": "y_maze",
    "epm": "elevated_plus",
    "elevated_plus": "elevated_plus",
    "light_dark_box": "light_dark_box",
    "ldb": "light_dark_box",
    "three_chamber": "three_chamber",
}
_ASSAY_SCHEMAS = {
    "tst": {
        "point_fields": {"points": (3, 200, True)},
        "geometry": (),
        "roi_label": "ROI",
    },
    "forced_swim": {
        "point_fields": {
            "points": (3, 200, True),
            "diameter_points": (2, 2, True),
            "waterline_points": (2, 2, False),
        },
        "geometry": ("cylinder_diameter_cm",),
        "roi_scalars": ("cylinder_diameter_cm",),
        "roi_label": "FST",
    },
    "water_maze": {
        "special": "water_maze",
        "geometry": ("pool_diameter_cm", "platform_radius_cm"),
        "roi_scalars": ("pool_diameter_cm", "platform_radius_cm"),
        "roi_label": "MWM",
    },
    "barnes_maze": {
        "special": "barnes_maze",
        "point_fields": {"arena_points": (5, 200, True)},
        "geometry": ("arena_diameter_cm", "hole_radius_cm", "hole_count"),
        "roi_scalars": ("arena_diameter_cm", "hole_radius_cm", "hole_count"),
        "roi_label": "Barnes",
    },
    "open_field": {
        "point_fields": {"points": (4, 4, True)},
        "geometry": ("arena_side_cm", "arena_width_cm", "arena_height_cm"),
        "roi_label": "OFT",
    },
    "novel_object": {
        "point_fields": {"points": (4, 4, True)},
        "special": "novel_object",
        "geometry": (
            "arena_side_cm",
            "arena_width_cm",
            "arena_height_cm",
            "interaction_radius_cm",
        ),
        "roi_label": "NOR",
    },
    "y_maze": {
        "zones": {"center": 3, "arm_a": 4, "arm_b": 4, "arm_c": 4},
        "geometry": ("arm_length_cm", "arm_width_cm"),
        "roi_label": "Y-Maze",
    },
    "elevated_plus": {
        "zones": {"center": 4, "open_1": 4, "open_2": 4, "closed_1": 4, "closed_2": 4},
        "geometry": ("center_size_cm", "open_arm_length_cm", "closed_arm_length_cm"),
        "roi_label": "EPM",
    },
    "light_dark_box": {
        "zones": {"light": (3, 200)},
        "optional_zones": {"dark": (3, 200), "doorway": (3, 200)},
        "geometry": (
            "light_box_length_cm",
            "light_box_width_cm",
            "dark_box_length_cm",
            "dark_box_width_cm",
            "shared_edge_length_cm",
        ),
        "roi_label": "LDB",
    },
    "three_chamber": {
        "zones": {"left": (3, 200), "center": (3, 200), "right": (3, 200)},
        "optional_zones": {"left_door": (3, 200), "right_door": (3, 200)},
        "special": "three_chamber",
        "geometry": (
            "arena_width_cm",
            "arena_height_cm",
            "interaction_radius_cm",
        ),
        "roi_label": "3CST",
    },
}


def canonical_assay(value: Any) -> str:
    assay = _ASSAY_ALIASES.get(str(value or "").strip().lower())
    if assay is None:
        raise RoiTemplateValidationError("不支持此实验类型的 ROI 模板。")
    return assay


def _clean_name(value: Any) -> str:
    name = _NAME_WHITESPACE.sub(" ", str(value or "").strip())
    if not name:
        raise RoiTemplateValidationError("模板名称不能为空。")
    if len(name) > 80:
        raise RoiTemplateValidationError("模板名称不能超过 80 个字符。")
    return name


def _source_frame(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RoiTemplateValidationError("保存模板时需要有效的源画面尺寸。")
    try:
        width = int(value["width"])
        height = int(value["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RoiTemplateValidationError("源画面宽高必须是整数。") from exc
    if not (1 <= width <= 100_000 and 1 <= height <= 100_000):
        raise RoiTemplateValidationError("源画面尺寸超出允许范围。")
    try:
        orientation = int(value.get("orientation_degrees") or 0) % 360
    except (TypeError, ValueError) as exc:
        raise RoiTemplateValidationError("画面方向无效。") from exc
    if orientation not in {0, 90, 180, 270}:
        raise RoiTemplateValidationError("画面方向只支持 0/90/180/270 度。")
    return {
        "width": width,
        "height": height,
        "orientation_degrees": orientation,
        "aspect_ratio": width / float(height),
    }


def _point_list(
    value: Any,
    *,
    minimum: int,
    maximum: int,
    source: dict[str, Any],
    already_normalized: bool = False,
) -> list[dict[str, float]] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise RoiTemplateValidationError(f"ROI 点数必须在 {minimum} 到 {maximum} 之间。")
    points: list[dict[str, float]] = []
    width, height = float(source["width"]), float(source["height"])
    for raw in value:
        if not isinstance(raw, dict):
            raise RoiTemplateValidationError("ROI 坐标格式无效。")
        try:
            x, y = float(raw["x"]), float(raw["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RoiTemplateValidationError("ROI 坐标必须是数字。") from exc
        nx, ny = (x, y) if already_normalized else (x / width, y / height)
        # Small tolerance permits points drawn exactly on antialiased canvas edges.
        if not (-0.01 <= nx <= 1.01 and -0.01 <= ny <= 1.01):
            raise RoiTemplateValidationError("ROI 坐标超出源画面范围。")
        points.append({"x": min(1.0, max(0.0, nx)), "y": min(1.0, max(0.0, ny))})
    return points


def normalize_template_payload(assay: Any, payload: Any) -> dict[str, Any]:
    assay_key = canonical_assay(assay)
    if not isinstance(payload, dict):
        raise RoiTemplateValidationError("ROI 模板格式无效。")
    if assay_key == "barnes_maze":
        payload = {**payload, "settings": {}}
    source = _source_frame(payload.get("source_frame"))
    raw_rois = payload.get("rois")
    if not isinstance(raw_rois, list) or not raw_rois:
        raise RoiTemplateValidationError("请先绘制至少一个完整 ROI。")
    if len(raw_rois) > 32:
        raise RoiTemplateValidationError("单个模板最多保存 32 个 ROI。")

    schema = _ASSAY_SCHEMAS[assay_key]
    already_normalized = str(payload.get("coordinate_space") or "").lower() == "normalized"
    rois: list[dict[str, Any]] = []
    for index, raw_roi in enumerate(raw_rois):
        if not isinstance(raw_roi, dict):
            raise RoiTemplateValidationError(f"ROI {index + 1} 格式无效。")
        zones: dict[str, list[dict[str, float]]] = {}
        zone_schema = schema.get("zones") or {}
        if zone_schema and not isinstance(raw_roi.get("zones"), dict):
            raise RoiTemplateValidationError(f"ROI {index + 1} 缺少区域。")
        for zone_key, expected_value in zone_schema.items():
            minimum, maximum = (
                expected_value
                if isinstance(expected_value, tuple)
                else (expected_value, expected_value)
            )
            points = _point_list(
                raw_roi["zones"].get(zone_key),
                minimum=minimum,
                maximum=maximum,
                source=source,
                already_normalized=already_normalized,
            )
            if points is None:
                raise RoiTemplateValidationError(f"ROI {index + 1} 的 {zone_key} 尚未完成。")
            zones[zone_key] = points
        for zone_key, bounds in (schema.get("optional_zones") or {}).items():
            raw_points = (raw_roi.get("zones") or {}).get(zone_key)
            if not raw_points:
                continue
            zones[zone_key] = _point_list(
                raw_points,
                minimum=bounds[0],
                maximum=bounds[1],
                source=source,
                already_normalized=already_normalized,
            )
        detection_points = _point_list(
            raw_roi.get("detection_points") or None,
            minimum=3,
            maximum=200,
            source=source,
            already_normalized=already_normalized,
        )
        roi: dict[str, Any] = {
            "name": str(raw_roi.get("name") or f"{schema['roi_label']} {index + 1}")[:120],
            "detection_points": detection_points,
        }
        if schema.get("special") == "water_maze":
            def clean_ellipse(raw_shape: Any, label: str) -> dict[str, Any]:
                if not isinstance(raw_shape, dict) or not isinstance(raw_shape.get("center"), dict):
                    raise RoiTemplateValidationError(f"ROI {index + 1} 缺少{label}椭圆。")
                center = _point_list(
                    [raw_shape["center"]],
                    minimum=1,
                    maximum=1,
                    source=source,
                    already_normalized=already_normalized,
                )[0]
                try:
                    rx = float(raw_shape["radius_x_px"])
                    ry = float(raw_shape["radius_y_px"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RoiTemplateValidationError(f"ROI {index + 1} 的{label}半径无效。") from exc
                if not already_normalized:
                    rx /= float(source["width"])
                    ry /= float(source["height"])
                if not (0 < rx <= 1.01 and 0 < ry <= 1.01):
                    raise RoiTemplateValidationError(f"ROI {index + 1} 的{label}半径超出画面。")
                return {
                    "center": center,
                    "radius_x_px": min(1.0, rx),
                    "radius_y_px": min(1.0, ry),
                    "angle_deg": float(raw_shape.get("angle_deg") or 0),
                }

            roi["pool"] = clean_ellipse(raw_roi.get("pool"), "泳池")
            roi["platform"] = clean_ellipse(raw_roi.get("platform"), "平台")
        elif schema.get("special") == "barnes_maze":
            holes = []
            ids = set()
            for raw_hole in raw_roi.get("holes") or []:
                hole_id = str(raw_hole.get("id") or "").strip()
                if not hole_id or hole_id in ids:
                    raise RoiTemplateValidationError("洞位编号不能为空或重复。")
                ids.add(hole_id)
                point = _point_list([raw_hole.get("point")], minimum=1, maximum=1,
                                    source=source, already_normalized=already_normalized)[0]
                hole = {"id": hole_id, "point": point}
                if raw_hole.get("radius_cm") is not None:
                    radius = float(raw_hole['radius_cm'])
                    if not 0 < radius <= 1000:
                        raise RoiTemplateValidationError("洞 ROI 半径无效。")
                    hole['radius_cm'] = radius
                holes.append(hole)
            if len(holes) < 2 or len(holes) != int(raw_roi.get("hole_count") or len(holes)):
                raise RoiTemplateValidationError("请完成全部洞位后保存模板。")
            roi["holes"] = holes
        elif schema.get("special") in {"novel_object", "three_chamber"}:
            objects = []
            for raw_object in raw_roi.get("objects") or []:
                if not isinstance(raw_object, dict) or not isinstance(raw_object.get("point"), dict):
                    continue
                point = _point_list(
                    [raw_object["point"]],
                    minimum=1,
                    maximum=1,
                    source=source,
                    already_normalized=already_normalized,
                )[0]
                if schema.get("special") == "three_chamber":
                    side = str(raw_object.get("side") or "")
                    if side not in {"left", "right"}:
                        continue
                    objects.append({"side": side, "name": f"{side.title()} cup", "point": point})
                else:
                    objects.append({"name": f"Object {len(objects) + 1}", "point": point})
            if len(objects) != 2:
                raise RoiTemplateValidationError(f"ROI {index + 1} 缺少两个物体位置。")
            roi["objects"] = objects
        if zone_schema or zones:
            roi["zones"] = zones
        for field, (minimum, maximum, required) in (schema.get("point_fields") or {}).items():
            raw_points = raw_roi.get(field)
            if not raw_points and not required:
                continue
            points = _point_list(
                raw_points,
                minimum=minimum,
                maximum=maximum,
                source=source,
                already_normalized=already_normalized,
            )
            if points is None:
                raise RoiTemplateValidationError(f"ROI {index + 1} 的 {field} 尚未完成。")
            roi[field] = points
        for key in schema.get("roi_scalars") or ():
            raw_value = raw_roi.get(key)
            if raw_value is None:
                continue
            try:
                number = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise RoiTemplateValidationError("设备物理尺寸必须是数字。") from exc
            if not 0 < number <= 10_000:
                raise RoiTemplateValidationError("设备物理尺寸超出允许范围。")
            roi[key] = number
        rois.append(roi)

    geometry: dict[str, float] = {}
    raw_geometry = payload.get("geometry")
    if isinstance(raw_geometry, dict):
        for key in schema["geometry"]:
            if raw_geometry.get(key) is None:
                continue
            try:
                number = float(raw_geometry[key])
            except (TypeError, ValueError) as exc:
                raise RoiTemplateValidationError("设备物理尺寸必须是数字。") from exc
            if not 0 < number <= 10_000:
                raise RoiTemplateValidationError("设备物理尺寸超出允许范围。")
            geometry[key] = number
    return {
        "assay": assay_key,
        "coordinate_space": "normalized",
        "source_frame": source,
        "rois": rois,
        "geometry": geometry,
    }


def compatibility_warnings(
    source_frame: dict[str, Any], target_frame: dict[str, Any], *, aspect_tolerance: float = 0.02
) -> list[str]:
    target = _source_frame(target_frame)
    warnings: list[str] = []
    if not isinstance(source_frame, dict) or source_frame.get("unknown"):
        return ["legacy_frame_unknown"]
    source = _source_frame(source_frame)
    if source["orientation_degrees"] != target["orientation_degrees"]:
        warnings.append("orientation_mismatch")
    source_ratio = float(source["aspect_ratio"])
    target_ratio = float(target["aspect_ratio"])
    if abs(source_ratio - target_ratio) / max(source_ratio, 1e-9) > float(aspect_tolerance):
        warnings.append("aspect_ratio_mismatch")
    return warnings


def materialize_template(template: dict[str, Any], target_frame: dict[str, Any]) -> dict[str, Any]:
    target = _source_frame(target_frame)
    width, height = float(target["width"]), float(target["height"])
    result = copy.deepcopy(template)
    legacy_pixels = str(result.get("coordinate_space")) == "legacy_pixels"
    source_frame = result.get("source_frame") or {}
    unknown_legacy_frame = legacy_pixels and bool(source_frame.get("unknown"))
    legacy_source = None if unknown_legacy_frame else _source_frame(source_frame)
    source_width = float(legacy_source["width"]) if legacy_source else 1.0
    source_height = float(legacy_source["height"]) if legacy_source else 1.0
    for roi in result.get("rois") or []:
        for points in (roi.get("zones") or {}).values():
            for point in points or []:
                if unknown_legacy_frame:
                    point["x"], point["y"] = float(point["x"]), float(point["y"])
                else:
                    nx = float(point["x"]) / source_width if legacy_pixels else float(point["x"])
                    ny = float(point["y"]) / source_height if legacy_pixels else float(point["y"])
                    point["x"], point["y"] = nx * width, ny * height
        for point in roi.get("detection_points") or []:
            if unknown_legacy_frame:
                point["x"], point["y"] = float(point["x"]), float(point["y"])
            else:
                nx = float(point["x"]) / source_width if legacy_pixels else float(point["x"])
                ny = float(point["y"]) / source_height if legacy_pixels else float(point["y"])
                point["x"], point["y"] = nx * width, ny * height
        for field in ("points", "diameter_points", "waterline_points", "arena_points"):
            for point in roi.get(field) or []:
                if unknown_legacy_frame:
                    point["x"], point["y"] = float(point["x"]), float(point["y"])
                else:
                    nx = float(point["x"]) / source_width if legacy_pixels else float(point["x"])
                    ny = float(point["y"]) / source_height if legacy_pixels else float(point["y"])
                    point["x"], point["y"] = nx * width, ny * height
        if str(result.get("assay")) == "barnes_maze":
            for hole in roi.get("holes") or []:
                point = hole["point"]
                point["x"], point["y"] = float(point["x"]) * width, float(point["y"]) * height
        if str(result.get("assay")) == "water_maze":
            for field in ("pool", "platform"):
                shape = roi.get(field)
                if not isinstance(shape, dict):
                    continue
                center = shape.get("center") or {}
                center["x"] = float(center.get("x") or 0) * width
                center["y"] = float(center.get("y") or 0) * height
                shape["radius_x_px"] = float(shape.get("radius_x_px") or 0) * width
                shape["radius_y_px"] = float(shape.get("radius_y_px") or 0) * height
        if str(result.get("assay")) in {"novel_object", "three_chamber"}:
            for item in roi.get("objects") or []:
                point = item.get("point") or {}
                point["x"] = float(point.get("x") or 0) * width
                point["y"] = float(point.get("y") or 0) * height
    result["coordinate_space"] = "pixels"
    result["target_frame"] = target
    result["warnings"] = compatibility_warnings(result.get("source_frame") or {}, target)
    return result


class RoiTemplateStore:
    """Disk-backed reusable device geometry, isolated by experiment type."""

    def __init__(self, path: str | Path, legacy_y_maze_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.legacy_y_maze_path = Path(legacy_y_maze_path) if legacy_y_maze_path else None
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy()

    def _read(self) -> list[dict[str, Any]]:
        data = self._read_document()
        templates = data.get("templates", [])
        return templates if isinstance(templates, list) else []

    def _read_document(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = read_json(self.path)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write(
        self,
        templates: list[dict[str, Any]],
        *,
        migrated_legacy_ids: list[str] | None = None,
    ) -> None:
        document = self._read_document()
        migrations = document.get("migrations") if isinstance(document.get("migrations"), dict) else {}
        if migrated_legacy_ids is not None:
            migrations["y_maze_template_ids"] = sorted(set(migrated_legacy_ids))
        atomic_write_json(
            self.path,
            {"version": 2, "templates": templates, "migrations": migrations},
        )

    def _migrate_legacy(self) -> None:
        legacy = self.legacy_y_maze_path
        if legacy is None or not legacy.exists():
            return
        with self._lock:
            document = self._read_document()
            templates = self._read()
            known_ids = {str(item.get("id")) for item in templates}
            migrations = document.get("migrations") if isinstance(document.get("migrations"), dict) else {}
            migrated_ids = {
                str(item) for item in migrations.get("y_maze_template_ids", []) if item
            }
            try:
                data = read_json(legacy)
                old_templates = data.get("templates", []) if isinstance(data, dict) else []
            except Exception:
                return
            changed = False
            for raw in old_templates if isinstance(old_templates, list) else []:
                if not isinstance(raw, dict):
                    continue
                legacy_id = str(raw.get("id") or "")
                if legacy_id in known_ids or legacy_id in migrated_ids:
                    continue
                # Legacy coordinates were pixels but did not record the video frame.
                # ROI bounds are not frame bounds, so never infer dimensions from
                # max coordinates. Preserve exact pixels and require visual review.
                migrated = copy.deepcopy(raw)
                migrated.update(
                    {
                        "assay": "y_maze",
                        "coordinate_space": "legacy_pixels",
                        "source_frame": {"unknown": True},
                    }
                )
                templates.append(migrated)
                known_ids.add(str(migrated.get("id")))
                migrated_ids.add(str(migrated.get("id")))
                changed = True
            if changed or (old_templates and not migrations.get("y_maze_template_ids")):
                self._write(templates, migrated_legacy_ids=sorted(migrated_ids))

    def list_templates(self, assay: Any) -> list[dict[str, Any]]:
        assay_key = canonical_assay(assay)
        with self._lock:
            templates = [
                copy.deepcopy(item)
                for item in self._read()
                if str(item.get("assay")) == assay_key
            ]
        return sorted(templates, key=lambda item: float(item.get("updated_at") or 0), reverse=True)

    def save_template(self, assay: Any, name: Any, payload: Any) -> dict[str, Any]:
        assay_key = canonical_assay(assay)
        clean_name = _clean_name(name)
        normalized = normalize_template_payload(assay_key, payload)
        now = time.time()
        with self._lock:
            templates = self._read()
            existing = next(
                (
                    item
                    for item in templates
                    if str(item.get("assay")) == assay_key
                    and str(item.get("name", "")).casefold() == clean_name.casefold()
                ),
                None,
            )
            if existing is None:
                existing = {"id": uuid.uuid4().hex, "created_at": now}
                templates.append(existing)
            existing.update({"name": clean_name, "updated_at": now, **normalized})
            self._write(templates)
            return copy.deepcopy(existing)

    def delete_template(self, assay: Any, template_id: str) -> bool:
        assay_key = canonical_assay(assay)
        template_id = str(template_id or "").strip()
        if not template_id:
            return False
        with self._lock:
            templates = self._read()
            kept = [
                item
                for item in templates
                if not (
                    str(item.get("assay")) == assay_key
                    and str(item.get("id")) == template_id
                )
            ]
            if len(kept) == len(templates):
                return False
            self._write(kept)
        return True
