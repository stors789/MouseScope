from __future__ import annotations

import copy
import hashlib
import math
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from services.atomic_json import atomic_write_json, read_json


SUPPORTED_ASSAYS = {"open_field", "novel_object", "barnes_maze"}
DEFAULT_KEYPOINT_SCHEMA = [
    {"name": "nose", "label_zh": "鼻尖", "required": True},
    {"name": "left_ear", "label_zh": "左耳", "required": False},
    {"name": "right_ear", "label_zh": "右耳", "required": False},
    {"name": "body_center", "label_zh": "身体中心", "required": True},
    {"name": "tail_base", "label_zh": "尾根", "required": False},
]
SUPERANIMAL_MODEL_REF = "deeplabcut://superanimal_topviewmouse/hrnet_w32"


class PoseModelValidationError(ValueError):
    pass


_NAME_WHITESPACE = re.compile(r"\s+")
_KEYPOINT_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _clean_name(value: Any, label: str = "模型档案名称") -> str:
    name = _NAME_WHITESPACE.sub(" ", str(value or "").strip())
    if not name:
        raise PoseModelValidationError(f"{label}不能为空。")
    if len(name) > 100:
        raise PoseModelValidationError(f"{label}不能超过 100 个字符。")
    return name


def _clean_assays(value: Any) -> list[str]:
    raw = value if isinstance(value, list) else [value]
    aliases = {"oft": "open_field", "nor": "novel_object", "barnes": "barnes_maze"}
    assays: list[str] = []
    for item in raw:
        assay = aliases.get(str(item or "").strip().lower(), str(item or "").strip().lower())
        if assay not in SUPPORTED_ASSAYS:
            raise PoseModelValidationError("姿态模型仅支持旷场、新奇物体识别和巴恩斯迷宫实验。")
        if assay not in assays:
            assays.append(assay)
    if not assays:
        raise PoseModelValidationError("请至少选择一个适用实验。")
    return assays


def _clean_schema(value: Any) -> list[dict[str, Any]]:
    raw_schema = value if isinstance(value, list) and value else DEFAULT_KEYPOINT_SCHEMA
    schema: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in raw_schema:
        if isinstance(raw, str):
            raw = {"name": raw}
        if not isinstance(raw, dict):
            raise PoseModelValidationError("关键点 schema 格式无效。")
        name = str(raw.get("name") or "").strip().lower()
        if not _KEYPOINT_NAME.fullmatch(name):
            raise PoseModelValidationError(f"关键点名称无效：{name or '-'}")
        if name in names:
            raise PoseModelValidationError(f"关键点名称重复：{name}")
        names.add(name)
        schema.append(
            {
                "name": name,
                "label_zh": str(raw.get("label_zh") or name)[:80],
                "required": bool(raw.get("required", False)),
            }
        )
    if "nose" not in names or "body_center" not in names:
        raise PoseModelValidationError("关键点 schema 必须包含 nose 和 body_center。")
    return schema


def _clean_metadata(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    allowed = (
        "description",
        "camera_view",
        "camera_name",
        "resolution",
        "mouse_color",
        "arena_color",
        "lighting",
        "laboratory",
    )
    return {
        key: copy.deepcopy(raw[key])
        for key in allowed
        if raw.get(key) not in (None, "")
    }


def _clean_weight_reference(value: Any, *, training_status: str) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    kind = str(raw.get("kind") or "remote_model").strip().lower()
    if kind not in {"remote_model", "local_file", "external_artifact", "pending"}:
        raise PoseModelValidationError("权重引用类型无效。")
    reference = str(raw.get("reference") or SUPERANIMAL_MODEL_REF).strip()
    if kind != "pending" and not reference:
        raise PoseModelValidationError("权重引用不能为空。")
    checksum = str(raw.get("sha256") or "").strip().lower() or None
    if checksum is not None and not _HEX_SHA256.fullmatch(checksum):
        raise PoseModelValidationError("权重 SHA-256 校验值必须是 64 位十六进制字符串。")
    if training_status == "ready" and kind in {"local_file", "external_artifact"} and not checksum:
        raise PoseModelValidationError("可用的自定义权重必须提供 SHA-256 校验值。")
    return {
        "kind": kind,
        "reference": reference,
        "checksum": {
            "algorithm": "sha256",
            "value": checksum,
            "status": "recorded" if checksum else "unverified_remote",
        },
    }


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _clean_annotation(raw: Any, schema_names: set[str], *, default_split: str = "train") -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PoseModelValidationError("人工标注记录格式无效。")
    split = str(raw.get("split") or default_split).strip().lower()
    if split not in {"train", "validation"}:
        raise PoseModelValidationError("标注 split 只能是 train 或 validation。")
    try:
        frame_index = int(raw["frame_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PoseModelValidationError("标注必须包含有效的 frame_index。") from exc
    if frame_index < 0:
        raise PoseModelValidationError("frame_index 不能小于 0。")
    video_ref = str(raw.get("video_ref") or "").strip()
    if not video_ref:
        raise PoseModelValidationError("标注必须包含 video_ref。")
    raw_points = raw.get("keypoints")
    if not isinstance(raw_points, dict) or not raw_points:
        raise PoseModelValidationError("标注必须包含关键点坐标。")
    keypoints: dict[str, dict[str, Any]] = {}
    for name, point in raw_points.items():
        key = str(name or "").strip().lower()
        if key not in schema_names:
            raise PoseModelValidationError(f"标注包含 schema 之外的关键点：{key}")
        if point is None:
            keypoints[key] = {"visible": False}
            continue
        if not isinstance(point, dict):
            raise PoseModelValidationError(f"关键点 {key} 的坐标格式无效。")
        visible = bool(point.get("visible", True))
        if not visible:
            keypoints[key] = {"visible": False}
            continue
        try:
            x, y = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PoseModelValidationError(f"关键点 {key} 缺少有效坐标。") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise PoseModelValidationError(f"关键点 {key} 坐标必须是有限数字。")
        keypoints[key] = {"x": x, "y": y, "visible": True}
    annotation_id = str(raw.get("annotation_id") or uuid.uuid4().hex).strip()
    if not annotation_id:
        annotation_id = uuid.uuid4().hex
    result = {
        "annotation_id": annotation_id,
        "video_ref": video_ref,
        "frame_index": frame_index,
        "roi_id": str(raw.get("roi_id") or "").strip() or None,
        "coordinate_space": str(raw.get("coordinate_space") or "pixels").strip().lower(),
        "keypoints": keypoints,
        "split": split,
        "created_at": float(raw.get("created_at") or time.time()),
    }
    if isinstance(raw.get("frame_size"), dict):
        try:
            width = int(raw["frame_size"]["width"])
            height = int(raw["frame_size"]["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PoseModelValidationError("标注 frame_size 无效。") from exc
        if width <= 0 or height <= 0:
            raise PoseModelValidationError("标注 frame_size 必须大于 0。")
        result["frame_size"] = {"width": width, "height": height}
    return result


class PoseModelProfileStore:
    """Versioned pose archives. Existing versions are never updated in-place."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read_profiles(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            document = read_json(self.path)
        except Exception:
            return []
        profiles = document.get("profiles", []) if isinstance(document, dict) else []
        return profiles if isinstance(profiles, list) else []

    def _write_profiles(self, profiles: list[dict[str, Any]]) -> None:
        atomic_write_json(self.path, {"schema_version": 1, "profiles": profiles})

    @staticmethod
    def _public_profile(profile: dict[str, Any], *, include_annotations: bool) -> dict[str, Any]:
        result = copy.deepcopy(profile)
        versions = result.get("versions") or []
        for version in versions:
            annotations = version.get("annotations") or []
            version["annotation_count"] = sum(1 for item in annotations if item.get("split") == "train")
            version["validation_count"] = sum(1 for item in annotations if item.get("split") == "validation")
            if not include_annotations:
                version.pop("annotations", None)
        result["latest_version_id"] = versions[-1]["version_id"] if versions else None
        return result

    def list_profiles(self, assay: Any | None = None, *, include_annotations: bool = False) -> list[dict[str, Any]]:
        assay_key = None
        if assay not in (None, ""):
            assay_key = _clean_assays([assay])[0]
        with self._lock:
            profiles = [
                self._public_profile(item, include_annotations=include_annotations)
                for item in self._read_profiles()
                if assay_key is None or assay_key in (item.get("assays") or [])
            ]
        return sorted(profiles, key=lambda item: float(item.get("updated_at") or 0), reverse=True)

    def get_profile(
        self,
        profile_id: str,
        *,
        version_id: str | None = None,
        include_annotations: bool = False,
    ) -> dict[str, Any]:
        with self._lock:
            profile = next(
                (item for item in self._read_profiles() if str(item.get("profile_id")) == str(profile_id)),
                None,
            )
            if profile is None:
                raise KeyError(profile_id)
            result = self._public_profile(profile, include_annotations=include_annotations)
        if version_id:
            versions = [item for item in result.get("versions") or [] if str(item.get("version_id")) == str(version_id)]
            if not versions:
                raise KeyError(version_id)
            result["versions"] = versions
            result["latest_version_id"] = versions[0]["version_id"]
        return result

    def create_profile(
        self,
        *,
        name: Any,
        assays: Any,
        metadata: Any = None,
        keypoint_schema: Any = None,
        base_model: str = "SuperAnimal-TopViewMouse",
    ) -> dict[str, Any]:
        clean_name = _clean_name(name)
        clean_assays = _clean_assays(assays)
        schema = _clean_schema(keypoint_schema)
        now = time.time()
        version_id = uuid.uuid4().hex
        profile = {
            "profile_id": uuid.uuid4().hex,
            "name": clean_name,
            "assays": clean_assays,
            "metadata": _clean_metadata(metadata),
            "created_at": now,
            "updated_at": now,
            "versions": [
                {
                    "version_id": version_id,
                    "version_number": 1,
                    "parent_version": None,
                    "base_model": str(base_model or "SuperAnimal-TopViewMouse"),
                    "keypoint_schema": schema,
                    "weights": _clean_weight_reference(
                        {
                            "kind": "remote_model",
                            "reference": SUPERANIMAL_MODEL_REF,
                        },
                        training_status="pretrained",
                    ),
                    "annotations": [],
                    "fixed_validation_annotation_ids": [],
                    "training": {
                        "status": "pretrained",
                        "adapter": "deeplabcut-superanimal",
                        "message": "零样本基础版本；远程权重校验值将在 DLC 运行时下载并登记。",
                    },
                    "created_at": now,
                }
            ],
        }
        with self._lock:
            profiles = self._read_profiles()
            if any(str(item.get("name", "")).casefold() == clean_name.casefold() for item in profiles):
                raise PoseModelValidationError("已存在同名姿态模型档案。")
            profiles.append(profile)
            self._write_profiles(profiles)
        return self._public_profile(profile, include_annotations=True)

    def create_version(
        self,
        profile_id: str,
        *,
        parent_version: str,
        annotations: Any,
        metadata: Any = None,
        weight_reference: Any = None,
        training_status: str = "awaiting_external_training",
    ) -> dict[str, Any]:
        status = str(training_status or "awaiting_external_training").strip().lower()
        if status not in {"awaiting_external_training", "ready"}:
            raise PoseModelValidationError("新版本状态只能是 awaiting_external_training 或 ready。")
        raw_annotations = annotations if isinstance(annotations, list) else []
        if not raw_annotations and status != "ready":
            raise PoseModelValidationError("生成新版本前请至少追加一帧人工纠错或固定验证标注。")
        with self._lock:
            profiles = self._read_profiles()
            profile = next((item for item in profiles if str(item.get("profile_id")) == str(profile_id)), None)
            if profile is None:
                raise KeyError(profile_id)
            parent = next(
                (item for item in profile.get("versions") or [] if str(item.get("version_id")) == str(parent_version)),
                None,
            )
            if parent is None:
                raise KeyError(parent_version)
            schema_names = {item["name"] for item in parent.get("keypoint_schema") or []}
            inherited = copy.deepcopy(parent.get("annotations") or [])
            inherited_ids = {str(item.get("annotation_id")) for item in inherited}
            additions = [_clean_annotation(item, schema_names) for item in raw_annotations]
            for item in additions:
                if item["annotation_id"] in inherited_ids:
                    raise PoseModelValidationError("新增标注的 annotation_id 与历史标注重复。")
                inherited_ids.add(item["annotation_id"])
            combined = inherited + additions
            fixed_validation_ids = list(parent.get("fixed_validation_annotation_ids") or [])
            fixed_validation_ids.extend(
                item["annotation_id"]
                for item in additions
                if item.get("split") == "validation" and item["annotation_id"] not in fixed_validation_ids
            )
            version_number = max(int(item.get("version_number") or 0) for item in profile["versions"]) + 1
            now = time.time()
            version = {
                "version_id": uuid.uuid4().hex,
                "version_number": version_number,
                "parent_version": parent["version_id"],
                "base_model": parent.get("base_model") or "SuperAnimal-TopViewMouse",
                "keypoint_schema": copy.deepcopy(parent.get("keypoint_schema") or DEFAULT_KEYPOINT_SCHEMA),
                "weights": _clean_weight_reference(
                    weight_reference or {"kind": "pending", "reference": f"pending://{profile_id}/v{version_number}"},
                    training_status=status,
                ),
                "annotations": combined,
                "fixed_validation_annotation_ids": fixed_validation_ids,
                "training": {
                    "status": status,
                    "adapter": "external",
                    "message": (
                        "已登记权重与校验值。"
                        if status == "ready"
                        else "版本已保存，等待外部 DeepLabCut 训练适配器生成并登记权重。"
                    ),
                },
                "version_metadata": _clean_metadata(metadata),
                "created_at": now,
            }
            profile["versions"].append(version)
            profile["updated_at"] = now
            self._write_profiles(profiles)
            return copy.deepcopy(version)
