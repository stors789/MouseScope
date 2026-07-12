from __future__ import annotations

import copy
import hashlib
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

from services.atomic_json import atomic_write_json, read_json


RUN_SNAPSHOT_FILENAME = "run_snapshot.json"
RESULT_FINGERPRINT_FILENAME = "result_fingerprint.json"
REPRODUCIBILITY_REPORT_FILENAME = "reproducibility_report.json"
RUN_SNAPSHOT_VERSION = 1
DEFAULT_RANDOM_SEED = 20260710
_SCIENTIFIC_RESULT_SUFFIXES = {".csv", ".tsv", ".txt"}
_METADATA_FILENAMES = {
    RUN_SNAPSHOT_FILENAME,
    RESULT_FINGERPRINT_FILENAME,
    REPRODUCIBILITY_REPORT_FILENAME,
}


class ReproducibilityError(ValueError):
    """Raised when an earlier run cannot be reconstructed safely."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except Exception:
            pass
    return str(value)


def capture_input_file_states(video_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for raw_path in video_paths or []:
        path = Path(str(raw_path)).expanduser()
        absolute = path.resolve(strict=False)
        state: dict[str, Any] = {
            "path": str(absolute),
            "exists": absolute.is_file(),
        }
        if state["exists"]:
            stat = absolute.stat()
            state.update({
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
        states.append(state)
    return states


def build_run_snapshot(
    *,
    session_id: str,
    app_version: str,
    sess: dict[str, Any],
    resolved_config: dict[str, Any],
) -> dict[str, Any]:
    output_dir = str(resolved_config.get("output_dir") or sess.get("output_dir") or "")
    payload = {
        "snapshot_version": RUN_SNAPSHOT_VERSION,
        "app_version": str(app_version),
        "created_at": time.time(),
        "session_id": str(session_id),
        "experiment_type": str(sess.get("experiment_type") or "tst"),
        "random_seed": int(sess.get("reproducibility_seed") or DEFAULT_RANDOM_SEED),
        "video_paths": [str(Path(str(path)).resolve(strict=False)) for path in (sess.get("video_paths") or [])],
        "video_infos": _json_safe(sess.get("video_infos") or []),
        "input_files": capture_input_file_states(sess.get("video_paths") or []),
        "draft_payload": _json_safe(sess.get("draft_payload") or {}),
        "resolved_config": _json_safe(resolved_config),
        "output_root_dir": str(sess.get("output_root_dir") or ""),
        "output_dir": output_dir,
        "is_batch": bool(sess.get("is_batch")),
        "selected_video_idx": int(sess.get("selected_video_idx") or 0),
        "start_seconds": sess.get("start_seconds"),
        "end_seconds": sess.get("end_seconds"),
        "fps_override": sess.get("fps_override"),
        "project_context": _json_safe(sess.get("project_context")) if isinstance(sess.get("project_context"), dict) else None,
        "replay_of": _json_safe(sess.get("replay_of")) if isinstance(sess.get("replay_of"), dict) else None,
    }
    return _json_safe(payload)


def write_run_snapshot(output_dir: str | Path, snapshot: dict[str, Any]) -> str:
    path = Path(output_dir) / RUN_SNAPSHOT_FILENAME
    atomic_write_json(path, snapshot)
    return str(path)


def load_run_snapshot(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ReproducibilityError("运行快照格式无效。")
    if int(payload.get("snapshot_version") or 0) != RUN_SNAPSHOT_VERSION:
        raise ReproducibilityError("运行快照版本不受支持。")
    return payload


def find_run_snapshot(record: dict[str, Any] | None) -> str | None:
    if not isinstance(record, dict):
        return None
    for raw_path in record.get("result_files") or []:
        path = Path(str(raw_path))
        if path.name == RUN_SNAPSHOT_FILENAME and path.is_file():
            return str(path)
    output_dir = str(record.get("output_dir") or "").strip()
    if output_dir:
        candidate = Path(output_dir) / RUN_SNAPSHOT_FILENAME
        if candidate.is_file():
            return str(candidate)
    return None


def find_reproducibility_report(record: dict[str, Any] | None) -> str | None:
    if not isinstance(record, dict):
        return None
    for raw_path in record.get("result_files") or []:
        path = Path(str(raw_path))
        if path.name == REPRODUCIBILITY_REPORT_FILENAME and path.is_file():
            return str(path)
    output_dir = str(record.get("output_dir") or "").strip()
    if output_dir:
        candidate = Path(output_dir) / REPRODUCIBILITY_REPORT_FILENAME
        if candidate.is_file():
            return str(candidate)
    return None


def reproducibility_status(record: dict[str, Any] | None) -> str | None:
    report_path = find_reproducibility_report(record)
    if not report_path:
        return None
    try:
        payload = read_json(report_path)
    except (OSError, ValueError):
        return "report_unreadable"
    status = str(payload.get("status") or "").strip().lower() if isinstance(payload, dict) else ""
    return status or "report_unreadable"


def validate_replay_snapshot(snapshot: dict[str, Any], *, app_version: str) -> list[str]:
    errors: list[str] = []
    snapshot_version = str(snapshot.get("app_version") or "")
    if snapshot_version != str(app_version):
        errors.append(
            f"运行快照来自 MouseScope v{snapshot_version or '未知'}，只能在生成它的同一软件版本中重跑。"
        )

    expected_states = snapshot.get("input_files") or []
    if not expected_states:
        errors.append("运行快照没有输入视频状态记录。")
        return errors

    for index, expected in enumerate(expected_states):
        if not isinstance(expected, dict):
            errors.append(f"第 {index + 1} 个输入视频状态记录无效。")
            continue
        path = Path(str(expected.get("path") or ""))
        if not path.is_file():
            errors.append(f"原视频不存在或已被项目缓存清理：{path}")
            continue
        stat = path.stat()
        if int(expected.get("size") or -1) != int(stat.st_size):
            errors.append(f"原视频大小已变化，不能按原配置重跑：{path.name}")
            continue
        if int(expected.get("mtime_ns") or -1) != int(stat.st_mtime_ns):
            errors.append(f"原视频修改时间已变化，不能确认它仍是同一文件：{path.name}")
    return errors


def resolved_config_for_replay(snapshot: dict[str, Any], *, output_dir: str | Path) -> dict[str, Any]:
    config = copy.deepcopy(snapshot.get("resolved_config") or {})
    if not isinstance(config, dict) or not config:
        raise ReproducibilityError("运行快照缺少最终生效配置。")
    config["output_dir"] = str(output_dir)
    config["reproducibility_seed"] = int(snapshot.get("random_seed") or DEFAULT_RANDOM_SEED)
    return config


def apply_deterministic_seed(seed: int | None = None) -> int:
    resolved_seed = int(seed if seed is not None else DEFAULT_RANDOM_SEED)
    random.seed(resolved_seed)
    try:
        import numpy as np

        np.random.seed(resolved_seed)
    except Exception:
        pass
    try:
        import cv2

        cv2.setRNGSeed(resolved_seed)
    except Exception:
        pass
    return resolved_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


_VOLATILE_REPORT_LINE = re.compile(
    r"^\s*(?:"
    r"分析时间(?:\s*(?:/\s*Analysis Time|\(\s*Analysis Time\s*\)))?"
    r"|Analysis Time(?:\s*(?:/\s*分析时间|\(\s*分析时间\s*\)))?"
    r")\s*:",
    re.IGNORECASE,
)


def _scientific_result_sha256(path: Path) -> tuple[str, str]:
    if path.suffix.lower() != ".txt":
        return _sha256(path), "raw-bytes"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return _sha256(path), "raw-bytes"
    canonical_lines = [line for line in lines if not _VOLATILE_REPORT_LINE.match(line)]
    digest = hashlib.sha256(("\n".join(canonical_lines) + "\n").encode("utf-8")).hexdigest()
    return digest, "utf8-lines-without-analysis-time"


def _relative_result_name(path: Path, output_dir: Path) -> str:
    try:
        return path.resolve().relative_to(output_dir.resolve()).as_posix()
    except (OSError, ValueError):
        return path.name


def build_result_fingerprint(
    result_files: Iterable[str | Path],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    root = Path(output_dir)
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in result_files or []:
        path = Path(str(raw_path))
        if not path.is_file() or path.suffix.lower() not in _SCIENTIFIC_RESULT_SUFFIXES:
            continue
        if path.name in _METADATA_FILENAMES:
            continue
        relative_name = _relative_result_name(path, root)
        if relative_name in seen:
            continue
        seen.add(relative_name)
        digest, normalization = _scientific_result_sha256(path)
        entries.append({
            "path": relative_name,
            "size": int(path.stat().st_size),
            "sha256": digest,
            "normalization": normalization,
        })
    entries.sort(key=lambda item: item["path"])
    return {
        "algorithm": "sha256",
        "scope": "scientific-text-results",
        "files": entries,
    }


def _fingerprint_map(payload: dict[str, Any]) -> dict[str, str]:
    return {
        str(item.get("path")): str(item.get("sha256"))
        for item in (payload.get("files") or [])
        if isinstance(item, dict) and item.get("path") and item.get("sha256")
    }


def compare_result_fingerprints(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_map = _fingerprint_map(reference)
    candidate_map = _fingerprint_map(candidate)
    missing = sorted(set(reference_map) - set(candidate_map))
    extra = sorted(set(candidate_map) - set(reference_map))
    changed = sorted(
        name for name in set(reference_map) & set(candidate_map)
        if reference_map[name] != candidate_map[name]
    )
    return {
        "consistent": not missing and not extra and not changed,
        "compared_file_count": len(set(reference_map) & set(candidate_map)),
        "missing_files": missing,
        "extra_files": extra,
        "changed_files": changed,
    }


def finalize_reproducibility_artifacts(analysis: Any, sess: dict[str, Any] | None) -> list[str]:
    """Write deterministic result evidence once a run has completed.

    Only scientific CSV/TSV/TXT outputs are hashed. Generated MP4 files are
    deliberately excluded because encoder metadata can change without changing
    the underlying behavioral measurements.
    """

    if str(getattr(analysis, "status", "") or "").lower() != "completed":
        return []
    sess = sess or {}
    output_value = str((getattr(analysis, "config", {}) or {}).get("output_dir") or sess.get("output_dir") or "").strip()
    if not output_value:
        return []
    output_dir = Path(output_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = build_result_fingerprint(getattr(analysis, "result_files", []) or [], output_dir=output_dir)
    fingerprint.update({
        "session_id": str(getattr(analysis, "session_id", "") or ""),
        "created_at": time.time(),
    })
    fingerprint_path = output_dir / RESULT_FINGERPRINT_FILENAME
    atomic_write_json(fingerprint_path, fingerprint)
    created = [str(fingerprint_path)]

    replay_of = sess.get("replay_of") if isinstance(sess.get("replay_of"), dict) else None
    if replay_of:
        reference_snapshot_path = Path(str(replay_of.get("snapshot_path") or ""))
        reference_path = reference_snapshot_path.parent / RESULT_FINGERPRINT_FILENAME
        report: dict[str, Any] = {
            "session_id": str(getattr(analysis, "session_id", "") or ""),
            "replay_of_session_id": str(replay_of.get("session_id") or ""),
            "reference_fingerprint": str(reference_path),
            "candidate_fingerprint": str(fingerprint_path),
            "created_at": time.time(),
        }
        if reference_path.is_file():
            reference = read_json(reference_path)
            if not (reference.get("files") or []) or not (fingerprint.get("files") or []):
                report.update({
                    "status": "no_comparable_results",
                    "consistent": None,
                    "compared_file_count": 0,
                    "missing_files": [],
                    "extra_files": [],
                    "changed_files": [],
                })
            else:
                report.update(compare_result_fingerprints(reference, fingerprint))
                report["status"] = "consistent" if report["consistent"] else "different"
        else:
            report.update({
                "status": "reference_unavailable",
                "consistent": None,
                "compared_file_count": 0,
                "missing_files": [],
                "extra_files": [],
                "changed_files": [],
            })
        report_path = output_dir / REPRODUCIBILITY_REPORT_FILENAME
        atomic_write_json(report_path, report)
        created.append(str(report_path))

    append_result = getattr(analysis, "_append_result_file", None)
    for path in created:
        if callable(append_result):
            append_result(path, group="metadata")
        elif path not in analysis.result_files:
            analysis.result_files.append(path)
    return created
