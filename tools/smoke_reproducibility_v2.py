#!/usr/bin/env python3
"""Same-machine, same-version reproducibility smoke test.

Runs one single-video and one two-video OFT pair against the bundled sample,
then requires every comparable CSV/TXT output in each replay to match its
reference SHA-256 digest.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile
import time


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import app as mousescope_app  # noqa: E402
from services.analysis_builders import build_analysis_config  # noqa: E402
from services.open_field_session import OpenFieldMultiArenaAnalysisSession  # noqa: E402
from services.reproducibility import (  # noqa: E402
    REPRODUCIBILITY_REPORT_FILENAME,
    apply_deterministic_seed,
    build_run_snapshot,
    finalize_reproducibility_artifacts,
    write_run_snapshot,
)
from services.version import APP_VERSION  # noqa: E402


SAMPLE_VIDEO = REPO_ROOT / "test.mp4"


def _payload(video_count: int) -> dict:
    roi = {
        "name": "Arena 1",
        "points": [[0, 0], [543, 0], [543, 959], [0, 959]],
        "time_config": {"mode": "global", "start": None, "end": None},
    }
    config = {
        "gray_threshold": 127,
        "rois": [roi],
        "generate_video": False,
    }
    config["per_video_configs"] = [
        {"gray_threshold": 127, "rois": [copy.deepcopy(roi)]}
        for _ in range(video_count)
    ]
    return config


def _run_once(
    *,
    run_id: str,
    output_dir: pathlib.Path,
    video_count: int,
    reference_snapshot: str | None = None,
) -> tuple[str, dict | None]:
    video_path = str(SAMPLE_VIDEO.resolve())
    info = mousescope_app._probe_video_info(video_path)
    info.update({"path": video_path, "original_name": SAMPLE_VIDEO.name})
    video_paths = [video_path for _ in range(video_count)]
    video_infos = [copy.deepcopy(info) for _ in range(video_count)]
    payload = _payload(video_count)
    sess = {
        "experiment_type": "open_field",
        "video_paths": video_paths,
        "video_infos": video_infos,
        "output_dir": str(output_dir),
        "output_root_dir": str(output_dir.parent),
        "start_seconds": 0.0,
        "end_seconds": 1.0,
        "fps_override": None,
        "is_batch": video_count > 1,
        "draft_payload": payload,
    }
    if reference_snapshot:
        sess["replay_of"] = {"session_id": run_id.replace("replay", "reference"), "snapshot_path": reference_snapshot}

    config = build_analysis_config("open_field", sess, payload)
    seed = apply_deterministic_seed()
    sess["reproducibility_seed"] = seed
    config["reproducibility_seed"] = seed
    snapshot = build_run_snapshot(
        session_id=run_id,
        app_version=APP_VERSION,
        sess=sess,
        resolved_config=config,
    )
    snapshot_path = write_run_snapshot(output_dir, snapshot)

    analysis = OpenFieldMultiArenaAnalysisSession(run_id, video_paths, video_infos, config)
    analysis.is_running = True
    analysis.status = "analyzing"
    analysis.start_time = time.time()
    analysis._run()
    if analysis.status != "completed":
        raise RuntimeError(f"{run_id} failed: {analysis.message}")
    finalize_reproducibility_artifacts(analysis, sess)

    report_path = output_dir / REPRODUCIBILITY_REPORT_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    return snapshot_path, report


def _verify_pair(root: pathlib.Path, label: str, video_count: int) -> int:
    reference_snapshot, _ = _run_once(
        run_id=f"{label}-reference",
        output_dir=root / f"{label}-reference",
        video_count=video_count,
    )
    _, report = _run_once(
        run_id=f"{label}-replay",
        output_dir=root / f"{label}-replay",
        video_count=video_count,
        reference_snapshot=reference_snapshot,
    )
    if not report or report.get("status") != "consistent" or not report.get("consistent"):
        raise RuntimeError(f"{label} reproducibility mismatch: {report}")
    compared = int(report.get("compared_file_count") or 0)
    if compared <= 0:
        raise RuntimeError(f"{label} produced no comparable scientific result files")
    return compared


def main() -> int:
    if not SAMPLE_VIDEO.is_file():
        raise FileNotFoundError(f"Missing sample video: {SAMPLE_VIDEO}")
    with tempfile.TemporaryDirectory(prefix="mousescope_repro_") as temp_dir:
        root = pathlib.Path(temp_dir)
        single_count = _verify_pair(root, "single", 1)
        batch_count = _verify_pair(root, "batch", 2)
    print(
        f"v{APP_VERSION} reproducibility smoke passed: "
        f"single={single_count} files, batch={batch_count} files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
