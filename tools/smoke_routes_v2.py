#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Route and inline-script smoke test for the v2.1.0 workbench flow."""

from __future__ import annotations

import copy
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import app, analysis_sessions, _now_ts  # noqa: E402


SAMPLE_VIDEO = REPO_ROOT / "test.mp4"
BASE_VIDEO_INFO = {
    "path": str(SAMPLE_VIDEO),
    "original_name": SAMPLE_VIDEO.name,
    "fps": 30.0,
    "detected_fps": 30.0,
    "width": 640,
    "height": 480,
    "total_frames": 900,
}
EXPERIMENTS = ("tst", "open_field", "novel_object", "elevated_plus", "y_maze")


def _rendered_script_blocks(html: str) -> list[str]:
    return re.findall(r"<script>(.*?)</script>", html, flags=re.S)


def _check_js_syntax(name: str, blocks: Iterable[str]) -> None:
    node_path = shutil.which("node")
    if not node_path:
        print(f"[skip] node not found, skip inline JS syntax check for {name}")
        return

    for index, block in enumerate(blocks, start=1):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as handle:
            handle.write(block)
            temp_path = handle.name
        proc = subprocess.run([node_path, "--check", temp_path], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{name} script #{index} failed JS syntax check:\n{proc.stderr}")


def _seed_session(session_id: str, experiment_type: str, is_batch: bool) -> None:
    video_infos = [copy.deepcopy(BASE_VIDEO_INFO)]
    video_paths = [str(SAMPLE_VIDEO)]
    if is_batch:
        video_infos.append(copy.deepcopy(BASE_VIDEO_INFO))
        video_paths.append(str(SAMPLE_VIDEO))

    analysis_sessions[session_id] = {
        "experiment_type": experiment_type,
        "video_paths": video_paths,
        "video_infos": video_infos,
        "video_info": copy.deepcopy(BASE_VIDEO_INFO),
        "start_seconds": 0 if is_batch else None,
        "end_seconds": 30 if is_batch else None,
        "output_dir": None,
        "fps_override": None,
        "is_batch": is_batch,
        "draft_payload": {"per_video_configs": [{} for _ in video_infos]} if is_batch else None,
        "draft_updated_at": _now_ts() if is_batch else None,
        "selected_video_idx": 1 if is_batch and len(video_infos) > 1 else 0,
        "created_at": _now_ts(),
        "last_accessed_at": _now_ts(),
    }


def main() -> int:
    if not SAMPLE_VIDEO.exists():
        raise FileNotFoundError(f"Missing sample video for smoke test: {SAMPLE_VIDEO}")

    public_paths = [
        "/",
        "/upload",
        "/manual?module=tst",
        "/manual?module=oft",
        "/manual?module=nor",
        "/manual?module=epm",
        "/manual?module=ymaze",
    ]

    with app.test_client() as client:
        with client.session_transaction() as sess:
            sess["license_accepted"] = True

        for path in public_paths:
            response = client.get(path)
            if response.status_code != 200:
                raise RuntimeError(f"{path} returned {response.status_code}")
            _check_js_syntax(path, _rendered_script_blocks(response.get_data(as_text=True)))

        for experiment in EXPERIMENTS:
            single_id = f"smoke-single-{experiment}"
            batch_id = f"smoke-batch-{experiment}"
            _seed_session(single_id, experiment, is_batch=False)
            _seed_session(batch_id, experiment, is_batch=True)

            checks = [
                (f"/calibrate/{single_id}", 200),
                (f"/monitor/{single_id}", 200),
                (f"/batch/{batch_id}", 200),
                (f"/calibrate/{batch_id}", 302),
                (f"/calibrate/{batch_id}/video/1", 200),
                (f"/monitor/{batch_id}", 200),
                (f"/api/batch_state/{batch_id}", 200),
                (f"/api/draft/{batch_id}", 200),
            ]
            for path, expected_status in checks:
                response = client.get(path, follow_redirects=False)
                if response.status_code != expected_status:
                    raise RuntimeError(f"{experiment} {path} returned {response.status_code}, expected {expected_status}")
                if expected_status == 200 and response.content_type.startswith("text/html"):
                    _check_js_syntax(path, _rendered_script_blocks(response.get_data(as_text=True)))

    print("v2.1.0 route smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
