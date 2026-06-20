#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Short end-to-end analysis smoke test for MouseScope v3.4.0."""

from __future__ import annotations

import copy
import pathlib
import shutil
import sys
import tempfile
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app import app, analysis_sessions, _now_ts  # noqa: E402
from services.experiment_registry import list_experiments  # noqa: E402


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

RECT = [
    {"x": 100, "y": 80},
    {"x": 300, "y": 80},
    {"x": 300, "y": 420},
    {"x": 100, "y": 420},
]


def _seed_session(session_id: str, experiment_type: str, output_dir: str) -> None:
    analysis_sessions[session_id] = {
        "experiment_type": experiment_type,
        "video_paths": [str(SAMPLE_VIDEO)],
        "video_infos": [copy.deepcopy(BASE_VIDEO_INFO)],
        "video_info": copy.deepcopy(BASE_VIDEO_INFO),
        "start_seconds": 0,
        "end_seconds": 0.3,
        "output_dir": output_dir,
        "fps_override": None,
        "is_batch": False,
        "draft_payload": None,
        "draft_updated_at": None,
        "selected_video_idx": 0,
        "created_at": _now_ts(),
        "last_accessed_at": _now_ts(),
    }


def _epm_roi() -> dict:
    return {
        "name": "EPM 1",
        "zones": {
            "center": [
                {"x": 190, "y": 190},
                {"x": 230, "y": 190},
                {"x": 230, "y": 230},
                {"x": 190, "y": 230},
            ],
            "open_1": [
                {"x": 230, "y": 198},
                {"x": 330, "y": 198},
                {"x": 330, "y": 222},
                {"x": 230, "y": 222},
            ],
            "open_2": [
                {"x": 90, "y": 198},
                {"x": 190, "y": 198},
                {"x": 190, "y": 222},
                {"x": 90, "y": 222},
            ],
            "closed_1": [
                {"x": 198, "y": 90},
                {"x": 222, "y": 90},
                {"x": 222, "y": 190},
                {"x": 198, "y": 190},
            ],
            "closed_2": [
                {"x": 198, "y": 230},
                {"x": 222, "y": 230},
                {"x": 222, "y": 330},
                {"x": 198, "y": 330},
            ],
        },
        "time_config": {"mode": "global", "start": 0, "end": 0},
    }


def _ymaze_roi() -> dict:
    return {
        "name": "Y 1",
        "zones": {
            "center": [
                {"x": 198, "y": 190},
                {"x": 222, "y": 190},
                {"x": 210, "y": 228},
            ],
            "arm_a": [
                {"x": 198, "y": 90},
                {"x": 222, "y": 90},
                {"x": 222, "y": 190},
                {"x": 198, "y": 190},
            ],
            "arm_b": [
                {"x": 230, "y": 208},
                {"x": 320, "y": 168},
                {"x": 334, "y": 194},
                {"x": 242, "y": 234},
            ],
            "arm_c": [
                {"x": 168, "y": 194},
                {"x": 182, "y": 168},
                {"x": 270, "y": 208},
                {"x": 256, "y": 234},
            ],
        },
        "time_config": {"mode": "global", "start": 0, "end": 0},
    }


def _ldb_roi() -> dict:
    return {
        "name": "LDB 1",
        "zones": {
            "light": [{"x": 100, "y": 100}, {"x": 200, "y": 100}, {"x": 200, "y": 380}, {"x": 100, "y": 380}],
            "dark": [{"x": 210, "y": 100}, {"x": 310, "y": 100}, {"x": 310, "y": 380}, {"x": 210, "y": 380}],
            "doorway": [{"x": 195, "y": 210}, {"x": 215, "y": 210}, {"x": 215, "y": 270}, {"x": 195, "y": 270}],
        },
        "time_config": {"mode": "global", "start": 0, "end": 0},
    }


def _three_chamber_roi() -> dict:
    return {
        "name": "3CST 1",
        "zones": {
            "left": [{"x": 80, "y": 100}, {"x": 180, "y": 100}, {"x": 180, "y": 380}, {"x": 80, "y": 380}],
            "center": [{"x": 185, "y": 100}, {"x": 255, "y": 100}, {"x": 255, "y": 380}, {"x": 185, "y": 380}],
            "right": [{"x": 260, "y": 100}, {"x": 360, "y": 100}, {"x": 360, "y": 380}, {"x": 260, "y": 380}],
            "left_door": [{"x": 176, "y": 210}, {"x": 190, "y": 210}, {"x": 190, "y": 270}, {"x": 176, "y": 270}],
            "right_door": [{"x": 250, "y": 210}, {"x": 264, "y": 210}, {"x": 264, "y": 270}, {"x": 250, "y": 270}],
        },
        "objects": [
            {"side": "left", "name": "Left cup", "point": {"x": 130, "y": 240}},
            {"side": "right", "name": "Right cup", "point": {"x": 310, "y": 240}},
        ],
        "time_config": {"mode": "global", "start": 0, "end": 0},
    }


def _water_maze_roi() -> dict:
    return {
        "name": "MWM 1",
        "pool": {
            "center": {"x": 220, "y": 240},
            "radius_x_px": 140,
            "radius_y_px": 110,
            "angle_deg": 0,
            "diameter_cm": 120.0,
        },
        "platform": {
            "center": {"x": 260, "y": 205},
            "radius_x_px": 8,
            "radius_y_px": 7,
            "radius_cm": 5.0,
        },
        "pool_diameter_cm": 120.0,
        "time_config": {"mode": "global", "start": 0, "end": 0},
    }


def _payload_for(experiment_type: str) -> dict:
    if experiment_type == "tst":
        return {
            "generate_video": False,
            "per_video_configs": [{
                "pixel_to_mm": 1.0,
                "bg_method": "threshold",
                "bg_type": "light",
                "gray_threshold": 127,
                "erosion_kernel": 3,
                "erosion_iter": 2,
                "dilation_kernel": 5,
                "dilation_iter": 3,
                "min_area": 100,
                "rois": [{"points": RECT, "name": "ROI 1", "time_config": {"mode": "global", "start": 0, "end": 0}}],
            }],
        }
    if experiment_type == "open_field":
        return {
            "generate_video": False,
            "per_video_configs": [{
                "arena_side_cm": 50.0,
                "bg_method": "threshold",
                "bg_type": "light",
                "gray_threshold": 127,
                "erosion_kernel": 3,
                "erosion_iter": 1,
                "dilation_kernel": 5,
                "dilation_iter": 2,
                "min_area": 120,
                "movement_threshold_cm_s": 2.0,
                "bin_seconds": 30,
                "rois": [{"points": RECT, "name": "Arena 1", "time_config": {"mode": "global", "start": 0, "end": 0}}],
            }],
        }
    if experiment_type == "forced_swim":
        return {
            "generate_video": False,
            "per_video_configs": [{
                "cylinder_diameter_cm": 10.0,
                "bg_method": "threshold",
                "bg_type": "light",
                "gray_threshold": 127,
                "erosion_kernel": 3,
                "erosion_iter": 1,
                "dilation_kernel": 5,
                "dilation_iter": 2,
                "min_area": 80,
                "auto_threshold": False,
                "immobility_threshold": 0.08,
                "threshold_sensitivity": 0,
                "smoothing_window_s": 0.5,
                "min_immobility_duration_s": 0.2,
                "gap_merge_s": 0.1,
                "bin_seconds": 30,
                "rois": [{
                    "points": RECT,
                    "name": "Cylinder 1",
                    "cylinder_diameter_cm": 10.0,
                    "diameter_points": [{"x": 100, "y": 250}, {"x": 300, "y": 250}],
                    "time_config": {"mode": "global", "start": 0, "end": 0},
                }],
            }],
        }
    if experiment_type == "water_maze":
        return {
            "generate_video": False,
            "per_video_configs": [{
                "trial_type": "acquisition_hidden",
                "pool_diameter_cm": 120.0,
                "platform_radius_cm": 5.0,
                "target_quadrant": "NE",
                "start_position": "S",
                "bg_method": "threshold",
                "bg_type": "light",
                "gray_threshold": 127,
                "erosion_kernel": 3,
                "erosion_iter": 1,
                "dilation_kernel": 5,
                "dilation_iter": 2,
                "min_area": 80,
                "max_area": 50000,
                "thigmotaxis_band_cm": 10.0,
                "platform_dwell_s": 0.1,
                "track_smoothing_max_jump_cm": 25.0,
                "max_gap_s": 0.1,
                "movement_threshold_cm_s": 2.0,
                "bin_seconds": 10,
                "rois": [_water_maze_roi()],
            }],
        }
    if experiment_type == "novel_object":
        return {
            "generate_video": False,
            "per_video_configs": [{
                "phase": "recognition",
                "arena_width_cm": 50.0,
                "arena_height_cm": 50.0,
                "interaction_radius_cm": 4.0,
                "rois": [{
                    "points": RECT,
                    "name": "Arena 1",
                    "time_config": {"mode": "global", "start": 0, "end": 0},
                    "objects": [
                        {"name": "熟悉物体", "role": "familiar", "point": {"x": 160, "y": 220}},
                        {"name": "新奇物体", "role": "novel", "point": {"x": 240, "y": 220}},
                    ],
                }],
            }],
        }
    if experiment_type == "elevated_plus":
        return {
            "generate_video": False,
            "per_video_configs": [{
                "center_size_cm": 5.0,
                "open_arm_length_cm": 30.0,
                "closed_arm_length_cm": 30.0,
                "movement_threshold_cm_s": 2.0,
                "entry_depth_ratio": 0.2,
                "bin_seconds": 30,
                "rois": [_epm_roi()],
            }],
        }
    if experiment_type == "y_maze":
        return {
            "generate_video": False,
            "per_video_configs": [{
                "arm_length_cm": 30.0,
                "arm_width_cm": 10.0,
                "movement_threshold_cm_s": 2.0,
                "entry_depth_ratio": 0.2,
                "bin_seconds": 30,
                "rois": [_ymaze_roi()],
            }],
        }
    if experiment_type == "light_dark_box":
        return {
            "generate_video": False,
            "per_video_configs": [{
                "arena_width_cm": 40.0,
                "arena_height_cm": 20.0,
                "door_open_delay_s": 0.0,
                "transition_min_dwell_s": 0.01,
                "doorway_reset_s": 0.05,
                "movement_threshold_cm_s": 2.0,
                "bin_seconds": 30,
                "rois": [_ldb_roi()],
            }],
        }
    if experiment_type == "three_chamber":
        return {
            "generate_video": False,
            "per_video_configs": [{
                "analysis_mode": "single_phase",
                "selected_phase": "sociability",
                "arena_width_cm": 60.0,
                "arena_height_cm": 40.0,
                "interaction_radius_cm": 2.0,
                "cup_exclusion_radius_cm": 1.0,
                "transition_min_dwell_s": 0.01,
                "movement_threshold_cm_s": 2.0,
                "bin_seconds": 30,
                "role_schedule": {
                    "sociability": {"left": "social_stranger_1", "right": "empty_object"},
                    "social_novelty": {"left": "familiar_stranger_1", "right": "novel_stranger_2"},
                    "habituation_all": {"left": "empty_object", "right": "empty_object"},
                    "habituation_center": {"left": "", "right": ""},
                },
                "rois": [_three_chamber_roi()],
            }],
        }
    raise ValueError(f"Unknown experiment type: {experiment_type}")


def main() -> int:
    if not SAMPLE_VIDEO.exists():
        raise FileNotFoundError(f"Missing sample video for smoke test: {SAMPLE_VIDEO}")

    experiments = tuple(spec.key for spec in list_experiments())
    with tempfile.TemporaryDirectory(prefix="mousescope_smoke_") as temp_dir:
        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess["license_accepted"] = True

            for experiment_type in experiments:
                session_id = f"short-{experiment_type}-{int(time.time() * 1000)}"
                out_dir = str(pathlib.Path(temp_dir) / experiment_type)
                pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
                _seed_session(session_id, experiment_type, out_dir)

                response = client.post(f"/api/start_analysis/{session_id}", json=_payload_for(experiment_type))
                if response.status_code != 200:
                    raise RuntimeError(f"{experiment_type} start_analysis returned {response.status_code}: {response.get_data(as_text=True)}")

                deadline = time.time() + 90
                while time.time() < deadline:
                    status = analysis_sessions[session_id]["analysis"].get_status()
                    if status["status"] == "completed":
                        break
                    if status["status"] in {"error", "stopped"}:
                        raise RuntimeError(f"{experiment_type} finished with status={status['status']}: {status.get('message')}")
                    time.sleep(0.25)
                else:
                    raise TimeoutError(f"{experiment_type} did not complete within timeout")

        shutil.rmtree(temp_dir, ignore_errors=True)

    print("v3.4.0 short analysis smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
