import math
import pathlib
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from core.water_maze import PoolGeometry, WaterMazeTracker
from services.analysis_builders import build_water_maze_config
from services.analysis_runtime import AnalysisSessionBase
from services.experiment_registry import get_experiment_spec


class FakeBg:
    method = "threshold"

    def apply(self, frame, learning_rate=-1, gray=None):
        return np.full(frame.shape[:2], 255, dtype=np.uint8)


class FakeMorph:
    def process(self, mask, inplace=False):
        return mask


class QueuedDetector:
    def __init__(self, centroids):
        self.centroids = list(centroids)

    def detect(self, _mask):
        if not self.centroids:
            return []
        centroid = self.centroids.pop(0)
        if centroid is None:
            return []
        x, y = centroid
        return [{
            "centroid": (float(x), float(y)),
            "area": 100.0,
            "bbox": (int(x) - 2, int(y) - 2, 4, 4),
            "contour": None,
        }]


def ellipse_boundary(center=(120.0, 110.0), axes=(150.0, 110.0), angle_deg=18.0, count=80):
    cx, cy = center
    a, b = axes[0] * 0.5, axes[1] * 0.5
    angle = math.radians(angle_deg)
    out = []
    for idx in range(count):
        theta = 2.0 * math.pi * idx / count
        x = a * math.cos(theta)
        y = b * math.sin(theta)
        out.append([
            cx + x * math.cos(angle) - y * math.sin(angle),
            cy + x * math.sin(angle) + y * math.cos(angle),
        ])
    return out


def run_tracker(tracker, detector_points, dt=0.1):
    frame = np.zeros((240, 260, 3), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tracker.initialize(frame.shape)
    x0, y0, _w, _h = tracker.detection_bbox
    local_points = [
        None if point is None else (float(point[0]) - x0, float(point[1]) - y0)
        for point in detector_points
    ]
    detector = QueuedDetector(local_points)
    for idx in range(len(detector_points)):
        tracker.update(frame, idx * dt, gray, FakeBg(), FakeMorph(), detector, dt_s=dt)
    return tracker.build_summary()


class WaterMazeCoreTests(unittest.TestCase):
    def test_ellipse_boundary_maps_to_standard_circle(self):
        boundary = ellipse_boundary()
        geometry = PoolGeometry(boundary, pool_diameter_cm=120.0)

        radii = [math.hypot(*geometry.image_to_pool(point)) for point in boundary]

        self.assertAlmostEqual(float(np.mean(radii)), 60.0, delta=0.8)
        self.assertLess(float(np.std(radii)), 0.8)
        self.assertLess(geometry.boundary_residual_max_percent, 2.0)

    def test_escape_latency_path_and_platform_entries(self):
        boundary = ellipse_boundary()
        geometry = PoolGeometry(boundary, pool_diameter_cm=120.0)
        platform_img = geometry.pool_to_image((30.0, -30.0))
        detector_points = [
            geometry.pool_to_image((-35.0, 40.0)),
            geometry.pool_to_image((-10.0, 20.0)),
            geometry.pool_to_image((10.0, 0.0)),
            geometry.pool_to_image((24.0, -20.0)),
            geometry.pool_to_image((30.0, -30.0)),
            geometry.pool_to_image((31.0, -29.0)),
        ]
        tracker = WaterMazeTracker(
            pool_boundary_points=boundary,
            platform={"center": platform_img, "radius_cm": 8.0},
            pool_diameter_cm=120.0,
            platform_dwell_s=0.2,
            fps=10.0,
            track_smoothing_max_jump_cm=80.0,
        )

        summary = run_tracker(tracker, detector_points)

        self.assertTrue(summary["success"])
        self.assertAlmostEqual(summary["escape_latency_s"], 0.5, places=2)
        self.assertEqual(summary["platform_entries"], 1)
        self.assertGreater(summary["total_distance_cm"], 80.0)
        self.assertIsNotNone(summary["path_length_to_platform_cm"])
        self.assertAlmostEqual(summary["detected_rate_percent"], 100.0, places=2)
        self.assertEqual(summary["search_strategy"], "direct_path")
        self.assertEqual(summary["search_strategy_score"], 7)
        self.assertIn("directness", summary["search_strategy_basis"])

    def test_probe_trial_counts_target_quadrant_and_platform_crossings(self):
        boundary = ellipse_boundary()
        geometry = PoolGeometry(boundary, pool_diameter_cm=120.0)
        platform_img = geometry.pool_to_image((30.0, -30.0))
        detector_points = [
            geometry.pool_to_image((42.0, -42.0)),
            geometry.pool_to_image((30.0, -30.0)),
            geometry.pool_to_image((45.0, -45.0)),
            geometry.pool_to_image((0.0, 25.0)),
            geometry.pool_to_image((30.0, -30.0)),
        ]
        tracker = WaterMazeTracker(
            pool_boundary_points=boundary,
            platform={"center": platform_img, "radius_cm": 7.0},
            pool_diameter_cm=120.0,
            trial_type="probe",
            target_quadrant="NE",
            fps=10.0,
            platform_dwell_s=99.0,
            track_smoothing_max_jump_cm=100.0,
        )

        summary = run_tracker(tracker, detector_points)

        self.assertEqual(summary["trial_type"], "probe")
        self.assertEqual(summary["target_quadrant"], "NE")
        self.assertGreater(summary["target_quadrant_percent"], 70.0)
        self.assertEqual(summary["platform_crossings"], 2)
        self.assertFalse(summary["success"])

    def test_thigmotaxis_percent_uses_standard_pool_coordinates(self):
        boundary = ellipse_boundary()
        geometry = PoolGeometry(boundary, pool_diameter_cm=120.0)
        platform_img = geometry.pool_to_image((0.0, 0.0))
        detector_points = [
            geometry.pool_to_image((55.0, 0.0)),
            geometry.pool_to_image((56.0, 6.0)),
            geometry.pool_to_image((54.0, -8.0)),
            geometry.pool_to_image((5.0, 0.0)),
        ]
        tracker = WaterMazeTracker(
            pool_boundary_points=boundary,
            platform={"center": platform_img, "radius_cm": 5.0},
            pool_diameter_cm=120.0,
            thigmotaxis_band_cm=10.0,
            fps=10.0,
            platform_dwell_s=99.0,
            track_smoothing_max_jump_cm=100.0,
        )

        summary = run_tracker(tracker, detector_points)

        self.assertGreaterEqual(summary["thigmotaxis_percent"], 75.0)
        self.assertGreater(summary["thigmotaxis_time_s"], 0.0)
        self.assertEqual(summary["search_strategy"], "thigmotaxis")
        self.assertEqual(summary["search_strategy_score"], 0)

    def test_large_jump_is_filtered_from_path_length(self):
        boundary = ellipse_boundary()
        geometry = PoolGeometry(boundary, pool_diameter_cm=120.0)
        platform_img = geometry.pool_to_image((0.0, 0.0))
        detector_points = [
            geometry.pool_to_image((0.0, 0.0)),
            geometry.pool_to_image((45.0, 45.0)),
        ]
        tracker = WaterMazeTracker(
            pool_boundary_points=boundary,
            platform={"center": platform_img, "radius_cm": 5.0},
            pool_diameter_cm=120.0,
            fps=10.0,
            track_smoothing_max_jump_cm=5.0,
        )

        summary = run_tracker(tracker, detector_points)

        self.assertEqual(summary["jump_filtered_count"], 1)
        self.assertAlmostEqual(summary["total_distance_cm"], 0.0, places=3)
        self.assertIn("large_jump_filtered", summary["qc_flags"])

    def test_registry_and_builder_include_water_maze(self):
        spec = get_experiment_spec("water_maze")
        self.assertEqual(spec.short_label, "MWM")
        complete, missing = spec.validator({
            "pool_diameter_cm": 120,
            "rois": [{
                "pool_boundary_points": ellipse_boundary(),
                "platform": {"center": {"x": 120, "y": 110}, "radius_cm": 5},
            }],
        })
        self.assertTrue(complete, missing)

        config = build_water_maze_config(
            {"video_infos": [{}, {}], "output_dir": "out", "start_seconds": "", "end_seconds": ""},
            {
                "pool_diameter_cm": 120,
                "platform_radius_cm": 5,
                "target_quadrant": "NE",
                "rois": [{
                    "pool_boundary_points": ellipse_boundary(),
                    "platform": {"center": {"x": 120, "y": 110}, "radius_cm": 5},
                }],
            },
        )
        self.assertEqual(config["target_quadrant"], "NE")
        self.assertEqual(len(config["per_video_configs"]), 2)
        self.assertEqual(config["per_video_configs"][0]["rois"][0]["platform"]["radius_cm"], 5)

    def test_calibration_template_exposes_core_tuning_controls(self):
        html = (ROOT / "templates" / "calibrate_water_maze.html").read_text(encoding="utf-8")
        for control_id in [
            "thigmotaxisBandCm",
            "platformDwellS",
            "movementThreshold",
            "binSeconds",
            "maxJumpCm",
            "maxGapS",
            "excludePlatform",
        ]:
            self.assertIn(f'id="{control_id}"', html)

        manual = (ROOT / "templates" / "manual_mwm_content.html").read_text(encoding="utf-8")
        for field_name in [
            "thigmotaxis_band_cm",
            "platform_dwell_s",
            "track_smoothing_max_jump_cm",
            "max_gap_s",
            "exclude_platform_from_detection",
        ]:
            self.assertIn(field_name, manual)

    def test_registry_accepts_dragged_pool_ellipse(self):
        spec = get_experiment_spec("water_maze")
        complete, missing = spec.validator({
            "pool_diameter_cm": 120,
            "rois": [{
                "pool": {
                    "center": {"x": 120, "y": 110},
                    "radius_x_px": 76,
                    "radius_y_px": 54,
                    "angle_deg": 0,
                },
                "platform": {
                    "center": {"x": 145, "y": 92},
                    "radius_x_px": 8,
                    "radius_y_px": 6,
                    "radius_cm": 5,
                },
            }],
        })
        self.assertTrue(complete, missing)

    def test_dragged_pool_payload_preserves_upper_right_platform(self):
        pool = {
            "center": {"x": 120, "y": 110},
            "radius_x_px": 76,
            "radius_y_px": 54,
            "angle_deg": 0,
        }
        tracker = WaterMazeTracker(
            pool=pool,
            platform={"center": {"x": 158, "y": 83}, "radius_cm": 5},
            pool_diameter_cm=120.0,
        )

        self.assertGreater(tracker.platform_center_cm[0], 0.0)
        self.assertLess(tracker.platform_center_cm[1], 0.0)
        self.assertEqual(tracker.target_quadrant, "NE")
        self.assertEqual(tracker.quadrant_label(tracker.target_quadrant), "I")

    def test_coordinate_rotation_controls_quadrant_and_release_point(self):
        pool = {
            "center": {"x": 100, "y": 100},
            "radius_x_px": 100,
            "radius_y_px": 100,
            "angle_deg": 0,
        }
        tracker = WaterMazeTracker(
            pool=pool,
            platform={"center": {"x": 150, "y": 100}, "radius_cm": 5},
            pool_diameter_cm=120.0,
            quadrant_rotation_deg=90.0,
            start_position="N",
        )

        self.assertEqual(tracker.target_quadrant, "NE")
        self.assertAlmostEqual(tracker.release_point_cm[0], 60.0, places=3)
        self.assertAlmostEqual(tracker.release_point_cm[1], 0.0, places=3)
        display_platform = tracker.point_to_coordinate_frame(tracker.platform_center_cm)
        self.assertIsNotNone(display_platform)
        self.assertAlmostEqual(display_platform[0], 0.0, places=3)
        self.assertLess(display_platform[1], 0.0)

    def test_sampling_scaling_handles_dragged_ellipse_payload(self):
        roi = {
            "pool": {
                "center": {"x": 120, "y": 80},
                "radius_x_px": 60,
                "radius_y_px": 40,
                "angle_deg": 0,
                "diameter_cm": 120,
            },
            "platform": {
                "center": {"x": 150, "y": 60},
                "radius_x_px": 8,
                "radius_y_px": 6,
                "radius_cm": 5,
            },
            "pool_boundary_points": [{"x": 120, "y": 40}, {"x": 180, "y": 80}],
        }

        scaled = AnalysisSessionBase._scale_geometry_value(roi, 0.5, 0.25)

        self.assertEqual(scaled["pool"]["center"], {"x": 60.0, "y": 20.0})
        self.assertEqual(scaled["pool"]["radius_x_px"], 30.0)
        self.assertEqual(scaled["pool"]["radius_y_px"], 10.0)
        self.assertEqual(scaled["pool"]["diameter_cm"], 120)
        self.assertEqual(scaled["platform"]["center"], {"x": 75.0, "y": 15.0})
        self.assertEqual(scaled["platform"]["radius_x_px"], 4.0)
        self.assertEqual(scaled["platform"]["radius_y_px"], 1.5)
        self.assertEqual(scaled["platform"]["radius_cm"], 5)
        self.assertEqual(scaled["pool_boundary_points"][0], {"x": 60.0, "y": 10.0})

    def test_txt_report_uses_trial_specific_metric_sections(self):
        import app  # noqa: F401 - configures analysis runtime helpers
        from services.water_maze_session import WaterMazeMultiPoolAnalysisSession

        base_summary = {
            "roi_id": 1,
            "roi_name": "Pool 1",
            "video_name": "mwm.mp4",
            "target_quadrant": "NE",
            "target_quadrant_label": "I",
            "coordinate_rotation_deg": 0,
            "time_mode": "global",
            "total_time_s": 60,
            "detected_rate_percent": 98.5,
            "success": True,
            "escape_latency_s": 18.2,
            "path_length_to_platform_cm": 320.0,
            "time_to_first_platform_entry_s": 17.8,
            "latency_to_first_platform_crossing_s": 19.0,
            "platform_entries": 2,
            "platform_crossings": 2,
            "time_in_platform_zone_s": 3.0,
            "path_efficiency": 0.42,
            "mean_distance_to_platform_cm": 28.0,
            "gallagher_proximity_score": 28.0,
            "cipl": 31.0,
            "target_quadrant_time_s": 32.0,
            "target_quadrant_percent": 53.3,
            "target_quadrant_distance_cm": 280.0,
            "quadrant_ne_time_s": 32.0,
            "quadrant_ne_percent": 53.3,
            "quadrant_nw_time_s": 8.0,
            "quadrant_nw_percent": 13.3,
            "quadrant_sw_time_s": 9.0,
            "quadrant_sw_percent": 15.0,
            "quadrant_se_time_s": 11.0,
            "quadrant_se_percent": 18.4,
            "total_distance_cm": 700.0,
            "mean_swim_speed_cm_s": 11.7,
            "max_swim_speed_cm_s": 34.5,
            "thigmotaxis_time_s": 6.0,
            "thigmotaxis_percent": 10.0,
            "whishaw_corridor_time_s": 12.0,
            "whishaw_corridor_percent": 20.0,
            "moving_time_s": 55.0,
            "floating_time_s": 5.0,
            "initial_heading_error_deg": 22.0,
            "heading_error_deg": 35.0,
            "search_strategy": "focal_search",
            "search_strategy_label": "Focal Search",
            "search_strategy_label_zh": "局灶搜索",
            "search_strategy_display": "局灶搜索 / Focal Search",
            "search_strategy_score": 5,
            "search_strategy_confidence": 0.82,
            "search_strategy_basis": "directness=0.44; target_quadrant_percent=53.30; near_platform_percent=37.00",
            "qc_flags": "",
        }
        session = WaterMazeMultiPoolAnalysisSession("report-test", [], [], {})
        with tempfile.TemporaryDirectory() as tmpdir:
            probe_path = pathlib.Path(tmpdir) / "probe.txt"
            probe_summary = dict(base_summary, trial_type="probe", success=False, escape_latency_s=None)
            session._generate_txt_report([probe_summary], str(probe_path))
            probe_text = probe_path.read_text(encoding="utf-8")

            self.assertIn("Probe Trial Memory Metrics", probe_text)
            self.assertIn("Quadrant Occupancy", probe_text)
            self.assertIn("Former Platform Crossings", probe_text)
            self.assertIn("Search Strategy", probe_text)
            self.assertIn("Focal Search", probe_text)
            self.assertNotIn("Escape Latency", probe_text)
            self.assertNotIn("Hidden Platform Learning Metrics", probe_text)

            visible_path = pathlib.Path(tmpdir) / "visible.txt"
            visible_summary = dict(base_summary, trial_type="visible_platform")
            session._generate_txt_report([visible_summary], str(visible_path))
            visible_text = visible_path.read_text(encoding="utf-8")

            self.assertIn("Visible Platform Control Metrics", visible_text)
            self.assertIn("Escape Latency", visible_text)
            self.assertIn("Search Strategy Basis", visible_text)
            self.assertNotIn("Probe Trial Memory Metrics", visible_text)


if __name__ == "__main__":
    unittest.main()
