import unittest
import pathlib
import sys
import os
from unittest.mock import patch

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CORE = ROOT / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from core.light_dark_box import LightDarkBoxTracker, _render_tracker_template_canvas
from core.three_chamber import ThreeChamberTracker
from services.analysis_builders import build_light_dark_box_config, build_three_chamber_config
from services.experiment_registry import (
    get_experiment_spec,
    is_experiment_release_hidden,
    is_three_chamber_enabled,
    list_experiments,
    normalize_experiment_type,
)
from vision_tracker import MouseDetector


class FakeBg:
    method = "threshold"

    def apply(self, frame, gray=None):
        return np.full(frame.shape[:2], 255, dtype=np.uint8)


class FakeMorph:
    def process(self, mask, inplace=False):
        return mask


class MaskBg:
    method = "static"

    def __init__(self, mask):
        self.mask = mask

    def apply(self, frame, gray=None):
        return self.mask.copy()


class QueuedMaskBg:
    method = "static"

    def __init__(self, masks):
        self.masks = [mask.copy() for mask in masks]
        self.last = self.masks[-1].copy() if self.masks else None

    def apply(self, frame, gray=None):
        if self.masks:
            self.last = self.masks.pop(0)
        if self.last is not None:
            return self.last.copy()
        return np.zeros(frame.shape[:2], dtype=np.uint8)


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


class NewAssayTrackerTests(unittest.TestCase):
    def test_light_dark_doorway_jitter_counts_one_transition(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "dark": [[0, 0], [45, 0], [45, 100], [0, 100]],
                "doorway": [[45, 35], [55, 35], [55, 65], [45, 65]],
                "light": [[55, 0], [110, 0], [110, 100], [55, 100]],
            },
            arena_width_cm=40,
            arena_height_cm=20,
            fps=10,
            door_open_delay_s=0,
            transition_min_dwell_s=0.2,
            doorway_reset_s=0.2,
        )
        detector = QueuedDetector([(20, 50), (20, 50), (50, 50), (50, 50), (80, 50), (80, 50), (80, 50)])
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for idx in range(7):
            tracker.update(frame, idx * 0.1, gray, FakeBg(), FakeMorph(), detector, dt_s=0.1)
        summary = tracker.build_summary()
        self.assertEqual(summary["total_transitions"], 1)
        self.assertEqual(summary["light_entries"], 1)
        self.assertAlmostEqual(summary["first_light_latency_s"], 0.5, places=2)

    def test_light_dark_dwell_debounces_without_doorway(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "dark": [[0, 0], [50, 0], [50, 100], [0, 100]],
                "light": [[50, 0], [110, 0], [110, 100], [50, 100]],
            },
            fps=10,
            door_open_delay_s=0,
            transition_min_dwell_s=0.2,
        )
        detector = QueuedDetector([(20, 50), (20, 50), (80, 50), (20, 50), (80, 50), (80, 50)])
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for idx in range(6):
            tracker.update(frame, idx * 0.1, gray, FakeBg(), FakeMorph(), detector, dt_s=0.1)
        summary = tracker.build_summary()
        self.assertEqual(summary["total_transitions"], 1)
        self.assertEqual(summary["light_entries"], 1)

    def test_light_dark_default_detection_mask_excludes_doorway(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "dark": [[0, 0], [50, 0], [50, 100], [0, 100]],
                "doorway": [[45, 35], [55, 35], [55, 65], [45, 65]],
                "light": [[50, 0], [110, 0], [110, 100], [50, 100]],
            },
        )
        tracker.initialize((120, 120, 3))
        self.assertIsNotNone(tracker.detection_mask)
        self.assertEqual(int(tracker.detection_mask[50, 50]), 0)
        self.assertEqual(int(tracker.detection_mask[50, 20]), 255)
        self.assertEqual(int(tracker.detection_mask[50, 80]), 255)

    def test_light_dark_doorway_bridge_merges_split_body_without_door_pixels(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "dark": [[0, 0], [50, 0], [50, 100], [0, 100]],
                "doorway": [[45, 35], [55, 35], [55, 65], [45, 65]],
                "light": [[50, 0], [110, 0], [110, 100], [50, 100]],
            },
        )
        mask = np.zeros((120, 120), dtype=np.uint8)
        cv2.rectangle(mask, (35, 45), (45, 54), 255, thickness=-1)
        cv2.rectangle(mask, (55, 45), (65, 54), 255, thickness=-1)
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = MouseDetector(min_area=120, max_area=50000, min_solidity=0.1)

        detection, fg_mask = tracker.detect_mouse(frame, gray, MaskBg(mask), FakeMorph(), detector)

        self.assertIsNotNone(detection)
        self.assertTrue(detection.get("doorway_bridge"))
        self.assertGreaterEqual(detection["area"], 120)
        self.assertAlmostEqual(detection["centroid"][0], 50.0, delta=1.0)
        self.assertEqual(int(fg_mask[50, 50]), 0)
        self.assertEqual(detector.detect(fg_mask), [])

    def test_light_dark_output_maps_do_not_draw_doorway(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "dark": [[0, 0], [50, 0], [50, 100], [0, 100]],
                "doorway": [[45, 35], [55, 35], [55, 65], [45, 65]],
                "light": [[50, 0], [110, 0], [110, 100], [50, 100]],
            },
        )
        _canvas, translated, _offset = _render_tracker_template_canvas(tracker, pixels_per_cm=4)
        self.assertEqual(set(translated), {"dark", "light"})

    def test_light_dark_invisible_dark_uses_light_visibility_only(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "light": [[50, 0], [110, 0], [110, 100], [50, 100]],
            },
            dark_box_visible=False,
            fps=10,
            door_open_delay_s=0,
            transition_min_dwell_s=0.2,
        )
        detector = QueuedDetector([None, None, (30, 50), (30, 50), None, None])
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for idx in range(6):
            tracker.update(frame, idx * 0.1, gray, FakeBg(), FakeMorph(), detector, dt_s=0.1)
        summary = tracker.build_summary()
        self.assertFalse(summary["dark_box_visible"])
        self.assertFalse(summary["tracking_reliable"])
        self.assertEqual(summary["light_entries"], 1)
        self.assertEqual(summary["total_transitions"], 2)
        self.assertAlmostEqual(summary["light_time_s"], 0.2, places=2)
        self.assertAlmostEqual(summary["dark_time_s"], 0.4, places=2)
        self.assertIsNone(summary["total_distance_cm"])
        self.assertIsNone(summary["mean_speed_cm_s"])
        self.assertFalse(tracker.trajectory_template_cm_stats)

    def test_light_dark_invisible_dark_doorway_excludes_head_peek(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "light": [[50, 0], [110, 0], [110, 100], [50, 100]],
                "doorway": [[50, 35], [66, 35], [66, 65], [50, 65]],
            },
            dark_box_visible=False,
            fps=10,
            door_open_delay_s=0,
            transition_min_dwell_s=0.2,
        )
        tracker.initialize((120, 120, 3))
        self.assertEqual(int(tracker.detection_mask[50, 55]), 0)
        self.assertEqual(int(tracker.detection_mask[50, 80]), 255)

        masks = []
        for kind in ["dark", "dark", "doorway", "doorway", "light", "light"]:
            mask = np.zeros((120, 120), dtype=np.uint8)
            if kind == "doorway":
                cv2.rectangle(mask, (54, 45), (62, 55), 255, thickness=-1)
            elif kind == "light":
                cv2.rectangle(mask, (72, 45), (82, 55), 255, thickness=-1)
            masks.append(mask)

        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = MouseDetector(min_area=30, max_area=50000, min_solidity=0.1)
        bg = QueuedMaskBg(masks)
        for idx in range(6):
            tracker.update(frame, idx * 0.1, gray, bg, FakeMorph(), detector, dt_s=0.1)

        summary = tracker.build_summary()
        self.assertFalse(summary["dark_box_visible"])
        self.assertEqual(summary["detected_frames"], 2)
        self.assertEqual(summary["light_entries"], 1)
        self.assertEqual(summary["total_transitions"], 1)
        self.assertAlmostEqual(summary["first_light_latency_s"], 0.5, places=2)
        self.assertAlmostEqual(summary["light_time_s"], 0.2, places=2)
        self.assertAlmostEqual(summary["dark_time_s"], 0.4, places=2)

    def test_light_dark_trackmap_uses_rectified_template(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "dark": [[10, 10], [48, 0], [58, 90], [20, 100]],
                "light": [[58, 0], [110, 12], [98, 112], [50, 90]],
            },
            arena_width_cm=40,
            arena_height_cm=20,
            fps=10,
            door_open_delay_s=0,
            transition_min_dwell_s=0.1,
        )
        detector = QueuedDetector([(30, 50), (30, 50), (78, 50), (78, 50)])
        frame = np.zeros((130, 130, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for idx in range(4):
            tracker.update(frame, idx * 0.1, gray, FakeBg(), FakeMorph(), detector, dt_s=0.1)
        self.assertTrue(tracker.trajectory_template_cm_stats)
        xs = [p[0] for p in tracker.trajectory_template_cm_stats]
        ys = [p[1] for p in tracker.trajectory_template_cm_stats]
        self.assertGreaterEqual(min(xs), 0.0)
        self.assertLessEqual(max(xs), 40.0)
        self.assertGreaterEqual(min(ys), 0.0)
        self.assertLessEqual(max(ys), 20.0)

    def test_light_dark_default_template_forms_square_apparatus(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "dark": [[0, 0], [100, 0], [100, 50], [0, 50]],
                "light": [[0, 50], [100, 50], [100, 100], [0, 100]],
            },
        )
        self.assertEqual(tracker.template_layout, "vertical")
        self.assertAlmostEqual(tracker.light_box_length_cm, 40.0, places=3)
        self.assertAlmostEqual(tracker.light_box_width_cm, 20.0, places=3)
        self.assertAlmostEqual(tracker.dark_box_length_cm, 40.0, places=3)
        self.assertAlmostEqual(tracker.dark_box_width_cm, 20.0, places=3)
        self.assertAlmostEqual(tracker.shared_edge_length_cm, 40.0, places=3)
        self.assertAlmostEqual(tracker.arena_width_cm, 40.0, places=3)
        self.assertAlmostEqual(tracker.arena_height_cm, 40.0, places=3)

    def test_light_dark_old_bad_default_dimensions_are_normalized(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "dark": [[0, 0], [100, 0], [100, 50], [0, 50]],
                "light": [[0, 50], [100, 50], [100, 100], [0, 100]],
            },
            light_box_length_cm=20,
            light_box_width_cm=40,
            dark_box_length_cm=20,
            dark_box_width_cm=40,
            shared_edge_length_cm=40,
        )
        self.assertAlmostEqual(tracker.light_box_length_cm, 40.0, places=3)
        self.assertAlmostEqual(tracker.light_box_width_cm, 20.0, places=3)
        self.assertAlmostEqual(tracker.arena_width_cm, 40.0, places=3)
        self.assertAlmostEqual(tracker.arena_height_cm, 40.0, places=3)

    def test_light_dark_template_uses_independent_chamber_dimensions(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "dark": [[0, 0], [50, 0], [50, 100], [0, 100]],
                "light": [[50, 20], [150, 20], [150, 80], [50, 80]],
            },
            light_box_length_cm=30,
            light_box_width_cm=15,
            dark_box_length_cm=20,
            dark_box_width_cm=10,
            shared_edge_length_cm=15,
        )
        self.assertEqual(tracker.template_layout, "horizontal")
        self.assertEqual(tracker.template_dark_shared_side, "length")
        self.assertEqual(tracker.template_light_shared_side, "width")
        self.assertAlmostEqual(tracker.arena_width_cm, 40.0, places=3)
        self.assertAlmostEqual(tracker.arena_height_cm, 20.0, places=3)
        self.assertAlmostEqual(tracker.shared_edge_length_cm, 15.0, places=3)
        dark = tracker.template_polygons_cm["dark"]
        light = tracker.template_polygons_cm["light"]
        self.assertAlmostEqual(float(dark[:, 0].max()), float(light[:, 0].min()), places=3)

    def test_light_dark_template_can_follow_vertical_layout(self):
        tracker = LightDarkBoxTracker(
            zone_polygons={
                "dark": [[0, 0], [80, 0], [80, 40], [0, 40]],
                "light": [[20, 40], [70, 40], [70, 140], [20, 140]],
            },
            light_box_length_cm=30,
            light_box_width_cm=18,
            dark_box_length_cm=24,
            dark_box_width_cm=12,
            shared_edge_length_cm=18,
        )
        self.assertEqual(tracker.template_layout, "vertical")
        self.assertAlmostEqual(tracker.arena_width_cm, 24.0, places=3)
        self.assertAlmostEqual(tracker.arena_height_cm, 42.0, places=3)
        dark = tracker.template_polygons_cm["dark"]
        light = tracker.template_polygons_cm["light"]
        self.assertAlmostEqual(float(dark[:, 1].max()), float(light[:, 1].min()), places=3)

    def test_three_chamber_full_protocol_ignores_gap(self):
        tracker = ThreeChamberTracker(
            zone_polygons={
                "left": [[0, 0], [40, 0], [40, 100], [0, 100]],
                "center": [[40, 0], [70, 0], [70, 100], [40, 100]],
                "right": [[70, 0], [110, 0], [110, 100], [70, 100]],
            },
            objects=[
                {"side": "left", "point": {"x": 20, "y": 50}},
                {"side": "right", "point": {"x": 90, "y": 50}},
            ],
            role_schedule={
                "sociability": {"left": "social_stranger_1", "right": "empty_object"},
                "social_novelty": {"left": "familiar_stranger_1", "right": "novel_stranger_2"},
            },
            protocol_segments=[
                {"id": "soc", "phase": "sociability", "start_s": 0.0, "end_s": 1.0, "analyze": True},
                {"id": "gap", "phase": "ignore", "start_s": 1.0, "end_s": 2.0, "analyze": False},
                {"id": "nov", "phase": "social_novelty", "start_s": 2.0, "end_s": 3.0, "analyze": True},
            ],
            analysis_mode="full_protocol",
            interaction_radius_cm=20,
            cup_exclusion_radius_cm=0,
            transition_min_dwell_s=0.1,
            fps=10,
        )
        detector = QueuedDetector([(20, 50), (20, 50), (20, 50), (90, 50), (90, 50)])
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for t in [0.0, 0.1, 1.5, 2.0, 2.1]:
            tracker.update(frame, t, gray, FakeBg(), FakeMorph(), detector, dt_s=0.1)
        phase_rows = tracker.build_phase_summaries()
        self.assertEqual([row["phase"] for row in phase_rows], ["sociability", "social_novelty"])
        self.assertAlmostEqual(sum(float(row["total_time_s"]) for row in phase_rows), 0.4, places=3)
        summary = tracker.build_summary()
        self.assertIsNotNone(summary["sociability_index"])
        self.assertIsNotNone(summary["social_novelty_index"])


class NewAssayIntegrationTests(unittest.TestCase):
    def test_three_chamber_hidden_by_default_and_flag_enabled(self):
        disabled_env = {
            "MOUSESCOPE_ENABLE_3CST": "",
            "MOUSESCOPE_ENABLE_THREE_CHAMBER": "",
        }
        with patch.dict(os.environ, disabled_env):
            self.assertTrue(is_experiment_release_hidden("three_chamber"))
            self.assertEqual(normalize_experiment_type("three_chamber"), "tst")
            self.assertNotIn("three_chamber", [spec.key for spec in list_experiments()])
            self.assertEqual(get_experiment_spec("three_chamber").key, "three_chamber")

        with patch.dict(os.environ, {"MOUSESCOPE_ENABLE_3CST": "1", "MOUSESCOPE_ENABLE_THREE_CHAMBER": ""}):
            self.assertTrue(is_three_chamber_enabled())
            self.assertFalse(is_experiment_release_hidden("three_chamber"))
            self.assertEqual(normalize_experiment_type("three_chamber"), "three_chamber")
            self.assertIn("three_chamber", [spec.key for spec in list_experiments()])

    def test_registry_validators_accept_minimal_payloads(self):
        ldb_spec = get_experiment_spec("light_dark_box")
        ok, missing = ldb_spec.validator({"rois": [{
            "zones": {
                "light": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
                "dark": [{"x": 2, "y": 0}, {"x": 3, "y": 0}, {"x": 3, "y": 1}],
            }
        }]})
        self.assertTrue(ok, missing)
        ok, missing = ldb_spec.validator({"dark_box_visible": False, "rois": [{
            "zones": {
                "light": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
                "doorway": [{"x": 0, "y": 0}, {"x": 0.5, "y": 0}, {"x": 0.5, "y": 1}],
            }
        }]})
        self.assertTrue(ok, missing)
        ok, missing = ldb_spec.validator({"dark_box_visible": False, "rois": [{
            "zones": {
                "light": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
                "doorway": [{"x": 0, "y": 0}, {"x": 0.5, "y": 0}],
            }
        }]})
        self.assertFalse(ok)
        three_spec = get_experiment_spec("three_chamber")
        ok, missing = three_spec.validator({"analysis_mode": "single_phase", "selected_phase": "sociability", "rois": [{
            "zones": {
                "left": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}],
                "center": [{"x": 2, "y": 0}, {"x": 3, "y": 0}, {"x": 3, "y": 1}],
                "right": [{"x": 4, "y": 0}, {"x": 5, "y": 0}, {"x": 5, "y": 1}],
            },
            "objects": [
                {"side": "left", "point": {"x": 0, "y": 0}},
                {"side": "right", "point": {"x": 1, "y": 0}},
            ],
        }]})
        self.assertTrue(ok, missing)

    def test_builders_create_per_video_configs(self):
        sess = {"video_infos": [{}, {}], "output_dir": None, "start_seconds": None, "end_seconds": None}
        ldb = build_light_dark_box_config(sess, {
            "dark_box_visible": False,
            "light_box_length_cm": 30,
            "dark_box_length_cm": 18,
            "shared_edge_length_cm": 12,
            "per_video_configs": [{"rois": [{"zones": {"light": [1, 2, 3]}}]}],
        })
        self.assertEqual(len(ldb["per_video_configs"]), 2)
        self.assertFalse(ldb["per_video_configs"][0]["dark_box_visible"])
        self.assertEqual(ldb["per_video_configs"][0]["light_box_length_cm"], 30)
        self.assertEqual(ldb["per_video_configs"][0]["shared_edge_length_cm"], 12)
        defaults = build_light_dark_box_config(sess, {})
        self.assertEqual(defaults["light_box_length_cm"], 40)
        self.assertEqual(defaults["light_box_width_cm"], 20)
        self.assertEqual(defaults["dark_box_length_cm"], 40)
        self.assertEqual(defaults["dark_box_width_cm"], 20)
        self.assertEqual(defaults["shared_edge_length_cm"], 40)
        three = build_three_chamber_config(sess, {"analysis_mode": "full_protocol", "per_video_configs": [{}]})
        self.assertEqual(three["per_video_configs"][0]["analysis_mode"], "full_protocol")
        self.assertTrue(three["per_video_configs"][0]["protocol_segments"])


if __name__ == "__main__":
    unittest.main()
