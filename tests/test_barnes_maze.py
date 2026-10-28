import math
import pathlib
import sys
import unittest

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "core")]
from core.barnes_maze import (
    BarnesArena,
    BarnesScorer,
    estimate_silhouette_landmarks,
    validate_manual_events,
)


def arena(target="1"):
    points = [
        [50 + 40 * math.cos(t), 50 + 40 * math.sin(t)]
        for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)
    ]
    holes = [
        dict(id=str(i + 1), point=p)
        for i, p in enumerate([(80, 50), (50, 80), (20, 50), (50, 20)])
    ]
    return BarnesArena(points, holes, target, diameter_cm=80, hole_radius_cm=2)


def observation(nose=None, body=(50, 50), confidence=1.0):
    points = {}
    for name, p in [("body_center", body), ("nose", nose)]:
        if p is not None:
            points[name] = dict(x=p[0], y=p[1], confidence=confidence, visible=True)
    return dict(keypoints=points, source="test_fixture")


class BarnesGeometryTests(unittest.TestCase):
    def test_ellipse_calibration_and_hole_roundtrip(self):
        a = arena()
        for hole in a.holes:
            self.assertLess(
                math.dist(a.to_image(a.to_cm(hole["point"])), hole["point"]), 1e-4
            )
            self.assertEqual(a.hole_at(a.to_cm(hole["point"])), hole["id"])

    def test_outside_nose_is_not_clamped_to_rim(self):
        a = arena()
        self.assertGreater(math.hypot(*a.to_cm((180, 50))), a.radius_cm)
        self.assertIsNone(a.hole_at(a.to_cm((180, 50))))

    def test_duplicate_and_overlapping_holes_are_rejected(self):
        a = arena()
        holes = [dict(id="1", point=[80, 50]), dict(id="1", point=[50, 80])]
        with self.assertRaises(ValueError):
            BarnesArena(a.points, holes, "1", 80, 2)
        holes[1] = dict(id="2", point=[80.5, 50])
        with self.assertRaises(ValueError):
            BarnesArena(a.points, holes, "1", 80, 2)

    def test_missing_target_or_invalid_dimensions_rejected(self):
        a = arena()
        with self.assertRaises(ValueError):
            BarnesArena(
                a.points,
                [dict(id="1", point=[80, 50]), dict(id="2", point=[50, 80])],
                "3",
                80,
                2,
            )
        with self.assertRaises(ValueError):
            BarnesArena(a.points, a.holes, "1", float("nan"), 2)


class BarnesScoringTests(unittest.TestCase):
    def scorer(self, **kw):
        return BarnesScorer(
            arena(),
            visit_confirm_s=0.1,
            visit_reset_s=0.1,
            tracking_max_jump_cm=0,
            **kw
        )

    def test_dwell_uses_source_intervals_and_onset_timestamp(self):
        s = self.scorer()
        s.update(1.0, 0.04, observation((80, 50)))
        self.assertFalse(s.events)
        s.update(1.04, 0.06, observation((80, 50)))
        self.assertEqual(len(s.events), 1)
        self.assertEqual(s.events[0]["time_s"], 1.0)
        self.assertEqual(s.build_summary()["primary_latency_s"], 1.0)

    def test_withdrawal_and_repeat_same_hole_count_twice(self):
        s = self.scorer()
        for t, nose in [
            (0, (50, 80)),
            (0.1, (50, 80)),
            (0.2, (50, 50)),
            (0.3, (50, 80)),
        ]:
            s.update(t, 0.1, observation(nose))
        self.assertEqual([e["hole_id"] for e in s.events], ["2", "2"])
        self.assertEqual(s.build_summary()["total_errors"], 2)

    def test_missing_nose_does_not_rearm_same_hole(self):
        s = self.scorer()
        for t, nose in [(0, (50, 80)), (0.1, None), (0.2, None), (0.3, (50, 80))]:
            s.update(t, 0.1, observation(nose))
        self.assertEqual(len(s.events), 1)

    def test_unknown_gap_does_not_join_confirmation_streak(self):
        s = self.scorer()
        s.update(0, 0.05, observation((50, 80)))
        s.update(0.05, 0.05, observation())
        s.update(0.1, 0.05, observation((50, 80)))
        self.assertEqual(len(s.events), 0)
        s.update(0.15, 0.05, observation((50, 80)))
        self.assertEqual(s.events[0]["time_s"], 0.1)

    def test_primary_and_escape_endpoints_are_distinct(self):
        s = self.scorer(manual_escape_time_s=4.0)
        for t, nose in [
            (0, (50, 80)),
            (1, (20, 50)),
            (2, (80, 50)),
            (3, (50, 80)),
            (4, (80, 50)),
            (5, (20, 50)),
        ]:
            s.update(t, 1.0, observation(nose))
        result = s.finish()
        self.assertEqual(result["primary_latency_s"], 2.0)
        self.assertEqual(result["total_latency_s"], 4.0)
        self.assertEqual(result["primary_errors"], 2)
        self.assertEqual(result["total_errors"], 3)
        self.assertEqual(result["escape_status"], "confirmed_manual")
        self.assertEqual(len(s.events), 5)

    def test_detection_loss_is_only_escape_candidate(self):
        s = self.scorer()
        s.update(0, 0.1, observation((80, 50), body=(78, 50)))
        for i in range(1, 8):
            s.update(i * 0.1, 0.1, observation(body=None))
        result = s.finish()
        self.assertIsNotNone(result["escape_candidate_time_s"])
        self.assertIsNone(result["total_latency_s"])
        self.assertEqual(result["escape_status"], "not_confirmed")
        self.assertTrue(result["total_latency_censored"])

    def test_reappearance_retracts_disappearance_candidate(self):
        s = self.scorer()
        s.update(0, 0.1, observation((80, 50), body=(78, 50)))
        for i in range(1, 8):
            s.update(i * 0.1, 0.1, observation(body=None))
        s.update(0.8, 0.1, observation(body=(50, 50)))
        self.assertIsNone(s.finish()["escape_candidate_time_s"])

    def test_probe_does_not_require_escape(self):
        s = self.scorer(trial_type="probe")
        s.update(0, 0.1, observation((80, 50)))
        result = s.finish()
        self.assertEqual(result["escape_status"], "not_applicable")
        self.assertFalse(result["total_errors_censored"])
        with self.assertRaises(ValueError):
            self.scorer(trial_type="probe", manual_escape_time_s=3.0)

    def test_missing_target_is_censored_not_zero_latency(self):
        s = self.scorer()
        s.update(0, 0.1, observation((50, 80)))
        result = s.finish()
        self.assertIsNone(result["primary_latency_s"])
        self.assertTrue(result["primary_latency_censored"])

    def test_low_confidence_nose_does_not_create_visit(self):
        s = self.scorer()
        s.update(0, 0.1, observation((80, 50), confidence=0.1))
        self.assertFalse(s.events)
        self.assertEqual(s.finish()["nose_observed_percent"], 0)

    def test_no_path_is_drawn_across_missing_body(self):
        s = self.scorer()
        for i, p in enumerate([(30, 50), None, (70, 50)]):
            s.update(i * 0.1, 0.1, observation(body=p))
        self.assertEqual(s.finish()["total_distance_cm"], 0)

    def test_manual_events_are_explicit_and_keep_repeated_pokes(self):
        s = self.scorer(
            manual_events=[
                dict(time_s=0.1, hole_id="2"),
                dict(time_s=0.2, hole_id="2"),
                dict(time_s=0.3, hole_id="1"),
            ]
        )
        for i in range(4):
            s.update(i * 0.1, 0.1, observation())
        result = s.finish()
        self.assertEqual(result["hole_visit_sequence"], "2-2-1")
        self.assertEqual(result["primary_errors"], 2)
        self.assertEqual(result["event_source"], "manual")

    def test_explicit_empty_manual_list_does_not_enable_automatic_events(self):
        s = self.scorer(manual_events=[])
        s.update(0, 0.1, observation((80, 50)))
        self.assertFalse(s.events)

    def test_manual_events_require_valid_increasing_source_times(self):
        for events in [
            [dict(time_s=0.1, hole_id="99")],
            [dict(time_s=0.2, hole_id="1"), dict(time_s=0.1, hole_id="2")],
        ]:
            with self.assertRaises(ValueError):
                validate_manual_events(events, arena())

    def test_escape_at_final_frame_interval_boundary_is_preserved(self):
        s = self.scorer(manual_escape_time_s=1.0)
        for i in range(10):
            s.update(i * 0.1, 0.1, observation((80, 50)))
        self.assertEqual(s.finish()["total_latency_s"], 1.0)

    def test_manual_hole_events_after_escape_are_rejected(self):
        with self.assertRaises(ValueError):
            self.scorer(
                manual_escape_time_s=1.0, manual_events=[dict(time_s=2.0, hole_id="2")]
            )

    def test_nonmonotonic_source_time_is_rejected(self):
        s = self.scorer()
        s.update(1.0, 0.1, observation())
        with self.assertRaises(ValueError):
            s.update(1.0, 0.1, observation())


class ClassicalLandmarkTests(unittest.TestCase):
    def test_tail_resolves_head_without_hole_information(self):
        mask = np.zeros((100, 160), np.uint8)
        cv2.ellipse(mask, (85, 50), (22, 13), 0, 0, 360, 255, -1)
        cv2.line(mask, (64, 50), (20, 50), 255, 2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = estimate_silhouette_landmarks(contours[0])
        self.assertIn("nose", result["keypoints"])
        self.assertGreater(
            result["keypoints"]["nose"]["x"], result["keypoints"]["body_center"]["x"]
        )

    def test_ambiguous_round_silhouette_has_no_fabricated_nose(self):
        mask = np.zeros((100, 100), np.uint8)
        cv2.circle(mask, (50, 50), 15, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertNotIn(
            "nose", estimate_silhouette_landmarks(contours[0])["keypoints"]
        )


if __name__ == "__main__":
    unittest.main()
