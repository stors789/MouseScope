from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from services.analysis_builders import build_analysis_config
from services.experiment_defaults import (
    ExperimentDefaultsValidationError,
    experiment_defaults_schema,
    validate_experiment_defaults,
)
from services.experiment_defaults_store import ExperimentDefaultsStore


class ExperimentDefaultsTest(unittest.TestCase):
    def test_store_persists_all_modules_and_rejects_unknown_fields(self):
        with TemporaryDirectory() as temp_dir:
            store = ExperimentDefaultsStore(Path(temp_dir) / "experiment_defaults.json")
            defaults = store.get_defaults()
            self.assertIn("open_field", defaults)
            self.assertIn("water_maze", defaults)
            self.assertEqual(defaults["open_field"]["arena_side_cm"], 50.0)

            defaults["open_field"]["arena_side_cm"] = 66
            defaults["open_field"]["not_a_setting"] = "ignored"
            saved = store.save_defaults(defaults)

            self.assertEqual(saved["open_field"]["arena_side_cm"], 66.0)
            self.assertNotIn("not_a_setting", saved["open_field"])
            self.assertEqual(store.get_defaults()["open_field"]["arena_side_cm"], 66.0)

    def test_global_defaults_are_baseline_and_per_video_values_win(self):
        sess = {
            "video_infos": [{"path": "video.mp4"}],
            "output_dir": "results",
            "start_seconds": None,
            "end_seconds": None,
            "fps_override": None,
        }
        global_defaults = {
            "open_field": {
                "arena_side_cm": 66,
                "bin_seconds": 120,
                "movement_threshold_cm_s": 3.5,
            }
        }
        config = build_analysis_config(
            "open_field",
            sess,
            {"per_video_configs": [{"arena_side_cm": 55}]},
            global_defaults=global_defaults,
        )

        self.assertEqual(config["arena_side_cm"], 66.0)
        self.assertEqual(config["bin_seconds"], 120.0)
        self.assertEqual(config["per_video_configs"][0]["arena_side_cm"], 55.0)
        self.assertEqual(config["per_video_configs"][0]["bin_seconds"], 120.0)
        self.assertEqual(config["per_video_configs"][0]["movement_threshold_cm_s"], 3.5)

    def test_tst_defaults_use_one_algorithm_switch_and_propagate_max_area(self):
        sess = {
            "video_infos": [{"path": "video.mp4"}],
            "output_dir": "results",
            "start_seconds": None,
            "end_seconds": None,
            "fps_override": None,
        }
        config = build_analysis_config(
            "tst",
            sess,
            {"per_video_configs": [{}]},
            global_defaults={"tst": {"max_area": 43210, "use_tst_lis_primary": True}},
        )

        self.assertEqual(config["max_area"], 43210)
        self.assertEqual(config["per_video_configs"][0]["max_area"], 43210)
        self.assertTrue(config["use_tst_lis_primary"])
        self.assertFalse(config["use_tst_legacy_primary"])
        self.assertTrue(config["enable_tst_lis"])
        schema_keys = {field["key"] for field in experiment_defaults_schema()["modules"]["tst"]}
        self.assertIn("use_tst_lis_primary", schema_keys)
        self.assertNotIn("use_tst_legacy_primary", schema_keys)
        self.assertNotIn("enable_tst_lis", schema_keys)

    def test_each_experiment_module_propagates_global_defaults_to_runtime(self):
        sess = {
            "video_infos": [{"path": "video.mp4"}],
            "output_dir": "results",
            "start_seconds": None,
            "end_seconds": None,
            "fps_override": None,
        }
        cases = {
            "tst": ("max_area", 40101),
            "forced_swim": ("cylinder_diameter_cm", 13.1),
            "water_maze": ("pool_diameter_cm", 151.0),
            "open_field": ("arena_side_cm", 61.0),
            "novel_object": ("interaction_radius_cm", 6.1),
            "elevated_plus": ("center_size_cm", 7.1),
            "y_maze": ("arm_length_cm", 41.0),
            "light_dark_box": ("door_open_delay_s", 4.1),
            "three_chamber": ("interaction_radius_cm", 3.1),
        }
        for experiment, (field, value) in cases.items():
            with self.subTest(experiment=experiment, field=field):
                config = build_analysis_config(
                    experiment,
                    sess,
                    {"per_video_configs": [{}]},
                    global_defaults={experiment: {field: value}},
                )
                self.assertEqual(config[field], value)
                self.assertEqual(config["per_video_configs"][0][field], value)

    def test_invalid_numeric_defaults_are_rejected_before_writing(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiment_defaults.json"
            store = ExperimentDefaultsStore(path)
            defaults = store.get_defaults()
            defaults["open_field"]["erosion_kernel"] = 0

            with self.assertRaises(ExperimentDefaultsValidationError) as caught:
                store.save_defaults(defaults)

            self.assertFalse(path.exists())
            self.assertEqual(caught.exception.errors[0]["field"], "erosion_kernel")

    def test_cross_field_and_protocol_validation(self):
        defaults = ExperimentDefaultsStore(Path("unused.json")).get_defaults()
        defaults["open_field"]["min_area"] = 500
        defaults["open_field"]["max_area"] = 100
        defaults["three_chamber"]["protocol_segments"] = [
            {"id": "first", "phase": "a", "start_s": 0, "end_s": 10},
            {"id": "second", "phase": "b", "start_s": 9, "end_s": 20},
        ]

        errors = validate_experiment_defaults(defaults)
        keys = {(item["experiment"], item["field"]) for item in errors}
        self.assertIn(("open_field", "min_area"), keys)
        self.assertIn(("three_chamber", "protocol_segments"), keys)

    def test_schema_exposes_bounds_and_correct_three_chamber_phase_options(self):
        schema = experiment_defaults_schema()["modules"]
        open_field = {field["key"]: field for field in schema["open_field"]}
        self.assertEqual(open_field["gray_threshold"]["min"], 0)
        self.assertEqual(open_field["gray_threshold"]["max"], 255)
        self.assertEqual(open_field["erosion_kernel"]["min"], 1)
        self.assertEqual(open_field["bg_method"]["label"]["zh"], "背景建模方法")

        three_chamber = {field["key"]: field for field in schema["three_chamber"]}
        phase_values = {item["value"] for item in three_chamber["phase"]["options"]}
        self.assertIn("sociability", phase_values)
        self.assertNotIn("recognition", phase_values)

    def test_unsafe_legacy_file_is_repaired_when_loaded(self):
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "experiment_defaults.json"
            path.write_text(json.dumps({"open_field": {"erosion_kernel": 0, "min_area": 900, "max_area": 10}}), encoding="utf-8")
            loaded = ExperimentDefaultsStore(path).get_defaults()

        self.assertEqual(loaded["open_field"]["erosion_kernel"], 3)
        self.assertEqual(loaded["open_field"]["min_area"], 120)
        self.assertEqual(loaded["open_field"]["max_area"], 50000)

    def test_geometry_cross_field_validation(self):
        defaults = ExperimentDefaultsStore(Path("unused.json")).get_defaults()
        defaults["open_field"]["arena_side_cm"] = 30
        defaults["open_field"]["wall_margin_cm"] = 15
        defaults["water_maze"]["pool_diameter_cm"] = 20
        defaults["water_maze"]["platform_radius_cm"] = 10

        errors = validate_experiment_defaults(defaults)
        keys = {(item["experiment"], item["field"]) for item in errors}
        self.assertIn(("open_field", "wall_margin_cm"), keys)
        self.assertIn(("water_maze", "platform_radius_cm"), keys)


if __name__ == "__main__":
    unittest.main()
