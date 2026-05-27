import unittest

from services.analysis_builders import build_tst_config


class TSTAlgorithmDefaultTest(unittest.TestCase):
    def _session(self):
        return {
            "video_infos": [{"path": "sample.mp4", "original_name": "sample.mp4"}],
            "output_dir": None,
            "start_seconds": None,
            "end_seconds": None,
            "fps_override": None,
        }

    def test_tst_defaults_to_legacy_algorithm(self):
        cfg = build_tst_config(self._session(), {})

        self.assertFalse(cfg["use_tst_lis_primary"])
        self.assertTrue(cfg["use_tst_legacy_primary"])
        self.assertFalse(cfg["enable_tst_lis"])

        per_video = cfg["per_video_configs"][0]
        self.assertFalse(per_video["use_tst_lis_primary"])
        self.assertTrue(per_video["use_tst_legacy_primary"])
        self.assertFalse(per_video["enable_tst_lis"])

    def test_tst_new_algorithm_can_be_enabled_explicitly(self):
        cfg = build_tst_config(
            self._session(),
            {
                "use_tst_lis_primary": True,
                "auto_behavior_threshold": True,
                "per_video_configs": [
                    {
                        "use_tst_lis_primary": True,
                        "auto_behavior_threshold": True,
                    }
                ],
            },
        )

        self.assertTrue(cfg["use_tst_lis_primary"])
        self.assertFalse(cfg["use_tst_legacy_primary"])
        self.assertTrue(cfg["enable_tst_lis"])
        self.assertFalse(cfg["auto_behavior_threshold"])

        per_video = cfg["per_video_configs"][0]
        self.assertTrue(per_video["use_tst_lis_primary"])
        self.assertFalse(per_video["use_tst_legacy_primary"])
        self.assertTrue(per_video["enable_tst_lis"])
        self.assertFalse(per_video["auto_behavior_threshold"])

    def test_tst_legacy_key_still_overrides_per_video_new_default(self):
        cfg = build_tst_config(
            self._session(),
            {
                "use_tst_lis_primary": True,
                "per_video_configs": [{"use_tst_legacy_primary": True}],
            },
        )

        self.assertTrue(cfg["use_tst_lis_primary"])
        per_video = cfg["per_video_configs"][0]
        self.assertFalse(per_video["use_tst_lis_primary"])
        self.assertTrue(per_video["use_tst_legacy_primary"])


if __name__ == "__main__":
    unittest.main()
