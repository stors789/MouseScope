from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.output_runs import build_run_output_dir, prepare_session_run_output


class OutputRunIsolationTest(unittest.TestCase):
    def test_same_second_sessions_get_different_directories(self):
        with TemporaryDirectory() as temp_dir:
            first = build_run_output_dir(
                temp_dir,
                session_id="a" * 32,
                experiment_type="open_field",
                timestamp="20260710_120000",
            )
            second = build_run_output_dir(
                temp_dir,
                session_id="b" * 32,
                experiment_type="open_field",
                timestamp="20260710_120000",
            )

        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).parent, Path(second).parent)
        self.assertIn("open_field", Path(first).name)

    def test_quick_analysis_preserves_root_and_uses_run_directory(self):
        with TemporaryDirectory() as temp_dir:
            sess = {"experiment_type": "tst", "output_dir": temp_dir}
            run_dir = prepare_session_run_output(
                sess,
                session_id="c" * 32,
                default_output_root=temp_dir,
                resolve_output_dir=lambda value: str(Path(value).resolve()),
            )

            self.assertEqual(sess["output_root_dir"], str(Path(temp_dir).resolve()))
            self.assertEqual(sess["run_output_dir"], run_dir)
            self.assertEqual(sess["output_dir"], run_dir)
            self.assertEqual(Path(run_dir).parent, Path(temp_dir).resolve())

    def test_project_run_directory_is_not_nested_again(self):
        with TemporaryDirectory() as temp_dir:
            reserved = str(Path(temp_dir) / "results" / "20260710_tst_deadbeef")
            sess = {
                "experiment_type": "tst",
                "output_dir": reserved,
                "run_output_dir": reserved,
                "project_context": {"project_id": "p1"},
            }
            result = prepare_session_run_output(
                sess,
                session_id="d" * 32,
                default_output_root=temp_dir,
                resolve_output_dir=lambda value: str(Path(value).resolve()),
            )

        self.assertEqual(result, reserved)
        self.assertEqual(sess["output_dir"], reserved)

    def test_retry_uses_a_new_directory_below_the_same_root(self):
        with TemporaryDirectory() as temp_dir:
            sess = {"experiment_type": "tst", "output_dir": temp_dir}
            first = prepare_session_run_output(
                sess,
                session_id="e" * 32,
                default_output_root=temp_dir,
                resolve_output_dir=lambda value: str(Path(value).resolve()),
            )
            second = prepare_session_run_output(
                sess,
                session_id="e" * 32,
                default_output_root=temp_dir,
                resolve_output_dir=lambda value: str(Path(value).resolve()),
                force_new=True,
            )

        self.assertNotEqual(first, second)
        self.assertEqual(Path(first).parent, Path(second).parent)
        self.assertTrue(Path(second).name.endswith("_retry2"))


if __name__ == "__main__":
    unittest.main()
