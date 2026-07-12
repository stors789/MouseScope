from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from services.atomic_json import read_json
from services.reproducibility import (
    REPRODUCIBILITY_REPORT_FILENAME,
    RESULT_FINGERPRINT_FILENAME,
    RUN_SNAPSHOT_FILENAME,
    build_result_fingerprint,
    build_run_snapshot,
    compare_result_fingerprints,
    finalize_reproducibility_artifacts,
    load_run_snapshot,
    resolved_config_for_replay,
    reproducibility_status,
    validate_replay_snapshot,
    write_run_snapshot,
)


class _DummyAnalysis:
    def __init__(self, output_dir: Path, result_files: list[str], session_id: str = "candidate"):
        self.status = "completed"
        self.session_id = session_id
        self.config = {"output_dir": str(output_dir)}
        self.result_files = list(result_files)

    def _append_result_file(self, path, group=None):
        if path not in self.result_files:
            self.result_files.append(path)


class ReproducibilityTest(unittest.TestCase):
    def test_snapshot_round_trip_and_replay_config(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "input.mp4"
            video.write_bytes(b"video-state")
            output = root / "results"
            sess = {
                "experiment_type": "open_field",
                "video_paths": [str(video)],
                "video_infos": [{"path": str(video), "original_name": "input.mp4"}],
                "draft_payload": {"rois": [{"name": "Arena 1"}]},
                "output_root_dir": str(root),
                "output_dir": str(output),
            }
            snapshot = build_run_snapshot(
                session_id="run-a",
                app_version="3.5.0",
                sess=sess,
                resolved_config={"output_dir": str(output), "threshold": 127},
            )
            snapshot_path = write_run_snapshot(output, snapshot)
            loaded = load_run_snapshot(snapshot_path)

            self.assertEqual(Path(snapshot_path).name, RUN_SNAPSHOT_FILENAME)
            self.assertEqual(validate_replay_snapshot(loaded, app_version="3.5.0"), [])
            replay_config = resolved_config_for_replay(loaded, output_dir=root / "replay")
            self.assertEqual(replay_config["threshold"], 127)
            self.assertEqual(replay_config["output_dir"], str(root / "replay"))

    def test_replay_validation_detects_changed_input(self):
        with TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "input.mp4"
            video.write_bytes(b"before")
            snapshot = build_run_snapshot(
                session_id="run-a",
                app_version="3.5.0",
                sess={"video_paths": [str(video)]},
                resolved_config={"output_dir": temp_dir},
            )
            video.write_bytes(b"after-with-different-size")

            errors = validate_replay_snapshot(snapshot, app_version="3.5.0")

            self.assertTrue(any("大小已变化" in error for error in errors))

    def test_replay_validation_requires_same_app_version(self):
        with TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "input.mp4"
            video.write_bytes(b"video")
            snapshot = build_run_snapshot(
                session_id="run-a",
                app_version="3.5.0",
                sess={"video_paths": [str(video)]},
                resolved_config={"output_dir": temp_dir},
            )

            errors = validate_replay_snapshot(snapshot, app_version="3.5.1")

            self.assertTrue(any("同一软件版本" in error for error in errors))

    def test_fingerprint_only_tracks_scientific_text_outputs(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "summary.csv"
            txt_path = root / "report.txt"
            video_path = root / "annotated.mp4"
            csv_path.write_text("value\n1\n", encoding="utf-8")
            txt_path.write_text("distance=1", encoding="utf-8")
            video_path.write_bytes(b"encoded-video")

            fingerprint = build_result_fingerprint(
                [str(csv_path), str(txt_path), str(video_path)],
                output_dir=root,
            )

            self.assertEqual([item["path"] for item in fingerprint["files"]], ["report.txt", "summary.csv"])

    def test_completed_replay_writes_consistency_report(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference_dir = root / "reference"
            candidate_dir = root / "candidate"
            reference_dir.mkdir()
            candidate_dir.mkdir()
            reference_csv = reference_dir / "summary.csv"
            candidate_csv = candidate_dir / "summary.csv"
            reference_csv.write_text("value\n1\n", encoding="utf-8")
            candidate_csv.write_text("value\n1\n", encoding="utf-8")
            reference_fingerprint = build_result_fingerprint([reference_csv], output_dir=reference_dir)
            from services.atomic_json import atomic_write_json

            atomic_write_json(reference_dir / RESULT_FINGERPRINT_FILENAME, reference_fingerprint)
            atomic_write_json(reference_dir / RUN_SNAPSHOT_FILENAME, {"snapshot_version": 1})
            analysis = _DummyAnalysis(candidate_dir, [str(candidate_csv)])

            created = finalize_reproducibility_artifacts(
                analysis,
                {"replay_of": {"session_id": "reference", "snapshot_path": str(reference_dir / RUN_SNAPSHOT_FILENAME)}},
            )

            self.assertIn(str(candidate_dir / RESULT_FINGERPRINT_FILENAME), created)
            self.assertIn(str(candidate_dir / REPRODUCIBILITY_REPORT_FILENAME), created)
            report = read_json(candidate_dir / REPRODUCIBILITY_REPORT_FILENAME)
            self.assertEqual(report["status"], "consistent")
            self.assertTrue(report["consistent"])
            self.assertEqual(
                reproducibility_status({"output_dir": str(candidate_dir), "result_files": created}),
                "consistent",
            )

    def test_fingerprint_comparison_identifies_changed_files(self):
        comparison = compare_result_fingerprints(
            {"files": [{"path": "summary.csv", "sha256": "a"}]},
            {"files": [{"path": "summary.csv", "sha256": "b"}]},
        )
        self.assertFalse(comparison["consistent"])
        self.assertEqual(comparison["changed_files"], ["summary.csv"])

    def test_report_analysis_timestamp_is_excluded_from_scientific_hash(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text(
                "分析时间 (Analysis Time): 2026-07-10 12:00:00\nTotal Distance: 42 cm\n",
                encoding="utf-8",
            )
            second.write_text(
                "分析时间 (Analysis Time): 2026-07-10 12:00:01\nTotal Distance: 42 cm\n",
                encoding="utf-8",
            )

            first_fp = build_result_fingerprint([first], output_dir=root)
            second_fp = build_result_fingerprint([second], output_dir=root)

            self.assertEqual(first_fp["files"][0]["sha256"], second_fp["files"][0]["sha256"])


if __name__ == "__main__":
    unittest.main()
