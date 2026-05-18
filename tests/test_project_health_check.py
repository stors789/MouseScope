from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import project_health_check


class ProjectHealthCheckTest(unittest.TestCase):
    def test_gitignore_requires_user_data_and_cache_patterns(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".gitignore").write_text("data/\n*.db\n", encoding="utf-8")

            with patch.object(project_health_check, "ROOT", root):
                findings = project_health_check.check_gitignore()

        messages = [finding.message for finding in findings]
        self.assertTrue(any("*.sqlite" in message for message in messages))
        self.assertTrue(any(".pytest_cache/" in message for message in messages))

    def test_stale_runtime_reference_scans_runtime_files_but_allows_architecture_note(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "core").mkdir()
            (root / "docs").mkdir()
            (root / "core" / "bad.py").write_text("OLD = 'core/tst_v2.py'\n", encoding="utf-8")
            (root / "docs" / "system_design.md").write_text("current runtime has no core/tst_v2.py\n", encoding="utf-8")

            with (
                patch.object(project_health_check, "ROOT", root),
                patch.object(project_health_check, "SCAN_ROOTS", ("core", "docs")),
                patch.object(project_health_check, "ALLOWED_STALE_REFERENCE_FILES", {Path("docs/system_design.md")}),
            ):
                findings = project_health_check.check_stale_runtime_references()

        self.assertEqual(len(findings), 1)
        self.assertIn("core", findings[0].message)
        self.assertIn("tst_v2", findings[0].message)

    def test_missing_required_files_are_reported_as_errors(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(project_health_check, "ROOT", root):
                findings = project_health_check.check_required_files()

        self.assertTrue(findings)
        self.assertTrue(all(finding.severity == "ERROR" for finding in findings))


if __name__ == "__main__":
    unittest.main()
