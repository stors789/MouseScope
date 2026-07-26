from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DesktopPlatformIsolationTest(unittest.TestCase):
    def test_windows_dependencies_remain_free_of_macos_frameworks(self):
        runtime = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        windows_build = (ROOT / "build_exe.bat").read_text(encoding="utf-8").lower()
        for mac_dependency in ("pywebview", "pyobjc", "cocoa", "webkit"):
            self.assertNotIn(mac_dependency, runtime)
            self.assertNotIn(mac_dependency, windows_build)

        mac_requirements = (ROOT / "requirements-macos.txt").read_text(encoding="utf-8").lower()
        self.assertIn("pywebview==", mac_requirements)

    def test_spec_selects_desktop_entry_only_on_macos(self):
        spec = (ROOT / "MouseScope.spec").read_text(encoding="utf-8")
        self.assertIn("ENTRY_SCRIPT = 'desktop.py' if IS_MACOS else 'app.py'", spec)
        self.assertIn("MACOS_HIDDEN_IMPORTS", spec)
        self.assertIn("console=not IS_MACOS", spec)

    def test_importing_desktop_module_does_not_import_webview(self):
        import desktop  # noqa: F401

        self.assertNotIn("webview", sys.modules)


if __name__ == "__main__":
    unittest.main()
