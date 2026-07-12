from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UserFacingWordingTest(unittest.TestCase):
    def _public_copy_sources(self) -> list[Path]:
        return [
            *sorted((ROOT / "templates").glob("*.html")),
            ROOT / "static" / "js" / "i18n.js",
            ROOT / "static" / "js" / "tour.js",
            ROOT / "routes" / "api.py",
            ROOT / "routes" / "pages.py",
            ROOT / "routes" / "project_api.py",
            ROOT / "services" / "experiment_defaults.py",
            ROOT / "services" / "novel_object_session.py",
            ROOT / "services" / "tst_session.py",
        ]

    def test_public_copy_avoids_release_history_language(self):
        forbidden = re.compile(
            r"当前版本|本版本|此版本|原本(?:判断)?逻辑|新算法|旧算法|原始算法|"
            r"现在怎么|现在支持|现在可以|现在已|目前支持|仍然保留|仍保留原有|不再在页面|"
            r"current version|this version|original (?:decision )?logic|new algorithm|old algorithm|"
            r"legacy algorithm|now supports|now acts|how [^\n]{0,40} now|"
            r"what [^\n]{0,40} now really means|no longer considered",
            re.IGNORECASE,
        )
        for path in self._public_copy_sources():
            source = path.read_text(encoding="utf-8")
            match = forbidden.search(source)
            self.assertIsNone(
                match,
                f"release-history wording {match.group(0)!r} remains in {path.relative_to(ROOT)}" if match else "",
            )

    def test_tst_algorithm_names_are_stable_across_user_surfaces(self):
        surfaces = [
            ROOT / "templates" / "calibrate.html",
            ROOT / "templates" / "monitor.html",
            ROOT / "templates" / "manual_tst_content.html",
            ROOT / "static" / "js" / "tour.js",
        ]
        for path in surfaces:
            source = path.read_text(encoding="utf-8")
            self.assertIn("经典行为算法", source, f"missing classic algorithm name in {path.name}")
            self.assertIn("微小挣扎优化算法", source, f"missing optimized algorithm name in {path.name}")


if __name__ == "__main__":
    unittest.main()
