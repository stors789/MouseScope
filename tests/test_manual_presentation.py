from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"


class ManualPresentationTest(unittest.TestCase):
    def test_manual_selector_is_compact_and_not_repeated_as_manual_buttons(self):
        template = (TEMPLATES / "manual.html").read_text(encoding="utf-8")
        self.assertIn('class="manual-switch-heading"', template)
        self.assertIn('data-i18n="manual.switch.hint"', template)
        for redundant_label in (
            "TST 手册",
            "FST 手册",
            "MWM 手册",
            "OFT 手册",
            "NOR 手册",
            "EPM 手册",
            "Y 迷宫手册",
            "LDB 手册",
            "3CST 手册",
        ):
            self.assertNotIn(redundant_label, template)

    def test_assay_hero_titles_do_not_repeat_complete_manual_wording(self):
        redundant = re.compile(r"完整(?:操作)?手册|Complete\s+.*Manual|.*Complete Manual")
        for path in sorted(TEMPLATES.glob("manual_*_content.html")):
            self.assertIsNone(
                redundant.search(path.read_text(encoding="utf-8")),
                f"redundant manual title remains in {path.name}",
            )

    def test_assay_heroes_do_not_render_redundant_tag_chips(self):
        sources = [TEMPLATES / "manual.html", *sorted(TEMPLATES.glob("manual_*_content.html"))]
        for path in sources:
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("manual-chip", source, f"manual chip markup remains in {path.name}")
            self.assertNotIn("heroChips", source, f"manual chip data remains in {path.name}")


if __name__ == "__main__":
    unittest.main()
