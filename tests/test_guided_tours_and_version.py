from __future__ import annotations

import re
import unittest
from pathlib import Path

from services.project_store import PROJECT_SCHEMA_VERSION
from services.version import APP_VERSION, APP_VERSION_TAG


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "templates"
I18N_PATH = ROOT / "static" / "js" / "i18n.js"
TOUR_PATH = ROOT / "static" / "js" / "tour.js"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class GuidedTourConsistencyTest(unittest.TestCase):
    def test_every_tour_translation_key_exists_in_both_languages(self):
        i18n = read(I18N_PATH)
        zh_source, en_source = i18n.split("    en: {", 1)
        sources = [*TEMPLATES.glob("*.html"), ROOT / "static" / "js" / "project_home.js", TOUR_PATH]
        keys: set[str] = set()
        for source in sources:
            keys.update(
                re.findall(
                    r"(?:titleKey|bodyKey)\s*:\s*['\"]([^'\"]+)",
                    read(source),
                )
            )

        self.assertTrue(keys)
        for key in sorted(keys):
            pattern = rf"['\"]{re.escape(key)}['\"]\s*:"
            self.assertRegex(zh_source, pattern, f"missing Chinese tour text: {key}")
            self.assertRegex(en_source, pattern, f"missing English tour text: {key}")

    def test_data_tour_targets_exist_in_their_page_markup(self):
        base = read(TEMPLATES / "base.html")
        project_home = read(TEMPLATES / "project_home.html")
        checks = [
            (TEMPLATES / "home.html", base),
            (TEMPLATES / "projects.html", base),
            (TEMPLATES / "tasks.html", base),
            (TEMPLATES / "upload.html", base),
            (TEMPLATES / "batch_workspace.html", base),
            (TEMPLATES / "visual_style.html", base),
            (TEMPLATES / "experiment_defaults.html", base),
            (ROOT / "static" / "js" / "project_home.js", base + project_home),
            (TEMPLATES / "calibrate_water_maze.html", base),
        ]
        for source_path, included_markup in checks:
            source = read(source_path)
            markup = source + included_markup
            selectors = re.findall(r"target\s*:\s*(['\"])(.*?)\1", source)
            for _quote, selector in selectors:
                for tour_id in re.findall(r"data-tour-id=['\"]([^'\"]+)['\"]", selector):
                    self.assertRegex(
                        markup,
                        rf"data-tour-id=['\"]{re.escape(tour_id)}['\"]",
                        f"{source_path.name} targets missing data-tour-id={tour_id}",
                    )

    def test_declared_tour_pages_have_matching_registration(self):
        pages = {
            "home": (TEMPLATES / "home.html", TEMPLATES / "home.html"),
            "projects": (TEMPLATES / "projects.html", TEMPLATES / "projects.html"),
            "project_home": (TEMPLATES / "project_home.html", ROOT / "static" / "js" / "project_home.js"),
            "tasks": (TEMPLATES / "tasks.html", TEMPLATES / "tasks.html"),
            "upload": (TEMPLATES / "upload.html", TEMPLATES / "upload.html"),
            "analysis-config": (TEMPLATES / "batch_workspace.html", TEMPLATES / "batch_workspace.html"),
            "visual-style": (TEMPLATES / "visual_style.html", TEMPLATES / "visual_style.html"),
            "experiment-defaults": (TEMPLATES / "experiment_defaults.html", TEMPLATES / "experiment_defaults.html"),
            "calibrate-water-maze": (TEMPLATES / "calibrate_water_maze.html", TEMPLATES / "calibrate_water_maze.html"),
            "monitor": (TEMPLATES / "monitor_water_maze.html", TOUR_PATH),
        }
        for page_id, (markup_path, registration_path) in pages.items():
            self.assertIn(f'data-tour-page="{page_id}"', read(markup_path))
            self.assertRegex(
                read(registration_path),
                rf"register\(['\"]{re.escape(page_id)}['\"]",
                f"tour page {page_id} has no matching registration",
            )

    def test_global_settings_guide_no_longer_uses_old_image_style_nav_identity(self):
        sources = read(TEMPLATES / "base.html") + read(TEMPLATES / "home.html") + read(I18N_PATH)
        self.assertNotIn("nav-visual-style", sources)
        self.assertNotIn("tour.home.visual_style", sources)
        self.assertIn("nav-global-settings", sources)
        self.assertIn("tour.home.global_settings", sources)


class VersionConsistencyTest(unittest.TestCase):
    def test_current_version_is_3_5_0(self):
        self.assertEqual(APP_VERSION, "3.5.0")
        self.assertEqual(APP_VERSION_TAG, "v3.5.0")
        self.assertEqual(PROJECT_SCHEMA_VERSION, APP_VERSION)

    def test_current_product_files_do_not_contain_superseded_version_labels(self):
        current_files = [
            ROOT / "README.md",
            ROOT / "CITATION.cff",
            ROOT / "docs" / "system_design.md",
            ROOT / "templates" / "base.html",
            ROOT / "templates" / "home.html",
            ROOT / "templates" / "project_home.html",
            ROOT / "static" / "js" / "i18n.js",
            ROOT / "services" / "__init__.py",
            ROOT / "tools" / "smoke_routes_v2.py",
            ROOT / "tools" / "smoke_short_analyses_v2.py",
        ]
        superseded = re.compile(r"(?:v|Version\s+)(?:3\.3\.0|3\.4\.0|4\.5\.0)")
        for path in current_files:
            self.assertIsNone(superseded.search(read(path)), f"superseded version in {path}")


if __name__ == "__main__":
    unittest.main()
