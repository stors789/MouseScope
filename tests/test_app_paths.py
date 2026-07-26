from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from services.app_paths import resolve_application_data_directory


class ApplicationDataPathTest(unittest.TestCase):
    def test_explicit_data_directory_override_is_used(self):
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / 'custom-data'
            resolved = resolve_application_data_directory(
                Path(temp_dir) / 'app',
                frozen=True,
                environ={'MOUSESCOPE_DATA_DIR': str(target)},
            )
            self.assertEqual(resolved, target)
            self.assertTrue(target.is_dir())

    def test_packaged_app_falls_back_when_program_directory_is_not_writable(self):
        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / 'Program Files' / 'MouseScope'
            fallback = Path(temp_dir) / 'LocalAppData' / 'MouseScope' / 'data'

            def ensure(path):
                if path == base / 'data':
                    raise PermissionError('access denied')
                path.mkdir(parents=True, exist_ok=True)
                return path

            with patch('services.app_paths._ensure_writable_directory', side_effect=ensure):
                resolved = resolve_application_data_directory(
                    base,
                    frozen=True,
                    environ={'LOCALAPPDATA': str(Path(temp_dir) / 'LocalAppData')},
                    platform_name='win32',
                )

            self.assertEqual(resolved, fallback)
            self.assertTrue(fallback.is_dir())

    def test_macos_bundle_uses_application_support_and_migrates_legacy_data(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / 'MouseScope.app' / 'Contents' / 'MacOS'
            legacy = base / 'data'
            legacy.mkdir(parents=True)
            (legacy / 'history.json').write_text('legacy', encoding='utf-8')

            resolved = resolve_application_data_directory(
                base,
                frozen=True,
                environ={'HOME': str(root / 'home')},
                platform_name='darwin',
            )

            expected = root / 'home' / 'Library' / 'Application Support' / 'MouseScope' / 'data'
            self.assertEqual(resolved, expected)
            self.assertEqual((resolved / 'history.json').read_text(encoding='utf-8'), 'legacy')
            self.assertEqual((legacy / 'history.json').read_text(encoding='utf-8'), 'legacy')


if __name__ == '__main__':
    unittest.main()
