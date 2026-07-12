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
                )

            self.assertEqual(resolved, fallback)
            self.assertTrue(fallback.is_dir())


if __name__ == '__main__':
    unittest.main()
