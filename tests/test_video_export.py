from __future__ import annotations

import io
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from services.video_export import FfmpegVideoWriter


class _FakeFfmpegProcess:
    def __init__(self):
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO(b"")

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        return None


class VideoExportResourceTest(unittest.TestCase):
    def test_ffmpeg_writer_closes_both_process_pipes(self):
        process = _FakeFfmpegProcess()
        with TemporaryDirectory() as temp_dir, patch(
            "services.video_export.subprocess.Popen", return_value=process
        ):
            writer = FfmpegVideoWriter(
                Path(temp_dir) / "result.mp4",
                30,
                64,
                48,
                ffmpeg_exe="ffmpeg",
                label="test",
            )
            writer.release()

        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stderr.closed)


if __name__ == "__main__":
    unittest.main()
