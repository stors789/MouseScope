from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.browser_video import ensure_browser_video


def test_mp4_is_served_without_transcoding(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"mp4")

    resolved, is_proxy = ensure_browser_video(source, tmp_path / "cache")

    assert resolved == source
    assert is_proxy is False


def test_legacy_video_is_transcoded_once_and_cached(tmp_path: Path) -> None:
    source = tmp_path / "source.wmv"
    source.write_bytes(b"wmv")
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"browser mp4")

        class Result:
            returncode = 0
            stderr = b""

        return Result()

    with patch("services.browser_video.find_ffmpeg_exe", return_value="/fake/ffmpeg"), patch(
        "services.browser_video.run_tracked_ffmpeg",
        side_effect=fake_run,
    ):
        first, first_is_proxy = ensure_browser_video(source, tmp_path / "cache")
        second, second_is_proxy = ensure_browser_video(source, tmp_path / "cache")

    assert first == second
    assert first.read_bytes() == b"browser mp4"
    assert first_is_proxy is second_is_proxy is True
    assert len(calls) == 1
    assert "libx264" in calls[0]
    assert "yuv420p" in calls[0]
