from __future__ import annotations

import inspect

from services.novel_object_session import NovelObjectMultiArenaAnalysisSession
from services.open_field_session import OpenFieldMultiArenaAnalysisSession


def test_nor_annotated_render_uses_pose_cache_without_dlc_coordinator() -> None:
    source = inspect.getsource(
        NovelObjectMultiArenaAnalysisSession._render_annotated_video
    )
    assert "pose_cache.get" in source
    assert "PoseInferenceCoordinator(" not in source
    assert ".infer(" not in source
    assert ".infer_many(" not in source


def test_oft_annotated_render_uses_pose_cache_without_dlc_coordinator() -> None:
    source = inspect.getsource(
        OpenFieldMultiArenaAnalysisSession._render_annotated_video
    )
    assert "pose_cache.get" in source
    assert "PoseInferenceCoordinator(" not in source
    assert ".infer(" not in source
    assert ".infer_many(" not in source
