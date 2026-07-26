from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from core.vision_tracker import (
    BackgroundSubtractor,
    MorphologyProcessor,
    MouseDetector,
    create_temporal_background,
    uniform_temporal_sample_indices,
)
from core.y_maze import YMazeTracker
from services.analysis_runtime import FrameCachedBackgroundSubtractor, begin_frame_cache
from services.experiment_defaults import experiment_defaults_schema


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.position = 0
        self.requested_indices: list[int] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        return 0.0

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
            self.requested_indices.append(self.position)
        return True

    def read(self):
        if not (0 <= self.position < len(self.frames)):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def test_uniform_sampling_covers_entire_requested_window_and_orientation() -> None:
    frames = [np.full((12, 16, 3), index, dtype=np.uint8) for index in range(100)]
    capture = FakeCapture(frames)
    with patch('core.vision_tracker.open_video_capture', return_value=capture) as opener:
        background, noise, indices = create_temporal_background(
            'video.mp4',
            num_samples=5,
            start_frame=10,
            end_frame=90,
            orientation_override=90,
            analysis_size=(8, 6),
        )

    assert indices == [10, 30, 50, 70, 90]
    assert capture.requested_indices == indices
    assert background.shape == (6, 8)
    assert int(background[0, 0]) == 50
    assert noise.shape == background.shape
    assert capture.released is True
    opener.assert_called_once_with('video.mp4', orientation_override=90)


def test_fixed_bright_spot_is_background_while_moving_mouse_survives() -> None:
    frames = []
    mouse_positions = [(4 + index * 3, 25) for index in range(9)]
    for mouse_x, mouse_y in mouse_positions:
        gray = np.full((48, 48), 100, dtype=np.uint8)
        cv2.circle(gray, (38, 8), 4, 210, -1)  # fixed reflection
        cv2.circle(gray, (mouse_x, mouse_y), 3, 20, -1)  # moving mouse
        frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    capture = FakeCapture(frames)
    with patch('core.vision_tracker.open_video_capture', return_value=capture):
        background, noise, _ = create_temporal_background('video.mp4', num_samples=9)

    current = cv2.cvtColor(frames[4], cv2.COLOR_BGR2GRAY)
    subtractor = BackgroundSubtractor(
        method='temporal',
        threshold=20,
        static_bg=background,
        temporal_noise_map=noise,
    )
    mask = subtractor.apply(frames[4], gray=current)

    assert mask[8, 38] == 0
    assert mask[25, mouse_positions[4][0]] == 255


def test_temporal_difference_threshold_is_independent_and_effective() -> None:
    background = np.full((10, 10), 100, dtype=np.uint8)
    frame = background.copy()
    frame[4:6, 4:6] = 120
    noise = np.zeros_like(background)

    sensitive = BackgroundSubtractor(
        method='temporal',
        threshold=15,
        static_bg=background,
        temporal_noise_map=noise,
    )
    conservative = BackgroundSubtractor(
        method='temporal',
        threshold=25,
        static_bg=background,
        temporal_noise_map=noise,
    )

    assert sensitive.apply(frame, gray=frame)[4, 4] == 255
    assert conservative.apply(frame, gray=frame)[4, 4] == 0


def _zones(offset_x: int) -> dict[str, list[tuple[int, int]]]:
    return {
        'center': [(offset_x + 40, 40), (offset_x + 60, 40), (offset_x + 50, 60)],
        'arm_a': [(offset_x + 40, 40), (offset_x + 60, 40), (offset_x + 60, 0), (offset_x + 40, 0)],
        'arm_b': [(offset_x + 60, 40), (offset_x + 50, 60), (offset_x + 90, 90), (offset_x + 100, 70)],
        'arm_c': [(offset_x + 50, 60), (offset_x + 40, 40), (offset_x, 70), (offset_x + 10, 90)],
    }


def test_temporal_mask_is_cached_once_for_multiple_y_maze_rois() -> None:
    background = np.full((100, 220), 100, dtype=np.uint8)
    gray = background.copy()
    cv2.circle(gray, (50, 20), 5, 20, -1)
    cv2.circle(gray, (160, 20), 5, 20, -1)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    wrapped = BackgroundSubtractor(
        method='temporal',
        threshold=20,
        static_bg=background,
        temporal_noise_map=np.zeros_like(background),
    )
    call_count = 0
    original_apply = wrapped.apply

    def counted_apply(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_apply(*args, **kwargs)

    wrapped.apply = counted_apply  # type: ignore[method-assign]
    subtractor = FrameCachedBackgroundSubtractor(wrapped)
    begin_frame_cache(subtractor, 7)
    morph = MorphologyProcessor(
        erosion_iterations=0,
        dilation_iterations=0,
        use_closing=False,
    )
    detector = MouseDetector(min_area=20, max_area=500)
    trackers = [
        YMazeTracker(_zones(0), fps=10, zone_confirm_s=0),
        YMazeTracker(_zones(110), fps=10, zone_confirm_s=0),
    ]

    results = [
        tracker.update(frame, 0.0, gray, subtractor, morph, detector, dt_s=0.1)
        for tracker in trackers
    ]

    assert [result['detected'] for result in results] == [True, True]
    assert call_count == 1


def test_old_background_modes_remain_unchanged_and_temporal_is_ymaze_only() -> None:
    gray = np.array([[20, 200]], dtype=np.uint8)
    threshold = BackgroundSubtractor(method='threshold', threshold=100, invert=False)
    assert threshold.apply(gray, gray=gray).tolist() == [[255, 0]]

    static = BackgroundSubtractor(method='static', static_bg=np.array([[20, 200]], dtype=np.uint8))
    assert static.apply(np.array([[60, 200]], dtype=np.uint8)).tolist() == [[255, 0]]

    schema = experiment_defaults_schema()
    ymaze_bg = next(field for field in schema['modules']['y_maze'] if field['key'] == 'bg_method')
    epm_bg = next(field for field in schema['modules']['elevated_plus'] if field['key'] == 'bg_method')
    assert 'temporal' in {option['value'] for option in ymaze_bg['options']}
    assert 'temporal' not in {option['value'] for option in epm_bg['options']}
    assert uniform_temporal_sample_indices(4, num_samples=90) == [0, 1, 2, 3]
