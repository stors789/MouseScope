from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from flask import Flask

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / 'core'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from core.y_maze import (
    YMazeTracker,
    build_manual_y_maze_summary,
    normalize_manual_arm_entries,
)
from services.debug_preview import _y_maze_debug_preview
from services.y_maze_session import YMazeMultiArenaAnalysisSession


ZONES = {
    'center': [{'x': 90, 'y': 90}, {'x': 110, 'y': 90}, {'x': 100, 'y': 110}],
    'arm_a': [{'x': 90, 'y': 90}, {'x': 110, 'y': 90}, {'x': 110, 'y': 10}, {'x': 90, 'y': 10}],
    'arm_b': [{'x': 110, 'y': 90}, {'x': 200, 'y': 90}, {'x': 200, 'y': 110}, {'x': 110, 'y': 110}],
    'arm_c': [{'x': 0, 'y': 90}, {'x': 90, 'y': 90}, {'x': 90, 'y': 110}, {'x': 0, 'y': 110}],
}


def roi(name: str, mode: str = 'automatic', events=None):
    payload = {
        'name': name,
        'zones': ZONES,
        'time_config': {'mode': 'global', 'start': None, 'end': None},
        'scoring_mode': mode,
    }
    if events is not None:
        payload['manual_events'] = events
    return payload


def session(config=None):
    return YMazeMultiArenaAnalysisSession('session', [], [], config or {'rois': []})


def test_manual_abc_sequence_summary_and_event_rows() -> None:
    summary, rows = build_manual_y_maze_summary(
        roi_id=2,
        roi_name='Right',
        events=[
            {'timestamp_s': 1.0, 'arm': 'A'},
            {'timestamp_s': 2.0, 'arm': 'B'},
            {'timestamp_s': 3.0, 'arm': 'C'},
        ],
        window_start_s=0.0,
        window_end_s=10.0,
    )

    assert summary['scoring_method'] == 'manual'
    assert summary['entry_sequence'] == 'A-B-C'
    assert summary['arm_a_entries'] == 1
    assert summary['arm_b_entries'] == 1
    assert summary['arm_c_entries'] == 1
    assert summary['total_arm_entries'] == 3
    assert summary['alternation_count'] == 1
    assert summary['alternation_percent'] == 100.0
    assert [row['arm'] for row in rows] == ['A', 'B', 'C']
    assert all(row['scoring_method'] == 'manual' for row in rows)


def test_undo_is_recomputed_from_remaining_sequence() -> None:
    events = [
        {'timestamp_s': 1.0, 'arm': 'A'},
        {'timestamp_s': 2.0, 'arm': 'B'},
        {'timestamp_s': 3.0, 'arm': 'C'},
    ]
    events.pop()
    summary, _ = build_manual_y_maze_summary(
        roi_id=1,
        roi_name='Mouse',
        events=events,
        window_start_s=0.0,
        window_end_s=5.0,
    )
    assert summary['entry_sequence'] == 'A-B'
    assert summary['total_arm_entries'] == 2
    assert summary['alternation_opportunities'] == 0
    assert summary['alternation_percent'] is None


def test_manual_window_filter_and_strict_timestamp_validation() -> None:
    summary, rows = build_manual_y_maze_summary(
        roi_id=1,
        roi_name='Mouse',
        events=[
            {'timestamp_s': 1.0, 'arm': 'A'},
            {'timestamp_s': 5.0, 'arm': 'B'},
            {'timestamp_s': 11.0, 'arm': 'C'},
        ],
        window_start_s=2.0,
        window_end_s=10.0,
    )
    assert [row['arm'] for row in rows] == ['B']
    assert summary['manual_events_excluded'] == 2

    with pytest.raises(ValueError, match='strictly increasing'):
        normalize_manual_arm_entries([
            {'timestamp_s': 2.0, 'arm': 'A'},
            {'timestamp_s': 2.0, 'arm': 'B'},
        ])


def test_old_roi_defaults_to_automatic_and_mixed_session_skips_manual_tracker() -> None:
    mixed = session()
    parsed = mixed._parse_rois({
        'rois': [
            roi('Left'),
            roi('Right', 'manual', [{'timestamp_s': 1.0, 'arm': 'A'}]),
        ],
    })
    assert [item['scoring_mode'] for item in parsed] == ['automatic', 'manual']

    trackers = mixed._create_trackers(parsed, {}, 30.0, 0.0)
    assert len(trackers) == 1
    assert trackers[0].roi_name == 'Left'
    assert trackers[0].build_summary()['scoring_method'] == 'automatic'


def test_batch_manual_events_are_isolated_by_video_config() -> None:
    analyzer = session()
    first = analyzer._parse_rois({
        'rois': [roi('Right', 'manual', [{'timestamp_s': 1.0, 'arm': 'A'}])],
    })
    second = analyzer._parse_rois({
        'rois': [roi('Right', 'manual', [{'timestamp_s': 2.0, 'arm': 'C'}])],
    })
    first[0]['manual_events'].append({'timestamp_s': 3.0, 'arm': 'B'})

    assert second[0]['manual_events'] == [{'timestamp_s': 2.0, 'arm': 'C'}]


def test_manual_preview_does_not_create_automatic_tracker() -> None:
    app = Flask(__name__)
    frame = np.full((120, 220, 3), 100, dtype=np.uint8)
    background = type(
        'Background',
        (),
        {'method': 'threshold', 'apply': lambda self, frame, gray=None: np.zeros(gray.shape, dtype=np.uint8)},
    )()
    data = {
        'rois': [roi('Right', 'manual', [{'timestamp_s': 1.0, 'arm': 'A'}])],
        'bg_method': 'threshold',
    }
    with app.test_request_context(), patch(
        'services.debug_preview._build_debug_bg_subtractor',
        return_value=background,
    ), patch('services.debug_preview.YMazeTracker') as tracker:
        response = _y_maze_debug_preview(frame, data, 'video.mp4')

    assert response.status_code == 200
    assert response.get_json()['roi_count'] == 1
    tracker.assert_not_called()


def test_manual_workbench_persists_per_roi_events_and_shortcuts() -> None:
    template = (ROOT / 'templates' / 'calibrate_maze.html').read_text(encoding='utf-8')

    assert "roi.scoring_mode==='manual'?'manual':'automatic'" in template
    assert 'payload.manual_events=cloneState(roi.manual_events||[])' in template
    assert 'workspace_video_configs' in template
    assert "recordManualArm(`arm_${key}`)" in template
    assert "event.code==='Space'" in template
    assert "pageFormat('alert.manual_empty'" in template
    assert 'return fetch(`/api/get_first_frame/' in template
    assert 'await loadManualFrame(next)' in template
    assert 'scheduleManualPlaybackStep(generation)' in template
    assert 'setInterval(' not in template


def test_manual_workbench_is_not_rendered_for_epm() -> None:
    template = (ROOT / 'templates' / 'calibrate_maze.html').read_text(encoding='utf-8')
    workbench = template.index('id="manualWorkbench"')
    guard = template.rfind('{% if not is_epm %}', 0, workbench)
    closing = template.find('{% endif %}', workbench)

    assert guard >= 0
    assert closing > workbench
