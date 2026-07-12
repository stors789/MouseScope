from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))

from services.resource_policy import (
    batch_worker_count,
    current_resource_policy,
    opencv_thread_limit,
    video_queue_capacity,
    resolved_run_resource_policy,
)
from services.analysis_runtime import AnalysisSessionBase


class _ConcurrentProbeSession(AnalysisSessionBase):
    def __init__(self, count: int):
        self.video_paths = [f"video-{index}.mp4" for index in range(count)]
        self.video_infos = [{"original_name": f"video-{index}.mp4"} for index in range(count)]
        self.config = {"parallel_videos": True, "batch_workers": 99}
        self.is_running = True
        self.status = "analyzing"
        self.progress = 0
        self.message = ""
        self.logs = []
        self.end_time = None
        self._probe_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def _log(self, message, *, force_console=False):
        self.logs.append(str(message))

    def _analyze_single_video(self, _path, _info, batch_index=None):
        with self._probe_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._probe_lock:
            self.active -= 1


class ResourcePolicyTest(unittest.TestCase):
    def test_resolved_run_policy_records_effective_batch_workers(self):
        policy = resolved_run_resource_policy(9, requested_workers=99, cpu_count=8)
        self.assertEqual(policy["max_parallel_videos"], 2)
        self.assertEqual(policy["batch_workers"], 2)

    def test_default_parallelism_does_not_oversubscribe_opencv(self):
        with patch.dict(os.environ, {}, clear=True):
            policy = current_resource_policy(cpu_count=20)

        self.assertEqual(policy.max_parallel_videos, 4)
        self.assertEqual(policy.opencv_threads, 4)
        self.assertLessEqual(policy.max_parallel_videos * policy.opencv_threads, policy.cpu_count)

    def test_requested_batch_workers_are_clamped_to_process_limit(self):
        with patch.dict(os.environ, {"MOUSESCOPE_MAX_CONCURRENT_VIDEOS": "3"}, clear=True):
            self.assertEqual(batch_worker_count(12, requested=10, cpu_count=20), 3)
            self.assertEqual(batch_worker_count(2, requested=10, cpu_count=20), 2)

    def test_queue_capacity_respects_memory_budget(self):
        capacity_1080p = video_queue_capacity(1920, 1080, requested=120, target_megabytes=96)
        capacity_4k = video_queue_capacity(3840, 2160, requested=120, target_megabytes=96)

        self.assertLessEqual(capacity_1080p, 16)
        self.assertLessEqual(capacity_4k, 4)
        self.assertGreaterEqual(capacity_4k, 2)

    def test_opencv_override_is_clamped_to_cpu_count(self):
        with patch.dict(os.environ, {"MOUSESCOPE_OPENCV_THREADS": "99"}, clear=True):
            self.assertEqual(opencv_thread_limit(cpu_count=8, parallel_limit=2), 8)

    def test_batch_runtime_respects_global_video_limit(self):
        session = _ConcurrentProbeSession(8)
        session._run()

        self.assertEqual(session.status, "completed")
        self.assertLessEqual(session.max_active, current_resource_policy().max_parallel_videos)


if __name__ == "__main__":
    unittest.main()
