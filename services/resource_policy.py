from __future__ import annotations

import os
from dataclasses import asdict, dataclass


def _positive_env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 10_000) -> int:
    try:
        value = int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, min(maximum, value))


def default_parallel_video_limit(cpu_count: int | None = None) -> int:
    cpu = max(1, int(cpu_count or os.cpu_count() or 1))
    calculated = 1 if cpu <= 4 else 2 if cpu <= 8 else min(4, max(2, cpu // 4))
    return _positive_env_int("MOUSESCOPE_MAX_CONCURRENT_VIDEOS", calculated, maximum=16)


def opencv_thread_limit(cpu_count: int | None = None, parallel_limit: int | None = None) -> int:
    cpu = max(1, int(cpu_count or os.cpu_count() or 1))
    workers = max(1, int(parallel_limit or default_parallel_video_limit(cpu)))
    calculated = max(1, min(4, max(1, cpu - 1) // workers))
    return _positive_env_int("MOUSESCOPE_OPENCV_THREADS", calculated, maximum=cpu)


def batch_worker_count(total_videos: int, requested: int | str | None = None, cpu_count: int | None = None) -> int:
    total = max(1, int(total_videos or 1))
    process_limit = default_parallel_video_limit(cpu_count)
    if requested is None or str(requested).strip() == "":
        desired = process_limit
    else:
        try:
            desired = int(requested)
        except (TypeError, ValueError):
            desired = 1
    return max(1, min(total, process_limit, desired))


def video_queue_capacity(
    width: int,
    height: int,
    *,
    requested: int = 120,
    target_megabytes: int | None = None,
) -> int:
    """Limit queued BGR frames to a per-writer memory budget."""

    frame_bytes = max(1, int(width or 1)) * max(1, int(height or 1)) * 3
    target_mb = target_megabytes or _positive_env_int("MOUSESCOPE_VIDEO_QUEUE_MB", 96, minimum=16, maximum=512)
    memory_capacity = max(2, int(target_mb * 1024 * 1024 // frame_bytes))
    return max(2, min(max(2, int(requested or 2)), memory_capacity))


@dataclass(frozen=True)
class ResourcePolicy:
    cpu_count: int
    max_parallel_videos: int
    opencv_threads: int
    video_queue_target_mb: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def current_resource_policy(cpu_count: int | None = None) -> ResourcePolicy:
    cpu = max(1, int(cpu_count or os.cpu_count() or 1))
    parallel = default_parallel_video_limit(cpu)
    return ResourcePolicy(
        cpu_count=cpu,
        max_parallel_videos=parallel,
        opencv_threads=opencv_thread_limit(cpu, parallel),
        video_queue_target_mb=_positive_env_int("MOUSESCOPE_VIDEO_QUEUE_MB", 96, minimum=16, maximum=512),
    )


def resolved_run_resource_policy(
    total_videos: int,
    requested_workers: int | str | None = None,
    *,
    cpu_count: int | None = None,
) -> dict[str, int]:
    policy = current_resource_policy(cpu_count).as_dict()
    policy["batch_workers"] = batch_worker_count(
        total_videos,
        requested=requested_workers,
        cpu_count=cpu_count,
    )
    return policy
