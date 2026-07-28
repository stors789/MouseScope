# DLC Batch Inference and In-Memory Pose Cache

## Goal

Improve OFT/NOR DeepLabCut throughput by sending multiple frames to the GPU per
request, while ensuring NOR annotated-video rendering reuses the first scoring
pass instead of running DeepLabCut a second time.

## Scope

- Add ordered multi-frame inference to the bundled DLC worker protocol.
- Add configurable DLC batch size to OFT/NOR pose controls and experiment
  defaults.
- Use bounded frame prefetching so video decoding can overlap GPU inference.
- Keep per-frame NOR pose results in memory until annotated-video rendering
  completes.
- Preserve the existing single-frame protocol and classical-vision fallback.

Persistent caches and interrupted-job resume are out of scope.

## Configuration

Expose `pose_batch_size` in OFT/NOR DeepLabCut settings.

- Valid range: 1 through 16.
- Default: 4.
- Recommended for the target RTX 4060 8 GB system: 4.
- The requested and effective batch sizes are logged and written to result
  summaries.

The pipeline prefetch capacity is two batches. Capacity scales with the
effective batch size and remains bounded.

## Worker Protocol

Keep the current single-frame `preview` and `infer` operations unchanged. Add
an `infer_batch` operation with:

- an ordered `image_paths` array;
- the existing model reference, keypoint schema, and maximum-individual count;
- one ordered result item for every input image;
- the existing request ID for persistent-worker response matching.

The official detector and pose runners are created with the effective batch
size. The detector receives all image paths in the micro-batch, and the pose
runner receives the corresponding detector outputs. The final partial batch is
valid and preserves input order.

## Runtime and Backpressure

`DeepLabCutRuntime` gains a multi-frame method while retaining `infer_frame`.
`PoseInferenceCoordinator` gains `infer_many` and performs ROI assignment per
returned frame in source order.

For formal OFT/NOR analysis:

1. A bounded producer reads, orients, samples, and resizes frames.
2. The consumer sends one micro-batch to the persistent DLC worker.
3. Returned poses are applied to trackers sequentially in source-frame order.
4. At most two prepared batches may wait, preventing unbounded RAM or temporary
   image growth.

Temporary batch images are removed after each worker response.

## Adaptive OOM Handling

If CUDA reports out-of-memory while creating or running a batch, the worker and
runtime reduce the effective batch size to the largest lower step in
`16, 8, 4, 2, 1`, clear failed-batch resources, and retry that batch once at
each lower step. Batch size 1 retains the existing per-frame fallback behavior.

Non-OOM errors keep the existing circuit breaker and classical fallback. A
failed batch must not reorder frames or abort unrelated videos.

## NOR Pose Cache

During the scoring pass, store each sampled source frame number and its
`PoseFrameResult` in a video-local in-memory cache. The cache contains numeric
pose observations and errors only; it never stores decoded video frames.

The annotated-video pass:

- does not create a DeepLabCut coordinator;
- looks up the cached result by sampled source frame number;
- runs the existing tracker and drawing path with that cached observation;
- uses the same classical fallback when the cached pose is absent or failed.

The cache is released immediately after rendering, when rendering is disabled,
or when the video is stopped or fails. Each concurrently analyzed video owns
its own cache.

## Compatibility

- Classical tracking behavior is unchanged.
- Single-frame previews remain single-frame and keep their current API.
- Existing model references remain valid.
- Batch size 1 is behaviorally equivalent to the current persistent worker.
- Result ordering, source-frame numbers, timestamps, and ROI assignment remain
  deterministic.

## Testing

Add tests for:

- ordered worker batch responses and final partial batches;
- predictor runner reuse with configurable batch size;
- runtime request/response matching for batch operations;
- per-frame ROI assignment from `infer_many`;
- bounded pipeline order and cancellation;
- NOR cache hits and misses;
- annotated rendering making zero DLC calls when cache entries exist;
- classical fallback from cached failures;
- CUDA OOM downgrade and retry;
- validation and persistence of batch-size settings;
- single-frame protocol regression coverage.

Run the existing pose, NOR, API, and desktop test suites. On the target RTX
4060, benchmark batch sizes 1, 2, 4, 8, and 16 using representative NOR frames.
The selected default must complete without OOM and improve aggregate frame
throughput over batch size 1.

## Acceptance Criteria

- NOR annotated-video rendering performs no repeated DLC inference.
- User-selected batch size is visible in logs and result summaries.
- GPU batch inference preserves frame and ROI ordering.
- Memory remains bounded to the configured cache plus two prepared batches.
- OOM automatically reduces batch size without losing or duplicating frames.
- Existing single-frame preview and classical tracking tests continue to pass.
