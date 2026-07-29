# Strict DeepLabCut Tracking Semantics

## Problem

MouseScope currently treats every non-classical tracking mode as eligible for
frame-level classical fallback. As a result, selecting `deeplabcut` can still
invoke grayscale segmentation and PCA-based head direction. PCA estimates an
unoriented body axis and can reverse head and tail by 180 degrees.

## Backend Semantics

- `classical`: use only background subtraction, grayscale segmentation, and
  contour-based tracking.
- `deeplabcut`: use only DeepLabCut observations. If an arena has no usable
  pose for a frame, record a missing pose frame and do not run classical
  detection or head-direction estimation.
- `auto`: prefer DeepLabCut and use the existing classical pipeline when a
  usable pose is unavailable.

The semantics apply equally to analysis and annotated-video rendering. Cached
pose results must produce the same backend choice during rendering.

## Outputs

Strict DLC gaps use the tracking source `deeplabcut_missing`. Summaries report
the number of these frames separately. An analysis with successful and missing
strict-DLC frames remains a DeepLabCut analysis rather than being reported as
classical.

Object interaction circles remain unchanged. They define the physical
nose-to-object distance used for exploration scoring and are independent of
the grayscale detector.

## Verification

Tests must prove that:

1. strict DLC does not invoke the classical detector when pose assignment is
   missing;
2. strict DLC does not invoke grayscale/PCA head estimation when pose
   head-point confidence is insufficient;
3. automatic mode still performs classical fallback;
4. summary counters distinguish DLC gaps from classical fallback frames.
