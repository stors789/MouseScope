# DeepLabCut Head-Direction Stabilization

## Goal

Reduce one- to three-frame DeepLabCut head-direction dropouts without lowering
the pose keypoint confidence threshold, while preserving strict DeepLabCut
tracking semantics and reproducible NOR scoring.

## Configuration

Add two NOR parameters to defaults, parameter metadata, calibration payloads,
saved configurations, and run snapshots:

- `pose_head_hold_s`, default `0.12`, range `0.0` to `1.0` seconds.
- `pose_head_smoothing`, default `0.35`, range `0.0` to `1.0`. This is the
  weight assigned to the newest reliable direction observation; `1.0` disables
  smoothing.

## Direction State

Each tracker retains the last reliable DeepLabCut direction, its confidence,
and the elapsed time since it was observed.

When a reliable observation arrives, normalize its direction and blend it with
the previous direction using `pose_head_smoothing`. Re-normalize the blended
vector before using it.

When the nose is temporarily unavailable but the current body center remains
available, retain the smoothed direction for at most `pose_head_hold_s`.
Reconstruct a display head point from the current body origin, retained unit
direction, and the most recent reliable head-vector length. Mark these frames
as held rather than observed.

Held frames may continue exploration scoring because this is the behavior the
hold is intended to stabilize. Object proximity and alignment are recomputed
from the reconstructed current head point; a stale absolute nose position must
not be reused.

After the hold expires, or before any reliable direction has been observed,
there is no valid head direction.

## Strict And Auto Modes

With explicit `tracking_backend=deeplabcut` and head-direction scoring enabled,
only observed or held DeepLabCut head directions may contribute exploration
time. Missing head direction must never fall back to body-center proximity.

`tracking_backend=auto` retains its existing classical and proximity fallback
behavior. Strict DeepLabCut must not construct or invoke grayscale,
background-subtraction, contour, or PCA tracking components.

## Output

Observed frames keep the existing `deeplabcut_ears` or `deeplabcut_body`
method and `head` exploration mode. Held frames use explicit values such as
`deeplabcut_held` and `head_held`, so CSV consumers can distinguish measured
and retained direction.

The annotated arrow uses the same stabilized direction state as scoring.
Analysis stores pose inference in the existing in-memory cache; rendering does
not invoke DeepLabCut again.

## Verification

Tests cover reliable direction smoothing, a short held gap, hold expiration,
strict-mode suppression of proximity fallback, preservation of auto fallback,
configuration defaults and payload serialization, and CSV method/mode labels.
