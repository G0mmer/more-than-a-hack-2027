"""C2 synthetic-stage velocity estimators at 25 Hz.

Both return (vx, vy, valid), each shaped like points.shape[:-1].
Inputs are (frames, 2) or (frames, players, 2), in metres.
Optional integer track_ids have shape points.shape[:-1]; negative means unknown.
Without IDs, each column must already be one stable track. Call separately per
half. Invalid estimates are NaN, and all input arrays are treated as read-only.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from possession_lab.kinematics import velocity as original_velocity

FPS = 25.0
WINDOW = 16


def _inputs(points, track_ids, fps, maximum_speed):
    values = np.asarray(points, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[-1] != 2:
        raise ValueError("Expected (frames, 2) or (frames, players, 2)")
    if fps != FPS:
        raise ValueError("This C2 comparison requires 25 Hz")
    if not np.isfinite(maximum_speed) or maximum_speed <= 0:
        raise ValueError("maximum_speed must be positive and finite")
    ids = np.zeros(values.shape[:-1], dtype=np.int64) if track_ids is None else np.asarray(track_ids)
    if ids.shape != values.shape[:-1] or ids.dtype.kind not in "iu":
        raise ValueError("track_ids must be integer IDs with the observation shape")
    return values, ids


def _same_track(ids, window):
    windows = sliding_window_view(ids, window, axis=0)
    return (windows[..., -1] >= 0) & np.all(windows == windows[..., -1:], axis=-1)


def local_linear_velocity(points, *, track_ids=None, fps=FPS, maximum_speed=60.0):
    """Causal OLS slope over [t-15, t], spanning exactly 0.6 s.

    All 16 observations must have finite XY and one known, unchanged track ID.
    The first 15 frames, missing windows, and cross-track windows are invalid.
    No interpolation or short-window fallback is used. A finite slope exceeding
    maximum_speed is rejected using the same norm criterion as the baseline.
    """
    values, ids = _inputs(points, track_ids, fps, maximum_speed)
    output = np.full_like(values, np.nan)
    valid = np.zeros(values.shape[:-1], dtype=bool)
    if len(values) < WINDOW:
        return output[..., 0], output[..., 1], valid

    windows = sliding_window_view(values, WINDOW, axis=0)
    finite = np.isfinite(windows)
    usable = finite.all(axis=(-2, -1)) & _same_track(ids, WINDOW)
    times = (np.arange(WINDOW, dtype=np.float64) - (WINDOW - 1) / 2) / fps
    safe = np.where(finite, windows, 0.0)
    # Subtract a reference position to avoid cancellation from large offsets.
    slope = ((safe - safe[..., :1]) @ times) / np.dot(times, times)
    usable &= np.isfinite(slope).all(axis=-1)
    usable &= np.linalg.norm(slope, axis=-1) <= maximum_speed
    output[WINDOW - 1:] = np.where(usable[..., None], slope, np.nan)
    valid[WINDOW - 1:] = usable
    return output[..., 0], output[..., 1], valid


def five_frame_velocity(points, *, track_ids=None, fps=FPS, maximum_speed=60.0):
    """Call the unchanged team estimator, with a shared track-boundary guard.

    The original function assumes stable tracks and does not accept IDs.
    Its six-observation window must therefore additionally have the same ID.
    On a stable known track, results equal the original implementation.
    """
    values, ids = _inputs(points, track_ids, fps, maximum_speed)
    output = original_velocity(values, fps=fps, lag=5, maximum_speed=maximum_speed)
    if len(values) >= 6:
        output[5:] = np.where(_same_track(ids, 6)[..., None], output[5:], np.nan)
    return output[..., 0], output[..., 1], np.isfinite(output).all(axis=-1)
