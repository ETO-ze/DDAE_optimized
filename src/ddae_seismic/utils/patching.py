from __future__ import annotations

import numpy as np


def window_starts(n: int, win: int, stride: int) -> list[int]:
    """Return shared extraction/reconstruction positions, including the tail.

    A short signal gets one edge-padded window. Strides larger than the window
    are rejected because they leave samples without a prediction.
    """
    if n <= 0 or win <= 0 or stride <= 0:
        raise ValueError("n, win and stride must be > 0")
    if stride > win:
        raise ValueError("stride must be <= win to cover every sample")
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, stride))
    if starts[-1] != n - win:
        starts.append(n - win)
    return starts


def extract_windows_1d(x2d: np.ndarray, win: int, stride: int) -> np.ndarray:
    """Extract time windows in trace-major order, padding short traces at the end."""
    if x2d.ndim != 2 or x2d.shape[1] == 0:
        raise ValueError(f"Expected nonempty 2D array, got {x2d.shape}")
    nt, ntr = x2d.shape
    starts = window_starts(nt, win, stride)
    padded = np.pad(x2d, ((0, max(0, win - nt)), (0, 0)), mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(padded, win, axis=0)
    windows = sw[starts].transpose(1, 0, 2).reshape(ntr * len(starts), win)
    return windows.astype(np.float32, copy=False)


def overlap_add_1d(windows: np.ndarray, n: int, win: int, stride: int) -> np.ndarray:
    """Reconstruct a signal with positive Hann weights at every sample.

    The interior of a length-(win + 2) Hann window retains the taper without
    zero endpoints, including for non-overlapping windows. Padding is cropped.
    """
    starts = window_starts(n, win, stride)
    if windows.shape != (len(starts), win):
        raise ValueError(f"Expected windows shape {(len(starts), win)}, got {windows.shape}")
    y = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    weights = np.hanning(win + 2)[1:-1]
    for start, window in zip(starts, windows):
        length = min(win, n - start)
        y[start:start + length] += window[:length] * weights[:length]
        wsum[start:start + length] += weights[:length]
    return (y / wsum).astype(np.float32)
