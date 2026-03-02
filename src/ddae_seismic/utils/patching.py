from __future__ import annotations

import numpy as np

def extract_windows_1d(x2d: np.ndarray, win: int, stride: int) -> np.ndarray:
    """Extract sliding windows along time for each trace.

    x2d: (nt, ntr)
    returns: (N, win) where windows are concatenated trace-by-trace.
    """
    if x2d.ndim != 2:
        raise ValueError(f"Expected 2D array, got {x2d.shape}")
    nt, ntr = x2d.shape
    if nt < win:
        raise ValueError(f"nt={nt} < win={win}")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    try:
        sw = np.lib.stride_tricks.sliding_window_view(x2d, window_shape=(win,), axis=0)  # (nt-win+1, ntr, win)
        sw = sw[::stride, :, :]  # (n_win, ntr, win)
        n_win = sw.shape[0]
        # reorder to trace-major: (ntr, n_win, win) -> (N, win)
        windows = np.transpose(sw, (1, 0, 2)).reshape(ntr * n_win, win)
        return windows.astype(np.float32, copy=False)
    except Exception:
        # fallback loop
        windows = []
        for tr in range(ntr):
            x = x2d[:, tr]
            for start in range(0, nt - win + 1, stride):
                windows.append(x[start:start+win])
        return np.stack(windows, axis=0).astype(np.float32, copy=False)

def overlap_add_1d(windows: np.ndarray, n: int, win: int, stride: int) -> np.ndarray:
    """Reconstruct 1D signal from overlapping windows using overlap-add with Hann weights."""
    if windows.ndim != 2 or windows.shape[1] != win:
        raise ValueError(f"windows must be (n_win, {win}), got {windows.shape}")

    y = np.zeros((n,), dtype=np.float32)
    wsum = np.zeros((n,), dtype=np.float32)
    hann = np.hanning(win).astype(np.float32)
    if np.all(hann == 0):
        hann = np.ones((win,), dtype=np.float32)

    pos = 0
    for w in windows:
        y[pos:pos+win] += w * hann
        wsum[pos:pos+win] += hann
        pos += stride
        if pos + win > n + stride:  # loose safety
            break
    wsum = np.maximum(wsum, 1e-8)
    return y / wsum
