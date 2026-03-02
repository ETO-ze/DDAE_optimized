from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import scipy.io as sio

from .utils.patching import extract_windows_1d, overlap_add_1d

@dataclass
class NormState:
    mode: str
    scale: Optional[np.ndarray] = None  # scalar or per-trace vector
    eps: float = 1e-8

def load_mat(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    mat = sio.loadmat(str(path))
    out = {}
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray):
            out[k] = v
    return out

def _ensure_2d_time_trace(x: np.ndarray) -> np.ndarray:
    # Accept (nt, ntr) or (ntr, nt). Heuristic: nt usually >= ntr.
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (nt, ntr) or (ntr, nt), got shape {x.shape}.")
    nt, ntr = x.shape
    if nt < ntr:
        x = x.T
    return x.astype(np.float32, copy=False)

def normalize(x: np.ndarray, mode: str = "global_max", eps: float = 1e-8) -> Tuple[np.ndarray, NormState]:
    x = x.astype(np.float32, copy=False)
    if mode == "none":
        return x, NormState(mode=mode, scale=None, eps=eps)

    if mode == "global_max":
        s = float(np.max(np.abs(x))) + eps
        return x / s, NormState(mode=mode, scale=np.array([s], dtype=np.float32), eps=eps)

    if mode == "per_trace_std":
        # robust-ish: per-trace std
        nt, ntr = x.shape
        s = np.std(x, axis=0, keepdims=True).astype(np.float32) + eps
        return x / s, NormState(mode=mode, scale=s, eps=eps)

    raise ValueError(f"Unknown norm mode: {mode}")

def denormalize(x: np.ndarray, state: NormState) -> np.ndarray:
    if state.mode == "none" or state.scale is None:
        return x
    if state.mode == "global_max":
        return x * float(state.scale.squeeze())
    if state.mode == "per_trace_std":
        return x * state.scale
    raise ValueError(f"Unknown norm mode: {state.mode}")

def to_training_pairs(
    noisy: np.ndarray,
    clean: Optional[np.ndarray],
    win: int,
    stride: int,
    patch_mode: str = "trace",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Convert (nt,ntr) matrices into (N, win) training samples.

    patch_mode:
      - 'trace': sliding window along time for each trace (recommended)
      - 'flatten': mimic original notebook: flatten whole matrix then chunk
    """
    noisy = _ensure_2d_time_trace(noisy)
    if clean is not None:
        clean = _ensure_2d_time_trace(clean)
        if clean.shape != noisy.shape:
            raise ValueError(f"clean shape {clean.shape} != noisy shape {noisy.shape}")

    if patch_mode == "trace":
        X = extract_windows_1d(noisy, win=win, stride=stride)  # (N, win)
        Y = extract_windows_1d(clean, win=win, stride=stride) if clean is not None else None
        return X, Y

    if patch_mode == "flatten":
        # Flatten in (ntr, nt) order like the notebook
        n = noisy.size
        n_win = n // win
        X = noisy.T.reshape(-1)[: n_win * win].reshape(n_win, win)
        if clean is None:
            return X.astype(np.float32), None
        Y = clean.T.reshape(-1)[: n_win * win].reshape(n_win, win)
        return X.astype(np.float32), Y.astype(np.float32)

    raise ValueError(f"Unknown patch_mode: {patch_mode}")

def reconstruct_from_windows(
    windows: np.ndarray,
    nt: int,
    ntr: int,
    win: int,
    stride: int,
) -> np.ndarray:
    """Inverse of extract_windows_1d for patch_mode='trace'."""
    # windows correspond to traces sequentially: trace0 windows, trace1 windows, ...
    out = np.zeros((nt, ntr), dtype=np.float32)
    idx = 0
    for tr in range(ntr):
        n_win = 1 + max(0, (nt - win) // stride)
        w = windows[idx : idx + n_win]  # (n_win, win)
        idx += n_win
        out[:, tr] = overlap_add_1d(w, n=nt, win=win, stride=stride)
    if idx != windows.shape[0]:
        raise RuntimeError(f"Window count mismatch: used {idx} / total {windows.shape[0]}")
    return out
