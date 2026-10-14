from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import scipy.io as sio

from .utils.patching import extract_windows_1d, overlap_add_1d, window_starts

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

def prepare_time_traces(
    x: np.ndarray, mode: str = "global_max", eps: float = 1e-8
) -> Tuple[np.ndarray, NormState]:
    """Orient time/trace axes before normalization in both training and inference."""
    return normalize(_ensure_2d_time_trace(x), mode=mode, eps=eps)


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
      - 'flatten': trace-major chunks; edge-pad the last chunk (stride is unused)
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
        # Preserve the notebook's trace-major, non-overlapping chunk order.
        # Pad rather than silently discard a partial final chunk.
        if win <= 0 or noisy.size == 0:
            raise ValueError("win and input size must be > 0")
        padding = (-noisy.size) % win

        def chunk(x: np.ndarray) -> np.ndarray:
            flat = np.pad(x.T.reshape(-1), (0, padding), mode="edge")
            return flat.reshape(-1, win).astype(np.float32, copy=False)

        return chunk(noisy), chunk(clean) if clean is not None else None

    raise ValueError(f"Unknown patch_mode: {patch_mode}")

def reconstruct_from_windows(
    windows: np.ndarray,
    nt: int,
    ntr: int,
    win: int,
    stride: int,
) -> np.ndarray:
    """Inverse of extract_windows_1d for patch_mode='trace'."""
    n_win = len(window_starts(nt, win, stride))
    if ntr <= 0 or windows.shape != (ntr * n_win, win):
        raise ValueError(f"Expected windows shape {(ntr * n_win, win)}, got {windows.shape}")
    out = np.empty((nt, ntr), dtype=np.float32)
    for tr in range(ntr):
        out[:, tr] = overlap_add_1d(
            windows[tr * n_win:(tr + 1) * n_win], n=nt, win=win, stride=stride
        )
    return out


def reconstruct_flattened(windows: np.ndarray, nt: int, ntr: int, win: int) -> np.ndarray:
    """Invert flatten-mode chunking, removing only the final padding samples."""
    if nt <= 0 or ntr <= 0 or win <= 0:
        raise ValueError("nt, ntr and win must be > 0")
    count = (nt * ntr + win - 1) // win
    if windows.shape != (count, win):
        raise ValueError(f"Expected windows shape {(count, win)}, got {windows.shape}")
    return windows.reshape(-1)[:nt * ntr].reshape(ntr, nt).T.astype(np.float32, copy=False)
