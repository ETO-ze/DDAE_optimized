from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio

from ..config import load_config, RunConfig
from ..data_io import (
    load_mat, prepare_time_traces, denormalize, to_training_pairs,
    reconstruct_from_windows, reconstruct_flattened,
)
from ._common import resolve_model_path


def denoise_array(model, x: np.ndarray, cfg: RunConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict every sample and return input, denoised, residual in input orientation."""
    if x.ndim != 2 or not x.size:
        raise ValueError(f"Expected nonempty 2D input, got {x.shape}")
    # Apply the existing time/trace heuristic before per-trace normalization.
    transposed = x.shape[0] < x.shape[1]
    x_norm, nstate = prepare_time_traces(x, mode=cfg.norm, eps=cfg.norm_eps)
    nt, ntr = x_norm.shape
    X, _ = to_training_pairs(
        x_norm, clean=None, win=cfg.win, stride=cfg.stride, patch_mode=cfg.patch_mode
    )

    bs = max(1, cfg.batch_size)
    preds = []
    for i in range(0, X.shape[0], bs):
        batch = X[i:i + bs]
        prediction = np.asarray(model.predict(batch, verbose=0), dtype=np.float32)
        if prediction.shape != batch.shape:
            raise ValueError(f"Model output shape {prediction.shape} != input window shape {batch.shape}")
        preds.append(prediction)
    P = np.concatenate(preds, axis=0)

    if cfg.patch_mode == "trace":
        den = reconstruct_from_windows(P, nt=nt, ntr=ntr, win=cfg.win, stride=cfg.stride)
    else:
        den = reconstruct_flattened(P, nt=nt, ntr=ntr, win=cfg.win)
    den = denormalize(den, nstate)
    inp = denormalize(x_norm, nstate)
    if transposed:
        inp, den = inp.T, den.T
    return inp, den, inp - den


def main() -> None:
    ap = argparse.ArgumentParser(description="Run inference on a .mat file and save denoised output.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--model", required=True, help="Path to .keras model; latest aliases support LATEST.txt.")
    ap.add_argument("--out", default=None, help="Output .mat path (default: runs/infer/<input>_denoised.mat).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_path = resolve_model_path(args.model)
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    mat = load_mat(cfg.input_mat)
    inp, den, resid = denoise_array(model, mat[cfg.input_key], cfg)

    out_path = Path(args.out) if args.out else Path(cfg.out_dir) / "infer" / (Path(cfg.input_mat).stem + "_denoised.mat")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(out_path), {
        "input": inp.astype(np.float32),
        "denoised": den.astype(np.float32),
        "residual": resid.astype(np.float32),
        "meta": {
            "input_mat": str(cfg.input_mat),
            "input_key": cfg.input_key,
            "model": str(model_path),
            "win": cfg.win,
            "stride": cfg.stride,
            "patch_mode": cfg.patch_mode,
            "norm": cfg.norm,
        }
    })
    print(f"[done] saved: {out_path}")


if __name__ == "__main__":
    main()
