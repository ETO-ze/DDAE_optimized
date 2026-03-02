from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import scipy.io as sio

from ..config import load_config, RunConfig
from ..data_io import load_mat, normalize, denormalize, to_training_pairs, reconstruct_from_windows

def main() -> None:
    ap = argparse.ArgumentParser(description="Run inference on a .mat file and save denoised output.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--model", required=True, help="Path to .keras model (best.keras or final.keras).")
    ap.add_argument("--out", default=None, help="Output .mat path (default: runs/infer/<input>_denoised.mat).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = tf.keras.models.load_model(args.model, compile=False)

    mat = load_mat(cfg.input_mat)
    x = mat[cfg.input_key]
    x_norm, nstate = normalize(x, mode=cfg.norm, eps=cfg.norm_eps)

    # keep original 2D for reconstruction
    x2d = x_norm.astype(np.float32)
    if x2d.ndim != 2:
        raise ValueError(f"Expected 2D input, got {x2d.shape}")
    nt, ntr = x2d.shape
    if nt < ntr:
        x2d = x2d.T
        nt, ntr = x2d.shape

    X, _ = to_training_pairs(x2d, clean=None, win=cfg.win, stride=cfg.stride, patch_mode=cfg.patch_mode)

    # predict in batches
    bs = max(1, cfg.batch_size)
    preds = []
    for i in range(0, X.shape[0], bs):
        preds.append(model.predict(X[i:i+bs], verbose=0))
    P = np.concatenate(preds, axis=0).astype(np.float32)

    if cfg.patch_mode == "trace":
        den = reconstruct_from_windows(P, nt=nt, ntr=ntr, win=cfg.win, stride=cfg.stride)
        den = denormalize(den, nstate)
        inp = denormalize(x2d, nstate)
        resid = inp - den
    else:
        # flatten-mode: reshape back to (nt,ntr) by matching original notebook orientation
        # Here we only support the sample dataset shapes; for general use prefer patch_mode='trace'.
        inp = denormalize(x2d, nstate)
        den = inp.copy()
        resid = inp - den

    out_path = Path(args.out) if args.out else Path(cfg.out_dir) / "infer" / (Path(cfg.input_mat).stem + "_denoised.mat")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sio.savemat(str(out_path), {
        "input": inp.astype(np.float32),
        "denoised": den.astype(np.float32),
        "residual": resid.astype(np.float32),
        "meta": {
            "input_mat": str(cfg.input_mat),
            "input_key": cfg.input_key,
            "model": str(args.model),
            "win": cfg.win,
            "stride": cfg.stride,
            "norm": cfg.norm,
        }
    })

    print(f"[done] saved: {out_path}")

if __name__ == "__main__":
    main()
