from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

from ..config import load_config, save_effective_config, RunConfig
from ..data_io import load_mat, normalize, to_training_pairs
from ..models import build_dense_dae, build_conv1d_dae
from ..utils.seed import set_seed
from ._common import make_run_dir

def _build_model(cfg: RunConfig) -> tf.keras.Model:
    if cfg.model_type == "dense":
        return build_dense_dae(
            input_dim=cfg.win,
            widths=cfg.dense_widths,
            l2=cfg.l2,
            dropout=cfg.dropout,
            name="dense_dae_syn",
        )
    if cfg.model_type == "conv1d":
        return build_conv1d_dae(win=cfg.win, name="conv1d_dae_syn")
    raise ValueError(f"Unknown model_type: {cfg.model_type}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1 supervised training on synthetic clean/noisy pairs.")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    out_dir = make_run_dir(cfg.out_dir, tag="syn")
    save_effective_config(cfg, out_dir / "config_effective.yaml")

    mat = load_mat(cfg.input_mat)
    noisy = mat[cfg.input_key]
    if not cfg.target_mat:
        raise ValueError("target_mat must be set for supervised synthetic training.")
    mat2 = load_mat(cfg.target_mat)
    clean = mat2[cfg.target_key]

    noisy, nstate = normalize(noisy, mode=cfg.norm, eps=cfg.norm_eps)
    clean, _ = normalize(clean, mode=cfg.norm, eps=cfg.norm_eps)  # use same scaling style

    X, Y = to_training_pairs(noisy, clean, win=cfg.win, stride=cfg.stride, patch_mode=cfg.patch_mode)
    assert Y is not None

    print(f"[data] X={X.shape} Y={Y.shape} norm={cfg.norm} win={cfg.win} stride={cfg.stride} patch_mode={cfg.patch_mode}")

    model = _build_model(cfg)
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
    model.compile(optimizer=opt, loss="mse")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(3, cfg.patience // 3)),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]

    hist = model.fit(
        X,
        Y,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
        validation_split=cfg.val_split,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(out_dir / "final.keras")
    print(f"[done] saved: {out_dir}")

if __name__ == "__main__":
    main()
