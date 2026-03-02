from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

from ..config import load_config, save_effective_config, RunConfig
from ..data_io import load_mat, normalize, to_training_pairs
from ..models import build_dense_dae, build_conv1d_dae, transfer_compatible_weights
from ..losses import CorrelationLossConfig, make_correlation_denoise_loss
from ..utils.seed import set_seed
from ._common import make_run_dir

def _build_model(cfg: RunConfig) -> tf.keras.Model:
    if cfg.model_type == "dense":
        return build_dense_dae(
            input_dim=cfg.win,
            widths=cfg.dense_widths,
            l2=cfg.l2,
            dropout=cfg.dropout,
            name="dense_dae_field",
        )
    if cfg.model_type == "conv1d":
        return build_conv1d_dae(win=cfg.win, name="conv1d_dae_field")
    raise ValueError(f"Unknown model_type: {cfg.model_type}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 field training (unsupervised/self-supervised).")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    out_dir = make_run_dir(cfg.out_dir, tag="field")
    save_effective_config(cfg, out_dir / "config_effective.yaml")

    mat = load_mat(cfg.input_mat)
    noisy = mat[cfg.input_key]
    noisy, nstate = normalize(noisy, mode=cfg.norm, eps=cfg.norm_eps)

    X, _ = to_training_pairs(noisy, clean=None, win=cfg.win, stride=cfg.stride, patch_mode=cfg.patch_mode)
    Y = X  # auto-encoding target (loss enforces denoising)
    print(f"[data] X={X.shape} norm={cfg.norm} win={cfg.win} stride={cfg.stride} patch_mode={cfg.patch_mode}")

    model = _build_model(cfg)

    # transfer weights
    if cfg.pretrained_model:
        src = tf.keras.models.load_model(cfg.pretrained_model, compile=False)
        transfer_compatible_weights(src, model, verbose=True)

    if cfg.freeze_encoder:
        for layer in model.layers:
            if layer.name.startswith("enc_") or layer.name in ("bottleneck",):
                layer.trainable = False
        print("[train] encoder frozen.")

    loss_cfg = CorrelationLossConfig(
        corr_target=cfg.corr_target,
        corr_residual=cfg.corr_residual,
        w_target=cfg.w_target,
        w_residual=cfg.w_residual,
        w_mse=cfg.w_mse,
        eps=cfg.corr_eps,
    )
    loss_fn = make_correlation_denoise_loss(loss_cfg)

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
    model.compile(optimizer=opt, loss=loss_fn)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.keras"),
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=cfg.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=max(3, cfg.patience // 3)),
        tf.keras.callbacks.CSVLogger(str(out_dir / "history.csv")),
    ]

    model.fit(
        X,
        Y,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    model.save(out_dir / "final.keras")
    print(f"[done] saved: {out_dir}")

if __name__ == "__main__":
    main()
