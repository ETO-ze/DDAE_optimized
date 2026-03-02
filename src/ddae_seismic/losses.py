from __future__ import annotations

from dataclasses import dataclass
import tensorflow as tf

def _corrcoef_batch(x: tf.Tensor, y: tf.Tensor, eps: float = 1e-8) -> tf.Tensor:
    """Pearson correlation coefficient per-sample, then averaged.

    x,y: (batch, dim) or any shape with batch first
    returns: scalar mean corr in [-1,1] (numerically clipped)
    """
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    y = tf.reshape(y, [tf.shape(y)[0], -1])

    x = x - tf.reduce_mean(x, axis=1, keepdims=True)
    y = y - tf.reduce_mean(y, axis=1, keepdims=True)

    num = tf.reduce_sum(x * y, axis=1)
    den = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) * tf.reduce_sum(tf.square(y), axis=1) + eps)
    r = num / den
    r = tf.clip_by_value(r, -1.0, 1.0)
    return tf.reduce_mean(r)

@dataclass
class CorrelationLossConfig:
    corr_target: float = 1.0
    corr_residual: float = 0.0
    w_target: float = 1.0
    w_residual: float = 1.0
    w_mse: float = 0.0
    eps: float = 1e-8

def make_correlation_denoise_loss(cfg: CorrelationLossConfig):
    """Loss used for field (unsupervised/self-supervised) stage.

    Goal:
      - corr(y_true, y_pred) -> cfg.corr_target (typically 1)
      - corr(y_true - y_pred, y_pred) -> cfg.corr_residual (typically 0)
      - optional MSE anchor to preserve amplitude/phase scale
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        resid = y_true - y_pred
        r_sig = _corrcoef_batch(y_true, y_pred, eps=cfg.eps)
        r_res = _corrcoef_batch(resid, y_pred, eps=cfg.eps)

        l_sig = tf.square(cfg.corr_target - r_sig)
        l_res = tf.square(cfg.corr_residual - r_res)

        l = cfg.w_target * l_sig + cfg.w_residual * l_res
        if cfg.w_mse and cfg.w_mse > 0:
            l = l + cfg.w_mse * tf.reduce_mean(tf.square(resid))
        return l
    return loss
