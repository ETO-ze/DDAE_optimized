from __future__ import annotations

from typing import Tuple, Optional

import tensorflow as tf

def build_dense_dae(
    input_dim: int,
    widths: Tuple[int, int, int] = (512, 256, 128),
    l2: float = 0.0,
    dropout: float = 0.0,
    name: str = "dense_dae",
) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(l2) if l2 and l2 > 0 else None
    inp = tf.keras.Input(shape=(input_dim,), name="x")
    x = inp
    if dropout and dropout > 0:
        x = tf.keras.layers.Dropout(dropout, name="in_dropout")(x)

    x = tf.keras.layers.Dense(widths[0], activation="relu", kernel_regularizer=reg, name="enc_dense_1")(x)
    x = tf.keras.layers.Dense(widths[1], activation="relu", kernel_regularizer=reg, name="enc_dense_2")(x)
    z = tf.keras.layers.Dense(widths[2], activation="relu", kernel_regularizer=reg, name="bottleneck")(x)

    x = tf.keras.layers.Dense(widths[2], activation="relu", kernel_regularizer=reg, name="dec_dense_1")(z)
    x = tf.keras.layers.Dense(widths[1], activation="relu", kernel_regularizer=reg, name="dec_dense_2")(x)
    x = tf.keras.layers.Dense(widths[0], activation="relu", kernel_regularizer=reg, name="dec_dense_3")(x)

    out = tf.keras.layers.Dense(input_dim, activation="linear", name="y")(x)
    return tf.keras.Model(inp, out, name=name)

def build_conv1d_dae(
    win: int,
    n_ch: int = 1,
    base: int = 32,
    name: str = "conv1d_dae",
) -> tf.keras.Model:
    """A light Conv1D autoencoder for trace-wise windows.

    Input: (win,) -> internally reshape to (win,1)
    Output: (win,)
    """
    inp = tf.keras.Input(shape=(win,), name="x")
    x = tf.keras.layers.Reshape((win, n_ch), name="reshape_in")(inp)

    # Encoder
    x = tf.keras.layers.Conv1D(base, 5, padding="same", activation="relu", name="enc_c1")(x)
    x = tf.keras.layers.Conv1D(base, 5, padding="same", activation="relu", name="enc_c2")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)

    x = tf.keras.layers.Conv1D(base*2, 5, padding="same", activation="relu", name="enc_c3")(x)
    x = tf.keras.layers.Conv1D(base*2, 5, padding="same", activation="relu", name="enc_c4")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="pool2")(x)

    # Bottleneck
    x = tf.keras.layers.Conv1D(base*4, 3, padding="same", activation="relu", name="bottleneck")(x)

    # Decoder
    x = tf.keras.layers.UpSampling1D(2, name="up1")(x)
    x = tf.keras.layers.Conv1D(base*2, 5, padding="same", activation="relu", name="dec_c1")(x)
    x = tf.keras.layers.Conv1D(base*2, 5, padding="same", activation="relu", name="dec_c2")(x)

    x = tf.keras.layers.UpSampling1D(2, name="up2")(x)
    x = tf.keras.layers.Conv1D(base, 5, padding="same", activation="relu", name="dec_c3")(x)
    x = tf.keras.layers.Conv1D(base, 5, padding="same", activation="relu", name="dec_c4")(x)

    x = tf.keras.layers.Conv1D(1, 3, padding="same", activation="linear", name="out_c")(x)
    out = tf.keras.layers.Reshape((win,), name="reshape_out")(x)
    return tf.keras.Model(inp, out, name=name)

def transfer_compatible_weights(src: tf.keras.Model, dst: tf.keras.Model, verbose: bool = True) -> None:
    """Copy weights layer-by-layer when shapes match.

    This is robust to different input/output dims (first/last layers usually mismatch),
    while copying the shared middle layers (like the original notebook did).
    """
    src_layers = {layer.name: layer for layer in src.layers}
    copied = 0
    for layer in dst.layers:
        if layer.name not in src_layers:
            continue
        sw = src_layers[layer.name].get_weights()
        dw = layer.get_weights()
        if len(sw) == 0 or len(dw) == 0:
            continue
        if len(sw) != len(dw):
            continue
        if all(a.shape == b.shape for a, b in zip(sw, dw)):
            layer.set_weights(sw)
            copied += 1
            if verbose:
                print(f"[transfer] copied layer: {layer.name}")
    if verbose:
        print(f"[transfer] total layers copied: {copied}")
