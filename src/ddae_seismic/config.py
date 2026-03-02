from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

@dataclass
class RunConfig:
    # I/O
    input_mat: str
    input_key: str
    target_mat: Optional[str] = None
    target_key: Optional[str] = None
    out_dir: str = "runs"

    # Patching
    win: int = 256
    stride: int = 128
    patch_mode: str = "trace"  # trace | flatten (trace = per-trace sliding windows)

    # Normalization
    norm: str = "global_max"   # global_max | per_trace_std | none
    norm_eps: float = 1e-8

    # Model
    model_type: str = "dense"  # dense | conv1d
    dense_widths: tuple[int, int, int] = (512, 256, 128)
    l2: float = 0.0
    dropout: float = 0.0

    # Training
    seed: int = 42
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 100
    val_split: float = 0.1
    patience: int = 15

    # Unsupervised loss weights (used in field training)
    corr_target: float = 1.0        # want corr(y_true, y_pred) -> 1
    corr_residual: float = 0.0      # want corr(y_true - y_pred, y_pred) -> 0
    w_target: float = 1.0
    w_residual: float = 1.0
    w_mse: float = 0.0              # optional amplitude anchor for field training
    corr_eps: float = 1e-8

    # Transfer learning
    pretrained_model: Optional[str] = None  # path to .keras model from synthetic stage
    freeze_encoder: bool = False

def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    cfg: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig(**cfg)

def save_effective_config(cfg: RunConfig, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg.__dict__, out_path.open("w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
