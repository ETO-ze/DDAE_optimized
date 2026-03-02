from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json

def make_run_dir(root: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(root) / tag / ts
    out.mkdir(parents=True, exist_ok=True)
    # create convenience symlink-like pointer on platforms that support it (optional)
    latest = Path(root) / tag / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            if latest.is_symlink():
                latest.unlink()
            else:
                # if it's a folder created by previous run on windows, skip
                pass
        latest.symlink_to(out, target_is_directory=True)
    except Exception:
        # fallback: write a text pointer
        (Path(root) / tag / "LATEST.txt").write_text(str(out), encoding="utf-8")
    return out
