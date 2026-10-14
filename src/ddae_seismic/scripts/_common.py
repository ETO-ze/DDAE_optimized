from __future__ import annotations

from pathlib import Path
from datetime import datetime


def make_run_dir(root: str, tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    parent = Path(root) / tag
    suffix = 0
    while True:
        out = parent / (ts if suffix == 0 else f"{ts}_{suffix}")
        try:
            out.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            # Some Windows clocks repeat even with microseconds; never reuse a run.
            suffix += 1
    # The text pointer is authoritative even when a legacy 'latest' folder exists.
    (out.parent / "LATEST.txt").write_text(out.name + "\n", encoding="utf-8")
    latest = out.parent / "latest"
    try:
        if latest.is_symlink():
            latest.unlink()
        if not latest.exists():
            # A relative symlink target is resolved from latest's parent, not cwd.
            latest.symlink_to(out.name, target_is_directory=True)
    except OSError:
        # Windows commonly lacks symlink privileges; model loading reads LATEST.txt.
        pass
    return out


def resolve_model_path(path: str | Path) -> Path:
    """Resolve a model file or a runs/<tag>/latest/... alias on every platform.

    New text pointers are relative to LATEST.txt. Legacy absolute and cwd-relative
    pointers are also accepted. A present pointer takes precedence over a stale
    manually copied 'latest' directory; a bad pointer is never silently ignored.
    """
    path = Path(path)
    for directory in path.parents:
        if directory.name != "latest":
            continue
        pointer = directory.parent / "LATEST.txt"
        if pointer.is_file():
            value = pointer.read_text(encoding="utf-8").strip()
            if not value:
                raise FileNotFoundError(f"Empty model pointer: {pointer}")
            target = Path(value)
            roots = [target] if target.is_absolute() else [pointer.parent / target, target]
            for root in roots:
                candidate = root / path.relative_to(directory)
                if candidate.is_file():
                    return candidate.resolve()
            raise FileNotFoundError(f"Model {path.name} not found using {pointer}: {value}")
    if path.is_file():
        return path.resolve()
    raise FileNotFoundError(
        f"Model not found: {path}. Train the required stage first, or pass an existing .keras file."
    )
