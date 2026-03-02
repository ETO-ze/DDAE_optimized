from __future__ import annotations

import argparse
from pathlib import Path
import tensorflow as tf

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert legacy Keras JSON+H5 weights into a single .keras file.")
    ap.add_argument("--json", required=True, help="Path to model json (e.g., bestInitial_syn.json)")
    ap.add_argument("--weights", required=True, help="Path to weights h5 (e.g., bestInitial_synW.h5)")
    ap.add_argument("--out", required=True, help="Output path (.keras)")
    args = ap.parse_args()

    json_text = Path(args.json).read_text(encoding="utf-8")
    model = tf.keras.models.model_from_json(json_text)
    model.load_weights(args.weights)
    model.save(args.out)
    print(f"[done] saved: {args.out}")

if __name__ == "__main__":
    main()
