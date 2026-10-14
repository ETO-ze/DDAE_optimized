# DDAE for Seismic Random Noise Attenuation

A Python/TensorFlow refactor of the notebook-based
`Deep-Denoising-Autoencoder-for-Seismic-Random-Noise-Attenuation` workflow.
The project separates data preparation, two-stage training and `.mat` inference
so experiments can be configured and reproduced from the command line.

## What is implemented

- Supervised synthetic-data pretraining with MSE
- Field-data self-supervised fine-tuning with a correlation-based loss and optional MSE anchor
- Dense and Conv1D autoencoders with compatible-layer weight transfer
- Trace-wise window extraction and weighted overlap-add reconstruction
- Legacy trace-major flattened chunks, including reconstruction from model predictions
- YAML configurations and timestamped runs containing configuration, history and model checkpoints

## Quick start

Use Python 3.10 or 3.11 with a TensorFlow-compatible environment. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ddae_seismic.scripts.train_syn --config configs/syn_demo.yaml
python -m ddae_seismic.scripts.train_field --config configs/field_demo.yaml
python -m ddae_seismic.scripts.infer --config configs/infer_field.yaml --model runs/field/latest/best.keras
```

The demo configurations use files in [`data/sample`](data/sample). Training can
be time-consuming; the default field stage allows up to 300 epochs. Edit the
configuration before running if a shorter experiment is needed.

The synthetic and field demos use different window lengths. Weight transfer
copies only layers with matching names and shapes, not incompatible input/output
layers. Inference must use the window length of its saved model.

`latest` model aliases work through either a symlink or `LATEST.txt`, including
Windows without symlink privileges. Both inference and field-stage pretrained
model loading resolve the pointer automatically. A direct timestamped `.keras`
path is also supported.

Inference writes `input`, `denoised`, `residual` and `meta` to
`runs/infer/<input>_denoised.mat`, preserving the input matrix's orientation.

## Window and data conventions

- Inputs must be nonempty 2D matrices. The existing orientation heuristic treats
  the longer axis as time; square matrices are treated as `(time, traces)`.
  Data where the actual time axis is shorter than the trace axis needs an
  explicit orientation-aware workflow instead of relying on this heuristic.
- Trace mode requires `1 <= stride <= win`. A final window covers the tail;
  traces shorter than `win` are edge-padded and cropped after prediction.
- Reconstruction uses the positive interior of a Hann window, so first/last
  samples are retained even when `stride == win`.
- Flatten mode keeps trace-major, non-overlapping chunk order and ignores
  `stride`. Its final chunk is edge-padded and cropped after prediction.
- Tail and short-trace handling changes the training sample count compared with
  older versions that discarded incomplete windows. Record the code revision
  alongside experiment settings when comparing runs.

## Tests

The numerical and path regression suite needs only NumPy, SciPy and PyYAML:

```bash
python -m pip install numpy scipy pyyaml
# Linux/macOS:
PYTHONPATH=src python -m unittest discover -s tests -v
# Windows PowerShell:
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

With TensorFlow installed, the same command also runs a tiny saved-model CLI
smoke test. It verifies actual prediction, `.keras` loading, pointer fallback
and `.mat` output without training a model. Without TensorFlow, that test is
explicitly skipped. GitHub Actions runs the numerical suite on Linux/Windows
and a separate TensorFlow integration job.

These are correctness checks, not denoising-quality benchmarks. No SNR gain,
field-data quality improvement or reproduction claim is established by these
tests. Evaluate on held-out data before using the results for research claims.

## Documentation

- [VSCode walkthrough and troubleshooting](docs/vscode定向优化使用.md)
- [Engineering changes from the notebook](docs/项目优化说明.md)
- [Original example notebook](data/sample/DDAE_Denoising.ipynb)
