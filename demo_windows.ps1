# VSCode Terminal (PowerShell) demo runner
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .

python -m ddae_seismic.scripts.train_syn --config configs/syn_demo.yaml
python -m ddae_seismic.scripts.train_field --config configs/field_demo.yaml
python -m ddae_seismic.scripts.infer --config configs/infer_field.yaml --model runs/field/latest/best.keras
