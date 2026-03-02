## 1.解压 zip，VSCode 打开该文件夹

## 2.终端（PowerShell）执行：

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .

## 3.跑 Stage-1（合成监督预训练）：

python -m ddae_seismic.scripts.train_syn --config configs/syn_demo.yaml

## 4.跑 Stage-2（野外自监督微调）：

python -m ddae_seismic.scripts.train_field --config configs/field_demo.yaml

## 5.推理输出 .mat（含 denoised+residual）：

python -m ddae_seismic.scripts.infer --config configs/infer_field.yaml --model runs/field/latest/best.keras

输出会在：runs/infer/NEWREAL3_1_denoised.mat