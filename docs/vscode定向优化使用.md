# DDAE 地震随机噪声衰减（优化版）使用说明书（VSCode 优先）

> 适用对象：Windows/VSCode 用户，先脚本化跑通；若 VSCode Notebook 不顺，再用 Jupyter 作为兜底。  
> 项目目录：`ddae_optimized/`（已包含示例数据与脚本）。  
> 核心流程：Stage‑1 合成数据监督预训练（noisy→clean）→ Stage‑2 野外数据自监督微调（noisy→noisy + 相关性去噪损失）→ 推理输出 denoised/residual。

---

## 0. 项目结构速览（你需要知道的文件/目录）

- `src/ddae_seismic/`：核心包
  - `scripts/train_syn.py`：Stage‑1 合成监督训练
  - `scripts/train_field.py`：Stage‑2 野外自监督训练（含迁移学习）
  - `scripts/infer.py`：推理并导出 `.mat`
  - `losses.py`：相关系数去噪损失（Correlation denoise loss）
  - `models.py`：Dense/Conv1D 自编码器
  - `data_io.py` + `utils/patching.py`：`.mat` 读写、切窗/拼接
- `configs/`
  - `syn_demo.yaml`：Stage‑1 示例配置
  - `field_demo.yaml`：Stage‑2 示例配置（**注意 corr_eps 写法**，见“故障排查”）
  - `infer_field.yaml`：推理示例配置
- `data/sample/`：示例数据（与原论文复现仓库一致）
  - `DDAE_SYN.mat`：合成 noisy/clean 对
  - `NEWREAL3_1.mat`：野外数据（含 `dn1` 归一化噪声、以及对比基线）
  - `bestInitial_syn.json` / `bestInitial_synW.h5`：原版预训练模型结构/权重（可用 `import_legacy.py` 转成 `.keras`）
- `runs/`：训练/推理输出目录（脚本自动创建）

---

## 1. 环境准备（Windows + VSCode）

### 1.1 安装/确认 Python（推荐 3.11 x64）
在 PowerShell 执行：

```powershell
py -0p
python --version
```

应当能看到类似：
- `Python 3.11.x`
- `C:\Users\...\Python311\python.exe`

> 注意：PowerShell 里 `where` 是 `Where-Object` 的别名，不等于 `where.exe`。  
> 要查路径请用：
> `Get-Command python` 或 `where.exe python`

---

## 2. VSCode 优先：创建虚拟环境 + 安装依赖

在项目根目录（例如：`...\ddae_optimized\ddae_optimized`）执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 2.1 若激活脚本被拦截（running scripts is disabled）
执行一次：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后重新激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2.2 VSCode 选择解释器（非常重要）
VSCode：`Ctrl + Shift + P` → `Python: Select Interpreter` → 选择：

- `.\.venv\Scripts\python.exe`

---

## 3. 运行流程（Stage‑1 → Stage‑2 → Infer）

### 3.1 Stage‑1：合成数据监督训练（预训练）
命令：

```powershell
python -m ddae_seismic.scripts.train_syn --config configs/syn_demo.yaml
```

正常输出会包含：
- `[data] X=(..., 256) Y=(..., 256)`（切窗后样本数可能随 stride 变化）
- 训练 epoch 日志（loss/val_loss）
- `[done] saved: runs\syn\YYYYMMDD_HHMMSS`

最终模型位置（示例）：
- `runs\syn\YYYYMMDD_HHMMSS\best.keras`

---

### 3.2 Stage‑2：野外数据自监督微调（迁移学习 + 相关性损失）
命令：

```powershell
python -m ddae_seismic.scripts.train_field --config configs/field_demo.yaml
```

Stage‑2 会先加载 Stage‑1 的 `.keras` 作为 `pretrained_model`，并在打印中显示：
- `[transfer] copied layer: ...`
- `[transfer] total layers copied: N`

**关键前置条件：`configs/field_demo.yaml` 里的 `pretrained_model` 路径必须指向真实存在的 `.keras`。**
默认是：
- `pretrained_model: runs/syn/latest/best.keras`

Windows 上如果没有 symlink 权限，`runs/syn/latest` 可能不存在（见“故障排查 A”）。

---

### 3.3 推理（infer）：输出 `.mat`（含 denoised + residual）
Stage‑2 成功后，推荐用 `LATEST.txt` 自动定位模型：

```powershell
$fld = Get-Content runs\field\LATEST.txt
python -m ddae_seismic.scripts.infer --config configs/infer_field.yaml --model "$fld\best.keras"
```

输出通常在：
- `runs\infer\*.mat`

输出变量（通常包含）：
- `denoised`：去噪结果
- `residual`：残差（输入-输出）
- `meta`：窗口、步长、归一化信息等（用于复现实验）

---

## 4. 故障排查（结合你这次真实报错整理）

### A) `ValueError: File not found: runs/syn/latest/best.keras`
**原因：Windows 未启用 symlink（开发者模式/权限），脚本 fallback 写了 `runs/syn/LATEST.txt`，但不会自动创建 `runs/syn/latest/`。**

**方案 1（推荐）：按 LATEST.txt 改配置或复制模型**
执行：

```powershell
New-Item -ItemType Directory -Force runs\syn\latest | Out-Null
$syn = Get-Content runs\syn\LATEST.txt
Copy-Item "$syn\best.keras" runs\syn\latest\best.keras -Force
```

然后再跑 Stage‑2。

**方案 2：直接改 `configs/field_demo.yaml`**
把：
- `pretrained_model: runs/syn/latest/best.keras`

改为：
- `pretrained_model: runs/syn/YYYYMMDD_HHMMSS/best.keras`（用实际路径）

**方案 3：启用 Windows 开发者模式（长期方案）**
Settings → Privacy & security → For developers → Developer Mode = On  
之后 symlink 更容易成功。

---

### B) `TypeError: ... AddV2 ... type string does not match type float32`
你这次的根因是：`configs/field_demo.yaml` 中 `corr_eps: 1e-8` 被 YAML 解析成 **字符串**，导致 loss 里 `+ eps` 发生 string/float 混算。

**修复：把 `corr_eps` 改成带小数点的科学计数法或十进制**
在 `configs/field_demo.yaml` 中将：

```yaml
corr_eps: 1e-8
```

改成任意一种：

```yaml
corr_eps: 1.0e-8
# 或
corr_eps: 0.00000001
```

然后重新运行 Stage‑2：

```powershell
python -m ddae_seismic.scripts.train_field --config configs/field_demo.yaml
```

> 建议：如果你在其它配置里也写了 `1e-3 / 5e-4` 这种格式，同样改成 `1.0e-3 / 5.0e-4`。

---

### C) `No suitable Python runtime found`（你最早的报错）
原因：你当时执行了 `py -3.10 ...`，但机器上没有 3.10；你实际装的是 3.11。

修复：改用：

```powershell
py -3.11 -m venv .venv
```

---

### D) `where python` 没输出（PowerShell 里）
PowerShell 的 `where` 不是 `where.exe`。请用：

```powershell
Get-Command python
where.exe python
```

---

### E) oneDNN / CPU 指令提示（信息级，不是错误）
看到类似：
- `oneDNN custom operations are on...`
- `This TensorFlow binary is optimized to use available CPU instructions...`

这不影响使用；只是提示数值微小差异或 CPU 特性。

---

## 5. 参数说明（配置文件关键字段）

### 5.1 切窗/拼接（patching）
- `win`：窗口长度（时间采样点数）
- `stride`：滑窗步长
- `patch_mode: trace`：按每条 trace 进行滑窗（推荐）
- 推理端使用 overlap‑add 拼回完整信号，减少窗边伪影。

### 5.2 归一化（norm）
- `global_max`：全局最大值归一化（合成阶段默认）
- `none`：不归一化（示例 field 里 `dn1` 已预归一化）

### 5.3 Field 自监督损失（相关性去噪）
- `corr_target`：希望 corr(input, output) → 1
- `corr_residual`：希望 corr(residual, output) → 0
- `w_target / w_residual`：两项权重
- `w_mse`：可选振幅锚定（建议 1e-4 ~ 1e-2 微调）
- `corr_eps`：数值稳定项（**写成 1.0e-8**）

---

## 6. 针对本项目的优化建议（基于你这次真实运行暴露的问题）

### 6.1 可靠性/易用性（优先）
1) **修复 latest 指针机制（推荐在代码层改）**
   - 目前 symlink 失败只写 `LATEST.txt`，但默认配置依赖 `runs/*/latest/`，会导致“找不到 best.keras”。
   - 建议改为：symlink 失败时自动创建 `runs/tag/latest/` 并复制/硬链接 `best.keras`，或自动回退读取 `LATEST.txt`。

2) **配置解析强制类型转换**
   - 对 `corr_eps / lr / norm_eps` 等字段在加载时 `float(...)`，彻底规避 YAML “1e-8 字符串”坑。

3) **在 scripts 中打印更明确的提示**
   - 若 `pretrained_model` 不存在，打印：
     - “symlink 可能失败 → 请查看 runs/syn/LATEST.txt”
     - 并给出一条可复制的修复命令。

### 6.2 效果上限（可选）
1) `model_type` 从 `dense` 升级到 `conv1d`（更适配地震时序结构）。
2) Field 阶段加入轻量频域/平滑约束（避免过平滑与幅值漂移）。
3) 增加评估：SNR 提升（合成），以及结构相干/残差能量（野外）。

---

## 7. Jupyter 兜底用法（如果你想直接看原 Notebook）
你也可以直接打开示例 Notebook：
- `data/sample/DDAE_Denoising.ipynb`

在 VSCode 里安装 Jupyter 扩展即可运行；或：

```powershell
pip install jupyter
jupyter notebook
```

---

## 8. 常用命令速查（复制即用）

### Stage‑1
```powershell
python -m ddae_seismic.scripts.train_syn --config configs/syn_demo.yaml
```

### 修复 syn latest（Windows）
```powershell
New-Item -ItemType Directory -Force runs\syn\latest | Out-Null
$syn = Get-Content runs\syn\LATEST.txt
Copy-Item "$syn\best.keras" runs\syn\latest\best.keras -Force
```

### Stage‑2
```powershell
python -m ddae_seismic.scripts.train_field --config configs/field_demo.yaml
```

### Infer（自动取最新 field 模型）
```powershell
$fld = Get-Content runs\field\LATEST.txt
python -m ddae_seismic.scripts.infer --config configs/infer_field.yaml --model "$fld\best.keras"
```

---

**如果你要把自己的真实数据接入**：  
只要保证 `.mat` 里有一个二维数组键（例如 `dn1`），形状建议为 `(nt, ntraces)`，然后在配置里改：
- `input_mat: path/to/your.mat`
- `input_key: your_key`
- `win/stride` 按你的采样率与波形周期选即可。

