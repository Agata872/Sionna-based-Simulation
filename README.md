# Sionna-based Simulation

基于 NVIDIA Sionna 的链路级与系统级仿真环境（搭建于 2026-10-01）。

## 环境概要

| 项目 | 版本 / 配置 |
|---|---|
| Sionna | `sionna-no-rt` 2.2.0（PHY 链路级 + SYS 系统级，不含 RT 射线追踪） |
| PyTorch | 2.14.1+cu126（CUDA 12.6 构建） |
| Python | 3.12.3（虚拟环境 `.venv`） |
| GPU | 2 × Tesla T4（每块约 14.6 GiB 可用显存），NVIDIA 驱动 535.288.01 |
| 系统 | Ubuntu 24.04.4 LTS |

## 目录结构

```
Sionna-based Simulation/
├── .venv/              # Python 虚拟环境，所有依赖都在这里（不纳入 git）
├── sionna/             # 官方仓库 v2.2.0 源码（git 子模块，含 sionna-rt 子模块）
│   └── tutorials/      #   phy/ 链路级、sys/ 系统级、rt/ 射线追踪 教程
├── env.sh              # 激活环境
├── verify_env.py       # 环境验证脚本
├── requirements.txt    # 锁定的依赖版本
└── README.md
```

## 快速开始

```bash
cd "/home/user/Sionna-based Simulation"   # 路径含空格，务必加引号
source env.sh                             # 激活环境；退出用 deactivate
python verify_env.py                      # 验证环境，全部通过约需 15 秒
```

在 VS Code 中使用：`Ctrl+Shift+P` → `Python: Select Interpreter` →
选择 `/home/user/Sionna-based Simulation/.venv/bin/python`。

最小示例：

```python
import sionna.phy
sionna.phy.config.device = "cuda:0"   # 选择 GPU："cuda:0" / "cuda:1" / "cpu"
sionna.phy.config.seed = 42
```

## 注意事项

### 1. 不要从 PyPI 安装或升级 torch

PyPI 上默认的 torch 是 CUDA 13 构建，需要驱动 ≥ 580；本机驱动是 535，装上后 GPU 会不可用。
安装其他包时，建议把 torch 一并写上，防止被替换：

```bash
pip install <包名> "torch==2.14.1+cu126"
```

装完新包后，可以运行 `python verify_env.py` 确认 GPU 仍然可用。

### 2. 未安装射线追踪模块 Sionna RT

Sionna RT 2.1.0 及以后的版本依赖 Dr.Jit 1.5.0，它生成的 `copysign` 指令不被 535 驱动自带的
OptiX 支持，GPU 射线追踪会直接崩溃（CPU 后端可用）。因此本环境只装了 `sionna-no-rt`。

以下教程依赖 RT，在本环境中无法运行：

- `tutorials/rt/` 下的全部教程
- `tutorials/phy/Link_Level_Simulations_with_RT.ipynb`
- `tutorials/sys/SYS_Meets_RT.ipynb`
- `tutorials/sys/LinkAdaptation.ipynb`

以后如果需要 RT，可以选择：

- 由管理员把驱动升级到较新版本（如 580 系列）后，换装完整版：
  ```bash
  pip uninstall -y sionna-no-rt      # sionna 与 sionna-no-rt 共用同一个包目录，须先卸载
  pip install "sionna==2.2.0" "torch==2.14.1+cu126"
  ```
- 不升级驱动：sionna-rt 2.0.1（Dr.Jit 1.3.1 / Mitsuba 3.8.0）已实测可在本机 GPU 上运行，
  但没有 2.1/2.2 的新功能（RCS 模块、新版绕射模型等）。

### 3. 双 GPU 并行

Sionna 的每个进程使用一块 GPU。可以同时开两个进程，分别跑不同的仿真：

```bash
CUDA_VISIBLE_DEVICES=0 python sim_a.py &
CUDA_VISIBLE_DEVICES=1 python sim_b.py &
```

显存不足（OOM）时，减小 `batch_size`。

### 4. 运行教程 notebook

本环境未安装 JupyterLab。在 VS Code 中打开 `.ipynb` 并运行单元格时，VS Code 会提示在 `.venv`
中安装 `ipykernel`，同意即可。

## 复现环境

```bash
git clone --recursive git@github.com:Agata872/Sionna-based-Simulation.git   # --recursive 会一并拉取 sionna 子模块
cd Sionna-based-Simulation
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 参考

- 文档：https://nvlabs.github.io/sionna/
- 源码：https://github.com/NVlabs/sionna
