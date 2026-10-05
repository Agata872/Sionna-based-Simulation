# Sionna-based Simulation

基于 NVIDIA Sionna 的链路级、系统级与射线追踪仿真环境（搭建于 2026-10-01）。

## 环境概要

| 项目 | 版本 / 配置 |
|---|---|
| Sionna | `sionna` 2.2.0（PHY 链路级 + SYS 系统级 + RT 射线追踪） |
| Sionna RT | `sionna-rt` 2.2.0（Mitsuba 3.9.1 / Dr.Jit 1.5.0），GPU 运行依赖 580 用户态库，见注意事项 2 |
| PyTorch | 2.14.1+cu126（CUDA 12.6 构建） |
| Python | 3.12.3（虚拟环境 `.venv`） |
| GPU | 2 × Tesla T4（每块约 14.6 GiB 可用显存） |
| NVIDIA 驱动 | 内核模块 535.288.01（全机共用）；本账号额外使用 580.178.04 用户态库 |
| 系统 | Ubuntu 24.04.4 LTS |

## 目录结构

```
Sionna-based Simulation/
├── .venv/              # Python 虚拟环境，所有依赖都在这里（不纳入 git）
├── sionna/             # 官方仓库 v2.2.0 源码（git 子模块，含 sionna-rt 子模块）
│   └── tutorials/      #   phy/ 链路级、sys/ 系统级、rt/ 射线追踪 教程
├── oran/               # O-RAN 7.2x 下行 Cat A / Cat B 链路级仿真（说明见 oran/README.md）
│   └── compression/    #   前传 IQ 压缩算法（BFP 等）
├── scripts/            # 仿真脚本，如 compare_cat_a_b.py（Cat A/B × 压缩位宽 × SNR 扫描）
├── tests/              # 单元测试：python -m unittest discover -s tests -v
├── results/            # 仿真输出（不纳入 git）
├── env.sh              # 激活环境（终端用）
├── .env                # VS Code 启动 Python / Jupyter 内核时读取的环境变量
├── verify_env.py       # 环境验证脚本
├── requirements.txt    # 锁定的依赖版本
└── README.md

~/.local/nvidia/580.178.04/   # 580 驱动的用户态库（不纳入 git，安装方法见注意事项 2）
```

## 快速开始

```bash
cd "/home/user/Sionna-based Simulation"   # 路径含空格，务必加引号
source env.sh                             # 激活环境；退出用 deactivate
python verify_env.py                      # 验证环境，全部通过约需 20 秒
```

在 VS Code 中使用：`Ctrl+Shift+P` → `Python: Select Interpreter` →
选择 `/home/user/Sionna-based Simulation/.venv/bin/python`。VS Code 会自动读取项目根目录的 `.env`，
无需执行 `env.sh`。

最小示例：

```python
import sionna.phy
sionna.phy.config.device = "cuda:0"   # 选择 GPU："cuda:0" / "cuda:1" / "cpu"
sionna.phy.config.seed = 42
```

## O-RAN 7.2x 下行链路仿真

`oran/` 基于 Sionna PHY 实现了 O-RAN 7.2x 划分下的 PDSCH 下行链路（单小区、2 个 UE 的 MU-MIMO），
包括 Category A（预编码在 O-DU）和 Category B（预编码在 O-RU），以及前传 IQ 压缩（BFP / 块缩放 / μ 律）。
闭环模式用 Sionna RT 对物理场景（默认 munich）射线追踪，UE 随机撒点，按 CQI 选 MCS（含 OLLA）后下行传输。
设计、配置与扩展方法见 [oran/README.md](oran/README.md)。

```bash
source env.sh
python -m unittest discover -s tests -v     # 单元测试
python scripts/compare_cat_a_b.py           # Cat A/B 对比，结果写入 results/cat_ab/
```

## 注意事项

### 1. 不要从 PyPI 安装或升级 torch

PyPI 上默认的 torch 是 CUDA 13 构建，需要驱动 ≥ 580；本机驱动是 535，装上后 GPU 会不可用。
安装其他包时，建议把 torch 一并写上，防止被替换：

```bash
pip install <包名> "torch==2.14.1+cu126"
```

装完新包后，可以运行 `python verify_env.py` 确认 GPU 仍然可用。

### 2. 射线追踪 Sionna RT 依赖 580 用户态库

**问题**：Sionna RT 2.1.0 及以后的版本依赖 Dr.Jit 1.5.0，它生成的 `copysign` 指令不被 535 驱动
自带的 OptiX 支持，GPU 射线追踪会直接崩溃（`optix.ptx.copysign.f32` 未实现）。

**不能直接升级驱动**：驱动的内核模块全机共用，升级需要 root 并重启。2026-02-27 曾尝试升级到
590-server，机器未能正常启动，随后被退回 535 并用 `apt-mark hold` 锁定。升级前须先与管理员确认。

**本环境的做法**：内核模块保持 535 不变，只让本账号的进程加载 580 驱动的用户态库（libcuda、
OptiX 等）。T4 是数据中心 GPU，支持 NVIDIA 的 CUDA 前向兼容，新版 libcuda 可以运行在旧版内核模块上。
这些库放在 `~/.local/nvidia/580.178.04/`，通过 `LD_LIBRARY_PATH` 生效：

- 终端：`source env.sh` 时自动设置
- VS Code：Python / Jupyter 内核启动时读取项目根目录的 `.env`；修改 `.env` 后需重启内核

不设置时（例如没有 `source env.sh` 就运行脚本），PHY / SYS 照常可用，但 GPU 射线追踪会崩溃。
`python verify_env.py` 的第 6 项会检查这一点。

已验证：GPU 射线追踪结果与 CPU 后端一致（信道增益最大差 0.009 dB），
`tutorials/sys/LinkAdaptation.ipynb` 可完整运行。其余 RT 教程未逐一测试。

注意：

- NVIDIA 官方的前向兼容只覆盖 CUDA 相关库，OptiX 以这种方式使用不在官方保证范围内。
  遇到异常的 OptiX 报错时，先考虑这个原因。
- 如果以后系统驱动升级到 580.178.04 以后的版本，须删掉 `env.sh` 和 `.env` 中的这项设置，
  否则用户态库会比内核模块旧，GPU 将不可用。

安装 580 用户态库（不需要 sudo，只解压不安装，约 500 MB）：

```bash
mkdir -p /tmp/nv580 && cd /tmp/nv580
apt-get download libnvidia-compute-580-server libnvidia-gl-580-server
for d in *.deb; do dpkg-deb -x "$d" root; done
ls root/usr/lib/x86_64-linux-gnu/libcuda.so.580.*   # 确认版本号，若不是 580.178.04，须同步修改 env.sh 和 .env
mkdir -p ~/.local/nvidia/580.178.04
cd root/usr/lib/x86_64-linux-gnu
# 只拷贝 CUDA / OptiX 相关的库，不要拷贝 OpenGL / Vulkan 等图形库
cp -a libcuda.so* libcudadebugger.so* libnvidia-ptxjitcompiler.so* libnvidia-nvvm.so* \
      libnvidia-nvvm70.so* libnvidia-gpucomp.so* libnvoptix.so* libnvidia-rtcore.so* \
      ~/.local/nvidia/580.178.04/
rm -rf /tmp/nv580
```

### 3. 双 GPU 并行

Sionna 的每个进程使用一块 GPU。可以同时开两个进程，分别跑不同的仿真：

```bash
CUDA_VISIBLE_DEVICES=0 python sim_a.py &
CUDA_VISIBLE_DEVICES=1 python sim_b.py &
```

显存不足（OOM）时，减小 `batch_size`。

### 4. 运行教程 notebook

本环境未安装 JupyterLab，`ipykernel` 已包含在 `requirements.txt` 中。在 VS Code 中打开 `.ipynb`，
内核选择 `.venv` 即可运行。

## 复现环境

```bash
git clone --recursive git@github.com:Agata872/Sionna-based-Simulation.git   # --recursive 会一并拉取 sionna 子模块
cd Sionna-based-Simulation
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

然后按注意事项 2 安装 580 用户态库。如果新机器的驱动已经 ≥ 580，则不需要这些库，
删掉 `env.sh` 和 `.env` 中的 `LD_LIBRARY_PATH` 设置即可。

## 参考

- 文档：https://nvlabs.github.io/sionna/
- 源码：https://github.com/NVlabs/sionna
