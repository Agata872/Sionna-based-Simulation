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

第一级放各方向共用的东西（标准文档、参考论文、仿真库、工具函数、测试），各研究方向放在 `studies/` 下，
每个方向自带脚本、验证和研究记录。

```
Sionna-based Simulation/
├── 3GPP and ORAN report/   # 3GPP / O-RAN 标准与规范 PDF（不纳入 git）
├── Paper reference/        # 参考论文 PDF（不纳入 git）
├── oran/                   # 公共仿真库：O-RAN 7.2x 下行 Cat A / Cat B、3GPP split 7-1 / 8 链路级仿真、RT 与 UMi 撒点闭环（说明见 oran/README.md）
│   └── compression/        #   前传 IQ 压缩算法（BFP、块缩放、μ 律、调制压缩）
├── common/                 # 各方向共用的工具函数
│   └── runio.py            #   运行目录记录：Run（自动写 metadata 与源码快照）、Recorder、read_rows
├── tests/                  # oran/ 的单元测试：python -m unittest discover -s tests -v
├── studies/                # 各研究方向（见下节"研究方向"）
│   ├── oran_baseline/
│   ├── csirs_du_ru/
│   ├── srsbf/
│   ├── twin_ci_compression/
│   └── dynamic_split/
├── results/                # 仿真输出（不纳入 git），各方向对应哪些子目录见该方向的 README
├── sionna/                 # 官方仓库 v2.2.0 源码（git 子模块，含 sionna-rt 子模块）
│   └── tutorials/          #   phy/ 链路级、sys/ 系统级、rt/ 射线追踪 教程
├── .venv/                  # Python 虚拟环境，所有依赖都在这里（不纳入 git）
├── env.sh                  # 激活环境（终端用）
├── .env                    # VS Code 启动 Python / Jupyter 内核时读取的环境变量
├── verify_env.py           # 环境验证脚本
├── requirements.txt        # 锁定的依赖版本
└── README.md

~/.local/nvidia/580.178.04/   # 580 驱动的用户态库（不纳入 git，安装方法见注意事项 2）
```

## 研究方向

按时间顺序。每个方向的 README 写明要回答的问题、文件表（运行 / 汇总 / 诊断 / 验证）和对应的 `results/` 子目录，
各轮的研究记录在该方向的 `notes/` 中。

| 方向 | 时间 | 内容 | 状态 |
|---|---|---|---|
| [oran_baseline](studies/oran_baseline/README.md) | 2026-10-04 ~ 05；2026-11-05 | `oran/` 仿真器的基线验证：Cat A/B × 压缩位宽 × SNR；射线追踪闭环中 CSI-RS / CQI / OLLA 能看到多少压缩失真；cell-free UMi 撒点中 7-1 / Cat A / Cat B / option 8 的前传负载（F1） | 完成；F1 见 experiments.md |
| [csirs_du_ru](studies/csirs_du_ru/README.md) | 2026-10-06 | 调研与第 1–3 轮：研究现状；CSI-RS 压缩与 DU–RU 预编码协作（权值表示、信息时效、协作消息预算、RU 自主预编码后的 MCS 决策） | 结束，问题转入第 4 轮 |
| [srsbf](studies/srsbf/README.md) | 2026-10-07 | 第 4 轮：O-RU 侧 SRS 波束赋形下 O-DU 的多用户调度与链路自适应 | 方向已修正（把 O-RAN 接口本身当成了研究对象），由第 5 轮取代 |
| [twin_ci_compression](studies/twin_ci_compression/README.md) | 2026-10-07 ~ 11 | 第 5–8 轮与第一篇论文：数字孪生（信道知识图）辅助的 cell-free 前传信道信息压缩（URTS-CODE）；第 8 轮是 CSI-RS 压缩调研和第二篇方案 | 第一篇有初稿；第二篇暂停 |
| [dynamic_split](studies/dynamic_split/README.md) | 2026-10-12 起 | Girycki 等（EuCNC 2024）的扩展：动态切分 + 压缩 + CI；期刊扩展的验证实验在其 `cqi_compression/` 下 | 阶段 0–14 完成；期刊扩展验证 V-a 完成，V-b、V-c 未开始 |

**约定**
- 脚本都从仓库根目录运行：`python studies/<方向>/<脚本>.py`。脚本自己把仓库根目录加入 `sys.path`，
  以导入 `oran/` 与 `common/`；同一方向内的脚本按模块名互相导入。
- 每次运行写到 `results/` 下带日期的目录，保存 metadata、源码快照、逐行结果和日志（`common/runio.py`），
  并登记在该方向的实验索引中（`experiments.md`；早期方向登记在 `notes/` 的轮次记录里）。
- 新方向：在 `studies/` 下新建文件夹，放 `README.md`、`notes/`（研究记录）和脚本；
  被两个以上方向用到的函数移到 `common/`。

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
包括 Category A（预编码在 O-DU）、Category B（预编码在 O-RU）和 3GPP split option 8（时域 IQ），
以及前传 IQ 压缩（BFP / 块缩放 / μ 律）。
闭环模式用 Sionna RT 对物理场景（默认 munich）射线追踪，UE 随机撒点，按 CQI 选 MCS（含 OLLA）后下行传输。
设计、配置与扩展方法见 [oran/README.md](oran/README.md)，逐模块演示见
[studies/oran_baseline/closed_loop_walkthrough.ipynb](studies/oran_baseline/closed_loop_walkthrough.ipynb)。

```bash
source env.sh
python -m unittest discover -s tests -v              # 单元测试
python studies/oran_baseline/compare_cat_a_b.py      # Cat A/B 对比，结果写入 results/cat_ab/
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

### 5. WSL2（Windows 笔记本）

已在 WSL2 Ubuntu 24.04 + RTX 4060 Laptop（Windows 驱动 617.42，WSL 内显示为 615.78.05）上验证，
`verify_env.py` 全部通过，GPU 与 CPU 射线追踪结果最大差 0.009 dB。与服务器的差别：

- **虚拟环境放在 Linux 文件系统**：项目位于 `/mnt/d`（Windows 盘），在上面装约 7 GB 的依赖和 import 都很慢。
  虚拟环境建在 `~/.venvs/sionna-sim`，项目中的 `.venv` 是指向它的符号链接，`env.sh` 和 VS Code 照常使用：

  ```bash
  python3 -m venv ~/.venvs/sionna-sim
  ln -s ~/.venvs/sionna-sim .venv
  .venv/bin/python -m pip install -r requirements.txt
  ```

- **不需要 580 用户态库**：libcuda 由 Windows 驱动提供（`/usr/lib/wsl/lib`），版本已 ≥ 580。
- **OptiX 需另行提供**：Windows 驱动不向 WSL 提供可用的 OptiX，Sionna RT 默认选 GPU 后端，
  `load_scene` 会报 `Could not initialize OptiX!`。做法是从版本最接近的 Linux 驱动中只提取 OptiX 相关的库
  （不需要 sudo，不修改 Windows），`env.sh` 在 WSL 下自动把该目录加入 `LD_LIBRARY_PATH`，`.env` 同理：

  ```bash
  mkdir -p /tmp/nvwsl && cd /tmp/nvwsl
  V=615.78.08   # 与 WSL 中 nvidia-smi 显示的版本最接近的已发布版本；若更改，须同步修改 env.sh 和 .env
  curl -fLO https://download.nvidia.com/XFree86/Linux-x86_64/$V/NVIDIA-Linux-x86_64-$V-no-compat32.run
  sh NVIDIA-Linux-x86_64-$V-no-compat32.run -x --target driver   # 只解压，不安装
  mkdir -p ~/.local/nvidia/wsl-optix-$V
  cp -a driver/libnvoptix.so.$V driver/libnvidia-rtcore.so.$V driver/libnvidia-gpucomp.so.$V \
        driver/nvoptix.bin ~/.local/nvidia/wsl-optix-$V/
  ln -s libnvoptix.so.$V ~/.local/nvidia/wsl-optix-$V/libnvoptix.so.1
  rm -rf /tmp/nvwsl
  ```

  不要拷贝 libcuda 等其他库，否则会覆盖 Windows 驱动提供的版本，GPU 将不可用。
  Windows 驱动升级后若出现 OptiX 报错，按新的版本号重新提取。
- **换行符**：在 Windows 下检出时文件会被转成 CRLF，bash 无法 `source` 这样的 `env.sh`。
  `.gitattributes` 已规定 `*.sh` 和 `.env` 始终为 LF。
- **中文字体**：notebook 和部分脚本的图用 `Droid Sans Fallback` 显示中文，缺少时会反复报
  `findfont: Font family 'Droid Sans Fallback' not found`，中文显示为方框。不需要 sudo 的安装方法：

  ```bash
  mkdir -p /tmp/font && cd /tmp/font
  apt-get download fonts-droid-fallback && dpkg-deb -x fonts-droid-fallback_*.deb root
  mkdir -p ~/.local/share/fonts && cp root/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf ~/.local/share/fonts/
  rm -rf /tmp/font ~/.cache/matplotlib/fontlist-*.json   # 删除 matplotlib 字体缓存，下次 import 时重建
  ```

  装好后须重启 Jupyter 内核。
- **GPU 计算不可靠（2026-10-29 发现）**：这台 RTX 4060 Laptop 在持续负载下会间歇性读错显存，
  表现为 Sionna 的 LDPC 译码在相同输入、相同种子下结果不同，以及下标越界断言或 `illegal instruction`。
  不经过 Sionna 的纯 PyTorch 循环也能复现，同一脚本在 CPU 上完全一致；自检有时也会碰巧通过。
  在这台笔记本上请用 CPU 跑链路级仿真（`CUDA_VISIBLE_DEVICES=`），或改用 Colab / 服务器。
  换新机器后先运行 `python studies/dynamic_split/cqi_compression/gpu_selfcheck.py`，详见 [studies/dynamic_split/cqi_compression/README.md](studies/dynamic_split/cqi_compression/README.md)。
  此前在这台笔记本 GPU 上得到的链路级结果（包括重新运行的 `studies/oran_baseline/closed_loop_walkthrough.ipynb`）需要复核。

## 复现环境

```bash
git clone --recursive git@github.com:Agata872/Sionna-based-Simulation.git   # --recursive 会一并拉取 sionna 子模块
cd Sionna-based-Simulation
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

然后按注意事项 2 安装 580 用户态库。如果新机器的驱动已经 ≥ 580，则不需要这些库，
删掉 `env.sh` 和 `.env` 中的 `LD_LIBRARY_PATH` 设置即可。WSL2 下的做法见注意事项 5。

## 参考

- 文档：https://nvlabs.github.io/sionna/
- 源码：https://github.com/NVlabs/sionna
