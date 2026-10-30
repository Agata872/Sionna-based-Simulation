# CQI × 压缩实验索引

结果目录：本机为 `results/cqi_compression_20261029/`、`results/cqi_compression_20261030/`（不进 git）。Q5 在 Colab 上写到 Google Drive 的 `MyDrive/sionna_results/cqi_compression_20261029/`；Q7 起写到运行时本地磁盘，经笔记本输出取回本机。每个运行目录都有 metadata.json（参数、git 版本、源码哈希）、源码快照、rows.jsonl、complete.json（含 `recheck_mismatches`、Mitsuba 变体、torch 版本、GPU 型号）。框架见 [README.md](README.md)。

| ID | 目的 | 命令（省略 `python cqi_compression/run_cqi_bler.py`） | 设备 | 运行目录 | 状态 |
|---|---|---|---|---|---|
| Q0 | 冒烟测试：流程、四种 CQI 的行为 | `--device cuda:0 --drops 3 --periods 2 --widths ideal 6 5 4` | 本机 GPU | `smoke/` | 作废：GPU 不可靠。只用于检查代码 |
| Q1 | 正式运行（第一次） | `--device cuda:0 --drops 50` | 本机 GPU | `main.log`（目录已删） | 作废：撒点 0 的 3 bit 处因 GPU 读错显存出现 NaN 而中断 |
| Q2 | CPU 计时 | `--drops 2` | 本机 CPU | `cpu_timing/` | 手动停止（改用 Colab）。7 分 6 秒写入 206 行（约 26 个"位宽 × 4 周期"单元），据此估计完整实验约 3.5 小时 |
| Q3 | 检查运行中复核与汇总脚本 | `--drops 1 --periods 1 --widths ideal 5 --categories A` | 本机 CPU | `cpu_check/` | 完成：`recheck_mismatches` = 0 |
| Q4 | 检查 `--rt-llvm`（链路 GPU、射线追踪 CPU） | `--device cuda:0 --rt-llvm --drops 1 --periods 1 --widths ideal --categories A` | 本机 GPU + LLVM | `rtllvm_check/` | 完成：Mitsuba 变体为 llvm_ad_mono_polarized |
| Q5 | 正式运行 | `--device cuda:0 --drops 50`（按 `colab_continue.ipynb`） | Colab A100-SXM4-80GB，torch 2.11.0+cu130，射线追踪 cuda_ad_mono_polarized | Drive 上的 `main/`；本机 `colab_main/`（从笔记本输出提取的报告、日志、图，无 rows.jsonl） | 完成 2026-10-29，约 14.5 分钟。GPU 自检两项 PASS，日志中无复核不一致。结论见 README"结果" |
| Q6 | 冒烟测试：加入 split option 8 后的流程、笔记本打包与本机解包 | `--drops 2 --periods 1 --widths ideal 5 --o8-widths ideal 8 5`（`--categories` 默认 A B 8） | 另一台电脑的 WSL Ubuntu 22.04 CPU，torch 2.15.0+cpu，射线追踪 llvm_ad_mono_polarized（LLVM 18） | `cqi_compression_20261030/smoke_local/` | 完成 2026-10-30，约 4 分钟，112 行，`recheck_mismatches` = 0。见下 |
| Q7 | 正式运行：Cat A、Cat B、option 8 同一批撒点 | `--device cuda:0 --drops 50 --categories A B 8`（按 `colab_continue.ipynb`） | Colab GPU | 运行时 `/content/results/cqi_compression_20261030/main/`，经笔记本输出取回本机 `cqi_compression_20261030/colab_main/` | 待运行 |

## Q5 的登记

- 运行中复核：脚本只在 `recheck_mismatches` 不为 0 时打印警告，日志中没有警告，也没有任何一行标注"复核不一致"。`complete.json` 在 Drive 上，下载后可再核对。
- 汇总：200 个 UE-撒点中排除 26 个（理想前传下精灵 CQI 也失败）。各 Cat、各位宽下四种 CQI 的 BLER、失败 UE 比例、平均吞吐、平均 MCS 见 `colab_main/report.txt`，主要数字与结论见 [README.md](README.md) 的"结果"。
- 冒烟测试（`--drops 2 --periods 1 --widths ideal 5`）同在 Colab 上运行，日志为 `colab_main/smoke_colab.log`。

## Q6 的登记

- 环境：这台电脑上没有项目的 `.venv`，在 WSL（Ubuntu 22.04）中用 uv 另建了 `/root/sionna-venv`（Python 3.12、torch 2.15.0+cpu、sionna / sionna-rt 2.2.0），不在仓库里。系统自带的 LLVM 15 编译 Dr.Jit 1.5 生成的代码时报 `LLVM ERROR: Cannot select ... fminimum`，改装 apt.llvm.org 的 LLVM 18，并设 `DRJIT_LIBLLVM_PATH=/usr/lib/x86_64-linux-gnu/libLLVM-18.so.1` 后正常。
- Cat A / B 在第一个撒点上的 5 bit SQNR 与 Q0 的记录基本相同：Cat A 块缩放 27.5 dB、BFP 25.3 dB、μ 律 25.1 dB（Q0 为 27.5、25.2、25.0）；Cat B 28.5、27.4、23.8 dB（与 Q0 相同）。CPU 与 GPU 撒出的位置不同，不能逐位比较。
- option 8 的理想前传与 Cat A 的理想前传逐项相同（CQI、MCS、均衡后 SINR）。均匀定点 8 bit：PRB 上 SQNR 42.6 dB，时域 40.3 dB，相差 2.3 dB（理论值 10·log10(1024/612) = 2.2 dB）。时域 SQNR 比高斯分量最优值（40.6 dB）低 0.3 dB。5 bit：26.6 / 24.4 dB，与 Cat A 块缩放 5 bit 的 27.5 dB 接近，但前传比特率为 2.55 Gbps，Cat A 为 1.58 Gbps。
- 作对照的时域分块算法（每块 16 个样点）在第一个撒点上 8 bit 时：块缩放 47.6 dB、μ 律 45.4 dB、BFP 44.8 dB，都高于均匀定点的 42.6 dB；但每个样点多出 0.5 bit 的压缩参数。
- CPU 耗时：每个撒点每个位宽约 15–25 秒，option 8 与 Cat A 相当。
- 笔记本第 7 单元的打包代码在本机对 `smoke_local/` 执行，输出写进笔记本副本，再用 `extract_colab_results.py` 解压：rows.jsonl 与原文件逐行相同，sha256 校验通过，`analyse_cqi_bler.py` 能在 Windows 的 Python 上汇总并画出两张图。
