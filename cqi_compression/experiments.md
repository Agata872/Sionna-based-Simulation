# CQI × 压缩实验索引

结果目录：本机为 `results/cqi_compression_20261029/`（不进 git），Colab 上为 Google Drive 的 `MyDrive/sionna_results/cqi_compression_20261029/`。每个运行目录都有 metadata.json（参数、git 版本、源码哈希）、源码快照、rows.jsonl、complete.json（含 `recheck_mismatches`、Mitsuba 变体、torch 版本、GPU 型号）。框架见 [README.md](README.md)。

| ID | 目的 | 命令（省略 `python cqi_compression/run_cqi_bler.py`） | 设备 | 运行目录 | 状态 |
|---|---|---|---|---|---|
| Q0 | 冒烟测试：流程、四种 CQI 的行为 | `--device cuda:0 --drops 3 --periods 2 --widths ideal 6 5 4` | 本机 GPU | `smoke/` | 作废：GPU 不可靠。只用于检查代码 |
| Q1 | 正式运行（第一次） | `--device cuda:0 --drops 50` | 本机 GPU | `main.log`（目录已删） | 作废：撒点 0 的 3 bit 处因 GPU 读错显存出现 NaN 而中断 |
| Q2 | CPU 计时 | `--drops 2` | 本机 CPU | `cpu_timing/` | 手动停止（改用 Colab）。7 分 6 秒写入 206 行（约 26 个"位宽 × 4 周期"单元），据此估计完整实验约 3.5 小时 |
| Q3 | 检查运行中复核与汇总脚本 | `--drops 1 --periods 1 --widths ideal 5 --categories A` | 本机 CPU | `cpu_check/` | 完成：`recheck_mismatches` = 0 |
| Q4 | 检查 `--rt-llvm`（链路 GPU、射线追踪 CPU） | `--device cuda:0 --rt-llvm --drops 1 --periods 1 --widths ideal --categories A` | 本机 GPU + LLVM | `rtllvm_check/` | 完成：Mitsuba 变体为 llvm_ad_mono_polarized |
| Q5 | 正式运行 | `--device cuda:0 --drops 50`（按 `colab_continue.ipynb`） | Colab A100-SXM4-80GB，torch 2.11.0+cu130，射线追踪 cuda_ad_mono_polarized | Drive 上的 `main/`；本机 `colab_main/`（从笔记本输出提取的报告、日志、图，无 rows.jsonl） | 完成 2026-10-29，约 14.5 分钟。GPU 自检两项 PASS，日志中无复核不一致。结论见 README"结果" |

## Q5 的登记

- 运行中复核：脚本只在 `recheck_mismatches` 不为 0 时打印警告，日志中没有警告，也没有任何一行标注"复核不一致"。`complete.json` 在 Drive 上，下载后可再核对。
- 汇总：200 个 UE-撒点中排除 26 个（理想前传下精灵 CQI 也失败）。各 Cat、各位宽下四种 CQI 的 BLER、失败 UE 比例、平均吞吐、平均 MCS 见 `colab_main/report.txt`，主要数字与结论见 [README.md](README.md) 的"结果"。
- 冒烟测试（`--drops 2 --periods 1 --widths ideal 5`）同在 Colab 上运行，日志为 `colab_main/smoke_colab.log`。
