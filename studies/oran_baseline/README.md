# O-RAN 7.2x 仿真器基线：Cat A/B、IQ 压缩与射线追踪闭环

时间：2026-10-04 ~ 05；2026-11-05 补充 cell-free 下四种功能切分的前传负载对比（实验索引见 [experiments.md](experiments.md)）。验证 `oran/` 仿真器本身，并给出后续各方向沿用的基线设置。仿真器的设计说明、默认配置、
运行命令和这些实验的结论见 [oran/README.md](../../oran/README.md)。

**要回答的问题**
- 固定 MCS 时，Cat A（预编码在 O-DU）与 Cat B（预编码在 O-RU）的 BLER、EVM 随 IQ 压缩位宽和 SNR 怎样变化；BFP、块缩放、μ 律之间差多少。
- 射线追踪闭环（CSI-RS → CQI → MCS + OLLA）中，压缩失真有多少能被 CSI-RS 测量看到，有多少只能靠 OLLA 吸收。
- cell-free（TR 38.901 UMi 撒点）中，option 7-1、7.2x Cat A、7.2x Cat B、option 8 在各自的压缩方案下前传负载多大，
  CQI 查表选 MCS 时哪些压缩不损失吞吐（F1）。

## 文件

| 文件 | 类型 | 内容 |
|---|---|---|
| `compare_cat_a_b.py` | 运行 + 汇总 | Cat A/B × 压缩位宽 × SNR 扫描；可在两块 GPU 上分开跑，再用 `--merge` 合并 |
| `rt_closed_loop.py` | 运行 | 射线追踪闭环：随机撒点 → CSI-RS → CQI → MCS（+ OLLA）→ PDSCH → CRC |
| `rt_compare_compression.py` | 运行 + 汇总 | 闭环中比较 Cat A/B 的压缩失真：CQI 看到多少、OLLA 补了多少；配对撒点 |
| `closed_loop_walkthrough.ipynb` | 演示 | 按信号流向逐模块演示闭环仿真，每一步画出中间结果 |
| `split_fronthaul_load.ipynb` | 运行 + 汇总 | F1：cell-free UMi 撒点，四种切分 × 压缩方案（含 Cat B 调制压缩），CQI 查表选 MCS；负载与吞吐、最小负载、UE 星座图、负载随天线数与 PRB 占用率的变化 |
| `experiments.md` | 索引 | 2026-11 之后的实验登记（目的、设置、运行目录、结论） |

`rt_closed_loop.py` 与 `rt_compare_compression.py` 复用 `compare_cat_a_b.py` 中的压缩配置解析。

## 结果目录（`results/` 下）

| 目录 | 来源 |
|---|---|
| `cat_a`、`cat_b`、`cat_ab` | `compare_cat_a_b.py`，两块 GPU 分跑后合并 |
| `methods_a`、`methods_b`、`methods` | `compare_cat_a_b.py`，比较压缩算法 |
| `rt_su`、`rt_mu` | `rt_closed_loop.py` |
| `csi_su_a`、`csi_su_b`、`csi_su`、`csi_mu_a`、`csi_mu_b`、`csi_mu`、`drops_su.json`、`drops_mu.json` | `rt_compare_compression.py`；`drops_*.json` 是两块 GPU 共用的撒点 |
| `split_fronthaul_load_<日期>/run_<时间>_drops<N>_seed<S>/` | `split_fronthaul_load.ipynb`（F1）：metadata.json、源码快照、rows.jsonl、complete.json（含汇总）、fig_*.png |
