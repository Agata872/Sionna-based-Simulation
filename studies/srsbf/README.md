# O-RU 侧 SRS 波束赋形下 O-DU 的多用户调度与链路自适应（第 4 轮）

时间：2026-10-07。接续第 3 轮 [csirs_du_ru](../csirs_du_ru/README.md)，把问题落到 O-RAN CUS v22.00 的 SRS-BF-DL 上。

> **方向已修正。** 用户 2026-10-07 判断本轮把 O-RAN 接口本身当成了研究对象，偏离了"cell-free 前传压缩与 DU–RU 协作"的主线，
> 现行选题见第 5 轮 [twin_ci_compression](../twin_ci_compression/README.md)。本轮代码和结果保留。

记录：[notes/oran_stage4_srsbf_convergence_2026-10-07.md](notes/oran_stage4_srsbf_convergence_2026-10-07.md)（问题、模型假设、结果、实验命令）。

## 文件

| 文件 | 类型 | 内容 |
|---|---|---|
| `srsbf_core.py` | 模型 | SRS-BF-DL MU-MIMO 的抽象层模型（假设列表见 `ASSUMPTIONS`） |
| `explore_srsbf_scheduling.py` | 运行 | O-DU 的 PF 调度 + 链路自适应，比较各种 O-DU 信息策略（同一批撒点、信道、ICI 和 ACK 种子） |
| `diagnose_srsbf_prediction.py` | 诊断 | 分解 O-DU 的 MU-SINR 预测误差 |
| `validate_srsbf_abstraction.py` | 验证 | 用 `oran/` 的 NR 链路（PDSCH、DMRS LS 估计、LMMSE、LDPC、CRC）检验抽象层模型 |
| `verify_srsbf_identities.py` | 验证 | 两个解析构件的数值检查 |
| `srsbf_fronthaul_cost.py` | 分析 | 各 SRS-BF-DL 选项的 O-RU → O-DU 信息开销（CUS v22.00 字段长度） |
| `summarize_srsbf.py` | 汇总 | 按撒点聚类的配对 bootstrap 汇总 |
| `paired_srsbf.py` | 汇总 | 各策略之间的配对差（同种子、同撒点，可跨运行） |
| `plot_stage4_srsbf.py` | 作图 | 本轮记录中的图 |

## 结果目录（`results/` 下）

`stage4_srsbf_20261007/`
