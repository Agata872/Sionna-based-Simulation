# CSI-RS 压缩与 DU–RU 预编码协作（调研与第 1–3 轮探索）

时间：2026-10-06。从研究现状调研出发，在 `oran/` 之上用独立脚本做配对实验（`oran/` 未修改）。
后续见第 4 轮 [srsbf](../srsbf/README.md)。

## 各轮记录（`notes/`）

| 轮次 | 记录 | 问题 |
|---|---|---|
| 调研 | [oran_soa_2026-10-06.md](notes/oran_soa_2026-10-06.md) | 前传压缩与 DU–RU 协作的研究现状、空白与选题 |
| 路线 | [oran_csirs_precoding_roadmap_2026-10-06.md](notes/oran_csirs_precoding_roadmap_2026-10-06.md) | 有限前传下 DU 与 RU 应共享什么信息、RU 可以怎样更新预编码 |
| 第 1 轮 | [oran_experiment_log_2026-10-06.md](notes/oran_experiment_log_2026-10-06.md) | CSI-RS IQ 压缩、权值表示、双 RU 信息时效 |
| 第 2 轮 | [oran_stage2_followup_2026-10-06.md](notes/oran_stage2_followup_2026-10-06.md) | 强基线、双向消息预算、实际链路自适应 |
| 第 3 轮 | [oran_stage3_gap_2026-10-06.md](notes/oran_stage3_gap_2026-10-06.md) | RU 自主预编码后的质量信息是否足以支持 MCS 决策 |

各轮的结论、决定和实验命令都在对应记录中。

## 文件

`explore_csirs_precoding.py` 是本方向的公共入口，多数运行脚本从它导入 `Recorder`（定义已移到 `common/runio.py`，这里只是转导入），
`explore_compression_closed_loop.py` 还用它的 `weight_cases`；汇总和诊断脚本复用 `summarize_csirs_exploration.py` 中的 bootstrap 置信区间 `ci_mean`。

| 轮次 | 文件 | 类型 | 内容 |
|---|---|---|---|
| 1 | `explore_csirs_precoding.py` | 运行 | 配对实验：CSI-RS 整符号 IQ 量化与权值量化（数值模式），以及用真实 DMRS 接收机验证数据权值的影响（CRC 模式） |
| 1 | `explore_du_ru_information.py` | 诊断 | 双 RU 中信息时效与权值精度的关系 |
| 1 | `summarize_csirs_exploration.py` | 汇总 | 按撒点聚类的 bootstrap 置信区间与图 |
| 1 | `summarize_du_ru_information.py` | 汇总 | 双 RU 信息时效诊断的汇总 |
| 2 | `explore_coordination_budget.py` | 运行 | 独立复现 Team-MMSE 方程，有限消息位数的筛查 |
| 2 | `diagnose_coordination_quantization.py` | 诊断 | 区分统计量量化与系数量化各自造成的失败 |
| 2 | `explore_compression_closed_loop.py` | 验证 | 静态 CDL 重复时隙：固定 MCS 下的压缩收益在链路自适应后是否还在 |
| 2 | `combine_closed_loop_exploration.py` | 汇总 | 检查设置一致后合并独立的闭环批次 |
| 2 | `compare_closed_loop_measurements.py` | 汇总 | CSI-RS 表示匹配 / 失配的配对比较，含极限检查 |
| 2 | `summarize_next_exploration.py` | 汇总 | 协作预算与实际链路自适应实验的汇总 |
| 3 | `explore_dynamic_measurement.py` | 验证 | 因果的双 RU 连续 CDL：导频 / 预编码 / 链路自适应的证伪实验 |
| 3 | `explore_receiver_certificates.py` | 运行 | 接收端投影的 RU 消息与完整有效信道上报的对比筛查 |
| 3 | `audit_dynamic_information.py` | 验证 | 信息反例与有向线速率审计 |
| 3 | `summarize_dynamic_exploration.py` | 汇总 | 按撒点聚类的汇总、配对区间、图与可复现性审计 |

## 结果目录（`results/` 下）

| 目录 | 内容 |
|---|---|
| `exploration_20261006/` | 第 1–2 轮（`cdl_c_*`、`compression_loop_*`、`coordination_*` 等） |
| `exploration_20261007/` | 第 3 轮（`causal_*`、`dynamic_*`、`receiver_*`、`information_*` 等） |
