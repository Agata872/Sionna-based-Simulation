# 数字孪生辅助的 cell-free O-RAN 前传信道信息压缩（第 5–8 轮与第一篇论文）

时间：2026-10-07 ~ 11。接续第 4 轮 [srsbf](../srsbf/README.md)，按用户 2026-10-07 的意见修正方向：O-RU 把 SRS 信道信息（CI）
送往 O-DU 做集中式预编码时，用数字孪生（信道知识图，CKM）合成的区域子空间压缩 CI。方法为 URTS-CODE，最终版即 `urts_vq`（第 7 轮定型）。
代码是独立探索脚本，`oran/` 未修改。

## 各轮记录（`notes/`）与论文

| 轮次 | 记录 | 问题 |
|---|---|---|
| 第 5 轮 | [oran_stage5_twin_ckm_fronthaul_2026-10-07.md](notes/oran_stage5_twin_ckm_fronthaul_2026-10-07.md) | 方向修正；孪生辅助 CI 压缩的可行性（单 O-RU、cell-free） |
| 第 6 轮 | [oran_stage6_tracking_scaling_mismatch_2026-10-08.md](notes/oran_stage6_tracking_scaling_mismatch_2026-10-08.md) | 位置来源（O-DU 闭环定位）、UE 规模与 Gram 基线、慕尼黑内的孪生失配、可变精度量化 |
| 第 7 轮 | [oran_stage7_fair_quantization_final_method_2026-10-08.md](notes/oran_stage7_fair_quantization_final_method_2026-10-08.md) | 所有方案真正量化后的主比较，最终方法定型 |
| 第 8 轮 | [oran_stage8_csirs_compression_survey_2026-10-08.md](notes/oran_stage8_csirs_compression_survey_2026-10-08.md) | CSI-RS 压缩文献调研与第二篇方案（URTS-PORT）。**已暂停（用户 2026-10-08 决定），先完成第一篇** |
| 第一篇 | [paper1/](paper1/outline.md) | 框架 `outline.md`、实验索引 `experiments.md`、理论笔记 `theory_notes.md`、初稿 `manuscript/`、随仓库保存的汇总与图 `evidence/` |

## 文件

第 5–7 轮的脚本也是第一篇的实验入口（命令和运行目录见 [paper1/experiments.md](paper1/experiments.md)）。
`explore_twin_cellfree.py` 是主实验；`explore_twin_tracking.py`、`paper1_*` 和 `diagnose_precision_allocation.py`
从它（以及 `explore_twin_ci_compression.py`）导入场景与编码函数。

| 文件 | 类型 | 内容 |
|---|---|---|
| `explore_twin_ci_compression.py` | 运行 | 第 5 轮：单 O-RU 上孪生（CKM）字典与通用基的 CI 压缩对比 |
| `explore_twin_cellfree.py` | 运行 | 主实验：cell-free 前传上的 CI 压缩，以 O-DU 集中式预编码性能评判；URTS-CODE 等全部方案 |
| `explore_twin_tracking.py` | 运行 | 第 6 轮：UE 位置从哪来——在孪生上做 O-DU 闭环定位，位置后验回送 O-RU |
| `twin_ae.py` | 模型 | 数据驱动基线：在孪生信道上训练的每 O-RU 自编码器 |
| `diagnose_cellfree_leakage.py` | 诊断 | 对 `explore_twin_cellfree.py --dump` 导出的 CI 做泄漏诊断 |
| `diagnose_precision_allocation.py` | 诊断 | 第 6 轮（负结果）：精度分配的几种变体 |
| `summarize_stage5_twin.py` | 汇总 | 第 5–7 轮的汇总与图 |
| `paper1_ae_study.py` | 运行 | 第一篇 E5b：离线调自编码器基线 |
| `paper1_region_dimension.py` | 运行 | 第一篇 E6（理论 T2）：区域孪生子空间的有效维度随位置不确定度、阵列孔径和带宽的变化 |
| `paper1_jobs.sh`、`paper1_jobs2.sh`、`paper1_jobs3.sh` | 运行 | 第一篇的实验队列（单 GPU 依次运行；已完成的跳过）。队列运行时不要修改正在执行的文件 |
| `summarize_paper1.py` | 汇总 | 第一篇的汇总 JSON、清单与图 |
| `paper1_figures.py` | 作图 | 论文插图（矢量 PDF），写入 `paper1/manuscript/figures/` |
| `paper1_scene_figure.py` | 作图 | 论文中的慕尼黑孪生场景图 |

## 结果目录（`results/` 下）

| 目录 | 内容 |
|---|---|
| `stage5_twin_20261007/` | 第 5–7 轮 |
| `paper1_20261008/` | 第一篇全部运行；汇总、清单和图另复制了一份到 `paper1/evidence/` |
