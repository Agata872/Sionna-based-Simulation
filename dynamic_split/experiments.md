# 动态切分实验索引

结果目录：`results/dynamic_split_20261012/`（不进 git）。每个运行目录都有 `metadata.json`（参数、假设、git 版本、源码哈希）、`source_*.py`（源码快照）、`rows.jsonl`、`complete.json`，以及同名 `.log`。框架见 [README.md](README.md)。

## 实验清单

| ID | 目的 | 命令 | 运行目录 | 状态 |
|---|---|---|---|---|
| S0 | 复现 Girycki 等图 3–5（服务天线曲线、BF 选择、论文口径负载） | `.venv/bin/python dynamic_split/run_stage0.py --out results/dynamic_split_20261012/stage0_repro`，再 `summarize_stage0.py <运行目录>` | `stage0_repro/` | 完成 2026-10-12 |
| S1a | 只看负载的筛查：显式报头、BFP9、调制压缩、CI、权值对动态收益的影响 | `.venv/bin/python dynamic_split/screen_stage1a.py results/dynamic_split_20261012/stage0_repro --out results/dynamic_split_20261012/stage1a_load_screen` | `stage1a_load_screen/` | 完成 2026-10-12 |
| S1b | 真实编解码器的失真计入和速率；峰值（链路）约束下最小化平均负载，比较静态 / 动态位宽 / 动态切分 / 联合动态 | `.venv/bin/python dynamic_split/run_stage1b.py results/dynamic_split_20261012/stage0_repro --out results/dynamic_split_20261012/stage1b_distortion`，再 `analyse_stage1b.py <运行目录>` | `stage1b_distortion/` | 完成 2026-10-12 |
| S3 | 方向 ②+③：在论文场景上联合选服务集合、每 RU 切分（7-1 / 7-2）与压缩；P1 负载最小（κ = 0.98），P2 给定前传容量最大化和速率；逐撒点与逐状态两种粒度；对照 B0 论文、B1 论文两步法 + 统一压缩、B2 论文服务集合 + 逐 RU 选择 | `.venv/bin/python dynamic_split/run_stage3.py results/dynamic_split_20261012/stage0_repro --out results/dynamic_split_20261012/stage3_joint`，再 `analyse_stage3.py <运行目录>` | `stage3_joint/` | 完成 2026-10-12 |
| S4 | 方向 1：检验 `theory_notes.md` 的命题 2 / 推论 2（每 RU 的切分门限 n*）能否预测阶段 3 的逐 RU 选择；比较 P 与 B2 的服务集合结构 | `.venv/bin/python dynamic_split/explore_rule.py results/dynamic_split_20261012/stage3_joint results/dynamic_split_20261012/stage0_repro --out results/dynamic_split_20261012/stage4_rule --drops 30` | `stage4_rule/` | 完成 2026-10-13 |
| S4b | 服务集合路径消融：论文顺序 / 只看和速率增量的贪心 / 负载感知贪心，后接同样的逐 RU 拉格朗日选择；逐撒点 P1 负载（MCS） | `.venv/bin/python dynamic_split/ablate_path.py results/dynamic_split_20261012/stage0_repro --out results/dynamic_split_20261012/stage4_path_ablation` | `stage4_path_ablation/` | 完成 2026-10-13（结果行由 npz 补写，见 complete.json） |
| S4c | 闭式策略 CF（和速率贪心服务集合 + 全网一个位宽 + 命题 2 的逐 RU 切分）对比逐 RU 拉格朗日优化器 | `.venv/bin/python dynamic_split/explore_cf.py results/dynamic_split_20261012/stage0_repro results/dynamic_split_20261012/stage4_path_ablation --out results/dynamic_split_20261012/stage4_cf` | `stage4_cf/` | 完成 2026-10-13 |
| S5 | 按 [10] 忠实实现服务 AP 选择（MRT 算法 1，ZF 算法 2 + EuCNC 的加入规则）：A 复现论文图 5；B 与近似论文路径、和速率贪心比较逐撒点 P1 负载 | `.venv/bin/python dynamic_split/compare_paths10.py results/dynamic_split_20261012/stage0_repro results/dynamic_split_20261012/stage4_path_ablation --out results/dynamic_split_20261012/stage5_paths10` | `stage5_paths10/` | 完成 2026-10-13 |
| S6 | 命题 3 检验：逐 RU 位宽注水（CF-WF）对比全网一个位宽（CF-U）与逐 RU 拉格朗日优化器（OPT），服务集合用 [10] 与和速率贪心两种 | `.venv/bin/python dynamic_split/explore_prop3.py results/dynamic_split_20261012/stage0_repro --out results/dynamic_split_20261012/stage6_prop3` | `stage6_prop3/` | 完成 2026-10-13 |
| S7 | 方向 3：用三步闭式方法（[10] 服务集合 + 命题 3 位宽 + 命题 2 切分）分解动态调整的价值：逐撒点 / 逐状态 / 全局固定位宽 / 固定切分 / 全网统一配置；SE_max 改为 [10] 路径自身的最大值 | `.venv/bin/python dynamic_split/run_stage7.py results/dynamic_split_20261012/stage0_repro --out results/dynamic_split_20261012/stage7_dynamics`，再 `analyse_stage7.py <运行目录>` | `stage7_dynamics/` | 完成 2026-10-13 |
| S8 | 方向 3 的理论边界：由命题 2 预测动态切分失效的 K*（= max_ZF N n*/M̂），并用服务集合统计预测各状态选 7-2 的 RU 比例，对照 S7 | `.venv/bin/python dynamic_split/predict_boundary.py results/dynamic_split_20261012/stage7_dynamics --out results/dynamic_split_20261012/stage8_boundary` | `stage8_boundary/` | 完成 2026-10-13 |
| S9 | 复现差距检验：MRT 在 −50 dBm 偏低约 30% 是否来自 [10] 的噪声模型（对数正态、4.8 dB、线性均值）或宽带 SINR 定义（式 (6)）；Shannon，[10] 的启发式 | `.venv/bin/python dynamic_split/check_reproduction.py --out results/dynamic_split_20261012/stage9_reproduction` | `stage9_reproduction/`（第一版 `stage9_reproduction_v0/` 只含前 6 个变体） | 完成 2026-10-13，结论见 theory_notes 第 10 节 |
| S10 | 忠实模型（`--model paper10`：每 UE 对数正态噪声 4.8 dB + [10] 式 (18) 的 ZF）下重做 S7 的动态分解；BF 按论文规则重选 | `.venv/bin/python dynamic_split/run_stage7.py results/dynamic_split_20261012/stage0_repro --out results/dynamic_split_20261012/stage10_paper10 --model paper10 --se-model mcs`，再 `analyse_stage7.py` | `stage10_paper10/`（第一次运行因 SE_max = 0 的撒点崩溃，日志 `stage10_paper10_crash.log`） | 完成 2026-10-13 |
| S11 | 1024QAM 敏感性：忠实模型 + MCS 表 4（`--se-model mcs4`），同 S10 | `.venv/bin/python dynamic_split/run_stage7.py results/dynamic_split_20261012/stage0_repro --out results/dynamic_split_20261012/stage11_paper10_mcs4 --model paper10 --se-model mcs4`，再 `analyse_stage7.py` | `stage11_paper10_mcs4/` | 完成 2026-10-13 |
| S12 | 忠实模型下检验命题 3（CF-WF 对比全网一个位宽与逐 RU 拉格朗日优化器），MCS 表 2 与表 4 | `.venv/bin/python dynamic_split/explore_prop3.py results/dynamic_split_20261012/stage0_repro --out results/dynamic_split_20261012/stage12_prop3_paper10 --model paper10 --se-model mcs`（表 4：`--se-model mcs4`，目录 `stage12_prop3_paper10_mcs4/`） | `stage12_prop3_paper10*/` | 完成 2026-10-13 |
| S13 | 忠实模型下的边界预测（S8 方法）与真实编解码器复核（validate_cf），针对 S10 与 S11 | `predict_boundary.py <S10/S11 目录> --out results/dynamic_split_20261012/stage13_boundary_{paper10,paper10_mcs4}`；`validate_cf.py <S10/S11 目录> --drops 20` | `stage13_boundary_*/`、S10/S11 目录内的 `validation_*` | 完成 2026-10-13 |
| S14 | 论文插图数据（忠实口径、MCS 表 2）：命题 1 的 SINR 上限（真实编解码器）、命题 2 的门限阶跃与逐 RU 一致率、命题 3 的位宽斜率、动态分解、边界、P2 容量曲线（含论文不压缩方法） | `.venv/bin/python dynamic_split/fig_data.py --out results/dynamic_split_20261012/stage14_figdata --copy-to /home/user/Dynamic-Compression-With-Fronthaul-Load-Estimation/figures/data` | `stage14_figdata/` | 完成 2026-10-13（命题 3 的图数据由 `fig_prop3_post.py` 后处理） |
| S2a（旁支，已偏离论文框架） | 双 O-RU（各 32 阵元）、简化 UMi 簇信道（不用 Sionna RT）：集中式（7-1 / 7-2）与 O-RU 本地预编码、位宽、CI 方案（天线级 SRS IQ、ST 6 + BFP、URTS-CODE 代理）的联合选择；状态含 SRS 周期 | `.venv/bin/python dynamic_split/run_stage2a.py --out results/dynamic_split_20261012/stage2a_two_ru`，再 `analyse_stage2a.py <运行目录>` | `stage2a_two_ru/`（第一版 `stage2a_two_ru_v0_equalpower/` 作废） | 完成 2026-10-12 |

## S0：复现

- 设置：100 个撒点 × UE 数 {5, 10, 20} × 噪声 {−90, −70, −50} dBm/PRB × {ZF, MRT} × {shannon, mcs}，种子 2024，M.2135 UMi 路损。
- 输出：`rows.jsonl`（每行一条逐根考察曲线、天线顺序、是否加入、MRT 的 UE 分配），`per_ue_se.npz`（每步的每 UE 频谱效率），`stage0_summary.json`，`stage0_table.txt`（与论文图 5 对照），`fig_load_vs_k.png`。
- 结论（shannon 口径，对照论文）：
  - 9 个状态的波束赋形选择全部一致。
  - ZF 和速率：64.4/67、125.6/127、240.6/238（−90 dBm）；60.8/62、111.0/109（−70 dBm）。
  - MRT 和速率低 10–30%：33.9/37、10.1/14、15.7/22、22.7/33。
  - 27 个切分判断有 26 个一致；唯一不一致的是 MRT、−50 dBm、5 UE、K = 4，两边负载是 25.1 对 25.7 Gbit/s，几乎打平。
  - 7-1 负载（ZF 状态）与论文差 1–2 Gbit/s，例如 10 UE、−90 dBm、K = 1：23.3 对 25.0，因为我们的 M̂ 是 29，论文是 32。
- `mcs` 口径下，ZF 在 −90 dBm 时 M̂ 大幅下降（5/10/20 UE：17→8、29→14、45→28），和速率被单层 7.41 bit/s/Hz 截顶。

## S1a：只看负载的筛查

口径（`screen_stage1a.PROFILES`）：

| 口径 | 7-1 | 7-2 | 报头 | CI 与权值 |
|---|---|---|---|---|
| P0_paper | 32 bit | 32 bit | γ 常数，全部 RU | 无 |
| E0_headers | 32 bit | 32 bit | 显式 O-RAN | 无 |
| E1_bfp9 | BFP9 | BFP9 | 显式 | 无 |
| E2_modcomp | BFP9 | 调制压缩 | 显式 | 无 |
| E3_modcomp_ci | BFP9 | 调制压缩 | 显式 | SRS 每 10 ms，权值每时隙 |
| E4_modcomp_ci_bfwsrs | BFP9 | 调制压缩 | 显式 | SRS 每 10 ms，权值每次 SRS |
| E5_modcomp_ci_fast | BFP9 | 调制压缩 | 显式 | SRS 每 2.5 ms，权值每时隙 |
| E6_bfp9_ci | BFP9 | BFP9 | 显式 | SRS 每 10 ms，权值每时隙 |

输出：`stage1a_summary.json`（每个 K、口径、频谱效率模型、目标：各状态的选择、静态最优、四种收益、平均负载矩阵），`fig_dynamic_gain_vs_k.png`。

按状态动态比最佳静态的平均节省（下行目标，%）：

| 口径 | 模型 | K=1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| P0_paper | shannon | 0 | 0 | 1.0 | 14.9 | 24.6 | 0 | 0 |
| P0_paper | mcs | 0 | 0 | 1.5 | 16.8 | 28.1 | 15.2 | 0 |
| E1_bfp9 | mcs | 0 | 10.3 | 21.4 | 29.1 | 34.6 | 13.5 | 0 |
| E2_modcomp | mcs | 41.9 | 44.7 | 47.7 | 33.1 | 11.1 | 0 | 0 |
| E3_modcomp_ci | mcs | 41.9 | 44.7 | 47.7 | 34.7 | 12.4 | 0 | 0 |

- 峰值节省：有动态收益的 K 上多为 10–11%，个别点 4–6%，其余为 0。原因是 7-1 的峰值状态（MRT、−50 dBm）换成 7-2 后，峰值移到第二重的 ZF 状态（20 UE、−70 dBm），后者本来就选 7-1。
- 按 RU、按撒点选择比按状态多 0–4 个百分点。
- 目标改为下行加上行，差别 ≤ 1 个百分点。
- 结论写在 README 的"阶段 1a 的结论"。

## S1b：失真耦合的联合选择

- 用户决定（2026-10-12）：继续用阶段 0 的随机信道；真实编解码器（`oran/compression` 的 BFP、无损调制压缩、权值 BFP）；以 MCS 截顶为主、Shannon 只作对照；κ = 0.98（另给 0.95、0.99）；每个状态联合选切分、U-plane 位宽（BFP4–12、16 bit、调制压缩）和权值位宽（BFP4–12、16 bit），ZF/MRT 与服务天线数按论文方法；目标为峰值约束下最小化平均负载。
- κ 的含义：压缩后的和速率 ≥ κ × 同一服务天线集合下理想前传的和速率（撒点平均）。论文的 κ 用于选服务天线数，这里再给压缩留同样比例的余量。
- 峰值约束：每条 RU 链路的下行负载（每个撒点取最忙的 RU，再对撒点平均），在 9 个状态上的最大值 ≤ C。C 取最佳静态设计能做到的最小值，即按静态设计配置链路容量，各类策略在同一容量下比平均负载。注意：如果约束加在网络总负载上，按状态选择时"最小化平均"和"最小化峰值"会同时达到，约束不起作用，所以这里约束放在单条链路上。
- 与 S1a 的口径差别：权值 bundle 改为 1 PRB（ZF 按 PRB 计算）；SE-11 每块只计一个 8 bit 压缩参数（`fhload.bfw_block_bits`，S1a 按每 12 个值一个参数计，只在 K ≥ 16 的 ZF 7-2 上有极小差别）。
- 每个撒点只模拟一个时隙（每 PRB 168 个 RE）。UE 用这 168 个 RE 做 LMMSE 估计有效增益，因此 MRT 的干扰估计带有少量样本噪声，个别配置的和速率比例可能略大于 1。

输出：`per_drop.npz`（模型 × 状态 × 撒点 × 配置 × K 的和速率、总负载、最忙链路负载、上行 CI），`rows.jsonl`（撒点平均），`stage1b_summary.json`、`stage1b_report.txt`（四类策略在各 K、κ 下的解与每状态选择），`fig_policy_saving_vs_k.png`、`fig_distortion_curves_k8.png`。

结果（MCS 截顶，κ = 0.98，同一链路容量下平均下行负载的节省）：

| K | 链路容量 C [Gbit/s] | 静态最优 | 静态平均 [Gbit/s] | 只调位宽 | 只换切分 | 联合 |
|---|---|---|---|---|---|---|
| 1 | 0.23 | 7-1/BFP6 | 7.83 | 16.5% | 35.1% | 36.0% |
| 2 | 0.45 | 7-1/BFP6 | 7.83 | 16.5% | 39.2% | 40.1% |
| 4 | 0.91 | 7-1/BFP6 | 7.83 | 16.5% | 43.5% | 44.4% |
| 8 | 1.82 | 7-1/BFP6 | 7.83 | 16.5% | 46.5% | 47.4% |
| 16 | 3.27 | 7-2/调制压缩/权值 BFP7 | 5.06 | 0.9% | 24.1% | 25.5% |
| 32 | 3.48 | 7-2/调制压缩/权值 BFP7 | 2.83 | 0.5% | 4.0% | 4.4% |
| 64 | 4.37 | 7-2/调制压缩/权值 BFP8 | 1.70 | 2.5% | 0 | 2.5% |

- 联合动态的每状态选择（K ≤ 8）：ZF 状态 7-1 + BFP5/6，MRT 状态 7-2 + 调制压缩。K = 16：ZF 在 −70 dBm（10、20 UE）换成 7-2 + 调制压缩 + 权值 BFP6。
- Shannon 口径（对照）：K ≤ 8 静态最优为 7-1/BFP9（13.2 Gbit/s），联合动态省 44.5–52.4%，只换切分 37–45%，只调位宽 31.6%；K ≥ 16 只剩 3–5%。
- κ = 0.95（MCS）：静态 7-1/BFP5，联合省 31.3–48.8%（K ≤ 16），K = 32 为 9.7%。κ = 0.99 与 0.98 相差 ≤ 2 个百分点。
- 动态策略自己能做到的最小链路峰值与静态相同或低 0–7%（K = 16：3.04 对 3.27 Gbit/s）。
- 失真曲线（K = 8，`fig_distortion_curves_k8.png`）：MCS 口径下，ZF 的 7-1 BFP 和 7-2 权值 BFP 都在约 5 bit/分量（10 bit/IQ 对）处达到 0.98；MRT 在 −50 dBm 下 BFP4 也几乎无损。Shannon 口径下 −90 dBm 的 ZF 要到约 8–9 bit。

## S2a：双 O-RU、各 32 阵元（不用 Sionna RT）

- 用户决定（2026-10-12）：双 O-RU，各 32 阵元，其余按阶段 2 的建议：集中式 7-1 / 7-2、O-RU 本地预编码、CI 方案作为选择变量、状态含移动性（SRS 周期），先不用 Sionna RT。
- 配置（51 个，两个 O-RU 相同）：7-1 × U-plane {BFP4–9, 16 bit}；7-2 集中式 × 权值 {BFP4–9, 12} × CI {天线级 SRS IQ（raw）、ST 6 BFP4/6/9、KLT 孪生准确、KLT 孪生 2° 误差}；7-2 本地 × 功率规则 ν ∈ {0, 0.5}。U-plane 层符号一律用调制压缩（S1b 中它在 7-2 上总是最优）。
- 状态：UE 数 {8, 16, 32} × 噪声底 {−110, −90, −70} dBm/PRB × SRS 周期 {5, 10, 20, 40} ms；每个 UE 数 100 个撒点，种子 20261012；频谱效率 MCS 截顶（主）与 Shannon。
- 目标与约束：上下行总负载的状态平均最小；单条 O-RU 链路的 max(下行, 上行) 在所有状态上的峰值不超过最佳静态设计。另外比较四种方案集合：全部、去掉 KLT（只用 O-RAN 已有的 CI）、去掉本地预编码、两者都去掉。
- 输出：`per_drop.npz`、`rows.jsonl`、`stage2a_summary.json`、`stage2a_report.txt`、`fig_stage2a_policies.png`。
- 第一版（`stage2a_two_ru_v0_equalpower/`）中集中式 RZF 每 UE 等功率，作为参考偏弱：低 SNR 下被本地 ν = 0.5 反超 3–10%。已改为集中式也在 ν ∈ {0, 0.5} 中选（由 DU 用自己的 CI 预测和速率），第一版只保留作记录。

结果（MCS 截顶，同一链路容量下平均总负载的节省）：

| 方案集合 | κ | 静态最优（平均 Gbit/s） | 只调参数 | 只换方案族 | 联合 |
|---|---|---|---|---|---|
| 全部 | 0.98 | 7-2 集中式 w9 / ST6 BFP9（5.25） | 7.5% | 0 | 7.5% |
| 去掉 KLT | 0.98 | 同上（5.25） | 6.1% | 0 | 6.1% |
| 全部 | 0.95 | 7-2 集中式 w8 / ST6 BFP9（5.11） | 7.2% | 3.3% | 9.3% |
| 去掉 KLT | 0.95 | 同上（5.11） | 6.5% | 3.3% | 8.8% |

- 链路峰值：所有策略类都是 4.96 Gbit/s（κ = 0.98），动态不降峰值。
- 每状态选择（κ = 0.98）：始终是 7-2 集中式；权值位宽 K = 8 时 4–5，K = 16 时 5–6，K = 32 时 5–9；CI 多为 KLT 孪生准确，K = 32、−110 dBm 时为 ST 6 BFP9。
- 和速率比例（全部配置的表见 `stage2a_ratio_table.txt`）：本地 ν = 0.5 在 −70 dBm 时为 0.978 / 0.980 / 0.942（K = 8 / 16 / 32），高 SNR 时 0.74–0.92；权值 BFP5 在 K = 32、−110 dBm 时只有 0.52；KLT 孪生准确在 K = 32、−110 dBm 时为 1.23（去噪），2° 误差时为 0.999。
- 权值占比：SE-11 BFP9 每时隙占集中式下行的 17.3% / 17.7% / 20.7%（K = 8 / 16 / 32），BFP5 时为 9.7–11.6%。CI（ST6 BFP9，SRS 每 5 ms）占总负载 1.7–2.0%。
- 敏感性（权值改为每次 SRS 下发，只改负载口径）：静态最优降到 4.11 Gbit/s（−22%），联合动态只剩 1.6%（κ = 0.98）或 2.5%（κ = 0.95）。

## S3：方向 ②+③（回到论文框架）

- 用户决定（2026-10-12）：方向 ② 与 ③ 一起做；Split 8 以后再说；动态分配逐撒点与逐状态两种粒度都做。
- 设置：阶段 0 的场景与撒点（种子 2024，M.2135），ZF/MRT 与 M̂ 按阶段 0 的状态级选择；K ∈ {1, 2, 4, 8, 16, 32, 64}；MCS 截顶（主）与 Shannon；κ = 0.98；κ_s ∈ {0.98, 0.984, 0.988, 0.992, 0.996, 1.0}；λ 取 10⁻³–10³ 的 31 个对数点。
- 失真模型标定：7-1 的 BFP 相对失真 −18.6 / −24.6 / −30.7 / −36.7 / −42.7 / −48.8 / −54.8 / −60.7 / −66.7 dB（BFP4–12），16 bit 定点底 −86.1 dB；权值按 BFP 规则解析计算。与 S1b 的真实编解码器对照：7-1 BFP6（Shannon）103.9 对 105.4，权值 w6 111.8 对 112.2，w9 126.5 对 126.5。
- 输出：`stage3.npz`（逐撒点的候选点）、`stage3_summary.json`、`stage3_report.txt`、`fig_p1_load_vs_k.png`、`fig_p2_capacity_curves.png`、`validation_real_codec.json`（`validate_stage3.py --drops 20`）。
- 结果表与结论见 README 的"阶段 3 的结论"。复核：P / B2 / B1 的真实和速率比例平均为 0.9851 / 0.9854 / 0.9856（MCS），对应预测值 0.9852 / 0.9855 / 0.9855；低于 0.98 的撒点占 18.5% / 17.2% / 16.4%，最低 0.972。

## S4 / S4b / S4c：方向 1 的检验（2026-10-13）

结论写在 `theory_notes.md` 第 5 节。要点：
- S4（`stage4_rule/`，30 个撒点，MCS）：命题 2 / 推论 2 对 44036 个逐 RU 切分决策的预测与优化器 100% 一致。
- S4b（`stage4_path_ablation/`，100 个撒点，MCS）：服务集合路径消融。逐撒点 P1 负载（9 个状态平均，三条路径取最好时以论文路径为基准）：

| K | 论文路径 | + 和速率贪心 | + 负载感知（= 阶段 3 P） | 三者取最好 |
|---|---|---|---|---|
| 1 | 4.84 | 4.38（−9.6%） | 4.34（−10.4%） | 4.33 |
| 4 | 4.40 | 3.96（−10.1%） | 4.07（−7.6%） | 3.94 |
| 8 | 4.17 | 3.76（−10.0%） | 3.91（−6.2%） | 3.74 |
| 16 | 3.69 | 3.38（−8.4%） | 3.45（−6.6%） | 3.35 |
| 32 | 2.65 | 2.49（−6.0%） | 2.51（−5.2%） | 2.48 |
| 64 | 1.72 | 1.67（−2.8%） | 1.67（−2.7%） | 1.67 |

  负载感知路径在 MRT、K ≥ 2 时常常达不到 κ（每个 K 有 288–392 个撒点不可行），和速率贪心有 77 个不可行；以上为与论文路径取较好者后的平均。运行在写结果行时因 inf 报错，结果行由 `ablation.npz` 补写，脚本已修正。
- S4c（`stage4_cf/`，100 个撒点中三者都可行的 823 个状态-撒点组合，MCS）：闭式策略相对同一路径上的优化器的负载差：+6.7%（K = 1）、+4.9%（2）、+3.2%（4）、+2.0%（8）、+2.2%（16）、+0.6%（32）、−0.4%（64）；平均位宽 5.1–5.8，7-2 占比 0.39–1.00。

## S5 / S6（2026-10-13）

结论写在 `theory_notes.md` 第 7、8 节。
- S5（`stage5_paths10/`）：A 部分逐状态对照见 `rows.jsonl`（part = A）；B 部分在三条路径都可行的 652 个状态-撒点上：[10] 4.88 / 4.88 / 4.89 / 4.81 / 4.35 / 3.19 / 2.12 Gbit/s（K = 1 … 64），近似论文路径 5.20 … 2.13，和速率贪心 4.65 … 2.06；[10] 路径有 217 个状态-撒点达不到 κ·SE_max（SE_max 沿用阶段 0 近似路径的最大值）。
- S6（`stage6_prop3/`）：命题 3（CF-WF）相对逐 RU 拉格朗日优化器 −0.4% 到 0.0%；全网一个位宽 +9.2%（K = 1，[10] 服务集合）到 −0.3%（K = 64）。

## S7（2026-10-13）

- 输出：`stage7.npz`（每个状态、撒点、K、κ_s、模式、τ 的和速率比例与负载）、`stage7_summary.json`、`stage7_report.txt`、`validation_real_codec.json` 与 `validation_report.txt`（`validate_cf.py --drops 20`）。
- 结论写在 `theory_notes.md` 第 9 节：动态总收益 41–47%（K ≤ 8）、30%（16）、11%（32）、1%（64）；其中切分随状态切换贡献 26–37%，压缩比随状态调整 2–6%，逐 RU 调整 2–6%，逐撒点 ≤ 0.6%。真实编解码器：预测 0.9877、实测 0.9881。

## S8（2026-10-13）

- 每个 ZF 状态取 S7 在 K = 16 处的逐状态最优 (κ_s, τ)；n* 用 w = b = τ 和平均 Qm。
- 结果：(5,−90) M̂ 5.2、n* 3.75、K* 45.7；(10,−90) 11.3、7.39、41.9；(10,−70) 33.5、6.37、12.2；(20,−90) 25.7、15.08、37.5；(20,−70) 50.5、12.86、16.3。预测边界 K* = 45.7（S7：K = 32 动态收益 10.8%，K = 64 为 0.8%）。
- 半解析预测的 7-2 比例 P(n_r > n*) 与 S7 观察对照：除 (20,−70) 在 K = 16 时 0.58 对 0.41、(10,−70) 在 K = 8 时 0.13 对 0.22 外，其余偏差 ≤ 0.03。

## S9–S13（2026-10-13）

结论写在 `theory_notes.md` 第 10–12 节，命题的正式陈述与证明在 `propositions.md`。
- S9：复现差距来自 [10] 的噪声模型（每 UE 对数正态、4.8 dB、线性均值）与 ZF 归一化（式 (18)），两者合用后 9 个状态都与论文吻合。代码口径 `faithful.py`（`--model paper10`）；默认 `fixed` 与第一版逐位相同（回归检查：S7 前两个撒点的 grid 差为 0）。
- S10 / S11：忠实口径（表 2 / 表 4）下的动态分解，结构与 S7 相同；1024QAM 让压缩比随状态调整的价值升到 7–10%（K ≤ 8）。各有 1 个无效撒点（SE_max = 0）。
- S12：命题 3 在忠实口径、两种 MCS 表下仍与优化器持平（−0.4% 到 +0.1%）。
- S13：K* = 43.6 / 39.1；真实编解码器复核实测 0.9882 / 0.9870。

## S14：论文插图数据（2026-10-13）

- 输出：`stage14_figdata/` 下的 `fig_*.dat`（已复制到论文仓库 `figures/data/`）、`rows.jsonl`（OPT 的逐 RU 选择）、`summary.json`。论文仓库中对应的 TikZ 源文件为 `figures/fig_{prop1,prop2,prop3,dyn,boundary,p2}.tex`。
- 命题 1（10 UE、−90 dBm、ZF、K = 8，40 个撒点）：理想 SINR 31.3 dB；模型与 O-RAN 编解码器在 ≥ 6 bit 时差 ≤ 0.6 dB，5 bit 时 ≤ 1.2 dB，4 bit 时模型偏保守 1.0 dB（7-1）/ 2.2 dB（权值）。
- 命题 2（忠实口径）：10063 个 ZF 状态的逐 RU 决策 100% 符合门限，MRT 状态全部 7-2。
- 命题 3：每组一个 t̂ 时，clip(round(t̂ + x_r), 4, 12) 与 OPT 所选位宽相同的比例为 94.4%（7-1）/ 97.7%（7-2 权值），偏差从不超过 1 bit。直接"组内去均值"拟合的斜率为 0.69（7-1，受 4 bit 下限截断影响）/ 1.01（7-2）。
- 论文不压缩方法（忠实口径）：平均负载 16.6 Gbit/s（K = 4）/ 12.5 Gbit/s（K = 16），SE 比例 0.982；闭式方法逐状态 4.17 / 3.60 Gbit/s（−75% / −71%）。
- P2：闭式方法平均达到 0.98 SE_max 约需 9 Gbit/s（K = 4、16），不压缩方法需 31–34 Gbit/s；C = 5 Gbit/s 时 0.80–0.82 对 0.11–0.13；逐 RU 相对全网统一最多多 1.1（K = 4）/ 2.8（K = 16）个百分点。
