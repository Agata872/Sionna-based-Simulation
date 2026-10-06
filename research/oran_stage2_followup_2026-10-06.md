# 第二轮探索：强基线、双向消息预算与实际链路自适应

本轮得到的主要决策是：**暂停把通用低维摘要压缩或静态 CSI-RS/数据权值失配作为独立论文主线。** 前者的一个显著量化问题可由简单残差表示解决；后者的大部分静态吞吐损失可被普通 OLLA 吸收。低位宽的有效信道目标仍有实测收益，但在当前工具链的完整前传开销下，不能据此声称已经找到 TWC 问题。

原有 `oran/`、测试和模块组织保持不变。本轮新增独立实验与分析脚本，在 GPU0/GPU1 完成运行。全部原始数据、参数、运行时源码快照、统计结果和图保存在 `results/exploration_20261006/`。该目录沿用上一轮日期，不表示所有 runs 属于同一个统计批次。

## 1. 强前序核对与复现边界

[Delayed-global Team-MMSE 作者稿](https://arxiv.org/html/2404.12786v1) 已给出“本地 MMSE 因子 × 全局协调矩阵”的分工，并讨论 DU/AP 功能划分。因此，局部新鲜 CSI 加全局旧 CSI、以及下发 K×K 矩阵，都不能单独当作新颖性。

本轮独立实现该模型的两 RU、单接收天线 UE 方程，未拷贝作者代码。在无老化、无估计误差、无量化且不做事后功率归一化时，Team 解应等于集中式 MMSE。三组负载的相对误差为 4.11×10⁻¹⁵、1.05×10⁻¹⁴、3.94×10⁻¹⁴，检查通过。见 [validation.json](../results/exploration_20261006/coordination_budget_256/validation.json)。这验证方程和张量方向，不等于复现论文全部网络仿真或性能图。

后续 SE 筛查统一把每 RU 发射功率恢复为 0.5，指标为带瞬时接收 CSI 的 sum-SE。这一步改变原始优化问题；不能沿用 Team-MMSE 对原二次目标或 hardening-rate 的最优性结论。确实有低负载条件下局部 MMSE 的该指标更高，已保留这些结果。

[SOPIP 作者全文](https://www.comm.toronto.edu/~liang/publications/ToWC_HierarchicalSemiOnlineInformationParsing.pdf) 使用有效信道偏差驱动局部更新。上一轮 proximal 跟踪、以及本轮 target-only ridge，仍作为与其结构相近的筛查基线；**完整多时刻 SOPIP 消息管线尚未复现**。

新增的相关前序，按截至 2026-10-06 检索到的一手材料记录如下：

| 工作 | 当前核对依据 | 对选题的约束 |
|---|---|---|
| Optimal Local Precoding for Cell-Free Massive MIMO under the UatF Bound，TSP 2026，DOI 10.1109/TSP.2026.3694919 | [作者公开代码与论文说明](https://github.com/NicholasSeu/MAT_UatF_SCA) | 已有更强局部优化和双向统计量量化实现；不能只比较未经优化的 local RZF。本文尚未复现该 QoS/UatF 框架。 |
| Cell-Free Distributed Precoding Without Iterations on Unreliable Fronthaul by Quadratic Team Learning，TCOMM 2026 | [作者代码](https://github.com/hzy238221seu/QTL4CF-Precoding)、[IEEE 官方目录](https://www.comsoc.org/system/files/2026-02/publications_contents_digest_2025-jan_0.pdf) | 随机可得协作信息、非迭代协作和学习已存在相邻工作；需获取正文后做精确信息集合比较。 |
| Asynchronous Centralized and Distributed Precoding for Extensive Cell-Free OFDM With Adaptive Fronthaul Overhead，TWC 2026，DOI 10.1109/TWC.2025.3596109 | [IEEE 官方目录](https://www.comsoc.org/system/files/2026-02/publications_contents_digest_2025-jan_0.pdf) | 异步、OFDM 与前传开销已有直接相关前序；目前只确认书目信息，未获取全文，不能推断其覆盖的具体量化/测量约束。 |
| Multi-Antenna Users in Cell-Free Massive MIMO: Stream Allocation and Necessity of Downlink Pilots，TCOMM 2026 | [作者 v2 全文](https://arxiv.org/html/2505.02951v2)、上述 IEEE 目录 | 多天线 UE 的有效信道估计与流分配已有系统分析。继续探索时要区分 CSI-RS 的质量反馈用途与 DMRS 的数据解调用途。 |

这些工作进一步降低了泛化表述的“gap”可信度。当前没有宣称完成穷尽检索，也没有宣称已经证明某个问题无人研究。

## 2. 双向预算、估计误差与摘要表示

正式目录：[coordination_budget_256](../results/exploration_20261006/coordination_budget_256/summary.csv)；条件采样加密复核：[coordination_mc2048](../results/exploration_20261006/coordination_mc2048/summary.csv)。

扫描 7 种 (每 RU 天线 N, 用户 K) 负载、5 个端点相关系数 ρ、3 个新鲜 CSI 观测误差方差 η；每负载 256 个独立信道 drops。信道为 iid Gauss–Markov/Rayleigh，两 RU 独立；不是 CDL 移动轨迹。η 为观测中附加 CN 噪声相对单位信道功率的方差；RU 还拥有准确的旧本地 CSI，并融合旧/新观测得到后验，因此 η 不等于最终后验 NMSE。

因果信息明确为：t=0 取得旧本地 CSI，t=1 上传消息到 DU，t=2 DU 计算完成，t=3 RU 收到协调消息，再使用自己的新鲜观测形成数据权值。ρ 是 t=0 到 t=3 的相关性；这些 ticks 没有映射到毫秒，也没有包队列。原始完整 CSI 获取/SRS 开销仍未实现。

比较完整旧 W、局部 MMSE、proximal 跟踪、有效信道 target，以及 Team 协调。完整 W 路线上传 H、下发 W；摘要路线由 RU 计算条件统计 B_l，上传 B_l，DU 求 C_l 并下发，RU 计算 F_l C_l。后一种摘要路线是基于已知方程构造的筛查实现，并非已验证的 O-RAN 消息格式。

两方向分别计费：列分块 BFP、字节对齐、每消息 66 字节开销和 MTU1500 分包。66 字节包含假定的 16 字节 timestamp/version/format/scale-ID 头、8 字节 eCPRI-like 头和 42 字节 Ethernet 开销。上下行容量分别限制，不把上传 CSI 当成免费。共享层数据流不进入本节“协调控制开销”的数字，不能据此宣称整条前传节省相同比例。

**发现 A：直接量化接近单位阵的 B，可能破坏协调求解。** N=16、K=8、ρ=0.95、η=0、20 dB，加密条件采样至 2048 后：

| 方法 | sum-SE，bit/s/Hz | 每 RU 每方向 wire bits/update |
|---|---:|---:|
| 旧全局 W，上传/下发 BFP9 | 39.807 | 2896 |
| 上述消息加本地 proximal 跟踪 | 61.855 | 2896 |
| 未量化 Team 参考 | 63.925 | 不分配有限预算 |
| 直接 BFP9 上传 B、下发 C | 49.598 | 1744 |
| BFP9 上传 D=I−B、下发 C | 63.851 | 1744 |
| D/C 各 BFP16，放入完整 W9 的双向容量上限 | 63.926 | 2640 |

未量化参考不附会“16-bit 理想传输”。残差 9-bit 相对参考的配对 SE 差区间约 [−0.0810, −0.0666]，损失约 0.115%；同容量的残差 16-bit 差区间约 [−0.00144, 0.00336]。这说明在该条件下，简单表示加位宽分配已基本消除量化损失。

**故障隔离**：[coordination_failure_diagnosis/summary.json](../results/exploration_20261006/coordination_failure_diagnosis/summary.json)。直接 B9 后，约 88.3% drops 至少一个统计矩阵的对称部分出现特征值 ≥1；协调系统条件数中位数从约 527 增至 1060，95 分位从约 625 增至 7856。只量化下发 C 的 SE 损失约 0.066，而只量化上传 B 已损失约 14.29，故主要问题在上传统计及其求解放大。PSD 投影/谱截断也作为普通修复基线：固定 margin=0.01 把结果恢复到 62.794，但仍逊于残差编码。未根据测试集把该 margin 声称为泛化最优值。

D=I−B 使用相同 K×K 维度和同一 BFP 编码，格式 ID 在假设消息头中；单位阵无需额外发送。它是常规残差表示对照，**目前不作为新算法贡献**。

**发现 B：摘要是否省开销取决于负载。** N=4、K=8 时，完整 H/W9 的每方向预算为 1168 bits；B/C9 需要 1744 bits，反而更贵。按两个方向相同容量上限，摘要只能使用 4-bit，每方向 1104 bits；ρ=0.95、η=0 时，SE 从未量化约 20.027 降至 17.868。固定“9-bit 摘要优于 W9”会混入不相同的预算。这条负载边界是工程结论，尚不构成新颖性。

**发现 C：估计误差明显改变收益来源。** 同样 N=16、K=8、ρ=0.95，在 η=0.1 时，Team 参考 SE 约 46.333，本地跟踪约 45.728，而容量匹配残差表示约 46.333。若旧 CSI 理想、统计规律准确，优先级更像局部信道估计质量和新鲜度；继续追求该配置下的小量化损失缺少动机。这不能外推到有噪声旧 CSI、未知协方差或真实 SRS。

512→2048 条件 MC 复核使用相同 evaluated drops，独立于实际新鲜信道的采样。参考方案在复核分组的最大平均 SE 变化约 0.0457；不稳定的直接 B9 方案会因量化门限跨越出现较大逐 drop 变化。详见 [mc_convergence.json](../results/exploration_20261006/coordination_mc2048/mc_convergence.json)。这里验证趋势稳定，不宣称有限 MC 得到了解析最优解。条件预测计算成本也需要未来计入 RU 算力约束。

## 3. CSI-RS→CQI→MCS→DMRS/CRC→OLLA 已实际接通

[32 个独立 CDL drops 的合并结果](../results/exploration_20261006/compression_loop_combined32/summary.csv)来自两批不同种子；两个原批次分别保留，不重复当作额外独立样本。每 drop 的信道固定，模拟 128 次数据传输，前 64 次用于预热，后 64 次统计。每 8 次数据传输插入一个真实 CSI-RS 测量时隙；该时隙的打孔 PDSCH 不译码，整个时隙均从 goodput 的可用数据时间扣除。ACK/NACK 在 4 次传输的批次结束后更新 OLLA，CQI 在下一数据批次生效。

配置为 CDL-C、8 Tx、2 UE×4 Rx、每 UE 2 层、12 PRB、20 dB，U-plane BFP9，所有权值列恢复到已知功率 1/4，DMRS 估计实际压缩 W。仍使用理想 DU CSI，未引入物理移动或真实 RU CSI 获取。

实际 CSI-RS＋OLLA：

| 权值方法 | 两 UE 合计首传 goodput，Mbit/s | 首传 BLER |
|---|---:|---:|
| 普通 BFP3＋范数恢复 | 45.752 | 10.72% |
| 有效信道相位选择 BFP3＋范数恢复 | 48.906 | 9.84% |
| 普通 BFP9＋范数恢复 | 55.000 | 7.59% |

3-bit 有效信道目标相对普通 BFP3 的配对 goodput 增益约 3.153 Mbit/s，95% 区间 [2.164, 4.126]，约 6.9%。使用实际有效信道的 genie CQI 再接同一 OLLA，增益约 3.101 Mbit/s，区间 [2.274, 4.014]。这说明优势仍存在，并非全由 CSI-RS 估计误差解释。genie CQI 仍使用 EESM/AWGN 门限，不是最优 MCS。

固定 MCS24 的**同 drops、同 U-plane、同噪声**对照另存于 `compression_loop_fixed24_paired32`，完成后在后续补充段给出与上述闭环的精确比较；不要直接把上一轮不同配置的约 22% 增益当成这一轮配对结果。

**完整前传预算会削弱极低权值位宽的应用动机。** 现有 full-grid U-plane 与 SE11 分包模型下，BFP3/BFP9 权值对应总 wire rate 为 362.880/367.488 Mbit/s，只节省约 1.25%。即使用相位优化的 BFP3，首传吞吐仍比 BFP9 少约 11.1%。当前配置不能证明把 W9 改为 W3 是划算的系统选择。[完整前传权衡图](../results/exploration_20261006/compression_loop_combined32/full_fronthaul_tradeoff.png)；[精确开销](../results/exploration_20261006/compression_loop_combined32/total_fronthaul_cost.json)。该判断基于当前 U-plane 格式；加入 modulation compression 或其他层流表示后要重新计费。

## 4. 主动证伪：静态测量失配可被普通 OLLA 消除

[配对失配结果](../results/exploration_20261006/compression_loop_measurement_mismatch16/measurement_comparison.json)使用第一批相同 16 drops、同噪声。CSI-RS 始终使用 DU BFP9 权值，数据使用各自的 BFP3 或 BFP9。它只模拟表示失配，不等同于完整 RU 自主跟踪过程。

| 数据权值方法 | 关闭 OLLA：失配相对匹配的 goodput 差 | 开启 OLLA：失配相对匹配的 goodput 差 |
|---|---:|---:|
| 普通 BFP3 | −36.200 Mbit/s，[−42.011, −30.681] | +0.402 Mbit/s，[−0.173, 1.077] |
| 有效信道相位 BFP3 | −13.802 Mbit/s，[−22.359, −6.067] | +0.230 Mbit/s，[−0.703, 1.218] |

开启 OLLA 后，两者后半段 BLER 分别约 10.30% 和 9.62%，吞吐差异区间包含零。关闭 OLLA 的大损失不足以论证需要一种新协作机制。BFP9 同表示控制的 8192 条 TB 记录在 CRC、MCS、CQI、成功位数及 OLLA 状态上逐条一致，极限检查通过。[恢复过程图](../results/exploration_20261006/compression_loop_measurement_mismatch16/measurement_mismatch_recovery.png)。

结论限定在静态、恒定表示失配、当前 0.5 dB NACK 步长和反馈批次。尚未证明 OLLA 能消除快速非平稳更新、低延迟尾部失败或有移动的 CSI-RS 失效。

## 5. 更新后的探索路径与进入论文阶段的门槛

当前暂停三条泛化主线：纯 CSI-RS IQ 降位宽节省前传；通用低维矩阵压缩；静态 CSI-RS/数据表示失配。保留其实现和负结果作为后续强基线。

下一阶段暂定问题：**RU 在 DU 协调消息与 UE CSI-RS 反馈到达之间做本地预编码更新时，怎样用有限的协作信息判断更新对 UE 已测有效信道、干扰协方差和短时 BLER 的影响，并决定本地更新、补充测量或保持旧权值？** 这是待证伪的问题表述，尚未确认学术空白或给出新算法。

推进顺序：

1. 获取上述异步 OFDM、QTL、robust cascaded Team-MMSE 前序的全文并建立假设/消息/测量对照表；补完整 SOPIP 因果基线。若正文已完整覆盖拟研究约束，调整问题。
2. 接入连续的可复现 CDL/几何移动轨迹，保持旧/新本地 CSI 与估计误差的因果关系；不能把独立 CDL samples 拼成移动轨迹。明确 SRS 获取、CSI-RS 测量、反馈、DU 协调到达和数据权值的版本。
3. 将双 RU 的局部更新与实际 CSI-RS/CQI、DMRS/CRC 接在同一时间轴上。先比较周期性协调＋SOPIP/Team、普通 OLLA、状态分组 OLLA、普通低通 CQI、权值冻结和测量随更新触发。零老化、零局部更新必须回到现有结果。
4. 先看短时超目标 BLER、失败连续长度和吞吐/完整双向前传权衡，不只看平均 genie SE。同步和互易误差另设对照，避免把未知相位误差误认成消息压缩效应。
5. 只有简单基线留下显著损失、改善在合理 6/9-bit 范围内存在、真实 measurement/CRC 仍支持结论，且可以推导信息约束下的更新边界，才讨论 TWC 方法。若状态分组 OLLA、冻结/重测已解决问题，再换方向。

需要下一次讨论的实质是问题范围：继续把 CSI-RS 放在“保护测量与调度可靠性”的位置，还是转向高负载多 RU 的协作信息与用户/流分配。当前数据支持讨论这个取舍，不支持选定论文标题或声称已有可投稿贡献。

## 6. 复现与审计

新入口：`scripts/explore_coordination_budget.py`、`scripts/diagnose_coordination_quantization.py`、`scripts/explore_compression_closed_loop.py`。汇总：`scripts/summarize_next_exploration.py`；两批独立 drops 合并与失配比较有独立脚本。先 `source env.sh`，用 `CUDA_VISIBLE_DEVICES=0` 或 `1` 选择物理 GPU。

每个正式 run 使用新目录，不覆盖旧原始数据；`experiment_source.py` 是运行时版本，避免用后来扩展了参数的脚本误解释旧结果。`metadata.json` 保存 args 和假设；结果逐条写入 `rows.jsonl`，完成后保存 CSV 和 `complete.json`。统计 CI 按独立信道 drop 聚类，时间 slots 和两个 UE 不伪装成独立 drops。

两个失败冒烟目录保留 `failure.json` 并排除正式结论：一次暴露统计量量化导致的系数饱和，一次为资源网格属性引用错误。修正后正式运行均完成。最终统一的源码/数据 SHA 审计清单见 `stage2_manifest.json`。
