
# 第三轮探索：RU 自主预编码后的质量信息是否足以支持 MCS 决策

后续探索见 [第四轮记录：O-RU 侧 SRS 波束赋形下 O-DU 的多用户调度与链路自适应](oran_stage4_srsbf_convergence_2026-10-07.md)。第四轮把本轮的问题落到 CUS v22.00 的 SRS-BF-DL 上，并收敛为 TWC 选题；以下保留第三轮当时的结论与复现边界。

## 研究选择

建议把下一阶段主线收敛到：**延迟、有限协作信息下，如何观察 RU 自主预编码后的接收质量，并可靠地选择 MCS。** 先固定预编码策略、用户和层数，把研究变量限定为 RU 上报什么、何时上报，以及 DU 如何利用这些消息。无需再把预编码、压缩、用户选择、流分配和学习全部合成一个大问题。

本轮已经找到这个方向的实际动机和信息失效边界，尚未建立可投稿的新算法或完成新颖性排查。**周期性上传有效信道本身、接收器投影本身、事件触发本身，以及普通不确定性回退，都不能直接作为 TWC 创新。** 候选贡献应落在：在明确的私有信息集合和消息时延下，判定哪些消息足以支持调度决策，刻画不充分信息造成的决策损失，再设计达到该边界的上报机制。

相较第二轮，本轮补上了连续 Doppler 轨迹、RU 局部更新、实际 CSI-RS/CQI、延迟 ACK/OLLA、DMRS/LDPC/CRC 的同一时间轴，并加入 SOPIP 因果管线及普通有效信道上报对照。结果支持继续这个切口；不支持继续“静态失配加 backoff”或“直接压缩一个小矩阵”的泛化主线。

结果批次目录为 `results/exploration_20261007/`，这是创建时保留的批次名称。实际运行 UTC 时间以各目录 `metadata.json` 和 `complete.json` 为准，不能用目录名推定日期。本文整理于 2026-10-06 UTC。

## 1. 已有工作对问题范围的约束

| 一手材料 | 已覆盖的内容及对本选题的约束 |
|---|---|
| [Delayed-global Team-MMSE 作者稿](https://arxiv.org/html/2404.12786v1) | 本地新鲜 CSI 与全局旧 CSI 的分工、局部因子乘全局协调矩阵已有分析。本文不把这种架构或矩阵维度当成新意。 |
| [SOPIP 作者稿，Algorithms 1/2](https://www.comm.toronto.edu/~liang/publications/ToWC_HierarchicalSemiOnlineInformationParsing.pdf) | 旧全局信息解析成有效信道残差，再用新鲜本地 CSI 做更新已有具体算法。本轮独立实现其消息管线；新方向应研究这些权值实际生效后，DU 的质量判断是否充分。 |
| [CoMP 系统级联合传输与动态点选择，2012，作者存档](https://aaltodoc.aalto.fi/server/api/core/bitstreams/d43c45e4-09f4-44ec-a7bb-459344f430b5/content) | CQI 对预编码、干扰及点间相位的依赖，以及相关链路自适应问题已有讨论。不能把“CQI 与数据波束不同”当作首次发现。 |
| [Distributed antenna MIMO with limited feedback，2013，全文](https://link.springer.com/article/10.1186/1687-6180-2013-54) | 已比较多接收天线合并、量化有效信道及利用时间相关性的反馈。本文的固定投影反例是诊断工具，不是首次提出接收器影响反馈。 |
| [Robust Link Adaptation with Flashlight Interference，2024，作者稿](https://acris.aalto.fi/ws/portalfiles/portal/161685824/Robust_Link_Adaptation_in_Multiantenna_URLLC_Systems_With_Flashlight_Interference.pdf) | 动态波束/干扰使旧 SINR 不可靠、统计回退和稳健链路自适应已有直接前序。只改 OLLA 步长或增加预测 backoff 不够。 |
| [Contextual Multi-Armed Bandits for Link Adaptation，2019，作者稿](https://kth.diva-portal.org/smash/get/diva2%3A1366093/FULLTEXT01) | 上下文辅助的链路自适应已有工作。年龄分组和上下文学习应作为基线。 |
| [Task Oriented CSI Quantization，2019](https://arxiv.org/abs/1904.04057)；[分布式函数量化研究](https://arxiv.org/abs/2009.12879) | 面向最终任务而非 CSI 均方误差的量化，并非新的总方向；需要针对本问题推导具体边界和方法。 |
| [Self-Nomination，TWC 2026，作者 v1 全文](https://arxiv.org/html/2504.16351v1) | 已根据私有 CSI 的调度价值决定是否反馈，并施加平均反馈约束。v1 模型为单 Rx UE、入选后上传未量化 H、BS 集中 ZF、sum-rate/PF 目标；不能把“分布式决定信息是否值得发送”当作新贡献。 |
| [Robust Fronthaul-Aware Link Adaptation，2025，作者机构记录](https://portal.research.lu.se/en/publications/robust-fronthaul-aware-link-adaptation-for-uav-communication-with/) | 摘要明确针对前传丢包被误当空口错误，利用 RU 反馈的码字损失信息区分两者。本文不能泛称“首次把 RU 状态用于前传感知链路自适应”；该文完整正文尚未获取。 |
| [Distributed MIMO vRAN via Cross-Layer Link Abstraction，2026，作者全文](https://arxiv.org/html/2607.07520v1) | 已用 EESM/AWGN 校准支持离散 MCS、rank/power 和能效优化。接通标准链路和使用实际门限是验证基础，不能单独当作算法新颖性。 |

还有三篇必须在正式立项前逐式核对的强相关文献。本轮确认了 DOI/出版条目，QTL 还检查了公开代码，但**未取得三篇完整论文正文**：

- Asynchronous Centralized and Distributed Precoding for Extensive Cell-Free OFDM With Adaptive Fronthaul Overhead：[IEEE 条目](https://ieeexplore.ieee.org/document/11123632/)，DOI 10.1109/TWC.2025.3596109。
- Robust Cascaded Team MMSE Precoding for Cell-Free Distributed Downlink Under Hierarchical Fronthaul：[IEEE 条目](https://ieeexplore.ieee.org/document/10571814/)，DOI 10.1109/TWC.2024.3416296。
- Cell-Free Distributed Precoding Without Iterations on Unreliable Fronthaul by Quadratic Team Learning：[IEEE 条目](https://ieeexplore.ieee.org/document/11296898/)，[作者代码](https://github.com/hzy238221seu/QTL4CF-Precoding)，DOI 10.1109/TCOMM.2025.3642707。

不能依据标题、摘要或代码缺少某个模块，推断论文没有覆盖该约束。第二轮提及的最优局部 UatF 预编码也尚未在本轮网络中复现。因此，“这个问题值得继续”与“现有论文都没有解决”仍是两个不同判断。

## 2. 正式实验的因果模型

入口：[explore_dynamic_measurement.py](../scripts/explore_dynamic_measurement.py)。使用项目现有 `oran/` 数据链路，未修改核心模块。

- 两 RU；每 RU 4 或 8 根天线；两 UE，各 4 Rx、2 层；6 PRB、3 PRG；CDL-C、3.5 GHz、20 dB；U-plane 和初始 DU 权值均 BFP9。
- 每个独立 drop 的整段信道只调用一次 CDL；固定角度/时延，沿真实 Doppler 连续演化。每个 UE 在所有 RU 链路上使用相同速度向量。不是把独立 CDL samples 拼成移动轨迹，也不是空间一致的几何移动/遮挡模型。
- 时隙 0.5 ms，128 个时隙，前 32 个预热，后 96 个统计。正式 CRC 结果主要为 3 m/s。每 drop 每个策略占用相同物理时间，导频时隙成功载荷记为零。
- t=0 使用共同的初始 DU 快照与权值作为启动状态；启动阶段不用于正式统计。
- RU 在时隙边界 symbol 0 得到本地 CSI，随后形成本时隙权值。DU 每 16 个时隙更新一次，使用延迟 4 个时隙的快照。两个 RU 同时收到版本；尚无异步包到达、队列或丢包。
- 局部跟踪用 ridge=1 的普通最小二乘修正保持自己的旧有效信道贡献，并恢复每 RU 功率 0.5。固定整个数据时隙的权值；DMRS 采用该实际权值。
- CSI-RS 每 16 个时隙测量一次；CQI 和 ACK 均延迟 4 个时隙。ACK 更新的是发送时所属的 OLLA 年龄组，避免按 ACK 到达时的状态错分组。
- 普通 OLLA NACK 步长为 0.5 dB；另比较 2 dB 快速 OLLA、四个年龄组、CQI 低通、每 4 个时隙重测和导频间冻结权值。
- `genie` 用当前真实数据 SINR 生成 CQI，再接相同 OLLA。它是诊断参考，仍使用 EESM/AWGN 门限，并非最优 MCS 或可部署的有限消息方案。

普通有效信道上报对照由各 RU 独立编码 `G_l=H_l W_l`；DU 只组合已到达的报告，再计算 LMMSE SINR 和 CQI。报告采用自定义 float BFP6/9，每列 8-bit 共享指数、实虚部有符号尾数；不利用未计费的精确范数。它不是已实现的 O-RAN 标准扩展。

**获取 CSI 仍是抽象假设。** 主实验使用理想时隙边界 CSI；误差实验给所有策略相同的独立复高斯观测噪声。未实现物理 SRS、TDD 校准、UE 发射天线能力限制或外部干扰观测。项目当前 `channel.csi()` 的 SRS 说明也只是理想 CSI 假设，不存在物理 SRS 收发链路。

## 3. 实际 CRC 留下了可观测性损失

以下为每 RU 4 根天线、无局部观测误差、8 个独立 CDL drops。各策略使用相同 drops、噪声种子和评估时间。goodput 为两 UE 合计首传成功位数除以总物理时间，扣除完整导频时隙，没有 HARQ 收益。

| 策略 | goodput，Mbit/s | 首传 BLER | 相对年龄分组 OLLA 的配对 goodput 差，95% CI |
|---|---:|---:|---:|
| 旧 DU 权值 + 普通 OLLA | 16.454 | 7.64% | −1.437，[−2.813, 0.076] |
| 局部跟踪 + 普通 OLLA | 15.679 | 6.04% | −2.212，[−2.687, −1.759] |
| 局部跟踪 + 年龄分组 OLLA | 17.891 | 10.35% | 基线 |
| 局部跟踪 + 快速 OLLA | 15.013 | 8.68% | −2.878，[−3.862, −1.907] |
| 局部跟踪 + CQI 低通 | 15.795 | 6.18% | −2.096，[−2.831, −1.049] |
| 每 4 时隙 CSI-RS + 普通 OLLA | 14.496 | 10.68% | −3.395，[−4.299, −2.458] |
| 导频间冻结权值 | 14.443 | 9.65% | −3.449，[−4.565, −2.150] |
| 同一局部跟踪 + 当前 genie CQI | 22.534 | 9.58% | +4.643，[4.034, 5.229] |
| 同一局部跟踪 + G9 每 4 时隙、延迟 1 时隙 | 21.243 | 9.51% | +3.351，[2.747, 3.979] |

年龄分组 OLLA 比普通 OLLA 明显更强，因此后续不能只以普通 OLLA 作比较。当前更频繁 CSI-RS 的完整时隙代价很重，结果不能外推为实际 NR 稀疏导频都无效；更合理的局部打孔和非周期测量还需作为强基线。

G9 每 4 时隙方案相对年龄分组 OLLA 提升 **18.7%**，而预编码数据权值、瞬时理想 SE 和 CSI-RS 周期完全相同。这里改变的是 DU 可用质量信息及其 MCS 决策，不能把增益归因于更好的波束。

局部跟踪与旧权值的平均 SE 分别为 21.337 和 20.936 bit/s/Hz，但两者 SE 差和 goodput 差的 drop 配对区间都包含零。**不能宣称本轮已显著证明“SE 上升而 goodput 下降”。** 可证明的核心是：同一跟踪权值下，增强质量信息后首传 goodput 显著改变。

短时失败也是诊断指标：年龄 OLLA 的每 drop 最大连续失败 TB 数均值为 9.375，G9 每 4 时隙方案为 3.750；8-TB 窗口内 BLER>25% 的比例均值为 13.33% 和 9.41%。窗口相互重叠，这些数字不能当成独立样本，也不能当成 URLLC 尾部可靠性证明。

## 4. 信息时延比 6/9-bit 差异更有影响

仍使用完全相同的跟踪权值与 CSI-RS 周期。上行费用为每 RU 在旧 CSI 上传之外增加的 wire rate：

| 有效信道报告 | 每 RU 增加上行，Mbit/s | goodput，Mbit/s | 相对年龄 OLLA 的配对差，95% CI |
|---|---:|---:|---:|
| G6 每时隙，延迟 1 时隙 | 3.552 | 21.819 | +3.928，[3.374, 4.460] |
| G9 每时隙，延迟 1 时隙 | 4.704 | 21.765 | +3.874，[3.283, 4.465] |
| G9 每时隙，延迟 4 时隙 | 4.704 | 19.008 | +1.116，[0.560, 1.642] |
| G9 每 4 时隙，延迟 1 时隙 | 1.176 | 21.243 | +3.351，[2.747, 3.979] |
| G9 每 8 时隙，延迟 1 时隙 | 0.588 | 18.358 | +0.466，[−0.008, 0.901] |

G6−G9 的配对差仅 +0.054 Mbit/s，区间 [−0.058, 0.173]，不能声称 6-bit 更优。相同 G9 消息成本，延迟从 1 增至 4 时隙损失 2.757 Mbit/s，区间 [2.503, 2.973]。报告周期从 4 放宽到 8 时隙，损失 2.885 Mbit/s，区间 [2.443, 3.335]；后者相对年龄 OLLA 的改善已不稳健。

因此更有价值的研究变量是“某条质量信息何时仍足以支持 MCS”，而非继续把尾数从 9-bit 压到极低位宽。

前传审计：[directional_fronthaul.json](../results/exploration_20261007/information_audit_verified/directional_fronthaul.json)。每次报告打包同一 RU 的全部 PRG，列块字节对齐、含 8-bit 指数、66-byte 假设报文开销和 MTU1500 分包。G9 一次为 2352 wire bits。理想旧 CSI 按实际 complex64 无损上传计费，每 16 时隙一次，每 RU 4/8 天线分别为 0.834/1.668 Mbit/s；没有把理想 CSI 虚构成 H9。

旧 CSI 与新增报告都属 **RU→DU 上行**。两 RU full-grid U-plane、每时隙调度 C-plane 和每 16 时隙一次的 SE11 权值对应下行合计约 417.376 Mbit/s（N=4）。本轮没有减少这部分数据流，不能把额外上报说成下行前传节省。物理 SRS、空口 CQI/ACK、重传、基准消息头以外的管理流量和队列均未计入。

**部署动机的否决条件：** 若每 RU 可以提供额外约 3.6–4.7 Mbit/s 的上行并满足 0.5 ms 报告时延，普通 G6/G9 已接近 genie 参考。当前两 RU 小配置并未证明前传容量稀缺；仅以“前传有限”为由提出更复杂算法，动机不足。应先验证实际 SRS 获取/质量信息时延，或在真实多用户、多 RU 聚合容量中说明为何不能使用这条简单路径。

## 5. 两组外推检查与 SOPIP 管线

| 条件 | 年龄 OLLA | G9 每 4 时隙 | 提升 | 配对差 95% CI，Mbit/s |
|---|---:|---:|---:|---:|
| 每 RU 8 天线，理想本地 CSI | 27.893 | 29.135 | 4.5% | [0.663, 1.894] |
| 每 RU 4 天线，本地观测误差方差 0.01 | 18.156 | 20.908 | 15.2% | [1.997, 3.591] |

两组各使用相同 8 个配对 drops。误差组实际观测 NMSE 范围约 0.0083–0.0124；不代表物理 SRS 或互易校准误差。N=8 的 genie 参考为 30.153 Mbit/s，说明损失依赖空间负载，不宜用单一“失配幅度”概括所有配置。

SOPIP 独立实现采用作者 Algorithms 1/2 的上传延迟 u=1、下发延迟 d=3：CC 用收到的 H/V 迭代形成目标和公共残差，AP 用新鲜本地 H、收到的 V 及公共残差减去自己的旧 H 分量进行局部更新。把全部 UE 接收天线并入 H 行维度，是本轮的多 Rx 扩展。

正式参考使用**未量化、无限消息预算**，每时隙更新，CC/AP 各 20 次迭代，固定 alpha=512 并检查其覆盖评估 Gram 谱范数；目标使用与其他权值路线相同的全功率 global BD。AP 输出仅投影到每 RU 功率球 ≤0.5，不强制恢复功率等号。这是有效信道跟踪目标，非 goodput 最优化；未扫描到最优步长/迭代次数，未复现作者网络性能图。

该参考的普通 CSI-RS+OLLA goodput 为 12.063 Mbit/s，增加同一 G9 每 4 时隙报告后为 13.966，配对改善 1.903，[1.404, 2.376]；数据权值与 SE 保持相同。缓存为每 16 时隙更新的 SOPIP 变体为 9.364 Mbit/s，但这不是论文原始每时隙策略。

这些结果说明质量信息问题也可在另一类更新管线中出现。**不能用此结果宣称本地 ridge 在一般意义上优于 SOPIP**：目标函数、功率等号和步长选择都影响比较；本文也未完成动态 Team-MMSE 的同配置闭环。

## 6. 接收器信息的失效边界

入口：[explore_receiver_certificates.py](../scripts/explore_receiver_certificates.py)。16 个独立 drops，N=8、3 m/s，统计 64 个时隙；只做 SINR/SE 筛查，无空口 CQI 或 CRC。DU 由旧全局 CSI 和旧 W 形成接收方向，量化下发；RU 只投影自己的当前 G。两个端点使用相同解码后的参考方向，计入两方向消息。

普通投影点估计和三角不等式下界已经作为基线实现。另加入在旧 MMSE 接收子空间内重新优化接收器的更强基线，避免把“固定旧滤波器”当成唯一对手。同上行容量下，完整 G 使用更低位宽；投影 9-bit 对应完整 G 4-bit，投影 6-bit 对应完整 G 3-bit。该筛查按每 PRG 报文计费，不能直接复用第 4 节跨 PRG 打包后的 Mbps。

投影 9-bit 的正式结果：

| 估计方式 | 平均估计−当前最优 SINR，dB | 有某流高估的时隙比例 |
|---|---:|---:|
| 未量化旧接收方向 | −4.521 | 0% |
| 新鲜固定方向投影点估计 | −4.524 | 0%（本样本） |
| 新鲜固定方向量化下界 | −4.908 | 0% |
| 新鲜旧子空间内重新优化的下界 | −3.467 | 0% |
| 同上行容量完整 G4 点估计 | −0.522 | 99.02% |
| 上述子空间下界延迟 1 时隙 | −3.470 | 10.64% |
| 上述子空间下界延迟 4 时隙 | −3.470 | 62.60% |

高估事件是“一个时隙的任意 PRG/流超过当前最优 SINR 0.001 dB”，并非所有流的概率；完整 G4 量化点估计最严重高估约 7.01 dB。所有新鲜下界逐条验证不超过当前最优 SINR，延迟后的高估最大约 2.05/5.07 dB。

新鲜下界约束的是具有瞬时完美接收 CSI 的最优线性合并；实际 UE 的滤波器来自带噪声 DMRS 估计。因此它还不是实际接收机 SINR 或 CRC BLER 的保证，后续分析必须纳入该差异。

这给出两个需要区分的误差源：固定旧接收方向丢掉空间适应收益；延迟使原本正确的量化下界不再约束当前信道。完整 G 点估计较准，但也不能未经误差处理就叫“可靠证书”。

还有一个不依赖 Monte Carlo 的构造。单流、两 Rx、每 RU 功率 0.5，噪声 0.01，令 u=(0.1,0)、v=(0,sqrt(0.99))；RU2 的信道恒为 u+v，RU1 为 u+v 或 u−v。两世界 RU1 的信道范数相同，沿 (1,0) 的投影消息完全相同，但自适应接收器的 SINR 分别为 200 与 2，差 **20 dB**。见 [counterexample.json](../results/exploration_20261007/information_audit_verified/counterexample.json)。

一般地，只要固定线性投影 A 有非零核，u±v、v∈ker(A) 就能隐藏相干叠加所需的信息。这个反例只能排除“固定低秩投影普遍充分”；不能排除自适应/交互编码、受限信道族或概率保证，也不是新的通信复杂度定理。它提示下一步应研究接收子空间之外的决策相关补充信息，而不是只证明一个常规三角下界。

## 7. 可以收敛成论文的最小问题与待完成贡献

建议问题表述：**给定 RU 自主更新策略，当 DU 只有旧全局 CSI、旧 UE CQI 和延迟到达的有限 RU 消息时，怎样以最小额外质量信息支持可靠 MCS 决策？** “可靠”首先定义为模型内的条件 SINR/outage 约束；实际 BLER 必须再经接收机和 CRC 验证，不能由 Shannon SE 或 AWGN 门限直接推出。

每 RU 的私有信息限定为本地观测 H_l、自己实际采用的 W_l、已收到的 DU 版本和本地历史。编码器不能读取远端新鲜 CSI 或当前全局 SINR。DU 只能用已经到达、带时间戳与权值版本的消息及反馈。相干求和包含跨 RU 相位项，局部标量 SINR/范数一般不能直接相加。下发参考方向、版本确认、增量消息和触发通知也必须分别计费。

可争取的三项贡献，**以下均是下一阶段待完成内容**：

1. **决策所需信息的边界。** 定义相同 DU 信息下仍可成立的有效信道集合，研究该集合是否跨越 MCS 门限，以及私有 RU 状态、接收维度、空间负载、消息位数和信息年龄如何限制可保证的 MCS。把当前核空间反例推进为有明确假设的损失下界或位数/时延标度；不能把 elementary 反例本身当主定理。
2. **按决策不确定性补充信息。** 从普通 G6/G9、旧接收子空间和常规残差编码出发，仅在消息不能支持当前 MCS 时补充关键空间分量、提高更新频率或请求测量。量化误差、估计误差与老化误差分别处理；过去的确定性界加时间戳不等于当前保证。证明充分条件或条件风险控制，设计无需远端当前 CSI 的编码/触发规则。
3. **相同预算下的实际链路验证。** 固定波束，比较不同质量信息机制，报告首传 goodput、实际 CRC BLER、短时失败和消息成本；随后替换为 Team/SOPIP/较强局部预编码检查可迁移性。把预编码增益与观测增益分开，避免只靠 genie SE 得出结论。

不建议直接写“首次 receiver-aware task-oriented compression”：相邻文献已经很多。暂定工作题目可用 *Quality Observation for Decentralized Precoding With Delayed Fronthaul*，它只用于限定研究范围，不代表已确认的新颖性。

## 8. 下一阶段的顺序与停止条件

1. **先解决新颖性门槛。** 逐式比较上述三篇未获取全文的消息集合、接收 CSI、测量模型和可靠性目标；同时比较 flashlight、CoMP CQI 和任务量化。若已有方法完整覆盖本问题，沿用并研究其真正未覆盖的限制，不能靠改名立项。
2. **先验证是否真的需要复杂方法。** 加物理 SRS 或至少明确的 SRS 采样周期、噪声和因果外推，再加入互易误差/相位误差、实际外部干扰观测。比较报告延迟与 UE CSI-RS 反馈延迟一致时的收益，并扫描 0.5–2 ms 质量信息年龄。若普通 G6/G9 在目标部署中成本可承受并已充分解决问题，就停止“更省摘要”分支。
3. **同预算强基线优先。** 比较周期 G6/G9、差分 G、普通局部创新范数触发、预测 G、年龄/上下文 OLLA、稀疏非周期 CSI-RS，以及接收子空间内重优化。不能给新方法即时消息而给基线迟到消息；触发头与下发参考信息都计费。
4. **再实现决策相关方案。** 从接收子空间外的残差和 MCS 门限区间入手，处理不同 RU 报告年龄及不同权值版本；选择事件必须在 RU 私有信息和 DU 已到达历史内完成。先找出有价值的预算/时延区间，避免为本例的极小消息开销堆复杂控制器。
5. **最后扩展统计与负载。** 增加 32–64 独立 drops、较长轨迹、不同移动速度/SNR/空间负载和 4+ RU；当前 8 drops 的显著结果只支持初筛。无须先引入用户/层数联合优化，把单一问题做扎实。

可预先设定项目筛选目标，例如：在实际有依据的上行容量/时延区间，相比最强同预算周期/预测 G，上报成本减少约 25% 而 goodput 损失不超过 2%，或同成本下获得约 5% 的稳健提升，并且模型保证和实际 CRC 校准一致。这是研发筛选标准，不是 TWC 的统一录用门槛。如果完整 G、普通差分或简单触发已经做到这些，则必须进一步收窄理论问题或换方向。

## 9. 复现、核查与保留的负结果

正式统计：[causal_final_summary](../results/exploration_20261007/causal_final_summary/summary.csv)，[配对区间](../results/exploration_20261007/causal_final_summary/paired.csv)，[信息成本图](../results/exploration_20261007/causal_final_summary/quality_information_frontier.png)。汇总脚本：[summarize_dynamic_exploration.py](../scripts/summarize_dynamic_exploration.py)。接收器及额外配对分析源码和结果也在该目录。

正式 CRC 五个输入批次合计 **34,272 个预热后数据 TB**；不是 34,272 个独立信道样本。每配置只有 8 个独立 drops，各策略配对复用。置信区间采用 10,000 次按独立 drop 重采样的配对 percentile bootstrap，不把 UE、时隙或重叠失败窗口伪装成独立样本。

核查通过：

- [独立公式与固定点检查](../results/exploration_20261007/information_audit_verified/equation_validation.json)：上报矩阵的 SINR 与逐流干扰协方差公式一致；零老化未量化 SOPIP 在目标初值处保持固定点。另保留 float32 数值检查，不把数值一致性当作完整算法复现。
- [零移动实际 CRC 极限](../results/exploration_20261007/causal_static_crc_validation/validation.json)：一个静态 drop、每策略 48 条 UE 记录，旧权值/局部跟踪的 CQI、MCS、CRC、成功位数及 OLLA 逐条一致，SE 最大差约 1.91e−6。
- [接收摘要新鲜界检查](../results/exploration_20261007/causal_receiver_subspace_n8_16/validation.json)：固定方向与子空间界在实验中逐条通过；未对延迟报告声称确定性保证。
- 新增四个研究脚本通过 Python 语法编译；正式运行包含实际 CSI-RS、DMRS、LDPC 和 CRC。原有核心源码未修改。

早期诊断曾用第一 DMRS symbol 的 CSI 形成同一时隙 W，存在时隙内未来信息。发现后统一改为边界 symbol 0，并重跑正式 `causal_*` 批次；旧结果保留 `causality_audit.json`，排除最终统计。早期 SOPIP 目标功率归一化与其他方案不同，保留 `baseline_scope_audit.json` 并排除；正式 `causal_crc_sopip_ideal8` 使用一致目标。两个失败冒烟和被替代的旧运行保留 `failure.json`。

下面给出主结果的重跑入口。新目录必须不存在，避免覆盖原始数据；具体运行版本由各 run 内的 `experiment_source.py` 决定，不能用后来的新增参数误解释旧结果。

```bash
source env.sh
CUDA_VISIBLE_DEVICES=0 python scripts/explore_dynamic_measurement.py \
  --out results/stage3_reproduce_baselines --mode crc \
  --drops 8 --seed-start 20263000 --slots 128 --warmup 32 \
  --speeds 3 --antennas 4 --device cuda:0 \
  --policies stale16 track16 track16_ageolla track16_fastolla \
    track16_lowpass track4 pilot_locked16 track16_genie

CUDA_VISIBLE_DEVICES=1 python scripts/explore_dynamic_measurement.py \
  --out results/stage3_reproduce_reports --mode crc \
  --drops 8 --seed-start 20263000 --slots 128 --warmup 32 \
  --speeds 3 --antennas 4 --device cuda:0 \
  --policies track16_g6d1 track16_g9d1 track16_g9d4 \
    track16_g9e4d1 track16_g9e8d1

python scripts/audit_dynamic_information.py --out results/stage3_reproduce_audit
python scripts/summarize_dynamic_exploration.py \
  --runs results/stage3_reproduce_baselines results/stage3_reproduce_reports \
  --out results/stage3_reproduce_summary \
  --wire-audit results/stage3_reproduce_audit/directional_fronthaul.json
```

N=8、观测误差、SOPIP 和接收器筛查的完整参数保存在正式目录的 `metadata.json`。主汇总保存输入哈希清单；新增源码、核查、研究记录和补充分析的汇总审计见 [stage3_manifest.json](../results/exploration_20261007/stage3_manifest.json)。所有正式运行已经完成。
