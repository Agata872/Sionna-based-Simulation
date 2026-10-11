# 第八轮：CSI-RS 压缩文献调研，以及它在前传压缩与 DU–RU 协作中的位置

> **状态：已暂停（用户 2026-10-08 决定），先完成第一篇。** 恢复时从这里接着做：
> - 第二篇方案见第 5 节（URTS-PORT），哪些是经典结论、哪些需要证明见第 5 节的表；
> - 四个待决问题见第 6 节（FDD/TDD、CJT 还是单 RU 端口选择、UE 是否保持标准、Cai 等是否补进第一篇对照——最后一项已在第一篇框架中处理）；
> - 已下载并精读的文献在 `Paper reference/CSI-RS compression/` 与 `3GPP and ORAN report/`；
> - 尚未写任何第二篇代码，也没有第二篇实验结果。

整理日期：2026-10-08。本轮接续 [第七轮记录](oran_stage7_fair_quantization_final_method_2026-10-08.md)，只核查文献和标准，没有改代码，也没有跑实验。

阅读深度不一，表中逐条注明：
- 本地全文：O-RAN CUS v22 第 12.8 节（CDBF），TS 38.214 V18.2.0 与 V19.3.0 第 5.2 节的 CSI 条目，TR 38.843 V19.0.0 的 CSI 压缩评估结论。
- 用户下载后精读全文（2026-10-08 晚，`Paper reference/CSI-RS compression/`）：Anbukani 等 WCNC 2024、Choi–Love–Bidigare JSTSP 2014、Noh 等 JSTSP 2014、JSDM（Adhikary 等 TIT 2013）。精读后的修订集中在第 0 节第 6 条、第 2.3 节、第 3.1 与 3.5 节和第 5 节"哪些是经典结论"。
- Cai 等（arXiv 2508.05142）：读了 HTML 正文的方法部分。
- 其余：多数只读到摘要或检索摘要。

文中的"空白"是本次检索下的判断，没有穷尽证明。

## 0. 结论先行

1. **文献里的"CSI-RS 压缩"其实是四种对象。** 它们是：空口导频维度（端口数、密度）；UE 上报（码本或 AI 双边模型）；前传上与 CSI-RS 相关的载荷（导频 IQ、端口到阵元的波束权值、部分互易所需的 SRS 信道信息）；网络侧的 CSI 重建与下发。绝大多数工作只做前两种，而且是单基站。后两种基本只出现在 O-RAN 规范和少数 IITM 工作中。
2. **第一篇有一个需要补进对照表的新竞争者。** Cai、Zhang、…、Zeng（[arXiv 2508.05142](https://arxiv.org/abs/2508.05142)，2025-09）把场景划成 5 m 网格，对每个网格四个顶点上射线追踪的信道做 EVD，得到"环境特定信道基"（ECB），用来减少导频。3 m 定位误差下 NMSE 只增加 0.5 dB。这与 URTS 的"区域孪生子空间"思路接近。区别仍然清楚：它是单基站、导频受限的估计与预测；基按固定网格离线存储，维度固定为 15。它没有前传信源编码、码率分配和失配码率定律，也没有多 RU 和泄漏定律。
3. **和我们主题结合最紧的一点：端口波束由谁计算、另一端怎样知道，标准和学术都当成免费的。**
   - 3GPP：Rel-17/18 的端口选择码本（含 CJT）要求按上行角度和时延，对每个 CSI-RS 端口做空频波束赋形。
   - O-RAN CUS v22 的码本波束赋形（CDBF）把 PDSCH 权值写成 W = W_p2a·W_cb，O-DU 只发 PMI。但端口到阵元的映射 W_p2a 要么在 O-RU 预定义，要么由 O-DU 先下发权值来定义。CDBF 只支持 Type I、eType II 和 generic 码本，不支持 Rel-17 端口选择和 Rel-18 CJT。
   - 学术：FDD cell-free 方案（Kim–Choi–Shim、Zhang 等）都假设 CPU 能免费拿到各 AP 的上行角度或时延估计，而这恰好是第一篇在压缩的前传量。
4. **建议第二篇的方向（FDD 部分互易 cell-free O-RAN）。**
   - O-DU 与 O-RU 共享孪生，在下行频点合成区域子空间，用作 CSI-RS 端口波束。
   - O-RU 用新鲜的上行 SRS，只补孪生之外的少量"探测端口"，并只把这些端口的索引发给 O-DU。
   - UE 用标准端口选择码本或 CJT 码本上报。
   - O-DU 用同一组端口波束重建信道，再经 CDBF（generic 码本）下发预编码。

   与第一篇的结构差别在于残差不再可观测：UE 只看到导频照到的子空间。问题因此从"码率受限"变成"可观测性受限"，导频端口、上报比特和两个方向的前传比特要放进同一个账本分配。这是第二篇能独立成立的理论点。
5. **第二篇若留在 TDD，CSI-RS 的作用就只剩 CQI/干扰测量，以及 SRS 未探测到的 UE 接收天线。** 第一轮测得含 CSI-RS 的整个 OFDM 符号只占 U-plane 约 0.9%（[实验记录](oran_experiment_log_2026-10-06.md)），压缩价值弱。所以不推荐 TDD 作为第二篇的主设定，见第 6 节的待决问题。FDD 下情况不同：Anbukani 等按每天线一个非预编码端口（N = 32）、每时隙一次 CSI-RS 计算，CSI-RS 最多占前传数据的 19%；第一轮的 0.9% 只适用于低端口波束赋形 CSI-RS。
6. **精读四篇经典文献后的修正：第 5 节第 2 步本身不是新结果。** 已知信道协方差 R 时，单次训练的 MSE 最优导频就是 R 的主特征方向（Choi 等 Lemma 1）。用卡尔曼记忆时，序贯最优导频是预测误差协方差的主特征向量，而这些协方差与 R 共享特征向量，功率按注水分配（Noh 等 Proposition 1–3）。JSDM 的"统计预波束 × 瞬时预编码"两级结构，正是 O-RAN CDBF 的 W = W_p2a·W_cb。所以"孪生 KLT 方向作端口"只是把 R 换成孪生区域协方差 R̂。第二篇的新意必须落在 R̂ 的来源与失配（无需反馈或上行估计）、失配下的可观测误差底（Choi 等"天花板效应"在 R̂ ≠ R 时的推广）、cell-free 泄漏驱动的跨 RU 端口分配，以及端口波束不经前传这四点上。

## 1. 四种"CSI-RS 压缩"

| 压缩对象 | 典型做法 | 节省的资源 | 是否经过 7.2x 前传 | 与本课题的关系 |
|---|---|---|---|---|
| A 空口导频维度 | 波束赋形 CSI-RS、统计预波束（JSDM）、主动信道稀疏化、端口选择码本、学习导频、孪生/CKM 子空间、降低端口密度 | CSI-RS 占用的 RE | 端口波束要过前传：Cat B 为 C-plane 权值，Cat A 为天线域 IQ | 第二篇主体 |
| B UE 上报 | Type II、eType II、端口选择、CJT、Doppler 码本；AI 双边模型；孪生训练的自编码器 | PUCCH/PUSCH 比特 | 不经过：UCI 在 O-DU 译码，前传上只是 PUSCH 的 IQ | 用标准码本作接口，不作为贡献主体 |
| C 前传上的 CSI-RS 相关载荷 | CSI-RS 结构复用、调制压缩、SE 19 紧凑多端口波束信息、CDBF、SRS 信道信息上送 | 前传比特 | 经过 | 与第一篇共用"共享先验 → 少传"的机制 |
| D 网络侧 CSI 重建与下发 | 集中式 CPU 重建、分层 CSIT、Team 预编码 | 前传与回传 | 经过：O-DU→O-RU 的权值或协作信息 | DU–RU 协作的信息结构问题 |

## 2. 标准现状

### 2.1 3GPP

本地 TS 38.214 V18.2.0 已核对条目；Rel-19/20 和 6GR 的内容来自检索结果。

- **Rel-16 eType II**（5.2.2.2.5）：W = W1·W̃2·Wf^H。空间域取 L 个 DFT 波束，频域取 M 个 DFT 基，只上报部分非零系数。这相当于由 UE 完成的角度-时延压缩。
- **Rel-17 FeType II 端口选择**（5.2.2.2.7，typeII-PortSelection-r17）：基站用上行 SRS 估计角度和时延，对每个 CSI-RS 端口同时做空间和频域的波束赋形。端口数 P_CSI-RS ≤ 32，UE 在其中选择端口；频域基 M ∈ {1, 2}，参数由 paramCombination-r17 给出。这是"FDD 部分互易下压缩 CSI-RS"的标准形态。
- **Rel-18 CJT**（5.2.2.2.8，以及端口选择版 5.2.2.2.9）：配置 N_TRP ∈ {1,…,4} 个 CSI-RS 资源，每个最多 32 个端口。UE 可以用位图只选其中 N_0 个资源上报，各资源的阵列参数（N1, N2）相同。Rel-18 另有 Doppler 码本（5.2.2.2.10/11，用于预测 PMI）。
- **Rel-19**（本地 TS 38.214 V19.3.0，2026-04 已核对）：
  - 端口扩展：Type I 单/多面板、eType II 和 Doppler 的 r19 码本（5.2.2.2.1a、2a、5a、11a）支持 48、64、128 个端口，做法是把 2–4 个同端口数的 CSI-RS 资源拼接。端口选择码本 typeII-FePortSelection-r19（5.2.2.2.9a）支持 48 或 64 端口：2 个资源各 24/32 端口、3 个资源各 16 端口，或 4 个资源各 16/32 端口。
  - CJT 校准上报：CJTC-Dd（时延偏差）、CJTC-F（频偏）、CJTC-P（相偏），见 5.2.1.4.9–11。它们对应非理想同步下的多 TRP 相干传输。
  - 另有 CSI-PAI（CSI 预测精度指示，5.2.1.4.6）和 UE 发起的上报（5.2.1.5.4）。
  - 来源：本地规范；另见 [RAN1 #118bis FL summary](https://3gpp.org/ftp/Meetings_3gpp_SYNC/RAN1/Inbox/drafts/9.2(NR_MIMO_Ph5)/9.2.2%20CSI/Round%204/DRAFT%20R1-2409060%20Rel-19%20MIMO%20CSI%20Round%204%20-%20VFinal.docx)。
- **Rel-20（5G-Advanced）**：
  - AI/ML 双边 CSI 压缩已转入规范性工作（WI NR_AIML_air_Ph2，2025-06 的 RAN#108 立项），见 [Release 20 综述](https://arxiv.org/pdf/2506.11828)。
  - 据[二手汇总](https://apexstandards.com/2026-04-04.3GPP.pre-meeting/publish/3GPP.RAN1.124-bis.Saint-Julians-Malta.html)，MIMO Phase 6 在讨论 48/64/128 端口时降低 CSI-RS 频域密度，以及"early CSI"。正式结论未核到。
- **6GR 研究**：RAN1 议程把 CSI 分为基于下行导频的获取、基于上行导频的获取和其他三项。公司提案的端口数为 128–256 甚至更多，也有用 DMRS 辅助获取 CSI 的提案。本次没有找到已同意的 CSI 结论。

### 2.2 O-RAN CUS v22（本地全文）

- **第 12.8 节 CDBF。**
  - O-RU 内部计算 W = W_p2a·W_cb。O-DU 每层每子带只发码本描述（SE 35），不发权值。
  - W_p2a 由 p2aBeamId 指定，可以在 O-RU 预定义，也可以由 O-DU 先用 ST 1/SE 1 下发权值来定义。
  - 支持的码本：Type I 单面板（两种模式）、eType II（R = 1、2）和 TYPE-GENERIC（任意 2·N1·N2 维权值向量，用 cbBeamId 传）。
  - 规范说明 CDBF 的价值在于 O-DU 不必按子带、按层发送权值。
- **p2aMappingType 0001b** 允许不同 PRB 块使用不同的端口到阵元映射，所以能表达 Rel-17 那种带频域赋形的端口。但每个块的波束仍须事先定义。
- **CDBF 没有覆盖的部分**：没有 Rel-17 端口选择码本和 Rel-18 CJT 码本对应的 CDBF 类型；多 O-RU 协同的 CJT 也不在 CDBF 范围内。
- **SE 19** 为多端口 CSI-RS 提供紧凑的波束信息表示。规范用 32 端口 CSI-RS 作例子，说明它能节省 section 开销。
- **SRS 信道信息上送**（7.2.12 节、ST 6，含 beamspace type II 压缩）是第一篇的压缩对象。FDD 端口选择需要的角度和时延，也只能通过这条路径或在 O-RU 内部处理获得。

### 2.3 TR 38.843 V19.0.0：AI/ML CSI 压缩的评估结论（本地全文，6.2.2.14、6.2.2.19–20）

- **相对 Rel-16 eType II 的收益不大。** 1 对 1 联合训练、训练与测试同场景（generalization Case 1）时，平均 UPT 增益与负载有关：资源利用率 RU ≤ 39% 时，最大秩 1/2/4 分别为 0.2–2%、−0.3–6%、−4–6%；RU ≥ 70% 时分别为 0.23–9%、−0.2–15%、−1–17%。FTP 业务下，4 个来源报告上报开销减少 8–79%。
- **输入宜用预编码矩阵，不宜用原始信道。**
- **"本地化模型"（按小区或区域训练，即站点特定）有稳定收益。** 同复杂度下，SGCS 比全局模型高约 1–19%（Option 1，即在本区域内撒点）。一个来源用约 27% 参数、1% FLOPs 即与全局模型相当或更好（payload Y 时高 11.9%）。
- **合成数据可用。** RAN1 观察到：用合成数据训练的参考模型，无论是否用现场数据微调，都能保证最低性能要求。
- **对本课题的含义有三点。**
  - UE 上报压缩是拥挤的方向，标准评估中增益有限，不宜作为贡献主体。
  - "站点特定先验有用"已是 3GPP 评估中的共识，再次说明"用孪生"本身不构成卖点，与第五轮的结论一致。
  - 标准的 KPI 是 SGCS 和 UPT，与 eType II 比较。第二篇如果要和 AI 方案对照，应沿用这套口径。

## 3. 学术文献

### 3.1 统计/协方差驱动的导频降维（第二篇最近的经典前序）

| 工作 | 已经做了什么 | 与第二篇的区别 | 阅读状态 |
|---|---|---|---|
| Adhikary, Nam, Ahn, Caire, "Joint Spatial Division and Multiplexing—The Large-Scale Array Regime", IEEE TIT 2013 | 预编码 = 只依赖二阶统计的预波束 B_g × 基于降维"有效信道"的瞬时多用户预编码。下行训练和上报维度都降到有效维度（例如有效维度 10 时，每个用户只上报 10 维向量）。给出了 JSDM 相对全 CSIT 无损的条件；对大规模 ULA，用 Szegő 定理证明 DFT 预波束近似最优，只需粗略的到达角和角度扩展 | 结构上就是 CDBF 的 W = W_p2a·W_cb。我们的预波束由两端从孪生后验算出，不经前传；扩展到多 RU 的 CJT；失配时由 O-RU 用 SRS 补探测端口 | 全文已读 |
| Khalilsarai, Haghighatshoar, Yi, Caire, "FDD Massive MIMO via UL/DL Channel Covariance Extrapolation and Active Channel Sparsification", IEEE TWC 2019（[arXiv 1803.05754](https://arxiv.org/abs/1803.05754)） | 由上行估计角度散射函数，再外推下行协方差；主动稀疏化让固定导频维度 T_dl 也能估计多用户信道 | 孪生直接在下行频点合成，替代上行外推；我们还加入位置不确定、cell-free 和前传 | 摘要 |
| 同组扩展：[1710.07993](https://arxiv.org/abs/1710.07993)、[2104.13309](https://arxiv.org/abs/2104.13309)（DP-UPA 多维稀疏化） | 同上 | 同上 | 摘要 |
| Choi, Love, Bidigare, "Downlink Training Techniques for FDD Massive MIMO Systems: Open-Loop and Closed-Loop Training With Memory", IEEE JSTSP 2014 | 1. Lemma 1：已知 R 时，单次训练的最优导频沿 R 的主特征方向。2. Lemma 3"天花板效应"：训练长度 T < M 固定时，接收 SNR 随天线数增加而饱和。3. 他们认为基站拿不到 R（反馈 R 的开销随 M² 增长），因此让 BS 与 UE 事先共享一组低秩训练信号：开环按轮转发送，UE 用卡尔曼记忆估计；闭环由 UE 反馈下一次训练信号的索引（B bit），按 MSE 或 SNR 选择，训练信号集用 Grassmann 子空间设计 | 孪生让基站不经反馈就得到 R̂，消除了他们的核心约束。他们的天花板效应在 R̂ ≠ R 时的推广，就是我们 T5 的可观测误差底。"UE 反馈下一个导频索引"可作为 O-RU 选探测端口的 UE 侧对照 | 全文已读 |
| Noh, Zoltowski, Sung, Love, "Pilot Beam Pattern Design for Channel Estimation in Massive MIMO Systems", IEEE JSTSP 2014 | 高斯–马尔可夫信道加卡尔曼滤波。Prop. 1：序贯最优导频波束是预测误差协方差的主特征向量。Prop. 2：在这种设计下，所有误差协方差与 R 共享特征向量，所以导频序列就是在 R 的特征方向上排程。Prop. 3：每个特征方向的功率集中在该方向在时隙中最后一次使用时，跨方向按注水分配。假设基站已知 R（由上行估计或反馈得到） | 第 5 节第 2–3 步（KLT 方向作端口、按特征值分配）是它在 R = R̂ 下的特例。可借用的是"跨 CSI-RS 时机轮转孪生特征方向 + O-DU 卡尔曼跟踪"，用较少的每时机端口数覆盖更多维度 | 全文已读 |

### 3.2 部分互易与 cell-free FDD

| 工作 | 已经做了什么 | 与第二篇的区别 | 阅读状态 |
|---|---|---|---|
| Kim, Choi, Shim, "Downlink Pilot Precoding and Compressed Channel Feedback for FDD-Based Cell-Free Systems"（[arXiv 1909.07117](https://arxiv.org/abs/1909.07117)） | 各 AP 由上行导频得到出发角，据此预编码下行导频，只反馈主路径增益；反馈减少 80% 以上 | 没有计入 AP→CPU 传角度和导频权值的前传 | 摘要 |
| Zhang, Du, Ding, Jing, Huang, "Joint Port Selection Based Channel Acquisition for FDD Cell-Free Massive MIMO", IEEE TCOM（[arXiv 2307.10730](https://arxiv.org/abs/2307.10730)） | 跨 AP 联合端口选择，用端口相关性做 EVD 变换；贪心算法和深度学习两种实现 | 同上；端口基来自上行测量而非孪生 | 摘要 |
| Kim, Choi, Park, Alkhateeb, Lee, "FDD Massive MIMO: How to Optimally Combine UL Pilot and Limited DL CSI Feedback?"（[arXiv 2405.08614](https://arxiv.org/abs/2405.08614)） | 上行信息作边信息，只反馈路径相位；MSE 最优联合重建和比特分配 | 单基站；边信息来自上行而非孪生；无前传 | 摘要 |
| Lin, Lee, Ding（[arXiv 2205.14558](https://arxiv.org/abs/2205.14558)） | 部分互易下的波束化导频与学习反馈 | 单基站，学习方法 | 摘要 |

### 3.3 学习导频与站点特定导频

| 工作 | 已经做了什么 | 与第二篇的区别 | 阅读状态 |
|---|---|---|---|
| Sohrabi, Attiah, Yu, "Deep Learning for Distributed Channel Feedback and Multiuser Precoding in FDD Massive MIMO", IEEE TWC 2021（[arXiv 2007.06512](https://arxiv.org/abs/2007.06512)，有代码） | 把多用户估计与反馈视为分布式信源编码，端到端联合学习导频、反馈比特和预编码 | 理论视角与我们最接近；我们用孪生给出可解释的结构化解，并计入前传 | 摘要 |
| Heng, Andrews 等，[Learning Site-Specific Probing Beams](https://arxiv.org/abs/2107.13121)、[Grid-Free MIMO Beam Alignment](https://arxiv.org/abs/2209.08198) | 学习站点特定的探测波束，用少量探测预测最优波束 | 毫米波波束管理，不是完整 CSI；学习得到，而非来自孪生 | 摘要 |
| Gadamsetty, Hari, Hanzo, "Learning a Common Dictionary for CSI Feedback in FDD Massive MU-MIMO-OFDM Systems", IEEE OJVT 2023（[arXiv 2307.15943](https://arxiv.org/abs/2307.15943)） | BS 与 UE 共享学习得到的字典，反馈减少 50% | "两端共享字典"本身已有；它的字典来自数据，不随位置后验变化 | 摘要 |

### 3.4 孪生、CKM 与无线地图

| 工作 | 已经做了什么 | 与我们的区别 | 阅读状态 |
|---|---|---|---|
| Cai, Zhang, Yu, Zhang, Zhang, Shi, Qiu, Zeng（[arXiv 2508.05142](https://arxiv.org/abs/2508.05142)） | 每个 5 m 网格取 4 个顶点 × 10 个时刻 = 40 个射线追踪样本做 EVD，取 15 维基；TDD 上行只在部分子载波上发导频；导频开销最多省 50% | 单基站估计；网格和维度固定；无前传、码率和多 RU | HTML 方法部分 |
| Zheng, Chen, "A Radio Map Approach for Reduced Pilot CSI Tracking in Massive MIMO Networks", IEEE TSP 2025（[arXiv 2410.05803](https://arxiv.org/abs/2410.05803)） | 把无线地图嵌入切换卡尔曼滤波，联合跟踪位置与 CSI；每 10 ms 一个导频即达到完美 CSI 容量的 97% | 位置与 CSI 联合跟踪与第六轮的闭环定位相近，可作参照 | 摘要 |
| Wu, Ho, "Electromagnetic-Twin Channel Estimation: From Sparse Pilots to Persistent CSI"（[arXiv 2609.27894](https://arxiv.org/abs/2609.27894)，2026-08） | 递归同步的持久信道状态，只用 20% 导频 | 单链路估计 | 摘要 |
| Luo, Jiang, Khosravirad, Alkhateeb（[arXiv 2509.25793](https://arxiv.org/abs/2509.25793)，TCOM 2025）；Luo, Khosravirad, Alkhateeb, "Wireless Digital Twin Calibration: Refining DFT-Domain Channel Information"（[arXiv 2603.16126](https://arxiv.org/abs/2603.16126)）；Huang, Wen 等（[arXiv 2505.00660](https://arxiv.org/abs/2505.00660)） | 孪生生成数据训练 CSI 反馈，孪生保真度按四个方面分解；在 DFT 域校准孪生输出，以码本 CSI 反馈为例；真实测量验证 | 孪生在这里是训练集，我们则把它用作两端共享的结构先验。2603.16126 的 DFT 域校准可作为第三篇（CKM 在线校正）的前序 | 摘要 |
| Haider 等（[arXiv 2609.27017](https://arxiv.org/abs/2609.27017)） | 射线追踪锚点加 Transformer 做时间内插 | 时间内插，不涉及导频降维或前传 | 摘要 |
| Zeng 等，CKM 教程，IEEE COMST 2024（[arXiv 2309.07460](https://arxiv.org/abs/2309.07460)）；位置不确定下用 CKM 做多用户分层波束训练（[arXiv 2511.22902](https://arxiv.org/abs/2511.22902)） | 用 CKM 实现免训练或少训练的波束 | 第五轮已列入对照 | 已列 |

### 3.5 前传侧

| 工作 | 已经做了什么 | 阅读状态 |
|---|---|---|
| Anbukani, Harivignesh, Giridhar, "Exploiting CSI-RS Structure for Efficient Fronthaul Functional Split in 5G O-RAN and Low Complexity Channel Estimation", WCNC 2024 | 1. 对象：FDD 非预编码 CSI-RS，端口数等于天线数（N = 32）。现有 7.2x 把一个 CDM 组的 L 个端口打包成 L 层，O-RU 用预定义的单位块矩阵映射到天线。2. 所提 "7-2x+"：O-DU 每个 CDM 组只发一次 QPSK 基序列（相当于 L = 1），O-RU 用预定义的块对角 Walsh 矩阵作波束权值来施加正交码，其余零 RE 用 selective RE sending 去掉。3. 效果：CSI-RS 前传量降为原来的 1/L，L = 2/4/8 时节省 49%、75%、87%（100 MHz、N = 32、每时隙一次时从 2236 降到 283 Mbit/s）；O-RU 运算减少 67–93%。4. 自称 CSI-RS 最多占前传数据 19%，因此最多省 16.6%；这是按每时隙都发 CSI-RS 计算的上限。5. 另给出 UE 侧利用 Walsh 蝶形结构的快速 LS 估计。不改 3GPP 和 O-RAN 规范 | 全文已读。它只去掉了码域冗余，剩余的前传量仍随端口数 N 增长；它不降端口维度，也不涉及波束赋形端口的权值。我们的端口降维（N → P）加上孪生共享权值，与它正交、可以叠加。它的参考文献 [8] Cavalcante 等 ICC 2019（beamspace 波束系数降前传）、[9] Hewavithana 等 WCNC 2022（O-RAN 前传下的信道老化）值得补读 |
| Modulation compression（[arXiv 2011.03734](https://arxiv.org/abs/2011.03734)），以及 O-RAN 的 ModComp | 按星座压缩数据和参考信号 | 作为波形层基线 |

### 3.6 信息结构

| 工作 | 已经做了什么 | 与我们的关系 | 阅读状态 |
|---|---|---|---|
| Miretti, Björnson, Gesbert, Team MMSE（TWC 2022）；Miretti, Stańczak 2024 | 不同节点掌握不同 CSI 时的最优线性预编码 | 第二轮已列 | 已列 |
| de Kerret, Fritzsche, Gesbert, Salim, "Robust Precoding for Network MIMO with Hierarchical CSIT", ISWCS 2014（[arXiv 1406.7708](https://arxiv.org/abs/1406.7708)）；Atzeni, Gesbert, "Cooperative MIMO Precoding with Distributed CSI: A Hierarchical Approach"（[arXiv 1802.09820](https://arxiv.org/abs/1802.09820)）；[arXiv 2004.11918](https://arxiv.org/abs/2004.11918) | 分层 CSIT：精度高的节点也知道精度低的节点掌握的信息；计入信息交换的成本 | O-DU 有 UE 上报（全局，但有延迟、经过量化），O-RU 有新鲜的本地 SRS，正好构成一种分层信息结构 | 摘要 |
| Caire 等, "Channel State Acquisition in FDD Massive MIMO: Rate-Distortion Bound and Effectiveness of 'Analog' Feedback"（[arXiv 2202.01515](https://arxiv.org/abs/2202.01515)） | FDD CSI 获取的率失真界 | 第二篇的理论可以从这里出发 | 摘要 |

## 4. 空白判断

- **G1：端口波束由谁算、谁需要知道。** Rel-17/18 端口选择要求端口波束来自上行角度和时延。在 7.2x 中只有两种做法：一是把 SRS 信道信息送到 O-DU，这就是第一篇压缩的前传量；二是 O-RU 本地计算波束后告诉 O-DU，否则 O-DU 无法用 UE 上报的系数重建信道。CDBF 不覆盖这两种码本。本次检索没有发现把这部分前传计入成本、或用共享先验消除它的工作。
- **G2：导频降维与 cell-free 目标脱节。** Choi、Noh、ACS 和孪生 ECB 以单基站 MSE 为目标；JSDM 以单小区和速率为目标，但假设协方差已知、组间近似正交。而 cell-free 集中式预编码的损失主要来自泄漏（第五到七轮的 γ），端口应该覆盖哪些方向由跨 RU 的泄漏决定，不由单链路 MSE 决定。
- **G3：可观测性。** 第一篇中 O-RU 拿到的是完整信道，孪生之外的成分可以用残差层补上。第二篇中 UE 只能看到导频照到的子空间，孪生之外的成分不额外花端口就永远看不到。这时孪生失配（漫散射、材料、几何）直接变成误差底，不再是多花几个 bit 就能解决的问题。第七轮已表明，有漫散射时孪生端口只够达到 80–88%，所以必须有测量驱动的端口，这正是 O-RU 新鲜 SRS 的用处。
- **G4：信息结构可以作为设计变量。** 分层 CSIT 和 Team 理论都把信息结构当成给定条件。没有找到在共享先验下，把导频维度、上报比特和两个方向的前传比特一起分配的工作。

## 5. 第二篇候选算法：URTS-PORT（暂名）

**设定。** FDD（部分互易）；L 个 O-RU，各 M 个阵元；K 个 UE；Cat B（端口波束在 O-RU 执行）；CJT。两端共享 CKM 和 UE 位置后验。第一篇和第六轮的闭环定位照用，因为 FDD 下上行 SRS 仍然存在。

**两端（确定性计算，不需信令）：**

1. 按位置后验抽 S 个区域样本，在下行载频上用孪生合成每条链路（UE k, O-RU l）的空频信道，做 KLT，得到基 U_kl 和特征值 λ_kl。孪生直接在下行频点合成，不需要 ACS 那样从上行外推协方差。
2. 把 U_kl 的主方向转成端口波束：每个端口取一个空间波束，再按主时延加频域相位预补偿，使 UE 只用 M ∈ {1, 2} 个频域基就够。这样与 Rel-17/18 端口选择码本兼容。
3. 端口预算分配：
   - 约束是每个 O-RU 的端口上限（每个资源最多 32 个，CJT 最多 4 个资源）和 UE 上报比特上限。
   - 按第五轮的一阶模型 Σ_k log₂(1 + a(E_k − D_k)/(1 + γ c D_k)) 贪心地给链路加端口。这里 D_k 等于端口子空间之外的能量（不可观测部分）加上子空间内的估计与量化误差。
   - 同一 O-RU 下位置相近的 UE 共享端口（JSDM 式分组），端口数按它们子空间的并集维度计。

**O-RU l：**

4. 用最新的上行 SRS 估计本链路信道，计算落在孪生端口子空间之外的能量，称为创新量。若创新量超过门限，就在角度-时延 DFT 网格上选 n_l 个"探测端口"；部分互易保证角度和时延在上下行一致。只把这些格点的索引发给 O-DU，每个约 10 bit；第七轮下一步中的组合编码同样适用。这是第一篇残差层在下行的对应物，代价从"系数比特"变成"索引加端口 RE"。
5. 发送 CSI-RS，端口波束为 [孪生端口, 探测端口]，由 O-RU 本地生成。前传上 O-DU 只发未预编码的 CSI-RS 符号。

**UE：**

6. 用标准端口选择码本上报：单 O-RU 用 typeII-PortSelection-r17，多 O-RU 用 typeII-CJT-PortSelection-r18，不改 UE。可选的非标准扩展：按孪生特征值给各端口系数分配量化位宽。

**O-DU：**

7. 译码 UCI。自行复算孪生端口，加上收到的探测端口索引，重建各链路信道，做集中式 RZF 或 MMSE。
8. 经 CDBF 下发预编码，使用 TYPE-GENERIC 码本：每层每子带只发 P 维端口系数（cbBeamId）。W_p2a 就是两端都已知的端口波束，O-RU 自己合成 W = W_p2a·W_cb。下行权值的规模从 M×层数×子带数降到 P×层数×子带数，且 P ≪ M。这一步需要扩展 CDBF：让 p2a 映射由共享孪生按确定规则生成，而不是查预定义表或逐次下发权值。论文中必须写明这一点。

**哪些是经典结论，哪些需要我们证明（精读后补充）：**

| 步骤 | 经典来源 | 我们的增量 |
|---|---|---|
| 第 2 步：取协方差主特征方向作导频 | Choi 等 Lemma 1；Noh 等 Prop. 1–2 | 协方差是孪生按位置后验合成的 R̂，不需要 UE 反馈或长期上行估计；端口要符合 Rel-17/18 端口选择码本（带每端口频域预补偿，M ≤ 2） |
| 第 3 步：按特征值分配导频 | Noh 等的注水功率；JSDM 的有效维度与用户分组 | 目标从单基站 MSE 换成 cell-free 泄漏模型（γ）；约束换成每资源端口上限、CJT 资源数和 UCI 比特 |
| 第 8 步：两级预编码 W_p2a·W_cb | JSDM 的两级结构；CDBF 已标准化这种形式 | W_p2a 由两端从孪生复算，不经前传；需要扩展 CDBF |
| 第 4 步：O-RU 用 SRS 补探测端口 | 部分互易端口选择（Rel-17）；Kim–Choi–Shim 的角度预编码导频 | 只为孪生之外的创新部分付端口和索引；这一划分的最优性就是 T5 |
| 可观测误差底 | Choi 等 Lemma 3 的天花板效应（R 已知、T < M） | 推广到 R̂ ≠ R：误差底由孪生失配决定，这是第二篇的理论核心 |

**理论问题：**

- **T5 可观测性受限的失真。** 端口子空间之外的能量 ε_out(P) 构成误差底。它推广了 Choi 等的天花板效应：那里 R 已知、误差来自 T < M；这里端口沿 R̂ 选择，误差底是真实 R 落在 R̂ 端口子空间之外的能量。要推导孪生端口与探测端口的最优划分怎样随孪生失配和 SRS SNR 变化；孪生失配就是第一篇 T1 中 R̂ 与 R 的差异。直觉是：孪生准确时端口全部给孪生；失配大时，探测端口的边际收益超过孪生端口。
- **T6 统一账本。** 成本 = CSI-RS RE + UCI 比特 + 前传上行（探测端口索引）+ 前传下行（端口系数）。基线是"O-DU 收到 SRS 信道信息后自己算端口"，它的前传上行正是第一篇压缩的量。
- **T7 分层信息结构。** O-RU 掌握 SRS 和孪生；O-DU 掌握 UE 上报、孪生和探测端口索引。与 Team MMSE 和分层 CSIT 对照，说明共享孪生怎样把需要交换的信息缩小到创新部分。

**与第一篇的统一。** 同一组区域孪生子空间有三种用途：
- 上行 O-RU→O-DU 的信源编码基（第一篇）；
- 下行空口的模拟投影，即 CSI-RS 端口，投影由 UE 完成；
- 下行前传的 CDBF 端口映射。

三者只差在残差是否可观测。

**基线（同等资源计费）：**

- (a) Rel-17 端口选择，端口波束由 O-DU 用上行 SRS 信道信息计算。SRS 信道信息走前传，可以用第一篇的方法压缩。
- (b) Rel-18 CJT eType II，每个 O-RU 用 32 个非预编码端口。
- (c) JSDM/ACS 及 Noh 等的特征导频，协方差由上行估计并外推。它与本方法只差 R 的来源，是检验"孪生是否值得"的关键对照。
- (c') 在 (c) 的基础上，再叠加 Anbukani 等的 7-2x+ CSI-RS 波形传输，用来公平计算非预编码 32 端口基线的前传量。
- (d) Cai 等的网格 ECB 端口。
- (e) Kim–Choi–Shim 的角度预编码导频。
- (f) 孪生数据训练的双边自编码器（Luo 等），作为数据驱动参照。

**评价。**
- 指标：总频谱效率和 5% 用户频谱效率随总成本的变化。
- CRC/BLER 用现有的 CSI-RS→CQI→MCS 链路。
- 孪生失配沿用第七轮的慕尼黑扰动集。

## 6. 待用户决定

1. **第二篇的双工方式。** 推荐 FDD 部分互易：此时 CSI-RS 是下行 CSI 的唯一来源，压缩价值最大。也可以留在 TDD。选 FDD 的话，孪生要在下行频点重算（现在 `oran/config.py:67` 是 3.5 GHz TDD），并选一个 FDD 频段对，例如 n1（上行 1920–1980 MHz，下行 2110–2170 MHz）。
2. **主码本。** 推荐 Rel-18 typeII-CJT-PortSelection-r18，直接对应 cell-free。也可以先做单 O-RU 的 Rel-17 端口选择，再扩展。
3. **UE 是否保持标准不变。** 推荐保持；非标准的端口系数位宽分配只作扩展。
4. **是否把 Cai 等（2508.05142）补进第五轮笔记中第一篇的新颖性对照表。**

## 7. 文献获取

**已由用户下载并精读（2026-10-08）：**
- `Paper reference/CSI-RS compression/`：Anbukani 等 WCNC 2024；Choi–Love–Bidigare JSTSP 2014；Noh 等 JSTSP 2014；Adhikary 等 TIT 2013（JSDM）。
- `3GPP and ORAN report/`：TS 38.214 V19.3.0（`ts_138214v190300p.pdf`）；TR 38.843 V19.0.0（`38843-j00.pdf`）。

**可选补读（Anbukani 等的参考文献，与权值前传和 DU–RU 协作有关）：**
1. A. Cavalcante, I. Almeida, E. Medeiros, M. Berg, "A Novel Model of Beamspace Beamforming Coefficients for Fronthaul Load Reduction," IEEE ICC 2019.
2. T. Hewavithana, A. Chopra, B. Mondal, S. Wong, A. Davydov, M. Majmundar, "Overcoming Channel Aging in Massive MIMO Basestations With Open RAN Fronthaul," IEEE WCNC 2022.

有 arXiv 版本的文献，我自己读即可，不在此列。
