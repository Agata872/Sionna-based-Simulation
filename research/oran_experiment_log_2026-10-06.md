O-RAN 探索实验记录：CSI-RS、权值表示与双 RU 信息时效

日期：2026-10-06。使用用户授权的 GPU0/GPU1（两块 Tesla T4）。完成 8 批正式实验、2 次小规模入口验证。保持原有目录结构，未修改 `oran/` 或 Sionna 源码；新增实验与分析脚本。当前结果属于选题筛查，尚未形成可宣称新颖性的 TWC 问题。

**本轮决定：降低“通用 CSI-RS IQ 有损压缩”的优先级，保留“面向有效信道的权值表示”，优先继续“有限协作信息与 RU 局部跟踪”。** 决定依据是开销构成、实际 CRC 和局部强基线结果，而不是只看 EVM 或理论 SE。旧候选“局部刷新必然造成难以补偿的质量预测失配”也被弱化：简单有效信道跟踪在轻度老化时已能大幅消除这一诊断风险。

所有原始数据、配置、源码快照、日志、CSV、统计及 PNG/PDF 位于 [results/exploration_20261006](../results/exploration_20261006/)。[manifest.json](../results/exploration_20261006/manifest.json) 包含运行参数和原始数据/源码 SHA256；[environment.json](../results/exploration_20261006/environment.json) 保存版本；[baseline_validation.json](../results/exploration_20261006/baseline_validation.json) 保存实验前现有测试通过记录。源码快照逐一与元数据中的 SHA256 核对一致。

既有 `test_link.py` 8 项测试在 GPU0 通过，`test_closed_loop.py` 12 项在 GPU1 通过，后者包含 RT smoke。正式实验中的信道、输入比特和噪声按配置配对；一个 drop 内所有 UE 一起做 bootstrap，避免把协同用户当成独立重复。95% 区间使用 3000 次 drop-cluster 重采样，是探索性估计，没有多重比较校正。零误块结果不证明总体 BLER 为零。

| 运行目录 | 独立 drop/条件 | 作用 |
|---|---:|---|
| `cdl_c_mu_numerical` | 128 | 8 Tx、2 UE×4 Rx、每 UE 2 层，CSI-RS 符号与权值精度分离 |
| `cdl_c_mu_crc` | 128 | MCS14 的实际 DMRS、LDPC、CRC 对照 |
| `cdl_c_mu64_numerical` | 128 | 64 Tx 的阵列规模复核 |
| `cdl_c_mu_mcs24_crc` | 256 | MCS24 的实际译码复核 |
| `cdl_c_mu_power_control` | 512 | 所有表示恢复相同列功率后的机制复核 |
| `cdl_c_mu_power_control_crc` | 256 | 相同权值比特/列功率、MCS24、20 dB 的实际译码复核 |
| `two_ru_information_screen` | 256 | 双 RU 的 CSI 时效、权值量化与简单局部混合筛查 |
| `two_ru_tracking_screen` | 256 | 添加局部有效信道跟踪强基线，检验先前现象能否被简单方法解决 |

以上“条件”包括不同 SNR/精度/算法；同一数据集跨条件复用 drop。10 次运行共保存 463168 行 UE/配置观测，其中正式 CRC 实验有 48384 次 TB 译码评价，包含重复配置，不能把这个数当作独立样本数。

**发现一：当前低端口 CSI-RS 配置的波形压缩空间有限。** 按 51 PRB、4 层、BFP9、PRG=2、CSI-RS 周期 8 时隙计算，整个 CSI-RS 所在符号约占 U-plane payload 的 0.893%。这包含该符号上的数据和零 RE，是该符号全部传输成本的比例，不能等同于 CSI-RS 净开销或标准 codec 的可实现节省。当前 CSI-RS/data 共用 W，所以没有重复计算额外的 W_RS。

| 发射天线 | U-plane payload | 共享权值 payload | 解析估计的总前传（含 U/C-plane 头部） |
|---:|---:|---:|---:|
| 8 | 1.279488 Gbit/s | 0.031616 Gbit/s | 1.372800 Gbit/s |
| 32 | 1.279488 Gbit/s | 0.121472 Gbit/s | 1.465856 Gbit/s |
| 64 | 1.279488 Gbit/s | 0.241280 Gbit/s | 1.588864 Gbit/s |

权值按每时隙刷新计算，改变刷新周期会改变比例；总前传为现有解析模型，未实现包队列、截止时限及完整 SE19 打包。[前传构成图](../results/exploration_20261006/cost_inventory.png)；[开销扫描原始 JSON](../results/exploration_20261006/cdl_c_mu_numerical/cost_inventory.json)。

导频 IQ 的实验按完整 CSI-RS-bearing OFDM 符号分区，每块仍为 12 RE，包含与导频同符号的数据。没有对任意导频 RE 免费切换量化指数。8 Tx、30 dB 下，导频符号 BFP9 相对理想测量的平均增量 SINR 偏差约 −0.0035 dB；BFP3/4 约 +0.348 dB。数据权值和数据 IQ 保持理想，因此这些实验中的数据 oracle SE 不随导频符号量化改变，这是设计上的控制条件，不是吞吐改善证据。

该结果仅覆盖每 UE 两个波束赋形端口的配置。不能推广为所有 8/16/32 端口宽波束 CSI-RS 场景都无研究价值。若后续研究高端口探测、多波束扫掠或独立 W_RS，需扩展 mapper 并重新核算。

**发现二：BFP9 在当前静态链路中的损失很小，极低位宽的收益高度依赖 MCS。** 静态 CDL-C、理想发射 CSI、30 dB 下，普通 BFP9 权值相对浮点理想权值的平均 oracle sum-SE 损失为：8 Tx 约 0.0263%，64 Tx 约 0.0368%。这两个数字不包含 CSI 获取误差、老化、不同服务集合或闭环 MCS。

MCS14 在 20/30 dB 下，即便 3-bit 权值有明显 oracle SE 损失，128 个 drops 中所有相关配置仍无首传误块，因而固定 MCS 吞吐没有下降。MCS24、20 dB 下则出现明显差异：理想权值 BLER=9.77%，普通 BFP3=36.13%，BFP5=11.52%，BFP9=10.35%。BFP9 与理想的配对吞吐差区间包含零，现有样本未确立其损失。

研究含义：不要围绕“修复默认 BFP9 的巨大损失”设计主张；也不能把高 SNR 下的理论 SE 差直接换算成 NR 吞吐收益。低位宽能节省多少完整前传、节省是否值得牺牲链路质量，仍要结合调制压缩、权值占比和自适应 MCS。

[导频与权值精度图](../results/exploration_20261006/cdl_c_mu_numerical/pilot_vs_weights.png)；[MCS24 CRC 图](../results/exploration_20261006/cdl_c_mu_mcs24_crc/crc_weight_baselines.png)。

**发现三：压缩目标会影响实际吞吐，功率差异无法解释全部差异。** 使用两种非常简单的诊断表示：对每个 PRG/层尝试 32 个列相位，BFP 重建后分别选择权值相位对齐 NMSE 最小的结果，或全局信道加权误差 ||H(Q(W exp(jθ)) exp(−jθ)−W)||² 最小的结果。改变列相位在理想预编码中不改变 SINR；实际 CSI-RS、DMRS 与数据全部使用所选重建权值，不需要另传去旋转相位。

再增加 RU 端按已知每层功率恢复列范数的对照，保证每列功率与原预编码一致。严格功率对照中，8 Tx、3-bit、30 dB、512 个数值 drops：

| 方法（均恢复列功率） | 相位对齐权值 NMSE 均值 | 相对理想 oracle sum-SE 损失 |
|---|---:|---:|
| 普通 BFP | 0.04026 | 16.11% |
| 按权值 NMSE 选择相位 | 0.02653 | 14.06% |
| 按有效信道误差选择相位 | 0.03249 | 11.78% |

随后独立的 256 个 CRC drops，固定 MCS24、20 dB，同为 3-bit、相同列功率：

| 方法 | 首传 BLER | 两 UE 合计首传吞吐 |
|---|---:|---:|
| 普通 BFP＋范数恢复 | 37.30% | 39.182 Mbit/s |
| 权值 NMSE 选择＋范数恢复 | 29.30% | 44.187 Mbit/s |
| 有效信道目标选择＋范数恢复 | 23.44% | 47.849 Mbit/s |

有效信道目标相对普通 BFP 的直接配对吞吐增益约 8.67 Mbit/s（约 22.1%），95% 区间约 [6.84, 10.74] Mbit/s；相对 NMSE 目标的增益约 3.66 Mbit/s（约 8.3%），区间约 [2.08, 5.37] Mbit/s。精确区间保存在 [objective_comparisons.json](../results/exploration_20261006/cdl_c_mu_power_control_crc/objective_comparisons.json)，与[严格功率 CRC 图](../results/exploration_20261006/cdl_c_mu_power_control_crc/crc_weight_baselines.png)对应。

这支持“评价权值表示应看有效信道与实际译码”的研究方向。相位搜索、信道加权量化和量化感知预编码都有相邻先例；该算法只是筛查用强基线，不能作为已确认的新方法。实验使用理想全局 CSI、固定 MCS、极低位宽、较小带宽，没有证明在完整前传约束下或面对已发表量化感知方案仍有优势。

**发现四：双 RU 场景中，信息年龄比 BFP9 精度更值得关注，但简单局部混合并不可靠。** 这部分明确使用抽象模型：2 RU×4 Tx、2 UE×4 Rx、每 UE 2 层，相关 Rayleigh 信道，H_new=ρH_old+√(1−ρ²)E，ρ 与用户相关性分别扫描。所有重建权值恢复为每 RU 功率 0.5。ρ 是信道相关系数，不是毫秒；没有真实 SRS、CSI-RS 测量、CQI 或 CRC。

20 dB、用户相关性 0 时：

| ρ | 新鲜全局参考 sum-SE | 过时全局＋BFP9 sum-SE | 相对新鲜参考损失 |
|---:|---:|---:|---:|
| 1.00 | 27.789 | 27.786 | 0.0104% |
| 0.99 | 27.870 | 26.009 | 6.68% |
| 0.95 | 28.052 | 23.778 | 15.24% |
| 0.85 | 28.111 | 21.647 | 22.99% |

单位为 bit/s/Hz；新鲜全局 BD 为不可部署参考，不是证明过的上界。单纯混合旧全局 W 与新鲜局部 RZF，在某些低老化条件下反而比旧全局 W 更差。局部 RZF 还使用收到的旧 DU 权值进行列相位对齐，相关消息按完整权值计费，未宣称是零协作成本的纯局部方案。

**发现五：应把局部有效信道跟踪作为下一阶段的强基线。** RU 保存自己的旧 CSI 与收到的 BFP9 权值，并用新鲜本地 CSI 解一个局部 ridge 问题：

min_Δ ||H_new,l(W_cached,l+Δ) − H_old,l W_cached,l||²_F + λ||Δ||²_F。

最终恢复每 RU 功率约束。该更新只使用本地旧/新 CSI 和实际收到的压缩 W，没有使用新鲜全局 CSI，也没有让 RU 获得未量化 DU 权值。固定测试 λ∈{0.01,0.1,1,10}；未根据同一测试集声称找到了可泛化的最优 λ。

在上述 20 dB、用户相关性 0 的条件下，λ=1：

| ρ | 旧全局＋BFP9 sum-SE | 局部跟踪 sum-SE | 配对 SE 增益 95% 区间 | 旧质量预测高估 >3 dB 的比例：旧全局→跟踪 |
|---:|---:|---:|---|---|
| 0.99 | 26.009 | 26.954 | [0.854, 1.035] | 7.23%→0.39% |
| 0.95 | 23.778 | 25.345 | [1.387, 1.735] | 50.78%→23.44% |
| 0.85 | 21.647 | 23.229 | [1.326, 1.840] | 72.46%→56.64% |

这里的质量预测是旧信道、旧权值的 genie SINR，相对于当前实际有效信道计算值的差；不能称为真实 UE CQI 的误差或 MCS 失败率。零老化时，该跟踪对 BFP9 基线的最大平均 SE 差约 1.5×10⁻⁷，符合不改变静态基线的极限检查。

[包含跟踪的双 RU 图](../results/exploration_20261006/two_ru_tracking_screen/information_age_screen.png)；[全部相关性/ρ/λ 结果](../results/exploration_20261006/two_ru_tracking_screen/summary.csv)。

重新阅读 [SOPIP 作者全文](https://www.comm.toronto.edu/~liang/publications/ToWC_HierarchicalSemiOnlineInformationParsing.pdf) 后，必须谨慎处理这项结果：该工作已基于有效信道偏差和精简全局信息支持局部更新。上述 ridge 跟踪与其特定目标/信息结构相近，并非完整 SOPIP 实现，也不能当作新颖算法。下一阶段要把 SOPIP、[Team-MMSE](https://arxiv.org/abs/2104.15027) 及其[作者代码](https://github.com/emilbjornson/team-MMSE)纳入同信息与预算的比较，再讨论新增约束是否带来独立贡献。

**下一阶段的明确问题与停止条件。** 优先探索：在估计误差、有限更新和有限消息位宽下，DU 传完整 W、有效信道目标或干扰摘要，RU 分别能保留多少局部跟踪/全局协调收益；哪些消息维度决定质量预测可靠性。CSI-RS 用于实际测量验证，权值表示作为对照。当前尚未宣称这个问题无人研究。

接下来按以下顺序推进：

1. 核对并复现强前序的原始适用模型，避免把本文的 multi-antenna 扩展直接标成原算法。分析 ridge 跟踪与 SOPIP 信息摘要之间的关系。
2. 固定一个处理架构与用户集合，加入 RU CSI 估计误差、CSI 获取成本和明确的消息到达时间。保持双向前传预算分别计费；同步/互易误差在多 RU 相干传输中单独扫描。
3. 将跟踪接回现有 CSI-RS→CQI→MCS→DMRS/CRC，保留 OLLA、oracle MCS 与接收机对照，确认 genie 风险能否转化为真实可改善的闭环问题。
4. 在相同消息预算下比较完整权值与低维目标/摘要；补充 modulation compression 和合适的 beamspace 方案，防止只在 U-plane 占绝对主导的旧基线下优化小开销。
5. 若强前序或简单跟踪＋常规 OLLA 消除收益，及时停止该分支；若剩余损失只出现在不合理精度/免费 CSI 条件下，也停止。只有出现稳定机制及可推导边界，才进入论文算法和理论设计。

目前没有完成物理移动轨迹、实际 SRS 估计、包级截止时限或完整因果闭环。下一阶段最需要解决这些信息可得性与强基线问题，不能把增加仿真维度本身当成 TWC 贡献。

复现时先 `source env.sh`，选择 `CUDA_VISIBLE_DEVICES=0` 或 `1`。每个目录的 `metadata.json` 给出原始 args，`experiment_source.py` 为运行时源码，`analysis_source.py` 为汇总源码。当前入口：`scripts/explore_csirs_precoding.py --mode numerical|crc --out 新目录` 与 `scripts/explore_du_ru_information.py --out 新目录`；默认不覆盖已有结果。摘要入口：`scripts/summarize_csirs_exploration.py 目录...` 和 `scripts/summarize_du_ru_information.py 目录`。双 RU 分析的 bootstrap 辅助函数依赖前一个摘要模块，已另存为 `analysis_dependency.py`。
