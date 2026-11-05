# oran_baseline 实验索引

早期（2026-10-04 ~ 05）的运行见 [README](README.md) 的结果目录表。之后的运行在这里登记：目的、命令、运行目录、状态和结论。

## F1：cell-free 下四种功能切分的前传负载（2026-11-05）

**目的。** 在同一个 cell-free 下行中比较 option 7-1、7.2x Cat A、7.2x Cat B、option 8 的前传负载，以及各自的压缩方案在
CQI 查表选 MCS 的闭环中带来的吞吐代价；画出 UE 均衡后的星座图。

**设置。** TR 38.901 UMi-Street Canyon 随机撒点（Sionna `tr38901.UMi`，表 V16.1.0 = 仓库中的 V18.0.0；§7.6.9 绝对到达时间），
不用射线追踪。4 个 O-RU（2×2 方格，间距 200 m = UMi ISD，高 10 m）× 8 TRX，每个 O-RU 最大 35 dBm / 20 MHz，UE 噪声系数 9 dB；
4 个静止室外 UE × 2 天线 × 1 层；51 PRB、30 kHz；全协作集中式 BD，每 O-RU 功率约束（整体缩放），O-DU 按天线功率设定定点增益。
CSI-RS 经各切分自己的前传 → CQI（表 2）→ MCS 查表（MCS 表 2，关闭 OLLA）。每个撒点 4 个 CSI-RS 周期（1 测量时隙 + 7 数据时隙），
所有配置用同一撒点和同一随机种子。负载含 eCPRI 与以太网头，按每 O-RU 链路计，另换算到 100 MHz（273 PRB）。

压缩：7-1 与 option 8 为均匀定点 16/10/8/6 bit；Cat A 为 BFP 9/7/5、块缩放 5、μ 律 5；Cat B 同上（权值同方法同位宽），
另加调制压缩（CUS 附录 A.5）配 BFP9 / BFP7 权值；参照为理想前传。

**命令。** 在 VS Code 中运行 [`split_fronthaul_load.ipynb`](split_fronthaul_load.ipynb)（`NUM_DROPS = 20`，`SEED = 2026`）。

**运行目录。** `results/split_fronthaul_load_20261105/run_115332_drops20_seed2026/`（metadata.json、源码快照、rows.jsonl 共 420 行、
complete.json 含汇总、fig_*.png）。notebook 中保存的输出就是这次运行。

**状态。** 完成（20 个撒点 × 21 种配置，T4 上 12.5 分钟）。

**结论。**（20 个撒点 × 4 个 UE；理想前传时 CQI 15 占 55/80，其余在 10–14；均衡后 SINR 中位数 29 dB，主要受 PRG 内 BD 残余干扰限制）

| 切分 | 吞吐不低于理想前传 99% 的最小负载（100 MHz，4 条 O-RU 链路合计） | 每条链路 |
|---|---|---|
| option 7-1 | 8 bit 均匀定点：48.8 Gbit/s（吞吐 99.9%） | 12.2 Gbit/s |
| 7.2x Cat A | BFP7：45.2 Gbit/s（99.7%） | 11.3 Gbit/s |
| 7.2x Cat B | 调制压缩 + BFP7 权值：11.7 Gbit/s（100%） | 2.9 Gbit/s，其中 C-plane 0.18 |
| option 8 | 8 bit 均匀定点：65.1 Gbit/s（100%；6 bit 为 98.9%） | 16.3 Gbit/s |

1. **Cat B + 调制压缩的负载约为其他切分的 1/4**（11.7 对 45–65 Gbit/s），且无损。原因有两个：每个 O-RU 只传 4 层而不是 8 根天线；
   每个 RE 只要 Qm bit，而不是约 15–16 bit。负载随 MCS 变化，但本场景中多数 UE 用 256QAM，每条链路只在 2.79–2.96 Gbit/s 之间。
   Cat B 不用调制压缩时，BFP7 为 23.2 Gbit/s（99.8%），BFP5 吞吐只剩 79%。
2. **7-1 与 Cat A 满负荷时几乎一样**（8 bit 均匀定点 16 bit/RE，BFP7 为 14.7 bit/RE 加报头），差别只来自压缩方法。7-1 的真正劣势是
   与调度无关：PRB 占用率低时 Cat A 负载成比例下降，7-1 不变（notebook 第 6 节）。
3. **option 8 比同位宽的 7-1 多约 1/3（100 MHz）**：option 8 每根天线 122.88 Msps（4096 点 IFFT，含保护带和 CP），7-1 只传
   3276 个子载波 × 28000 符号/s = 91.7 M RE/s（20 MHz 时两者之比为 1.8）。但时域量化噪声分摊到全部 IFFT 点，20 MHz 仿真中带内 SQNR
   比同位宽的 7-1 高约 2 dB，所以 6 bit 时损失更小（98.9% 对 98.1%）。
4. **5 bit 是失真的转折点，且 CSI-RS 基本看不到它**（纯查表、无 OLLA，失真直接变成误块）：
   - Cat B BFP5：BLER 0.215，ΔCQI 仅 −0.05；层符号本身被量化，5 bit 的 BFP 步长比 256QAM 点距还粗，星座点被吸到量化网格上（见星座图）；
   - Cat A BFP5：BLER 0.038，ΔCQI −0.33，配对 UE 的失真经 NZP-IMR 进入 CQI，所以 MCS 有所下调；
   - μ 律 5 bit 在 Cat A 中最差（吞吐 85%），块缩放 5 bit 最好（97–96%），三者比特数相同。
5. **吞吐损失集中在未达到最高 MCS 的 UE**：CQI 为 15 的 UE 的 SINR 高于该档门限、有余量；其余 UE 的 SINR 正好卡在所选 MCS 的门限附近，
   失真一来就出错（例：Cat A BFP5 两类 UE 的 BLER 分别为 0.005 和 0.110）。
6. 口径核对：7-1、8 天线、2×16 bit、100 MHz 时这里为 24.3 Gbit/s，TR 38.816 原式为 21.5 Gbit/s；差别来自子载波密度（NR 30 kHz 下
   3276 个，TR 按 LTE 每 20 MHz 1200 个计）和 eCPRI 报头。

**局限。** 全协作（无用户中心服务簇），满缓冲、所有 PRB 都调度，静止 UE，无 HARQ；只撒室外 UE；每个 UE 1 层；纯 CQI 查表，
有 OLLA 时误块会部分转成 MCS 下调。下一步可做：用户中心服务簇（负载随撒点变化）、PRB 部分占用、每 UE 2 层（需双符号 DMRS）。
