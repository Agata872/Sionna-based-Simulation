# O-RAN 7.2x 下行链路级仿真（Category A / B）

基于 Sionna PHY（PyTorch 版 2.2.0）实现 O-RAN 7.2x 功能划分下的 PDSCH 下行链路，比较
Category A（预编码在 O-DU）与 Category B（预编码在 O-RU）在前传 IQ 压缩下的性能与前传带宽。
另有闭环模式：用 Sionna RT 对物理环境做射线追踪，UE 随机撒点，按 CQI 选 MCS 后进行下行传输。

规范版本以 `3GPP and ORAN report/` 中的文档为准：TS 38.211 V18.6.0、TS 38.212 V18.5.0、
TS 38.214 V18.2.0（Rel-18），O-RAN WG4 CUS v22.00（O-RAN.WG4.TS.CUS.0-R005-v22.00）。
已实现的部分（附录 A 的三种压缩算法、udCompMeth / bfwCompMeth 编码、段类型 1 与 SE 11 的报文格式）
与此前使用的 v08.00 相同。

## 仿真范围

两种模式共用同一条 PDSCH 链路（O-DU → 前传 → O-RU → 信道 → UE）。

### 开环链路级（`ORANDownlink`，CDL 信道，扫描 SNR）

只涉及物理层：从 MAC 经 FAPI 交下来的传输块（TB）开始，到 UE 的 CRC 校验为止。
RRC / PDCP / RLC / MAC 不参与仿真，只体现为配置参数（MCS、层数、RNTI、DMRS 端口等），信息比特用随机比特代替。

```
O-DU（High-PHY）
  TB CRC → 码块分割 + CB CRC → LDPC → 速率匹配 → 加扰 → 调制 → 层映射 → RE 映射（数据 + DMRS）
  ├─ Cat A：→ 预编码 → 前传 N_TRX 路天线端口流
  └─ Cat B：→ 前传 L 路层流 + C-plane 预编码权值
前传（7.2x，eCPRI）
  U-plane：16 bit 定点 → 按 PRB 压缩（BFP / 块缩放 / μ 律）→（传输）→ 解压
  C-plane（Cat B）：预编码权值按 PRB bundle 压缩（Section Extension 11，方法同上）
O-RU（Low-PHY）
  [Cat B：按权值逐 RE 预编码] → [时域：PRB 映射到 IFFT 中间 → IFFT + CP] → [DAC：预留]
信道
  每个 (UE, O-RU) 一条独立的 3GPP CDL 链路 + AWGN
UE
  [时域：去 CP + FFT] → DMRS 信道估计 → LMMSE 均衡 → 软解调 → 解层映射 → 解扰/速率解匹配/LDPC 译码 → CRC
```

不压缩时 Cat A 与 Cat B 天线口信号逐比特相同（单元测试中验证），两者的差别只来自前传：

| | Cat A | Cat B |
|---|---|---|
| U-plane 承载 | N_TRX 路天线端口流（已预编码） | L 路层流 |
| 压缩噪声注入位置 | 天线端口上，空间白色，不经预编码 | 层上，随信号一起被预编码（也随 BD 一起对其他 UE 置零） |
| C-plane | 只有段描述 | 另带 N_TRX × L 预编码权值（可压缩） |

另有 3GPP split option 8（`category="8"`，TR 38.801 §11.1.2.8）：O-DU 做完预编码后再做 IFFT + CP，前传承载 N_TRX 路
时域 IQ，没有 C-plane，O-RU 只剩射频。3GPP 不规定 option 8 的接口与压缩（实际由 CPRI 实现），默认按每个 I/Q 分量
w bit 的均匀定点量化（`CompressionConfig("fixed", w)`，削波电平按位宽取最优），量化噪声落在全部 IFFT 点上（含保护带）。
实现与比特率见 `fronthaul.py` 的 `TimeDomainCodec`、`option8_rate`，以及 `../cqi_compression/README.md` 的 option 8 一节。
理想前传时 option 8 与 Cat A 逐比特相同。

### 闭环（`ClosedLoopDownlink`，Sionna RT 信道，随机撒点）

在上面的链路外加上 UE 的 CSI 测量上报和 gNB MAC 的链路自适应。各层的处理方式如下：

| 层 | 内容 | 依据 | 本仿真 |
|---|---|---|---|
| 物理环境 | 场景几何与材料、O-RU / UE 位置、天线方向图 → 每条链路的 CIR | TR 38.901 §8（射线追踪）、§7.3（阵元方向图） | `rt_channel.py`，Sionna RT 2.2（GPU） |
| 链路预算 | 发射功率、路径增益、kTB·NF | — | SNR 由位置决定，不再扫描 |
| L1（下行） | 测量时隙：每 UE 一个预编码 NZP CSI-RS（CMR）、配对 UE 的 CMR 兼作 NZP-IMR、CSI-IM；与 PDSCH 一起经前传发送 | TS 38.211 §7.4.1.5；TS 38.214 §5.2.1.4.1、§5.2.2.3.1、§5.2.2.4、§5.1.4.2 | `csi_rs.py`（设计第 10 条） |
| L1（UE） | CSI 测量 → 宽带 CQI，reportQuantity = 'cri-RI-CQI'（非 PMI 反馈） | TS 38.214 §5.2.2.1、§5.2.2.5.1、§5.2.1.4.2 | `csi_rs.py` 估计 + `csi.py` 算 CQI；精灵测量作对照 |
| L1（上行） | CQI 经 UCI 上报 | TS 38.212 §6.3 | 无误码、无时延 |
| MAC | CQI → MCS、OLLA | 3GPP 不规定算法；MCS 表见 TS 38.214 §5.1.3.1 | `link_adaptation.py` |
| L1（下行） | DCI 1_1 指示 MCS；PDSCH | TS 38.212 §7.3.1.2.2；原有链路 | DCI 只体现为参数 |
| RRC | cqi-Table、mcs-Table、CSI-ReportConfig | TS 38.331 | 只是配置参数 |
| 初始接入 | SSB、PRACH、RRC 建立 | — | 不仿真，假设 UE 已处于 RRC_CONNECTED |

```
撒点：覆盖区内均匀随机取室外位置 → 射线追踪 → 各 (UE, O-RU) 信道（含链路预算），UE 静止
每个 CSI-RS 周期（1 个测量时隙 + batch_size 个数据时隙，默认共 8 个时隙）：
  O-DU：由 CSI（TDD 互易，假设理想）计算 BD 预编码
  测量时隙：PDSCH（在 CSI-RS / CSI-IM 处打孔）+ 预编码 CSI-RS + CSI-IM
            → 前传（Cat A/B、压缩）→ O-RU → 信道 → UE
  UE：CMR 估信道、NZP-IMR 估配对 UE 的干扰、CSI-IM 估噪声 → 每 PRB 的 LMMSE SINR → EESM → CQI → UCI 上报
  MAC：CQI → SINR 估计 + OLLA 偏移 → MCS → DCI
  数据时隙：PDSCH：O-DU → 前传 → O-RU → 信道 → UE → CRC
  ACK/NACK → OLLA
```

## 快速开始

逐模块演示完整闭环流程（带开关，可切换信道、Cat A/B、压缩、SU/MU、CQI 来源等）：
[`notebooks/closed_loop_walkthrough.ipynb`](../notebooks/closed_loop_walkthrough.ipynb)。

```bash
cd "/home/user/Sionna-based Simulation"
source env.sh
python -m unittest discover -s tests -v          # 单元测试，约 45 秒
python scripts/compare_cat_a_b.py --widths ideal 9 5 --snr 16 30 2 --max-batches 5   # 快速试跑
python scripts/compare_cat_a_b.py --widths ideal 7 bs:7 mu:7 --bfw-width bs:9        # 比较压缩算法

# 完整扫描（Cat A/B × 压缩位宽 × SNR）：两块 GPU 各跑一类，再合并画图
# Sionna 的随机种子按设备序号偏移，用 CUDA_VISIBLE_DEVICES 让两个进程都在 "cuda:0" 上运行，
# 两边的比特、信道、噪声才完全相同（配对比较）
CUDA_VISIBLE_DEVICES=0 python scripts/compare_cat_a_b.py --categories A --out results/cat_a &
CUDA_VISIBLE_DEVICES=1 python scripts/compare_cat_a_b.py --categories B --out results/cat_b &
wait
python scripts/compare_cat_a_b.py --merge results/cat_a results/cat_b --out results/cat_ab

# 闭环：munich 场景随机撒点，CQI → MCS → OLLA（结果与图在 results/rt_su、results/rt_mu）
python scripts/rt_closed_loop.py --drops 5 --batches 8                 # 快速试跑（单 UE）
python scripts/rt_closed_loop.py --num-ue 1                            # SU-MIMO，30 个撒点
python scripts/rt_closed_loop.py --num-ue 2                            # 2 UE MU-MIMO（BD）
python scripts/rt_closed_loop.py --num-ue 2 --category B --uplane bfp:7 --bfw bfp:7 --out results/rt_mu_b7
python scripts/rt_closed_loop.py --cqi-source genie                    # 对照：精灵 CQI（不发 CSI-RS）

# 闭环中比较 Cat A/B 的压缩失真：先生成撒点文件，两块 GPU 各跑一类（同一批撒点上依次跑各压缩配置），再合并
python scripts/rt_compare_compression.py --num-ue 2 --save-drops results/drops_mu.json
CUDA_VISIBLE_DEVICES=0 python scripts/rt_compare_compression.py --categories A --drops-file results/drops_mu.json --out results/csi_mu_a &
CUDA_VISIBLE_DEVICES=1 python scripts/rt_compare_compression.py --categories B --drops-file results/drops_mu.json --out results/csi_mu_b &
wait; python scripts/rt_compare_compression.py --merge results/csi_mu_a results/csi_mu_b --out results/csi_mu
```

在代码中使用：

```python
import sionna.phy
from oran import ORANConfig, ORANDownlink, CompressionConfig

sionna.phy.config.device = "cuda:0"
cfg = ORANConfig(category="B", mcs_table=2, mcs_index=20,
                 uplane=CompressionConfig("bfp", iq_width=9),   # U-plane 压缩
                 bfw=CompressionConfig("bfp", iq_width=9))      # Cat B 权值压缩
sim = ORANDownlink(cfg)
results = sim.simulate([18, 20, 22, 24], batch_size=16, max_batches=20, seed=42)
rate = sim.fronthaul_rate(num_prb=273)        # 100 MHz 下单个 O-RU 的前传比特率

# 闭环
from oran import ClosedLoopDownlink, LinkAdaptationConfig
cfg = ORANConfig(channel_model="rt", num_ue=2, mcs_table=2)
cfg.rt.tx_power_dbm = 35.0                    # 场景、O-RU 位置等见 RTConfig
sim = ClosedLoopDownlink(cfg, LinkAdaptationConfig(bler_target=0.1))
drops = sim.simulate(num_drops=10, num_batches=16, batch_size=8, warmup_batches=4, seed=1)
print(drops[0].summary(cfg.slots_per_second))
```

## 默认配置（`oran/config.py`）

| 项目 | 默认值 |
|---|---|
| 载波 | 30 kHz SCS，51 PRB（20 MHz），3.5 GHz |
| 节点 | 1 个 O-RU，8 个发射通道（1×4 双极化）；2 个 UE，各 4 根天线（1×2 双极化） |
| PDSCH | 每 UE 2 层、1 个码字；MCS 表 1、序号 14（对比脚本默认用表 2 序号 20，即 256QAM） |
| DMRS | Type 1，单符号，附加位置 1（符号 2、11）；UE0 端口 0/1，UE1 端口 2/3，2 个 CDM 组不放数据 |
| 预编码 | 块对角化（BD），PRG = 2 PRB，O-DU 已知理想 CSI |
| 前传 | U-plane BFP 9 bit；Cat B 权值 BFP 9 bit；16 bit 定点，回退 15 dB；MTU 1500 |
| 信道 | CDL-C，时延扩展 100 ns，静止 UE，各链路 BS 方位在 ±60° 内随机旋转 |
| UE | DMRS LS 估计（PRG 内局部线性 CDM 分离 + PRG 内线性插值），LMMSE 均衡 |
| 仿真域 | 频域（可切换为时域：O-RU 用 1024 点 IFFT，30.72 Msps） |

闭环模式另有（`RTConfig`、`LinkAdaptationConfig`）：

| 项目 | 默认值 |
|---|---|
| 场景 | Sionna 内置 munich；最多 3 次反射/绕射，含绕射与折射 |
| O-RU | 位于 (8.5, 21, 30) m，即 26 m 楼顶加 4 m 抱杆；朝北、下倾约 11°，覆盖北侧广场和几条街道 |
| 链路预算 | O-RU 35 dBm（20 MHz）、UE 噪声系数 9 dB（TR 38.901 表 7.8-1 的 UMi 校准值） |
| 撒点 | 距 O-RU 10~250 m，单阵元路径增益 ≥ −135 dB，室外，UE 高 1.5 m，静止 |
| CSI-RS | 符号 12；每 UE 一个 ν 端口 CMR（ν = 2 时为表 7.4.1.5.3-1 第 3 行，fd-CDM2，密度 1），用该 UE 的 BD 权值预编码；配对 UE 的 CMR 兼作 NZP-IMR；CSI-IM pattern1；powerControlOffset 0 dB；2 UE × 2 层时每 PRB 子载波 0–3 为 CMR、4–7 为 CSI-IM |
| CSI / LA | CQI 来自 CSI-RS（`cqi_source="csi-rs"`，可改为 `"genie"`）；CQI 表与 MCS 表同号（脚本默认表 2）；目标 BLER 0.1；OLLA 步长 0.5 dB，范围 ±10 dB |
| 周期 | CSI-RS 周期 8 个时隙 = 1 个测量时隙 + 7 个数据时隙（共用一个 MCS）；每个撒点 16 个周期，前 4 个（OLLA 收敛）不计入统计 |

## 指标

- **SNR**：每个 RE 的总发射功率为 1（所有 O-RU 所有天线之和），每对收发天线的平均信道增益为 1，
  SNR = 1/N0，N0 为每根接收天线每个 RE 的噪声方差。
- **BLER**：按 TB 的 CRC 统计，每个 UE 分别统计；吞吐量 = (1 − BLER) × TBS × 每秒时隙数。
- **接收 EVM**：UE 均衡后的数据符号相对发送符号的误差（含噪声、干扰、信道估计误差和压缩失真）。
- **发射 EVM**：O-RU 天线口频域信号相对"理想前传"（不量化、不压缩）的误差，只反映前传失真。
- **U-plane SQNR**：前传 IQ 经定点化 + 压缩后的信号与量化噪声功率比。
- **前传比特率**：单个 O-RU 链路的 U-plane + C-plane 比特率，含 eCPRI 和以太网开销
  （见 `fronthaul.py` 中头部长度的说明）。100 MHz、每层不压缩时为 3276 × 14 × 2000 × 32 = 2.93 Gbps，
  与书中数值一致。option 8 为天线数 × 采样率（IFFT 点数 × 子载波间隔）× 2w，按 eCPRI IQ 数据消息加以太网开销。
- **宽带 SNR（闭环）**：P_RE·mean|h|²/N0，按每根接收天线、各发射天线等功率计算，不含波束赋形和接收合并增益。
- **吞吐量（闭环）**：warmup 之后数据时隙中首传成功的 TB 比特数除以数据时隙时长（没有 HARQ 重传；测量时隙的
  PDSCH 视为与数据时隙相同）。
- **数据时隙实测 SINR**：数据时隙上均衡后的发送符号功率与误差功率之比，包含噪声、干扰、信道估计误差和前传失真，
  是 PDSCH 实际经历的 SINR。CSI 测得的有效 SINR 与它之差反映 CQI 的高估量。
- **ΔCQI**：CSI-RS 测得的 CQI 与精灵 CQI（真实等效信道、不经前传）之差。
- **INR（NZP-IMR）**：UE 在配对 UE 的 CSI-RS 上测得的干扰功率与 CSI-IM 噪声之比（每接收天线平均）。
- **OLLA 偏移**：收敛期之后的平均偏移，负值表示 MAC 需要把 CQI 下调这么多 dB 才能满足目标 BLER。

## 文件

| 文件 | 内容 |
|---|---|
| `config.py` | `ORANConfig`、`CompressionConfig`、`RTConfig` |
| `pdsch.py` | `PDSCHConfig`（复用 `PUSCHConfig` 的 DMRS / TBS 逻辑）、各 UE 配置、资源网格 |
| `odu.py` | O-DU High-PHY |
| `precoding.py` | BD、特征 ZF / RZF，按 PRG 计算与应用 |
| `compression/` | IQ 压缩算法：`base.py`（基类、定点、比特打包）、`bfp.py`（块浮点）、`block_scaling.py`（块缩放）、`mu_law.py`（μ 律）、`none.py` |
| `fronthaul.py` | 前传链路（U-plane / C-plane 编解码与统计；option 8 的时域 IQ 量化 `TimeDomainCodec`）、比特率估算 |
| `oru.py` | O-RU Low-PHY、空口参数（IFFT 点数、保护带、CP） |
| `channel.py` | 多链路信道（CDL 或射线追踪）、CSI 提取、加信道与噪声 |
| `rt_channel.py` | Sionna RT 场景、撒点、链路预算、CIR → 频域信道 |
| `channel_estimation.py` | PRG 感知的 DMRS 信道估计与插值、定时偏差补偿 |
| `ue.py` | UE 接收机 |
| `link.py` | 端到端仿真器 `ORANDownlink`、统计量 `LinkStats` |
| `bler_model.py` | AWGN 链路抽象：(Qm, R, RE 数) 的 TBLER 门限、EESM |
| `csi.py` | CQI 表、CSI 参考资源、LMMSE SINR、`CQIReporter` |
| `csi_rs.py` | CSI-RS / CSI-IM 资源与序列（`CSIRS`），UE 侧 CMR / NZP-IMR / CSI-IM 估计（`CSIRSReceiver`） |
| `link_adaptation.py` | `LinkAdaptation`：CQI → MCS、OLLA |
| `closed_loop.py` | 闭环仿真器 `ClosedLoopDownlink`、逐撒点结果 `DropResult` |
| `../scripts/rt_closed_loop.py` | 闭环仿真脚本（撒点、统计、画图） |
| `../scripts/rt_compare_compression.py` | 闭环中 Cat A/B × 压缩配置的对比（同一批撒点），汇总、CDF、CQI 高估量 |

## 主要设计与简化

1. **PDSCH**：Sionna 2.2 只有 PUSCH。CP-OFDM 下两者的 DMRS 与 TBS 计算一致，`PDSCHConfig` 继承
   `PUSCHConfig`，只把 TB 的信道类型改为 PDSCH（MCS 表、加扰初始化不同）。整个时隙 14 个符号都用于
   PDSCH，没有预留 PDCCH。DMRS 序列初始化中的 λ 项取 0：对照 TS 38.211 V18.6.0 §7.4.1.1.1，
   Type 1 DMRS 且未配置 dmrs-Downlink-r16 时 ⌊λ̄/2⌋ = 0、n̄_SCID = n_SCID，与 Rel-18 的公式相同。
2. **资源网格的"虚拟发射机"**：每个 UE 的层组作为一个虚拟发射机（num_tx = UE 数），预编码后由 O-RU 天线发出。
   UE 侧因此可以直接沿用 Sionna 的多发射机信道估计和 LMMSE 均衡；UE 估计全部 DMRS 端口，
   把其他 UE 的层当作已估计信道的干扰做白化。
3. **预编码**：O-DU 用第一个 DMRS 符号、每个 PRG 中心子载波处的理想信道计算预编码，整个时隙不变。
   ||W||_F = 1，每层功率相同。BD 要求 N_t − (K−1)·N_r ≥ 每 UE 层数，不满足时用 `ezf` / `rzf`。
4. **信道估计**：Sionna 自带的估计器跨整个带宽插值，而等效信道 H·W 在 PRG 边界跳变，导致约 −17 dB 的
   估计误差；且它对 CDM 组内两个端口的分离是相邻导频直接平均，频选信道下会产生与时延扩展成正比的
   泄漏（CDL-C 100 ns 时约 −32 dB 平底）。`DMRSChannelEstimator` 在 PRG 内对每个导频做局部线性模型的
   最小二乘，平底降到约 −40 dB；插值也只在 PRG 内进行。
   此外先补偿定时偏差（`ce_delay_compensation`，默认开启）：主要能量相对 FFT 窗口有时延 τ 时，信道带有
   线性相位，τ ≈ 1 µs 时每个子载波旋转约 0.19 rad，局部线性模型跟不上。估计器用同端口、同符号、同 PRG
   且 OCC 相位相同的导频对（Type 1 间隔 4 个子载波）估计每个 UE 的相位斜率，去旋转后再估计，最后旋转回去。
   CDL-C 100 ns 下影响很小（EVM 变化不到 0.1 个百分点）；射线追踪信道中首径弱、强径晚到
   （平均时延约 1.5 µs）时，不补偿 EVM 约 55%，补偿后与理想 CSI 相当（约 7%~17%）。
5. **UE 不知道压缩噪声**：LMMSE 只使用热噪声方差 N0，前传压缩失真对 UE 是未知的附加失真。
6. **定点化**：前传 IQ 先按 16 bit 定点化（O-RAN 不压缩时的格式），缩放使标称功率的 I/Q 分量 RMS 比满量程
   低 15 dB；之后按 O-RAN WG4 CUS v22.00 附录 A（规范性附录）压缩，三种方法每 PRB 都带 1 字节 udCompParam：
   - BFP（A.1）：4 bit 指数；
   - 块缩放（A.2）：8 bit 缩放因子 sblockScaler（Q1.7），粒度比 2 的幂次指数细，同位宽下 SQNR 比 BFP 高 2~5 dB；
   - μ 律（A.3）：4 bit compBitWidth + 4 bit compShift，先整块左移再做 μ = 8 的三段分段线性压扩。
   三者都与规范伪代码逐比特一致（`tests/test_compression.py` 中用伪代码的逐行翻译做对照）。规范未说明的
   取整溢出一律做饱和；μ 律解压最后的 "/ 2^compShift" 按伪代码做实数除法。
7. **信道**：CDL 只描述一对收发机，且簇离开角固定。为使两个 UE 在空间上可分，每条链路、每个样本随机旋转
   BS 阵列方位。每条链路逐样本归一化为单位平均增益，再乘以 `link_gain_db`（cell-free 大尺度衰落）。
8. **时域仿真**：O-RU 把 PRB 放在 IFFT（默认 1024 点）中间，两侧为保护带，UE 用同样的 FFT 后取出 PRB。
   CIR 每个 OFDM 符号生成一次、符号内不变（忽略符号内多普勒，适合低速），否则 CDL 显存随样点数线性增长。
   频域与时域结果一致（单元测试中验证）。
9. **射线追踪信道**（`rt_channel.py`）：O-RU、UE 天线与 CDL 模式相同（O-RU 天线单元为 TR 38.901 方向图）。
   信道乘以 sqrt(P_RE/N0)，噪声方差取 1（仍满足"每 RE 总发射功率为 1"的约定，前传定点化不受影响）。
   每个 UE 以所有 O-RU 中最早到达的路径为时延零点；UE 静止，一个批次的各样本是同一位置的连续时隙。
   只支持频域仿真，时延超过 CP 的路径不产生符号间干扰。
   撒点：在覆盖图范围内均匀抽点，丢弃距离或路径增益不满足门限、以及头顶有遮挡的点。覆盖图为单阵元、
   无折射，避免把 UE 撒进建筑。覆盖图是蒙特卡洛结果，每次计算略有差异，拒绝采样使同一 seed 的撒点
   基本可复现。射线追踪后路径增益比门限低 10 dB 以上的撒点会重撒（覆盖图与 PathSolver 偶有不一致）。
   PathSolver 用确定性模式（`RTConfig.deterministic`，开销很小），同一位置在不同进程中得到相同的信道。
   覆盖图仍有蒙特卡洛差异，门限附近的个别撒点在不同进程中可能不同，跨进程配对比较时要用
   `rt_compare_compression.py --save-drops` 生成的撒点文件。
10. **CSI-RS 与 CQI（`csi_rs.py`、`csi.py`）**：对应 'cri-RI-CQI'，即 gNB 发预编码 CSI-RS、UE 在等效信道上计算 CQI，
    与 TDD 互易 BD 预编码相配。
    - 资源：每个 CSI-RS 周期的第一个时隙为测量时隙，CSI-RS / CSI-IM 都在符号 12。UE k 的 CMR 从子载波 k·ν 开始，
      端口 i 用第 i 层的权值预编码；CSI-IM 在其后第一个 4 的倍数处。序列和 RE 映射按 TS 38.211 §7.4.1.5.2/.3，
      单元测试中与公式逐项核对。
    - 测量时隙的 PDSCH 在这些 RE 上打孔（规范为速率匹配），保证前传压缩块的内容真实，但不译码。
    - 前传：Cat A 中 CSI-RS 与 PDSCH 一起在 O-DU 预编码；Cat B 中 CSI-RS 放在对应的层流里，由 O-RU 用（压缩后的）
      权值预编码。O-RAN 中可用 reMask 区分 CSI-RS 所在的 section，端口权值可用 SE 11 / 19 下发。
    - UE：先用 PDSCH DMRS 估计的相位斜率补偿定时；每个 PRB 由 CMR 的 2 个 RE 做 LS + CDM2 解扩，得到 N_r × ν
      信道；CSI-IM 上的平均功率作为噪声方差；配对 UE 的端口（NZP-IMR）给出干扰协方差，在 PRG 内平均并扣除估计噪声
      ν·σ²/2。不扣除会把 BD 零陷处的干扰高估到与噪声相当，MU 时 SINR 系统性低估约 3 dB。本 UE 信道估计中的
      噪声不做修正（低 SNR 时 SINR 略偏高）。之后每 PRB、每层算 LMMSE SINR → EESM → CQI。
    - 理想前传下 CSI-RS 测得的 CQI 与精灵 CQI 相差不超过 1（单元测试验证）。
    - 精灵 CQI（`cqi_source="genie"`）：直接用真实等效信道 H_k·W 算每个 RE 的 LMMSE SINR（与 Sionna LMMSE 均衡器的
      1/no_eff 逐 RE 一致），不经前传，作为对照始终记录。
    - CQI 对应的 TBS 按 CSI 参考资源（§5.2.2.5.1）计算：符号 2~13 去掉 DMRS 符号，共 10 个数据符号；
      实际 PDSCH 占满 14 个符号，有 12 个数据符号。UCI 无误码、无时延；层数固定，不上报 RI。
11. **AWGN 门限（`bler_model.py`）**：用 Sionna SYS 的 PDSCH BLER 表（按码块长度），经 TB 分段得到 TBLER。
    表中缺少的最低几档 QPSK（SNR 低于 −5 dB），按同调制下最近的有数据条目用 Shannon 间隔外推。
    EESM 的 β 用 Sionna 的标定值。
12. **链路自适应（`link_adaptation.py`）**：
    - CQI 先换成该 CQI 的门限 SINR（保守下界），加上 OLLA 偏移，再取门限不超过它的最大 MCS。MCS 门限按实际
      分配的 RE 数计算。偏移为 0 时等价于"CQI 查表"。
    - OLLA：每个 ACK 偏移 +δ·p/(1−p)，每个 NACK 偏移 −δ。
    - 一个批次共用一个 MCS，批末统一更新，相当于一个批次的 HARQ 反馈时延；已用最高 MCS 时不再上调。
    - 没有 HARQ 重传。
    - 静止 UE 的 BLER-MCS 关系近似阶跃，OLLA 会在阶跃附近来回调整，平均首传 BLER 接近目标值。
13. **可用 MCS**：Sionna 的 LDPC 编码器不支持重复编码，码率 R < 1/5 或 BG1 下 R < 1/3 的 MCS 无法实现。
    `ORANDownlink.supported_mcs()` 逐个尝试建立编码链来确定可用 MCS。51 PRB、2 层时：
    - MCS 表 2 可用 3~27；
    - MCS 表 1 中 0~2、4、10 不可用。

    SINR 低于 MCS 3 门限（约 0.6 dB）的 UE 只能以高 BLER 运行。真实系统中靠低码率（重复）和降秩解决。
14. **固定层数**：BD 之后，部分 UE 的等效信道接近秩 1，例如只有单一视距径、或与同调度 UE 方向相近。此时第二层
    SINR 很低，宽带 CQI 被拉低。RI 自适应可以改善（见扩展）。
15. **前传压缩失真对 CSI 的可见性**：三种压缩算法都把值为 0 的 RE 还原为严格的 0，所以 CSI-IM 上测不到任何压缩失真；
    失真能否进入 CQI，取决于它落在哪些 CSI-RS RE 上。

    | 失真来源 | 对 PDSCH 的影响 | CSI-RS / CSI-IM 能否测到 |
    |---|---|---|
    | Cat A U-plane | 天线域白噪声，功率 ∝ 该 RE 上所有 UE 信号之和，BD 不能置零：自失真 + 泄漏给配对 UE | 部分：配对 UE 的 CSI-RS 的失真泄漏到本 UE，被 NZP-IMR 测到；自失真只表现为 CMR 信道估计误差 |
    | Cat B U-plane | 层域噪声，随本 UE 权值波束赋形，在配对 UE 处被 BD 置零：只有自失真 | 基本测不到：只表现为 CMR 信道估计误差 |
    | Cat B 权值 | BD 零陷不准，残余多用户干扰 | 能：配对 UE 的 CSI-RS 用压缩后的权值发出，NZP-IMR 直接测到 |

    测不到的部分由 OLLA 根据 ACK/NACK 吸收，`rt_compare_compression.py` 中的 OLLA 偏移和"CQI 高估量"即量化这一部分。
    另外，即使理想前传，BD 只在 PRG 中心子载波置零，PRG 内信道变化带来的残余干扰在高 SNR 时也会被 NZP-IMR 测到
    （精灵 CQI 中同样包含）。

## 扩展

- **新的压缩算法**：在 `compression/` 下新建文件，继承 `IQCompressor`，设置 `method`（注册名）和
  `param_bits`（每块压缩参数的比特数），实现 `compress` / `decompress`，在 `compression/__init__.py`
  中用 `register_compressor` 注册。之后 `CompressionConfig("新名字", iq_width=...)` 即可在 U-plane
  和 Cat B 权值上使用。O-RAN 定义而尚未实现的方法：modulation compression、选择性 RE 发送。
- **cell-free**：`num_oru > 1` 时，所有 O-RU 的天线拼成一个 N_t 维阵列做集中式预编码；Cat A 每个 O-RU
  只收自己天线的端口流，Cat B 每个 O-RU 都收全部层流和自己那部分权值。`link_gain_db` 给出各链路的大尺度增益。
- **DAC / 模拟波束赋形**：`ORU(dac=callable)` 传入作用于时域样点的函数。
- **自建场景**：`RTConfig.scene` 指向 Mitsuba XML 文件（可用 Blender + OSM 导入），再按场景调整
  `oru_positions`、`oru_look_at` 和撒点参数。`num_oru > 1` 时每个 O-RU 都是场景中的一个发射机，
  各链路的大尺度增益直接由场景给出（数据结构已支持，尚未验证）。
- **闭环的升级方向**：
  - 失真感知的 CQI：38.214 §5.2.1.4.1 允许 UE 把 CMR RE 上的"其他干扰"计入 CQI。2 端口、密度 1 的 CMR 每 PRB
    只有 2 个 RE，没有自由度分离失真与信道，需要更密的测量资源或借助 TRS / DMRS 残差。
  - 测量时隙的 PDSCH 按速率匹配生成并译码（目前打孔、不译码）。
  - RI 自适应。
  - HARQ（冗余版本 + 软合并）。
  - UE 移动（射线追踪路径的多普勒）和 CSI 上报时延。
