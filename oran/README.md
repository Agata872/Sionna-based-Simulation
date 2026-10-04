# O-RAN 7.2x 下行链路级仿真（Category A / B）

基于 Sionna PHY（PyTorch 版 2.2.0）实现 O-RAN 7.2x 功能划分下的 PDSCH 下行链路，比较
Category A（预编码在 O-DU）与 Category B（预编码在 O-RU）在前传 IQ 压缩下的性能与前传带宽。

## 仿真范围

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

## 快速开始

```bash
cd "/home/user/Sionna-based Simulation"
source env.sh
python -m unittest discover -s tests -v          # 单元测试，约 20 秒
python scripts/compare_cat_a_b.py --widths ideal 9 5 --snr 16 30 2 --max-batches 5   # 快速试跑
python scripts/compare_cat_a_b.py --widths ideal 7 bs:7 mu:7 --bfw-width bs:9        # 比较压缩算法

# 完整扫描（Cat A/B × 压缩位宽 × SNR）：两块 GPU 各跑一类，再合并画图
# Sionna 的随机种子按设备序号偏移，用 CUDA_VISIBLE_DEVICES 让两个进程都在 "cuda:0" 上运行，
# 两边的比特、信道、噪声才完全相同（配对比较）
CUDA_VISIBLE_DEVICES=0 python scripts/compare_cat_a_b.py --categories A --out results/cat_a &
CUDA_VISIBLE_DEVICES=1 python scripts/compare_cat_a_b.py --categories B --out results/cat_b &
wait
python scripts/compare_cat_a_b.py --merge results/cat_a results/cat_b --out results/cat_ab
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

## 指标

- **SNR**：每个 RE 的总发射功率为 1（所有 O-RU 所有天线之和），每对收发天线的平均信道增益为 1，
  SNR = 1/N0，N0 为每根接收天线每个 RE 的噪声方差。
- **BLER**：按 TB 的 CRC 统计，每个 UE 分别统计；吞吐量 = (1 − BLER) × TBS × 每秒时隙数。
- **接收 EVM**：UE 均衡后的数据符号相对发送符号的误差（含噪声、干扰、信道估计误差和压缩失真）。
- **发射 EVM**：O-RU 天线口频域信号相对"理想前传"（不量化、不压缩）的误差，只反映前传失真。
- **U-plane SQNR**：前传 IQ 经定点化 + 压缩后的信号与量化噪声功率比。
- **前传比特率**：单个 O-RU 链路的 U-plane + C-plane 比特率，含 eCPRI 和以太网开销
  （见 `fronthaul.py` 中头部长度的说明）。100 MHz、每层不压缩时为 3276 × 14 × 2000 × 32 = 2.93 Gbps，
  与书中数值一致。

## 文件

| 文件 | 内容 |
|---|---|
| `config.py` | `ORANConfig`、`CompressionConfig` |
| `pdsch.py` | `PDSCHConfig`（复用 `PUSCHConfig` 的 DMRS / TBS 逻辑）、各 UE 配置、资源网格 |
| `odu.py` | O-DU High-PHY |
| `precoding.py` | BD、特征 ZF / RZF，按 PRG 计算与应用 |
| `compression/` | IQ 压缩算法：`base.py`（基类、定点、比特打包）、`bfp.py`（块浮点）、`block_scaling.py`（块缩放）、`mu_law.py`（μ 律）、`none.py` |
| `fronthaul.py` | 前传链路（U-plane / C-plane 编解码与统计）、比特率估算 |
| `oru.py` | O-RU Low-PHY、空口参数（IFFT 点数、保护带、CP） |
| `channel.py` | 多链路 CDL 信道、CSI 提取、加信道与噪声 |
| `channel_estimation.py` | PRG 感知的 DMRS 信道估计与插值 |
| `ue.py` | UE 接收机 |
| `link.py` | 端到端仿真器 `ORANDownlink`、统计量 `LinkStats` |

## 主要设计与简化

1. **PDSCH**：Sionna 2.2 只有 PUSCH。CP-OFDM 下两者的 DMRS 与 TBS 计算一致，`PDSCHConfig` 继承
   `PUSCHConfig`，只把 TB 的信道类型改为 PDSCH（MCS 表、加扰初始化不同）。整个时隙 14 个符号都用于
   PDSCH，没有预留 PDCCH；DMRS 序列初始化采用 Rel-15 形式（λ 项为 0）。
2. **资源网格的"虚拟发射机"**：每个 UE 的层组作为一个虚拟发射机（num_tx = UE 数），预编码后由 O-RU 天线发出。
   UE 侧因此可以直接沿用 Sionna 的多发射机信道估计和 LMMSE 均衡；UE 估计全部 DMRS 端口，
   把其他 UE 的层当作已估计信道的干扰做白化。
3. **预编码**：O-DU 用第一个 DMRS 符号、每个 PRG 中心子载波处的理想信道计算预编码，整个时隙不变。
   ||W||_F = 1，每层功率相同。BD 要求 N_t − (K−1)·N_r ≥ 每 UE 层数，不满足时用 `ezf` / `rzf`。
4. **信道估计**：Sionna 自带的估计器跨整个带宽插值，而等效信道 H·W 在 PRG 边界跳变，导致约 −17 dB 的
   估计误差；且它对 CDM 组内两个端口的分离是相邻导频直接平均，频选信道下会产生与时延扩展成正比的
   泄漏（CDL-C 100 ns 时约 −32 dB 平底）。`DMRSChannelEstimator` 在 PRG 内对每个导频做局部线性模型的
   最小二乘，平底降到约 −40 dB；插值也只在 PRG 内进行。
5. **UE 不知道压缩噪声**：LMMSE 只使用热噪声方差 N0，前传压缩失真对 UE 是未知的附加失真。
6. **定点化**：前传 IQ 先按 16 bit 定点化（O-RAN 不压缩时的格式），缩放使标称功率的 I/Q 分量 RMS 比满量程
   低 15 dB；之后按 O-RAN WG4 CUS v08.00 附录 A 压缩，三种方法每 PRB 都带 1 字节 udCompParam：
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

## 扩展

- **新的压缩算法**：在 `compression/` 下新建文件，继承 `IQCompressor`，设置 `method`（注册名）和
  `param_bits`（每块压缩参数的比特数），实现 `compress` / `decompress`，在 `compression/__init__.py`
  中用 `register_compressor` 注册。之后 `CompressionConfig("新名字", iq_width=...)` 即可在 U-plane
  和 Cat B 权值上使用。O-RAN 定义而尚未实现的方法：modulation compression、选择性 RE 发送。
- **cell-free**：`num_oru > 1` 时，所有 O-RU 的天线拼成一个 N_t 维阵列做集中式预编码；Cat A 每个 O-RU
  只收自己天线的端口流，Cat B 每个 O-RU 都收全部层流和自己那部分权值。`link_gain_db` 给出各链路的大尺度增益。
- **Sionna RT 信道**：写一个与 `channel.MultiLinkCDL` 调用接口相同（返回 a、tau）的类替换即可。
- **DAC / 模拟波束赋形**：`ORU(dac=callable)` 传入作用于时域样点的函数。
