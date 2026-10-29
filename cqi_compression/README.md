# CQI 查表时是否考虑前传压缩：对首传 BLER 的影响

起始日期：2026-10-29。背景与研究方向见 [research/girycki_journal_direction_2026-10-29.md](../research/girycki_journal_direction_2026-10-29.md)。实验索引见 [experiments.md](experiments.md)。

**要回答的问题。** 闭环中 UE 上报 CQI，MAC 查表选 MCS（关闭 OLLA），数据经过压缩前传发送。如果 CQI 没有反映前传压缩，BLER 会升高多少？随压缩位宽怎样变化？UE 自己测量的 CQI（CSI-RS 也经过压缩前传）能反映多少失真？

**为什么做。** 期刊扩展打算在 DU–RU 之间逐时隙自适应调整压缩位宽、不经过 MAC。MAC 按 CQI 选好 MCS 之后，前传再降位宽，实际 SINR 可能低于该 MCS 的门限。这个实验量化这一风险有多大，以及"保住已调度 MCS"的位宽规则需要多准的失真模型（aware 与 oracle 的差别）。

## 设置

沿用 `oran/` 闭环的默认配置：Sionna RT munich，1 个 O-RU（8 通道），2 个 UE × 4 天线，每 UE 2 层，BD 预编码，51 PRB，30 kHz，MCS 表与 CQI 表都用表 2（256QAM），目标 BLER 0.1。

- **链路自适应**：`LinkAdaptationConfig(olla=False)`，即 CQI → 门限 SINR → MCS 的纯查表，不做外环修正。
- **压缩算法**：块缩放（O-RAN CUS 附录 A.2，udCompMeth 2）。三种算法同位宽时比特率相同，块缩放的 SQNR 最高（`oran/README` 设计第 6 条）。冒烟测试中在实际 U-plane 信号上复核，5 bit 时：Cat A 块缩放 27.5 dB、BFP 25.2 dB、μ 律 25.0 dB；Cat B 28.5、27.4、23.8 dB。正式运行写入 `codec_sqnr.json`。
- **位宽**：理想前传、9、8、7、6、5、4、3 bit。Cat B 的权值用同一方法、同一位宽。
- **Cat A / Cat B 都做**，用同一批撒点。

## 四种 CQI

| 名称 | 含义 | 实现 |
|---|---|---|
| genie | 不考虑压缩：真实等效信道、理想权值，不经前传 | `ClosedLoopDownlink.measure_cqi_genie` |
| csi-rs | UE 实测：预编码 CSI-RS / CSI-IM 与 PDSCH 一起经压缩前传 | `ClosedLoopDownlink.measure_cqi_csirs` |
| aware | 计入数据 RE 上的压缩失真（解析） | `aware_cqi.measure_cqi_aware` |
| oracle | 计入全部压缩影响（PDSCH 实测） | `aware_cqi.measure_cqi_oracle` |

- **aware**：在精灵 CQI 的 LMMSE SINR 中加入前传实际造成的失真。
  - Cat B 用解压后的权值算等效信道，残余多用户干扰随之进入 SINR。
  - U-plane 失真 e = O-RU 天线口信号 − 用实际权值预编码的理想信号，经信道到达 UE，按 PRB 估计协方差 C_d。
  - UE 的合并向量只按热噪声设计（与实际接收机一致），SINR = a² / (a − a² + f^H C_d f)。失真为零时与 `oran.csi.lmmse_sinr` 相同（已核对，相对误差 ≤ 3e-5）；理想前传时 aware 与 genie 的 CQI、有效 SINR 完全相同。
- **oracle**：发一个经压缩前传的 PDSCH 时隙，UE 照常做 LS 信道估计和均衡，按 PRB、按层统计均衡后 SINR，再经 EESM 得到 CQI。它比 aware 多计入了 DMRS 失真带来的信道估计误差（DMRS 与数据在同一个压缩块里）。理想前传时 oracle 与 genie 的 CQI 相同，有效 SINR 低约 0.4–0.7 dB（LS 估计本身的损失）。

数据时隙的 PDSCH 对四种 CQI 都经过同一个压缩前传。UE 静止，每个撒点的信道只算一次。同一撒点、同一周期内，各位宽、各 CQI 来源的比特和噪声相同（配对比较），选出相同 MCS 的来源共用同一次 PDSCH 仿真。

**运行中复核。** 每个（撒点, 位宽）在第一个周期结束后，用同一种子把第一个 PDSCH 批次重跑一次，误块数和均衡误差必须逐位相同。结果写入每行的 `recheck_ok` 和 `complete.json` 的 `recheck_mismatches`；不为 0 时这次运行作废，`analyse_cqi_bler.py` 会在报告开头警告。

**汇总口径。** 理想前传下用精灵 CQI 也失败（BLER ≥ 0.5）的 UE-撒点不参与汇总：它们的 SINR 低于最低可用 MCS（`oran/README` 设计第 13 条），在任何位宽、任何 CQI 下都全错，与压缩无关。报告开头给出排除的个数。

## 文件

| 文件 | 内容 |
|---|---|
| `aware_cqi.py` | 考虑压缩失真的 CQI：解析式 `distortion_aware_sinr` / `measure_cqi_aware`，PDSCH 实测 `measure_cqi_oracle` |
| `run_cqi_bler.py` | 仿真：Cat × 撒点 × 位宽 × 周期，四种 CQI 各自查表选 MCS 后发送 PDSCH，写入运行目录 |
| `analyse_cqi_bler.py` | 汇总：BLER、失败 UE 比例、吞吐、MCS 表格（`report.txt`）和图（`fig_bler_tput_vs_width.png`） |
| `gpu_selfcheck.py` | GPU 自检：纯 PyTorch 类 BP 循环与 Sionna LDPC 重复译码，结果须逐位相同 |
| `colab_continue.ipynb` | 在 Colab GPU 上继续：克隆仓库、安装、自检、冒烟测试、正式运行、汇总 |

运行目录沿用 `dynamic_split/runio.py` 的约定（metadata.json、源码快照、rows.jsonl、complete.json）。

```bash
source env.sh
python cqi_compression/run_cqi_bler.py --drops 2 --periods 1 --widths ideal 5 --out results/cqi_compression_20261029/smoke
python cqi_compression/run_cqi_bler.py --out results/cqi_compression_20261029/main                   # 默认 CPU
python cqi_compression/run_cqi_bler.py --device cuda:0 --out results/cqi_compression_20261029/main   # GPU（先自检）
python cqi_compression/analyse_cqi_bler.py results/cqi_compression_20261029/main
```

设备选项：

- 默认只用 CPU，脚本在导入 torch / Dr.Jit 之前隐藏 GPU，射线追踪随之使用 Mitsuba 的 LLVM 后端。
- `--device cuda:0` 时链路仿真和射线追踪都用 GPU。
- GPU 的 OptiX 不可用时，再加 `--rt-llvm`：射线追踪用 CPU，链路仍在 GPU。
- 不同设备上同一种子撒出的位置不同（随机数序列不同），CPU 与 GPU 的结果不能逐撒点对比，只能比统计量。

## 结果（2026-10-29，Colab A100，运行 Q5）

**运行与可信度。** NVIDIA A100-SXM4-80GB，torch 2.11.0+cu130，射线追踪 `cuda_ad_mono_polarized`。GPU 自检两项 PASS；100 个（Cat, 撒点）的运行中复核全部一致。耗时约 14.5 分钟（每个撒点约 8 s）。原始数据在 Google Drive 的 `MyDrive/sionna_results/cqi_compression_20261029/main/`；本机 `results/cqi_compression_20261029/colab_main/` 中有从笔记本输出提取的 `report.txt`、`main.log` 和图，没有 `rows.jsonl`。

**汇总口径。** 200 个 UE-撒点中有 26 个在理想前传下用精灵 CQI 也失败，已排除。每格为"首传 BLER / 平均 UE 吞吐（Mbps）"；9 bit 与理想前传相同，3 bit 时链路基本失效，均从表中略去，完整表见 `report.txt`。前传比特率为 51 PRB 下单个 O-RU 的 U-plane + C-plane。

| Cat | 位宽 | 前传 Gbps | SQNR dB | genie（不考虑） | csi-rs（UE 实测） | aware（数据失真） | oracle（全部失真） |
|---|---|---|---|---|---|---|---|
| A | 理想 | 4.61 | – | 0.006 / 184 | 0.023 / 182 | 0.006 / 184 | 0.006 / 182 |
| A | 8 bit | 2.40 | 46.1 | 0.031 / 179 | 0.035 / 179 | 0.018 / 181 | 0.005 / 181 |
| A | 7 bit | 2.13 | 40.0 | 0.085 / 169 | 0.099 / 168 | 0.050 / 174 | 0.024 / 176 |
| A | 6 bit | 1.85 | 33.7 | 0.198 / 147 | 0.206 / 147 | 0.122 / 158 | 0.099 / 160 |
| A | 5 bit | 1.58 | 27.5 | 0.498 / 90 | 0.485 / 92 | 0.401 / 102 | 0.317 / 114 |
| A | 4 bit | 1.30 | 21.3 | 0.936 / 5 | 0.914 / 8 | 0.779 / 25 | 0.634 / 42 |
| B | 理想 | 2.36 | – | 0.005 / 184 | 0.025 / 182 | 0.005 / 184 | 0.005 / 182 |
| B | 7 bit | 1.09 | 42.7 | 0.019 / 181 | 0.028 / 181 | 0.008 / 183 | 0.006 / 182 |
| B | 6 bit | 0.95 | 37.0 | 0.044 / 177 | 0.032 / 179 | 0.017 / 182 | 0.006 / 181 |
| B | 5 bit | 0.81 | 29.0 | 0.262 / 137 | 0.163 / 152 | 0.170 / 150 | 0.066 / 163 |
| B | 4 bit | 0.67 | 21.7 | 0.948 / 4 | 0.842 / 15 | 0.779 / 22 | 0.189 / 100 |

**结论**

1. **CQI 不考虑压缩时，BLER 在某个位宽以下陡增。** Cat A：8 bit 时 BLER 从 0.006 升到 0.031，7 bit 为 0.085，6 bit 为 0.198，5 bit 为 0.50；吞吐分别下降 8%（7 bit）、20%（6 bit）、51%（5 bit）。Cat B 更耐压缩：7 bit 0.019、6 bit 0.044，仍在目标 0.1 以内；5 bit 升到 0.26，4 bit 为 0.95。不通知 MAC 就能直接降位宽的下限：Cat A 约 8 bit（SQNR ≈ 46 dB），Cat B 约 6–7 bit（SQNR ≈ 37–43 dB）。
2. **BLER 的上升主要来自一部分 UE 整批出错，而不是所有 UE 都略微变差。** 大多数格的 BLER 与"BLER ≥ 0.5 的 UE 比例"几乎相等（见 `report.txt`）。静止信道下 BLER 随 MCS 近似阶跃：MCS 选高了的 UE 几乎全错，其余几乎不错。例外是 Cat B 4 bit 的 oracle（BLER 0.189，失败 UE 比例只有 0.05）：许多 UE 只有部分 TB 出错，日志中这些撒点的 BLER 多为 0.25 左右，说明此时 MCS 恰好落在门限附近。
3. **UE 自己测的 CQI 基本看不到压缩。** Cat A 各位宽下 csi-rs 与 genie 几乎相同（5 bit：0.485 对 0.498）。Cat B 稍有帮助（5 bit：0.163 对 0.262），因为权值压缩造成的残余多用户干扰能被 NZP-IMR 测到。理想前传时 csi-rs 的 BLER 略高（0.023 对 0.006），这是 CSI-RS 测量本身的误差。这与 `oran/README` 设计第 15 条的分析一致。因此不能指望 UE 的 CQI 自动反映前传压缩，需要 DU / MAC 修正或 OLLA 吸收。
4. **考虑压缩能让位宽再降 1–2 bit。** 在 BLER ≤ 0.1 的前提下，用 oracle CQI 时 Cat A 可以从 8 bit 降到 6–7 bit（2.40 → 1.85–2.13 Gbps），Cat B 可以从 6 bit 降到 5 bit（0.95 → 0.81 Gbps）。同一位宽下吞吐也更高：Cat A 5 bit 从 90 提高到 114 Mbps，Cat B 5 bit 从 137 提高到 163 Mbps，Cat B 4 bit 从 4 提高到 100 Mbps。
5. **解析的 aware 只补回一部分。** Cat A 5 bit：aware 0.401，oracle 0.317；Cat B 5 bit：0.170 对 0.066；Cat B 4 bit：0.779 对 0.189。差额来自 DMRS 失真造成的信道估计误差（DMRS 与数据在同一个压缩块里），Cat B 低位宽时最大。"保 MCS 的位宽规则"如果用解析失真模型，必须把信道估计误差也算进去，或者在 MCS 门限上留余量。
6. **即使用 oracle CQI，低位宽下 BLER 仍超过目标**（Cat A 5 bit 0.317、4 bit 0.634；Cat B 4 bit 0.189）。原因尚未查明。最可能的解释是 UE 解调只按热噪声计算 LLR（`oran/README` 设计第 5 条）：压缩失真占主导时 LLR 过于自信，LDPC 译码比"按 SINR 查 AWGN 门限"预测的差。也可能是 EESM 的 β 只按 AWGN 标定。需要单独验证。
7. **相同前传比特率下 Cat B 明显更好。** Cat B 7 bit（1.09 Gbps）不考虑压缩时 BLER 0.019；Cat A 要达到同样水平需要 8–9 bit（2.40–2.67 Gbps）。原因有二：本配置 4 层、8 根天线，Cat B 传层、Cat A 传天线；Cat B 的 U-plane 失真随信号一起波束赋形，在配对 UE 处被 BD 置零，Cat A 的失真则在天线域、会泄漏给配对 UE。
8. **压缩算法。** Cat A 中块缩放在各位宽下都比 BFP 和 μ 律高 2–3 dB。Cat B 的层信号是 QAM 星座点，量化误差与星座格点有关，单个撒点上的 SQNR 不随位宽单调变化（如 7 bit 时 BFP 43.7 dB 高于块缩放 41.7 dB），排序不稳定。

**局限。** 单小区、1 个 O-RU、静止 UE、OLLA 关闭、50 个撒点；统计单位实际是 UE-撒点（每个 Cat 约 87 个），BLER 的分辨率约为 1/87。oracle 需要知道发送的数据，实际系统做不到，只能作为上界。Cat B 的权值与 U-plane 用同一位宽，两者的贡献没有分开。

**下一步**
1. 查明结论 6：在同一批撒点上，让 UE 的噪声方差计入压缩失真，看 oracle 的 BLER 能否回到 0.1 附近。
2. Cat B 分开扫描 U-plane 位宽与权值位宽，区分两者的贡献。
3. 打开 OLLA 对比：OLLA 能否吸收静态的压缩偏差、需要多少时隙收敛。如果 DU 逐时隙改位宽，OLLA 跟不上，本实验关闭 OLLA 的结果就是这种情形的近似。
4. 把 Drive 上的 `rows.jsonl` 下载到本机 `results/cqi_compression_20261029/colab_main/`，以便做逐 UE 的细分析（如 CQI 高估量与 SINR 的关系）。

## 在另一台电脑 / Colab 上继续

1. 在另一台电脑的 VS Code 中拉取仓库（`main` 分支），打开 `cqi_compression/colab_continue.ipynb`。
2. 用 Colab 扩展把内核连接到 Colab GPU 运行时。仓库目前是公开的，第 3 步直接克隆，不需要令牌；若改回私有，把笔记本第 1 步的 `PRIVATE_REPO` 设为 True，并准备只读令牌（Repository access 选中本仓库、Contents: Read-only），放进 Colab Secrets 的 `GITHUB_TOKEN`。
   远程隧道下登录 Colab 时，浏览器会跳到 `127.0.0.1:端口` 并报错：复制完整地址，在 VS Code 终端（远程机器）运行 `curl -s "完整地址"` 即可完成登录。
3. 从上到下运行。结果写到 Google Drive 的 `MyDrive/sionna_results/cqi_compression_20261029/`；Drive 挂载失败时写到运行时本地并打包，断开前要下载。
4. GPU 自检两项都 PASS 才用 GPU；正式运行后检查报告开头没有"复核不一致"警告。
5. 运行目录已存在 `rows.jsonl` 时脚本拒绝覆盖。运行中断后，换一个 `--out` 名字重跑。

## 只用 CPU 的原因（2026-10-29）

在 WSL2 笔记本（RTX 4060 Laptop，Windows 驱动 617.42，PyTorch 2.14.1+cu126）上，GPU 在持续负载下会间歇性读错显存：

- Sionna 的 LDPC 译码器在相同输入、相同种子下输出不确定（每隔几次就有约 5000 个比特不同），并且会触发下标越界断言或 `illegal instruction`。下标缓冲区本身检查过是合法的，也没有被改写；
- 打开 `torch.use_deterministic_algorithms(True)` 也不能消除；
- 不经过 Sionna、只用 PyTorch 写的类 BP 循环（gather → tanh → 按下标写回，每次 20 轮）同样会触发越界断言，而同一脚本在 CPU 上 236 次结果完全一致；
- 单独的 gather、`index_put_`、矩阵乘法在 GPU 上短时间重复 300–500 次都正确；
- 间歇性：几十分钟后 `gpu_selfcheck.py` 在同一块 GPU 上又两项都通过了。所以自检通过只是必要条件，运行中的复核才是每次结果可信的依据。

这台笔记本此前在 GPU 上得到的链路级结果（包括重新运行的 `notebooks/closed_loop_walkthrough.ipynb`）也可能受影响，需要在 CPU 或可靠的 GPU 上复核。
