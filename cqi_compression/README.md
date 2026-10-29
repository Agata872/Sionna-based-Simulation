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

## 进度与下一步（2026-10-29）

**已完成**
- 代码：上面四个脚本和 Colab 笔记本。
- 正确性检查：aware 在失真为零时与 `lmmse_sinr` 一致；理想前传下 aware、oracle 与 genie 的 CQI 相同；位宽降低时 aware 的有效 SINR 单调下降，CSI-RS 实测值介于 genie 与 aware 之间。
- 耗时：CPU（本机 32 线程）每个撒点、每个 Cat 约 2 分钟，完整实验（50 个撒点 × Cat A/B）约 3.5 小时；GPU 按本机冒烟测试的速度外推约 1 小时（含复核）。

**还没有可用的结果。** 本机已有的运行都不能作为结论（见 [experiments.md](experiments.md)）：
- GPU 上的冒烟测试显示，5 bit 和 4 bit 时用 genie CQI 选 MCS 会让部分 UE 整批出错，而 oracle CQI 能避开；aware 介于两者之间（Cat A 撒点 0 的一个 UE 在 5 bit 时 aware 仍全错，差距来自 DMRS 失真）。这些现象需要在可靠的设备上复现。
- 正式运行在本机 GPU 上因为显存读错而中断（见下）。

**下一步**
1. 在 Colab 上按 `colab_continue.ipynb` 运行：自检 → 冒烟测试 → 正式运行 → 汇总。
2. 结果出来后，把主要数字和结论补到本文件，并在 `experiments.md` 登记。
3. 视结果决定后续：若 aware 与 oracle 差别大，"保 MCS 的位宽规则"需要把 DMRS 失真对信道估计的影响也建模进去；若 csi-rs 已经足够接近 oracle，说明 UE 测量能吸收大部分压缩影响。

## 在另一台电脑 / Colab 上继续

1. 在另一台电脑的 VS Code 中拉取仓库（`main` 分支），打开 `cqi_compression/colab_continue.ipynb`。
2. 用 Colab 扩展把内核连接到 Colab GPU 运行时。仓库是私有的：先在 GitHub 生成只读令牌，放进 Colab Secrets 的 `GITHUB_TOKEN`（笔记本读不到时会提示输入）。
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
