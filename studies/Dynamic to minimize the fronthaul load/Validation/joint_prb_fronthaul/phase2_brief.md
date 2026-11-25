# 第二阶段实验委托

请在当前目录编写 phase2.py，保存至 results/phase2/。只写你自己的文件。另一会话在写 scheduler.py 和 phase1.py。可以先独立实现候选的压缩评分和吞吐数据，在 scheduler.py 提供接口后集成与运行。用已有 `../../../../.venv/bin/python -B phase2.py --workers 6`。这轮只用 7-1 DU 预编码后的天线 IQ，保持同一 PRB 上所有 UE 同时有数据。

- 场景与第一阶段相同。预先采用 32 个布局 d=0..31 的 4 个相邻 PRB 四径信道、完美 CSI、−70 dBm。每个 RU 每 PRB 10 mW，在同一时隙两类主容量分别等价于两个、三个 BFP9 PRB 净荷（6272 和 9408 bit）。每个 PRB 14 个符号计费。BFP 和块缩放采用同一净荷公式 (14(24q+8)) bit/RU–PRB。比较所有方案的实际净荷，并给出一种固定按 PRB 节段打包估算时加 eCPRI 及报头的口径作为旁证，注明容量实验的主口径。
- 预编码用 `pinv(H_S)` 加逐天线限制的共同归一化。同一布局同一 PRB 的全部候选共用 64-QAM 符号，随机数种子由 20261125、d 和 b 派生。用 `oran.fronthaul.IQCodec` 的 BFP 与 Block Scaling，(q) 可为 5、7、9，一块 12 RE。对解压的信号执行相同的功率上限检查，可用统一缩放保护所有方案并记录调整。根据真信道接收向量，估计每 UE 的相关增益，并明确把剩余的多用户干扰及失真加入分母，再加热噪声。逐 PRB sum SE 作为线性目标的候选贡献。附带原文按 UE 聚合信号功率、干扰失真和噪声之后再 log 的 sum SE；后者用同一选择复算。给出相同 RF 展开与原始数据。
- 共享候选掩码的范围可以为每个天线数的若干高得分掩码、以及具有不同的活跃 RU 或避免热点能力的掩码。请保留确定的可行备用组合（4 个互不相交的 size-4 掩码，每块候选都包含它们，均支持 BFP9），使所有主场景都在同一个可行的候选集合上比较。支持不同 (q) 和方法的候选加同样的射频与链路规则。固定档两组都用 BFP9，自适应两组均可选 BFP 或块缩放、5/7/9 bit。同一个 PRB 在其活跃 RU 上使用一种码法和位宽。比较四组逐块贪心/联合 × 固定/自适应。用相同绝对前传容量而非压缩后的 PRB 数配额。报告联合的候选最优性间隙和贪心排序策略。
- scheduler.py 的接口为 `joint_select(values, costs, budget, **options)` 和 `greedy_select(values, costs, budget, **options)`。values 为各 PRB 的一维浮点数组列表，costs 为每块相同候选顺序的 `[候选数, RU 数]` 整数非负用量数组列表，budget 为长度 M 的整数向量。返回 dict 有 `indices`、`value`、`status`、`gap`（可 None）和 `runtime_s`。支持可行保底方案 `fallback`。请在其出现后进行集成。
- results/phase2/paired.npz 至少记录 `drop,k,algorithm,mask,width,codec,link_bits,power_mw`（实际逐 PRB 发射功率，最好也写 `antenna_power_mw`）、`se_per_prb`、`twc_se`、`mip_gap`。原始记录要能让我在单独的 Sionna PDSCH TB 试验重建实际波束、编码符号与压缩包，测同一 MCS 同一噪声下的逐用户 CRC 和有效吞吐。请写 summary.json，记录实验设定、种子、功率缩放、固定档基线、显式包头假设、聚合指标、区间和耗时。
- 至少验证链路容量与所有方案的功率上限，重新构造若干个输出计算 SE、报告候选范围和最优性声明，其他文件由研究者使用。请实际运行并报告数字。如果 Bash 权限卡住，请立刻提供已审阅可执行的精确命令，我们将在同一项目手动运行。
