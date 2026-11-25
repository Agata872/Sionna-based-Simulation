# 跨 PRB 逐用户 ZF 验证委托

请独立实现和运行第一轮复现。用同目录下的 `claude_replay.py` 和结果文件 `claude_replay.json` 交付，文稿与原 E8 结果由研究者保留。

- 读取 `../results/same_total_20261124/samples.npz` 的 `val`、`key`。维度是 `[drop, K_index, PRB, power_rule, granularity, m]`。取 `K_index=0, power_rule=0, granularity=0, m=4..16`。用 `../same_total.py` 中的 `drop(20261124, d, 1, 4, 4)[0]` 重建真信道 `H[b,u,a]`。命令从项目根用 `.venv/bin/python`，OpenBLAS 单线程。500 个布局各有 4 个 PRB，4 个 UE 始终同时有数据。
- 原基线对每一 PRB 分别取参考增益 `gmax=max(val[m])`。噪声用 mW，`noise=10**(dBm/10)`，噪声为 −90、−70、−50 dBm。`r[b]=0.98*log2(1+gmax/noise)` 是每个 UE 的基线目标。取能达到 `noise*(2**r[b]-1)` 的最小 `m`。每个 m 使用 `key` 中的原赢家。原波束降到恰好达标后，其实际总功率作为该 PRB 的上限，每根天线另受 10 mW 上限。
- 两两匹配四个 PRB。对每一对块 i,j 和两名用户 u,v，按 `[+d,-d]` 和 `[-d,+d]` 调两块上的用户速率。`d` 可取零，或取 `±min(r[i],r[j])*{1/32,1/16,1/8,1/4}`。各 PRB 各 m 均用归档的最优掩码，重算 `W=pinv(H[:,S])@diag(sqrt(noise*(2**rate-1)))`。在所有满足逐天线和逐 PRB 实发功率约束的选择中，最小化四块上的天线流总数。必须能退回基线。正常的容量上限检查允许相对浮点容差 1e-9。
- 对至少若干胜出样本直接计算真实复数 `H@W` 和 SINR，检查两个方向的速率守恒、逐 PRB 总功率和逐天线上限。按布局配对计算均值、负载节省和 95% 区间。输出可复查的数据和运行命令，记录基线与候选比较用的同一门槛。
- 初次运行仅使用 7-1 的固定宽度天线 IQ，统计 RU–PRB 数。更进一步的公开核查将用多径频域相关信道及估计 CSI，在真信道独立测 SINR，并对同一个 MCS 做传输块 CRC 检查。
