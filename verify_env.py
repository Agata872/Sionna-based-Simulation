#!/usr/bin/env python3
"""Sionna 仿真环境验证脚本。

用法（先激活环境）：
    source env.sh
    python verify_env.py

依次检查：
  1. 软件包版本
  2. PyTorch 能否在每块 GPU 上计算（cuBLAS / cuFFT）
  3. 链路级 Sionna PHY：5G LDPC + 16-QAM + AWGN，每块 GPU 各跑一次
  4. torch.compile 加速的蒙特卡洛 BER 仿真（sim_ber）
  5. 系统级 Sionna SYS：PHY 抽象 + 内环链路自适应

注：本环境安装的是 sionna-no-rt，不含射线追踪模块 Sionna RT。
"""

import sys
import time
import traceback

RESULTS = []


def run_check(name, fn, *args):
    """运行一项检查，记录是否通过及耗时。"""
    print(f"\n=== {name} ===", flush=True)
    t0 = time.perf_counter()
    try:
        fn(*args)
        ok = True
    except Exception:
        traceback.print_exc()
        ok = False
    dt = time.perf_counter() - t0
    RESULTS.append((name, ok, dt))
    print(f"--> {'通过' if ok else '失败'}（{dt:.1f} s）", flush=True)


def check_versions():
    from importlib.metadata import version
    import torch

    print(f"{'python':<14}{sys.version.split()[0]}")
    for pkg in ("torch", "sionna-no-rt", "numpy", "scipy", "matplotlib", "h5py"):
        print(f"{pkg:<14}{version(pkg)}")
    print(f"torch 编译时的 CUDA 版本：{torch.version.cuda}")
    # 本机驱动为 535（最高支持 CUDA 12.2），CUDA 13 构建的 torch 无法使用 GPU
    assert torch.version.cuda and torch.version.cuda.startswith("12."), \
        "torch 不是 CUDA 12.x 构建，请按 README 从 cu126 源重新安装"


def check_torch_gpu():
    import torch

    assert torch.cuda.is_available(), "torch.cuda.is_available() 为 False"
    for i in range(torch.cuda.device_count()):
        dev = f"cuda:{i}"
        a = torch.randn(1024, 1024, device=dev)
        b = torch.randn(1024, 1024, device=dev)
        x = torch.randn(4096, dtype=torch.complex64, device=dev)
        mm_ok = torch.allclose((a @ b).cpu(), a.cpu() @ b.cpu(), rtol=1e-3, atol=1e-3)
        fft_ok = torch.allclose(torch.fft.fft(x).cpu(), torch.fft.fft(x.cpu()),
                                rtol=1e-3, atol=1e-3)
        mem = torch.cuda.get_device_properties(i).total_memory / 2**30
        print(f"{dev}: {torch.cuda.get_device_name(i)}，显存 {mem:.1f} GiB，"
              f"matmul {'正常' if mm_ok else '异常'}，FFT {'正常' if fft_ok else '异常'}")
        assert mm_ok and fft_ok


def build_ldpc_qam_blocks(k, n, m):
    """构建 5G LDPC + QAM + AWGN 链路所需的模块。"""
    from sionna.phy.mapping import BinarySource, Constellation, Mapper, Demapper
    from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
    from sionna.phy.channel import AWGN

    constellation = Constellation("qam", m)
    encoder = LDPC5GEncoder(k, n)
    return dict(
        source=BinarySource(),
        encoder=encoder,
        mapper=Mapper(constellation=constellation),
        channel=AWGN(),
        demapper=Demapper("app", constellation=constellation),
        decoder=LDPC5GDecoder(encoder, hard_out=True, num_iter=20),
    )


def check_phy(device):
    from sionna.phy import config
    from sionna.phy.utils import ebnodb2no, compute_ber

    config.device = device
    config.seed = 42
    k, n, m = 512, 1024, 4  # 码率 1/2，16-QAM
    blk = build_ldpc_qam_blocks(k, n, m)

    for ebno_db in (2.0, 4.0, 6.0):
        no = ebnodb2no(ebno_db, m, k / n)
        b = blk["source"]([2000, k])
        c = blk["encoder"](b)
        y = blk["channel"](blk["mapper"](c), no)
        llr = blk["demapper"](y, no)
        b_hat = blk["decoder"](llr)
        ber_uncoded = compute_ber(c, (llr > 0).float()).item()
        ber_coded = compute_ber(b, b_hat).item()
        print(f"Eb/N0 = {ebno_db:.0f} dB：译码前 BER = {ber_uncoded:.2e}，"
              f"译码后 BER = {ber_coded:.2e}")

    print(f"运行设备：{b_hat.device}")
    assert str(b_hat.device) == device, f"计算没有在 {device} 上进行"
    assert ber_coded < ber_uncoded, "LDPC 译码后 BER 没有下降"


def check_compile():
    import numpy as np
    from sionna.phy import Block, config
    from sionna.phy.utils import ebnodb2no, sim_ber

    class LinkModel(Block):
        def __init__(self, k=512, n=1024, m=4):
            super().__init__()
            self.k, self.n, self.m = k, n, m
            self.blk = build_ldpc_qam_blocks(k, n, m)

        def call(self, batch_size, ebno_db):
            blk = self.blk
            no = ebnodb2no(ebno_db, self.m, self.k / self.n)
            b = blk["source"]([batch_size, self.k])
            y = blk["channel"](blk["mapper"](blk["encoder"](b)), no)
            return b, blk["decoder"](blk["demapper"](y, no))

    config.device = "cuda:0"
    config.seed = 42
    ber, _ = sim_ber(LinkModel(), np.arange(0.0, 5.0, 1.0),
                     batch_size=1000, max_mc_iter=20,
                     num_target_block_errors=200,
                     compile_mode="reduce-overhead", verbose=True)
    ber = ber.cpu().numpy()
    assert ber[-1] < ber[0], "BER 没有随 Eb/N0 增大而下降"


def check_sys():
    import numpy as np
    from sionna.phy import config
    from sionna.phy.utils import db_to_lin
    from sionna.phy.utils.random import uniform, normal
    from sionna.sys import PHYAbstraction, InnerLoopLinkAdaptation

    config.device = "cuda:0"
    config.seed = 42
    num_ut, num_ofdm_symbols, num_subcarriers = 20, 14, 256

    # 随机生成各用户在各资源单元上的 SINR
    # 形状：[num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
    sinr_db = uniform([1, 1, num_ut, 1], low=-5, high=30) + \
        normal([num_ofdm_symbols, num_subcarriers, num_ut, 1], mean=0, std=2)
    sinr = db_to_lin(sinr_db)

    phy_abs = PHYAbstraction()
    illa = InnerLoopLinkAdaptation(phy_abs, bler_target=0.1)
    mcs_index = illa(sinr=sinr, mcs_table_index=1)
    num_decoded_bits, harq_feedback, sinr_eff, tbler, bler = phy_abs(
        mcs_index, sinr=sinr, mcs_table_index=1, mcs_category=0)

    harq = harq_feedback.cpu().numpy()
    bits = num_decoded_bits.cpu().numpy()
    print(f"{num_ut} 个用户选出的 MCS：{mcs_index.cpu().numpy().ravel().tolist()}")
    print(f"HARQ ACK 数：{int((harq == 1).sum())}/{num_ut}，"
          f"成功译码的总比特数：{int(bits.sum())}")
    assert set(np.unique(harq).tolist()) <= {0, 1}
    assert (bits[harq == 0] == 0).all(), "译码失败的传输块不应计入比特数"


def main():
    import torch

    run_check("1. 软件包版本", check_versions)
    run_check("2. PyTorch GPU 计算", check_torch_gpu)
    for i in range(torch.cuda.device_count()):
        run_check(f"3. 链路级 Sionna PHY（cuda:{i}）", check_phy, f"cuda:{i}")
    run_check("4. torch.compile 加速的 BER 仿真", check_compile)
    run_check("5. 系统级 Sionna SYS", check_sys)

    print("\n" + "=" * 60)
    for name, ok, dt in RESULTS:
        print(f"[{'通过' if ok else '失败'}] {name}（{dt:.1f} s）")
    num_failed = sum(not ok for _, ok, _ in RESULTS)
    print("=" * 60)
    print("全部检查通过" if num_failed == 0 else f"{num_failed} 项检查失败")
    sys.exit(1 if num_failed else 0)


if __name__ == "__main__":
    main()
