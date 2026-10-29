#!/usr/bin/env python3
"""GPU 自检：同一输入反复计算，结果必须逐位相同、不出现下标越界断言。

WSL 笔记本的 RTX 4060 在持续负载下会读错显存（见 README"只用 CPU 的原因"），短时间的单个算子测试
发现不了。换一台机器（Colab、服务器）跑仿真之前先运行本脚本，两项都 PASS 才使用 GPU。

    1. 纯 PyTorch 的类 BP 循环：gather → tanh / 乘积 → 按下标写回，每次 20 轮，持续 --seconds 秒
    2. Sionna 的 LDPC5GDecoder：同一组 LLR 反复译码 --repeats 次（码长取闭环中出现过的值）

用法：
    python cqi_compression/gpu_selfcheck.py                  # 默认 cuda:0，每项约 60 s
    python cqi_compression/gpu_selfcheck.py --device cpu --seconds 10 --repeats 5   # 对照
退出码：0 为全部通过，1 为有一项失败。
"""

import argparse
import sys
import time

import torch


def bp_like(device: str, seconds: float) -> str:
    gen = torch.Generator(device="cpu").manual_seed(0)
    edges, num_cn, deg = 47712, 4000, 12
    gidx = torch.randint(0, edges, (num_cn, deg), generator=gen, dtype=torch.int32).to(device)
    perm = torch.randperm(edges, generator=gen).to(torch.int32).to(device)
    valid = torch.randperm(num_cn * deg, generator=gen)[:edges].to(device)
    llr = torch.randn(42, edges, generator=gen).to(device)

    def decode():
        m = llr.clone()
        for _ in range(20):
            t = torch.tanh(m[:, perm][:, gidx].clamp(-20, 20) / 2)
            p = torch.prod(t, dim=-1, keepdim=True) / torch.where(t.abs() < 1e-7, torch.full_like(t, 1e-7), t)
            c = 2 * torch.atanh(p.clamp(-0.999999, 0.999999))
            out = torch.zeros(42, edges, device=device)
            out[:, perm] = c.reshape(42, -1)[:, valid]
            m = llr + out - out.mean(dim=1, keepdim=True)
        return m

    ref = decode().cpu()
    t0, runs, bad = time.time(), 0, 0
    while time.time() - t0 < seconds:
        bad += int(not torch.equal(decode().cpu(), ref))
        runs += 1
    return f"{runs} 次中 {bad} 次与第一次不同" if bad else f"{runs} 次结果全部相同"


def ldpc(device: str, repeats: int) -> str:
    import sionna.phy
    from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
    sionna.phy.config.device = device
    gen = torch.Generator(device="cpu").manual_seed(0)
    notes = []
    for k, n in ((7344, 10800), (6656, 10800), (8448, 12000)):
        dec = LDPC5GDecoder(LDPC5GEncoder(k, n, device=device), num_iter=20, device=device)
        llr = (torch.randn(42, n, generator=gen) * 2 + 1).to(device)
        ref = dec(llr).cpu()
        bad = sum(int(not torch.equal(dec(llr).cpu(), ref)) for _ in range(repeats))
        if bad:
            notes.append(f"k={k} n={n}：{repeats} 次中 {bad} 次不同")
    return "；".join(notes) if notes else f"3 种码长各 {repeats} 次结果全部相同"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()
    torch.set_grad_enabled(False)
    if args.device.startswith("cuda"):
        print(f"GPU：{torch.cuda.get_device_name(args.device)}，torch {torch.__version__}（CUDA {torch.version.cuda}）")
    ok = True
    for name, fn in (("类 BP 循环（纯 PyTorch）", lambda: bp_like(args.device, args.seconds)),
                     ("Sionna LDPC 重复译码", lambda: ldpc(args.device, args.repeats))):
        try:
            msg = fn()
            passed = "全部相同" in msg
        except Exception as e:                       # 越界断言、illegal instruction 等
            msg, passed = f"{type(e).__name__}: {str(e).splitlines()[0]}", False
        ok &= passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}：{msg}", flush=True)
        if not passed and args.device.startswith("cuda"):
            break                                    # CUDA 出错后上下文不可用，后续测试没有意义
    print("结论：" + ("可以使用该设备" if ok else "不要在该设备上跑仿真"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
