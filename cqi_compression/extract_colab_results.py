#!/usr/bin/env python3
"""从 colab_continue.ipynb 的单元输出中取出 Colab 上的运行结果，解压到本机。

笔记本最后一个单元把运行目录（rows.jsonl、metadata.json、codec_sqnr.json、complete.json、main.log 等，不含图）
打包成 zip，以 base64 文本写在输出里。VS Code 连接 Colab 运行时后，输出随笔记本保存在本机；保存笔记本后运行：

    python cqi_compression/extract_colab_results.py cqi_compression/colab_continue.ipynb results/cqi_compression_20261030/colab_main
    python cqi_compression/analyse_cqi_bler.py results/cqi_compression_20261030/colab_main

只依赖标准库。提交笔记本前再清空输出。
"""

import argparse
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

BEGIN, END = "-----BEGIN RESULTS ZIP-----", "-----END RESULTS ZIP-----"


def find_payload(notebook: Path):
    """返回 (zip 字节, 笔记本中记录的 sha256)；有多个时取最后一个。"""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    found = None
    for cell in nb["cells"]:
        # 同一单元的标准输出可能被拆成多段 stream 输出，先拼起来
        text = "".join("".join(out["text"]) if isinstance(out["text"], list) else out["text"]
                       for out in cell.get("outputs", [])
                       if out.get("output_type") == "stream" and out.get("name") == "stdout")
        if BEGIN in text and END in text:
            body = text.split(BEGIN, 1)[1].split(END, 1)[0]
            lines = [ln.strip() for ln in body.strip().splitlines()]
            sha = lines[0].removeprefix("sha256 ") if lines and lines[0].startswith("sha256 ") else None
            found = (base64.b64decode("".join(ln for ln in lines if not ln.startswith("sha256 "))), sha)
    if found is None:
        raise SystemExit(f"{notebook} 的输出中没有结果包：先在 Colab 上运行最后一个单元，再保存笔记本")
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("notebook")
    p.add_argument("out_dir")
    args = p.parse_args()
    data, sha = find_payload(Path(args.notebook))
    digest = hashlib.sha256(data).hexdigest()
    if sha and digest != sha:
        raise SystemExit(f"校验失败：sha256 {digest} 与笔记本中记录的 {sha} 不同，输出可能被截断")
    out = Path(args.out_dir)
    if (out / "rows.jsonl").exists():
        raise SystemExit(f"{out} 中已有 rows.jsonl，请换一个目录")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        z.extractall(out)
    print(f"已解压 {len(names)} 个文件到 {out.resolve()}（sha256 {digest[:16]}…{'，已校验' if sha else ''}）")
    for n in names:
        print("  ", n)


if __name__ == "__main__":
    main()
