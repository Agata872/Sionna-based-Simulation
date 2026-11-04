"""运行目录的记录工具，各研究方向共用。

两种记录器写出相同的 rows.jsonl 与 complete.json，区别在于元数据由谁写：
- Run：自动写 metadata.json（含 git 版本）和源码快照 source_<文件名>，供 dynamic_split 及以后的实验使用；
- Recorder：只写 rows.jsonl、rows.csv、complete.json，元数据和源码快照由调用脚本自己写
  （第 0–7 轮探索与 paper1 的脚本使用，原定义在 studies/csirs_du_ru/explore_csirs_precoding.py）。

同一输出目录已有 rows.jsonl 时两者都拒绝写入，避免覆盖已有结果。
"""

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path


class Run:
    def __init__(self, directory, sources, meta):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / 'rows.jsonl').exists():
            raise FileExistsError(f'Use a fresh output directory: {self.path}')
        for src in sources:
            src = Path(src)
            (self.path / f'source_{src.name}').write_bytes(src.read_bytes())
        meta = dict(meta, started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    git_revision=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                                                text=True).stdout.strip(),
                    source_sha256={Path(s).name: hashlib.sha256(Path(s).read_bytes()).hexdigest()
                                   for s in sources})
        (self.path / 'metadata.json').write_text(json.dumps(meta, indent=2))
        self.f = (self.path / 'rows.jsonl').open('w')
        self.count = 0

    def add(self, row):
        self.f.write(json.dumps(row, allow_nan=False) + '\n')
        self.count += 1

    def finish(self, **extra):
        self.f.close()
        (self.path / 'complete.json').write_text(json.dumps(dict(
            rows=self.count, completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)))


def read_rows(directory):
    with (Path(directory) / 'rows.jsonl').open() as f:
        return [json.loads(line) for line in f]


class Recorder:
    def __init__(self, directory):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        if (self.path / "rows.jsonl").exists():
            raise FileExistsError(f"Use a fresh output directory: {self.path}")
        self.f = (self.path / "rows.jsonl").open("w")
        self.rows = []

    def add(self, row):
        self.rows.append(row)
        self.f.write(json.dumps(row, allow_nan=False) + "\n")
        self.f.flush()

    def finish(self):
        self.f.close()
        keys = sorted({k for r in self.rows for k in r})
        with (self.path / "rows.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(self.rows)
        (self.path / "complete.json").write_text(json.dumps({"rows": len(self.rows), "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
