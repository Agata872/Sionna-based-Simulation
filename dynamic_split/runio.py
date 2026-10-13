"""运行目录：metadata.json、源码快照、rows.jsonl、complete.json（与 paper1 的约定一致）。"""

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
