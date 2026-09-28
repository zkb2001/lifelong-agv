"""Remove SH16 hier row so compare_100 --resume can overwrite."""
from __future__ import annotations

import json

from ml_research.benchmarks.coord_custom_ai.compare_100 import JSONL, _write_summary

rows = []
removed = 0
for line in JSONL.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("method_id") == "hier" and int(r.get("slot") or -1) == 16:
        removed += 1
        continue
    rows.append(r)
JSONL.write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    encoding="utf-8",
)
_write_summary(rows)
print(f"removed={removed} kept={len(rows)}")
