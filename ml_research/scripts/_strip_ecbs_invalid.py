"""Remove ECBS rows for slots 11 and 12."""
from __future__ import annotations

import json

from ml_research.benchmarks.coord_custom_ai.compare_100 import JSONL, _write_summary

WANT = {11, 12}
rows = []
removed = 0
for line in JSONL.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("method_id") == "ecbs" and int(r.get("slot") or -1) in WANT:
        removed += 1
        continue
    rows.append(r)
JSONL.write_text(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    encoding="utf-8",
)
_write_summary(rows)
print(f"removed={removed} kept={len(rows)}")
