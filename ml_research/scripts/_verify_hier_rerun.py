"""Print hier status for slots 6/8/12 after unlimited-wall rerun."""
from __future__ import annotations

import json

from ml_research.benchmarks.coord_custom_ai.compare_100 import JSONL, _write_summary

rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
_write_summary(rows)
hier = {int(r["slot"]): r for r in rows if r.get("method_id") == "hier"}
for s in (6, 8, 12):
    r = hier.get(s)
    if not r:
        print(f"SH{s:02d} missing")
        continue
    print(
        f"SH{s:02d} done={r.get('tasks_completed')}/{r.get('tasks_total')} "
        f"valid={r.get('validate_ok')} wall={r.get('wall_seconds')} "
        f"sim={r.get('sim_time')} summary={r.get('validate_summary')}"
    )
print(f"total_rows={len(rows)} hier={len(hier)}")
