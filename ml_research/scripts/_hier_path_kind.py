"""Classify Hier runs by trajectory path (M0-only vs handoff→ECBS)."""
from __future__ import annotations

import json
from pathlib import Path

JSONL = Path("ml_research/results/coord_custom_ai/compare_100/results.jsonl")
rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
hier = [r for r in rows if r.get("method_id") == "hier"]
for r in sorted(hier, key=lambda x: int(x["slot"])):
    t = str(r.get("trajectory") or "")
    name = Path(t).name if t else ""
    if "m0_engine" in name:
        kind = "m0_only"
    elif "ecbs_" in name:
        kind = "handoff_ecbs"
    else:
        kind = "other"
    print(f"SH{int(r['slot']):02d}\t{kind}\t{name}")
