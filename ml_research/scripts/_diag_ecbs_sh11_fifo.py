"""Dump FIFO / premature issues for latest ECBS SH11 traj."""
from __future__ import annotations

import json
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    format_validation_summary,
    validate_hybrid_trajectory,
)

JSONL = Path("ml_research/results/coord_custom_ai/compare_100/results.jsonl")
rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
r = next(x for x in rows if x.get("method_id") == "ecbs" and int(x["slot"]) == 11)
meta = dict(load_custom_meta(11, n_tasks=100, max_sim_time=500000, wall_timeout=0))
meta["id"] = r["scenario_id"]
val = validate_hybrid_trajectory(meta, Path(r["trajectory"]))
iss = val.get("issues") or {}
print("summary", format_validation_summary(val))
print("n_premature", iss.get("n_premature_unload"))
print("n_fifo", iss.get("n_fifo_violations"), "n_display", iss.get("n_display_mismatch"))
for e in (iss.get("fifo_violations") or [])[:10]:
    print("fifo", e)
for e in (iss.get("display_mismatch") or [])[:10]:
    print("disp", e)
for e in (iss.get("task_carry_violations") or [])[:5]:
    print("carry", e)
