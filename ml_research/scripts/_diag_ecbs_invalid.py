"""Dump premature_unload details for ECBS SH11/SH12."""
from __future__ import annotations

import json
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    validate_hybrid_trajectory,
)

JSONL = Path("ml_research/results/coord_custom_ai/compare_100/results.jsonl")
rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
for slot in (11, 12):
    r = next(x for x in rows if x.get("method_id") == "ecbs" and int(x["slot"]) == slot)
    meta = dict(load_custom_meta(slot, n_tasks=100, max_sim_time=500000, wall_timeout=0))
    meta["id"] = r["scenario_id"]
    meta["n_tasks"] = 100
    traj = Path(r["trajectory"])
    val = validate_hybrid_trajectory(meta, traj)
    iss = val.get("issues") or {}
    print(f"\n=== SH{slot:02d} ok={val.get('ok')} ===")
    print("summary fields:", {k: iss.get(k) for k in iss if str(k).startswith("n_")})
    for e in iss.get("task_carry_violations") or []:
        print(e)
