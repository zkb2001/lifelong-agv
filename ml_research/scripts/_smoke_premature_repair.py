"""Smoke-test premature_unload scrub on existing ECBS SH11/SH12 trajs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    _load_task_dropoffs,
    format_validation_summary,
    scrub_premature_unload_episodes,
    validate_hybrid_trajectory,
)

JSONL = Path("ml_research/results/coord_custom_ai/compare_100/results.jsonl")
rows_all = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]

for slot in (11, 12):
    r = next(x for x in rows_all if x.get("method_id") == "ecbs" and int(x["slot"]) == slot)
    meta = dict(load_custom_meta(slot, n_tasks=100, max_sim_time=500000, wall_timeout=0))
    meta["id"] = r["scenario_id"]
    traj = Path(r["trajectory"])
    with traj.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    before = validate_hybrid_trajectory(meta, traj)
    scrubbed = scrub_premature_unload_episodes(rows, _load_task_dropoffs(meta))
    tmp = traj.with_name(traj.stem + "_scrub_tmp.csv")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    after = validate_hybrid_trajectory(meta, tmp)
    print(
        f"SH{slot:02d} scrub={sorted(scrubbed)} before={format_validation_summary(before)} "
        f"after={format_validation_summary(after)} ok={after.get('ok')}"
    )
    tmp.unlink(missing_ok=True)
