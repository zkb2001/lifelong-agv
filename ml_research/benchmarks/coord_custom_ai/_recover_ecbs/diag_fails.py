"""Diagnose map_net_v1 FAIL slots."""
from __future__ import annotations

import json
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    format_validation_summary,
    validate_hybrid_trajectory,
)
from ml_research.common.paths import RESULTS

summ = json.loads(
    (RESULTS / "coord_custom_ai" / "hier_sh_smoke" / "map_net_v1" / "summary_t12_k6.json").read_text(
        encoding="utf-8"
    )
)
traj_dir = RESULTS / "coord_custom_ai" / "ecbs" / "trajectories"
for row in summ["rows"]:
    if row.get("status") != "FAIL":
        continue
    slot = int(row["slot"])
    meta = load_custom_meta(slot)
    cands = sorted(
        traj_dir.glob(f"SH_custom_{slot:02d}_ecbs_*_hier.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        print(f"SH{slot:02d} no traj")
        continue
    v = validate_hybrid_trajectory(meta, cands[0])
    iss = v.get("issues") or {}
    keys = [
        "n_collisions",
        "n_swaps",
        "n_hard_wall",
        "n_illegal_motion",
        "n_task_carry_violations",
        "n_fifo_violations",
        "n_display_mismatch",
    ]
    bad = {k: iss.get(k) for k in keys if iss.get(k)}
    print(
        f"SH{slot:02d} cr={row.get('completion_ratio')} "
        f"sim={row.get('sim_time')} failed={row.get('n_failed')} {bad}"
    )
    for kind in [
        "task_carry_violations",
        "illegal_motion",
        "collisions",
        "swaps",
        "fifo_violations",
    ]:
        arr = iss.get(kind) or []
        if arr:
            print(" ", kind, arr[0])
