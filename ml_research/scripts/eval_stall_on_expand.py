"""Run champion hier_stall_on on Δsim maps; compare vs base/v3."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_research.benchmarks.coord_custom_ai.compare_100 import _base_meta, _hier_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs

OUT = Path("ml_research/results/hier_coord/stall_sim30_eval/AI_stall_on_expand.json")
BASE = {2: 4117, 5: None, 9: 3366, 10: None, 12: 3932}
V3 = {2: 4858, 9: 4173, 12: 4771}


def main():
    slots = [int(x) for x in (sys.argv[1:] or ["12", "2"])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if OUT.is_file():
        try:
            rows = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    done = {int(r["slot"]) for r in rows if "slot" in r}

    os.environ["AGV_CONFLICT_PRED"] = "0"
    os.environ["AGV_ONLINE_GUIDANCE"] = "0"
    os.environ["AGV_M0_HANDOFF"] = "1"
    os.environ["AGV_PROMOTE_ON_ASSIGN"] = "1"
    os.environ["AGV_STATION_MAX_UNLOADED"] = "1"
    os.environ["AGV_ASTAR_WALL_S"] = "0.20"
    os.environ["AGV_ASSIGN_FAIL_BUDGET"] = "4"
    os.environ["AGV_TRIED_TTL"] = "8"
    os.environ["AGV_QUIET_ASSIGN"] = "1"

    for slot in slots:
        if slot in done:
            print(f"SKIP SH{slot:02d}", flush=True)
            continue
        print(f"\n===== SH{slot:02d} hier_stall_on =====", flush=True)
        meta = _hier_meta(_base_meta(slot, wall_timeout=0.0, max_sim_time=200000))
        meta["id"] = f"SH{slot:02d}_stall_on_expand"
        rep = solve_ecbs(
            slot=int(slot),
            meta=meta,
            max_tasks=0,
            max_active=6,
            weight=1.5,
            time_limit=35.0,
            plan="joint",
            pipeline=True,
            turn_aware=True,
            use_hierarchical=True,
            hierarchical_force=False,
            use_swapnet=True,
            use_wavenet=False,
            legacy_wave_ecbs=True,
            joint_core="prioritized",
            traffic_recovery=False,
            allow_mode_switch=True,
            plan_horizon=24,
            exec_horizon=12,
            wave_hard_cap=4,
        )
        sim = int(rep.get("sim_time") or 0)
        h = rep.get("hierarchical") or {}
        base = BASE.get(slot)
        v3 = V3.get(slot)
        row = {
            "slot": slot,
            "mode": "hier_stall_on",
            "sim": sim,
            "wall": round(float(rep.get("wall_seconds") or 0), 2),
            "valid": bool(rep.get("validate_ok")),
            "done": rep.get("tasks_completed"),
            "handoff_t": h.get("m0_handoff_t"),
            "handoff_reason": h.get("m0_handoff_reason"),
            "d_base": (sim - base) if base is not None else None,
            "d_v3": (sim - v3) if v3 is not None else None,
        }
        rows.append(row)
        print("RESULT", json.dumps(row, ensure_ascii=False), flush=True)
        OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
