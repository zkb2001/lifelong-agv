"""A/B: When-to-Switch 3-tier (local+T_lock) vs previous local-MCTS champion.

Compares sim on SH09/12/02 against cached local_mcts_on numbers.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_research.benchmarks.coord_custom_ai.compare_100 import _base_meta, _hier_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs

OUT = Path("ml_research/results/hier_coord/stall_sim30_eval/AI_switcher_ab.json")
BASE = {9: 3366, 12: 3932, 2: 4117}
# Prior local-MCTS+LNS champion (same hier stack, no tier switcher)
PREV_ON = {9: 3342, 12: 3816, 2: 3776}


def _run(slot: int) -> dict:
    os.environ["AGV_CONFLICT_PRED"] = "0"
    os.environ["AGV_ONLINE_GUIDANCE"] = "0"
    os.environ["AGV_LOCAL_MCTS"] = "1"
    os.environ["AGV_TIER_SWITCHER"] = "1"
    os.environ["AGV_M0_HANDOFF"] = "1"
    os.environ["AGV_PROMOTE_ON_ASSIGN"] = "1"
    os.environ["AGV_STATION_MAX_UNLOADED"] = "1"
    os.environ["AGV_ASTAR_WALL_S"] = "0.20"
    os.environ["AGV_ASSIGN_FAIL_BUDGET"] = "4"
    os.environ["AGV_TRIED_TTL"] = "8"
    os.environ["AGV_QUIET_ASSIGN"] = "1"

    meta = _hier_meta(_base_meta(slot, wall_timeout=0.0, max_sim_time=200000))
    meta["id"] = f"SH{slot:02d}_switcher_v2"
    meta["use_local_mcts"] = True
    meta["use_tier_switcher"] = True

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
    prev = PREV_ON.get(slot, 0)
    base = BASE.get(slot, 0)
    return {
        "slot": slot,
        "mode": "tier_switcher_v2",
        "sim": sim,
        "wall": round(float(rep.get("wall_seconds") or 0), 2),
        "valid": bool(rep.get("validate_ok")),
        "done": rep.get("tasks_completed"),
        "handoff_t": h.get("m0_handoff_t"),
        "handoff_reason": h.get("m0_handoff_reason"),
        "base": base,
        "prev_local_on": prev,
        "d_base": sim - base,
        "d_prev": sim - prev,
        "beats_base": sim < base,
        "beats_prev": sim < prev,
    }


def main():
    slots = [int(x) for x in (sys.argv[1:] or ["12", "9", "2"])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if OUT.is_file():
        try:
            rows = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    done = {int(r["slot"]) for r in rows if r.get("mode") == "tier_switcher_v2"}

    for slot in slots:
        if slot in done:
            print(f"SKIP SH{slot:02d}", flush=True)
            continue
        print(f"\n===== SH{slot:02d} tier_switcher =====", flush=True)
        row = _run(slot)
        rows.append(row)
        print("RESULT", json.dumps(row, ensure_ascii=False), flush=True)
        OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("WROTE", OUT, flush=True)
    print("\n=== SWITCHER vs local-MCTS ON (sim) ===", flush=True)
    for r in sorted(rows, key=lambda x: x["slot"]):
        print(
            f"  SH{r['slot']:02d}: switcher={r['sim']} prev_on={r['prev_local_on']} "
            f"Δ={r['d_prev']:+d} vs_base={r['d_base']:+d} "
            f"{'HELPS' if r['beats_prev'] else 'no_gain'} "
            f"{'BEATS_BASE' if r['beats_base'] else 'above_base'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
