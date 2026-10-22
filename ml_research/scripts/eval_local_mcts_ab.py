"""A/B: local MCTS unstick ON vs OFF — does AI shorten sim?

Runs hier (stall + sim-first gate) with/without AGV_LOCAL_MCTS on Δsim maps.
Compares against known base sims.
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

OUT = Path("ml_research/results/hier_coord/stall_sim30_eval/AI_local_mcts_ab.json")
BASE = {9: 3366, 12: 3932, 2: 4117}
V3 = {9: 4173, 12: 4771, 2: 4858}
STALL_ON = {9: 2878, 12: 5479, 2: 5001}  # prior expand without sim-gate delay nuance

MODES = [
    ("off", False),
    ("on", True),
]


def _run(slot: int, *, local_on: bool) -> dict:
    os.environ["AGV_CONFLICT_PRED"] = "0"
    os.environ["AGV_ONLINE_GUIDANCE"] = "0"
    os.environ["AGV_LOCAL_MCTS"] = "1" if local_on else "0"
    os.environ["AGV_M0_HANDOFF"] = "1"
    os.environ["AGV_PROMOTE_ON_ASSIGN"] = "1"
    os.environ["AGV_STATION_MAX_UNLOADED"] = "1"
    os.environ["AGV_ASTAR_WALL_S"] = "0.20"
    os.environ["AGV_ASSIGN_FAIL_BUDGET"] = "4"
    os.environ["AGV_TRIED_TTL"] = "8"
    os.environ["AGV_QUIET_ASSIGN"] = "1"

    meta = _hier_meta(_base_meta(slot, wall_timeout=0.0, max_sim_time=200000))
    meta["id"] = f"SH{slot:02d}_lmcts_{'on' if local_on else 'off'}"
    meta["use_local_mcts"] = bool(local_on)

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
    return {
        "slot": slot,
        "mode": "local_mcts_on" if local_on else "local_mcts_off",
        "sim": sim,
        "wall": round(float(rep.get("wall_seconds") or 0), 2),
        "valid": bool(rep.get("validate_ok")),
        "done": rep.get("tasks_completed"),
        "handoff_t": h.get("m0_handoff_t"),
        "handoff_reason": h.get("m0_handoff_reason"),
        "base": BASE.get(slot),
        "d_base": sim - BASE.get(slot, 0),
        "d_v3": sim - V3.get(slot, 0),
        "d_stall_on_prev": sim - STALL_ON.get(slot, 0),
        "beats_base": sim < BASE.get(slot, 10**9),
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
    done = {(int(r["slot"]), str(r["mode"])) for r in rows if "slot" in r}

    for slot in slots:
        for name, on in MODES:
            mode = "local_mcts_on" if on else "local_mcts_off"
            if (slot, mode) in done:
                print(f"SKIP SH{slot:02d} {mode}", flush=True)
                continue
            print(f"\n===== SH{slot:02d} {mode} =====", flush=True)
            row = _run(slot, local_on=on)
            rows.append(row)
            print("RESULT", json.dumps(row, ensure_ascii=False), flush=True)
            OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("WROTE", OUT, flush=True)
    print("\n=== VERDICT (sim / d_base; lower better) ===", flush=True)
    by_slot: dict = {}
    for r in rows:
        by_slot.setdefault(r["slot"], {})[r["mode"]] = r
    for slot in sorted(by_slot):
        off = by_slot[slot].get("local_mcts_off")
        on = by_slot[slot].get("local_mcts_on")
        if not off or not on:
            continue
        delta = on["sim"] - off["sim"]
        vs_base = on["d_base"]
        useful = delta < 0
        print(
            f"  SH{slot:02d}: off={off['sim']} on={on['sim']} "
            f"Δon-off={delta:+d} Δon-base={vs_base:+d} "
            f"{'AI_HELPS_SIM' if useful else 'AI_NO_HELP'} "
            f"{'BEATS_BASE' if on['beats_base'] else 'above_base'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
