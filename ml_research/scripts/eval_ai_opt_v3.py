"""A/B: CostTracer + disperse-goal MCTS (+ optional switcher v3) vs champions.

Champions from prior rounds:
  SH09 switcher 2471 | SH12 local 3816 | SH02 local 3776
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

OUT = Path("ml_research/results/hier_coord/stall_sim30_eval/AI_opt_v3_ab.json")
BASE = {9: 3366, 12: 3932, 2: 4117}
CHAMP = {9: 2471, 12: 3816, 2: 3776}  # best known per map
LOCAL_ON = {9: 3342, 12: 3816, 2: 3776}

# local_opt = disperse goals + CostTracer MCTS (no switcher)
# switch_v3b = local_opt + T_lock only on assign/pure-MCTS (not LNS force)
MODES = [
    ("local_opt", False),
    ("switch_v3b", True),
]


def _run(slot: int, *, switcher: bool) -> dict:
    os.environ["AGV_CONFLICT_PRED"] = "0"
    os.environ["AGV_ONLINE_GUIDANCE"] = "0"
    os.environ["AGV_LOCAL_MCTS"] = "1"
    os.environ["AGV_TIER_SWITCHER"] = "1" if switcher else "0"
    os.environ["AGV_M0_HANDOFF"] = "1"
    os.environ["AGV_PROMOTE_ON_ASSIGN"] = "1"
    os.environ["AGV_STATION_MAX_UNLOADED"] = "1"
    os.environ["AGV_ASTAR_WALL_S"] = "0.20"
    os.environ["AGV_ASSIGN_FAIL_BUDGET"] = "4"
    os.environ["AGV_TRIED_TTL"] = "8"
    os.environ["AGV_QUIET_ASSIGN"] = "1"

    mode = "switch_v3b" if switcher else "local_opt"
    meta = _hier_meta(_base_meta(slot, wall_timeout=0.0, max_sim_time=200000))
    meta["id"] = f"SH{slot:02d}_{mode}"
    meta["use_local_mcts"] = True
    meta["use_tier_switcher"] = bool(switcher)

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
        "mode": mode,
        "sim": sim,
        "wall": round(float(rep.get("wall_seconds") or 0), 2),
        "valid": bool(rep.get("validate_ok")),
        "done": rep.get("tasks_completed"),
        "handoff_t": h.get("m0_handoff_t"),
        "handoff_reason": h.get("m0_handoff_reason"),
        "base": BASE.get(slot),
        "champ": CHAMP.get(slot),
        "prev_local": LOCAL_ON.get(slot),
        "d_base": sim - BASE.get(slot, 0),
        "d_champ": sim - CHAMP.get(slot, 0),
        "d_prev_local": sim - LOCAL_ON.get(slot, 0),
        "beats_base": sim < BASE.get(slot, 10**9),
        "beats_champ": sim < CHAMP.get(slot, 10**9),
    }


def main():
    slots = [int(x) for x in (sys.argv[1:] or ["9", "12", "2"])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if OUT.is_file():
        try:
            rows = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    done = {(int(r["slot"]), str(r["mode"])) for r in rows if "slot" in r}

    for slot in slots:
        for name, sw in MODES:
            if (slot, name) in done:
                print(f"SKIP SH{slot:02d} {name}", flush=True)
                continue
            print(f"\n===== SH{slot:02d} {name} =====", flush=True)
            row = _run(slot, switcher=sw)
            rows.append(row)
            print("RESULT", json.dumps(row, ensure_ascii=False), flush=True)
            OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("WROTE", OUT, flush=True)
    print("\n=== VERDICT (sim; lower better) ===", flush=True)
    by_slot: dict = {}
    for r in rows:
        by_slot.setdefault(r["slot"], {})[r["mode"]] = r
    for slot in sorted(by_slot):
        lo = by_slot[slot].get("local_opt")
        sw = by_slot[slot].get("switch_v3b") or by_slot[slot].get("switch_v3")
        champ = CHAMP.get(slot)
        for label, r in (("local_opt", lo), ("switch", sw)):
            if not r:
                continue
            print(
                f"  SH{slot:02d} {label}: sim={r['sim']} "
                f"Δchamp={r['d_champ']:+d} Δlocal={r['d_prev_local']:+d} "
                f"handoff={r.get('handoff_t')} "
                f"{'BEATS_CHAMP' if r['beats_champ'] else 'below_champ'}",
                flush=True,
            )
        if lo and sw:
            print(
                f"  SH{slot:02d} pick={'switch' if sw['sim'] < lo['sim'] else 'local_opt'} "
                f"(champ={champ})",
                flush=True,
            )


if __name__ == "__main__":
    main()
