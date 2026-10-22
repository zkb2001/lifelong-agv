"""Ablate AI levers for lower sim on Δsim-hot maps (SH09/12/02).

Must call solve_ecbs with use_hierarchical=True (matches compare_100 hier).
Previous inline ablation omitted that flag → pure ECBS, useless for sim compare.
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

OUT = Path("ml_research/results/hier_coord/stall_sim30_eval/AI_sim_ablation_v2.json")
BASE = {9: 3366, 12: 3932, 2: 4117}
V3 = {9: 4173, 12: 4771, 2: 4858}

# (name, meta_overrides, env_overrides)
# Priority: online guidance (GGO-style) + conflict; stall on/off as controls.
# Dropped traffic_recovery: hier M0 call sites often force traffic_recovery=False,
# and m0_handoff_enabled=False risks infinite plateau thrash.
MODES = [
    (
        "hier_online_guidance",
        {"use_stall_escalate": False},
        {"AGV_CONFLICT_PRED": "0", "AGV_ONLINE_GUIDANCE": "1", "AGV_GUIDANCE_ALPHA": "0.75"},
    ),
    (
        "hier_guidance_plus_stall",
        {},
        {"AGV_CONFLICT_PRED": "0", "AGV_ONLINE_GUIDANCE": "1", "AGV_GUIDANCE_ALPHA": "0.75"},
    ),
    (
        "hier_conflict_pred",
        {"use_stall_escalate": False},
        {"AGV_CONFLICT_PRED": "1", "AGV_ONLINE_GUIDANCE": "0"},
    ),
    ("hier_stall_on", {}, {"AGV_CONFLICT_PRED": "0", "AGV_ONLINE_GUIDANCE": "0"}),
    ("hier_stall_off", {"use_stall_escalate": False}, {"AGV_CONFLICT_PRED": "0", "AGV_ONLINE_GUIDANCE": "0"}),
]


def _run_hier(slot: int, meta: dict):
    os.environ["AGV_M0_HANDOFF"] = "1"
    os.environ["AGV_PROMOTE_ON_ASSIGN"] = "1"
    os.environ["AGV_STATION_MAX_UNLOADED"] = "1"
    os.environ["AGV_ASTAR_WALL_S"] = "0.20"
    os.environ["AGV_ASSIGN_FAIL_BUDGET"] = "4"
    os.environ["AGV_TRIED_TTL"] = "8"
    os.environ["AGV_QUIET_ASSIGN"] = "1"
    return solve_ecbs(
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
        traffic_recovery=bool(meta.get("traffic_recovery", False)),
        allow_mode_switch=True,
        plan_horizon=24,
        exec_horizon=12,
        wave_hard_cap=4,
    )


def main():
    slots = [int(x) for x in (sys.argv[1:] or ["9", "12", "2"])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if OUT.is_file():
        try:
            rows = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    done = {(int(r["slot"]), str(r["mode"])) for r in rows if "slot" in r and "mode" in r}

    for slot in slots:
        for name, ov, env in MODES:
            key = (slot, name)
            if key in done:
                print(f"SKIP SH{slot:02d} {name}", flush=True)
                continue
            print(f"\n===== SH{slot:02d} mode={name} =====", flush=True)
            for k, v in env.items():
                os.environ[k] = str(v)
            # reset neural toggles that cache at import time
            try:
                from simulation import engine as eng

                eng.set_neural_astar(
                    conflict=env.get("AGV_CONFLICT_PRED", "0") in ("1", "true", "True"),
                    online_guidance=env.get("AGV_ONLINE_GUIDANCE", "0") in ("1", "true", "True"),
                )
                if "AGV_GUIDANCE_ALPHA" in env:
                    eng.GUIDANCE_ALPHA = float(env["AGV_GUIDANCE_ALPHA"])
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] set_neural_astar: {exc}", flush=True)

            meta = _hier_meta(_base_meta(slot, wall_timeout=0.0, max_sim_time=200000))
            meta["id"] = f"SH{slot:02d}_abl_{name}"
            meta.update(ov)
            rep = _run_hier(slot, meta)
            sim = int(rep.get("sim_time") or 0)
            h = rep.get("hierarchical") or {}
            row = {
                "slot": slot,
                "mode": name,
                "sim": sim,
                "wall": round(float(rep.get("wall_seconds") or 0), 2),
                "valid": bool(rep.get("validate_ok")),
                "done": rep.get("tasks_completed"),
                "handoff_t": h.get("m0_handoff_t"),
                "handoff_reason": h.get("m0_handoff_reason"),
                "d_base": sim - BASE.get(slot, 0),
                "d_v3": sim - V3.get(slot, 0),
            }
            rows.append(row)
            print("RESULT", json.dumps(row, ensure_ascii=False), flush=True)
            OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("WROTE", OUT, flush=True)
    # summary pivot
    by = {}
    for r in rows:
        by.setdefault(r["mode"], []).append(r)
    print("\n=== SUMMARY (sim / d_base) ===", flush=True)
    for mode, rs in by.items():
        sims = ", ".join(f"SH{r['slot']:02d}={r['sim']}(d{r['d_base']:+d})" for r in sorted(rs, key=lambda x: x["slot"]))
        print(f"  {mode}: {sims}", flush=True)


if __name__ == "__main__":
    main()
