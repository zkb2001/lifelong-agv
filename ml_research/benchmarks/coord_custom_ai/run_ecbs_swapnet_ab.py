"""A/B: pipeline ECBS vs ECBS+SwapNet on the same SH hard-map task subset."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.common.paths import RESULTS

OUT = RESULTS / "coord_custom_ai" / "ecbs_swapnet_ab"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", type=int, default=10)
    ap.add_argument("--max-tasks", type=int, default=20)
    ap.add_argument("--max-active", type=int, default=4)
    ap.add_argument("--weight", type=float, default=1.5)
    ap.add_argument("--time-limit", type=float, default=35.0)
    ap.add_argument("--deliver-batch", type=int, default=0)
    args = ap.parse_args()

    meta = load_custom_meta(args.slot, max_sim_time=300000, wall_timeout=1e9)
    assert meta, f"slot {args.slot} missing"

    common = dict(
        slot=args.slot,
        meta=meta,
        max_tasks=args.max_tasks,
        max_active=args.max_active,
        weight=args.weight,
        time_limit=args.time_limit,
        plan="joint",
        pipeline=True,
        turn_aware=True,
        deliver_batch=int(args.deliver_batch),
        initial_park=True,
    )

    print("\n=== arm A: ECBS pipeline (no SwapNet) ===", flush=True)
    t0 = time.perf_counter()
    a = solve_ecbs(**common, use_swapnet=False)
    a["wall_ab"] = round(time.perf_counter() - t0, 2)

    print("\n=== arm B: ECBS pipeline + SwapNet/TPTS ===", flush=True)
    t0 = time.perf_counter()
    b = solve_ecbs(**common, use_swapnet=True)
    b["wall_ab"] = round(time.perf_counter() - t0, 2)

    keys = [
        "sim_time",
        "tasks_completed",
        "completion_ratio",
        "wall_seconds",
        "validate_ok",
        "n_collisions",
        "n_swaps",
        "n_illegal_motion",
    ]
    print("\n========== ECBS ± SwapNet A/B ==========", flush=True)
    print(
        f"slot={args.slot} tasks={args.max_tasks} k={args.max_active}",
        flush=True,
    )
    print(
        f"{'metric':<22} {'ecbs':>12} {'ecbs+swap':>12} {'delta(sw-ecbs)':>14}",
        flush=True,
    )
    deltas = {}
    for k in keys:
        av, bv = a.get(k), b.get(k)
        dv = None
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            dv = bv - av
        deltas[k] = dv
        print(f"{k:<22} {av!s:>12} {bv!s:>12} {dv!s:>14}", flush=True)

    swap = b.get("swapnet") or {}
    print(
        f"swapnet mode={swap.get('swapnet_meta', {}).get('mode')} "
        f"wave_steals={swap.get('wave_steals')} "
        f"mid_handoffs={swap.get('mid_handoffs')} "
        f"yes/no={swap.get('swap_policy_yes')}/{swap.get('swap_policy_no')}",
        flush=True,
    )
    if a.get("sim_time") and b.get("sim_time"):
        pct = 100.0 * (a["sim_time"] - b["sim_time"]) / a["sim_time"]
        print(
            f"sim_time speedup: {pct:+.2f}% "
            f"(>0 means SwapNet finished earlier)",
            flush=True,
        )

    summary = {
        "slot": args.slot,
        "max_tasks": args.max_tasks,
        "max_active": args.max_active,
        "ecbs": {k: a.get(k) for k in keys + ["trajectory", "swapnet", "method"]},
        "ecbs_swapnet": {
            k: b.get(k) for k in keys + ["trajectory", "swapnet", "method"]
        },
        "delta_swap_minus_ecbs": deltas,
        "sim_time_pct_saved": (
            round(100.0 * (a["sim_time"] - b["sim_time"]) / max(1, a["sim_time"]), 3)
            if a.get("sim_time") is not None and b.get("sim_time") is not None
            else None
        ),
    }
    out = OUT / f"SH_custom_{args.slot:02d}_t{args.max_tasks}_k{args.max_active}_ab.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary -> {out}", flush=True)
    return 0 if a.get("validate_ok") and b.get("validate_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
