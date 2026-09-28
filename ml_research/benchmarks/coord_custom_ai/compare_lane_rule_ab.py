"""SH03 A/B: traffic OFF / TrafficNet-lite / neural TrafficRuleNet (aligned)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.common.paths import RESULTS

OUT = RESULTS / "coord_custom_ai" / "lane_rule_ab"
OUT.mkdir(parents=True, exist_ok=True)


def _run(mode: str, slot: int = 3, max_tasks: int = 12, max_active: int = 4) -> dict:
    meta = load_custom_meta(slot, max_sim_time=300000, wall_timeout=1e9)
    use_lite = mode == "lite"
    use_lane = mode == "neural"
    t0 = time.perf_counter()
    rep = solve_ecbs(
        slot=slot,
        meta=meta,
        max_tasks=max_tasks,
        max_active=max_active,
        plan="joint",
        pipeline=True,
        turn_aware=True,
        initial_park=False,
        use_hierarchical=True,
        use_wavenet=True,
        wave_hard_cap=4,
        use_traffic_accel=use_lite,
        use_lane_rules=use_lane,
        time_limit=25.0,
    )
    wall = round(time.perf_counter() - t0, 2)
    hier = rep.get("hierarchical") or {}
    row = {
        "mode": mode,
        "slot": slot,
        "sim_time": rep.get("sim_time"),
        "wall_seconds": wall,
        "completion_ratio": rep.get("completion_ratio"),
        "validate_ok": rep.get("validate_ok"),
        "tasks_completed": rep.get("tasks_completed"),
        "accel_waves": hier.get("accel_waves"),
        "lane_rule_waves": hier.get("lane_rule_waves"),
        "agents_dropped": hier.get("agents_dropped"),
        "last_k": hier.get("last_k"),
    }
    print(
        f"  [{mode}] sim={row['sim_time']} wall={wall}s "
        f"ok={row['validate_ok']} cr={row['completion_ratio']} "
        f"accel={row['accel_waves']} lane_w={row['lane_rule_waves']}",
        flush=True,
    )
    return row


def main() -> int:
    modes = ("off", "lite", "neural")
    print("[LANE-AB] SH03 t12 hier+wavenet modes=", modes, flush=True)
    rows = []
    for m in modes:
        print(f"\n======== mode={m} ========", flush=True)
        rows.append(_run(m))
    path = OUT / "SH03_t12_off_lite_neural.json"
    path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print("\n========== SUMMARY ==========", flush=True)
    for r in rows:
        print(
            f"  {r['mode']:7s} sim={r['sim_time']} ok={r['validate_ok']} "
            f"accel={r['accel_waves']} lane_w={r['lane_rule_waves']}",
            flush=True,
        )
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
