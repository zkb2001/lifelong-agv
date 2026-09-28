"""Short smoke: blocked unload pad → TrafficNet park/semaphore path (SH01, few tasks)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.common.paths import RESULTS

OUT = RESULTS / "lane_traffic_ai" / "smoke_blocked_pad"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=1)
    ap.add_argument("--max-tasks", type=int, default=12)
    ap.add_argument("--wall-timeout", type=float, default=120.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_custom_meta(
        int(args.slot),
        max_sim_time=50000,
        wall_timeout=float(args.wall_timeout),
    )
    if not meta:
        raise SystemExit(f"slot {args.slot} missing")
    meta = dict(meta)
    meta["id"] = f"SH{args.slot:02d}_traffic_smoke"
    meta["progress_every"] = 50
    meta["traffic_recovery"] = True

    rep = solve_ecbs(
        slot=int(args.slot),
        meta=meta,
        max_tasks=int(args.max_tasks),
        use_swapnet=True,
        traffic_recovery=True,
        legacy_wave_ecbs=False,
    )
    out = {
        "method": rep.get("method"),
        "tasks_completed": rep.get("tasks_completed"),
        "tasks_total": rep.get("tasks_total"),
        "sim_time": rep.get("sim_time"),
        "validate_ok": rep.get("validate_ok"),
        "validate_summary": rep.get("validate_summary"),
        "traffic_recovery": rep.get("traffic_recovery"),
        "hierarchical": {
            "last_reason": (rep.get("hierarchical") or {}).get("last_reason"),
            "traffic_stats": (rep.get("hierarchical") or {}).get("traffic_stats"),
        },
    }
    path = OUT / f"SH{args.slot:02d}_smoke.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    print(f"[smoke] wrote {path}", flush=True)
    return 0 if float(rep.get("completion_ratio") or 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
