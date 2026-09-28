"""Re-run M0 baseline on SH_custom slot, validate, optionally copy to paper traj."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    format_validation_summary,
    validate_hybrid_trajectory,
)
from ml_research.benchmarks.runner import run_m0

PAPER_TRAJ = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "curriculum_shape"
    / "paper_baseline_astar_custom"
    / "trajectories"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=10)
    ap.add_argument("--max-time", type=int, default=800)
    ap.add_argument("--wall", type=float, default=7200.0)
    ap.add_argument("--copy-paper", action="store_true", default=True)
    ap.add_argument("--no-copy-paper", action="store_false", dest="copy_paper")
    args = ap.parse_args()

    meta = load_custom_meta(args.slot, max_sim_time=args.max_time, wall_timeout=args.wall)
    if meta is None:
        raise SystemExit(f"SH_custom_{args.slot:02d} meta missing")

    sid = meta["id"]
    print(f"[rerun] {sid} max_time={args.max_time} wall={args.wall}s", flush=True)
    rep = run_m0(
        meta["task_csv"],
        position_csv=meta["position_csv"],
        max_time=args.max_time,
        scenario_tag=sid,
        wall_timeout=args.wall,
        extra_obstacles=meta.get("extra_obstacles"),
    )
    traj = Path(rep["trajectory"])
    val = validate_hybrid_trajectory(meta, traj)
    summary = format_validation_summary(val)
    print(f"[rerun] validate: {summary}", flush=True)
    print(
        "[rerun] run:",
        json.dumps(
            {
                k: rep[k]
                for k in (
                    "sim_time",
                    "tasks_completed",
                    "tasks_total",
                    "completion_ratio",
                    "collisions",
                    "swaps",
                    "conflict_free",
                    "forced_stop",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.copy_paper and val.get("ok"):
        out = PAPER_TRAJ / f"{sid}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(traj, out)
        print(f"[rerun] copied -> {out}", flush=True)
    return 0 if val.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
