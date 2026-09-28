"""Same-horizon LMAPF / lifelong MAPD throughput: M0 / M2 / Transformer."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path
from typing import Callable, List, Optional

from ml_research.benchmarks.allocators import allocator_priority_planning, patch_allocator
from ml_research.benchmarks.common import (
    load_scenario,
    patch_extra_obstacles,
    patch_moving_obstacle_horizon,
    sanitize_sim_end_points,
    sanitize_task_end_points,
)
from ml_research.common.paths import POSITION_CSV, RESULTS, TASK_CSV

from .allocator_transformer import allocator_transformer_pp, load_net, reset_load

OUT = RESULTS / "rl_rh_pp"
OUT.mkdir(parents=True, exist_ok=True)


def _resolve_scenario(args) -> dict:
    if args.slot is not None:
        from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta

        meta = load_custom_meta(args.slot, max_sim_time=args.max_time)
        if meta is None:
            raise FileNotFoundError(f"SH_custom_{args.slot:02d} exports missing")
        return {
            "tag": meta["id"],
            "task_csv": Path(meta["task_csv"]),
            "position_csv": Path(meta["position_csv"]),
            "extra_obstacles": list(meta.get("extra_obstacles") or []),
        }
    return {
        "tag": "original_csv",
        "task_csv": Path(args.task or TASK_CSV),
        "position_csv": Path(args.position or POSITION_CSV),
        "extra_obstacles": [],
    }


def _curve(
    name: str,
    allocator: Optional[Callable],
    *,
    task_csv: Path,
    position_csv: Path,
    extra_obstacles: list,
    max_time: int,
    every: int,
) -> dict:
    import sys

    mod, env, agv_states, task_states, n_tasks = load_scenario(task_csv, position_csv)
    if extra_obstacles:
        patch_extra_obstacles(env, list(extra_obstacles))
        sanitize_task_end_points(env, task_states)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )
    if extra_obstacles:
        sanitize_sim_end_points(sim)
    patch_moving_obstacle_horizon(sim)
    if name.startswith("TF"):
        reset_load()
        load_net(force=True)
    if allocator is not None:
        patch_allocator(sim, allocator)

    curve: List[dict] = []
    t_wall0 = time.perf_counter()
    forced = False
    with contextlib.redirect_stdout(io.StringIO()):
        while not sim.all_over():
            sim.time_forward()
            left = sum(len(v) for v in sim.task_states.values())
            done = int(n_tasks - left)
            if sim.time % every == 0 or left <= 0:
                curve.append({"t": int(sim.time), "done": done, "left": int(left)})
                print(
                    f"  [{name}] t={sim.time} done={done}/{n_tasks}",
                    file=sys.stderr,
                    flush=True,
                )
            if sim.time >= max_time:
                forced = True
                break
    wall = time.perf_counter() - t_wall0
    left = sum(len(v) for v in sim.task_states.values())
    done = int(n_tasks - left)

    def t_at(k: int) -> Optional[int]:
        for row in curve:
            if row["done"] >= k:
                return int(row["t"])
        return None

    return {
        "label": name,
        "tasks_total": int(n_tasks),
        "tasks_completed": done,
        "completion_ratio": round(done / max(1, n_tasks), 4),
        "sim_time": int(sim.time),
        "forced_stop": bool(forced and left > 0),
        "wall_seconds": round(wall, 3),
        "t_done_50": t_at(max(1, n_tasks // 2)),
        "t_done_80": t_at(max(1, int(0.8 * n_tasks))),
        "t_done_100": t_at(n_tasks) if done >= n_tasks else None,
        "throughput": round(done / max(1, int(sim.time)), 4),
        "curve": curve,
    }


def _done_at(curve: list, t: int) -> int:
    last = 0
    for row in curve:
        if row["t"] <= t:
            last = int(row["done"])
        else:
            break
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-time", type=int, default=8000, help="sim horizon (default 8000)")
    ap.add_argument("--every", type=int, default=50, help="log curve every N sim steps")
    ap.add_argument("--slot", type=int, default=None, help="SH_custom slot (lifelong maze)")
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--position", type=str, default=None)
    args = ap.parse_args()

    sc = _resolve_scenario(args)
    kw = dict(
        task_csv=sc["task_csv"],
        position_csv=sc["position_csv"],
        extra_obstacles=sc["extra_obstacles"],
        max_time=args.max_time,
        every=args.every,
    )

    rows = [
        _curve("M0_baseline", None, **kw),
        _curve("M2_priority_pp", allocator_priority_planning, **kw),
        _curve("TF_RH_PP_SIL", allocator_transformer_pp, **kw),
    ]

    if args.max_time >= 8000:
        snaps = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    elif args.max_time >= 2000:
        snaps = [200, 500, 1000, 1500, 2000]
    else:
        snaps = [50, 100, 150, 200, 250, 300, 400, 500]
    snaps = [t for t in snaps if t <= args.max_time]

    snapshot = {
        str(t): {r["label"]: _done_at(r["curve"], t) for r in rows} for t in snaps
    }
    summary = {
        "scenario": sc["tag"],
        "task_csv": str(sc["task_csv"]),
        "position_csv": str(sc["position_csv"]),
        "max_time": args.max_time,
        "same_time_done": snapshot,
        "methods": [
            {
                k: r[k]
                for k in (
                    "label",
                    "tasks_completed",
                    "completion_ratio",
                    "sim_time",
                    "t_done_50",
                    "t_done_80",
                    "t_done_100",
                    "throughput",
                    "wall_seconds",
                    "forced_stop",
                )
            }
            for r in rows
        ],
        "curves": {r["label"]: r["curve"] for r in rows},
    }
    suffix = sc["tag"].lower().replace(" ", "_")
    path = OUT / f"lmapf_speed_{suffix}_t{args.max_time}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("scenario", "max_time", "same_time_done", "methods")}, indent=2), flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
