"""Run baseline (M0) / PP (M2) / custom allocator on a scenario."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path
from typing import Callable, Optional

from ml_research.common.paths import POSITION_CSV, TASK_CSV

from .allocators import allocator_priority_planning, patch_allocator
from .common import (
    BENCH_OUT,
    analyze_trajectory_conflicts,
    collect_task_rt,
    load_scenario,
    patch_conflict_free_execution,
    patch_extra_obstacles,
    patch_moving_obstacle_horizon,
    run_sim_loop,
    urgent_completion_stats,
    write_trajectory,
)


def _finalize(
    method: str,
    sim,
    steps: dict,
    n_tasks: int,
    wall: float,
    forced: bool,
    task_csv: Path,
    traj_path: Path,
    notes: str = "",
    save_trajectory: bool = True,
) -> dict:
    if save_trajectory:
        write_trajectory(traj_path, steps)
    left = sum(len(v) for v in sim.task_states.values())
    completed = n_tasks - left
    rt = collect_task_rt(task_csv)
    urgent = urgent_completion_stats(steps, rt)
    conf = analyze_trajectory_conflicts(steps)
    return {
        "method": method,
        "sim_time": int(sim.time),
        "tasks_completed": int(completed),
        "tasks_total": int(n_tasks),
        "completion_ratio": round(completed / max(1, n_tasks), 4),
        "wall_seconds": round(wall, 3),
        "n_conflicts": int(conf["n_conflicts"]),
        "collisions": int(conf["collisions"]),
        "swaps": int(conf["swaps"]),
        "conflict_free": bool(conf["conflict_free"]),
        "urgent_on_time": urgent["urgent_on_time"],
        "urgent_late": urgent["urgent_late"],
        "forced_stop": bool(forced and left > 0),
        "trajectory": str(traj_path) if save_trajectory else "",
        "notes": notes,
    }


def _run_with_allocator(
    method: str,
    task_csv,
    position_csv,
    max_time: int,
    allocator: Optional[Callable],
    notes: str = "",
    scenario_tag: str = "normal",
    wall_timeout: Optional[float] = None,
    extra_obstacles=None,
    save_trajectory: bool = True,
    **_ignored,
) -> dict:
    task_csv = Path(task_csv)
    position_csv = Path(position_csv)
    mod, env, agv_states, task_states, n_tasks = load_scenario(task_csv, position_csv)
    if extra_obstacles:
        patch_extra_obstacles(env, list(extra_obstacles))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )
    patch_moving_obstacle_horizon(sim)
    patch_conflict_free_execution(sim)
    if allocator is not None:
        patch_allocator(sim, allocator)
    t0 = time.perf_counter()
    traj = (BENCH_OUT / f"{scenario_tag}_{method}.csv") if save_trajectory else None
    steps, forced = run_sim_loop(
        sim,
        max_time=max_time,
        label=f"{scenario_tag}/{method}",
        checkpoint_path=traj if save_trajectory else None,
        checkpoint_every=100 if save_trajectory else 0,
        wall_timeout=wall_timeout,
    )
    wall = time.perf_counter() - t0
    traj_out = traj if traj is not None else BENCH_OUT / f"{scenario_tag}_{method}.csv"
    return _finalize(
        method,
        sim,
        steps,
        n_tasks,
        wall,
        forced,
        task_csv,
        traj_out,
        notes,
        save_trajectory=save_trajectory,
    )


def run_m0(
    task_csv,
    position_csv=POSITION_CSV,
    max_time=2000,
    scenario_tag="normal",
    wall_timeout=None,
    extra_obstacles=None,
    **_kw,
):
    return _run_with_allocator(
        "M0_baseline_greedy",
        task_csv,
        position_csv,
        max_time,
        allocator=None,
        notes="engine greedy + spacetime A*",
        scenario_tag=scenario_tag,
        wall_timeout=wall_timeout,
        extra_obstacles=extra_obstacles,
    )


def run_m2(
    task_csv,
    position_csv=POSITION_CSV,
    max_time=2000,
    scenario_tag="normal",
    wall_timeout=None,
    extra_obstacles=None,
    **_kw,
):
    return _run_with_allocator(
        "M2_priority_planning",
        task_csv,
        position_csv,
        max_time,
        allocator=allocator_priority_planning,
        notes="PP + spacetime A*",
        scenario_tag=scenario_tag,
        wall_timeout=wall_timeout,
        extra_obstacles=extra_obstacles,
    )


METHODS = {"M0": run_m0, "M2": run_m2}


def run_method(name: str, task_csv, position_csv=POSITION_CSV, max_time=2000, **kwargs):
    key = name.upper().split("_")[0]
    if key not in METHODS:
        raise ValueError(f"Unknown method {name}; choose from {list(METHODS)}")
    return METHODS[key](task_csv, position_csv=position_csv, max_time=max_time, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["M0", "M2"])
    ap.add_argument("--task", default=str(TASK_CSV))
    ap.add_argument("--position", default=str(POSITION_CSV))
    ap.add_argument("--max-time", type=int, default=2000)
    ap.add_argument("--wall", type=float, default=None)
    args = ap.parse_args()
    rep = run_method(
        args.method,
        args.task,
        position_csv=args.position,
        max_time=args.max_time,
        wall_timeout=args.wall,
    )
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
