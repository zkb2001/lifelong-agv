"""Lifelong MAPD with pure spacetime-A* baseline (simulation/engine ≈ main copy).

No conflict-shield / M5 replan / path_fallback / Hungarian stack — only:
  - Simulation.time_forward + native greedy assign
  - engine.A_Star path planning
  - lifelong task injector
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from ml_research.benchmarks.lifelong.task_generator import (
    LifelongTaskGenerator,
    install_lifelong_generator,
)
from ml_research.common.data_utils import load_main_copy
from ml_research.common.paths import POSITION_CSV, RESULTS, ROOT

OUT = RESULTS / "lifelong"
TRAJ = OUT / "live_trajectory.csv"
TASKS = OUT / "live_tasks.csv"
STATUS = OUT / "live_status.txt"

TRAJ_HEADER = [
    "timestamp",
    "name",
    "X",
    "Y",
    "pitch",
    "loaded",
    "destination",
    "Emergency",
    "task-id",
]


def _write_trajectory(path: Path, steps_dict: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TRAJ_HEADER)
        for t in sorted(steps_dict.keys()):
            for s in steps_dict[t]:
                w.writerow(
                    [
                        s["timestamp"],
                        s["name"],
                        s["X"],
                        s["Y"],
                        s["pitch"],
                        str(s["loaded"]).lower(),
                        s["destination"],
                        str(s["Emergency"]).lower(),
                        s.get("task-id", ""),
                    ]
                )


def _write_status(sim, gen: LifelongTaskGenerator, wall: float) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(
        "\n".join(
            [
                f"t={sim.time}",
                f"backlog={gen.backlog(sim)}",
                f"surface={len(sim.surface_tasks)}",
                f"generated={gen.n_generated}",
                f"urgent={gen.n_urgent}",
                f"wall_s={wall:.1f}",
            ]
        ),
        encoding="utf-8",
    )


def run_lifelong(
    *,
    position_csv: Path = POSITION_CSV,
    max_time: int = 5000,
    wall_timeout: Optional[float] = None,
    seed: int = 0,
    min_backlog: int = 12,
    max_backlog: int = 28,
    inject_every: int = 3,
    urgent_prob: float = 0.12,
    checkpoint_every: int = 5,
    with_monitor: bool = False,
) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (TRAJ, TASKS, STATUS):
        if p.exists():
            p.unlink()

    mod = load_main_copy(force_reload=True)
    start_points, end_points, agv_list = mod.get_object_position(str(position_csv))
    env = mod.ENV(list(start_points.values()), list(end_points.values()))
    agv_states = mod.get_agv_state(agv_list)
    task_states = {name: [] for name in start_points}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )

    # Pure baseline: do NOT patch conflict shield / M5 / fallback / allocator.

    gen = LifelongTaskGenerator(
        pickups=list(start_points.keys()),
        dropoffs=list(end_points.keys()),
        start_points=start_points,
        end_points=end_points,
        mod=mod,
        seed=seed,
        min_backlog=min_backlog,
        max_backlog=max_backlog,
        inject_every=inject_every,
        urgent_prob=urgent_prob,
        catalog_csv=TASKS,
    )
    seeded = gen.seed(sim)
    install_lifelong_generator(sim, gen)
    print(
        f"[LIFELONG-A*] pure engine baseline | seeded={seeded} "
        f"pickups={len(start_points)} dropoffs={len(end_points)} "
        f"min/max backlog={min_backlog}/{max_backlog}",
        flush=True,
    )

    mon_proc = None
    if with_monitor:
        mon_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ml_research.benchmarks.lifelong.live_monitor",
                "--traj",
                str(TRAJ),
                "--tasks",
                str(TASKS),
                "--position",
                str(position_csv),
                "--fps",
                "12",
            ],
            cwd=str(ROOT),
        )
        print(f"[LIFELONG-A*] live monitor pid={mon_proc.pid}", flush=True)
        time.sleep(0.8)

    t0 = time.perf_counter()
    forced = False
    try:
        while True:
            sim.time_forward()
            wall = time.perf_counter() - t0
            if sim.time % checkpoint_every == 0:
                _write_trajectory(TRAJ, sim.steps_reorganize())
                _write_status(sim, gen, wall)
            if sim.time % 50 == 0:
                print(
                    f"  [lifelong-A*] t={sim.time} backlog={gen.backlog(sim)} "
                    f"gen={gen.n_generated} surface={len(sim.surface_tasks)} wall={wall:.1f}s",
                    flush=True,
                )
            if sim.time >= max_time:
                forced = True
                break
            if wall_timeout is not None and wall >= wall_timeout:
                forced = True
                break
    except KeyboardInterrupt:
        print("[LIFELONG-A*] interrupted by user", flush=True)
        forced = True
    finally:
        _write_trajectory(TRAJ, sim.steps_reorganize())
        _write_status(sim, gen, time.perf_counter() - t0)

    wall = time.perf_counter() - t0
    out = {
        "method": "astar_maincopy_lifelong",
        "sim_time": int(sim.time),
        "wall_seconds": round(wall, 3),
        "generated": gen.n_generated,
        "urgent_generated": gen.n_urgent,
        "backlog": gen.backlog(sim),
        "forced_stop": forced,
        "trajectory": str(TRAJ),
        "tasks": str(TASKS),
    }
    print(f"[LIFELONG-A*] done {out}", flush=True)
    if mon_proc is not None:
        print("[LIFELONG-A*] monitor still open — close the window when finished viewing", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Lifelong pure A* baseline (simulation/engine, no common shields)"
    )
    ap.add_argument("--position", type=Path, default=POSITION_CSV)
    ap.add_argument("--max-time", type=int, default=2000)
    ap.add_argument("--wall-timeout", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-backlog", type=int, default=12)
    ap.add_argument("--max-backlog", type=int, default=28)
    ap.add_argument("--inject-every", type=int, default=3)
    ap.add_argument("--urgent-prob", type=float, default=0.12)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--with-monitor", action="store_true")
    args = ap.parse_args(argv)

    os.environ.pop("AGV_GPU_BFS", None)
    run_lifelong(
        position_csv=args.position,
        max_time=args.max_time,
        wall_timeout=args.wall_timeout,
        seed=args.seed,
        min_backlog=args.min_backlog,
        max_backlog=args.max_backlog,
        inject_every=args.inject_every,
        urgent_prob=args.urgent_prob,
        checkpoint_every=args.checkpoint_every,
        with_monitor=args.with_monitor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
