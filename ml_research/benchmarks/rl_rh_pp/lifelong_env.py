"""Lifelong MAPD env: fixed schedule replay on competition map."""
from __future__ import annotations

import contextlib
import csv
import io
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from ml_research.benchmarks.lifelong.run_swapnet_ab import (
    FixedScheduleInjector,
    _count_unloads,
)
from ml_research.common.data_utils import load_main_copy
from ml_research.common.paths import POSITION_CSV, RESULTS

DEFAULT_SCHEDULE = RESULTS / "lifelong" / "live_tasks.csv"


def load_schedule(path: Path, *, max_spawn_t: int | None = None) -> List[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            st = int(r["spawn_t"])
            if max_spawn_t is not None and st > max_spawn_t:
                continue
            rows.append(
                {
                    "spawn_t": st,
                    "task_id": r["task_id"],
                    "start_point": r["start_point"],
                    "end_point": r["end_point"],
                    "priority": r.get("priority", "Normal"),
                    "remaining_time": r.get("remaining_time", ""),
                }
            )
    rows.sort(key=lambda x: (x["spawn_t"], x["task_id"]))
    return rows


def make_lifelong_sim(
    schedule: List[dict],
    *,
    position_csv: Path = POSITION_CSV,
    allocator: Optional[Callable] = None,
):
    mod = load_main_copy(force_reload=True)
    start_points, end_points, agv_list = mod.get_object_position(str(position_csv))
    env = mod.ENV(list(start_points.values()), list(end_points.values()))
    agv_states = mod.get_agv_state(agv_list)
    task_states = {k: [] for k in start_points}

    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )

    if allocator is not None:
        from ml_research.benchmarks.allocators import patch_allocator

        patch_allocator(sim, allocator)

    injector = FixedScheduleInjector(schedule, mod, start_points, end_points)
    injector.install(sim)
    return mod, sim, injector


def run_lifelong_episode(
    sim,
    *,
    max_time: int,
    quiet: bool = True,
) -> Tuple[dict, int]:
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        while sim.time < max_time:
            sim.time_forward()
    steps = sim.steps_reorganize()
    unloads = _count_unloads(steps)
    backlog = sum(len(v) for v in (sim.task_states or {}).values())
    return steps, int(unloads), int(backlog)


def unloads_from_steps(steps: dict) -> int:
    return _count_unloads(steps)
