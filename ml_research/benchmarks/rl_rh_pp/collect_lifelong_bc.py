"""Collect BC demos from M0 teacher on lifelong fixed schedule."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from ml_research.benchmarks.allocators import allocator_main_copy
from ml_research.common.paths import RESULTS

from .features import MAX_TASK, build_agv_tokens, build_task_tokens, teacher_m0_order
from .lifelong_env import DEFAULT_SCHEDULE, load_schedule, make_lifelong_sim

OUT = RESULTS / "rl_rh_pp"
NPZ = OUT / "bc_demos_lifelong.npz"


def collect_lifelong_bc(
    *,
    schedule_csv: Path = DEFAULT_SCHEDULE,
    max_time: int = 3000,
    min_surface: int = 3,
    max_snaps: int = 600,
) -> dict:
    schedule = load_schedule(schedule_csv, max_spawn_t=max_time)
    agv_fs, agv_ms, task_fs, task_ms, teachers = [], [], [], [], []

    def recording_allocator(sim, unassigned_agvs):
        if (
            unassigned_agvs
            and len(sim.surface_tasks or {}) >= min_surface
            and len(agv_fs) < max_snaps
        ):
            free = list(unassigned_agvs)
            tids = list(sim.surface_tasks.keys())[:MAX_TASK]
            af, am, _ = build_agv_tokens(sim, free)
            tf, tm, ordered_tids = build_task_tokens(sim, free, tids)
            order = teacher_m0_order(sim, free, ordered_tids)
            id2i = {tid: i for i, tid in enumerate(ordered_tids)}
            teacher = np.full((MAX_TASK,), -1, np.int64)
            for j, tid in enumerate(order):
                if tid in id2i:
                    teacher[j] = id2i[tid]
            agv_fs.append(af)
            agv_ms.append(am)
            task_fs.append(tf)
            task_ms.append(tm)
            teachers.append(teacher)
        return allocator_main_copy(sim, unassigned_agvs)

    _mod, sim, _inj = make_lifelong_sim(schedule, allocator=recording_allocator)

    with contextlib.redirect_stdout(io.StringIO()):
        while sim.time < max_time:
            sim.time_forward()

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        NPZ,
        agv_f=np.asarray(agv_fs, np.float32),
        agv_m=np.asarray(agv_ms, np.bool_),
        task_f=np.asarray(task_fs, np.float32),
        task_m=np.asarray(task_ms, np.bool_),
        teacher=np.asarray(teachers, np.int64),
    )
    meta = {
        "n": len(teachers),
        "path": str(NPZ),
        "schedule": str(schedule_csv),
        "max_time": max_time,
        "teacher": "m0_greedy_order",
        "mode": "on_assign_hook",
    }
    (OUT / "bc_demos_lifelong_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[RL-RH-PP] lifelong BC collected n={len(teachers)} -> {NPZ}", flush=True)
    return meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-time", type=int, default=3000)
    ap.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    ap.add_argument("--max-snaps", type=int, default=600)
    args = ap.parse_args()
    collect_lifelong_bc(
        schedule_csv=args.schedule,
        max_time=args.max_time,
        max_snaps=args.max_snaps,
    )
