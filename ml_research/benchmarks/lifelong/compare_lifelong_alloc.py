"""Fair lifelong MAPD compare: identical task schedule, different allocators.

Replays a fixed spawn catalog (default: results/lifelong/live_tasks.csv from the
8000s baseline video run) under M0 / M2 / TF-RH-PP on the competition map.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import time
from pathlib import Path
from typing import Callable, List, Optional

from ml_research.benchmarks.allocators import allocator_priority_planning, patch_allocator
from ml_research.benchmarks.lifelong.run_swapnet_ab import (
    FixedScheduleInjector,
    _count_pickups,
    _count_unloads,
    _loaded_steps,
    _manhattan_travel,
)
from ml_research.common.data_utils import load_main_copy
from ml_research.common.paths import POSITION_CSV, RESULTS

from ml_research.benchmarks.rl_rh_pp.allocator_transformer import (
    allocator_transformer_pp,
    load_net,
    reset_load,
)

OUT = RESULTS / "lifelong" / "alloc_compare"
DEFAULT_SCHEDULE = RESULTS / "lifelong" / "live_tasks.csv"


def load_schedule(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append(
            {
                "spawn_t": int(r["spawn_t"]),
                "task_id": r["task_id"],
                "start_point": r["start_point"],
                "end_point": r["end_point"],
                "priority": r.get("priority", "Normal"),
                "remaining_time": r.get("remaining_time", ""),
            }
        )
    out.sort(key=lambda x: (x["spawn_t"], x["task_id"]))
    return out


def run_arm(
    name: str,
    schedule: List[dict],
    *,
    position_csv: Path,
    max_time: int,
    allocator: Optional[Callable],
) -> dict:
    mod = load_main_copy(force_reload=True)
    start_points, end_points, agv_list = mod.get_object_position(str(position_csv))
    env = mod.ENV(list(start_points.values()), list(end_points.values()))
    agv_states = mod.get_agv_state(agv_list)
    task_states = {k: [] for k in start_points}

    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )

    if name.startswith("TF"):
        reset_load()
        load_net(force=True)
    if allocator is not None:
        patch_allocator(sim, allocator)

    injector = FixedScheduleInjector(schedule, mod, start_points, end_points)
    injector.install(sim)

    t0 = time.perf_counter()
    while sim.time < max_time:
        with contextlib.redirect_stdout(io.StringIO()):
            sim.time_forward()
        if sim.time % 500 == 0:
            print(
                f"  [{name}] t={sim.time} backlog={sum(len(v) for v in sim.task_states.values())} "
                f"injected={injector.n_injected} wall={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    wall = time.perf_counter() - t0
    steps = sim.steps_reorganize()
    traj_path = OUT / f"traj_{name}.csv"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
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
    with traj_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t in sorted(steps.keys()):
            for s in steps[t]:
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

    backlog_end = sum(len(v) for v in (sim.task_states or {}).values())
    return {
        "label": name,
        "sim_time": int(sim.time),
        "wall_seconds": round(wall, 3),
        "schedule_tasks": len(schedule),
        "injected": injector.n_injected,
        "pickups": _count_pickups(steps),
        "unloads": _count_unloads(steps),
        "loaded_agv_steps": _loaded_steps(steps),
        "manhattan_travel": _manhattan_travel(steps),
        "backlog_end": backlog_end,
        "throughput": round(_count_unloads(steps) / max(1, int(sim.time)), 4),
        "trajectory": str(traj_path),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Lifelong alloc compare (fixed schedule)")
    ap.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    ap.add_argument("--position", type=Path, default=POSITION_CSV)
    ap.add_argument("--max-time", type=int, default=8000)
    ap.add_argument("--only", type=str, default="", help="comma labels e.g. M0,M2,TF")
    ap.add_argument("--tf-ckpt", type=Path, default=None, help="Transformer ckpt (default: lifelong if exists)")
    args = ap.parse_args(argv)

    if args.tf_ckpt is not None:
        os.environ["AGV_RHPP_CKPT"] = str(args.tf_ckpt)
    elif (RESULTS / "rl_rh_pp" / "priority_transformer_lifelong.pt").exists():
        os.environ["AGV_RHPP_CKPT"] = str(RESULTS / "rl_rh_pp" / "priority_transformer_lifelong.pt")

    os.environ.pop("AGV_GPU_BFS", None)
    OUT.mkdir(parents=True, exist_ok=True)

    schedule = load_schedule(args.schedule)
    schedule = [r for r in schedule if r["spawn_t"] <= args.max_time]
    print(
        f"[LIFELONG-CMP] schedule={args.schedule.name} tasks={len(schedule)} "
        f"horizon={args.max_time}",
        flush=True,
    )

    arms = [
        ("M0_baseline", None),
        ("M2_priority_pp", allocator_priority_planning),
        ("TF_RH_PP_SIL", allocator_transformer_pp),
    ]
    if args.only.strip():
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        arms = [(n, a) for n, a in arms if n.split("_")[0] in keep or n in keep]

    rows = []
    for name, alloc in arms:
        print(f"\n===== {name} =====", flush=True)
        t0 = time.perf_counter()
        r = run_arm(
            name,
            schedule,
            position_csv=args.position,
            max_time=args.max_time,
            allocator=alloc,
        )
        r["wall_clock"] = round(time.perf_counter() - t0, 2)
        rows.append(r)
        print(
            f"[{name}] unloads={r['unloads']}/{r['schedule_tasks']} "
            f"backlog_end={r['backlog_end']} thr={r['throughput']} wall={r['wall_seconds']}s",
            flush=True,
        )

    summary = {
        "scenario": "competition_map_lifelong_fixed_schedule",
        "schedule_csv": str(args.schedule.resolve()),
        "max_time": args.max_time,
        "n_scheduled_tasks": len(schedule),
        "methods": rows,
        "winner_by_unloads": max(rows, key=lambda x: x["unloads"])["label"],
    }
    out_json = OUT / f"compare_t{args.max_time}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== Lifelong fixed-schedule compare ==========", flush=True)
    print(f"tasks in schedule: {len(schedule)}  horizon: {args.max_time}", flush=True)
    print(f"{'method':<20} {'unloads':>8} {'pickups':>8} {'backlog':>8} {'thr':>8} {'wall_s':>8}", flush=True)
    for r in rows:
        print(
            f"{r['label']:<20} {r['unloads']:>8} {r['pickups']:>8} "
            f"{r['backlog_end']:>8} {r['throughput']:>8.4f} {r['wall_seconds']:>8.1f}",
            flush=True,
        )
    print(f"winner (most unloads): {summary['winner_by_unloads']}", flush=True)
    print(f"wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
