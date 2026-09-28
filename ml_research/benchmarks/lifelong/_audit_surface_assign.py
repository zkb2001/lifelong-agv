"""Verify assignments are always current surface tasks; surface never skips FIFO head."""
from __future__ import annotations

import contextlib
import io
import os
from collections import defaultdict
from pathlib import Path

os.environ.pop("AGV_GPU_BFS", None)

from ml_research.benchmarks.allocators import make_complete_allocator, patch_allocator
from ml_research.benchmarks.common import (
    get_mod,
    patch_conflict_free_execution,
    patch_moving_obstacle_horizon,
    set_pair_cost_mode,
)
from ml_research.benchmarks.lifelong.task_generator import (
    LifelongTaskGenerator,
    install_lifelong_generator,
)
from ml_research.benchmarks.m5_replan import ensure_m5_hooks, m5_replan_enabled
from ml_research.benchmarks.path_fallback import patch_path_planning_fallback
from ml_research.common.paths import POSITION_CSV


def _station_surface(sim, pickup: str):
    return [
        tid
        for tid, info in (sim.surface_tasks or {}).items()
        if info.get("pickup_name") == pickup
    ]


def _queue_head(sim, pickup: str):
    q = sim.task_states.get(pickup) or []
    return q[0].get("task_id") if q else None


def install_surface_audit(sim) -> dict:
    """Record violations around assign / promote / expose."""
    stats = {
        "assigns": 0,
        "assign_not_on_surface": [],
        "surface_not_queue_head": [],
        "surface_while_head_inflight": [],
        "multi_surface_same_station": [],
        "checks": 0,
    }

    def audit(tag: str):
        stats["checks"] += 1
        # multi surface per station
        by_st = defaultdict(list)
        for tid, info in list((sim.surface_tasks or {}).items()):
            by_st[info.get("pickup_name")].append(tid)
        for st, tids in by_st.items():
            if st and len(tids) > 1:
                stats["multi_surface_same_station"].append(
                    {"t": sim.time, "tag": tag, "station": st, "tids": list(tids)}
                )

        for st, queue in list((sim.task_states or {}).items()):
            if not queue:
                continue
            head = queue[0].get("task_id")
            surf = _station_surface(sim, st)
            # Who holds head?
            holders = [a.name for a in sim.agvs if a.task_id == head]
            inflight = getattr(sim, "_inflight_tasks", {}) or {}
            inflight_hold = [
                agv
                for agv, info in inflight.items()
                if info.get("task_id") == head
            ]

            if surf:
                if head not in surf:
                    # Surface shows something that is not queue head
                    stats["surface_not_queue_head"].append(
                        {
                            "t": int(sim.time),
                            "tag": tag,
                            "station": st,
                            "head": head,
                            "surface": list(surf),
                            "holders": holders,
                        }
                    )
                if holders or inflight_hold:
                    # Head already assigned → station should have empty surface
                    stats["surface_while_head_inflight"].append(
                        {
                            "t": int(sim.time),
                            "tag": tag,
                            "station": st,
                            "head": head,
                            "surface": list(surf),
                            "holders": holders,
                            "inflight": inflight_hold,
                        }
                    )

    orig_reserve = sim.reserve_assigned_task

    def reserve_audited(task_id):
        # BEFORE reserve: task must be on surface
        stats["assigns"] += 1
        if task_id not in (sim.surface_tasks or {}):
            pickup = None
            for name, q in (sim.task_states or {}).items():
                if any(x.get("task_id") == task_id for x in q):
                    pickup = name
                    break
            stats["assign_not_on_surface"].append(
                {
                    "t": int(sim.time),
                    "task_id": task_id,
                    "station": pickup,
                    "surface": list((sim.surface_tasks or {}).keys()),
                    "head": _queue_head(sim, pickup) if pickup else None,
                }
            )
        else:
            info = sim.surface_tasks[task_id]
            pickup = info.get("pickup_name")
            head = _queue_head(sim, pickup) if pickup else None
            if head != task_id:
                stats["assign_not_on_surface"].append(
                    {
                        "t": int(sim.time),
                        "task_id": task_id,
                        "station": pickup,
                        "surface": list((sim.surface_tasks or {}).keys()),
                        "head": head,
                        "reason": "surface_but_not_queue_head",
                    }
                )
        audit("pre_reserve")
        out = orig_reserve(task_id)
        audit("post_reserve")
        return out

    sim.reserve_assigned_task = reserve_audited

    orig_tf = sim.time_forward

    def tf_audited():
        orig_tf()
        audit("post_tick")

    sim.time_forward = tf_audited
    return stats


def main() -> int:
    set_pair_cost_mode("astar")
    mod = get_mod(force_reload=True)
    start_points, end_points, agv_list = mod.get_object_position(str(POSITION_CSV))
    env = mod.ENV(list(start_points.values()), list(end_points.values()))
    agv_states = mod.get_agv_state(agv_list)
    task_states = {name: [] for name in start_points}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )

    patch_conflict_free_execution(sim)
    patch_moving_obstacle_horizon(sim)
    patch_path_planning_fallback(sim)
    allocator = make_complete_allocator(
        horizon=40, enable_replan=m5_replan_enabled(True), aggressive=False, station_eta=False
    )
    patch_allocator(sim, allocator)
    ensure_m5_hooks(sim, horizon=40, enable_replan=True, aggressive=False)

    gen = LifelongTaskGenerator(
        pickups=list(start_points.keys()),
        dropoffs=list(end_points.keys()),
        start_points=start_points,
        end_points=end_points,
        mod=mod,
        seed=42,
        min_backlog=12,
        max_backlog=28,
        inject_every=3,
        urgent_prob=0.12,
        catalog_csv=None,
    )
    gen.seed(sim)
    install_lifelong_generator(sim, gen)
    stats = install_surface_audit(sim)

    max_t = 400
    for _ in range(max_t):
        sim.time_forward()

    print("=== SURFACE / ASSIGN AUDIT ===")
    print(f"sim_time={sim.time} assigns={stats['assigns']} checks={stats['checks']}")
    print(f"assign_not_on_surface={len(stats['assign_not_on_surface'])}")
    print(f"surface_not_queue_head={len(stats['surface_not_queue_head'])}")
    print(f"surface_while_head_inflight={len(stats['surface_while_head_inflight'])}")
    print(f"multi_surface_same_station={len(stats['multi_surface_same_station'])}")

    for key in (
        "assign_not_on_surface",
        "surface_not_queue_head",
        "surface_while_head_inflight",
        "multi_surface_same_station",
    ):
        rows = stats[key]
        if not rows:
            continue
        print(f"\n-- {key} sample (up to 15) --")
        for r in rows[:15]:
            print(r)

    # Also FIFO pickup check online
    # (pickup order vs queue consumption already covered by assign_not_on_surface)

    bad = (
        len(stats["assign_not_on_surface"])
        + len(stats["surface_not_queue_head"])
        + len(stats["surface_while_head_inflight"])
        + len(stats["multi_surface_same_station"])
    )
    print(f"\nVERDICT: {'FAIL' if bad else 'PASS'} (bad_events={bad})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
