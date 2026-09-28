"""Baseline greedy + Priority Planning (M2 teacher for RL-RH-PP)."""
from __future__ import annotations

import os
from typing import Callable

from .common import (
    enhanced_pair_cost,
    is_urgent_task,
    make_assigned,
    maybe_evacuate_on_block,
    travel_distance,
)


def soft_pair_rank_cost(sim, agv, task_id: str, task_info: dict) -> float:
    d = float(travel_distance(sim, agv.state[:2], task_info["pickup_point"]))
    if d == float("inf"):
        if hasattr(sim, "manhattan_distance"):
            d = float(sim.manhattan_distance(agv.state[:2], task_info["pickup_point"]))
        else:
            d = float(
                abs(int(agv.state[0]) - int(task_info["pickup_point"][0]))
                + abs(int(agv.state[1]) - int(task_info["pickup_point"][1]))
            )
        d = d + 40.0

    if (agv.name, task_id) in (getattr(sim, "tried_tasks", None) or set()):
        d += 25.0
    if (getattr(sim, "assign_cooldown", None) or {}).get(agv.name, -1) > sim.time:
        d += 15.0

    from .common import soft_endpoint_multiplier

    mult = float(soft_endpoint_multiplier(sim, task_info))
    if mult == float("inf") or mult > 50.0:
        mult = 12.0
    d *= mult
    if task_info.get("numbers_before_urgent", -1) >= 0:
        d *= 0.7
    if is_urgent_task(task_info):
        d *= 0.65
    return d


def allocator_soft_propose(sim, unassigned_agvs):
    if not unassigned_agvs or not sim.surface_tasks:
        return []
    best = None
    for agv in unassigned_agvs:
        for tid, info in sim.surface_tasks.items():
            c = soft_pair_rank_cost(sim, agv, tid, info)
            if best is None or c < best[0]:
                best = (c, agv, tid)
    if best is None:
        return []
    _, agv, tid = best
    tried = getattr(sim, "tried_tasks", None)
    if isinstance(tried, set):
        tried.discard((agv.name, tid))
    cd = getattr(sim, "assign_cooldown", None)
    if isinstance(cd, dict):
        cd.pop(agv.name, None)
    assigned = make_assigned(sim, agv, tid)
    if not assigned:
        return []
    print(
        f"分配[soft-propose]: AGV {agv.name} → 任务 {tid}, rank={best[0]:.1f}",
        flush=True,
    )
    return assigned


def _with_soft_propose(sim, unassigned_agvs, assigned):
    if assigned:
        return assigned
    if not unassigned_agvs or not getattr(sim, "surface_tasks", None):
        return assigned if assigned else []
    return allocator_soft_propose(sim, unassigned_agvs)


def patch_allocator(sim, allocator: Callable) -> Callable:
    orig = sim.get_cost_matrix_and_allocate_task

    def wrapped(unassigned_agvs):
        return allocator(sim, unassigned_agvs)

    sim.get_cost_matrix_and_allocate_task = wrapped
    return orig


def allocator_main_copy(sim, unassigned_agvs):
    """Manhattan greedy (engine-style baseline assign)."""
    if not unassigned_agvs or not sim.surface_tasks:
        return []
    min_agv = None
    min_task_id = None
    min_cost = float("inf")
    tried = getattr(sim, "tried_tasks", None) or set()
    for agv in unassigned_agvs:
        for task_id, task_info in sim.surface_tasks.items():
            cost = float(sim.manhattan_distance(agv.state[:2], task_info["pickup_point"]))
            if task_info.get("numbers_before_urgent", -1) >= 0:
                cost *= 0.7
            if (agv.name, task_id) in tried:
                cost = float("inf")
            if cost < min_cost:
                min_agv = agv
                min_task_id = task_id
                min_cost = cost
    if min_agv is None or min_task_id is None or min_cost == float("inf"):
        return []
    assigned = make_assigned(sim, min_agv, min_task_id)
    print(f"分配[baseline]: AGV {min_agv.name} → 任务 {min_task_id}, 代价: {min_cost}", flush=True)
    return assigned


def _task_priority_key(sim, tid: str, unassigned_agvs) -> tuple:
    info = sim.surface_tasks[tid]
    is_urgent = 0 if is_urgent_task(info) else 1
    rt = info.get("remaining_time")
    rt_key = float(rt) if rt is not None else 1e9
    before_u = info.get("numbers_before_urgent", -1)
    before_key = before_u if before_u >= 0 else 1e6
    min_c = min(
        (
            enhanced_pair_cost(sim, a, tid, info, urgency_mode="m0_plus")
            for a in unassigned_agvs
        ),
        default=float("inf"),
    )
    return (is_urgent, before_key, rt_key, min_c, tid)


def allocator_priority_planning(sim, unassigned_agvs):
    """M2 PP: urgent/FIFO-first, then nearest AGV (teacher for RL-RH-PP)."""
    from .common import get_pair_cost_mode

    if not unassigned_agvs or not sim.surface_tasks:
        maybe_evacuate_on_block(sim)
        return []
    ordered = sorted(
        sim.surface_tasks.keys(),
        key=lambda tid: _task_priority_key(sim, tid, unassigned_agvs),
    )
    fallback = None
    far_thresh = 18.0
    if get_pair_cost_mode() == "astar":
        far_thresh = float(os.environ.get("AGV_PP_FAR_COST", "80"))
    for tid in ordered:
        info = sim.surface_tasks[tid]
        candidates = []
        for agv in unassigned_agvs:
            c = enhanced_pair_cost(sim, agv, tid, info, urgency_mode="m0_plus")
            if c < float("inf"):
                candidates.append((c, agv))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        c, agv = candidates[0]
        far_non_urgent = (
            not is_urgent_task(info)
            and info.get("numbers_before_urgent", -1) < 0
            and c > far_thresh
        )
        if far_non_urgent:
            if fallback is None or c < fallback[0]:
                fallback = (c, agv, tid)
            continue
        assigned = make_assigned(sim, agv, tid)
        print(f"分配[PP]: AGV {agv.name} → 任务 {tid}, 代价: {c}")
        return assigned
    if fallback is not None:
        c, agv, tid = fallback
        assigned = make_assigned(sim, agv, tid)
        print(f"分配[PP-far]: AGV {agv.name} → 任务 {tid}, 代价: {c}")
        return assigned
    maybe_evacuate_on_block(sim)
    return _with_soft_propose(sim, unassigned_agvs, [])
