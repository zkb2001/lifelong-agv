"""Station-aware allocator rules: multi-AGV same pickup + ETA/FIFO consistency.

Problem (dock wander)
---------------------
Far AGV gets earlier FIFO task T1 at station S; near AGV later gets T2 at S and
arrives first, then idles until T1 is picked.

Literature anchors
------------------
- **TPTS** (Ma et al., AAMAS 2017): nearer free agent may take over a task still
  en-route to pickup (task swap / cancel + rematch).
- **FIFO pickup reservation** (warehouse MAPD solvers): order at the dock must
  match assignment order — arrival inversion causes waiting.
- **Capacity / buffer limits** at stations: allow several in-flight agents, but
  not unbounded crowding.

Design (adaptive, multi-car OK)
-------------------------------
1. **Capacity** ``max_unloaded`` >= 1 (default 3, adaptive up to ``hard_cap``).
2. **ETA / FIFO consistency gate**: a new assignee may join only if they will not
   arrive substantially *before* existing unloaded in-flight (who hold earlier
   queue work). Inversion → block (swap tick should give them the earlier task).
3. **Soft occupancy penalty**: scales with queue length and ETA inversion risk.
4. **TPTS-style swap**: free nearer AGV cancels far unloaded in-flight so the
   restored surface head rematches to the near car.

Does not edit ``main copy.py``. Used by M5/M8 and optionally M4.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from .common import is_urgent_task, travel_distance
from .m5_replan import (
    _has_picked_up,
    _is_loaded,
    cancel_inflight,
    ensure_m5_hooks,
)

Cell = Tuple[int, int]

# Defaults: several concurrent approaches per station OK (surface promote pipeline).
DEFAULT_MAX_UNLOADED = int(os.environ.get("AGV_STATION_MAX_UNLOADED", "3") or "3")
DEFAULT_HARD_CAP = int(os.environ.get("AGV_STATION_HARD_CAP", "5") or "5")
DEFAULT_ARRIVAL_SLACK = float(os.environ.get("AGV_STATION_ARRIVAL_SLACK", "2.0") or "2.0")
DEFAULT_SWAP_MARGIN = float(os.environ.get("AGV_STATION_SWAP_MARGIN", "4.0") or "4.0")


def _pickup_key(info: dict) -> str:
    name = info.get("pickup_name")
    if name:
        return str(name)
    p = info.get("pickup_point")
    if p is not None:
        return f"xy:{int(p[0])},{int(p[1])}"
    return "unknown"


def _eta(sim, agv, pickup) -> float:
    d = float(travel_distance(sim, agv.state[:2], pickup))
    if d == float("inf"):
        return float("inf")
    return d


def adaptive_max_unloaded(
    sim,
    task_info: dict,
    *,
    base: int = DEFAULT_MAX_UNLOADED,
    hard_cap: int = DEFAULT_HARD_CAP,
) -> int:
    """Raise capacity when the station has backlog / many free AGVs nearby."""
    base = max(1, int(base))
    hard_cap = max(base, int(hard_cap))
    key = _pickup_key(task_info)
    # Surface heads at this station (usually 0–1) + pending queue depth proxy
    surface_here = 0
    fifo_depth = 0.0
    for info in (getattr(sim, "surface_tasks", None) or {}).values():
        if _pickup_key(info) != key:
            continue
        surface_here += 1
        before = info.get("numbers_before_urgent", -1)
        left = info.get("numbers_left", -1)
        if before is not None and int(before) >= 0:
            fifo_depth = max(fifo_depth, float(before) + 1.0)
        if left is not None and int(left) >= 0:
            fifo_depth = max(fifo_depth, float(left) + 1.0)
    free_n = len(getattr(sim, "get_unassigned_agvs", lambda: [])() or [])
    # More backlog / free fleet → allow more concurrent approaches
    bump = 0
    if fifo_depth >= 2 or surface_here >= 1:
        bump += 1
    if free_n >= 4:
        bump += 1
    return int(min(hard_cap, max(base, base + bump - 1)))


def unloaded_inflight_by_station(sim) -> Dict[str, List[dict]]:
    """pickup_key → list of {agv, eta, info, tid} for not-yet-loaded in-flight."""
    out: Dict[str, List[dict]] = {}
    inflight = getattr(sim, "_inflight_tasks", None) or {}
    for agv in getattr(sim, "agvs", []) or []:
        tid = agv.task_id
        if tid is None or str(tid).startswith("escape_"):
            continue
        info = inflight.get(agv.name)
        if not info:
            continue
        if _is_loaded(agv, sim.time) or _has_picked_up(agv, tid, sim.time, sim):
            continue
        key = _pickup_key(info)
        pickup = info.get("pickup_point")
        if pickup is None:
            continue
        eta = _eta(sim, agv, pickup)
        out.setdefault(key, []).append(
            {"agv": agv, "eta": eta, "info": info, "tid": tid, "name": agv.name}
        )
    for key in out:
        out[key].sort(key=lambda r: (r["eta"], r["name"]))
    return out


def station_gate_allows(
    sim,
    agv,
    task_info: dict,
    *,
    max_unloaded: Optional[int] = None,
    allow_urgent_bypass: bool = True,
    arrival_slack: float = DEFAULT_ARRIVAL_SLACK,
    enforce_eta_order: bool = True,
) -> bool:
    """Whether ``agv`` may take ``task_info`` at this station now.

    Multi-car is allowed up to capacity. Reject when:
      - station already has ``max_unloaded`` unloaded in-flight, or
      - assignee would arrive *much earlier* than existing unloaded traffic
        (FIFO arrival inversion → dock wander). Prefer swap of earlier task.
    """
    key = _pickup_key(task_info)
    groups = unloaded_inflight_by_station(sim)
    cur = [r for r in (groups.get(key) or []) if r["name"] != agv.name]

    cap = (
        int(max_unloaded)
        if max_unloaded is not None
        else adaptive_max_unloaded(sim, task_info)
    )
    if len(cur) >= cap:
        if allow_urgent_bypass and is_urgent_task(task_info):
            pickup = task_info.get("pickup_point")
            if pickup is None:
                return False
            my_eta = _eta(sim, agv, pickup)
            worst = max(r["eta"] for r in cur)
            return my_eta + 2.0 < worst
        return False

    if not enforce_eta_order or not cur:
        return True

    pickup = task_info.get("pickup_point")
    if pickup is None:
        return True
    my_eta = _eta(sim, agv, pickup)
    if my_eta >= float("inf"):
        return False
    # Existing unloaded AGVs hold earlier (or concurrent) FIFO work.
    # If I beat their ETA by more than slack, I would wait at the dock.
    earliest = min(r["eta"] for r in cur)
    if my_eta + float(arrival_slack) < earliest:
        return False
    return True


def station_occupancy_penalty(
    sim,
    agv,
    task_info: dict,
    *,
    weight: float = 8.0,
) -> float:
    """Soft cost: queue length + ETA inversion risk (adaptive, not a hard ban)."""
    key = _pickup_key(task_info)
    groups = unloaded_inflight_by_station(sim)
    cur = [r for r in (groups.get(key) or []) if r["name"] != agv.name]
    if not cur:
        return 0.0
    pickup = task_info.get("pickup_point")
    n = float(len(cur))
    if pickup is None:
        return weight * n

    my_eta = _eta(sim, agv, pickup)
    best_inflight = cur[0]["eta"]
    # Mild base for sharing the station (multi-car OK)
    pen = weight * 0.35 * n
    # Extra if I would arrive before the earliest in-flight (inversion)
    if my_eta + DEFAULT_ARRIVAL_SLACK < best_inflight:
        invert = best_inflight - my_eta
        pen += weight * (0.8 + 0.25 * invert)
    elif my_eta <= best_inflight + 1.0:
        # Arrive almost together — small crowding cost
        pen += weight * 0.25
    else:
        # Arrive after them — healthy FIFO-compatible convoy
        pen += 0.15 * (my_eta - best_inflight)
    return float(pen)


def station_eta_swap_tick(
    sim,
    *,
    margin: float = DEFAULT_SWAP_MARGIN,
    max_swaps: int = 2,
) -> int:
    """TPTS-style: cancel far unloaded in-flight when a free nearer AGV exists.

    Urgent/Emergency tasks use a lower ETA margin and optional SwapNet policy.
    """
    if not getattr(sim, "_inflight_tasks", None):
        return 0
    free = [
        a
        for a in sim.get_unassigned_agvs()
        if sim.assign_cooldown.get(a.name, -1) <= sim.time
    ]
    if not free:
        return 0

    margin = float(getattr(sim, "_station_eta_margin", margin))
    policy = getattr(sim, "_swap_policy", None)
    groups = unloaded_inflight_by_station(sim)
    cancelled = 0
    stats = getattr(sim, "_replan_stats", None)
    if stats is not None and "station_swap" not in stats:
        stats["station_swap"] = 0

    # Urgent candidates first (SLA rescue)
    def _row_sort_key(r):
        urgent = is_urgent_task(r["info"])
        rt = r["info"].get("remaining_time")
        sla_pressure = 0.0
        if urgent and rt is not None:
            try:
                sla_pressure = max(0.0, r["eta"] / max(1.0, float(rt)))
            except (TypeError, ValueError):
                sla_pressure = 0.5
        return (0 if urgent else 1, -sla_pressure, -r["eta"])

    for key, rows in groups.items():
        if cancelled >= max_swaps:
            break
        for row in sorted(rows, key=_row_sort_key):
            if cancelled >= max_swaps:
                break
            info = row["info"]
            pickup = info.get("pickup_point")
            if pickup is None:
                continue
            far = row["agv"]
            far_eta = row["eta"]
            urgent = is_urgent_task(info)
            best_free = None
            best_eta = float("inf")
            for a in free:
                e = _eta(sim, a, pickup)
                if e < best_eta:
                    best_eta = e
                    best_free = a
            if best_free is None or best_eta >= float("inf"):
                continue
            gap = far_eta - best_eta
            eff_margin = margin * (0.45 if urgent else 1.0)
            if gap < eff_margin:
                if not (
                    urgent
                    and info.get("remaining_time") is not None
                    and far_eta > float(info["remaining_time"])
                    and best_eta < float(info["remaining_time"])
                ):
                    continue
            do_swap = True
            if callable(policy):
                do_swap = bool(
                    policy(
                        sim,
                        far,
                        best_free,
                        key,
                        far_eta,
                        best_eta,
                        gap,
                        key,
                    )
                )
            if not do_swap:
                continue
            if cancel_inflight(sim, far, reason=f"station_eta_swap<{key}"):
                cancelled += 1
                if stats is not None:
                    stats["station_swap"] = int(stats.get("station_swap", 0)) + 1
                free = [a for a in free if a.name != best_free.name]
                tag = "URGENT" if urgent else "normal"
                print(
                    f"[M8-station] swap({tag}): free {best_free.name} eta={best_eta:.1f} "
                    f"< inflight {far.name} eta={far_eta:.1f} gap={gap:.1f} @ {key}",
                    flush=True,
                )
    return cancelled


def ensure_station_eta_hooks(
    sim,
    *,
    max_unloaded: Optional[int] = None,
    margin: float = DEFAULT_SWAP_MARGIN,
    arrival_slack: float = DEFAULT_ARRIVAL_SLACK,
) -> None:
    """Mark sim flags; M5 hooks must already track ``_inflight_tasks``."""
    ensure_m5_hooks(sim, enable_replan=True)
    sim._station_eta_on = True
    sim._station_eta_max_unloaded = (
        int(max_unloaded) if max_unloaded is not None else DEFAULT_MAX_UNLOADED
    )
    sim._station_eta_margin = float(margin)
    sim._station_eta_arrival_slack = float(arrival_slack)
    if not hasattr(sim, "_replan_stats") or sim._replan_stats is None:
        sim._replan_stats = {}
    sim._replan_stats.setdefault("station_swap", 0)
    sim._replan_stats.setdefault("station_block", 0)


def apply_station_mask_to_pairs(
    sim,
    agvs,
    task_ids,
    mask,
    costs,
    *,
    max_task: int,
    penalty_weight: float = 8.0,
) -> Tuple[Any, Any]:
    """In-place: clear mask / inflate costs for ETA-inconsistent pairs (for M4)."""
    import numpy as np

    blocked = 0
    for i, agv in enumerate(agvs):
        for j, tid in enumerate(task_ids):
            idx = i * max_task + j
            if idx >= len(mask) or not mask[idx]:
                continue
            info = sim.surface_tasks.get(tid)
            if info is None:
                continue
            if not station_gate_allows(sim, agv, info):
                mask[idx] = False
                costs[idx] = 1e6
                blocked += 1
                continue
            pen = station_occupancy_penalty(sim, agv, info, weight=penalty_weight)
            if pen > 0 and costs[idx] < 1e5:
                costs[idx] = float(costs[idx]) + float(pen)
    if blocked and hasattr(sim, "_replan_stats"):
        sim._replan_stats["station_block"] = int(
            sim._replan_stats.get("station_block", 0)
        ) + blocked
    return mask, costs


def wrap_allocator_with_station_eta(
    base_allocator,
    *,
    max_unloaded: Optional[int] = None,
    margin: float = DEFAULT_SWAP_MARGIN,
    penalty_weight: float = 8.0,
):
    """Wrap a (sim, unassigned) → assigned allocator with station ETA gate + swap."""

    def allocator(sim, unassigned_agvs):
        ensure_station_eta_hooks(sim, max_unloaded=max_unloaded, margin=margin)
        station_eta_swap_tick(sim, margin=margin, max_swaps=2)
        return allocator_station_eta_pp(
            sim,
            unassigned_agvs,
            max_unloaded=max_unloaded,
            penalty_weight=penalty_weight,
        )

    return allocator


def allocator_station_eta_pp(
    sim,
    unassigned_agvs,
    *,
    max_unloaded: Optional[int] = None,
    penalty_weight: float = 8.0,
):
    """Priority Planning + adaptive capacity + ETA-order gate + soft penalty."""
    from .common import enhanced_pair_cost, make_assigned, maybe_evacuate_on_block

    if not unassigned_agvs or not sim.surface_tasks:
        maybe_evacuate_on_block(sim)
        return []

    def _prio_key(tid: str) -> tuple:
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

    ordered = sorted(sim.surface_tasks.keys(), key=_prio_key)
    fallback = None
    blocked = 0
    for tid in ordered:
        info = sim.surface_tasks[tid]
        candidates = []
        for agv in unassigned_agvs:
            if not station_gate_allows(
                sim,
                agv,
                info,
                max_unloaded=max_unloaded,
                allow_urgent_bypass=True,
            ):
                blocked += 1
                continue
            c = enhanced_pair_cost(sim, agv, tid, info, urgency_mode="m0_plus")
            if c >= float("inf"):
                continue
            c = c + station_occupancy_penalty(sim, agv, info, weight=penalty_weight)
            candidates.append((c, agv))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0])
        c, agv = candidates[0]
        far_non_urgent = (
            not is_urgent_task(info)
            and info.get("numbers_before_urgent", -1) < 0
            and c > 18
        )
        if far_non_urgent:
            if fallback is None or c < fallback[0]:
                fallback = (c, agv, tid)
            continue
        assigned = make_assigned(sim, agv, tid)
        print(f"分配[M8-PP]: AGV {agv.name} → 任务 {tid}, 代价: {c:.2f}", flush=True)
        return assigned

    if blocked and hasattr(sim, "_replan_stats"):
        sim._replan_stats["station_block"] = int(
            sim._replan_stats.get("station_block", 0)
        ) + blocked

    if fallback is not None:
        c, agv, tid = fallback
        info = sim.surface_tasks.get(tid)
        if info is not None and station_gate_allows(
            sim, agv, info, max_unloaded=max_unloaded
        ):
            assigned = make_assigned(sim, agv, tid)
            print(
                f"分配[M8-PP-far]: AGV {agv.name} → 任务 {tid}, 代价: {c:.2f}",
                flush=True,
            )
            return assigned

    maybe_evacuate_on_block(sim)
    return []
