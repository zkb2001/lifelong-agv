"""Windowed lifelong MAPD via ECBS/WCBS (+ optional hierarchical WaveNet).

Rewritten recovery: the previous champion ``solve_ecbs.py`` was truncated and
no full source remained in-repo. This module restores a working pipeline that
matches the smoke / compare call API.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pymapf import Agent, GridMap, MAPFProblem, solve

from ml_research.benchmarks.common import TRAJ_HEADER, load_scenario, patch_extra_obstacles
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    format_validation_summary,
    validate_hybrid_trajectory,
)
from ml_research.common.paths import CKPT, RESULTS

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]

# Filled by solve_ecbs after map pickups load; False→True must land here.
_PICKUP_LEAVE: Set[Cell] = set()
# Pickup and dropoff centers. A stand is a neighbor the road can drive to.
_MAP_STATIONS: Set[Cell] = set()
_MAP_DROPOFFS: Set[Cell] = set()
_MAP_PICKUPS: Set[Cell] = set()
_DROP_NAME: Dict[Cell, str] = {}
_ROAD_ANCHORS: Tuple[Cell, ...] = ((6, 1), (10, 10), (2, 2), (19, 2), (10, 16))
# Overflow unloaders. Cargo stays on the car until a later wave has a free stand.
_STICKY: Dict[str, Dict[str, object]] = {}
_PICKUP_SKIP: Set[str] = set()
_CTX_LOADED: Optional[Dict[str, bool]] = None
_CTX_DEST: Optional[Dict[str, str]] = None
_CTX_TID: Optional[Dict[str, str]] = None
# task_id -> legal unload cells (validator ring). Used to refuse off-pad clears.
_TID_DROPOFFS: Dict[str, Set[Cell]] = {}

OUT = RESULTS / "coord_custom_ai" / "ecbs"
TRAJ = OUT / "trajectories"

# Medium/hard joint-core switch. ECBS code stays; default OFF → Prioritized.
# Override: solve_ecbs(joint_core=...) or env AGV_JOINT_CORE=ecbs|prioritized
import os as _os

DEFAULT_JOINT_CORE = str(_os.environ.get("AGV_JOINT_CORE", "prioritized") or "prioritized").strip().lower()
if DEFAULT_JOINT_CORE not in ("prioritized", "ecbs"):
    DEFAULT_JOINT_CORE = "prioritized"


def _write_queue_task_csv(queues: Dict[str, list], path: Path) -> int:
    """Materialize current FIFO queues into a task CSV for the engine runner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["task_id", "start_point", "end_point", "priority", "remaining_time"]
        )
        qs = {k: list(v) for k, v in queues.items() if v}
        while qs:
            progress = False
            for st in list(qs):
                if not qs[st]:
                    qs.pop(st, None)
                    continue
                t = qs[st].pop(0)
                w.writerow(
                    [
                        t.get("task_id"),
                        st,
                        t.get("destination") or t.get("end_point"),
                        t.get("priority", "Normal"),
                        t.get("remaining_time")
                        if t.get("remaining_time") is not None
                        else "None",
                    ]
                )
                n += 1
                progress = True
                if not qs[st]:
                    qs.pop(st, None)
            if not progress:
                break
    return n


def _legalize_engine_steps(steps: dict) -> dict:
    """Export-only: fix move_and_turn + in-place turn_gt_90 without rewriting XY."""
    by_name: Dict[str, List[dict]] = {}
    for t in sorted(steps.keys()):
        for s in steps[int(t)]:
            row = dict(s)
            row["timestamp"] = int(t)
            by_name.setdefault(str(row["name"]), []).append(row)

    for _name, seq in by_name.items():
        seq.sort(key=lambda r: int(r["timestamp"]))
        for i in range(1, len(seq)):
            a, b = seq[i - 1], seq[i]
            if int(b["timestamp"]) != int(a["timestamp"]) + 1:
                continue
            ax, ay = int(a["X"]), int(a["Y"])
            bx, by = int(b["X"]), int(b["Y"])
            dist = abs(ax - bx) + abs(ay - by)
            pa = int(a.get("pitch") or 0) % 360
            pb = int(b.get("pitch") or 0) % 360
            if dist == 1 and pa != pb:
                # move_and_turn → keep prior pitch on the move tick
                b["pitch"] = pa
            elif dist == 0:
                dp = (pb - pa) % 360
                if dp == 180:
                    # turn_gt_90 → take the +90° midpoint (next tick can finish)
                    b["pitch"] = (pa + 90) % 360

    out: dict = {}
    for _name, seq in by_name.items():
        for r in seq:
            out.setdefault(int(r["timestamp"]), []).append(r)
    for t in out:
        out[t].sort(key=lambda r: str(r["name"]))
    return out


def _m0_pose_from_sim(sim) -> Dict[str, Pose]:
    pose: Dict[str, Pose] = {}
    for agv in getattr(sim, "agvs", []) or []:
        st = agv.state
        pose[str(agv.name)] = (
            int(st[0]),
            int(st[1]),
            int(st[3]) if len(st) > 3 else 90,
        )
    return pose


def _m0_queues_from_sim(sim) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for name, q in (getattr(sim, "task_states", None) or {}).items():
        if not q:
            continue
        out[str(name)] = [dict(t) for t in q]
    return out


def _m0_assigned_from_sim(sim) -> Dict[str, dict]:
    assigned: Dict[str, dict] = {}
    inflight = getattr(sim, "_inflight_tasks", None) or {}
    for name, info in inflight.items():
        if isinstance(info, dict):
            assigned[str(name)] = dict(info)
    return assigned


def _m0_stall_signals(sim, *, n_tasks: int) -> Dict[str, Any]:
    """Detect M0 hub livelock / A* degrade signals for handoff triggers.

    Returns dict with:
      idle_all, has_work, done_est, assign_fail_streak, n_idle, n_surface,
      plan_fail_rate, plan_fail_samples, path_fail_streak
    """
    agvs = list(getattr(sim, "agvs", None) or [])
    n_idle = 0
    n_loaded = 0
    for agv in agvs:
        tid = agv.task_id
        if tid is None or str(tid).startswith("escape_"):
            n_idle += 1
            continue
        cur = _m0_step_at(agv, int(getattr(sim, "time", 0) or 0))
        loaded = False
        if cur is not None:
            loaded = str(cur.get("loaded", "")).lower() in ("true", "1", "yes")
        if loaded:
            n_loaded += 1
    n_surface = len(getattr(sim, "surface_tasks", None) or {})
    left_q = sum(len(v) for v in (getattr(sim, "task_states", None) or {}).values())
    has_work = n_surface > 0 or left_q > 0
    idle_all = bool(agvs) and n_idle >= len(agvs)
    n_busy = sum(
        1
        for a in agvs
        if a.task_id is not None and not str(a.task_id).startswith("escape_")
    )
    # Progress-compatible completion (matches run_sim_loop stderr done=).
    # Plateau must track done_completed — subtracting n_busy oscillates and
    # resets done_plateau_streak forever while true completions are stuck.
    done_completed = max(0, int(n_tasks) - int(left_q))
    done_est = max(0, int(done_completed) - int(n_busy))

    # Recent window plan/path failure rate (engine may expose counters).
    stats = getattr(sim, "_replan_stats", None)
    if not isinstance(stats, dict):
        stats = {}
    path_ok = int(stats.get("path_ok") or stats.get("plan_ok") or 0)
    path_fail = int(
        stats.get("path_fail")
        or stats.get("plan_fail")
        or stats.get("astar_fail")
        or 0
    )
    # Also fold assign failures into the window sample when present.
    assign_fail = int(getattr(sim, "_assign_fail_streak", 0) or 0)
    path_fail_streak = int(
        getattr(sim, "_path_fail_streak", None)
        or stats.get("path_fail_streak")
        or 0
    )
    samples = path_ok + path_fail
    # Do NOT bootstrap from assign_fail — that made plan_fail_rate trip too early.
    plan_fail_rate = float(path_fail) / float(max(1, samples)) if samples >= 10 else 0.0

    return {
        "idle_all": bool(idle_all),
        "has_work": bool(has_work),
        "done_est": int(done_est),
        "done_completed": int(done_completed),
        "assign_fail_streak": int(getattr(sim, "_assign_fail_streak", 0) or 0),
        "n_idle": int(n_idle),
        "n_surface": int(n_surface),
        "left_q": int(left_q),
        "n_loaded": int(n_loaded),
        "n_busy": int(n_busy),
        "plan_fail_rate": float(plan_fail_rate),
        "plan_fail_samples": int(samples),
        "path_fail_streak": int(path_fail_streak),
    }


def _m0_step_at(agv, t: int) -> Optional[dict]:
    t = int(t)
    best = None
    for s in getattr(agv, "steps", None) or []:
        ts = int(s.get("timestamp", -1))
        if ts == t:
            return dict(s)
        if ts <= t:
            best = dict(s)
    return best


def _m0_task_from_inflight(info: dict, tid: str) -> dict:
    task = dict(info)
    task["task_id"] = str(task.get("task_id") or tid)
    if "pickup_point" in task and task["pickup_point"] is not None:
        task["pickup_point"] = tuple(task["pickup_point"])
    if "end_points" in task:
        task["end_points"] = [tuple(e) for e in (task.get("end_points") or []) if e]
    return task


def _m0_task_from_step(sim, step: dict, tid: str) -> dict:
    """Rebuild a minimal delivery task from a loaded trajectory step + map pads."""
    tid_s = str(tid)
    dest = str(step.get("destination") or "").strip()
    end_points: List[Cell] = []
    # Prefer any still-cached task payload keyed by tid / agv.
    for store in (
        getattr(sim, "assigned_task_info", None),
        getattr(sim, "surface_tasks", None),
        getattr(sim, "_inflight_tasks", None),
    ):
        if not isinstance(store, dict):
            continue
        info = store.get(tid_s)
        if not isinstance(info, dict):
            for v in store.values():
                if isinstance(v, dict) and str(v.get("task_id") or "") == tid_s:
                    info = v
                    break
        if isinstance(info, dict):
            if not dest:
                dest = str(info.get("destination") or "").strip()
            for ep in info.get("end_points") or ():
                if ep is not None and len(ep) >= 2:
                    end_points.append((int(ep[0]), int(ep[1])))
            if end_points:
                break
    env = getattr(sim, "env", None)
    ends = getattr(env, "destination_points", None) if env is not None else None
    if dest and isinstance(ends, dict) and dest in ends:
        cx, cy = int(ends[dest][0]), int(ends[dest][1])
        ring = [(cx, cy)]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ring.append((cx + dx, cy + dy))
        for c in ring:
            if c not in end_points:
                end_points.append(c)
    task_out = {
        "task_id": tid_s,
        "destination": dest,
        "end_points": end_points,
        "start_point": str(step.get("pickup_name") or ""),
        "pickup_name": str(step.get("pickup_name") or ""),
    }
    task_out["end_points"] = _restrict_hub13_pads(task_out, end_points)
    return task_out


def _m0_handoff_bindings(sim) -> Tuple[Dict[str, dict], Dict[str, dict], Set[str]]:
    """pending_delivery / pending_pickup / picked_ids from live M0 engine.

    Primary source: ``agv.task_id`` + ``_inflight_tasks``. Fallback: any AGV
    whose *latest step at sim.time* is still loaded with a real task-id — this
    catches engine desync where ``task_id`` / inflight were cleared mid-trip
    while the trajectory still shows cargo (caused task_carry at handoff).
    """
    pending_delivery: Dict[str, dict] = {}
    pending_pickup: Dict[str, dict] = {}
    picked_ids: Set[str] = set(
        str(x) for x in (getattr(sim, "_pickup_committed", None) or set())
    )
    inflight = getattr(sim, "_inflight_tasks", None) or {}
    t_now = int(getattr(sim, "time", 0) or 0)
    for agv in getattr(sim, "agvs", []) or []:
        name = str(agv.name)
        tid = agv.task_id
        cur = _m0_step_at(agv, t_now)
        loaded = False
        step_tid = ""
        if cur is not None:
            loaded = str(cur.get("loaded", "")).lower() in ("true", "1", "yes")
            step_tid = str(cur.get("task-id") or "").strip()
        if tid is not None and not str(tid).startswith("escape_"):
            tid_s = str(tid)
            info = inflight.get(name)
            if isinstance(info, dict):
                task = _m0_task_from_inflight(info, tid_s)
            elif loaded and step_tid:
                task = _m0_task_from_step(sim, cur or {}, step_tid or tid_s)
            else:
                # Live task_id but no payload / not loaded → treat as approach.
                task = {"task_id": tid_s, "destination": "", "end_points": []}
            if loaded:
                pending_delivery[name] = task
                picked_ids.add(str(task.get("task_id") or tid_s))
            else:
                pending_pickup[name] = task
            continue
        # Live task_id gone/escape, but trajectory still carrying → recover.
        if (
            loaded
            and step_tid
            and not step_tid.startswith("escape_")
            and name not in pending_delivery
        ):
            task = _m0_task_from_step(sim, cur or {}, step_tid)
            pending_delivery[name] = task
            picked_ids.add(step_tid)
            # Re-attach live handles so later M0 helpers stay consistent.
            try:
                agv.task_id = step_tid
            except Exception:  # noqa: BLE001
                pass
            if isinstance(inflight, dict) and name not in inflight:
                inflight[name] = dict(task)
                inflight[name]["agv"] = name
    return pending_delivery, pending_pickup, picked_ids


def _steps_dict_to_steps_by(
    steps: dict, names: List[str], now: int
) -> Dict[str, List[dict]]:
    """Convert engine steps[t]->rows into wave-loop steps_by[name]->rows."""
    steps_by: Dict[str, List[dict]] = {n: [] for n in names}
    for t in sorted(int(x) for x in steps.keys()):
        if t > int(now):
            continue
        for s in steps[t]:
            name = str(s.get("name") or "")
            if name not in steps_by:
                steps_by[name] = []
            row = dict(s)
            row["timestamp"] = int(t)
            # Normalize loaded flags to wave-loop style strings.
            loaded = str(row.get("loaded", "")).lower() in ("true", "1", "yes")
            row["loaded"] = "TRUE" if loaded else "FALSE"
            em = str(row.get("Emergency", "")).lower() in ("true", "1", "yes")
            row["Emergency"] = "TRUE" if em else "FALSE"
            steps_by[name].append(row)
    for n in list(steps_by):
        seq = steps_by[n]
        if not seq:
            continue
        last_t = int(seq[-1]["timestamp"])
        if last_t < int(now):
            prev = seq[-1]
            for t_fill in range(last_t + 1, int(now) + 1):
                seq.append(
                    {
                        "timestamp": t_fill,
                        "name": n,
                        "X": int(prev["X"]),
                        "Y": int(prev["Y"]),
                        "pitch": int(prev.get("pitch") or 90) % 360,
                        "loaded": prev.get("loaded", "FALSE"),
                        "destination": prev.get("destination", ""),
                        "Emergency": prev.get("Emergency", "FALSE"),
                        "task-id": prev.get("task-id", ""),
                    }
                )
                prev = seq[-1]
    return steps_by


def _build_m0_handoff_snapshot(
    *,
    sim,
    steps: dict,
    n_tasks: int,
    traj: Path,
    remain_csv: Path,
    handoff_label: str,
) -> dict:
    """Freeze M0 state for wave-loop warm start."""
    now = int(getattr(sim, "time", 0) or 0)
    pose = _m0_pose_from_sim(sim)
    queues = _m0_queues_from_sim(sim)
    pending_delivery, pending_pickup, picked_ids = _m0_handoff_bindings(sim)

    # Unpicked assigned tasks stay as queue heads until rising-edge pickup.
    # Popping here orphans the tid when approach fails without requeue
    # (r55–r63 Dragon-3 → surface_head=66). Inflight blocks double-claim.
    for agv, task in list(pending_pickup.items()):
        st = str(
            task.get("pickup_name")
            or task.get("start_point")
            or (
                str(task.get("task_id") or "").rsplit("-", 1)[0]
                if "-" in str(task.get("task_id") or "")
                else ""
            )
        )
        tid_s = str(task.get("task_id") or "").strip()
        if tid_s:
            picked_ids.discard(tid_s)
        if st and queues.get(st):
            head = queues[st][0] if queues[st] else None
            if head is not None and str(head.get("task_id") or "") == tid_s:
                pending_pickup[agv] = dict(head)
            elif tid_s:
                for trow in list(queues.get(st) or []):
                    if str(trow.get("task_id") or "") == tid_s:
                        pending_pickup[agv] = dict(trow)
                        break
                else:
                    # Not in queue — insert as head so FIFO cannot skip it.
                    body = dict(task)
                    body.setdefault("task_id", tid_s)
                    queues.setdefault(st, []).insert(0, body)
                    pending_pickup[agv] = dict(queues[st][0])
        elif tid_s and st:
            body = dict(task)
            body.setdefault("task_id", tid_s)
            queues.setdefault(st, []).insert(0, body)
            pending_pickup[agv] = dict(queues[st][0])

    n_loaded = len(pending_delivery)
    left_q = sum(len(v) for v in queues.values())
    # Remaining work ≈ queue + carriers (pending_pickup already removed from queues).
    remaining = left_q + n_loaded + len(pending_pickup)
    done = max(0, int(n_tasks) - int(remaining))

    names = sorted(pose.keys())
    steps_by = _steps_dict_to_steps_by(steps, names, now)
    for n, p in pose.items():
        if not steps_by.get(n):
            steps_by[n] = [_hold(n, p, now)]
        else:
            # Align last pose with live sim pose at handoff tick.
            last = steps_by[n][-1]
            last["X"], last["Y"] = int(p[0]), int(p[1])
            last["pitch"] = int(p[2]) % 360
            if n in pending_delivery:
                task = pending_delivery[n]
                last["loaded"] = "TRUE"
                last["destination"] = str(task.get("destination") or "")
                last["task-id"] = str(task.get("task_id") or "")
            elif n in pending_pickup:
                task = pending_pickup[n]
                last["loaded"] = "FALSE"
                last["destination"] = ""
                last["task-id"] = str(task.get("task_id") or "")

    n_remain = _write_queue_task_csv(queues, remain_csv)
    return {
        "handoff": True,
        "handoff_label": str(handoff_label),
        "now": now,
        "pose": pose,
        "queues": queues,
        "pending_delivery": pending_delivery,
        "pending_pickup": pending_pickup,
        "picked_ids": sorted(picked_ids),
        "done": int(done),
        "tasks_remaining_csv": str(remain_csv),
        "tasks_remaining": int(n_remain),
        "steps_by": steps_by,
        "trajectory_prefix": str(traj),
        "agv_snapshot": {
            n: {
                "pose": pose[n],
                "loaded": n in pending_delivery,
                "task_id": str(
                    (pending_delivery.get(n) or pending_pickup.get(n) or {}).get(
                        "task_id"
                    )
                    or ""
                ),
            }
            for n in names
        },
    }


def _solve_via_m0_engine(
    *,
    meta: dict,
    task_csv: Path,
    total: int,
    use_swapnet: bool,
    swapnet_margin: float,
    hier_stats: dict,
    t0: float,
    plan: str,
    max_active: int,
    weight: float,
    pipeline: bool,
    turn_aware: bool,
    plan_horizon: int,
    exec_horizon: int,
    use_hierarchical: bool,
    use_wavenet: bool,
    gate: Any = None,
    allow_yield: bool = False,
    traffic_recovery: bool = True,
) -> dict:
    """Default path: Simulation greedy + spacetime A* (+ SwapNet + rule recovery).

    Wave-batched 'astar' / ECBS inside solve_ecbs is legacy-only
    (``legacy_wave_ecbs=True``). Rule recovery activates on path/assign
    failure (freq artery/parking + rule semaphore) — no SceneDifficulty handoff.

    ``allow_yield`` + gate handoff is disabled when ``traffic_recovery`` is on.
    """
    import contextlib
    import io

    from ml_research.benchmarks.common import (
        analyze_trajectory_conflicts,
        patch_conflict_free_execution,
        patch_moving_obstacle_horizon,
        run_sim_loop,
        write_trajectory,
    )

    # Rule recovery replaces classifier→ECBS escalate
    if traffic_recovery:
        allow_yield = False
        gate = None

    print(
        f"[M0] engine"
        f"{'+SwapNet' if use_swapnet else ''}"
        f"{'+RulePark' if traffic_recovery else ''} "
        f"tasks={total}",
        flush=True,
    )
    hier_stats = dict(hier_stats)
    hier_stats["last_planner"] = "m0_engine"
    hier_stats["last_reason"] = "m0_traffic_default" if traffic_recovery else "scene_easy_m0_engine"
    hier_stats["traffic_recovery"] = bool(traffic_recovery)
    hier_stats["waves_astar"] = int(hier_stats.get("waves_astar") or 0) + 1

    if use_swapnet:
        try:
            import simulation.engine_swapnet  # noqa: F401
            print("[M0] engine_swapnet patched time_forward", flush=True)
        except Exception as exc:  # noqa: BLE001
            hier_stats["swapnet_import_error"] = str(exc)
            print(f"[M0] WARN engine_swapnet import failed: {exc}", flush=True)

    mod, env, agv_states, task_states, n_tasks = load_scenario(
        task_csv, Path(meta["position_csv"]), force_reload=True
    )
    if meta.get("extra_obstacles"):
        patch_extra_obstacles(env, list(meta["extra_obstacles"]))

    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )
    patch_moving_obstacle_horizon(sim)
    # Field/runtime default: NO execution conflict shield.
    # Shield only patches the current tick and was a major source of teleports;
    # path-fail recovery stays via evacuate/hold + RulePark (nearest non-interfering parking).
    # Opt-in: meta["conflict_shield"]=True or AGV_CONFLICT_SHIELD=1.
    _shield_meta = meta.get("conflict_shield", None)
    if _shield_meta is None:
        _shield_on = str(_os.environ.get("AGV_CONFLICT_SHIELD", "0")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    else:
        _shield_on = bool(_shield_meta)
    if _shield_on:
        patch_conflict_free_execution(sim)
        print("[M0] conflict shield ON (opt-in)", flush=True)
    else:
        print(
            "[M0] conflict shield OFF (field default); "
            "assign-fail → evacuate/hold/RulePark",
            flush=True,
        )
    hier_stats["conflict_shield"] = bool(_shield_on)

    swap_stats = {"enabled": bool(use_swapnet), "station_swap": 0}
    if use_swapnet:
        try:
            from ml_research.benchmarks.m5_replan import ensure_m5_hooks
            from ml_research.benchmarks.station_eta import ensure_station_eta_hooks
            from ml_research.benchmarks.swap_net.hooks import attach_swap_net

            ensure_m5_hooks(sim, horizon=40, enable_replan=True, aggressive=False)
            ensure_station_eta_hooks(sim, margin=float(swapnet_margin))
            ckpt = CKPT / "swap_net_v3.pt"
            if not ckpt.exists():
                ckpt = CKPT / "swap_net.pt"
            attach_swap_net(
                sim,
                ckpt=ckpt if ckpt.exists() else None,
                margin=float(swapnet_margin),
            )
            sim._swapnet_enabled = True
            sim._swapnet_period = int(meta.get("swapnet_period") or 2)
            if not hasattr(sim, "_replan_stats") or not isinstance(
                sim._replan_stats, dict
            ):
                sim._replan_stats = {}
            sim._replan_stats.setdefault("station_swap", 0)
            hier_stats["swapnet"] = True
            hier_stats["swapnet_ckpt"] = str(ckpt) if ckpt.exists() else "untrained"
            hier_stats["swapnet_period"] = int(sim._swapnet_period)
        except Exception as exc:  # noqa: BLE001
            hier_stats["swapnet_error"] = str(exc)
            swap_stats["enabled"] = False
            print(f"[M0] WARN attach_swap_net failed: {exc}", flush=True)

    traffic_stats = {"enabled": bool(traffic_recovery), "triggers": 0, "parks": 0}
    if traffic_recovery:
        try:
            from ml_research.benchmarks.lane_traffic_ai.recovery import (
                attach_traffic_recovery,
            )

            ctrl = attach_traffic_recovery(sim, meta=meta)
            pol = ctrl.policy
            hier_stats["traffic_net"] = False
            hier_stats["traffic_rule"] = True
            hier_stats["traffic_teacher"] = True
            traffic_stats["rule"] = True
            traffic_stats["teacher"] = True
            mid = getattr(pol, "map_id", "") or meta.get("id")
            print(
                f"[M0] rule recovery attached map={mid} "
                f"(freq artery/parking + rule semaphore)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            hier_stats["traffic_error"] = str(exc)
            traffic_stats["enabled"] = False
            print(f"[M0] WARN rule recovery attach failed: {exc}", flush=True)

    sid = str(meta["id"])
    traj = TRAJ / f"{sid}_m0_engine{'_swap' if use_swapnet else ''}.csv"
    traj.parent.mkdir(parents=True, exist_ok=True)
    max_time = int(meta.get("max_sim_time") or 500000)
    # Prefer explicit meta wall; <=0 means unlimited; >>1e8 kept as-is (no 30min clamp).
    _w = meta.get("wall_timeout")
    raw_wall = 1800.0 if _w is None else float(_w)
    wall_timeout = None if raw_wall <= 0 else raw_wall
    prog_every = int(meta.get("progress_every") or 100)
    gate_every = int(meta.get("gate_every") or prog_every)
    allow_yield = bool(allow_yield) and gate is not None
    print(
        f"[M0] start max_time={max_time} wall_timeout={wall_timeout} "
        f"progress_every={prog_every} swapnet={bool(use_swapnet)} "
        f"traffic_recovery={bool(traffic_recovery)} "
        f"allow_yield={allow_yield} gate_every={gate_every}",
        flush=True,
    )

    escalate_hit = {"label": "", "t": -1, "reason": ""}
    # GENERALIZABLE_SWITCH_V1
    import time as _time
    stall_state = {
        "idle_surface_streak": 0,
        "done_plateau_streak": 0,
        "prev_done": None,
        "prev_t": None,
        "throughput_ema": None,
        "throughput_drop_streak": 0,
        "done_at_window_start": None,
        "t_at_window_start": None,
        "wall_at_gate": None,
        "wall_tick_streak": 0,
        "win_wall0": None,
        "win_t0": None,
        "win_done0": None,
        "wall_window_hit": False,
        "wall_window_dsim": 0,
        "wall_window_ddone": 0,
        "last_recover_plateau": -1,
    }
    idle_need = int(meta.get("m0_idle_surface_streak") or 2)
    plateau_need = int(meta.get("m0_done_plateau_streak") or 3)
    assign_fail_need = int(meta.get("m0_assign_fail_handoff") or 8)
    plan_fail_rate_need = float(meta.get("m0_plan_fail_rate") or 0.4)
    plan_fail_samples_need = int(meta.get("m0_plan_fail_samples") or 10)
    throughput_ratio_need = float(meta.get("m0_throughput_ratio") or 0.4)
    throughput_drop_need = int(meta.get("m0_throughput_drop_streak") or 2)
    min_handoff_t = int(meta.get("m0_min_handoff_t") or 80)
    min_handoff_done = int(meta.get("m0_min_handoff_done") or 0)
    min_handoff_wall_s = float(meta.get("m0_min_handoff_wall_s") or 45.0)
    wall_tick_s = float(meta.get("m0_wall_tick_s") or 1.0)
    wall_tick_streak_need = int(meta.get("m0_wall_tick_streak") or 2)
    wall_starve_s = float(meta.get("m0_wall_starve_s") or 90.0)
    wall_starve_done = int(meta.get("m0_wall_starve_done") or 15)
    wall_starve_min_t = int(meta.get("m0_wall_starve_min_t") or 40)
    wall_window_s = float(meta.get("m0_wall_window_s") or 45.0)
    wall_window_min_sim = int(meta.get("m0_wall_window_min_sim") or 20)
    wall_window_min_done = int(meta.get("m0_wall_window_min_done") or 2)
    wall_done_cost_s = float(meta.get("m0_wall_done_cost_s") or 8.0)
    wall_done_cost_min_done = int(meta.get("m0_wall_done_cost_min_done") or 6)
    m0_loop_t0 = _time.perf_counter()

    def _should_stop_gate(sim_live) -> bool:
        if not allow_yield:
            return False
        t = int(getattr(sim_live, "time", 0) or 0)
        if t <= 0 or (gate_every > 0 and t % max(1, gate_every) != 0):
            return False

        wall_elapsed = float(_time.perf_counter() - m0_loop_t0)
        sig = _m0_stall_signals(sim_live, n_tasks=int(n_tasks))
        if sig["idle_all"] and sig["has_work"]:
            stall_state["idle_surface_streak"] = (
                int(stall_state["idle_surface_streak"]) + 1
            )
        else:
            stall_state["idle_surface_streak"] = 0
        # Plateau on true completions (stderr done=), not done_est (busy-adjusted).
        prev_done = stall_state["prev_done"]
        done_est = int(sig["done_est"])
        done_plateau_metric = int(sig.get("done_completed") or done_est)
        if (
            prev_done is not None
            and done_plateau_metric <= int(prev_done)
            and sig["has_work"]
        ):
            stall_state["done_plateau_streak"] = (
                int(stall_state["done_plateau_streak"]) + 1
            )
        else:
            stall_state["done_plateau_streak"] = 0

        prev_t = stall_state["prev_t"]
        if prev_done is not None and prev_t is not None and t > int(prev_t):
            dt = max(1, t - int(prev_t))
            d_done = max(0, done_plateau_metric - int(prev_done))
            rate = float(d_done) / float(dt)
            ema = stall_state["throughput_ema"]
            if ema is None:
                stall_state["throughput_ema"] = rate
            else:
                stall_state["throughput_ema"] = 0.8 * float(ema) + 0.2 * rate
                base = float(stall_state["throughput_ema"])
                if (
                    sig["has_work"]
                    and base > 1e-6
                    and rate <= throughput_ratio_need * base
                ):
                    stall_state["throughput_drop_streak"] = (
                        int(stall_state["throughput_drop_streak"]) + 1
                    )
                else:
                    stall_state["throughput_drop_streak"] = 0
            prev_wall = stall_state["wall_at_gate"]
            if prev_wall is not None:
                wpt = float(wall_elapsed - float(prev_wall)) / float(dt)
                if wpt >= float(wall_tick_s) and sig["has_work"]:
                    stall_state["wall_tick_streak"] = (
                        int(stall_state["wall_tick_streak"]) + 1
                    )
                else:
                    stall_state["wall_tick_streak"] = 0
        stall_state["prev_done"] = done_plateau_metric
        stall_state["prev_t"] = t
        stall_state["wall_at_gate"] = wall_elapsed

        stall_state["wall_window_hit"] = False
        if stall_state["win_wall0"] is None:
            stall_state["win_wall0"] = wall_elapsed
            stall_state["win_t0"] = t
            stall_state["win_done0"] = done_plateau_metric
        else:
            d_wall = float(wall_elapsed) - float(stall_state["win_wall0"])
            if d_wall >= float(wall_window_s):
                d_sim = int(t) - int(stall_state["win_t0"] or 0)
                d_done = int(done_plateau_metric) - int(stall_state["win_done0"] or 0)
                stall_state["wall_window_dsim"] = int(d_sim)
                stall_state["wall_window_ddone"] = int(d_done)
                if d_sim < int(wall_window_min_sim) or d_done < int(
                    wall_window_min_done
                ):
                    stall_state["wall_window_hit"] = True
                stall_state["win_wall0"] = wall_elapsed
                stall_state["win_t0"] = t
                stall_state["win_done0"] = done_plateau_metric

        hotspot_discount = 1.0
        try:
            drop_s = float(getattr(gate, "last", None) and gate.last.dropoff_score or 0.0)
            thr = float(getattr(gate, "dropoff_escalate_score", 0.45) or 0.45)
            if drop_s >= thr:
                hotspot_discount = 0.75
        except Exception:  # noqa: BLE001
            hotspot_discount = 1.0

        def _need(v: float, *, floor: int = 1) -> int:
            return max(int(floor), int(round(float(v) * hotspot_discount)))

        force_reason = ""
        cold = (t < int(min_handoff_t) and wall_elapsed < float(min_handoff_wall_s)) or (
            done_plateau_metric < int(min_handoff_done)
        )
        remaining = max(0, int(n_tasks) - int(done_plateau_metric))
        near_done = remaining <= max(2, int(round(0.01 * float(n_tasks))))
        # Do not hand off on wall clock. A healthy M0 (completions still
        # arriving, sim/task still low) stays. A pinched map already started
        # on ECBS via narrow_cut. Here we only leave when work is stuck:
        # plateau, idle surface, assign/plan failure, throughput drop.
        # Wall-clock alone must NOT eject a healthy M0 run: SH02 M0 was ~14
        # sim/task (would finish <8000) but wall_done_cost handed off to ECBS
        # which blew sim to 20k+. Only escalate on wall when sim is also stalled
        # AND sim/task is already poor.
        sim_stalled = (
            int(stall_state["done_plateau_streak"]) >= 3
            or (
                int(stall_state["idle_surface_streak"]) >= 2
                and bool(sig.get("idle_all"))
            )
        )
        sim_per_task = float(t) / max(1, int(done_plateau_metric))
        plateau_now = int(stall_state["done_plateau_streak"])
        idle_now = int(stall_state["idle_surface_streak"])
        # Protect healthy M0: good cumulative sim/task AND recent progress.
        # SH02 stuck at done=140 for 400+ ticks still had sim/task≈20 — must
        # drop efficient once plateau is long (busy A* thrash ≠ idle_all).
        sim_efficient = (
            int(done_plateau_metric) >= 5
            and sim_per_task <= 35.0
            and plateau_now < 12
        )
        # Prefer M0-internal unstick over ECBS (ECBS blows sim to 20k+).
        # plateau 8..19: scatter once, wait escape finish, then assign; stay M0.
        if (
            sig["has_work"]
            and 8 <= plateau_now < 20
            and int(done_plateau_metric) >= 5
            and stall_state.get("last_recover_plateau") != plateau_now
        ):
            stall_state["last_recover_plateau"] = plateau_now
            try:
                cur_b = float(getattr(sim_live, "astar_wall_budget_s", 0.2) or 0.2)
                sim_live.astar_wall_budget_s = max(cur_b, 0.35)
                if hasattr(sim_live, "tried_tasks"):
                    sim_live.tried_tasks.clear()
                if hasattr(sim_live, "_tried_until"):
                    sim_live._tried_until = {}
                if hasattr(sim_live, "assign_cooldown") and isinstance(
                    sim_live.assign_cooldown, dict
                ):
                    sim_live.assign_cooldown.clear()
                sim_live._assign_fail_streak = 0
                prefixes = ("escape_", "fsm_", "yield_", "stage_", "park_", "recover_")
                t_now = int(getattr(sim_live, "time", 0) or 0)
                cleared = 0
                n_escaping = 0
                # Only clear finished escape_ (path done). Mid-flight clear+rescatter
                # on every gate made r11 assign_ok=0 forever.
                for agv in list(getattr(sim_live, "agvs", None) or []):
                    tid = getattr(agv, "task_id", None)
                    if tid is None or not str(tid).startswith(prefixes):
                        continue
                    path = getattr(agv, "path", None) or []
                    path_end = len(path) - 1 if path else -1
                    if path_end < 0 or t_now >= path_end:
                        agv.task_id = None
                        cleared += 1
                    else:
                        n_escaping += 1
                n_idle = sum(
                    1
                    for a in (getattr(sim_live, "agvs", None) or [])
                    if getattr(a, "task_id", None) is None
                )
                n_fleet = len(getattr(sim_live, "agvs", []) or [])
                scattered = 0
                moved = False
                # Scatter once at plateau==8; rare re-scatter if still fully idle
                # and no escapes in flight (every 4 gates).
                do_scatter = (plateau_now == 8) or (
                    plateau_now >= 12
                    and plateau_now % 4 == 0
                    and n_escaping == 0
                    and n_idle >= max(3, n_fleet // 2)
                )
                if do_scatter and hasattr(sim_live, "scatter_idle_agvs"):
                    scattered = int(
                        sim_live.scatter_idle_agvs(min_r=5, max_r=16) or 0
                    )
                if hasattr(sim_live, "evacuate_hub_idles_far"):
                    moved = bool(sim_live.evacuate_hub_idles_far(min_r=4, max_r=12))
                if hasattr(sim_live, "heal_orphaned_surface_tasks"):
                    sim_live.heal_orphaned_surface_tasks()
                if hasattr(sim_live, "_continue_or_release_stale_path_end_tasks"):
                    sim_live._continue_or_release_stale_path_end_tasks()
                n_ok = 0
                # Assign only when some AGVs are truly free (not mid-escape).
                n_free = sum(
                    1
                    for a in (getattr(sim_live, "agvs", None) or [])
                    if getattr(a, "task_id", None) is None
                )
                if n_free > 0 and hasattr(sim_live, "_try_assign_tasks"):
                    n_ok = int(
                        sim_live._try_assign_tasks(fail_budget=3, max_evacuate=2) or 0
                    )
                print(
                    f"[M0] plateau-recover t={t} plateau={plateau_now} "
                    f"done={done_plateau_metric} idle={n_idle} escaping={n_escaping} "
                    f"cleared_escape={cleared} scattered={scattered} "
                    f"moved={moved} assign_ok={n_ok} free={n_free} "
                    f"astar_wall>=0.35",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[M0] WARN plateau-recover failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        # hard_livelock: only after full recover window (plateau≥20).
        # Do NOT fire on idle_all@plateau≥8 — that aborted recover at 12 on SH02.
        hard_livelock = (
            (not sim_efficient)
            and sig["has_work"]
            and int(sig.get("n_surface") or 0) >= 3
            and t >= 80
            and plateau_now >= 20
        )
        if near_done:
            force_reason = ""
        elif sim_efficient:
            # Healthy M0 sim/task with recent progress — do not eject.
            force_reason = ""
        elif 8 <= plateau_now < 20:
            # Recover window — never handoff; give clear+assign time to work.
            force_reason = ""
        elif hard_livelock:
            force_reason = (
                f"hard_livelock plateau={plateau_now} idle={idle_now} "
                f"idle_all={bool(sig.get('idle_all'))} "
                f"surface={sig.get('n_surface')} sim/task={sim_per_task:.1f}"
            )
        elif (
            bool(stall_state.get("wall_window_hit"))
            and sig["has_work"]
            and t >= 10
            and sim_stalled
            and not sim_efficient
        ):
            force_reason = (
                f"wall_window<{wall_window_s:.0f}s "
                f"dsim={stall_state['wall_window_dsim']}<{wall_window_min_sim} "
                f"or ddone={stall_state['wall_window_ddone']}<{wall_window_min_done}"
            )
        elif (
            int(stall_state["wall_tick_streak"]) >= int(wall_tick_streak_need)
            and sig["has_work"]
            and t >= 10
            and sim_stalled
            and not sim_efficient
        ):
            force_reason = (
                f"wall_tick_streak={stall_state['wall_tick_streak']}"
                f"@{wall_tick_s:.2f}s/tick"
            )
        elif (
            wall_elapsed >= float(wall_starve_s)
            and done_est < int(wall_starve_done)
            and t >= int(wall_starve_min_t)
            and sig["has_work"]
            and remaining > int(wall_starve_done)
            and sim_stalled
            and not sim_efficient
        ):
            force_reason = (
                f"wall_starve wall={wall_elapsed:.1f}s done={done_est}"
                f"<{wall_starve_done}"
            )
        elif (
            done_est >= int(wall_done_cost_min_done)
            and sig["has_work"]
            and remaining > int(wall_done_cost_min_done)
            and (float(wall_elapsed) / max(1, int(done_est))) >= float(wall_done_cost_s)
            and sim_stalled
            and not sim_efficient
        ):
            cost = float(wall_elapsed) / max(1, int(done_est))
            force_reason = (
                f"wall_done_cost={cost:.1f}s/task>={wall_done_cost_s:.1f}"
            )
        elif (
            int(stall_state["idle_surface_streak"]) >= _need(idle_need, floor=2)
            and sig["has_work"]
            and not cold
            and bool(sig.get("idle_all"))
            and not sim_efficient
        ):
            force_reason = (
                f"idle_surface_streak={stall_state['idle_surface_streak']}"
            )
        elif (
            int(stall_state["done_plateau_streak"]) >= max(6, _need(plateau_need, floor=4))
            and sig["has_work"]
            and not cold
            and not sim_efficient
        ):
            force_reason = (
                f"done_plateau_streak={stall_state['done_plateau_streak']}"
            )
        elif (
            int(sig["assign_fail_streak"]) >= max(80, int(assign_fail_need))
            and sig["has_work"]
            and not cold
            and not sim_efficient
        ):
            force_reason = f"assign_fail_streak={sig['assign_fail_streak']}"
        elif (
            float(sig.get("plan_fail_rate") or 0) >= plan_fail_rate_need
            and int(sig.get("plan_fail_samples") or 0) >= plan_fail_samples_need
            and sig["has_work"]
            and not cold
            and not sim_efficient
        ):
            force_reason = (
                f"plan_fail_rate={float(sig['plan_fail_rate']):.2f}"
                f"@{sig['plan_fail_samples']}"
            )
        elif (
            int(stall_state["throughput_drop_streak"])
            >= max(4, _need(throughput_drop_need, floor=2))
            and sig["has_work"]
            and not cold
            and not sim_efficient
        ):
            force_reason = (
                f"throughput_drop_streak={stall_state['throughput_drop_streak']}"
            )

        if force_reason:
            try:
                label = gate.force_commit_at_least(
                    "medium", reason=f"m0_degrade:{force_reason}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[M0] WARN force_commit failed: {exc}", flush=True)
                label = "medium"
            escalate_hit["label"] = str(label or "medium")
            escalate_hit["t"] = t
            escalate_hit["reason"] = force_reason
            print(
                f"[HIER] M0 degrade escalate -> {escalate_hit['label']} at t={t} "
                f"wall={wall_elapsed:.1f}s reason={force_reason} "
                f"hotspot_discount={hotspot_discount}",
                flush=True,
            )
            return True

        pose_live = _m0_pose_from_sim(sim_live)
        queues_live = _m0_queues_from_sim(sim_live)
        assigned_live = _m0_assigned_from_sim(sim_live)
        try:
            gate.set_scene(
                static=set(getattr(gate, "scene_static", None) or set()),
                stations=set(getattr(gate, "scene_stations", None) or set()),
                pose=pose_live,
            )
            st = gate.decide(
                assigned=assigned_live,
                queues=queues_live,
                recent_joint_fail=0,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[M0] WARN gate.decide failed: {exc}", flush=True)
            return False
        label = str(
            getattr(gate, "last_scene_label", "")
            or getattr(st, "scene_label", "")
            or ""
        )
        # SceneNet/map band alone: never eject during recover window or when
        # sim_efficient. Band-only handoff killed healthy/recovering M0.
        if (
            label in ("medium", "hard")
            and not sim_efficient
            and plateau_now >= 20
        ):
            escalate_hit["label"] = label
            escalate_hit["t"] = t
            escalate_hit["reason"] = str(getattr(st, "reason", "") or "gate_decide")
            print(
                f"[HIER] M0 gate escalate -> {label} at t={t} "
                f"planner={getattr(st, 'planner', '')} "
                f"reason={getattr(st, 'reason', '')}",
                flush=True,
            )
            return True
        return False

    steps, forced = run_sim_loop(
        sim,
        max_time=max_time,
        label=f"{sid}/m0_engine",
        progress_every=max(1, prog_every),
        checkpoint_path=traj,
        checkpoint_every=max(25, min(100, prog_every)),
        wall_timeout=wall_timeout,
        should_stop=_should_stop_gate if allow_yield else None,
    )
    steps = _legalize_engine_steps(steps)
    write_trajectory(traj, steps)
    left = sum(len(v) for v in sim.task_states.values())
    # Crude done for non-handoff; handoff recomputes via snapshot.
    n_loaded = sum(
        1
        for a in getattr(sim, "agvs", []) or []
        if a.task_id
        and not str(a.task_id).startswith("escape_")
        and (
            str((_m0_step_at(a, int(sim.time)) or {}).get("loaded", "")).lower()
            in ("true", "1", "yes")
        )
    )
    done = max(0, int(n_tasks) - int(left) - int(n_loaded))
    rs = getattr(sim, "_replan_stats", {}) or {}
    swap_stats["station_swap"] = int(rs.get("station_swap") or 0)
    ctrl = getattr(sim, "_traffic_recovery", None)
    if ctrl is not None and getattr(ctrl, "stats", None):
        traffic_stats.update({k: int(v) for k, v in ctrl.stats.items()})
        hier_stats["traffic_stats"] = dict(ctrl.stats)

    handoff_label = str(escalate_hit.get("label") or "")
    do_handoff = bool(
        allow_yield
        and handoff_label in ("medium", "hard")
        and (left > 0 or n_loaded > 0 or any(
            a.task_id and not str(a.task_id).startswith("escape_")
            for a in (getattr(sim, "agvs", []) or [])
        ) or sum(len(v) for v in (getattr(sim, "task_states", None) or {}).values()) > 0
        or len(getattr(sim, "surface_tasks", None) or {}) > 0)
    )
    if do_handoff:
        # Clear pad-blocking idles before freezing the handoff snapshot.
        if hasattr(sim, "evacuate_hub_idles_far"):
            try:
                moved = bool(sim.evacuate_hub_idles_far(min_r=4, max_r=8))
                print(
                    f"[HIER] M0 handoff pre-evacuate moved={moved}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[M0] WARN handoff evacuate failed: {exc}", flush=True)
        remain_csv = OUT / f"{sid}_m0_handoff_remain.csv"
        snap = _build_m0_handoff_snapshot(
            sim=sim,
            steps=steps,
            n_tasks=int(n_tasks),
            traj=traj,
            remain_csv=remain_csv,
            handoff_label=handoff_label,
        )
        hier_stats["m0_handoff"] = True
        hier_stats["m0_handoff_label"] = handoff_label
        hier_stats["m0_handoff_t"] = int(snap["now"])
        hier_stats["m0_handoff_reason"] = str(escalate_hit.get("reason") or "")
        hier_stats["last_planner"] = "m0_handoff"
        hier_stats["last_reason"] = f"m0_yield_{handoff_label}"
        if gate is not None and hasattr(gate, "stats_dict"):
            try:
                hier_stats.update(gate.stats_dict())
            except Exception:  # noqa: BLE001
                pass
        rep = {
            "scenario_id": sid,
            "method": "m0_engine_handoff",
            "handoff": True,
            "handoff_label": handoff_label,
            "handoff_snapshot": snap,
            "plan_horizon": int(plan_horizon),
            "exec_horizon": int(exec_horizon),
            "weight": weight,
            "max_active": max_active,
            "plan": plan,
            "pipeline": bool(pipeline),
            "turn_aware": bool(turn_aware),
            "tasks_total": int(n_tasks),
            "tasks_completed": int(snap["done"]),
            "tasks_failed": [],
            "completion_ratio": round(int(snap["done"]) / max(1, n_tasks), 4),
            "sim_time": int(snap["now"]),
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "trajectory": str(traj),
            "validate_ok": False,
            "validate_summary": "handoff_pending_wave",
            "n_collisions": 0,
            "n_swaps": 0,
            "n_hard_wall": 0,
            "n_illegal_motion": 0,
            "n_fifo": 0,
            "n_pickup_cell": 0,
            "use_swapnet": bool(swap_stats["enabled"]),
            "swapnet": dict(swap_stats),
            "forced_stop": True,
            "hierarchical": hier_stats,
            "use_hierarchical": bool(use_hierarchical),
            "use_wavenet": bool(use_wavenet),
        }
        print(
            f"[HIER] M0 handoff label={handoff_label} t={snap['now']} "
            f"done={snap['done']}/{n_tasks} remain_q={snap['tasks_remaining']} "
            f"carriers={len(snap['pending_delivery'])} "
            f"approach={len(snap['pending_pickup'])} "
            f"reason={escalate_hit.get('reason') or ''}",
            flush=True,
        )
        return rep

    val = validate_hybrid_trajectory(meta, traj)
    iss = val.get("issues") or {}
    conf = analyze_trajectory_conflicts(steps)
    rep = {
        "scenario_id": sid,
        "method": "m0_engine_swapnet" if swap_stats["enabled"] else "m0_engine",
        "handoff": False,
        "plan_horizon": int(plan_horizon),
        "exec_horizon": int(exec_horizon),
        "weight": weight,
        "max_active": max_active,
        "plan": plan,
        "pipeline": bool(pipeline),
        "turn_aware": bool(turn_aware),
        "tasks_total": int(n_tasks),
        "tasks_completed": done,
        "tasks_failed": [],
        "completion_ratio": round(done / max(1, n_tasks), 4),
        "sim_time": int(sim.time),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "trajectory": str(traj),
        "validate_ok": bool(val.get("ok")),
        "validate_summary": format_validation_summary(val),
        "n_collisions": int(iss.get("n_collisions") or conf.get("collisions") or 0),
        "n_swaps": int(iss.get("n_swaps") or conf.get("swaps") or 0),
        "n_hard_wall": int(iss.get("n_hard_wall") or 0),
        "n_illegal_motion": int(iss.get("n_illegal_motion") or 0),
        "n_fifo": int(iss.get("n_fifo_violations") or 0),
        "n_pickup_cell": int(iss.get("n_pickup_cell_violations") or 0),
        "use_swapnet": bool(swap_stats["enabled"]),
        "swapnet": dict(swap_stats),
        "traffic_recovery": dict(traffic_stats),
        "forced_stop": bool(forced and left > 0),
        "hierarchical": hier_stats,
        "use_hierarchical": bool(use_hierarchical),
        "use_wavenet": bool(use_wavenet),
    }
    print(
        f"[HIER] M0-engine done={done}/{n_tasks} sim={rep['sim_time']} "
        f"wall={rep['wall_seconds']}s validate={rep['validate_summary']} "
        f"swap={swap_stats.get('station_swap', 0)} "
        f"traffic={traffic_stats}",
        flush=True,
    )
    return rep


def _wh_to_rc(x: int, y: int) -> Cell:
    return (y - 1, x - 1)


def _rc_to_wh(r: int, c: int) -> Cell:
    return (c + 1, r + 1)


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _delta_of_pitch(pitch: int) -> Tuple[int, int]:
    p = int(pitch) % 360
    if p == 0:
        return 1, 0
    if p == 90:
        return 0, 1
    if p == 180:
        return -1, 0
    if p == 270:
        return 0, -1
    return 0, 0


def _pitch_from_delta(dx: int, dy: int, prev: int) -> int:
    if dx == 1:
        return 0
    if dx == -1:
        return 180
    if dy == 1:
        return 90
    if dy == -1:
        return 270
    return int(prev) % 360


def _turn_pitch_chain(from_pitch: int, to_pitch: int) -> List[int]:
    """Pitches to apply in-place (≤90°/tick), ending at ``to_pitch``.

    A single 180° flip is illegal (``turn_gt_90``); emit mid then target.
    """
    a = int(from_pitch) % 360
    b = int(to_pitch) % 360
    if a == b:
        return []
    diff = (b - a + 360) % 360
    if diff == 180:
        mid = (a + 90) % 360
        return [mid, b]
    # 90° either way — one step.
    return [b]


def _turn_aware_st_astar(
    start: Pose,
    goal: Cell,
    static: Set[Cell],
    reserved_v: Dict[Tuple[int, Cell], str],
    reserved_e: Set[Tuple[int, Cell, Cell]],
    who: str,
    tmax: int = 120,
    max_expansions: int = 40000,
) -> Optional[List[Pose]]:
    """Earliest arrival in (x, y, pitch, t). Actions: wait, ±90° turn, forward.

    A 180° flip in one tick is illegal, so only quarter-turns are generated.
    Other agents are frozen obstacles via ``reserved_v`` / ``reserved_e``.
    """
    sx, sy = int(start[0]), int(start[1])
    sp = int(start[2]) % 360
    if (sx, sy) == goal:
        return []

    def cell_free(t: int, cell: Cell) -> bool:
        if cell in static and cell != goal and cell != (sx, sy):
            return False
        if not (1 <= cell[0] <= 20 and 1 <= cell[1] <= 20):
            return False
        other = reserved_v.get((t, cell))
        return other is None or other == who

    def edge_free(t: int, c0: Cell, c1: Cell) -> bool:
        if c0 == c1:
            return True
        return (t, c1, c0) not in reserved_e

    start_st = (sx, sy, sp, 0)
    open_h: List[Tuple[int, int, Tuple[int, int, int, int]]] = []
    heapq.heappush(open_h, (_manh((sx, sy), goal), 0, start_st))
    parent: Dict[Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]] = {
        start_st: None
    }
    g_best = {start_st: 0}
    closed: Set[Tuple[int, int, int, int]] = set()
    found = None
    exp = 0
    while open_h and exp < int(max_expansions):
        _f, _, st = heapq.heappop(open_h)
        if st in closed:
            continue
        closed.add(st)
        exp += 1
        x, y, pitch, t = st
        if (x, y) == goal:
            found = st
            break
        if t >= int(tmax):
            continue
        t1 = t + 1
        nxt_states: List[Tuple[int, int, int]] = []
        if cell_free(t1, (x, y)):
            nxt_states.append((x, y, pitch))
            for tp in (0, 90, 180, 270):
                if tp == pitch:
                    continue
                diff = (tp - pitch) % 360
                if diff == 90 or diff == 270:
                    nxt_states.append((x, y, tp))
        dx, dy = _delta_of_pitch(pitch)
        nx, ny = x + dx, y + dy
        if cell_free(t1, (nx, ny)) and edge_free(t1, (x, y), (nx, ny)):
            nxt_states.append((nx, ny, pitch))
        for nx, ny, npitch in nxt_states:
            nst = (nx, ny, npitch, t1)
            if nst in closed:
                continue
            ng = t1
            if ng >= g_best.get(nst, 10**9):
                continue
            g_best[nst] = ng
            parent[nst] = st
            heapq.heappush(
                open_h, (ng + _manh((nx, ny), goal), exp, nst)
            )
    if found is None:
        return None
    chain = []
    cur = found
    while cur is not None:
        chain.append(cur)
        cur = parent[cur]
    chain.reverse()
    return [(x, y, pitch) for x, y, pitch, _t in chain[1:]]


def _ring_via_pairs(start: Cell, goal: Cell) -> List[Tuple[Cell, Cell]]:
    """Shared bottom/top highways first, then the side rings."""
    out: List[Tuple[Cell, Cell, int]] = []
    for y_ring in (2, 19):
        w1 = (start[0], y_ring)
        w2 = (goal[0], y_ring)
        cost = abs(start[1] - y_ring) + abs(goal[1] - y_ring) + abs(start[0] - goal[0])
        out.append((w1, w2, cost))
    for x_ring in (2, 19):
        w1 = (x_ring, start[1])
        w2 = (x_ring, goal[1])
        cost = (
            abs(start[0] - x_ring)
            + abs(goal[0] - x_ring)
            + abs(start[1] - goal[1])
            + 5
        )
        out.append((w1, w2, cost))
    out.sort(key=lambda x: (x[2], x[0], x[1]))
    return [(a, b) for a, b, _ in out[:4]]


def _dropoff_of(cell: Cell) -> Optional[Cell]:
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        s = (cell[0] + dx, cell[1] + dy)
        if s in _MAP_DROPOFFS:
            return s
    return None


def _station_by_name(name: str) -> Optional[Cell]:
    for cell, nm in _DROP_NAME.items():
        if nm == name:
            return cell
    return None


def _road_pads(station: Cell, blocked: Set[Cell]) -> Set[Cell]:
    """Approach cells of a dropoff that the road can actually reach."""
    found: Set[Cell] = set()
    for anchor in _ROAD_ANCHORS:
        if not (1 <= anchor[0] <= 20 and 1 <= anchor[1] <= 20):
            continue
        if anchor in blocked:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = (station[0] + dx, station[1] + dy)
            if not (1 <= c[0] <= 20 and 1 <= c[1] <= 20) or c in blocked:
                continue
            if _bfs_len(anchor, c, blocked) < 10**6:
                found.add(c)
    return found


def _carrier_on_stand(
    task: dict,
    cell: Cell,
    free: Set[Cell],
    walls: Set[Cell],
    blocked: Set[Cell],
) -> bool:
    """True if ``cell`` is a legal unload stand for this task."""
    if cell in set(_valid_unload_pads(task, free, walls)):
        return True
    station = _station_by_name(str((task or {}).get("destination") or ""))
    if station is None:
        return False
    return cell in _road_pads(
        station, set(blocked) | set(_MAP_DROPOFFS) | set(_MAP_PICKUPS)
    )


def _all_unload_doors(blocked: Set[Cell]) -> Set[Cell]:
    """Every approach cell a dropoff can actually be unloaded on."""
    doors: Set[Cell] = set()
    for station in _MAP_DROPOFFS:
        doors |= _road_pads(station, blocked)
    return doors


def _park_off_doors(
    start: Cell,
    doors: Set[Cell],
    occupied: Set[Cell],
    free_cells: Set[Cell],
) -> Optional[Cell]:
    """Nearest parking cell that is not an unload door and not occupied."""
    q = deque([start])
    seen = {start}
    while q and len(seen) <= 160:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if nxt in seen or not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                continue
            if nxt not in free_cells:
                continue
            seen.add(nxt)
            if nxt not in doors and nxt not in occupied and nxt != start:
                return nxt
            q.append(nxt)
    return None


def _restore_sticky_cargo(
    loaded: Optional[Dict[str, bool]],
    dest: Optional[Dict[str, str]],
    tid: Optional[Dict[str, str]],
) -> None:
    """Overflow unloaders stay loaded. A later apply must not wipe the cargo."""
    if not isinstance(loaded, dict):
        return
    for agv, cargo in _STICKY.items():
        tid_s = str(cargo.get("tid") or "")
        if not tid_s:
            continue
        loaded[agv] = True
        if isinstance(dest, dict):
            dest[agv] = str(cargo.get("dest") or "")
        if isinstance(tid, dict):
            tid[agv] = tid_s


def _remember_sticky(agv: str, station: Optional[Cell]) -> None:
    cargo = _STICKY.get(agv) or {"dest": "", "tid": "", "station": station}
    if station is not None:
        cargo["station"] = station
    loaded = _CTX_LOADED
    dest = _CTX_DEST
    tid = _CTX_TID
    if isinstance(loaded, dict) and loaded.get(agv) and isinstance(tid, dict):
        tid_s = str(tid.get(agv) or "")
        if tid_s:
            cargo["tid"] = tid_s
            cargo["dest"] = str(dest.get(agv) or "") if isinstance(dest, dict) else ""
    _STICKY[agv] = cargo


def _select_joint_keepers(
    pose: Dict[str, Pose],
    goals: Dict[str, Cell],
    movers: Set[str],
    static: Set[Cell],
    *,
    allow_inject: bool = True,
) -> Tuple[Set[str], Dict[str, Optional[Cell]]]:
    """Capacity, sticky cargo, blocked-pickup skip, one later join, one-step yield.

    This is the 2026-09-19 guard that finished SH10, written against the live
    wave state instead of bytecode frames. Extra unloaders stay put and keep
    their cargo. An empty keep set is a wait, not a failed plan.
    """
    blocked = set(static) | set(_MAP_DROPOFFS) | set(_MAP_PICKUPS)
    hold: Set[str] = set()
    keep: Set[str] = set()
    used: Dict[Cell, Set[Cell]] = {}
    saved: Dict[str, Optional[Cell]] = {}
    standing: Dict[Cell, str] = {}
    for agv, p in pose.items():
        standing[(int(p[0]), int(p[1]))] = str(agv)
    delivery = any(
        isinstance(g, tuple) and _dropoff_of(g) is not None for g in goals.values()
    )
    if not delivery:
        _PICKUP_SKIP.clear()

    for agv in sorted(movers):
        start = pose[agv][:2]
        goal = goals.get(agv, start)
        station = _dropoff_of(goal)
        if agv in _STICKY:
            hold.add(agv)
            continue
        if goal == start or station is None:
            obstacles = blocked | {c for c, who in standing.items() if who != agv}
            reachable = goal == start or _bfs_len(start, goal, obstacles) < 10**6
            if not reachable:
                hold.add(agv)
                if not delivery:
                    _PICKUP_SKIP.add(agv)
                    door = next(
                        (
                            who
                            for cell, who in standing.items()
                            if abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]) == 1
                        ),
                        None,
                    )
                    print(
                        f"[PIPE] pickup-skip {agv} goal={goal} "
                        f"blocked-by={door or '?'} (requeued, car stays)",
                        flush=True,
                    )
                continue
            keep.add(agv)
            continue
        pads = _road_pads(station, blocked)
        taken = used.setdefault(station, set())
        if goal not in pads or goal in taken or len(taken) >= len(pads):
            hold.add(agv)
            _remember_sticky(agv, station)
            if not _STICKY[agv].get("_announced"):
                _STICKY[agv]["_announced"] = True
                print(
                    f"[PIPE] unload-hold {agv} stay={start} "
                    f"station={_DROP_NAME.get(station, station)} "
                    f"cap={len(pads)} tid={_STICKY[agv].get('tid') or '?'} "
                    f"(not in this joint plan)",
                    flush=True,
                )
            continue
        keep.add(agv)
        taken.add(goal)

    for cell, who in list(standing.items()):
        station = _dropoff_of(cell)
        if station is None or who in keep:
            continue
        if cell not in _road_pads(station, blocked):
            continue
        used.setdefault(station, set()).add(cell)

    if allow_inject and delivery and _STICKY:
        best: Optional[Tuple[int, Cell, str]] = None
        for agv, cargo in list(_STICKY.items()):
            station = cargo.get("station")
            if not isinstance(station, tuple) or agv not in pose:
                continue
            start = pose[agv][:2]
            pads = _road_pads(station, blocked)
            taken = used.setdefault(station, set())
            if start in pads:
                if start not in taken:
                    taken.add(start)
                continue
            if len(taken) >= len(pads):
                continue
            free = [
                p
                for p in sorted(pads)
                if p not in taken and standing.get(p) in (None, agv)
            ]
            if not free:
                continue
            pad = min(free, key=lambda p: (_bfs_len(start, p, blocked), p))
            dist = _bfs_len(start, pad, blocked)
            if dist >= 10**6:
                continue
            if best is None or (dist, agv) < (best[0], best[2]):
                best = (dist, pad, agv)
        if best is not None:
            _dist_b, pad, agv = best
            cargo = _STICKY.get(agv) or {}
            station = cargo.get("station")
            saved[agv] = goals.get(agv)
            goals[agv] = pad
            keep.add(agv)
            hold.discard(agv)
            if isinstance(station, tuple):
                used.setdefault(station, set()).add(pad)
            print(
                f"[PIPE] unload-join {agv} pad={pad} "
                f"tid={cargo.get('tid') or '?'}",
                flush=True,
            )

    if allow_inject and delivery and keep:
        occupied = set(standing)
        carrying: Set[str] = set()
        if isinstance(_CTX_LOADED, dict):
            carrying = {n for n, flag in _CTX_LOADED.items() if flag}
        for agv in list(keep):
            if agv not in pose:
                continue
            goal = goals.get(agv)
            if not isinstance(goal, tuple) or _dropoff_of(goal) is None:
                continue
            start = pose[agv][:2]
            others = {
                cell: who
                for cell, who in standing.items()
                if who not in keep and who not in _STICKY and who not in carrying
            }
            obstacles = blocked | set(others)
            opened = _bfs_len(start, goal, obstacles) < 10**6
            occupant = others.get(goal)
            if opened and occupant is None:
                continue
            ranked: List[Tuple[int, str, Cell]] = []
            for cell, who in others.items():
                if opened and who != occupant:
                    continue
                if (
                    not opened
                    and _bfs_len(start, goal, obstacles - {cell}) >= 10**6
                ):
                    continue
                ranked.append(
                    (abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]), who, cell)
                )
            if not ranked:
                continue
            ranked.sort()
            who = ranked[0][1]
            if who not in pose or who in saved:
                continue
            cell = pose[who][:2]
            opts: List[Tuple[int, Cell]] = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (cell[0] + dx, cell[1] + dy)
                if not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                    continue
                if nxt in blocked or nxt in (occupied - {cell}) or nxt == goal:
                    continue
                away = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                opts.append((away, nxt))
            if not opts:
                continue
            opts.sort(key=lambda item: (-item[0], item[1]))
            step = opts[0][1]
            saved[who] = goals.get(who)
            goals[who] = step
            keep.add(who)
            occupied.discard(cell)
            occupied.add(step)
            standing = dict(standing)
            standing.pop(cell, None)
            standing[step] = who
            print(
                f"[PIPE] door-yield {who} off={cell} step={step} "
                f"open={goal} for {agv}",
                flush=True,
            )
    return keep, saved


def _plan_turn_aware_joint(
    pose: Dict[str, Pose],
    goals: Dict[str, Cell],
    movers: Set[str],
    static: Set[Cell],
    *,
    tmax_scale: float = 2.0,
    prefer_outer_ring: bool = False,
    allow_inject: bool = True,
) -> Optional[Dict[str, List[Pose]]]:
    """Prioritized turn-aware spacetime plans. Non-movers freeze as obstacles.

    Tries short-first, long-first, and the y=2 / y=19 highways. Keeps the
    plan with the smallest makespan. This is the 2026-09-02 joint core:
    each AGV turns and steps on its own clock, so one turn does not stop
    the fleet.
    """
    names = list(pose.keys())
    movers = set(movers)
    if not movers:
        return {n: [] for n in names}
    movers, saved = _select_joint_keepers(
        pose, goals, movers, static, allow_inject=allow_inject
    )
    if not movers:
        return {n: [] for n in names}

    def _path_agent(
        n: str,
        reserved_v: Dict[Tuple[int, Cell], str],
        reserved_e: Set[Tuple[int, Cell, Cell]],
        horizon: int,
        force_ring: Optional[Tuple[Cell, Cell]] = None,
    ) -> Optional[List[Pose]]:
        start = pose[n]
        goal = goals[n]
        direct = _turn_aware_st_astar(
            start, goal, static, reserved_v, reserved_e, n, tmax=horizon
        )
        if direct is None:
            direct = _turn_aware_st_astar(
                start,
                goal,
                static,
                reserved_v,
                reserved_e,
                n,
                tmax=horizon * 2,
            )
        best = direct
        best_score = len(direct) if direct is not None else 10**9
        if not prefer_outer_ring and force_ring is None:
            return best

        def _try_via(w1: Cell, w2: Cell) -> Optional[List[Pose]]:
            if w1 in static or w2 in static:
                return None
            if not (1 <= w1[0] <= 20 and 1 <= w1[1] <= 20):
                return None
            if not (1 <= w2[0] <= 20 and 1 <= w2[1] <= 20):
                return None
            rv = dict(reserved_v)
            re = set(reserved_e)
            segs: List[Pose] = []
            cur: Pose = start
            for gp in (w1, w2, goal):
                if cur[:2] == gp:
                    continue
                bud = max(50, int(_bfs_len(cur[:2], gp, static) * 5) + 50)
                seg = _turn_aware_st_astar(
                    cur, gp, static, rv, re, n, tmax=bud
                )
                if seg is None:
                    seg = _turn_aware_st_astar(
                        cur, gp, static, rv, re, n, tmax=bud * 2
                    )
                if seg is None:
                    return None
                t0 = len(segs)
                prev = cur[:2]
                for i, p in enumerate(seg):
                    tt = t0 + i + 1
                    c = (int(p[0]), int(p[1]))
                    rv[(tt, c)] = n
                    if c != prev:
                        re.add((tt, prev, c))
                    prev = c
                segs.extend(seg)
                cur = segs[-1]
            return segs or None

        vias = [force_ring] if force_ring is not None else _ring_via_pairs(start[:2], goal)
        for pair in vias:
            if pair is None:
                continue
            w1, w2 = pair
            segs = _try_via(w1, w2)
            if segs is None:
                continue
            score = len(segs) - (10 if prefer_outer_ring else 0)
            if score < best_score:
                best = segs
                best_score = score
        return best

    reserved_v: Dict[Tuple[int, Cell], str] = {}
    reserved_e: Set[Tuple[int, Cell, Cell]] = set()
    base_h = 40
    for n in movers:
        bl = _bfs_len(pose[n][:2], goals[n], static)
        if bl >= 10**6:
            bl = 60
        base_h = max(base_h, int(bl * float(tmax_scale)) + 30)
    horizon = min(160, max(base_h, 80))
    for n in names:
        c = pose[n][:2]
        if n in movers:
            reserved_v[(0, c)] = n
        else:
            for t in range(0, horizon * 2 + 64):
                reserved_v[(t, c)] = n
    reserved_v0 = dict(reserved_v)
    base = sorted(
        movers, key=lambda n: (_bfs_len(pose[n][:2], goals[n], static), n)
    )
    # Cars on an unload door must reserve their exit before carriers do.
    # Short-first lets a nearby delivery claim the door and the leaver waits
    # until the slice is already cut.
    _door_cells = _all_unload_doors(set(static))
    leave_order = sorted(
        movers,
        key=lambda n: (
            0 if pose[n][:2] in _door_cells else 1,
            _bfs_len(pose[n][:2], goals[n], static),
            n,
        ),
    )
    orders: List[Tuple[str, List[str], Optional[int]]] = [
        ("leave_first", leave_order, None),
        ("direct_short", list(base), None),
        ("direct_long", list(reversed(base)), None),
    ]
    if prefer_outer_ring:
        for y_ring in (2, 19):
            orders.append((f"highway_y{y_ring}_short", list(base), y_ring))
            orders.append(
                (f"highway_y{y_ring}_long", list(reversed(base)), y_ring)
            )
    best_tl: Optional[Dict[str, List[Pose]]] = None
    best_T = 10**9
    for tag, order, y_hw in orders:
        reserved_v = dict(reserved_v0)
        reserved_e = set()
        timelines: Dict[str, List[Pose]] = {n: [] for n in names}
        ok = True
        for n in order:
            force = None
            if y_hw is not None:
                s = pose[n][:2]
                g = goals[n]
                force = ((s[0], y_hw), (g[0], y_hw))
            path = _path_agent(n, reserved_v, reserved_e, horizon, force)
            if path is None and force is not None:
                path = _path_agent(n, reserved_v, reserved_e, horizon, None)
            if path is None:
                ok = False
                break
            timelines[n] = path
            prev = pose[n][:2]
            for i, p in enumerate(path):
                t = i + 1
                c = (int(p[0]), int(p[1]))
                reserved_v[(t, c)] = n
                if c != prev:
                    reserved_e.add((t, prev, c))
                prev = c
            fin_t = len(path)
            final = path[-1][:2] if path else pose[n][:2]
            hold_to = max(horizon * 2 + 64, fin_t + 8)
            for t in range(fin_t + 1, hold_to + 1):
                reserved_v[(t, final)] = n
        if not ok:
            continue
        tmax_len = max((len(timelines[n]) for n in movers), default=0)
        for n in movers:
            last = timelines[n][-1] if timelines[n] else pose[n]
            while len(timelines[n]) < tmax_len:
                timelines[n].append(last)
        prev_c = {n: pose[n][:2] for n in names}
        conflict = False
        for t in range(tmax_len):
            cur_c = {}
            for n in names:
                seq = timelines[n]
                cur_c[n] = seq[t][:2] if t < len(seq) else prev_c[n]
            occ: Dict[Cell, str] = {}
            for n, c in cur_c.items():
                if c in occ:
                    conflict = True
                    break
                occ[c] = n
            if conflict:
                break
            for n, c in cur_c.items():
                pcell = prev_c[n]
                if c == pcell:
                    continue
                for m, mc in cur_c.items():
                    if m != n and mc == pcell and prev_c[m] == c:
                        conflict = True
                        break
                if conflict:
                    break
            prev_c = cur_c
        if conflict:
            continue
        best_T = tmax_len
        best_tl = timelines
        print(f"[PIPE] leave-plan {tag} T={tmax_len}", flush=True)
        break
    if best_tl is None:
        brief = ",".join(
            f"{n}:{goals.get(n)}" for n in list(movers)[:6]
        )
        print(f"[PIPE] joint-none n={len(movers)} {brief}", flush=True)
        if saved and allow_inject:
            for name, prev in saved.items():
                if prev is None:
                    goals.pop(name, None)
                else:
                    goals[name] = prev
            print(
                f"[PIPE] stand rollback {list(saved)} (joint failed)",
                flush=True,
            )
            return _plan_turn_aware_joint(
                pose,
                goals,
                set(movers) - set(saved),
                static,
                tmax_scale=tmax_scale,
                prefer_outer_ring=prefer_outer_ring,
                allow_inject=False,
            )
        return None
    for n in names:
        if n not in movers:
            best_tl[n] = []
        last = best_tl[n][-1] if best_tl[n] else pose[n]
        while len(best_tl[n]) < best_T:
            best_tl[n].append(last)
    return best_tl


def _wave_wait_cells(n: int, free: Set[Cell], forbidden: Set[Cell]) -> List[Cell]:
    """Distinct staging cells near the door hub (6, 1)."""
    if n <= 0:
        return []
    hx, hy = 6, 1
    ring = [
        (hx, hy),
        (hx - 1, hy),
        (hx + 1, hy),
        (hx - 2, hy),
        (hx + 2, hy),
        (hx, hy + 1),
        (hx - 1, hy + 1),
        (hx + 1, hy + 1),
        (hx + 2, hy + 1),
        (hx - 2, hy + 1),
    ]
    out: List[Cell] = []
    for c in ring:
        if c in free and c not in forbidden and c not in out:
            out.append(c)
        if len(out) >= n:
            return out
    extra = sorted(
        (c for c in free if c not in forbidden and c not in out),
        key=lambda c: (_manh(c, (hx, hy)), c),
    )
    for c in extra:
        out.append(c)
        if len(out) >= n:
            break
    return out


def _turn_aware_cell_paths(
    pose: Dict[str, Pose],
    goals: Dict[str, Cell],
    movers: Set[str],
    static: Set[Cell],
    *,
    prefer_outer_ring: bool = True,
) -> Optional[Dict[str, List[Cell]]]:
    """Cell paths from the turn-aware joint plan, including in-place turn ticks."""
    movers = {n for n in movers if n in pose and n in goals}
    if not movers:
        return None
    goals = dict(goals)
    # Idles sitting on a mover goal block every joint plan. Stage them on
    # the outer ring in the same plan, which is what the old pipeline did
    # before each epoch.
    free_cells = {
        (x, y)
        for x in range(1, 21)
        for y in range(1, 21)
        if (x, y) not in static
    }
    goal_cells = {goals[n] for n in movers}
    blocked_by_idle = [
        n
        for n in pose
        if n not in movers and pose[n][:2] in goal_cells
    ]
    if blocked_by_idle:
        ring = _wave_wait_cells(
            len(blocked_by_idle),
            free_cells,
            set(goal_cells) | {pose[n][:2] for n in movers},
        )
        for n, cell in zip(blocked_by_idle, ring):
            goals[n] = cell
            movers.add(n)
    tl = _plan_turn_aware_joint(
        pose,
        goals,
        movers,
        static,
        prefer_outer_ring=prefer_outer_ring and len(movers) >= 2,
    )
    if not tl:
        return None
    out: Dict[str, List[Cell]] = {}
    for n in pose:
        cells = [pose[n][:2]]
        for p in tl.get(n) or []:
            cells.append((int(p[0]), int(p[1])))
        out[n] = cells
    return out


def _load_resume_trajectory(
    path: Path,
) -> Tuple[Dict[str, List[dict]], int, Set[str], Dict[str, dict]]:
    """Load a cut-off trajectory so the wave loop can continue.

    Completed tasks are those whose cargo falling edge is already in the file.
    The last loaded row of each AGV is returned so the caller can rebuild
    pending deliveries from the original task queues.
    """
    steps_by: Dict[str, List[dict]] = {}
    completed: Set[str] = set()
    last: Dict[str, dict] = {}
    prev_ld: Dict[str, bool] = {}
    prev_tid: Dict[str, str] = {}
    now = 0
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            name = str(row.get("name") or "")
            if not name:
                continue
            t = int(row.get("timestamp") or 0)
            now = max(now, t)
            loaded_s = str(row.get("loaded") or "").strip().upper()
            loaded = loaded_s in ("TRUE", "1", "YES")
            tid_s = str(row.get("task-id") or "").strip()
            rec = {
                "timestamp": t,
                "name": name,
                "X": int(row.get("X") or 0),
                "Y": int(row.get("Y") or 0),
                "pitch": int(row.get("pitch") or 0) % 360,
                "loaded": "TRUE" if loaded else "FALSE",
                "destination": str(row.get("destination") or ""),
                "Emergency": str(row.get("Emergency") or "FALSE"),
                "task-id": tid_s,
            }
            steps_by.setdefault(name, []).append(rec)
            if prev_ld.get(name) and prev_tid.get(name):
                if not loaded or (tid_s and tid_s != prev_tid[name]):
                    completed.add(prev_tid[name])
            prev_ld[name] = loaded
            prev_tid[name] = tid_s
            last[name] = rec
    return steps_by, now, completed, last


def _hold(
    name: str,
    pose: Pose,
    t: int,
    *,
    loaded: bool = False,
    dest: str = "",
    tid: str = "",
) -> dict:
    return {
        "timestamp": t,
        "name": name,
        "X": pose[0],
        "Y": pose[1],
        "pitch": int(pose[2]) % 360,
        "loaded": "TRUE" if loaded else "FALSE",
        "destination": dest,
        "Emergency": "FALSE",
        "task-id": tid,
    }


def _fleet_hold_tick(
    steps_by: Dict[str, List[dict]],
    pose: Dict[str, Pose],
    names: List[str],
    now: int,
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    rise_ok: Optional[Dict[str, Set[Cell]]] = None,
    pickup_leave: Optional[Set[Cell]] = None,
) -> int:
    """Advance one sim tick: every AGV stays put (competition action dwell).

    False→True is allowed only when ``rise_ok`` lists the AGV cell (pickup
    commit). ``rise_ok=None`` or ``{}`` both suppress orphan rises (r98).
    ``pickup_leave``: hard whitelist (pad∪ring); blocks hub phantoms even if
    rise_ok is wrong (r106: Jazz rising at Xiamen (12,8)).
    """
    _restore_sticky_cargo(loaded, dest, tid)
    now = int(now) + 1
    leave = _PICKUP_LEAVE if pickup_leave is None else pickup_leave
    for n in names:
        ld = bool(loaded.get(n, False))
        d_s = str(dest.get(n, "") or "") if ld else ""
        t_s = str(tid.get(n, "") or "") if ld else ""
        prev = steps_by[n][-1] if steps_by.get(n) else None
        prev_ld = bool(
            prev and str(prev.get("loaded", "")).lower() in ("true", "1", "yes")
        )
        prev_tid = str(prev.get("task-id") or "").strip() if prev else ""
        # Refuse off-pad falling edges (fail/drop wipes → premature_unload).
        if prev_ld and prev_tid and not ld:
            cell = (int(pose[n][0]), int(pose[n][1]))
            pads = _TID_DROPOFFS.get(prev_tid) or set()
            if cell not in pads:
                ld = True
                d_s = str(prev.get("destination") or "")
                t_s = prev_tid
                loaded[n] = True
                dest[n] = d_s
                tid[n] = t_s
        if ld:
            if not prev_ld:
                cell = (int(pose[n][0]), int(pose[n][1]))
                cur_pitch = int(pose[n][2]) % 360
                prev_cell = (
                    (int(prev["X"]), int(prev["Y"])) if prev else None
                )
                prev_pitch = (
                    int(prev.get("pitch") or 0) % 360 if prev else None
                )
                # 1s no-turn dwell: same cell and same pitch as the previous tick.
                stayed = (
                    prev_cell == cell and prev_pitch == cur_pitch
                )
                allowed = rise_ok.get(n) if rise_ok is not None else None
                if (
                    not stayed
                    or not allowed
                    or cell not in allowed
                    or (leave and cell not in leave)
                ):
                    ld = False
                    d_s, t_s = "", ""
        steps_by.setdefault(n, []).append(
            _hold(
                n,
                pose[n],
                now,
                loaded=ld,
                dest=d_s,
                tid=t_s,
            )
        )
    return now


def _heal_same_tid_cargo_wipe(
    steps_by: Dict[str, List[dict]],
    agv: str,
    *,
    tid_s: str,
    dest_s: str,
) -> bool:
    """Return True iff the AGV's last frame is already loaded for ``tid_s``.

    Do not rewrite FALSE wipe streaks (r99: history rewrite + invent created
    pickup_cell phantoms). Callers must requeue / pending_pickup if False.
    """
    del dest_s  # kept for call-site compatibility
    seq = steps_by.get(agv) or []
    if not seq:
        return False
    last = seq[-1]
    if str(last.get("loaded", "")).lower() not in ("true", "1", "yes"):
        return False
    tid_s = str(tid_s or "").strip()
    if not tid_s:
        return False
    return str(last.get("task-id") or "").strip() == tid_s


def _build_grid(static: Set[Cell], stations: Set[Cell], W: int = 20, H: int = 20) -> GridMap:
    # Stations are walkable (capacity reserved in planning); only maze walls block.
    del stations
    grid = []
    for y in range(1, H + 1):
        row = []
        for x in range(1, W + 1):
            blocked = (x, y) in static
            row.append(1 if blocked else 0)
        grid.append(row)
    return GridMap(grid)


def _bfs_len(start: Cell, goal: Cell, static: Set[Cell]) -> int:
    if start == goal:
        return 0
    blocked = set(static)
    parent = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                continue
            if nxt in parent:
                continue
            if nxt in blocked and nxt != goal:
                continue
            parent[nxt] = cur
            q.append(nxt)
    if goal not in parent:
        return 10**6
    n = 0
    cur = goal
    while cur is not None and cur != start:
        n += 1
        cur = parent[cur]
    return n




def _snap_free(cell: Cell, free: Set[Cell], static: Set[Cell], prefer: Optional[Cell] = None) -> Cell:
    """Map a possibly-blocked/station cell onto a nearby free cell (BFS)."""
    c = (int(cell[0]), int(cell[1]))
    if c in free:
        return c
    if not free:
        return c
    # BFS to nearest free cell (handles maze walls covering station rings)
    q = deque([c])
    seen = {c}
    best: Optional[Cell] = None
    best_key = None
    anchor = prefer or c
    while q:
        cur = q.popleft()
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if nxt in seen:
                continue
            seen.add(nxt)
            if nxt in free:
                key = (_manh(nxt, anchor), _manh(nxt, c), nxt)
                if best is None or key < best_key:
                    best = nxt
                    best_key = key
            elif nxt not in static and 1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20:
                q.append(nxt)
            if len(seen) > 400:
                break
        if best is not None and len(seen) > 80:
                break
    if best is not None:
        return best
    return min(free, key=lambda q: (_manh(q, anchor), q))


def _component_from(start: Cell, free: Set[Cell]) -> Set[Cell]:
    """4-connected free component containing ``start`` (empty if start not free)."""
    if start not in free:
        return set()
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if nxt in free and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def _snap_reachable(
    start: Cell,
    goal: Cell,
    free: Set[Cell],
    static: Set[Cell],
) -> Cell:
    """Snap ``goal`` to a free cell reachable from ``start``.

    Maze generators occasionally leave station pads as isolated free cells
    fully enclosed by walls; A*/ECBS then fail forever at the same timestamp.

    If the station snap is unreachable from ``start``, still return the station
    snap — never a far cell in the start component (r71: that made idle AGVs
    "arrive" at a fake pickup goal and rise loaded off-station).
    """
    s0 = start if start in free else _snap_free(start, free, static, prefer=goal)
    g0 = _snap_free(goal, free, static, prefer=start)
    comp = _component_from(s0, free)
    if not comp:
        return g0
    if g0 in comp:
        return g0
    # Prefer reachable cell nearest the station (may be far from pad).
    # Commit gate (pad∪4-neighbor) prevents rising-edge off-station (r71/r76).
    return min(comp, key=lambda c: (_manh(c, g0), c))


def _station_name_of(task: dict, station_pads: Optional[Dict[str, Cell]] = None) -> str:
    """Station id for a task.

    Prefer ``task_id`` prefix when it is a known pickup station — ``start_point``
    is frequently polluted to another station or a destination (r85: Tiger-55
    committed at Dragon leave-cell). Fall back to pickup_name / start_point.
    """
    tid_s = str(task.get("task_id") or "")
    prefix = tid_s.rsplit("-", 1)[0].strip() if "-" in tid_s else ""
    for cand in (
        prefix,
        str(task.get("pickup_name") or "").strip(),
        str(task.get("start_point") or "").strip(),
    ):
        if not cand:
            continue
        if station_pads is not None and cand not in station_pads:
            continue
        return cand
    return prefix or tid_s or ""


def _pickup_approach(cell: Cell) -> Cell:
    """Stand beside an edge pickup station. The station cell is a dead end."""
    x, y = int(cell[0]), int(cell[1])
    if x <= 1:
        return (x + 1, y)
    if x >= 20:
        return (x - 1, y)
    return (x, y)


def _pickup_goal(
    task: dict,
    start: Cell,
    free: Set[Cell],
    static: Set[Cell],
    *,
    station_pads: Optional[Dict[str, Cell]] = None,
) -> Cell:
    """Approach cell beside the pickup station, never the station itself.

    Validator accepts the leave ring. Driving onto the station (map edge)
    traps the AGV and blocks the only exit.
    """
    st = _station_name_of(task, station_pads=station_pads)
    if station_pads and st and st in station_pads:
        pad = station_pads[st]
        if (
            pad not in static
            and 1 <= pad[0] <= 20
            and 1 <= pad[1] <= 20
        ):
            return _pickup_approach(pad)
        return _pickup_approach(_snap_free(pad, free, static, prefer=start))

    true_v = task.get("true_pickup_point")
    if true_v is not None:
        raw = (int(true_v[0]), int(true_v[1]))
        if (
            raw not in static
            and 1 <= raw[0] <= 20
            and 1 <= raw[1] <= 20
        ):
            return _pickup_approach(raw)
        return _pickup_approach(_snap_free(raw, free, static, prefer=start))

    pick_v = task.get("pickup_point")
    if pick_v is not None:
        raw = (int(pick_v[0]), int(pick_v[1]))
        if (
            raw not in static
            and 1 <= raw[0] <= 20
            and 1 <= raw[1] <= 20
        ):
            return _pickup_approach(raw)
        s0 = start if start in free else _snap_free(start, free, static, prefer=raw)
        return _pickup_approach(_snap_reachable(s0, raw, free, static))

    raw = (int(task["pickup_point"][0]), int(task["pickup_point"][1]))
    s0 = start if start in free else _snap_free(start, free, static, prefer=raw)
    return _pickup_approach(_snap_reachable(s0, raw, free, static))


# Display number 13 = Beijing. West pad (5,4) is the trunk; do not unload there.
_HUB13_DEST = "Beijing"
_HUB13_PADS = ((7, 4), (6, 5))  # 右, 上


def _restrict_hub13_pads(task: dict, pads: List[Cell]) -> List[Cell]:
    dest = str((task or {}).get("destination") or (task or {}).get("end_point") or "")
    if dest != _HUB13_DEST:
        return pads
    kept = [p for p in pads if (int(p[0]), int(p[1])) in set(_HUB13_PADS)]
    if kept:
        return kept
    return [(int(p[0]), int(p[1])) for p in _HUB13_PADS]


def _valid_unload_pads(
    task: dict,
    free: Set[Cell],
    static: Set[Cell],
    *,
    prefer: Optional[Cell] = None,
) -> List[Cell]:
    """Unload pads: true dropoff stations (not walls). Snap only if OOB/wall."""
    raw = [tuple(e) for e in (task.get("end_points") or []) if e is not None]
    pads: List[Cell] = []
    seen: Set[Cell] = set()
    for e in raw:
        e = (int(e[0]), int(e[1]))
        if e in static:
            continue
        if 1 <= e[0] <= 20 and 1 <= e[1] <= 20:
            if e not in seen:
                pads.append(e)
                seen.add(e)
            continue
        c = e if e in free else _snap_free(e, free, static, prefer=prefer)
        if prefer is not None:
            c = _snap_reachable(prefer, c, free, static)
        if c not in seen and c not in static:
            pads.append(c)
            seen.add(c)
    if pads:
        return _restrict_hub13_pads(task, pads)
    # Last resort: snap around destination name neighbors / raw ends
    for e in raw:
        c = _snap_free(e, free, static, prefer=prefer)
        if prefer is not None:
            c = _snap_reachable(prefer, c, free, static)
        if c not in seen and c not in static:
            pads.append(c)
            seen.add(c)
    return _restrict_hub13_pads(task, pads)


def _dest_inflight_cap(
    task: dict,
    free: Set[Cell],
    static: Set[Cell],
    n_agvs: int,
) -> int:
    """Per-destination concurrency = walkable unload pads (obstacle-free).

    Falls back to 1 if geometry yields no pad (avoid hard deadlock).
    ``n_agvs`` kept for API compat; pad count is the real ceiling.
    """
    del n_agvs
    pads = _valid_unload_pads(task, free, static)
    return max(1, len(pads) if pads else 1)


def _count_dest_inflight(
    dest: str,
    *bags: Dict[str, dict],
) -> int:
    d = str(dest or "")
    if not d:
        return 0
    n = 0
    for bag in bags:
        for t in bag.values():
            if str(t.get("destination") or "") == d:
                n += 1
    return n


def _wave_claim_budget(
    *,
    queues: Dict[str, list],
    order: List[str],
    free: Set[Cell],
    static: Set[Cell],
    bags: Tuple[Dict[str, dict], ...],
    n_free_agvs: int,
    max_active: int = 0,
    queue_peek: int = 8,
) -> Tuple[int, Dict[str, Any]]:
    """Rule-based wave claim size (dest-pad capacity, not SceneNet/k).

    Rules
    -----
    R1. ``pads(dest)`` = unique walkable unload cells for that destination
        (``_valid_unload_pads``: exclude static / obstacles).
    R2. Aggregate **by destination** (never sum pads once per surface task —
        two tasks to HubA with 3 pads ⇒ room 3, not 6).
    R3. ``room(dest) = max(0, |pads(dest)| − inflight(dest))`` where inflight
        counts assigned + pending_pickup + pending_delivery to that dest.
    R4. ``claim_cap = min(|free AGVs|, Σ_dest room(dest) [, max_active])``.
        If every dest is full, claim_cap=0 (wait for unload / carriers drain).
    R5. ``joint_k`` stays separate (gate / hard_cap); this only sizes **claim**.

    Destinations are discovered from inflight bags + a short queue peek so
    pad geometry is known before claim.
    """
    dest_task: Dict[str, dict] = {}
    for bag in bags:
        for t in bag.values():
            d = str((t or {}).get("destination") or "").strip()
            if d and d not in dest_task:
                dest_task[d] = t
    for st in order:
        for t in list(queues.get(st) or [])[: max(1, int(queue_peek))]:
            d = str((t or {}).get("destination") or "").strip()
            if d and d not in dest_task:
                dest_task[d] = t

    detail: Dict[str, Any] = {}
    total_room = 0
    for d, task in dest_task.items():
        pads = _valid_unload_pads(task, free, static)
        n_pads = len(pads) if pads else 1
        inflight = _count_dest_inflight(d, *bags)
        room = max(0, int(n_pads) - int(inflight))
        detail[d] = {
            "pads": int(n_pads),
            "inflight": int(inflight),
            "room": int(room),
        }
        total_room += int(room)

    if not dest_task:
        # No dest geometry yet (malformed rows): allow a single probe claim.
        cap = min(max(0, int(n_free_agvs)), 1) if any(queues.values()) else 0
    else:
        cap = min(max(0, int(n_free_agvs)), max(0, int(total_room)))
    if int(max_active) > 0:
        cap = min(cap, int(max_active))
    return int(cap), {
        "claim_cap": int(cap),
        "total_room": int(total_room),
        "n_free_agvs": int(n_free_agvs),
        "by_dest": detail,
    }


def _spacetime_bfs(
    start: Cell,
    goal: Cell,
    static: Set[Cell],
    reserved: Dict[Tuple[int, Cell], str],
    who: str,
    *,
    tmax: int = 240,
) -> Optional[List[Cell]]:
    if start == goal:
        return [start]
    blocked = set(static)
    parent: Dict[Tuple[int, Cell], Optional[Tuple[int, Cell]]] = {(0, start): None}
    q = deque([(0, start)])
    found = None
    while q:
        t, cur = q.popleft()
        if cur == goal:
            found = (t, cur)
            break
        if t >= tmax:
            continue
        t1 = t + 1
        x, y = cur
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                continue
            if nxt in blocked and nxt != goal:
                continue
            occ = reserved.get((t1, nxt))
            if occ is not None and occ != who:
                continue
            # edge / swap conflict with higher-priority agents
            if dx != 0 or dy != 0:
                swap_who = reserved.get((t1, cur))
                if (
                    swap_who is not None
                    and swap_who != who
                    and reserved.get((t, nxt)) == swap_who
                ):
                    continue
            key = (t1, nxt)
            if key in parent:
                continue
            parent[key] = (t, cur)
            q.append(key)
    if found is None:
        return None
    rev: List[Cell] = []
    curk: Optional[Tuple[int, Cell]] = found
    while curk is not None:
        rev.append(curk[1])
        curk = parent[curk]
    rev.reverse()
    return rev


def _prioritized_st_paths(
    starts: Dict[str, Cell],
    goals: Dict[str, Cell],
    movers: Set[str],
    static: Set[Cell],
    *,
    hold_after: int = 0,
    seed_reserved: Optional[Dict[Tuple[int, Cell], str]] = None,
) -> Optional[Dict[str, List[Cell]]]:
    """M0-style baseline: priority spacetime A* — all movers get paths in parallel time.

    Higher-priority agents reserved first; lower-priority avoid their spacetime cells.
    ``seed_reserved`` lets a higher-priority ECBS core own spacetime first (hybrid).
    Non-movers are seeded as stationary reservations so plans cannot ghost-through
    idle AGVs (avoids false ``_cell_paths_conflict`` → serial collapse).
    Returns None only if some mover cannot be routed.

    Default ``hold_after=0`` (pickup shared-pad). Unload dwell is ``_fleet_hold_tick``;
    pass hold_after=1 only when a short pad reservation is required.
    """
    reserved: Dict[Tuple[int, Cell], str] = (
        dict(seed_reserved) if seed_reserved else {}
    )
    out: Dict[str, List[Cell]] = {n: [starts[n]] for n in starts}
    movers = set(movers)
    rough = 80
    for n in movers:
        if n in starts and n in goals:
            rough = max(rough, _manh(starts[n], goals[n]) + 80)
    tmax_idle = max(120, min(400, int(rough)))
    # Seed idle / non-mover occupancy for the planning horizon.
    for n in starts:
        if n in movers:
            continue
        c = starts[n]
        for t in range(0, tmax_idle + int(hold_after)):
            key = (t, c)
            if key not in reserved:
                reserved[key] = n
    _door_cells = _all_unload_doors(set(static))
    order = sorted(
        movers,
        key=lambda n: (
            0 if starts.get(n) in _door_cells else 1,
            _manh(starts[n], goals[n]),
            n,
        ),
    )
    ha = max(0, int(hold_after))
    for n in order:
        s, g = starts[n], goals[n]
        span = _manh(s, g) + 60 + sum(
            max(0, len(out[m]) - 1) for m in order if m in out and out[m]
        ) // 2
        path = _spacetime_bfs(s, g, static, reserved, n, tmax=max(120, min(400, span)))
        if path is None or (path and path[-1] != g):
            return None
        out[n] = path
        for i, c in enumerate(path):
            reserved[(i, c)] = n
        # hold_after=0: free the goal next tick (shared-pad time multiplex).
        for t in range(len(path), len(path) + ha):
            reserved[(t, path[-1])] = n
        # Materialize hold ticks on the path (reservation alone does not dwell).
        for _ in range(ha):
            out[n].append(path[-1])
    T = max((len(out[n]) for n in movers), default=1)
    for n in starts:
        if n not in movers:
            out[n] = [starts[n]] * T
        else:
            while len(out[n]) < T:
                out[n].append(out[n][-1])
    return out


def _plan_prioritized_paths(
    starts: Dict[str, Cell],
    goals: Dict[str, Cell],
    movers: Set[str],
    static: Set[Cell],
    *,
    seed_reserved: Optional[Dict[Tuple[int, Cell], str]] = None,
    grid: Optional[GridMap] = None,
    weight: float = 1.5,
    time_limit: float = 2.0,
    max_expansions: int = 80_000,
) -> Optional[Dict[str, List[Cell]]]:
    """Fast joint-core planner for medium/hard: priority spacetime A*.

    1) In-repo reservation-table Prioritized (``_prioritized_st_paths``)
    2) Optional pymapf ``prioritized`` fallback when a GridMap is provided
    """
    movers = set(movers)
    if not movers:
        return {n: [starts[n]] for n in starts}
    paths = _prioritized_st_paths(
        starts, goals, movers, static, seed_reserved=seed_reserved
    )
    if paths is not None:
        return paths
    if grid is None or seed_reserved:
        return None
    # pymapf prioritized (no seed reservation support)
    return _ecbs(
        grid,
        starts,
        goals,
            weight=float(weight),
            time_limit=float(time_limit),
            max_expansions=int(max_expansions),
        movers=movers,
        plan="per_agent",
    )


def _plan_joint_core_paths(
    grid: GridMap,
    starts: Dict[str, Cell],
    goals: Dict[str, Cell],
    *,
    movers: Set[str],
    static: Set[Cell],
    joint_core: str,
    joint_plan: str,
    weight: float,
    time_limit: float,
    max_expansions: int,
    plan_horizon: int = 24,
    exec_horizon: int = 12,
    seed_reserved: Optional[Dict[Tuple[int, Cell], str]] = None,
) -> Optional[Dict[str, List[Cell]]]:
    """Medium/hard joint core: Prioritized (default) or ECBS (switch).

    ECBS implementation is retained; set ``joint_core='ecbs'`` to re-enable.
    """
    core = str(joint_core or DEFAULT_JOINT_CORE).strip().lower()
    if core == "ecbs":
        return _plan_wave_paths(
        grid,
            starts,
            goals,
            movers=set(movers),
            static=static,
            planner="ecbs",
            joint_plan=joint_plan,
        weight=weight,
            time_limit=time_limit,
        max_expansions=max_expansions,
            plan_horizon=plan_horizon,
            exec_horizon=exec_horizon,
        )
    return _plan_prioritized_paths(
        starts,
        goals,
        set(movers),
        static,
        seed_reserved=seed_reserved,
        grid=grid,
        weight=weight,
        time_limit=time_limit,
        max_expansions=max_expansions,
    )


def _seed_reserved_from_paths(
    paths: Dict[str, List[Cell]],
    movers: Set[str],
    *,
    hold_after: int = 8,
) -> Dict[Tuple[int, Cell], str]:
    """Build spacetime reservation owned by ``movers`` (ECBS core for hybrid)."""
    reserved: Dict[Tuple[int, Cell], str] = {}
    for n in movers:
        path = list(paths.get(n) or [])
        if not path:
            continue
        for i, c in enumerate(path):
            reserved[(i, c)] = n
        for t in range(len(path), len(path) + int(hold_after)):
            reserved[(t, path[-1])] = n
    return reserved


def _merge_pad_paths(
    starts: Dict[str, Cell],
    parts: List[Dict[str, List[Cell]]],
    movers: Set[str],
) -> Dict[str, List[Cell]]:
    merged: Dict[str, List[Cell]] = {n: [starts[n]] for n in starts}
    for part in parts:
        for n, path in part.items():
            if path:
                merged[n] = list(path)
    T = max((len(merged[n]) for n in movers), default=1)
    for n in starts:
        while len(merged[n]) < T:
            merged[n].append(merged[n][-1])
    return merged


def _plan_hybrid_wave_paths(
    grid: GridMap,
    starts: Dict[str, Cell],
    goals: Dict[str, Cell],
    *,
    movers_astar: Set[str],
    movers_ecbs: Set[str],
    static: Set[Cell],
    joint_plan: str,
    weight: float,
    time_limit: float,
    max_expansions: int,
    plan_horizon: int = 24,
    exec_horizon: int = 12,
    joint_core: str = DEFAULT_JOINT_CORE,
) -> Optional[Dict[str, List[Cell]]]:
    """Same-wave hybrid: joint core first, A* bystanders dodge its reservation.

    ``joint_core``: 'prioritized' (default, fast) or 'ecbs' (legacy switch).
    Core agent set is still called movers_ecbs historically (= conflict core).
    """
    movers_astar = set(movers_astar)
    movers_ecbs = set(movers_ecbs)
    movers = movers_astar | movers_ecbs
    core_mode = str(joint_core or DEFAULT_JOINT_CORE).strip().lower()
    if core_mode not in ("prioritized", "ecbs"):
        core_mode = "prioritized"
    if not movers:
        return {n: [starts[n]] for n in starts}
    if not movers_ecbs:
        return _prioritized_st_paths(starts, goals, movers_astar, static)
    if not movers_astar:
        return _plan_joint_core_paths(
        grid,
        starts,
        goals,
            movers=movers_ecbs,
            static=static,
            joint_core=core_mode,
            joint_plan=joint_plan,
        weight=weight,
        time_limit=time_limit,
        max_expansions=max_expansions,
            plan_horizon=plan_horizon,
            exec_horizon=exec_horizon,
        )

    core_paths = _plan_joint_core_paths(
        grid,
        starts,
        goals,
        movers=movers_ecbs,
        static=static,
        joint_core=core_mode,
        joint_plan=joint_plan,
        weight=weight,
        time_limit=time_limit,
        max_expansions=max_expansions,
        plan_horizon=plan_horizon,
        exec_horizon=exec_horizon,
    )
    if core_paths is None:
        # Core failed → fold everyone into same joint-core planner (not forced ECBS)
        return _plan_joint_core_paths(
            grid,
            starts,
            goals,
            movers=movers,
            static=static,
            joint_core=core_mode,
            joint_plan=joint_plan,
            weight=weight,
            time_limit=time_limit,
            max_expansions=max_expansions,
            plan_horizon=plan_horizon,
            exec_horizon=exec_horizon,
        )

    seed = _seed_reserved_from_paths(core_paths, movers_ecbs)
    astar_paths = _prioritized_st_paths(
        starts, goals, movers_astar, static, seed_reserved=seed
    )
    if astar_paths is None:
        return _plan_joint_core_paths(
            grid,
            starts,
            goals,
            movers=movers,
            static=static,
            joint_core=core_mode,
            joint_plan=joint_plan,
            weight=weight,
            time_limit=time_limit,
            max_expansions=max_expansions,
            plan_horizon=plan_horizon,
            exec_horizon=exec_horizon,
        )

    merged = _merge_pad_paths(starts, [core_paths, astar_paths], movers)
    if _cell_paths_conflict(merged, list(starts.keys())):
        return _plan_joint_core_paths(
            grid,
            starts,
            goals,
            movers=movers,
            static=static,
            joint_core=core_mode,
            joint_plan=joint_plan,
            weight=weight,
            time_limit=time_limit,
            max_expansions=max_expansions,
            plan_horizon=plan_horizon,
            exec_horizon=exec_horizon,
        )
    return merged


def _split_hybrid_movers(
    assigned: Dict[str, dict],
    pose: Dict[str, Pose],
    *,
    k_budget: int,
    hard_cap: int,
    escalate: bool,
    top_dest: str,
    wave_net: Optional[object],
    map_hardness: float,
    map_obs: Optional[object],
    static: Set[Cell],
    work_count: Dict[str, int],
    bfs_len,
    affinity: Optional[Dict[str, str]] = None,
    scene_label: str = "",
    force_promote: Optional[Set[str]] = None,
) -> Tuple[Set[str], Set[str], str]:
    """Pick ECBS conflict-core vs A* bystanders. Sticky astar affinity stays out.

    ECBS core is capped by ``hard_cap`` so a full-fleet claim never forces joint
    ECBS on all AGVs. Medium scenes use a hub-capable core (floor 3, ≤hard_cap).
    """
    names_a = set(assigned.keys())
    aff = dict(affinity or {})
    promote = set(force_promote or ())
    sticky = {
        n
        for n in names_a
        if aff.get(n) == "astar" and n not in promote
    }
    pool = names_a - sticky
    if not pool and sticky:
        # Everyone sticky-A*: no ECBS core this wave
        return set(), set(sticky), "all_sticky_astar"

    label = str(scene_label or "")
    k_req = max(1, min(int(k_budget), len(pool)))
    if label == "medium":
        # Was ≤2 (caused v2 livelock on hubs). Floor 3, cap by hard_cap.
        cap = int(hard_cap) if int(hard_cap) > 0 else 4
        floor_m = min(3, cap, len(pool))
        k_req = max(floor_m, min(k_req, cap, len(pool)))
    elif label == "hard":
        # Large-wave industrial: never a 1–2 agent ECBS core on hard maps.
        cap_h = int(hard_cap) if int(hard_cap) > 0 else 6
        floor_h = min(6, cap_h, len(pool))
        floor_h = max(4, floor_h) if len(pool) >= 4 else len(pool)
        k_req = max(floor_h, min(k_req, cap_h, len(pool)))
    # hard_cap bounds the joint ECBS subset; rest stay spacetime-A*
    cap = int(hard_cap) if int(hard_cap) > 0 else k_req
    k = max(1, min(k_req, cap, len(pool)))
    if len(pool) <= k and not sticky:
        return set(pool), set(), "all_ecbs"

    keepers: Set[str] = set()
    mode = "heuristic"
    pool_assigned = {n: assigned[n] for n in pool}
    if escalate and pool_assigned:
        from ml_research.benchmarks.wave_net.policy import select_wave_agents

        sel = select_wave_agents(
            pool_assigned,
            pose,
                static,
            bfs_len,
            work_count,
            escalate=True,
            k_budget=k,
            net=wave_net,
            map_hardness=float(map_hardness),
            map_obs=map_obs,
        )
        keepers = set(sel.keepers.keys()) & pool
        mode = f"wavenet:{sel.mode}"
    else:
        scored: List[Tuple[float, int, str]] = []
        for agv, task in pool_assigned.items():
            dest = str(task.get("destination") or "")
            pk = tuple(task.get("pickup_point") or pose[agv][:2])
            same = 0.0 if (top_dest and dest == top_dest) else 1.0
            scored.append((same, _manh(pose[agv][:2], pk), agv))
        scored.sort()
        keepers = {agv for _, _, agv in scored[:k]}
        mode = "hotspot_heuristic"

    if not keepers:
        keepers = set(list(sorted(pool))[:k])
    movers_ecbs = keepers
    movers_astar = (names_a - movers_ecbs) | sticky
    # Sticky must never remain in ECBS core
    movers_ecbs -= sticky
    movers_astar |= sticky
    if sticky:
        mode = f"{mode}+sticky{len(sticky)}"
    if movers_astar and movers_ecbs:
        mode = f"{mode}+cap{k}"
    elif not movers_ecbs:
        mode = f"{mode}+astar_only"
    return movers_ecbs, movers_astar, mode


def _planner_uses_hybrid_split(planner: str) -> bool:
    return str(planner or "") in ("ecbs", "hybrid")


def _individual_path(
    start: Cell, goal: Cell, static: Set[Cell]
) -> List[Cell]:
    """Space-only A* path; falls back to [start] if blocked."""
    if start == goal:
        return [start]
    path = _astar_cells(start, goal, set(static))
    if path:
        return path
    return [start]


def _assign_window_goals(
    cur: Dict[str, Cell],
    goals: Dict[str, Cell],
    movers: Set[str],
    static: Set[Cell],
    *,
    horizon: int,
) -> Dict[str, Cell]:
    """RHCR-style intermediate goals ≈ horizon steps along individual paths."""
    win: Dict[str, Cell] = {}
    used: Set[Cell] = set()
    # Farther agents first so closer ones can dodge occupied window cells.
    order = sorted(
        movers,
        key=lambda n: (-_manh(cur[n], goals[n]), n),
    )
    for n in order:
        path = _individual_path(cur[n], goals[n], static)
        idx = min(max(1, int(horizon)), max(0, len(path) - 1))
        g = path[idx]
        # Resolve collisions on intermediate targets by pushing further / wait.
        guard = 0
        while g in used and g != goals[n] and guard < len(path):
            idx = min(idx + 1, len(path) - 1)
            g = path[idx]
            guard += 1
        if g in used and g != goals[n]:
            # stay put this window if no free intermediate
            g = cur[n]
        win[n] = g
        used.add(g)
    for n in cur:
        if n not in win:
            win[n] = cur[n]
    return win


def _windowed_ecbs_paths(
    grid: GridMap,
    starts: Dict[str, Cell],
    goals: Dict[str, Cell],
    *,
    movers: Set[str],
    static: Set[Cell],
    weight: float,
    time_limit: float,
    max_expansions: int,
    plan: str = "joint",
    horizon: int = 24,
    exec_horizon: int = 12,
    max_rounds: int = 48,
) -> Optional[Dict[str, List[Cell]]]:
    """RHCR-style windowed WCBS: plan H steps, commit exec_h, replan.

    Speeds up maze/narrow maps by keeping each ECBS instance short-horizon.
    Returns full stitched cell paths from starts to goals (or None).
    """
    H = max(4, int(horizon))
    E = max(1, min(int(exec_horizon), H))
    movers = set(movers)
    cur = {n: starts[n] for n in starts}
    committed: Dict[str, List[Cell]] = {n: [starts[n]] for n in starts}
    # Per-window time budget: short and sharp.
    win_tl = min(float(time_limit), max(3.0, float(time_limit) * 0.35))
    win_exp = min(int(max_expansions), 80_000)

    for rnd in range(max(1, int(max_rounds))):
        if all(cur[n] == goals[n] for n in movers):
                break
        win_goals = _assign_window_goals(cur, goals, movers, static, horizon=H)
        # Idle agents stay; movers head to window goals.
        partial = _ecbs(
            grid,
            cur,
            win_goals,
            weight=weight,
            time_limit=win_tl,
            max_expansions=win_exp,
            movers=movers,
            plan=plan,
        )
        if partial is None:
            # One recovery: slightly longer window / full remaining for this round
            partial = _ecbs(
                grid,
                cur,
                win_goals,
                weight=max(weight, 1.8),
                time_limit=min(float(time_limit), win_tl * 2.0),
                max_expansions=min(int(max_expansions), win_exp * 2),
                movers=movers,
                plan=plan,
            )
        if partial is None:
            # Last ditch this round: go straight to true goals (classic ECBS)
            if rnd == 0:
                partial = _ecbs(
                    grid,
                    starts,
                    goals,
                    weight=weight,
                    time_limit=time_limit,
                    max_expansions=max_expansions,
                    movers=movers,
                    plan=plan,
                )
                if partial is not None:
                    return partial
                return None
            # Later rounds: keep what we already committed, or fail cleanly.
            if any(committed.get(n) for n in starts):
                return {
                    n: (committed[n] if committed[n] else [starts[n]])
                    for n in starts
                }
            return None

        if not isinstance(partial, dict):
            # Defensive: some planner backends may return a non-dict failure sentinel.
            print(
                f"[ECBS] windowed unexpected partial type={type(partial).__name__} "
                f"rnd={rnd} → abort round",
                flush=True,
            )
            if any(len(committed.get(n) or []) > 1 for n in starts):
                return {
                    n: (committed[n] if committed[n] else [starts[n]])
                    for n in starts
                }
            return None

        # Commit up to E steps, then pad EVERY agent to the same round length.
        # (Desynced committed lengths were stitching vertex conflicts at validate.)
        round_body: Dict[str, List[Cell]] = {}
        max_e = 0
        for n in starts:
            seq = partial.get(n) or [cur[n]]
            if not seq:
                seq = [cur[n]]
            body = list(seq)
            if body and body[0] == cur[n]:
                body = body[1:]
            step = body[:E]
            round_body[n] = step
            if step:
                max_e = max(max_e, len(step))

        if max_e <= 0:
            # No one moved — fall back to classic one-shot from here
            rest = _ecbs(
                grid,
                cur,
                goals,
                weight=weight,
                time_limit=time_limit,
                max_expansions=max_expansions,
                movers=movers,
                plan=plan,
            )
            if rest is None:
                return None
            for n in starts:
                seq = rest.get(n) or [cur[n]]
                body = list(seq)
                if body and body[0] == cur[n]:
                    body = body[1:]
                committed[n].extend(body)
                if committed[n]:
                    cur[n] = committed[n][-1]
                break

        for n in starts:
            step = list(round_body[n])
            if not step:
                step = [cur[n]] * max_e
            else:
                while len(step) < max_e:
                    step.append(step[-1])
            committed[n].extend(step)
            cur[n] = step[-1]

        # Keep global timelines aligned across rounds
        T = max((len(committed[n]) for n in starts), default=1)
        for n in starts:
            while len(committed[n]) < T:
                committed[n].append(committed[n][-1])

    # Pad to equal length for non-movers / early finishers
    if not all(cur[n] == goals[n] for n in movers):
        # Finish remaining with one classic ECBS from current poses
        rest = _ecbs(
            grid,
            cur,
            goals,
            weight=weight,
            time_limit=time_limit,
            max_expansions=max_expansions,
            movers={n for n in movers if cur[n] != goals[n]},
            plan=plan,
        )
        if rest is None:
                return None
        # Sync before appending the final segment
        T = max((len(committed[n]) for n in starts), default=1)
        for n in starts:
            while len(committed[n]) < T:
                committed[n].append(committed[n][-1])
        rest_body: Dict[str, List[Cell]] = {}
        max_rest = 0
        for n in starts:
            seq = rest.get(n) or [cur[n]]
            body = list(seq)
            if body and body[0] == cur[n]:
                body = body[1:]
            rest_body[n] = body
            if body:
                max_rest = max(max_rest, len(body))
        for n in starts:
            body = list(rest_body[n])
            if not body:
                body = [cur[n]] * max(1, max_rest) if max_rest else []
            else:
                while len(body) < max_rest:
                    body.append(body[-1])
            committed[n].extend(body)
            if committed[n]:
                cur[n] = committed[n][-1]

    T = max((len(committed[n]) for n in starts), default=1)
    for n in starts:
        while len(committed[n]) < T:
            committed[n].append(committed[n][-1])
    return committed


def _plan_wave_paths(
    grid: GridMap,
    starts: Dict[str, Cell],
    goals: Dict[str, Cell],
    *,
    movers: Set[str],
    static: Set[Cell],
    planner: str,
    joint_plan: str,
    weight: float,
    time_limit: float,
    max_expansions: int,
    plan_horizon: int = 24,
    exec_horizon: int = 12,
) -> Optional[Dict[str, List[Cell]]]:
    """Route a wave. Baseline uses parallel spacetime A*; hard uses windowed ECBS."""
    if planner == "astar":
        # True baseline: all AGVs move concurrently under reservation table.
        paths = _prioritized_st_paths(starts, goals, set(movers), static)
        if paths is not None:
            return paths
        # Soften: try again with only movers in starts/goals maps
        paths = _prioritized_st_paths(
            {n: starts[n] for n in movers},
            {n: goals[n] for n in movers},
            set(movers),
                    static,
        )
        if paths is not None:
            # pad non-movers
            T = max((len(p) for p in paths.values()), default=1)
            full = {n: [starts[n]] * T for n in starts}
            full.update(paths)
            for n in movers:
                while len(full[n]) < T:
                    full[n].append(full[n][-1])
            return full
        return None
    if int(plan_horizon) <= 0:
        return _ecbs(
            grid,
            starts,
            goals,
            weight=weight,
            time_limit=time_limit,
            max_expansions=max_expansions,
            movers=movers,
            plan=joint_plan,
        )
    return _windowed_ecbs_paths(
        grid,
        starts,
        goals,
        movers=set(movers),
        static=static,
        weight=weight,
        time_limit=time_limit,
        max_expansions=max_expansions,
        plan=joint_plan,
        horizon=int(plan_horizon),
        exec_horizon=int(exec_horizon),
    )


def _astar_cells(
    start: Cell,
    goal: Cell,
    blocked: Set[Cell],
) -> Optional[List[Cell]]:
    """Static A* treating ``blocked`` as permanent obstacles (others freeze).

    ``goal`` may sit on a blocked cell (station pads are in ``blocked_plan``);
    the goal cell itself is always traversable as the destination.
    """
    if start == goal:
        return [start]
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    gscore = {start: 0}
    open_h = [(_manh(start, goal), 0, start)]
    closed: Set[Cell] = set()
    while open_h:
        _f, _, cur = heapq.heappop(open_h)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal:
            rev: List[Cell] = []
            c: Optional[Cell] = cur
            while c is not None:
                rev.append(c)
                c = parent[c]
            rev.reverse()
            return rev
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                continue
            if nxt in blocked and nxt != goal:
                continue
            ng = gscore[cur] + 1
            if ng >= gscore.get(nxt, 10**9):
                continue
            gscore[nxt] = ng
            parent[nxt] = cur
            heapq.heappush(open_h, (ng + _manh(nxt, goal), ng, nxt))
    return None


def _cell_paths_conflict(
    paths: Dict[str, List[Cell]],
    names: List[str],
    *,
    pad_holds: bool = True,
) -> bool:
    """True if any same-time vertex or swap conflict in cell paths.

    ``pad_holds=False``: agents disappear after their path ends (no goal-hold
    padding). Use for shared pickup/unload pads where ST already timed arrivals.
    """
    if not names:
        return False
    lens = {n: len(paths.get(n) or []) for n in names}
    T = max(lens.values(), default=0)
    if T <= 0:
        return False
    prev: Dict[str, Cell] = {}
    for n in names:
        seq = paths.get(n) or []
        if seq:
            prev[n] = seq[0]
    # t=0 vertex
    occ0: Dict[Cell, str] = {}
    for n in names:
        seq = paths.get(n) or []
        if not seq:
            continue
        c = seq[0]
        if c in occ0:
            return True
        occ0[c] = n
    for t in range(1, T):
        cur: Dict[str, Cell] = {}
        for n in names:
            seq = paths.get(n) or []
            if not seq:
                continue
            if t < len(seq):
                cur[n] = seq[t]
            elif pad_holds:
                cur[n] = seq[-1]
            # else: arrived — leave conflict table
        occ: Dict[Cell, str] = {}
        for n, c in cur.items():
            if c in occ:
                return True
            occ[c] = n
        for n, c in cur.items():
            p = prev.get(n)
            if p is None or c == p:
                continue
            for m, mc in cur.items():
                if m == n:
                    continue
                if mc == p and prev.get(m) == c:
                    return True
        prev = {**{n: prev[n] for n in prev if n not in cur}, **cur}
    return False


def _trim_paths_to_goals(
    paths: Dict[str, List[Cell]],
    goals: Dict[str, Cell],
    movers: Set[str],
) -> Dict[str, List[Cell]]:
    """Drop post-arrival holds so shared unload pads do not false-conflict."""
    out: Dict[str, List[Cell]] = {}
    for n, seq in paths.items():
        if n not in movers or n not in goals or not seq:
            out[n] = list(seq)
            continue
        g = goals[n]
        trimmed = seq
        for i, c in enumerate(seq):
            if c == g:
                trimmed = seq[: i + 1]
                break
        out[n] = trimmed
    return out


def _vacate_extend_after_goals(
    paths: Dict[str, List[Cell]],
    movers: Set[str],
    goals: Dict[str, Cell],
    static: Set[Cell],
    starts: Dict[str, Cell],
) -> Dict[str, List[Cell]]:
    """Trim to unload pad, dwell one tick, then one-hop off (shared-pad safe)."""
    trimmed = _trim_paths_to_goals(paths, goals, movers)
    out: Dict[str, List[Cell]] = {
        n: list(trimmed.get(n) or paths.get(n) or [starts.get(n, (1, 1))])
        for n in starts
    }
    order = sorted(
        (n for n in movers if n in out and out[n]),
        key=lambda n: (len(out[n]), n),
    )
    reserved: Dict[Tuple[int, Cell], str] = {}
    for n in order:
        seq = list(out[n])
        if not seq:
            continue
        g = seq[-1]
        # Path to pad + one dwell tick on pad.
        seq_dwell = list(seq) + [g]
        for i, c in enumerate(seq_dwell):
            reserved[(i, c)] = n
        t_vac = len(seq_dwell)
        vac: Optional[Cell] = None
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (int(g[0]) + dx, int(g[1]) + dy)
            if not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                continue
            if nxt in static:
                continue
            if reserved.get((t_vac, nxt)):
                continue
            vac = nxt
            break
        if vac is not None:
            out[n] = seq_dwell + [vac]
            reserved[(t_vac, vac)] = n
        else:
            out[n] = seq_dwell
    T = max((len(out[n]) for n in movers if n in out), default=1)
    for n in movers:
        if n not in out or not out[n]:
            continue
        while len(out[n]) < T:
            out[n].append(out[n][-1])
    for n in starts:
        if n in movers:
            continue
        c0 = starts[n]
        out[n] = [c0] * T
    return out


def _st_paths_ok(
    paths: Optional[Dict[str, List[Cell]]],
    movers: Set[str],
    goals: Optional[Dict[str, Cell]] = None,
) -> bool:
    """True if mover paths are vertex/swap-safe w.r.t. each other and idlers.

    Movers: trim to goal (shared-pad arrivals). Non-movers: hold start cell
    for the mover horizon so ghost-through plans cannot pass the gate.
    """
    if paths is None or not movers:
        return False
    check: Dict[str, List[Cell]] = {}
    if goals is not None:
        trimmed = _trim_paths_to_goals(paths, goals, movers)
    else:
        trimmed = paths
    for n in movers:
        seq = list(trimmed.get(n) or paths.get(n) or [])
        if not seq:
            return False
        check[n] = seq
    T = max((len(check[n]) for n in movers), default=1)
    for n, seq in paths.items():
        if n in movers:
            continue
        if not seq:
            continue
        check[n] = [seq[0]] * T
    return not _cell_paths_conflict(check, list(check.keys()), pad_holds=False)


def _st_movers_only_ok(
    paths: Optional[Dict[str, List[Cell]]],
    movers: Set[str],
    goals: Optional[Dict[str, Cell]] = None,
) -> bool:
    """Vertex/swap-safe among movers only (ignore parked idlers).

    Joint ECBS/ST often plans around reserved idlers but ``_st_paths_ok`` still
    rejects when an idler sits on a trimmed cell. For pickup apply we accept
    movers-only safety then disperse idlers after (or before) apply — otherwise
    waves re-claim forever (r19 cell-conflict spin).
    """
    if paths is None or not movers:
        return False
    if goals is not None:
        trimmed = _trim_paths_to_goals(paths, goals, movers)
    else:
        trimmed = paths
    check: Dict[str, List[Cell]] = {}
    for n in movers:
        seq = list(trimmed.get(n) or paths.get(n) or [])
        if not seq:
            return False
        check[n] = seq
    return not _cell_paths_conflict(check, list(check.keys()), pad_holds=False)


def _pose_timelines_conflict(
    pose0: Dict[str, Pose],
    timelines: Dict[str, List[Pose]],
    *,
    pad_holds: bool = True,
) -> bool:
    """Vertex/swap check on pose timelines.

    ``pad_holds=False``: agents that finished their timeline leave the table
    (shared pickup/unload pads). Matches ``_st_paths_ok`` / cell pad_holds=False.
    """
    names = list(pose0.keys())
    T = max((len(timelines.get(n) or []) for n in names), default=0)
    prev = {n: pose0[n][:2] for n in names}
    for t in range(T):
        cur: Dict[str, Cell] = {}
        for n in names:
            seq = timelines.get(n) or []
            if t < len(seq):
                cur[n] = seq[t][:2]
            elif pad_holds:
                cur[n] = prev[n]
            # else: arrived — omit from occupancy
        occ: Dict[Cell, str] = {}
        for n, c in cur.items():
            if c in occ:
                return True
            occ[c] = n
        # swap
        for n, c in cur.items():
            p = prev.get(n)
            if p is None or c == p:
                continue
            for m, mc in cur.items():
                if m == n:
                    continue
                if mc == p and prev.get(m) == c:
                    return True
        prev = {**{n: prev[n] for n in prev if n not in cur}, **cur}
    return False


def _pose_movers_ok(
    pose: Dict[str, Pose],
    timelines: Dict[str, List[Pose]],
    movers: Set[str],
) -> bool:
    """Pose vertex/swap check among movers only (shared-pad friendly)."""
    if not movers:
        return True
    mp = {a: pose[a] for a in movers if a in pose}
    mt = {a: timelines.get(a) or [] for a in movers}
    return not _pose_timelines_conflict(mp, mt, pad_holds=False)


def _serial_move_to(
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    now: int,
    mover: str,
    goal: Cell,
    blocked_plan: Set[Cell],
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    _depth: int = 0,
) -> Tuple[int, bool]:
    """Move one AGV with others frozen; evacuate blockers if needed.

    Never applies a path that ignores other agents (collision-safe).
    Never targets maze walls / blocked_plan cells.
    """
    names = list(pose.keys())
    goal = (int(goal[0]), int(goal[1]))
    # Station pads are members of blocked_plan but are valid destinations.
    # Snapping them away (old behavior) made pickup serial/corridor never land
    # on the true pad (r35–38 spin corridor-miss).
    if pose[mover][:2] == goal:
        return now, True

    def _try_path() -> Optional[List[Cell]]:
        frozen = {pose[n][:2] for n in names if n != mover}
        blocked = set(blocked_plan) | frozen
        blocked.discard(pose[mover][:2])
        blocked.discard(goal)
        return _astar_cells(pose[mover][:2], goal, blocked)

    def _park_spot(other: str, keep_clear: Set[Cell]) -> Optional[Cell]:
        occ = {pose[n][:2] for n in names}
        ox, oy = pose[other][:2]
        # Prefer true free cells (not walls, not station pads).
        cands = []
        for dx in range(-8, 9):
            for dy in range(-8, 9):
                if dx == 0 and dy == 0:
                    continue
                c = (ox + dx, oy + dy)
                if not (1 <= c[0] <= 20 and 1 <= c[1] <= 20):
                    continue
                if c in blocked_plan or c in keep_clear or c in occ:
                    continue
                cands.append(c)
        # Prefer off-pad: if any cand is outside blocked_plan walls only,
        # still OK; callers pass walls-only blocked_plan. Filter pads via
        # keep_clear when known. Global fallback below.
        if not cands:
            # Global fallback: any free cell on the map (hub livelock escape).
            for x in range(1, 21):
                for y in range(1, 21):
                    c = (x, y)
                    if c in blocked_plan or c in keep_clear or c in occ:
                        continue
                    if c == (ox, oy):
                        continue
                    cands.append(c)
        if not cands:
            return None
        return min(cands, key=lambda c: (_manh((ox, oy), c), c))

    path = _try_path()
    if path is None and _depth < 6:
        # 1) Evict anyone sitting on the goal
        for other in list(names):
            if other == mover:
                continue
            if pose[other][:2] != goal:
                continue
            park = _park_spot(other, {goal, pose[mover][:2]})
            if park is None:
                return now, False
            now, ok = _serial_move_to(
                pose,
                steps_by,
                now,
                other,
                park,
                blocked_plan,
                loaded=loaded,
                dest=dest,
                tid=tid,
                _depth=_depth + 1,
            )
            if not ok:
                return now, False
        path = _try_path()

    if path is None and _depth < 6:
        # 2) Corridor soft path → evacuate blockers along it
        blocked = set(blocked_plan)
        blocked.discard(pose[mover][:2])
        blocked.discard(goal)
        soft = _astar_cells(pose[mover][:2], goal, blocked)
        if soft is None:
            return now, False
        path_set = set(soft[1:])
        for other in list(names):
            if other == mover:
                continue
            if pose[other][:2] not in path_set:
                continue
            park = _park_spot(other, path_set | {goal})
            if park is None:
                continue
            now, ok = _serial_move_to(
                pose,
                    steps_by,
                    now,
                other,
                park,
                blocked_plan,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                _depth=_depth + 1,
            )
            if not ok:
                return now, False
        path = _try_path()
    if path is None:
        return now, False

    paths = {n: [pose[n][:2]] for n in names}
    paths[mover] = path
    if _cell_paths_conflict(paths, names):
        return now, False
    timelines = _plan_paths_as_poses(pose, paths)
    if _pose_timelines_conflict(pose, timelines):
        return now, False
    now = _apply_pose_timelines(
        steps_by, pose, timelines, now, loaded=loaded, dest=dest, tid=tid,
        rise_ok={},
    )
    return now, True


def _one_hop_move(
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    now: int,
    mover: str,
    dest_cell: Cell,
    blocked_plan: Set[Cell],
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    allow_cells: Optional[Set[Cell]] = None,
) -> Tuple[int, bool]:
    """Collision-safe 4-neighbor step onto an empty cell."""
    cur = pose[mover][:2]
    dest_cell = (int(dest_cell[0]), int(dest_cell[1]))
    if dest_cell == cur:
        return now, True
    if abs(dest_cell[0] - cur[0]) + abs(dest_cell[1] - cur[1]) != 1:
        return now, False
    if not (1 <= dest_cell[0] <= 20 and 1 <= dest_cell[1] <= 20):
        return now, False
    occ = {pose[n][:2] for n in pose if n != mover}
    if dest_cell in occ:
        return now, False
    # Walls + station pads are in blocked_plan; pad commits pass allow_cells.
    if dest_cell in blocked_plan and dest_cell != cur:
        if not allow_cells or dest_cell not in allow_cells:
            return now, False
    # Full-fleet paths so non-movers get hold pads (keeps trajectory clocks
    # aligned). Conflict check skips duplicate occupants — jam/dump can leave
    # two AGVs on one cell; a raw t=0 vertex check then rejects every hop
    # (r47–49). One representative per cell is enough to block walking through.
    paths = {n: [pose[n][:2]] for n in pose}
    paths[mover] = [cur, dest_cell]
    seen: Set[Cell] = set()
    conflict_names: List[str] = []
    for n in pose:
        c = pose[n][:2]
        if n != mover and c in seen:
            continue
        seen.add(c)
        conflict_names.append(n)
    if mover not in conflict_names:
        conflict_names.insert(0, mover)
    if _cell_paths_conflict(paths, conflict_names, pad_holds=True):
        return now, False
    timelines = _plan_paths_as_poses(pose, paths)
    now = _apply_pose_timelines(
        steps_by, pose, timelines, now, loaded=loaded, dest=dest, tid=tid,
        rise_ok={},
    )
    return now, True


def _greedy_reach(
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    now: int,
    mover: str,
    goal: Cell,
    blocked_plan: Set[Cell],
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    max_steps: int = 80,
) -> Tuple[int, bool]:
    """Manhattan-greedy 1-hop with neighbor shove — hub livelock escape.

    Full frozen-A* (``_serial_move_to``) fails when the corridor is packed;
    this only ever moves one adjacent cell after clearing that cell.
    """
    goal = (int(goal[0]), int(goal[1]))
    if pose[mover][:2] == goal:
        return now, True
    names = list(pose.keys())

    def _occ_at(cell: Cell) -> Optional[str]:
        for n in names:
            if n != mover and pose[n][:2] == cell:
                return n
        return None

    def _shove(who: str, nxt: Cell) -> bool:
        nonlocal now
        ox, oy = pose[who][:2]
        parks = [(ox + 1, oy), (ox - 1, oy), (ox, oy + 1), (ox, oy - 1)]
        parks.sort(key=lambda c: (-_manh(c, goal), c))
        cur = pose[mover][:2]
        for park in parks:
            if park == cur or park == nxt:
                continue
            if park in blocked_plan:
                continue
            if not (1 <= park[0] <= 20 and 1 <= park[1] <= 20):
                continue
            if any(pose[n][:2] == park for n in names if n != who):
                continue
            now, ok = _one_hop_move(
                pose,
                steps_by,
                now,
                who,
                park,
                blocked_plan,
                loaded=loaded,
                dest=dest,
                tid=tid,
            )
            if ok:
                return True
        # Global park via serial (r34: 4-neighbor shove fails in packed hubs).
        occ = {pose[n][:2] for n in names if n != who} | {cur, nxt, goal}
        cands = [
            (x, y)
            for x in range(1, 21)
            for y in range(1, 21)
            if (x, y) not in blocked_plan and (x, y) not in occ
        ]
        if not cands:
            return False
        # Prefer non-goal-adjacent empties far from the mover's goal corridor.
        park = min(cands, key=lambda c: (-_manh(c, goal), c))
        now, ok = _serial_move_to(
            pose,
            steps_by,
            now,
            who,
            park,
            blocked_plan,
            loaded=loaded,
            dest=dest,
            tid=tid,
        )
        return bool(ok)

    for _ in range(max(1, int(max_steps))):
        cur = pose[mover][:2]
        if cur == goal:
            return now, True
        path = _astar_cells(cur, goal, set(blocked_plan))
        if path is None or len(path) < 2:
            break
        nxt = path[1]
        who = _occ_at(nxt)
        if who is not None:
            if not _shove(who, nxt):
                break
            if _occ_at(nxt) is not None:
                break
        now, ok = _one_hop_move(
            pose,
            steps_by,
            now,
            mover,
            nxt,
            blocked_plan,
            loaded=loaded,
            dest=dest,
            tid=tid,
            allow_cells={goal},
        )
        if not ok:
            break
    return now, pose[mover][:2] == goal


def _jam_fleet_unload(
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    now: int,
    unloader: str,
    pad: Cell,
    blocked_plan: Set[Cell],
    free: Set[Cell],
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
) -> Tuple[int, bool]:
    """Park everyone else, then move unloader to pad (collision-safe)."""
    names = list(pose.keys())
    pad = (int(pad[0]), int(pad[1]))
    if pose[unloader][:2] == pad:
        return now, True

    corners = [
        (1, 1), (1, 20), (20, 1), (20, 20),
        (1, 10), (20, 10), (10, 1), (10, 20),
        (5, 1), (1, 5), (20, 5), (5, 20),
        (15, 1), (1, 15), (20, 15), (15, 20),
    ]

    def _park_others(keep: Set[Cell]) -> None:
        nonlocal now
        used: Set[Cell] = set(keep) | {pose[unloader][:2]}
        others = sorted(
            (n for n in names if n != unloader),
            key=lambda n: (_manh(pose[n][:2], pad), n),
        )
        for i, n in enumerate(others):
            # Already far from pad and off keep — skip.
            if (
                pose[n][:2] not in keep
                and _manh(pose[n][:2], pad) >= 6
                and pose[n][:2] not in used
            ):
                used.add(pose[n][:2])
                continue
            occ = {pose[m][:2] for m in names if m != n} | used
            cands = [
                c
                for c in free
                if c not in occ and c not in keep and c not in blocked_plan
            ]
            if not cands:
                # Never fall back onto station pads (not in ``free``).
                continue
            raw = corners[i % len(corners)]
            park = min(
                cands,
                key=lambda c: (_manh(c, raw), -_manh(c, pad), c),
            )
            now, ok = _serial_move_to(
                pose,
                steps_by,
                now,
                n,
                park,
                blocked_plan,
                loaded=loaded,
                dest=dest,
                tid=tid,
            )
            if not ok:
                now, ok = _greedy_reach(
                    pose,
                    steps_by,
                    now,
                    n,
                    park,
                    blocked_plan,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                    max_steps=60,
                )
            if ok:
                used.add(pose[n][:2])

    soft = _astar_cells(pose[unloader][:2], pad, set(blocked_plan))
    keep = set(soft) if soft else {pad, pose[unloader][:2]}
    for _pass in range(2):
        _park_others(keep)
        if pose[unloader][:2] == pad:
            return now, True
        now, ok = _serial_move_to(
            pose,
            steps_by,
            now,
            unloader,
            pad,
            blocked_plan,
            loaded=loaded,
            dest=dest,
            tid=tid,
        )
        if ok and pose[unloader][:2] == pad:
            return now, True
        now, ok = _greedy_reach(
            pose,
            steps_by,
            now,
            unloader,
            pad,
            blocked_plan,
            loaded=loaded,
            dest=dest,
            tid=tid,
            max_steps=160,
        )
        if ok and pose[unloader][:2] == pad:
            return now, True
        soft = _astar_cells(pose[unloader][:2], pad, set(blocked_plan))
        keep = set(soft) if soft else {pad, pose[unloader][:2]}

    starts = {n: pose[n][:2] for n in names}
    goals_m = {n: starts[n] for n in names}
    goals_m[unloader] = pad
    sp = _prioritized_st_paths(
        starts, goals_m, {unloader}, blocked_plan, hold_after=0
    )
    if sp is not None and _st_paths_ok(sp, {unloader}, goals_m):
        timelines = _plan_paths_as_poses(pose, sp)
        if not _pose_timelines_conflict(pose, timelines, pad_holds=True):
            now = _apply_pose_timelines(
                steps_by,
                pose,
                timelines,
                now,
                loaded=loaded,
                dest=dest,
                tid=tid,
                rise_ok={},
)
    return now, pose[unloader][:2] == pad


def _force_step_toward(
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    now: int,
    mover: str,
    goal: Cell,
    blocked_plan: Set[Cell],
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    max_steps: int = 8,
) -> Tuple[int, bool]:
    """Edge-deadlock escape: shove neighbor aside, one-hop toward goal."""
    goal = (int(goal[0]), int(goal[1]))
    names = list(pose.keys())
    for _ in range(max(1, int(max_steps))):
        cur = pose[mover][:2]
        if cur == goal:
            return now, True
        x, y = cur
        nbrs = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        nbrs = [
            c
            for c in nbrs
            if 1 <= c[0] <= 20 and 1 <= c[1] <= 20
        ]
        nbrs.sort(key=lambda c: (_manh(c, goal), c))
        stepped = False
        for nxt in nbrs:
            who = next((n for n in names if n != mover and pose[n][:2] == nxt), None)
            if who is not None:
                occ = {pose[n][:2] for n in names if n != who} | {cur, goal, nxt}
                # Prefer cells far from goal; skip walls. Station pads may appear
                # here (blocked_plan is walls-only) — shove to any empty non-wall
                # then mover takes nxt; pad occupancy is cleared by the shove.
                cands = [
                    (a, b)
                    for a in range(1, 21)
                    for b in range(1, 21)
                    if (a, b) not in blocked_plan and (a, b) not in occ
                ]
                if not cands:
                    continue
                park = min(cands, key=lambda c: (-_manh(c, goal), c))
                now, ok = _serial_move_to(
                    pose,
                    steps_by,
                    now,
                    who,
                    park,
                    blocked_plan,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                )
                if not ok:
                    # Adjacent shove before giving up on this nxt.
                    ox, oy = pose[who][:2]
                    shoved = False
                    for park2 in (
                        (ox + 1, oy),
                        (ox - 1, oy),
                        (ox, oy + 1),
                        (ox, oy - 1),
                    ):
                        if park2 in occ or park2 == cur or park2 == nxt:
                            continue
                        if park2 in blocked_plan:
                            continue
                        if not (1 <= park2[0] <= 20 and 1 <= park2[1] <= 20):
                            continue
                        now, okp = _one_hop_move(
                            pose,
                            steps_by,
                            now,
                            who,
                            park2,
                            blocked_plan,
                            loaded=loaded,
                            dest=dest,
                            tid=tid,
                        )
                        if okp:
                            shoved = True
                            break
                    if not shoved:
                        continue
            now, ok = _one_hop_move(
                pose,
                steps_by,
                now,
                mover,
                nxt,
                blocked_plan,
                loaded=loaded,
                dest=dest,
                tid=tid,
                allow_cells={goal},
            )
            if ok:
                stepped = True
                break
        if not stepped:
            break
    return now, pose[mover][:2] == goal


def _adjacent_claim_pad(
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    now: int,
    mover: str,
    goal: Cell,
    blocked_plan: Set[Cell],
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
) -> Tuple[int, bool]:
    """Manhattan≤2 pad claim: shove pad occupant, then one/two hops.

    Heavy ``_jam_fleet_unload`` often scrambles the left/right corridors and
    still leaves manh=1 failures (r46). Prefer a surgical adjacent claim first.
    """
    goal = (int(goal[0]), int(goal[1]))
    cur = pose[mover][:2]
    if cur == goal:
        return now, True
    d0 = _manh(cur, goal)
    if d0 > 2:
        return now, False
    names = list(pose.keys())

    def _occ(cell: Cell) -> Optional[str]:
        for n in names:
            if n != mover and pose[n][:2] == cell:
                return n
        return None

    def _shove_off(who: str, keep: Cell) -> bool:
        nonlocal now
        ox, oy = pose[who][:2]
        occ = {pose[n][:2] for n in names if n != who} | {keep, cur, goal}
        for park in (
            (ox + 1, oy),
            (ox - 1, oy),
            (ox, oy + 1),
            (ox, oy - 1),
        ):
            if park in occ or park in blocked_plan:
                continue
            if not (1 <= park[0] <= 20 and 1 <= park[1] <= 20):
                continue
            now, ok = _one_hop_move(
                pose,
                steps_by,
                now,
                who,
                park,
                blocked_plan,
                loaded=loaded,
                dest=dest,
                tid=tid,
            )
            if ok:
                return True
        cands = [
            (a, b)
            for a in range(1, 21)
            for b in range(1, 21)
            if (a, b) not in blocked_plan and (a, b) not in occ
        ]
        if not cands:
            return False
        park = min(cands, key=lambda c: (_manh((ox, oy), c), -_manh(c, goal), c))
        now, ok = _serial_move_to(
            pose,
            steps_by,
            now,
            who,
            park,
            blocked_plan,
            loaded=loaded,
            dest=dest,
            tid=tid,
        )
        return bool(ok)

    # Clear goal if occupied, then step in (repeat up to 3 hops for manh=2).
    for _ in range(3):
        cur = pose[mover][:2]
        if cur == goal:
            return now, True
        who_g = _occ(goal)
        if who_g is not None:
            if not _shove_off(who_g, goal):
                return now, False
        if _manh(pose[mover][:2], goal) == 1 and _occ(goal) is None:
            now, ok = _one_hop_move(
                pose,
                steps_by,
                now,
                mover,
                goal,
                blocked_plan,
                loaded=loaded,
                dest=dest,
                tid=tid,
                allow_cells={goal},
            )
            if ok and pose[mover][:2] == goal:
                return now, True
            return now, False
        now, ok = _force_step_toward(
            pose,
            steps_by,
            now,
            mover,
            goal,
            blocked_plan,
            loaded=loaded,
            dest=dest,
            tid=tid,
            max_steps=4,
        )
        if ok and pose[mover][:2] == goal:
            return now, True
        if pose[mover][:2] == cur:
            break
    return now, pose[mover][:2] == goal


def _clear_edge_then_claim(
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    now: int,
    mover: str,
    goal: Cell,
    blocked_plan: Set[Cell],
    free: Set[Cell],
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
) -> Tuple[int, bool]:
    """Park everyone off the edge pickup column/row, then claim the pad.

    SH02 pickups sit on x=1 / x=20. After dump, AGVs pack that column and
    jam/serial/force all fail (r47: done stuck ~373, hard-reset storm).
    """
    goal = (int(goal[0]), int(goal[1]))
    if pose[mover][:2] == goal:
        return now, True
    gx, gy = goal
    names = list(pose.keys())
    edge_col = gx in (1, 20)
    edge_row = gy in (1, 20)
    if not edge_col and not edge_row:
        return now, False

    def _nudge_off_edge(n: str) -> None:
        nonlocal now
        # Walk toward map interior with one-hops (avoids serial A* fail in maze).
        for _ in range(24):
            px, py = pose[n][:2]
            on = (px == gx) if edge_col else (py == gy)
            if not on and _manh((px, py), goal) >= 4:
                return
            if edge_col:
                pref = [(px + (1 if gx == 1 else -1), py), (px, py + 1), (px, py - 1)]
            else:
                pref = [(px, py + (1 if gy == 1 else -1)), (px + 1, py), (px - 1, py)]
            moved = False
            occ = {pose[m][:2] for m in names if m != n}
            for nxt in pref:
                if nxt in blocked_plan or nxt in occ:
                    continue
                if not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                    continue
                if nxt == goal or nxt == pose[mover][:2]:
                    continue
                now, ok = _one_hop_move(
                    pose,
                    steps_by,
                    now,
                    n,
                    nxt,
                    blocked_plan,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                )
                if ok:
                    moved = True
                    break
            if not moved:
                return

    others = sorted(
        (n for n in names if n != mover),
        key=lambda n: (_manh(pose[n][:2], goal), n),
    )
    for n in others:
        px, py = pose[n][:2]
        on = (px == gx) if edge_col else (py == gy)
        if on or _manh((px, py), goal) < 5:
            _nudge_off_edge(n)

    # Second pass: serial park remaining column squatters into free interior.
    for n in others:
        px, py = pose[n][:2]
        on = (px == gx) if edge_col else (py == gy)
        if not on:
            continue
        occ = {pose[m][:2] for m in names if m != n} | {goal, pose[mover][:2]}
        band = [
            c
            for c in free
            if c not in occ
            and (
                (6 <= c[0] <= 15)
                if edge_col
                else (6 <= c[1] <= 15)
            )
        ]
        if not band:
            band = [c for c in free if c not in occ]
        if not band:
            continue
        park = min(band, key=lambda c: (_manh((px, py), c), -_manh(c, goal), c))
        now, ok = _serial_move_to(
            pose,
            steps_by,
            now,
            n,
            park,
            blocked_plan,
            loaded=loaded,
            dest=dest,
            tid=tid,
        )
        if not ok:
            now, _ = _greedy_reach(
                pose,
                steps_by,
                now,
                n,
                park,
                blocked_plan,
                loaded=loaded,
                dest=dest,
                tid=tid,
                max_steps=100,
            )

    now, ok = _adjacent_claim_pad(
        pose,
        steps_by,
        now,
        mover,
        goal,
        blocked_plan,
        loaded=loaded,
        dest=dest,
        tid=tid,
    )
    if ok:
        return now, True
    now, ok = _force_step_toward(
        pose,
        steps_by,
        now,
        mover,
        goal,
        blocked_plan,
        loaded=loaded,
        dest=dest,
        tid=tid,
        max_steps=40,
    )
    if ok and pose[mover][:2] == goal:
        return now, True
    now, ok = _serial_move_to(
        pose,
        steps_by,
        now,
        mover,
        goal,
        blocked_plan,
        loaded=loaded,
        dest=dest,
        tid=tid,
    )
    if ok and pose[mover][:2] == goal:
        return now, True
    now, ok = _greedy_reach(
        pose,
        steps_by,
        now,
        mover,
        goal,
        blocked_plan,
        loaded=loaded,
        dest=dest,
        tid=tid,
        max_steps=240,
    )
    return now, bool(ok and pose[mover][:2] == goal)


def _separate_pose_overlaps(
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    now: int,
    blocked_plan: Set[Cell],
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    max_rounds: int = 8,
) -> int:
    """Break multi-AGV cells left by jam/dump (r52: sustained vertex collisions)."""
    names = list(pose.keys())
    for _ in range(max(1, int(max_rounds))):
        buckets: Dict[Cell, List[str]] = {}
        for n in names:
            buckets.setdefault(pose[n][:2], []).append(n)
        overlaps = {c: ns for c, ns in buckets.items() if len(ns) > 1}
        if not overlaps:
            return now
        for cell, ns in sorted(overlaps.items(), key=lambda kv: kv[0]):
            for n in sorted(ns)[1:]:
                ox, oy = pose[n][:2]
                occ = {pose[m][:2] for m in names if m != n}
                nudged = False
                for park in (
                    (ox + 1, oy),
                    (ox - 1, oy),
                    (ox, oy + 1),
                    (ox, oy - 1),
                ):
                    if park in blocked_plan or park in occ:
                        continue
                    if not (1 <= park[0] <= 20 and 1 <= park[1] <= 20):
                        continue
                    now, ok = _one_hop_move(
                        pose,
                        steps_by,
                        now,
                        n,
                        park,
                        blocked_plan,
                        loaded=loaded,
                        dest=dest,
                        tid=tid,
                    )
                    if ok:
                        nudged = True
                        break
                if nudged:
                    continue
                cands = [
                    (a, b)
                    for a in range(1, 21)
                    for b in range(1, 21)
                    if (a, b) not in blocked_plan and (a, b) not in occ
                ]
                if not cands:
                    continue
                park = min(
                    cands,
                    key=lambda c: (_manh((ox, oy), c), -_manh(c, cell), c),
                )
                now, _ = _serial_move_to(
                    pose,
                    steps_by,
                    now,
                    n,
                    park,
                    blocked_plan,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                )
    return now


def _nearby_park(
    from_cell: Cell, free: Set[Cell], forbidden: Set[Cell]
) -> Optional[Cell]:
    """Nearest free cell not in ``forbidden`` (for post-unload clear)."""
    if from_cell in free and from_cell not in forbidden:
        # Prefer stepping off the pad: look at neighbors first
        pass
    x, y = from_cell
    cands: List[Cell] = []
    for rad in range(1, 8):
        for dx in range(-rad, rad + 1):
            for dy in range(-rad, rad + 1):
                if abs(dx) + abs(dy) != rad:
                    continue
                c = (x + dx, y + dy)
                if c in free and c not in forbidden:
                    cands.append(c)
        if cands:
            break
    if not cands:
        return None
    return min(cands, key=lambda c: (_manh(from_cell, c), c))


def _disperse_idle_agents(
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    now: int,
    *,
    movers: Set[str],
    free: Set[Cell],
    blocked_plan: Set[Cell],
    keep_clear: Set[Cell],
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    aggressive: bool = False,
) -> int:
    """Push idle AGVs off corridors needed by movers (maze choke-point recovery).

    Idle robots sitting in narrow corridors are the main reason k=1 serial
    pickup fails repeatedly at the same timestamp.

    ``aggressive=True`` also parks far-away idlers (incl. other carriers) so a
    single unloader can reach hub pads during delivery jams.
    """
    names = list(pose.keys())
    idle = [n for n in names if n not in movers]
    if not idle:
        return now
    # Prefer parking toward map corners, away from mover cells + keep_clear.
    corners = [(1, 1), (1, 20), (20, 1), (20, 20), (1, 10), (20, 10), (10, 1), (10, 20)]
    forbidden = set(keep_clear) | {pose[m][:2] for m in movers}
    # First pass: anyone sitting on keep_clear / mover cells must move.
    priority = sorted(
        idle,
        key=lambda n: (
            0 if pose[n][:2] in forbidden else 1,
            -min((_manh(pose[n][:2], c) for c in keep_clear), default=0),
            n,
        ),
    )
    for i, other in enumerate(priority):
        occ = {pose[n][:2] for n in names if n != other}
        # Target: farthest free corner-ish cell not forbidden/occupied.
        cands = [
            c
            for c in free
            if c not in forbidden and c not in occ and c != pose[other][:2]
        ]
        if not cands:
                    continue
        # Bias: near a corner, far from keep_clear centroid.
        if keep_clear:
            cx = sum(c[0] for c in keep_clear) / len(keep_clear)
            cy = sum(c[1] for c in keep_clear) / len(keep_clear)
        else:
            cx, cy = pose[other][0], pose[other][1]
        corner = corners[i % len(corners)]
        target = min(
        cands,
        key=lambda c: (
                _manh(c, corner),
                -abs(c[0] - cx) - abs(c[1] - cy),
            c,
        ),
    )
        # Only move if currently on/near critical cells or blocking tightly.
        # aggressive: always park idlers (needed when other carriers jam a hub).
        if (
            not aggressive
            and pose[other][:2] not in forbidden
            and _manh(pose[other][:2], target) > 8
        ):
            # Skip far-away idlers unless we need space (still nudge if on corridor
            # adjacent to keep_clear).
            adj = False
            ox, oy = pose[other][:2]
            for kx, ky in keep_clear:
                if abs(ox - kx) + abs(oy - ky) <= 2:
                    adj = True
                    break
            if not adj:
                continue
        now, ok = _serial_move_to(
            pose,
            steps_by,
            now,
            other,
            target,
            blocked_plan,
            loaded=loaded,
            dest=dest,
            tid=tid,
        )
        if ok:
            forbidden.add(pose[other][:2])
    return now


def _nearest_off_corridor(
    start: Cell,
    avoid: Set[Cell],
    walls: Set[Cell],
    taken: Set[Cell],
    free: Set[Cell],
) -> Optional[Cell]:
    """Closest walkable cell that is not on a mover corridor and not occupied."""
    q = deque([start])
    seen = {start}
    while q and len(seen) <= 80:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if nxt in seen:
                continue
            if not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                continue
            if nxt in walls or nxt not in free:
                continue
            seen.add(nxt)
            if nxt not in avoid and nxt not in taken:
                return nxt
            q.append(nxt)
    return None


def _yield_blockers_joint(
    pose: Dict[str, Pose],
    steps_by: Dict[str, List[dict]],
    now: int,
    *,
    movers: Set[str],
    goals: Dict[str, Cell],
    static: Set[Cell],
    free: Set[Cell],
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
) -> int:
    """Jointly park non-movers that sit on a mover's static path.

    Shrinking a wave used to freeze the dropped AGV on the remaining
    corridor, so the next plan — even k=1 — returned no path and the
    same carriers were retried forever. This steps those blockers off
    the corridor together. It does not walk them to their unload pads.
    """
    movers = {n for n in movers if n in pose and n in goals}
    if not movers:
        return now
    walls = set(static)
    corridor: Set[Cell] = set()
    for n in movers:
        path = _astar_cells(pose[n][:2], goals[n], walls)
        if path:
            corridor.update(path)
    if not corridor:
        return now
    blockers = [
        n
        for n in pose
        if n not in movers and pose[n][:2] in corridor
    ]
    if not blockers:
        return now
    blockers.sort(
        key=lambda n: (
            min(_manh(pose[n][:2], goals[m]) for m in movers),
            n,
        )
    )
    blockers = blockers[:4]
    names = list(pose.keys())
    taken = {pose[n][:2] for n in names if n not in blockers}
    taken |= {goals[n] for n in movers}
    goals_y = {n: pose[n][:2] for n in names}
    movers_y: Set[str] = set()
    for n in blockers:
        park = _nearest_off_corridor(
            pose[n][:2], corridor, walls, taken, free
        )
        if park is None or park == pose[n][:2]:
            continue
        goals_y[n] = park
        movers_y.add(n)
        taken.add(park)
    starts = {n: pose[n][:2] for n in names}
    sp = None
    if movers_y:
        sp = _prioritized_st_paths(
            starts, goals_y, movers_y, walls, hold_after=0
        )
        if sp is None or not _st_paths_ok(sp, movers_y, goals_y):
            sp = None
            movers_y = set()
    if not movers_y:
        worst = blockers[0]
        x, y = pose[worst][:2]
        hop = None
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if (
                nxt in free
                and nxt not in walls
                and nxt not in corridor
                and nxt not in taken
            ):
                hop = nxt
                break
        if hop is None:
            return now
        goals_y = {n: pose[n][:2] for n in names}
        goals_y[worst] = hop
        movers_y = {worst}
        sp = _prioritized_st_paths(
            starts, goals_y, movers_y, walls, hold_after=0
        )
        if sp is None or not _st_paths_ok(sp, movers_y, goals_y):
            return now
    timelines = _plan_paths_as_poses(pose, sp)
    if not _pose_movers_ok(pose, timelines, movers_y):
        return now
    now = _apply_pose_timelines(
        steps_by,
        pose,
        timelines,
        now,
        loaded=loaded,
        dest=dest,
        tid=tid,
        rise_ok={},
    )
    print(
        f"[PIPE] yield k={len(movers_y)} off-corridor @t={now}",
        flush=True,
    )
    return now


def _path_avoids_station_transit(
    path: List[Cell],
    goal: Cell,
    stations: Set[Cell],
) -> bool:
    """True if path never steps on a foreign station pad (start/goal OK)."""
    if not path or not stations:
        return True
    g = (int(goal[0]), int(goal[1]))
    start = path[0]
    for c in path:
        if c in stations and c != g and c != start:
            return False
    return True


def _paths_avoid_station_transit(
    paths: Optional[Dict[str, List[Cell]]],
    movers: Set[str],
    goals: Dict[str, Cell],
    stations: Set[Cell],
) -> bool:
    if paths is None:
        return False
    if not stations:
        return True
    for n in movers:
        seq = list(paths.get(n) or [])
        if not seq:
            return False
        g = goals.get(n, seq[-1])
        if not _path_avoids_station_transit(seq, g, stations):
            return False
    return True


def _pick_park_target(
    who: str,
    pose: Dict[str, Pose],
    free: Set[Cell],
    forbidden: Set[Cell],
    *,
    corner_idx: int = 0,
) -> Optional[Cell]:
    """Nearest free corner-biased parking cell for an evacuating idle AGV."""
    corners = [
        (1, 1),
        (1, 20),
        (20, 1),
        (20, 20),
        (1, 10),
        (20, 10),
        (10, 1),
        (10, 20),
    ]
    occ = {pose[n][:2] for n in pose if n != who}
    cands = [
        c
        for c in free
        if c not in forbidden and c not in occ and c != pose[who][:2]
    ]
    if not cands:
        return None
    corner = corners[int(corner_idx) % len(corners)]
    return min(cands, key=lambda c: (_manh(c, corner), c))


def _joint_wave_with_evac(
    starts: Dict[str, Cell],
    goals: Dict[str, Cell],
    wave_movers: Set[str],
    blocked_plan: Set[Cell],
    pose: Dict[str, Pose],
    free: Set[Cell],
    keep_clear: Set[Cell],
    names: List[str],
    *,
    n_evac: int = 1,
    stations: Optional[Set[Cell]] = None,
) -> Optional[Tuple[Dict[str, List[Cell]], Set[str], Dict[str, Cell]]]:
    """On batch fail: park ≤n_evac idlers while jointly ST-planning the wave.

    Returns ``(paths, all_movers, goals)`` or None. Does not mutate pose.
    """
    wave_movers = set(wave_movers)
    if not wave_movers:
        return None
    g2 = {n: starts.get(n, pose[n][:2]) for n in names}
    for a, g in goals.items():
        g2[a] = g
    movers = set(wave_movers)
    forbidden = set(keep_clear) | {pose[m][:2] for m in wave_movers}
    for a in wave_movers:
        forbidden.add(g2[a])
    idle = [n for n in names if n not in wave_movers]
    idle.sort(
        key=lambda n: (
            0 if pose[n][:2] in forbidden else 1,
            min((_manh(pose[n][:2], c) for c in keep_clear), default=99),
            n,
        )
    )
    evac_n = 0
    need = max(1, int(n_evac))
    for i, other in enumerate(idle):
        if evac_n >= need:
            break
        on_keep = pose[other][:2] in forbidden
        adj = False
        if not on_keep and keep_clear:
            ox, oy = pose[other][:2]
            for kx, ky in keep_clear:
                if abs(ox - kx) + abs(oy - ky) <= 2:
                    adj = True
                    break
        # First pass: only blockers; if none, still take the top-ranked idle.
        if not on_keep and not adj and evac_n == 0:
            has_blocker = any(
                pose[x][:2] in forbidden
                or (
                    keep_clear
                    and min((_manh(pose[x][:2], c) for c in keep_clear), default=99)
                    <= 2
                )
                for x in idle
            )
            if has_blocker:
                continue
        park = _pick_park_target(other, pose, free, forbidden, corner_idx=i)
        if park is None:
            continue
        g2[other] = park
        movers.add(other)
        forbidden.add(park)
        evac_n += 1
    if evac_n == 0:
        return None
    s2 = {n: starts.get(n, pose[n][:2]) for n in names}
    paths = _prioritized_st_paths(s2, g2, movers, blocked_plan, hold_after=0)
    if paths is None or not _st_paths_ok(paths, movers, g2):
        return None
    stn = stations or set()
    if stn and not _paths_avoid_station_transit(paths, movers, g2, stn):
        return None
    return paths, movers, g2


def _ecbs(
    grid: GridMap,
    starts_wh: Dict[str, Cell],
    goals_wh: Dict[str, Cell],
    *,
    weight: float,
    time_limit: float,
    max_expansions: int,
    movers: Optional[Set[str]] = None,
    plan: str = "joint",
) -> Optional[Dict[str, List[Cell]]]:
    movers = set(movers) if movers is not None else set(starts_wh)
    agents = []
    for name in sorted(starts_wh):
        sr, sc = _wh_to_rc(*starts_wh[name])
        gr, gc = _wh_to_rc(*goals_wh[name])
        if not grid.is_free((sr, sc)):
            return None
        # Stay-put goals may sit on station for idle; allow goal free-check only when moving
        if name in movers and not grid.is_free((gr, gc)):
            return None
        agents.append(Agent(name=name, start=(sr, sc), goal=(gr, gc)))
    problem = MAPFProblem(grid, agents)
    algo = "wcbs" if plan in ("joint", "eecbs", "wcbs", "ecbs") else "prioritized"
    try:
        sol = solve(
            problem,
            algo,
            weight=float(weight),
            time_limit=float(time_limit),
            max_expansions=int(max_expansions),
        )
    except Exception:
        return None
    if sol is None or not sol.paths:
        return None
    out: Dict[str, List[Cell]] = {}
    for name, path_rc in sol.paths.items():
        out[name] = [_rc_to_wh(r, c) for r, c in path_rc]
    return out


def _cell_path_to_pose_seq(start: Pose, cells: List[Cell]) -> List[Pose]:
    """Expand a cell path into turn-safe pose timeline (no move+turn same step)."""
    if not cells:
        return []
    seq: List[Pose] = []
    x, y, pitch = int(start[0]), int(start[1]), int(start[2]) % 360
    # Align to first cell if needed
    if cells[0] != (x, y):
        cells = [(x, y)] + list(cells)
    for i in range(1, len(cells)):
        nx, ny = cells[i]
        if (nx, ny) == (x, y):
            seq.append((x, y, pitch))
            continue
        need = _pitch_from_delta(nx - x, ny - y, pitch)
        if need != pitch:
            for p in _turn_pitch_chain(pitch, need):
                pitch = p
                seq.append((x, y, pitch))
        seq.append((nx, ny, pitch))
        x, y = nx, ny
    return seq


def _apply_pose_timelines(
    steps_by: Dict[str, List[dict]],
    pose: Dict[str, Pose],
    timelines: Dict[str, List[Pose]],
    now: int,
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    pad_unload: Optional[Dict[str, Set[Cell]]] = None,
    on_unload: Optional[Any] = None,
    rise_ok: Optional[Dict[str, Set[Cell]]] = None,
    pickup_leave: Optional[Set[Cell]] = None,
) -> int:
    """Apply pose timelines. Optional mid-timeline unload on true pads.

    ``pad_unload`` maps AGV → true unload pads. When a loaded AGV first steps
    onto a pad, ``on_unload(agv)`` runs before the step is recorded so
    subsequent vacate holds are cargo-free (shared-pad time multiplex).

    ``rise_ok``: False→True only on listed cells (pickup commit). ``None`` or
    ``{}`` both suppress rising — invent/heal must restore traj before apply.
    ``pickup_leave``: hard whitelist for any False→True (r106 hub phantoms).
    """
    _restore_sticky_cargo(loaded, dest, tid)
    names = list(pose.keys())
    T = max((len(timelines.get(n) or []) for n in names), default=0)
    if T <= 0:
        return now
    # A finished timeline stays on its last cell for the rest of the slice.
    # Stop before a later step would land on that cell or swap with it.
    prev_hold: Dict[str, Cell] = {n: pose[n][:2] for n in names}
    for k in range(T):
        cur_c: Dict[str, Cell] = {}
        for n in names:
            seq = timelines.get(n) or []
            if k < len(seq):
                cur_c[n] = (int(seq[k][0]), int(seq[k][1]))
            else:
                cur_c[n] = prev_hold[n]
        occ: Dict[Cell, str] = {}
        bad = False
        for n, c in cur_c.items():
            if c in occ:
                bad = True
                break
            occ[c] = n
        if not bad:
            for n, c in cur_c.items():
                p = prev_hold[n]
                if c == p:
                    continue
                for m, mc in cur_c.items():
                    if m != n and mc == p and prev_hold[m] == c:
                        bad = True
                        break
                if bad:
                    break
        if bad:
            T = k
            break
        prev_hold = cur_c
    if T <= 0:
        return now
    leave = _PICKUP_LEAVE if pickup_leave is None else pickup_leave
    unloaded: Set[str] = set()
    prev_cell: Dict[str, Cell] = {n: pose[n][:2] for n in names}
    for k in range(T):
        t = now + k + 1
        for n in names:
            seq = timelines.get(n) or []
            if k < len(seq):
                pose[n] = seq[k]
            if (
                pad_unload is not None
                and on_unload is not None
                and n not in unloaded
                and n in pad_unload
                and bool(loaded.get(n, False))
                and pose[n][:2] in pad_unload[n]
            ):
                # Unload on dwell (2nd consecutive tick on pad), not on move-in.
                if prev_cell.get(n) == pose[n][:2]:
                    on_unload(n)
                    unloaded.add(n)
            ld = bool(loaded.get(n, False))
            d_s = dest.get(n, "") if ld else ""
            t_s = tid.get(n, "") if ld else ""
            prev = steps_by[n][-1] if steps_by.get(n) else None
            prev_ld = bool(
                prev
                and str(prev.get("loaded", "")).lower()
                in ("true", "1", "yes")
            )
            prev_tid = str(prev.get("task-id") or "").strip() if prev else ""
            if prev_ld and prev_tid and not ld:
                cell = pose[n][:2]
                pads = _TID_DROPOFFS.get(prev_tid) or set()
                if cell not in pads:
                    ld = True
                    d_s = str(prev.get("destination") or "")
                    t_s = prev_tid
                    loaded[n] = True
                    dest[n] = d_s
                    tid[n] = t_s
            if ld:
                if not prev_ld:
                    cell = pose[n][:2]
                    cur_pitch = int(pose[n][2]) % 360
                    last_cell = (
                        (int(prev["X"]), int(prev["Y"])) if prev else None
                    )
                    prev_pitch = (
                        int(prev.get("pitch") or 0) % 360 if prev else None
                    )
                    stayed = last_cell == cell and prev_pitch == cur_pitch
                    allowed = rise_ok.get(n) if rise_ok is not None else None
                    if (
                        not stayed
                        or not allowed
                        or cell not in allowed
                        or (leave and cell not in leave)
                    ):
                        ld = False
                        d_s, t_s = "", ""
            steps_by[n].append(
                _hold(
                    n,
                    pose[n],
                    t,
                    loaded=ld,
                    dest=d_s,
                    tid=t_s,
                )
            )
            prev_cell[n] = pose[n][:2]
    return now + T


def _apply_timelines_until_first_goal(
    steps_by: Dict[str, List[dict]],
    pose: Dict[str, Pose],
    timelines: Dict[str, List[Pose]],
    goals: Dict[str, Cell],
    movers: Set[str],
    now: int,
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    pad_unload: Optional[Dict[str, Set[Cell]]] = None,
    on_unload: Optional[Any] = None,
    rise_ok: Optional[Dict[str, Set[Cell]]] = None,
    pickup_leave: Optional[Set[Cell]] = None,
    min_steps: int = 0,
) -> int:
    """Play a joint plan only until the first mover reaches its goal.

    The 2026-09-02 SH03 pipeline did this: the clock does not wait out the
    longest path. Whoever arrives is committed on the next loop; everyone
    else is replanned from where they stopped.
    """
    _restore_sticky_cargo(loaded, dest, tid)
    arrive: Dict[str, int] = {}
    for n in movers:
        seq = timelines.get(n) or []
        g = goals.get(n)
        if g is None:
            continue
        if pose[n][:2] == g:
            arrive[n] = 0
            continue
        for i, p in enumerate(seq):
            if (int(p[0]), int(p[1])) == g:
                arrive[n] = i + 1
                break
    if not arrive:
        return _apply_pose_timelines(
            steps_by,
            pose,
            timelines,
            now,
            loaded=loaded,
            dest=dest,
            tid=tid,
            pad_unload=pad_unload,
            on_unload=on_unload,
            rise_ok=rise_ok,
            pickup_leave=pickup_leave,
        )
    positive = [t for t in arrive.values() if t > 0]
    floor_steps = max(0, int(min_steps))
    if not positive and floor_steps <= 0:
        return now
    t_cut = min(positive) if positive else floor_steps
    if floor_steps > 0:
        t_cut = max(t_cut, floor_steps)
    cap = max((len(seq) for seq in timelines.values()), default=0)
    if cap <= 0:
        return now
    t_cut = min(t_cut, cap)
    trimmed: Dict[str, List[Pose]] = {}
    for n in list(pose.keys()):
        seq = list(timelines.get(n) or [])
        if not seq:
            trimmed[n] = []
            continue
        trimmed[n] = seq[:t_cut]
        while len(trimmed[n]) < t_cut:
            trimmed[n].append(trimmed[n][-1] if trimmed[n] else pose[n])
    return _apply_pose_timelines(
        steps_by,
        pose,
        trimmed,
        now,
        loaded=loaded,
        dest=dest,
        tid=tid,
        pad_unload=pad_unload,
        on_unload=on_unload,
        rise_ok=rise_ok,
        pickup_leave=pickup_leave,
    )


def _vacate_paths_after_goal(
    paths: Dict[str, List[Cell]],
    goals: Dict[str, Cell],
    movers: Set[str],
    free: Set[Cell],
    blocked: Set[Cell],
) -> Dict[str, List[Cell]]:
    """After first goal touch, rewrite holds to a park neighbor (free the pad)."""
    out: Dict[str, List[Cell]] = {n: list(p) for n, p in paths.items()}
    for n in movers:
        g = goals.get(n)
        seq = out.get(n) or []
        if g is None or not seq:
            continue
        arr = None
        for i, c in enumerate(seq):
            if c == g:
                arr = i
                break
        if arr is None:
            continue
        park = _nearby_park(g, free, set(blocked) | {g}) or g
        for j in range(arr + 1, len(seq)):
            seq[j] = park
        out[n] = seq
    return out


def _plan_paths_as_poses(
    pose: Dict[str, Pose],
    paths: Dict[str, List[Cell]],
) -> Dict[str, List[Pose]]:
    """Expand cell paths on a shared clock, facing the way each step moves.

    A move tick must not also change pitch. Before each cell step the whole
    fleet waits while anyone who is about to move turns in place by at most
    90°. Everyone else holds still, so the following move is the same
    simultaneous cell step the joint plan already checked.
    """
    names = list(pose.keys())
    cell_paths: Dict[str, List[Cell]] = {}
    for n in names:
        cells = list(paths.get(n) or [pose[n][:2]])
        if not cells:
            cells = [pose[n][:2]]
        if cells[0] != pose[n][:2]:
            cells = [pose[n][:2]] + cells
        # Drop redundant terminal holds for expansion; re-pad after.
        while len(cells) > 1 and cells[-1] == cells[-2]:
            cells.pop()
        cell_paths[n] = cells

    movers = [n for n in names if len(cell_paths[n]) > 1]
    if not movers:
        return {n: [] for n in names}

    # Do not pad a car back onto its goal after it has arrived. Shared pads
    # are free on the next tick; holding the arrival made the pose check
    # reject a follow that the cell plan already allowed, and the wave froze.
    T = max(len(cell_paths[n]) for n in movers)
    cur: Dict[str, Pose] = {n: pose[n] for n in names}
    out: Dict[str, List[Pose]] = {n: [] for n in names}
    for i in range(T - 1):
        active = [n for n in movers if i + 1 < len(cell_paths[n])]
        if not active:
            break
        nxt_cell = {n: cell_paths[n][i + 1] for n in active}
        # 180° is two legal 90° ticks. A fourth pass means the chain stalled.
        for _spin in range(4):
            turning: Dict[str, int] = {}
            for n in active:
                x, y, pitch = cur[n]
                nx, ny = nxt_cell[n]
                if (nx, ny) == (x, y):
                    continue
                need = _pitch_from_delta(nx - x, ny - y, pitch)
                chain = _turn_pitch_chain(pitch, need)
                if chain:
                    turning[n] = chain[0]
            if not turning:
                break
            for n in active:
                x, y, pitch = cur[n]
                cur[n] = (x, y, turning.get(n, pitch))
                out[n].append(cur[n])
        for n in active:
            x, y, pitch = cur[n]
            nx, ny = nxt_cell[n]
            cur[n] = (int(nx), int(ny), int(pitch) % 360)
            out[n].append(cur[n])
    return out


def _station_key(task: dict) -> str:
    tid = str(task.get("task_id") or "")
    if "-" in tid:
        return tid.rsplit("-", 1)[0]
    return str(task.get("destination") or "UNK")


def _requeue(
    queues: Dict[str, list],
    order: List[str],
    task: dict,
    *,
    picked_ids: Optional[Set[str]] = None,
) -> bool:
    """Requeue a *not-yet-picked* task. Returns False if blocked (already picked)."""
    tid = str(task.get("task_id") or "").strip()
    if tid and picked_ids is not None and tid in picked_ids:
        print(
            f"[ECBS] block requeue of already-picked {tid} (FIFO guard)",
            flush=True,
        )
        return False
    st = _station_key(task)
    q = queues.setdefault(st, [])
    # Avoid duplicate inserts when handoff left the tid as queue head.
    for i, trow in enumerate(list(q)):
        if str(trow.get("task_id") or "") == tid:
            if i != 0:
                q.insert(0, q.pop(i))
            if st not in order:
                order.append(st)
            return True
    q.insert(0, task)
    if st not in order:
        order.append(st)
    return True


def _dense_fill(steps_by: Dict[str, List[dict]], now: int) -> None:
    for name in list(steps_by):
        by = {int(s["timestamp"]): s for s in steps_by[name]}
        if not by:
            continue
        last = by[min(by)]
        filled = []
        for t in range(0, now + 1):
            if t in by:
                last = by[t]
                filled.append(dict(last))
        else:
                r = dict(last)
                r["timestamp"] = t
                filled.append(r)
        steps_by[name] = filled


def solve_ecbs(
    slot: Optional[int] = None,
    *,
    meta: Optional[dict] = None,
    max_tasks: int = 0,
    max_active: int = 0,
    weight: float = 1.5,
    time_limit: float = 35.0,
    max_expansions: int = 300_000,
    plan: str = "joint",
    allow_solo_fallback: bool = False,
    initial_park: bool = True,
    deliver_batch: int = 0,
    turn_aware: bool = True,
    pipeline: bool = True,
    task_csv: Optional[Path] = None,
    use_swapnet: bool = False,
    swapnet_margin: float = 4.0,
    use_hierarchical: bool = False,
    hierarchical_force: bool = False,
    hierarchical_threshold: float = 0.08,
    use_wavenet: bool = False,
    wave_hard_cap: int = 4,
    use_traffic_accel: bool = False,
    use_lane_rules: bool = False,
    task_phase_by_id: Optional[Dict[str, str]] = None,
    plan_horizon: int = 24,
    exec_horizon: int = 12,
    allow_mode_switch: Optional[bool] = None,
    joint_core: Optional[str] = None,
    legacy_wave_ecbs: bool = False,
    traffic_recovery: bool = True,
) -> dict:
    # legacy unused flags kept for CLI compatibility
    del use_traffic_accel, use_lane_rules, deliver_batch
    del initial_park  # reserved; open-start is fine for smoke

    OUT.mkdir(parents=True, exist_ok=True)
    TRAJ.mkdir(parents=True, exist_ok=True)

    phase_map = dict(task_phase_by_id or {})
    plan_horizon = int(plan_horizon)
    exec_horizon = int(exec_horizon)
    joint_core_arg = joint_core  # resolve after meta load

    if meta is None:
        if slot is None:
            slot = 3
        meta = load_custom_meta(int(slot), max_sim_time=300000, wall_timeout=1e9)
        assert meta, f"slot {slot} meta missing"
    if slot is None:
        slot = int(meta.get("slot") or 0)

    _jc = str(
        joint_core_arg
        if joint_core_arg is not None
        else meta.get("joint_core") or DEFAULT_JOINT_CORE
    ).strip().lower()
    if _jc not in ("prioritized", "ecbs"):
        _jc = "prioritized"
    joint_core_mode = _jc

    tcsv = Path(task_csv) if task_csv else Path(meta["task_csv"])
    mod, env, agv_states, task_states, _ = load_scenario(
        tcsv, Path(meta["position_csv"]), force_reload=True
    )
    if meta.get("extra_obstacles"):
        patch_extra_obstacles(env, list(meta["extra_obstacles"]))

    queues = {n: [dict(t) for t in q] for n, q in task_states.items() if q}
    if max_tasks > 0:
        kept, nq = 0, {k: [] for k in queues}
        while kept < max_tasks:
            prog = False
            for n in list(queues):
                if queues[n] and kept < max_tasks:
                    nq[n].append(queues[n].pop(0))
                    kept += 1
                    prog = True
            if not prog:
                break
        queues = {k: v for k, v in nq.items() if v}

    # Pickup goals are the cell beside the station, not the station itself.
    # The station is a dead-end; occupying it blocks the only exit (r115).
    true_pickups: Dict[str, Cell] = {}
    try:
        sp_map, _ep_ignore, _agv_ignore = mod.get_object_position(
            str(meta["position_csv"])
        )
        true_pickups = {
            str(k): (int(v[0]), int(v[1])) for k, v in (sp_map or {}).items()
        }
    except Exception:
        true_pickups = {}
    map_json_early = meta.get("map_json")
    if map_json_early and Path(map_json_early).exists():
        try:
            _j_early = json.loads(Path(map_json_early).read_text(encoding="utf-8"))
            for p in _j_early.get("pickups") or []:
                nm = str(p.get("name") or "")
                if nm:
                    true_pickups[nm] = (int(p["x"]), int(p["y"]))
        except Exception:
            pass
    global _PICKUP_LEAVE
    _PICKUP_LEAVE = set()
    for _px, _py in true_pickups.values():
        for _dx, _dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            _PICKUP_LEAVE.add((int(_px) + _dx, int(_py) + _dy))
    for station, q in queues.items():
        pad = true_pickups.get(str(station))
        if pad is None:
            continue
        door = _pickup_approach(pad)
        for t in q:
            t["pickup_point"] = door
            t["true_pickup_point"] = door
            t["start_point"] = str(station)

    def _restore_task_pads(
        qmap: Dict[str, List[dict]],
        *pending_bags: Dict[str, dict],
    ) -> None:
        for station, q in qmap.items():
            pad = true_pickups.get(str(station))
            if not pad:
                continue
            door = _pickup_approach(pad)
            for t in q:
                t["pickup_point"] = door
                t["true_pickup_point"] = door
                t["start_point"] = str(station)
        for bag in pending_bags:
            for _agv, task in bag.items():
                tid_s = str(task.get("task_id") or "")
                prefix = tid_s.rsplit("-", 1)[0].strip() if "-" in tid_s else ""
                st = ""
                for cand in (
                    prefix,
                    str(task.get("pickup_name") or "").strip(),
                    str(task.get("start_point") or "").strip(),
                ):
                    if cand and cand in true_pickups:
                        st = cand
                        break
                if not st:
                    st = prefix
                pad = true_pickups.get(st)
                if not pad:
                    continue
                door = _pickup_approach(pad)
                task["pickup_point"] = door
                task["true_pickup_point"] = door
                task["start_point"] = st

    total = sum(len(v) for v in queues.values())

    # ---- Default stack: M0 + SwapNet + rule park recovery ----
    # SceneDifficulty / wave ECBS only when explicitly requested.
    legacy_wave = bool(legacy_wave_ecbs) or bool(meta.get("legacy_wave_ecbs", False))
    if not legacy_wave:
        t0_m0 = time.perf_counter()
        eng_csv = tcsv
        if max_tasks > 0 or str(tcsv.resolve()) != str(Path(meta["task_csv"]).resolve()):
            eng_csv = OUT / f"{meta['id']}_m0_traffic_tasks.csv"
            n_w = _write_queue_task_csv(queues, eng_csv)
            if n_w <= 0:
                n_w = total
        else:
            n_w = total
        do_traffic = bool(meta.get("traffic_recovery", traffic_recovery))
        do_swap = True if use_swapnet or do_traffic else bool(use_swapnet)
        if do_traffic:
            do_swap = True
        hier_stats_m0: dict = {
            "waves_escalated": 0,
            "waves_passthrough": 0,
            "waves_astar": 0,
            "waves_ecbs": 0,
            "waves_hybrid": 0,
            "hybrid_ecbs_agents": 0,
            "hybrid_astar_agents": 0,
            "sticky_astar_agents": 0,
            "affinity_clears": 0,
            "accel_waves": 0,
            "last_reason": "m0_traffic_default",
            "last_planner": "m0_engine",
            "hardness": 0.0,
            "joint_core": "none",
            "legacy_wave_ecbs": False,
        }
        print(
            f"[SOLVE] default M0+SwapNet+RulePark tasks={n_w} "
            f"(set legacy_wave_ecbs=1 for old wave/ECBS path)",
            flush=True,
        )
        return _solve_via_m0_engine(
            meta=meta,
            task_csv=eng_csv,
            total=int(n_w),
            use_swapnet=bool(do_swap),
            swapnet_margin=float(swapnet_margin),
            hier_stats=hier_stats_m0,
            t0=t0_m0,
            plan=plan,
            max_active=max_active if int(max_active) > 0 else 0,
            weight=weight,
            pipeline=pipeline,
            turn_aware=turn_aware,
            plan_horizon=plan_horizon,
            exec_horizon=exec_horizon,
            use_hierarchical=False,
            use_wavenet=False,
            gate=None,
            allow_yield=False,
            traffic_recovery=bool(do_traffic),
        )

    static_list = list(env.get_static_obstacles() or [])
    static = {(int(p[0]), int(p[1])) for p in static_list}
    map_json = meta.get("map_json")
    stations: Set[Cell] = set()
    global _MAP_STATIONS, _MAP_DROPOFFS, _MAP_PICKUPS, _DROP_NAME, _TID_DROPOFFS
    _MAP_PICKUPS = set()
    _MAP_DROPOFFS = set()
    _TID_DROPOFFS = {}
    _DROP_NAME = {}
    if map_json and Path(map_json).exists():
        j = json.loads(Path(map_json).read_text(encoding="utf-8"))
        stations = {
                (int(p["x"]), int(p["y"]))
                for p in (j.get("pickups") or []) + (j.get("dropoffs") or [])
            }
        for p in j.get("pickups") or []:
            _MAP_PICKUPS.add((int(p["x"]), int(p["y"])))
        for p in j.get("dropoffs") or []:
            cell = (int(p["x"]), int(p["y"]))
            _MAP_DROPOFFS.add(cell)
            if p.get("name"):
                _DROP_NAME[cell] = str(p["name"])
    _MAP_STATIONS = set(stations)
    _STICKY.clear()
    _PICKUP_SKIP.clear()
    try:
        from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
            _load_task_dropoffs as _ltd,
        )

        _TID_DROPOFFS = {str(k): set(v) for k, v in _ltd(meta).items()}
    except Exception:  # noqa: BLE001
        _TID_DROPOFFS = {}
    # ENV marks pickup/dropoff pads as static obstacles. Strip them so true
    # pads are walkable goals (r45: all 6 pickups were in static → snap forever).
    # Transit: stations stay in blocked_plan so AGVs cannot cut through foreign
    # pads (r90: "撞取货台"). Goal-on-blocked is allowed by A*/ST (see _astar_cells).
    walls = set(static) - set(stations)
    static = set(walls)
    no_park: Set[Cell] = set(walls) | set(stations)
    blocked_plan = set(walls) | set(stations)
    free = {
        (x, y)
        for x in range(1, 21)
        for y in range(1, 21)
        if (x, y) not in no_park
    }
    # ECBS grid stays walls-only (pymapf neighbors never enter obstacles; pad
    # goals would be unreachable if stations were blocked on the grid).
    grid = _build_grid(walls, stations)

    pose: Dict[str, Pose] = {}
    for name, agv in agv_states.items():
        st = agv["state"] if isinstance(agv, dict) else agv.state
        pose[name] = (int(st[0]), int(st[1]), int(st[3]) if len(st) > 3 else 90)

    names = sorted(pose)
    # 0 / negative => no artificial cap: use all AGVs in this scene.
    # Explicit positive values are still clamped to fleet size.
    if int(max_active) <= 0:
        max_active = len(names)
    else:
        max_active = max(1, min(int(max_active), len(names)))
    steps_by = {n: [_hold(n, pose[n], 0)] for n in names}
    now = 0
    done = 0
    done_seen: Set[str] = set()
    failed: List[str] = []
    failed_seen: Set[str] = set()

    def _mark_failed(tid_s: str) -> None:
        tid_s = str(tid_s or "").strip()
        if not tid_s or tid_s in failed_seen:
            return
        failed_seen.add(tid_s)
        failed.append(tid_s)
        _clear_inflight_tid(tid_s)

    def _count_done(tid_s: str) -> None:
        """Count each task_id at most once (requeue must not inflate cr>1)."""
        nonlocal done
        tid_s = str(tid_s or "").strip()
        if not tid_s or tid_s in done_seen:
            return
        done_seen.add(tid_s)
        done += 1

    def _wcbs_paths(
        starts_g: Dict[str, Cell],
        goals_g: Dict[str, Cell],
        movers_g,
        *,
        tl: Optional[float] = None,
    ) -> Optional[Dict[str, List[Cell]]]:
        """Medium/hard joint route: Prioritized by default; ECBS if joint_core=ecbs."""
        budget = float(
            tl if tl is not None else min(2.5, max(1.0, float(time_limit) * 0.1))
        )
        mv = set(movers_g)
        paths = _plan_joint_core_paths(
                grid,
            starts_g,
            goals_g,
            movers=mv,
            static=blocked_plan,
            joint_core=_effective_joint_core(),
            joint_plan="joint",
                weight=weight,
            time_limit=budget,
                max_expansions=max_expansions,
            plan_horizon=plan_horizon,
            exec_horizon=exec_horizon,
        )
        # ECBS grid is walls-only; reject any pad cut-through (r90).
        if (
            paths is not None
            and stations
            and not _paths_avoid_station_transit(paths, mv, goals_g, stations)
        ):
            return None
        return paths

    t0 = time.perf_counter()
    order = sorted(queues.keys())
    rr = 0
    work_count = {n: 0 for n in names}
    recent_joint_fail = 0
    delivery_miss_count: Dict[str, int] = {}
    # AGVs that already picked up but missed unload — keep cargo, retry deliver.
    # Never requeue-to-pickup after a rising loaded edge (FIFO/display forbids it).
    pending_delivery: Dict[str, dict] = {}
    # Assigned but not yet physically on pickup pad — keep going, no loaded edge yet.
    pending_pickup: Dict[str, dict] = {}
    loaded: Dict[str, bool] = {n: False for n in names}
    dest: Dict[str, str] = {n: "" for n in names}
    tid: Dict[str, str] = {n: "" for n in names}
    # Sticky planner affinity until unload: "astar" | "joint"
    agv_affinity: Dict[str, str] = {}
    # Per-station in-flight queue order (assign-time promote, pickup-time FIFO).
    inflight_station: Dict[str, List[str]] = {}
    picked_ids: Set[str] = set()
    resume_path = str(meta.get("resume_traj") or "").strip()
    if resume_path:
        resume_steps, now, resume_done, resume_last = _load_resume_trajectory(
            Path(resume_path)
        )
        done_seen.update(resume_done)
        done = len(done_seen)
        by_id: Dict[str, dict] = {}
        for q in queues.values():
            for task in q:
                tid_q = str(task.get("task_id") or "").strip()
                if tid_q:
                    by_id[tid_q] = task
        drop_ids = set(resume_done)
        for name, rec in resume_last.items():
            if name in pose:
                pose[name] = (
                    int(rec["X"]),
                    int(rec["Y"]),
                    int(rec["pitch"]) % 360,
                )
            tid_c = str(rec.get("task-id") or "").strip()
            if str(rec.get("loaded") or "") == "TRUE" and tid_c:
                task_c = by_id.get(tid_c) or {
                    "task_id": tid_c,
                    "destination": str(rec.get("destination") or ""),
                    "end_points": [],
                }
                pending_delivery[name] = dict(task_c)
                drop_ids.add(tid_c)
                picked_ids.add(tid_c)
        for st in list(queues):
            kept = [
                t
                for t in queues[st]
                if str(t.get("task_id") or "").strip() not in drop_ids
            ]
            if kept:
                queues[st] = kept
            else:
                queues.pop(st, None)
        order = sorted(queues.keys())
        steps_by = resume_steps
        for name in names:
            if name not in steps_by or not steps_by[name]:
                steps_by[name] = [_hold(name, pose[name], now)]
        print(
            f"[ECBS] resume t={now} done={done}/{total} "
            f"carry={len(pending_delivery)} "
            f"from={resume_path}",
            flush=True,
        )
    # Guard against infinite requeue when every pickup wave fails at the same t.
    pickup_stall_streak = 0
    # Cell-conflict → pending_pickup → replan same set (wall-clock spin).
    pickup_conflict_spin = 0
    # Soft claim cap after conflicts (0 = use full dest-pad budget).
    pickup_soft_claim = 0
    # Waves to skip new claims after a failed conflict recover (disperse only).
    pickup_claim_cooldown = 0
    # Consecutive force-unload misses before dropping a carrier (r26 thrash).
    force_unload_miss_streak = 0
    # AGVs that failed spin-corridor to the same pocket — skip on reclaim.
    pickup_spin_ban: Set[str] = set()
    # P0 anti-spin: same sim-t / done with no progress
    stall_t = -1
    stall_done = -1
    no_progress_waves = 0
    done_stall_waves = 0
    serial_recover_budget = int(meta.get("serial_recover_budget") or 6)
    serial_recover_used = 0
    joint_core_override: Optional[str] = None
    # Handback to true M0 after sustained calm (anti ping-pong cooldown).
    handback_calm_need = int(meta.get("handback_calm") or 5)
    handback_cooldown_until = 0.0
    easy_calm_waves = 0
    m0_handback_done = False
    allow_switch_runtime = bool(
        allow_mode_switch
        if allow_mode_switch is not None
        else meta.get("allow_mode_switch", False)
    )

    def _effective_joint_core() -> str:
        """Prioritized by default; escalate to ECBS after repeated joint fails."""
        if joint_core_override:
            return str(joint_core_override)
        if joint_core_mode == "prioritized" and int(recent_joint_fail) >= 6:
            return "ecbs"
        return joint_core_mode

    def _force_wait_tick(tag: str = "stall") -> None:
        """Advance sim by 1 wait tick for all AGVs (break spacetime livelock)."""
        nonlocal now
        now += 1
        for n in names:
            x, y, pitch = pose[n]
            steps_by.setdefault(n, [])
            prev = steps_by[n][-1] if steps_by[n] else {}
            steps_by[n].append(
                {
                    "timestamp": now,
                    "name": n,
                    "X": x,
                    "Y": y,
                    "pitch": pitch,
                    "loaded": str(prev.get("loaded", "false")).lower(),
                    "destination": prev.get("destination", ""),
                    "Emergency": str(prev.get("Emergency", "false")).lower(),
                    "task-id": prev.get("task-id", ""),
                }
            )
        print(f"[RECOVER] force wait tick t={now} ({tag})", flush=True)

    def _task_station(task: dict) -> str:
        return _station_name_of(task, station_pads=true_pickups) or "UNK"

    def _pk(
        task: dict,
        start: Cell,
        free_c: Optional[Set[Cell]] = None,
        static_c: Optional[Set[Cell]] = None,
    ) -> Cell:
        """Pickup goal pinned to map station pads (r71)."""
        return _pickup_goal(
            task,
            start,
            free if free_c is None else free_c,
            static if static_c is None else static_c,
            station_pads=true_pickups,
        )

    def _clear_inflight_tid(tid_s: str) -> None:
        tid_s = str(tid_s or "").strip()
        if not tid_s:
            return
        for st, q in list(inflight_station.items()):
            if tid_s not in q:
                    continue
            nq = [x for x in q if x != tid_s]
            if nq:
                inflight_station[st] = nq
            else:
                inflight_station.pop(st, None)

    def _requeue_task(task: dict) -> bool:
        """Requeue task and any later same-station in-flight (FIFO-safe)."""
        tid_s = str(task.get("task_id") or "").strip()
        st = _task_station(task)
        q_inf = list(inflight_station.get(st) or [])
        later: List[str] = []
        if tid_s and tid_s in q_inf:
            later = q_inf[q_inf.index(tid_s) :]
        elif tid_s:
            later = [tid_s]

        # Pull later in-flight AGVs back to queue (reverse so order preserved)
        pull: List[tuple] = []
        for bag in (pending_pickup,):
            for agv, t2 in list(bag.items()):
                t2id = str(t2.get("task_id") or "")
                if t2id in later:
                    pull.append((agv, t2, "pending_pickup"))
        # assigned is local in the wave loop — handled by caller via return list
        ok_any = False
        for t2id in reversed(later):
            # find task body
            body = None
            if tid_s == t2id:
                body = task
            for agv, t2, _src in pull:
                if str(t2.get("task_id") or "") == t2id:
                    body = t2
                    pending_pickup.pop(agv, None)
                    break
            if body is None:
                _clear_inflight_tid(t2id)
                continue
            ok = _requeue(queues, order, body, picked_ids=picked_ids)
            if ok:
                ok_any = True
                _clear_inflight_tid(t2id)
        if not later:
            ok = _requeue(queues, order, task, picked_ids=picked_ids)
            if ok and tid_s:
                _clear_inflight_tid(tid_s)
            return ok
        return ok_any

    def _repair_inflight(extra_held: Optional[Set[str]] = None) -> None:
        """Drop orphan in-flight tids (requeued/failed without clearing)."""
        held: Set[str] = set(extra_held or [])
        for task in list(pending_pickup.values()) + list(pending_delivery.values()):
            tid_s = str(task.get("task_id") or "").strip()
            if tid_s:
                held.add(tid_s)
        for st, q in list(inflight_station.items()):
            nq = [t for t in q if t in held]
            if nq:
                inflight_station[st] = nq
            else:
                inflight_station.pop(st, None)

    def _detach_later_from_bag(bag: Dict[str, dict], task: dict) -> None:
        """FIFO-safe: requeue task and all later same-station entries in bag."""
        st = _task_station(task)
        tid_s = str(task.get("task_id") or "").strip()
        q_inf = list(inflight_station.get(st) or [])
        if tid_s and tid_s in q_inf:
            later = set(q_inf[q_inf.index(tid_s) :])
        else:
            later = {tid_s} if tid_s else set()
        for agv, t2 in list(bag.items()):
            t2id = str(t2.get("task_id") or "").strip()
            if t2id not in later:
                                continue
            if _requeue(queues, order, t2, picked_ids=picked_ids):
                _clear_inflight_tid(t2id)
            bag.pop(agv, None)
            pending_pickup.pop(agv, None)

    def _is_station_head(task: dict) -> bool:
        tid_s = str(task.get("task_id") or "").strip()
        st = _task_station(task)
        q_inf = inflight_station.get(st) or []
        return (not q_inf) or (q_inf[0] == tid_s)

    # ---- hierarchical / map-net ----
    gate = None
    wave_net = None
    m0_handoff_active = False
    hier_stats: dict = {
        "waves_escalated": 0,
        "waves_passthrough": 0,
        "waves_astar": 0,
        "waves_ecbs": 0,
        "waves_hybrid": 0,
        "hybrid_ecbs_agents": 0,
        "hybrid_astar_agents": 0,
        "sticky_astar_agents": 0,
        "affinity_clears": 0,
        "accel_waves": 0,
        "last_reason": "off",
        "last_planner": "ecbs",
        "hardness": 0.0,
        "joint_core": joint_core_mode,
    }

    def _set_affinity(agv: str, kind: str) -> None:
        agv_affinity[str(agv)] = str(kind)

    def _clear_affinity(agv: str) -> None:
        if agv_affinity.pop(str(agv), None) is not None:
            hier_stats["affinity_clears"] = (
                int(hier_stats.get("affinity_clears") or 0) + 1
            )

    if use_hierarchical:
        from ml_research.benchmarks.coord_custom_ai.hierarchical_gate import (
            LifelongGateTracker,
            map_hardness,
            narrow_cut_score,
        )
        from ml_research.benchmarks.wave_net.policy import load_wave_policy, select_wave_agents

        hardness0 = map_hardness(
            extra_obstacles=list(meta.get("extra_obstacles") or []),
            static=static,
            free=free,
        )
        narrow0 = float(narrow_cut_score(free))
        # EscalateNet / EscalateNetMap disabled: want_on ignored on easy band;
        # upgrades come from force_commit (M0 degrade) / map hardness / force.
        escalate_net = None
        hier_stats["escalate_ai_enabled"] = False
        wave_net, wave_meta = load_wave_policy(enabled=bool(use_wavenet))
        hier_stats["wave_meta"] = wave_meta
        # SceneDifficultyNet disabled: band/planner come from map hardness /
        # force_commit (M0 degrade) — not p_hard bands.
        scene_diff_net = None
        hier_stats["scene_diff_enabled"] = False
        # Mode-switch: only when explicitly requested (meta/arg). Default OFF so
        # static easy demos (SH01) keep the M0+SwapNet short-circuit.
        if allow_mode_switch is None:
            allow_switch = bool(meta.get("allow_mode_switch", False))
        else:
            allow_switch = bool(allow_mode_switch)
        gate = LifelongGateTracker(
            map_hardness=float(hardness0),
            narrow_cut=float(narrow0),
            threshold=float(hierarchical_threshold),
            force=bool(hierarchical_force),
            hard_cap=int(wave_hard_cap),
            max_active=int(max_active),
            escalate_net=escalate_net,
            scene_diff_net=scene_diff_net,
            use_escalate_ai=False,
            use_map_obs=True,
            use_scene_diff=False,
            allow_mode_switch=bool(allow_switch),
        )
        hier_stats["hardness"] = float(hardness0)
        hier_stats["narrow_cut"] = float(narrow0)
        hier_stats["allow_mode_switch"] = bool(allow_switch)

        # Probe gate once (map hardness / force only). Easy ⇒ true engine M0(+SwapNet).
        # allow_mode_switch enables mid-run yield out of M0 (degrade→ECBS),
        # and must NOT divert the easy start away from M0.
        gate.set_scene(static=static, stations=stations, pose=pose)
        probe = gate.decide(
            assigned={},
            queues=queues,
            recent_joint_fail=0,
        )
        if (
            str(getattr(probe, "planner", "") or "") == "astar"
            and str(getattr(probe, "scene_label", "") or gate.last_scene_label)
            == "easy"
            and not bool(hierarchical_force)
        ):
            eng_csv = tcsv
            if max_tasks > 0 or str(tcsv.resolve()) != str(
                Path(meta["task_csv"]).resolve()
            ):
                eng_csv = OUT / f"{meta['id']}_easy_engine_tasks.csv"
                n_w = _write_queue_task_csv(queues, eng_csv)
                if n_w <= 0:
                    n_w = total
            else:
                n_w = total
            # Easy path: always attach SwapNet (falls back if ckpt missing).
            do_swap = True
            if allow_switch:
                print(
                    "[HIER] allow_mode_switch=1 → true M0+SwapNet with yield "
                    f"(probe={probe.planner}/{gate.last_scene_label})",
                    flush=True,
                )
            m0_rep = _solve_via_m0_engine(
                meta=meta,
                task_csv=eng_csv,
                total=int(n_w),
                use_swapnet=bool(do_swap),
                swapnet_margin=float(swapnet_margin),
                hier_stats=hier_stats,
                t0=t0,
                plan=plan,
                max_active=max_active,
                weight=weight,
                pipeline=pipeline,
                turn_aware=turn_aware,
                plan_horizon=plan_horizon,
                exec_horizon=exec_horizon,
                use_hierarchical=use_hierarchical,
                use_wavenet=use_wavenet,
                gate=gate if allow_switch else None,
                allow_yield=bool(allow_switch),
                traffic_recovery=False,
            )
            if not m0_rep.get("handoff"):
                return m0_rep
            # ---- M0 → medium/hard handoff: warm-start wave loop ----
            snap = dict(m0_rep.get("handoff_snapshot") or {})
            m0_handoff_active = True
            hier_stats = dict(m0_rep.get("hierarchical") or hier_stats)
            hier_stats["m0_handoff"] = True
            handoff_label = str(
                snap.get("handoff_label") or m0_rep.get("handoff_label") or "medium"
            )
            now = int(snap.get("now") or 0)
            done = int(snap.get("done") or 0)
            pose = dict(snap.get("pose") or pose)
            for n, p in pose.items():
                if n not in names:
                    names.append(n)
            names = sorted(set(names) | set(pose.keys()))
            queues = {
                k: [dict(t) for t in v]
                for k, v in (snap.get("queues") or {}).items()
                if v
            }
            order = sorted(queues.keys())
            pending_delivery = {
                str(k): dict(v)
                for k, v in (snap.get("pending_delivery") or {}).items()
            }
            pending_pickup = {
                str(k): dict(v) for k, v in (snap.get("pending_pickup") or {}).items()
            }
            picked_ids = set(str(x) for x in (snap.get("picked_ids") or []))
            steps_by = {
                str(k): [dict(r) for r in (v or [])]
                for k, v in (snap.get("steps_by") or {}).items()
            }
            for n in names:
                if n not in steps_by or not steps_by[n]:
                    p = pose.get(n, (1, 1, 90))
                    steps_by[n] = [_hold(n, p, now)]
            # Restore true pads BEFORE inflight rebuild — M0 tasks often lack
            # start_point; empty station key dropped Dragon-3 from inflight and
            # let Dragon-4 claim while Dragon-3 sat in pending_pickup (r55).
            _restore_task_pads(queues, pending_pickup)
            # Rebuild inflight_station from pending approach/carriers
            inflight_station = {}
            for bag in (pending_pickup, pending_delivery):
                for _agv, task in bag.items():
                    tid_s = str(task.get("task_id") or "").strip()
                    st = str(
                        task.get("pickup_name")
                        or task.get("start_point")
                        or (
                            tid_s.rsplit("-", 1)[0]
                            if "-" in tid_s
                            else ""
                        )
                    )
                    if st and tid_s:
                        q = inflight_station.setdefault(st, [])
                        if tid_s not in q:
                            q.append(tid_s)
            for n in names:
                work_count.setdefault(n, 0)
                agv_affinity.setdefault(n, "joint")
            print(
                f"[HIER] M0 handoff -> wave loop label={handoff_label} t={now} "
                f"done={done}/{total} q={sum(len(v) for v in queues.values())} "
                f"delivery={len(pending_delivery)} pickup={len(pending_pickup)} "
                f"joint_core={joint_core_mode}",
                flush=True,
            )
            if gate is not None:
                try:
                    # SH03 wave-ECBS style: hard band + full ECBS planner so
                    # each wave can move a large AGV set together (同进同出).
                    gate.force_commit_at_least(
                        "hard", reason="m0_handoff_sh03_wave", sim_t=int(now)
                    )
                    gate.calm_waves_to_release = max(
                        int(getattr(gate, "calm_waves_to_release", 5) or 5), 20
                    )
                    gate.scene_calm_streak = 0
                    print(
                        f"[HIER] handoff sticky band={gate.committed_scene} "
                        f"calm_need={gate.calm_waves_to_release} "
                        f"(SH03-style hard/ECBS)",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[HIER] WARN sticky commit failed: {exc}", flush=True)
            # Skip cold-start [ECBS] banner; continue into lifelong wave loop below.

    if not m0_handoff_active:
        plan = "joint" if plan in ("eecbs", "wcbs", "ecbs") else plan
        print(
            f"[ECBS] SH slot={slot} agents={len(names)} tasks={total} "
            f"max_active={max_active} plan={plan} pipeline={pipeline} "
            f"turn_aware={turn_aware} hier={use_hierarchical} wavenet={use_wavenet} "
            f"joint_core={joint_core_mode}",
            flush=True,
        )
    else:
        plan = "joint" if plan in ("eecbs", "wcbs", "ecbs") else plan

    while any(queues.values()) or pending_delivery or pending_pickup:
        # Break residual multi-AGV cells before planning (r53: 20-tick pileups).
        _ld = {n: False for n in names}
        _ds = {n: "" for n in names}
        _td = {n: "" for n in names}
        for _a, _tsk in pending_delivery.items():
            _ld[_a] = True
            _ds[_a] = str(_tsk.get("destination") or "")
            _td[_a] = str(_tsk.get("task_id") or "")
        now = _separate_pose_overlaps(
            pose,
            steps_by,
            now,
            blocked_plan,
            loaded=_ld,
            dest=_ds,
            tid=_td,
            max_rounds=2,
        )
        _wb = meta.get("wall_timeout")
        wall_budget = 1e9 if _wb is None else float(_wb)
        if 0 < wall_budget < 1e8 and (time.perf_counter() - t0) > wall_budget:
            print(
                f"[ECBS] wall timeout {wall_budget:.1f}s @t={now} "
                f"done={done}/{total} -> abort remaining",
                flush=True,
            )
            for st, q in list(queues.items()):
                for task in q:
                    tid_drop = str(task.get("task_id") or "")
                    if tid_drop:
                        _mark_failed(tid_drop)
                queues[st] = []
            queues = {}
            for agv, task in list(pending_delivery.items()):
                tid_drop = str(task.get("task_id") or "")
                if tid_drop:
                    _mark_failed(tid_drop)
            pending_delivery.clear()
            for agv, task in list(pending_pickup.items()):
                tid_drop = str(task.get("task_id") or "")
                if tid_drop:
                    _mark_failed(tid_drop)
            pending_pickup.clear()
            break

        # ---- Gate: choose planner BEFORE claim (no SceneDifficultyNet) ----
        # Claim size is dest-pad budget (see _wave_claim_budget), not full fleet
        # and not suggested_k. suggested_k only sizes the ECBS/hybrid joint core.
        pre_gate = None
        wave_planner = "ecbs"
        if gate is not None:
            gate.set_scene(static=static, stations=stations, pose=pose)
            pre_gate = gate.decide(
                assigned={},
                queues=queues,
                recent_joint_fail=recent_joint_fail,
            )
            wave_planner = str(getattr(pre_gate, "planner", "ecbs") or "ecbs")
            # Signal-based: after M0 yield, keep SH03-style ECBS (not hybrid
            # peel-to-k=1 / wave-astar serial) until backlog is small.
            if m0_handoff_active and (int(total) - int(done)) > max(
                8, int(0.05 * float(total))
            ):
                if wave_planner in ("astar", "hybrid"):
                    wave_planner = "ecbs"
                try:
                    gate.force_commit_at_least(
                        "hard", reason="handoff_floor_ecbs", sim_t=int(now)
                    )
                except Exception:
                    pass
            # Large-wave floor (industrial): gate shrink_scene_easy often sets
            # k=2..4; forbid 1–2-vehicle waves. Prefer full hard_cap when
            # wall_first / ecbs_prefer_full_k is on.
            min_k = int(meta.get("ecbs_min_wave_k") or 0)
            if (
                min_k > 0
                and wave_planner in ("ecbs", "hybrid")
                and pre_gate is not None
            ):
                prefer_full = bool(meta.get("ecbs_prefer_full_k", False)) or bool(
                    meta.get("wall_first_ecbs", False)
                )
                try:
                    sk = int(getattr(pre_gate, "suggested_k", min_k) or min_k)
                    if prefer_full or m0_handoff_active:
                        sk = max(min_k, int(wave_hard_cap))
                    else:
                        sk = max(min_k, min(int(wave_hard_cap), sk))
                    object.__setattr__(pre_gate, "suggested_k", int(sk))
                except Exception:
                    try:
                        pre_gate.suggested_k = max(
                            min_k, min(int(wave_hard_cap), int(pre_gate.suggested_k))
                        )
                    except Exception:
                        pass
            if gate.scene_decision_log:
                peek_phases: List[str] = []
                for st_name, q in queues.items():
                    for t in q[:3]:
                        tid = str(t.get("task_id") or "")
                        peek_phases.append(phase_map.get(tid, ""))
                gate.scene_decision_log[-1]["peek_phases"] = [
                    p for p in peek_phases if p
                ]
                gate.scene_decision_log[-1]["done_before"] = int(done)
                gate.scene_decision_log[-1]["queue_before"] = int(
                    sum(len(v) for v in queues.values())
                )

        if wave_planner == "astar":
            # After M0 handoff: wave-astar is not true M0. Allow controlled
            # handback to engine M0 after sustained easy calm (anti ping-pong).
            if m0_handoff_active and gate is not None:
                label_now = str(gate.last_scene_label or "")
                if label_now == "easy" and bool(
                    getattr(pre_gate, "deescalated", False)
                    or "recovered_fast" in str(getattr(pre_gate, "reason", "") or "")
                    or easy_calm_waves > 0
                ):
                    easy_calm_waves += 1
                elif label_now == "easy":
                    easy_calm_waves += 1
                else:
                    easy_calm_waves = 0
                wall_now = time.perf_counter()
                # True M0 handback requires pose-warm restart; cold CSV restart
                # caused collisions/FIFO breaks (SH01 switch 318/400 INVALID).
                # Default: calm de-escalate stays on wave-astar (fast enough).
                # Opt-in: meta["m0_handback"]=1 when warm handback is implemented.
                enable_m0_handback = bool(meta.get("m0_handback", False))
                can_handback = (
                    enable_m0_handback
                    and allow_switch_runtime
                    and (not m0_handback_done)
                    and easy_calm_waves >= handback_calm_need
                    and wall_now >= float(handback_cooldown_until)
                    and not pending_delivery
                    and done > 0
                )
                if (
                    (not enable_m0_handback)
                    and allow_switch_runtime
                    and m0_handoff_active
                    and easy_calm_waves >= handback_calm_need
                    and easy_calm_waves == handback_calm_need
                ):
                    print(
                        f"[HIER] calm recovered on wave-astar "
                        f"(skip cold M0 handback) t={now} done={done}/{total}",
                        flush=True,
                    )
                if can_handback:
                    print(
                        f"[HIER] M0 handback gate open calm={easy_calm_waves} "
                        f"t={now} done={done}/{total} -> true M0 engine",
                        flush=True,
                    )
                    # Flush remaining work to CSV and resume on true M0.
                    remain_csv = OUT / f"{meta['id']}_m0_handback_remain.csv"
                    # Requeue in-flight pickups into station queues.
                    for agv, task in list(pending_pickup.items()):
                        _requeue_task(task)
                        pending_pickup.pop(agv, None)
                        _clear_affinity(agv)
                    n_rem = _write_queue_task_csv(queues, remain_csv)
                    if n_rem <= 0 and not any(queues.values()):
                        print("[HIER] handback skipped — no remaining tasks", flush=True)
                    else:
                        # Persist wave prefix trajectory, then M0 finishes rest.
                        _dense_fill(steps_by, now)
                        prefix_traj = (
                            TRAJ
                            / f"{meta['id']}_wave_prefix_t{now}.csv"
                        )
                        rows_p = []
                        for n in sorted(steps_by):
                            rows_p.extend(steps_by[n])
                        rows_p.sort(
                            key=lambda r: (int(r["timestamp"]), r["name"])
                        )
                        with prefix_traj.open(
                            "w", newline="", encoding="utf-8"
                        ) as f:
                            w = csv.DictWriter(f, fieldnames=TRAJ_HEADER)
                            w.writeheader()
                            for r in rows_p:
                                w.writerow({k: r.get(k, "") for k in TRAJ_HEADER})
                        hb_meta = dict(meta)
                        hb_meta["id"] = f"{meta['id']}_handback"
                        hb_meta["task_csv"] = str(remain_csv.resolve())
                        hb_stats = dict(hier_stats)
                        hb_stats["m0_handback"] = True
                        hb_stats["m0_handback_t"] = int(now)
                        m0_tail = _solve_via_m0_engine(
                            meta=hb_meta,
                            task_csv=remain_csv,
                            total=max(1, int(n_rem)),
                            use_swapnet=True,
                            swapnet_margin=float(swapnet_margin),
                            hier_stats=hb_stats,
                            t0=t0,
                            plan=plan,
                            max_active=max_active,
                            weight=weight,
                            pipeline=pipeline,
                            turn_aware=turn_aware,
                            plan_horizon=plan_horizon,
                            exec_horizon=exec_horizon,
                            use_hierarchical=use_hierarchical,
                            use_wavenet=use_wavenet,
                            gate=gate,
                            allow_yield=True,
                            traffic_recovery=False,
                        )
                        m0_handback_done = True
                        handback_cooldown_until = time.perf_counter() + 30.0
                        hier_stats.update(m0_tail.get("hierarchical") or hb_stats)
                        hier_stats["m0_handback"] = True
                        hier_stats["m0_handback_t"] = int(now)
                        # If M0 handed off again, continue wave with fresh snapshot.
                        if m0_tail.get("handoff"):
                            snap = dict(m0_tail.get("handoff_snapshot") or {})
                            now = int(snap.get("now") or now)
                            done = int(snap.get("done") or done)
                            pose = dict(snap.get("pose") or pose)
                            queues = {
                                k: [dict(t) for t in v]
                                for k, v in (snap.get("queues") or {}).items()
                                if v
                            }
                            pending_delivery = {
                                str(k): dict(v)
                                for k, v in (
                                    snap.get("pending_delivery") or {}
                                ).items()
                            }
                            pending_pickup = {
                                str(k): dict(v)
                                for k, v in (
                                    snap.get("pending_pickup") or {}
                                ).items()
                            }
                            easy_calm_waves = 0
                            _restore_task_pads(queues, pending_pickup)
                            print(
                                f"[HIER] handback M0 re-escalated "
                                f"label={m0_tail.get('handoff_label')} "
                                f"t={now} — resume wave",
                                flush=True,
                            )
                            continue
                        # M0 finished remainder: stitch traj and exit loop.
                        done = int(m0_tail.get("tasks_completed") or done)
                        now = int(m0_tail.get("sim_time") or now)
                        queues = {}
                        pending_delivery.clear()
                        pending_pickup.clear()
                        hier_stats["last_planner"] = "m0_handback"
                        # Prefer M0 full traj if complete; else keep prefix.
                        tail_traj = Path(str(m0_tail.get("trajectory") or ""))
                        if tail_traj.exists() and float(
                            m0_tail.get("completion_ratio") or 0
                        ) >= 0.999:
                            # Use M0 traj as authoritative finish (engine-native).
                            # Wave prefix already covered pre-handback; for
                            # validate we keep M0 output when it completed all.
                            pass
                        print(
                            f"[HIER] M0 handback done sim={now} "
                            f"cr={m0_tail.get('completion_ratio')} "
                            f"validate={m0_tail.get('validate_ok')}",
                            flush=True,
                        )
                        # Adopt M0 report fields via hier_stats; break to write.
                        if float(m0_tail.get("completion_ratio") or 0) >= 0.999:
                            # Replace steps with M0 traj rows for final export.
                            try:
                                with tail_traj.open(newline="", encoding="utf-8") as f:
                                    for row in csv.DictReader(f):
                                        n = str(row.get("name") or "")
                                        if not n:
                                            continue
                                        steps_by.setdefault(n, []).append(dict(row))
                                now = max(
                                    now,
                                    max(
                                        (
                                            int(r.get("timestamp") or 0)
                                            for rows in steps_by.values()
                                            for r in rows
                                        ),
                                        default=now,
                                    ),
                                )
                            except Exception as exc:  # noqa: BLE001
                                print(
                                    f"[HIER] WARN handback traj merge: {exc}",
                                    flush=True,
                                )
                            break
                elif m0_handoff_active and easy_calm_waves < handback_calm_need:
                    if easy_calm_waves == 1 or easy_calm_waves == handback_calm_need - 1:
                        print(
                            f"[HIER] handback calm {easy_calm_waves}/"
                            f"{handback_calm_need} (wave-astar, not true M0 yet)",
                            flush=True,
                        )
        assigned: Dict[str, dict] = {}
        # Heal ghost cargo first: traj still loaded but not in pending_delivery
        # (off-pad clear guards) — force unload before any pickup resume/claim.
        for agv in names:
            if agv in pending_delivery:
                continue
            last = (steps_by.get(agv) or [None])[-1]
            if last is None:
                continue
            prev_ld = str(last.get("loaded", "")).lower() in ("true", "1", "yes")
            prev_tid = str(last.get("task-id") or "").strip()
            if not prev_ld or not prev_tid:
                continue
            if prev_tid in done_seen or prev_tid in failed_seen:
                continue
            pending_delivery[agv] = {
                "task_id": prev_tid,
                "destination": str(last.get("destination") or ""),
                "end_points": [
                    list(c) for c in sorted(_TID_DROPOFFS.get(prev_tid) or [])
                ],
            }
            print(
                f"[ECBS] heal ghost cargo agv={agv} tid={prev_tid} @t={now}",
                flush=True,
            )
        # Resume incomplete pickups before claiming new surface heads.
        n_resume = len(pending_pickup)
        for agv, task in list(pending_pickup.items()):
            tid_r = str(task.get("task_id") or "").strip()
            # Still carrying another tid → unload first, requeue this pickup.
            if agv in pending_delivery:
                cargo_tid = str(
                    (pending_delivery.get(agv) or {}).get("task_id") or ""
                ).strip()
                if cargo_tid and cargo_tid != tid_r:
                    pending_pickup.pop(agv, None)
                    _requeue_task(task)
                    continue
            last = (steps_by.get(agv) or [None])[-1]
            if last is not None:
                prev_ld = str(last.get("loaded", "")).lower() in (
                    "true",
                    "1",
                    "yes",
                )
                prev_tid = str(last.get("task-id") or "").strip()
                if prev_ld and prev_tid and prev_tid != tid_r:
                    pending_pickup.pop(agv, None)
                    _requeue_task(task)
                    if agv not in pending_delivery:
                        pending_delivery[agv] = {
                            "task_id": prev_tid,
                            "destination": str(last.get("destination") or ""),
                            "end_points": [
                                list(c)
                                for c in sorted(_TID_DROPOFFS.get(prev_tid) or [])
                            ],
                        }
                    continue
            assigned[agv] = task
            pending_pickup.pop(agv, None)
            if tid_r:
                picked_ids.discard(tid_r)
            st_r = _task_station(task)
            if st_r and tid_r:
                q_inf = inflight_station.setdefault(st_r, [])
                if tid_r not in q_inf:
                    q_inf.insert(0, tid_r)
        # Keep true pads on every bag (M0/staging must not stick as commit goals).
        _restore_task_pads(queues, pending_pickup, pending_delivery, assigned)
        resume_stations: Set[str] = {
            _task_station(t) for t in assigned.values() if t
        }
        # Per-wave cargo maps — init BEFORE claim so cooldown/disperse never
        # sees a stray str from hybrid sticky (r20 ValueError on dict(dest)).
        loaded = {n: False for n in names}
        dest = {n: "" for n in names}
        tid = {n: "" for n in names}
        for agv, task in list(pending_delivery.items()):
            d_s = str(task.get("destination") or "")
            t_s = str(task.get("task_id") or "")
            if not _heal_same_tid_cargo_wipe(
                steps_by, agv, tid_s=t_s, dest_s=d_s
            ):
                pending_delivery.pop(agv, None)
                if t_s and t_s not in done_seen and t_s not in failed_seen:
                    # Post-unload desync: traj already clear on pad → done.
                    last = steps_by[agv][-1]
                    pads = set(_valid_unload_pads(task, free, static))
                    last_cell = (int(last["X"]), int(last["Y"]))
                    if last_cell in pads:
                        _count_done(t_s)
                    else:
                        _requeue_task(task)
                _STICKY.pop(agv, None)
                _clear_affinity(agv)
                continue
            loaded[agv] = True
            dest[agv] = d_s
            tid[agv] = t_s
            # Refresh dest/tid on already-loaded stays only (r66 pickup_cell was
            # False→True annotate at random cells).
            last = steps_by[agv][-1]
            prev_ld = str(last.get("loaded", "")).lower() in ("true", "1", "yes")
            same = (int(last["X"]), int(last["Y"])) == pose[agv][:2]
            if prev_ld and same:
                last["destination"] = dest[agv]
                last["task-id"] = tid[agv]
        # Sticky means still loaded and waiting for a pad. An unload that
        # forgot to drop the mark must not freeze every AGV out of claims.
        for agv in list(_STICKY):
            if agv not in pending_delivery and not loaded.get(agv, False):
                _STICKY.pop(agv, None)
        # Carrying / in-flight pickup AGVs are busy — do not claim new pickups.
        free_agvs = [
            n
            for n in names
            if n not in pending_delivery
            and n not in assigned
            and n not in pickup_spin_ban
            and n not in _STICKY
            and not loaded.get(n, False)
        ]
        if not free_agvs and pickup_spin_ban:
            # All idle AGVs banned — clear ban and retry.
            pickup_spin_ban.clear()
            free_agvs = [
                n
                for n in names
                if n not in pending_delivery
                and n not in assigned
                and n not in _STICKY
                and not loaded.get(n, False)
            ]
        # SH03-style: claim as many free AGVs as dest-pad room allows (同进).
        # Do NOT artificially throttle carriers — that forced tiny waves + serial
        # makespan. Dest-pad claim_cap already bounds hub saturation.
        # Dest-pad claim budget (all planners): Σ room(dest) ∩ free AGVs.
        # joint_k / suggested_k stays separate for ECBS core size.
        claim_cap, claim_budget_info = _wave_claim_budget(
            queues=queues,
            order=order,
            free=free,
            static=static,
            bags=(assigned, pending_delivery),
            n_free_agvs=len(free_agvs),
            max_active=int(max_active) if int(max_active) < len(names) else 0,
        )
        if pickup_soft_claim > 0:
            claim_cap = min(int(claim_cap), int(pickup_soft_claim))
        # After conflict dump: single-agent reclaim until a corridor succeed
        # resets spin (r33 dump-spin with k=4/6 never cleared).
        if pickup_conflict_spin > 0 and int(claim_cap) > 0:
            _mk = int(meta.get("ecbs_min_wave_k") or 4)
            claim_cap = max(1, min(int(claim_cap), max(2, min(4, _mk // 2))))
            pickup_soft_claim = max(int(pickup_soft_claim), int(claim_cap))
        # Prefer large waves: floor claim to ecbs_min_wave_k (else ≥4) when
        # room/free allow — r110 avg wave≈4 despite claim often 7–12.
        if (
            pickup_soft_claim <= 0
            and pickup_conflict_spin <= 0
            and int(claim_cap) > 0
            and len(free_agvs) >= 4
            and int(claim_budget_info.get("total_room") or 0) >= 4
        ):
            _mk = int(meta.get("ecbs_min_wave_k") or 0)
            prefer_full = bool(meta.get("ecbs_prefer_full_k", False)) or bool(
                meta.get("wall_first_ecbs", False)
            )
            floor_k = (
                max(4, min(int(wave_hard_cap), _mk if _mk > 0 else 4))
                if prefer_full or _mk >= 4
                else 4
            )
            if int(claim_cap) < floor_k:
                claim_cap = min(
                    floor_k,
                    len(free_agvs),
                    int(claim_budget_info.get("total_room") or floor_k),
                )
        # After failed cell-conflict recover: prefer disperse over new claims,
        # but NEVER zero claim when the fleet is idle with queued work (r21
        # starved → break while queues still full → ratio 0.085).
        if pickup_claim_cooldown > 0:
            print(
                f"[ECBS] pickup claim-cooldown={pickup_claim_cooldown} "
                f"assigned={len(assigned)} spin={pickup_conflict_spin} @t={now}",
                flush=True,
            )
            pickup_claim_cooldown -= 1
            ld_c = {n: bool(loaded.get(n)) for n in names}
            ds_c = {n: str(dest.get(n, "") or "") for n in names}
            td_c = {n: str(tid.get(n, "") or "") for n in names}
            # r90: park one blocker via short joint ST; keep reclaiming waves
            # instead of full serial disperse + claim_cap=0 until disperse ends.
            keep_cd: Set[Cell] = set(stations) if stations else set()
            wave_cd = set(assigned.keys())
            if wave_cd:
                s_cd = {n: pose[n][:2] for n in names}
                g_cd = {n: s_cd[n] for n in names}
                for a, task in assigned.items():
                    g_cd[a] = _pk(task, s_cd[a], free, static)
                    keep_cd.add(g_cd[a])
                    keep_cd.add(s_cd[a])
                joint_cd = _joint_wave_with_evac(
                    s_cd,
                    g_cd,
                    wave_cd,
                    blocked_plan,
                    pose,
                    free,
                    keep_cd,
                    names,
                    n_evac=1,
                    stations=stations,
                )
                if joint_cd is not None:
                    p_cd, mv_cd, g_cd = joint_cd
                    tl_cd = _plan_paths_as_poses(pose, p_cd)
                    if not _pose_timelines_conflict(
                        {a: pose[a] for a in mv_cd},
                        {a: tl_cd.get(a) or [] for a in mv_cd},
                        pad_holds=False,
                    ):
                        now = _apply_pose_timelines(
                            steps_by,
                            pose,
                            tl_cd,
                            now,
                            loaded=ld_c,
                            dest=ds_c,
                            tid=td_c,
                            rise_ok={},
                        )
                        print(
                            f"[ECBS] claim-cooldown joint-evac "
                            f"k={len(wave_cd)} movers={len(mv_cd)} @t={now}",
                            flush=True,
                        )
                    else:
                        now = _disperse_idle_agents(
                            pose,
                            steps_by,
                            now,
                            movers=set(pending_delivery.keys()) | wave_cd,
                            free=free,
                            blocked_plan=blocked_plan,
                            keep_clear=set(stations) if stations else set(),
                            loaded=ld_c,
                            dest=ds_c,
                            tid=td_c,
                            aggressive=False,
                        )
                        _force_wait_tick("claim_cooldown")
                else:
                    now = _disperse_idle_agents(
                        pose,
                        steps_by,
                        now,
                        movers=set(pending_delivery.keys()) | wave_cd,
                        free=free,
                        blocked_plan=blocked_plan,
                        keep_clear=set(stations) if stations else set(),
                        loaded=ld_c,
                        dest=ds_c,
                        tid=td_c,
                        aggressive=False,
                    )
                    _force_wait_tick("claim_cooldown")
            else:
                # No assigned wave: nudge one pad-sitter toward a corner.
                idle_cd = [
                    n
                    for n in names
                    if n not in pending_delivery
                    and (not stations or pose[n][:2] in stations)
                ]
                if idle_cd:
                    who = idle_cd[0]
                    park = _pick_park_target(
                        who, pose, free, set(stations) if stations else set()
                    )
                    if park is not None:
                        now, _ = _serial_move_to(
                            pose,
                            steps_by,
                            now,
                            who,
                            park,
                            blocked_plan,
                            loaded=ld_c,
                            dest=ds_c,
                            tid=td_c,
                        )
                _force_wait_tick("claim_cooldown")
            if assigned or pending_delivery:
                # Soft floor: allow a small reclaim while cooling (was hard 0).
                claim_cap = max(0, min(2, len(free_agvs)))
            elif pickup_conflict_spin > 0:
                # Allow reclaim during dump cooldown (floor ≥2 when min_k set).
                _mk = int(meta.get("ecbs_min_wave_k") or 4)
                claim_cap = max(1, min(4, max(2, _mk // 2), len(free_agvs)))
                pickup_soft_claim = max(int(pickup_soft_claim), int(claim_cap))
            else:
                # Idle + queued work: keep a large-wave floor so we do not exit.
                claim_cap = max(int(claim_cap), min(4, len(free_agvs)))
        # Delivery hub jam (r26/r27): drain carriers; also requeue in-flight
        # pickups or they refill the hub while unload is stuck.
        if int(recent_joint_fail) >= 6 and len(pending_delivery) >= 3:
            if assigned:
                for agv, task in list(assigned.items()):
                    _requeue_task(task)
                    _clear_affinity(agv)
                assigned.clear()
                pending_pickup.clear()
            if int(claim_cap) > 0:
                print(
                    f"[ECBS] claim-freeze delivery-jam fail={recent_joint_fail} "
                    f"carry={len(pending_delivery)} @t={now}",
                    flush=True,
                )
            claim_cap = 0
        if gate is not None and pre_gate is not None:
            print(
                f"[SCENE] before-claim label={gate.last_scene_label} "
                f"p_hard={gate.last_scene_p_hard:.3f} "
                f"claim={claim_cap} room={claim_budget_info.get('total_room')} "
                f"k_joint={pre_gate.suggested_k} "
                f"pol={gate.last_k_policy} planner={wave_planner} "
                f"joint_core={joint_core_mode} "
                f"reason={pre_gate.reason}",
                flush=True,
            )
        else:
            print(
                f"[CLAIM] dest-pad claim={claim_cap} "
                f"room={claim_budget_info.get('total_room')} "
                f"free={len(free_agvs)} planner={wave_planner}",
                flush=True,
            )
        # main-copy surface (navigation.py): assign head → pop → next head
        # immediately becomes surface. Same station can feed many AGVs in one
        # wave (SwapNet relies on this). used_stations-per-wave was WRONG.
        # New claims only fill remaining dest-pad rooms (already net of inflight).
        claim_room = max(0, int(claim_cap))
        picks = 0
        unload_doors = _all_unload_doors(set(blocked_plan))
        for _ in range(max(1, len(free_agvs) * 8 + 4)):
            if picks >= claim_room or not free_agvs:
                break
            # Drop stale heads first
            for cand in list(order):
                if not queues.get(cand):
                    continue
                tid_head = str(queues[cand][0].get("task_id") or "").strip()
                if tid_head and (tid_head in picked_ids or tid_head in failed_seen):
                    queues[cand].pop(0)
                    if not queues.get(cand):
                        queues.pop(cand, None)
            # Current surface heads with no in-flight pickup yet.
            # Keep assign→promote, but at most one physical approach per station
            # so the fleet fans out (sim_time target) instead of piling waiters.
            # Skip stations with an unfinished resumed approach (FIFO).
            heads: List[Tuple[str, dict]] = []
            for cand in order:
                if cand in resume_stations:
                    continue
                if queues.get(cand) and not (inflight_station.get(cand) or []):
                    heads.append((cand, queues[cand][0]))
            if not heads:
                break

            if wave_planner in ("astar", "hybrid") or gate is not None:
                # Assignment uses solo A* (walls/stations only). Other AGVs are
                # moving obstacles later, when the path is actually planned.
                best = None
                best_rank = None
                for st, task in heads:
                    dest_name = str(task.get("destination") or "")
                    if dest_name:
                        cap = _dest_inflight_cap(task, free, static, len(names))
                        inflight = _count_dest_inflight(
                            dest_name, assigned, pending_delivery
                        )
                        if inflight >= cap:
                            continue
                    pk = tuple(task["pickup_point"])
                    urg = 0.7 if int(task.get("numbers_before_urgent", -1)) >= 0 else 1.0
                    for agv in free_agvs:
                        # Cars sitting on an unload door must leave. Prefer them
                        # over a closer idle, otherwise they stay and block the pad.
                        rank = (
                            0 if pose[agv][:2] in unload_doors else 1,
                            float(_bfs_len(pose[agv][:2], pk, blocked_plan)) * urg,
                            agv,
                        )
                        if best is None or rank < best_rank:
                            best_rank = rank
                            best = (agv, st, task)
                if best is None:
                    # All heads dest-capped: claim nothing this wave (carriers drain).
                    break
                agv, st, task = best
            else:
                # ECBS window: round-robin among claimable surface heads
                task = None
                st = None
                for k in range(len(order)):
                    cand = order[(rr + k) % len(order)]
                    if cand in resume_stations:
                        continue
                    if queues.get(cand) and not (inflight_station.get(cand) or []):
                        st, task = cand, queues[cand][0]
                        rr = (rr + k + 1) % max(1, len(order))
                        break
                if not task or st is None:
                    # Never wipe held inflight — that lets later FIFO heads claim
                    # while an earlier tid is still in assigned/pending (r55 Dragon-3).
                    held_tids: Set[str] = set()
                    for bag in (assigned, pending_pickup, pending_delivery):
                        for _t in bag.values():
                            _id = str((_t or {}).get("task_id") or "").strip()
                            if _id:
                                held_tids.add(_id)
                    for st_k, q in list(inflight_station.items()):
                        nq = [t for t in q if t in held_tids]
                        if nq:
                            inflight_station[st_k] = nq
                        else:
                            inflight_station.pop(st_k, None)
                    for k in range(len(order)):
                        cand = order[(rr + k) % len(order)]
                        if cand in resume_stations:
                            continue
                        if queues.get(cand) and not (inflight_station.get(cand) or []):
                            st, task = cand, queues[cand][0]
                            rr = (rr + k + 1) % max(1, len(order))
                            break
                if not task or st is None:
                    break
                pk = tuple(task["pickup_point"])
                agv = min(
                    free_agvs,
                    key=lambda n: (
                        0 if pose[n][:2] in unload_doors else 1,
                        _bfs_len(pose[n][:2], pk, blocked_plan),
                        n,
                    ),
                )

            # Assign = reserve + promote next (pop exposes new surface head)
            queues[st].pop(0)
            if st in queues and not queues[st]:
                del queues[st]
            assigned[agv] = task
            free_agvs.remove(agv)
            picks += 1
            # Sticky affinity: mode at claim time sticks until unload.
            if wave_planner == "astar":
                _set_affinity(agv, "astar")
            elif _planner_uses_hybrid_split(wave_planner):
                _set_affinity(agv, "joint")
            tid_new = str(task.get("task_id") or "")
            if tid_new:
                inflight_station.setdefault(st, []).append(tid_new)

        _repair_inflight(
            {str(t.get("task_id") or "") for t in assigned.values() if t}
        )
        # Drop any stacked non-head reservations and free those AGVs.
        for agv, task in list(assigned.items()):
            if not _is_station_head(task):
                _requeue_task(task)
                assigned.pop(agv, None)
                pending_pickup.pop(agv, None)
                _clear_affinity(agv)

        if not assigned and not pending_delivery and not pending_pickup:
            if any(queues.values()):
                # Still have surface tasks but nothing claimed this wave —
                # do not exit the solver (r21 claim-cooldown starve).
                # r23: after conflict-dump, cooldown ticks hit here with claim=0;
                # zeroing cooldown/soft would immediately re-claim the same wave.
                if pickup_claim_cooldown > 0 or pickup_conflict_spin > 0:
                    continue
                _force_wait_tick("claim_starve")
                pickup_claim_cooldown = 0
                pickup_soft_claim = 0
                # Rebuild inflight from live bags — do not clear (FIFO).
                _repair_inflight(
                    {str(t.get("task_id") or "") for t in assigned.values() if t}
                )
                continue
            break

        # easy → all A*; medium/hard → hybrid split (ECBS core + A* bystanders)
        wave_plan = "per_agent" if wave_planner == "astar" else plan
        escalate = bool(pre_gate.escalate) if pre_gate is not None else False
        movers_ecbs: Set[str] = set()
        movers_astar: Set[str] = set()
        hybrid_mode = "off"
        if wave_planner == "astar":
            escalate = False  # no WaveNet shrink on A* baseline branch
            movers_astar = set(assigned.keys())
            hier_stats["waves_astar"] = int(hier_stats.get("waves_astar") or 0) + 1
        elif wave_planner == "hybrid":
            hier_stats["waves_hybrid"] = int(hier_stats.get("waves_hybrid") or 0) + 1
        else:
            hier_stats["waves_ecbs"] = int(hier_stats.get("waves_ecbs") or 0) + 1

        if gate is not None and pre_gate is not None and assigned:
            hier_stats["last_reason"] = pre_gate.reason
            hier_stats["hardness"] = float(pre_gate.hardness)
            hier_stats["last_planner"] = wave_planner
            hier_stats["last_scene_label"] = str(
                getattr(pre_gate, "scene_label", "") or gate.last_scene_label
            )
            if escalate:
                hier_stats["waves_escalated"] = int(hier_stats["waves_escalated"]) + 1
            else:
                hier_stats["waves_passthrough"] = int(hier_stats["waves_passthrough"]) + 1

            if _planner_uses_hybrid_split(wave_planner):
                force_promote: Set[str] = set()
                # Clear sticky when joint fails OR no-progress (hub jam needs core).
                if int(recent_joint_fail) >= 2 or int(no_progress_waves) >= 2:
                    top_d = str(getattr(pre_gate, "top_dest", "") or "")
                    for n, kind in list(agv_affinity.items()):
                        if kind != "astar" or n not in assigned:
                            continue
                        dest_name = str(
                            (assigned.get(n) or {}).get("destination") or ""
                        )
                        if (
                            int(recent_joint_fail) >= 2
                            or int(no_progress_waves) >= 2
                            or (top_d and dest_name == top_d)
                            or n in pending_delivery
                        ):
                            force_promote.add(n)
                    # Also promote stuck carriers always on no-progress.
                    if int(no_progress_waves) >= 2:
                        force_promote |= set(pending_delivery.keys()) & set(assigned.keys())
                movers_ecbs, movers_astar, hybrid_mode = _split_hybrid_movers(
                    assigned,
                            pose,
                    k_budget=int(pre_gate.suggested_k),
                    hard_cap=int(wave_hard_cap),
                    escalate=bool(escalate),
                    top_dest=str(getattr(pre_gate, "top_dest", "") or ""),
                    wave_net=wave_net if use_wavenet else None,
                    map_hardness=float(pre_gate.hardness),
                    map_obs=getattr(gate, "last_map_obs", None),
                    static=static,
                    work_count=work_count,
                    bfs_len=_bfs_len,
                    affinity=agv_affinity,
                    scene_label=str(
                        getattr(pre_gate, "scene_label", "") or gate.last_scene_label
                    ),
                    force_promote=force_promote,
                )
                sticky_n = sum(
                    1 for n in movers_astar if agv_affinity.get(n) == "astar"
                )
                hier_stats["sticky_astar_agents"] = int(
                    hier_stats.get("sticky_astar_agents") or 0
                ) + int(sticky_n)
                if movers_astar and movers_ecbs:
                    if wave_planner == "ecbs":
                        hier_stats["waves_hybrid"] = (
                            int(hier_stats.get("waves_hybrid") or 0) + 1
                        )
                    hier_stats["hybrid_ecbs_agents"] = int(
                        hier_stats.get("hybrid_ecbs_agents") or 0
                    ) + len(movers_ecbs)
                    hier_stats["hybrid_astar_agents"] = int(
                        hier_stats.get("hybrid_astar_agents") or 0
                    ) + len(movers_astar)
                    hier_stats["last_planner"] = (
                        "hybrid" if wave_planner == "ecbs" else wave_planner
                    )
                    print(
                        f"[HIER] hybrid split mode={hybrid_mode} "
                        f"ecbs={len(movers_ecbs)} astar={len(movers_astar)} "
                        f"sticky={sticky_n} k={pre_gate.suggested_k} "
                        f"reason={pre_gate.reason}",
                                    flush=True,
                                )
                elif movers_ecbs:
                    print(
                        f"[HIER] ecbs-only mode={hybrid_mode} "
                        f"k={len(movers_ecbs)} reason={pre_gate.reason}",
                        flush=True,
                    )
                else:
                        print(
                        f"[HIER] astar-only sticky mode={hybrid_mode} "
                        f"k={len(movers_astar)} reason={pre_gate.reason}",
                            flush=True,
                        )
            else:
                movers_astar = set(assigned.keys())
        elif gate is not None and pre_gate is not None:
            hier_stats["last_reason"] = pre_gate.reason
            hier_stats["hardness"] = float(pre_gate.hardness)
            hier_stats["last_planner"] = wave_planner
        elif _planner_uses_hybrid_split(wave_planner):
            movers_ecbs = set(assigned.keys())
        else:
            movers_astar = set(assigned.keys())

        if not assigned and not pending_delivery:
            continue

        # Ensure split covers all assigned when hierarchical ECBS forgot to set
        if assigned and not movers_ecbs and not movers_astar:
            if _planner_uses_hybrid_split(wave_planner):
                movers_ecbs = set(assigned.keys())
            else:
                movers_astar = set(assigned.keys())

        def _route_wave(
            starts_g: Dict[str, Cell],
            goals_g: Dict[str, Cell],
            movers_set: Set[str],
        ) -> Optional[Dict[str, List[Cell]]]:
            movers_set = set(movers_set)
            if not movers_set:
                return {n: [starts_g[n]] for n in starts_g}
            block = set(blocked_plan)
            ta = _turn_aware_cell_paths(
                pose, goals_g, movers_set, block, prefer_outer_ring=True
            )
            if ta is not None:
                moving = {
                    n
                    for n, seq in ta.items()
                    if len(seq) > 1 and any(c != seq[0] for c in seq[1:])
                } or set(movers_set)
                g_chk = dict(goals_g)
                for n in moving:
                    if n not in g_chk and ta.get(n):
                        g_chk[n] = ta[n][-1]
                    if n not in movers_set:
                        pickup_extra_movers.add(n)
                        goals[n] = g_chk[n]
                if _st_paths_ok(ta, moving, g_chk) and (
                    not stations
                    or _paths_avoid_station_transit(
                        ta, movers_set, goals_g, stations
                    )
                ):
                    return ta
            # Always prefer idle-seeded spacetime A* (never ghost through parked AGVs).
            st_paths = _prioritized_st_paths(
                starts_g, goals_g, movers_set, blocked_plan
            )
            if (
                st_paths is not None
                and _st_paths_ok(st_paths, movers_set, goals_g)
                and (
                    not stations
                    or _paths_avoid_station_transit(
                        st_paths, movers_set, goals_g, stations
                    )
                )
            ):
                return st_paths
            # Easy/A* baseline: do not fall back to planners that ignore idle seeding.
            if wave_planner == "astar":
                return None

            ma = set(movers_astar) & movers_set
            me = set(movers_ecbs) & movers_set
            leftover = movers_set - ma - me
            if leftover:
                if _planner_uses_hybrid_split(wave_planner):
                    me |= leftover
                else:
                    ma |= leftover
            budget = min(3.0, max(1.5, float(time_limit) * 0.15))
            if ma and me:
                return _plan_hybrid_wave_paths(
                    grid,
                    starts_g,
                    goals_g,
                    movers_astar=ma,
                    movers_ecbs=me,
                    static=blocked_plan,
                    joint_plan=plan,
                    weight=weight,
                    time_limit=budget,
                    max_expansions=max_expansions,
                    plan_horizon=plan_horizon,
                    exec_horizon=exec_horizon,
                    joint_core=_effective_joint_core(),
                )
            if me:
                # Prefer ST first; then joint-core (Prioritized default / ECBS switch).
                st2 = _prioritized_st_paths(
                    starts_g, goals_g, me, blocked_plan
                )
                if (
                    st2 is not None
                    and _st_paths_ok(st2, me, goals_g)
                    and (
                        not stations
                        or _paths_avoid_station_transit(
                            st2, me, goals_g, stations
                        )
                    )
                ):
                    return st2
                jc = _wcbs_paths(starts_g, goals_g, me, tl=budget)
                if (
                    jc is not None
                    and _st_paths_ok(jc, me, goals_g)
                    and (
                        not stations
                        or _paths_avoid_station_transit(
                            jc, me, goals_g, stations
                        )
                    )
                ):
                    return jc
                if len(me) > max(2, int(wave_hard_cap)):
                    return None
                return _plan_joint_core_paths(
                    grid,
                    starts_g,
                    goals_g,
                    movers=me,
                    static=blocked_plan,
                    joint_core=_effective_joint_core(),
                    joint_plan=plan,
                    weight=weight,
                    time_limit=budget,
                    max_expansions=max_expansions,
                    plan_horizon=plan_horizon,
                    exec_horizon=exec_horizon,
                )
            return _plan_wave_paths(
                grid,
                starts_g,
                goals_g,
                movers=ma or movers_set,
                static=blocked_plan,
                planner="astar",
                joint_plan=plan,
                weight=weight,
                time_limit=budget,
                max_expansions=max_expansions,
                plan_horizon=plan_horizon,
                exec_horizon=exec_horizon,
            )

        if not assigned and not pending_delivery:
            continue

        # ---- pickup (new claims only; carriers in pending_delivery skip) ----
        # Refresh cargo maps for this wave's apply (claim-time init may be stale
        # if hybrid sticky briefly shadowed names — keep dicts authoritative).
        loaded = {n: False for n in names}
        dest = {n: "" for n in names}
        tid = {n: "" for n in names}
        global _CTX_LOADED, _CTX_DEST, _CTX_TID
        _CTX_LOADED = loaded
        _CTX_DEST = dest
        _CTX_TID = tid
        for agv, task in list(pending_delivery.items()):
            d_s = str(task.get("destination") or "")
            t_s = str(task["task_id"])
            if not _heal_same_tid_cargo_wipe(
                steps_by, agv, tid_s=t_s, dest_s=d_s
            ):
                pending_delivery.pop(agv, None)
                if t_s and t_s not in done_seen and t_s not in failed_seen:
                    last = steps_by[agv][-1]
                    pads = set(_valid_unload_pads(task, free, static))
                    last_cell = (int(last["X"]), int(last["Y"]))
                    if last_cell in pads:
                        _count_done(t_s)
                    else:
                        _requeue_task(task)
                _STICKY.pop(agv, None)
                _clear_affinity(agv)
                continue
            loaded[agv] = True
            dest[agv] = d_s
            tid[agv] = t_s
            # Sync cargo onto current stay tick only when already loaded —
            # avoids fabricating False→True on a move-in cell (r57 pickup_cell).
            last = steps_by[agv][-1]
            prev_ld = str(last.get("loaded", "")).lower() in ("true", "1", "yes")
            same = (int(last["X"]), int(last["Y"])) == pose[agv][:2]
            if prev_ld and same:
                last["loaded"] = "TRUE"
                last["destination"] = dest[agv]
                last["task-id"] = tid[agv]

        pickup_serial_done = True
        paths = None
        if not assigned:
            # Delivery-retry wave only
            pass
        else:
            pickup_serial_done = False
            pickup_extra_movers: Set[str] = set()
            starts = {n: pose[n][:2] for n in names}
            goals = {n: starts[n] for n in names}
            # Only FIFO heads remain in assigned; go straight to true pad.
            for agv, task in assigned.items():
                goals[agv] = _pk(task, starts[agv], free, static)

            # One AGV per pickup pad per wave. Joint apply holds at goal until
            # T; sharing a pad passes trimmed ST checks but collides on apply.
            reserved_pk: Set[Cell] = set()
            kept_asg: Dict[str, dict] = {}
            for agv, task in sorted(
                assigned.items(),
                key=lambda kv: (_manh(starts[kv[0]], goals[kv[0]]), kv[0]),
            ):
                g = goals[agv]
                if g in reserved_pk:
                    pending_pickup[agv] = task
                    continue
                reserved_pk.add(g)
                kept_asg[agv] = task
            if len(kept_asg) < len(assigned):
                print(
                    f"[ECBS] pickup defer shared-pad "
                    f"n={len(assigned) - len(kept_asg)} keep={len(kept_asg)}",
                    flush=True,
                )
            assigned = kept_asg
            if not assigned:
                pickup_serial_done = True
                paths = None
            else:
                goals = {n: starts[n] for n in names}
                for agv, task in assigned.items():
                    goals[agv] = _pk(task, starts[agv], free, static)

            if not assigned:
                pass
            elif pickup_conflict_spin > 0:
                # Skip joint ECBS after dump — go straight to corridor clear
                # (r34: reclaim→ECBS→fail→dump loop burned 2k+ spins).
                ld_s = {n: False for n in names}
                ds_s = {n: "" for n in names}
                td_s = {n: "" for n in names}
                for a, tsk in pending_delivery.items():
                    ld_s[a] = True
                    ds_s[a] = str(tsk.get("destination") or "")
                    td_s[a] = str(tsk.get("task_id") or "")
                kept_c: Dict[str, dict] = {}
                print(
                    f"[ECBS] spin-enter n={len(assigned)} "
                    f"agvs={list(assigned.keys())} @t={now}",
                    flush=True,
                )
                now = _separate_pose_overlaps(
                    pose,
                    steps_by,
                    now,
                    blocked_plan,
                    loaded=ld_s,
                    dest=ds_s,
                    tid=td_s,
                )
                for agv, task in list(assigned.items()):
                    goal = goals.get(agv) or _pk(
                        task, pose[agv][:2], free, static
                    )
                    # Surgical adjacent claim before fleet jam (r46 manh=1 fail).
                    now, ok = _adjacent_claim_pad(
                        pose,
                        steps_by,
                        now,
                        agv,
                        goal,
                        blocked_plan,
                        loaded=ld_s,
                        dest=ds_s,
                        tid=td_s,
                    )
                    if not ok:
                        now, ok = _clear_edge_then_claim(
                            pose,
                            steps_by,
                            now,
                            agv,
                            goal,
                            blocked_plan,
                            free,
                            loaded=ld_s,
                            dest=ds_s,
                            tid=td_s,
                        )
                        if ok:
                            print(
                                f"[ECBS] edge-claim ok agv={agv} "
                                f"goal={goal} @t={now}",
                                flush=True,
                            )
                    if not ok:
                        now, ok = _jam_fleet_unload(
                            pose,
                            steps_by,
                            now,
                            agv,
                            goal,
                            blocked_plan,
                            free,
                            loaded=ld_s,
                            dest=ds_s,
                            tid=td_s,
                        )
                    if ok:
                        kept_c[agv] = task
                    else:
                        if pickup_conflict_spin < 8 or pickup_conflict_spin % 25 == 0:
                            print(
                                f"[ECBS] jam-miss agv={agv} "
                                f"at={pose[agv][:2]} goal={goal} "
                                f"manh={_manh(pose[agv][:2], goal)} @t={now}",
                                flush=True,
                            )
                        now, ok2 = _serial_move_to(
                            pose,
                            steps_by,
                            now,
                            agv,
                            goal,
                            blocked_plan,
                            loaded=ld_s,
                            dest=ds_s,
                            tid=td_s,
                        )
                        if ok2 and pose[agv][:2] == goal:
                            kept_c[agv] = task
                        else:
                            now, ok3 = _force_step_toward(
                                pose,
                                steps_by,
                                now,
                                agv,
                                goal,
                                blocked_plan,
                                loaded=ld_s,
                                dest=ds_s,
                                tid=td_s,
                                max_steps=12,
                            )
                            if pickup_conflict_spin < 8 or pickup_conflict_spin % 25 == 0:
                                print(
                                    f"[ECBS] force-step ok={ok3} "
                                    f"at={pose[agv][:2]} goal={goal} @t={now}",
                                    flush=True,
                                )
                            if ok3 and pose[agv][:2] == goal:
                                kept_c[agv] = task
                            else:
                                pickup_spin_ban.add(agv)
                                _requeue_task(task)
                                _clear_affinity(agv)
                if kept_c:
                    assigned = kept_c
                    pickup_serial_done = True
                    pickup_conflict_spin = 0
                    pickup_soft_claim = 0
                    pickup_spin_ban.clear()
                    paths = None
                    print(
                        f"[ECBS] pickup spin corridor-first "
                        f"k={len(kept_c)} @t={now}",
                        flush=True,
                    )
                else:
                    # All corridor attempts failed — tasks already requeued above.
                    assigned = {}
                    pickup_serial_done = True
                    paths = None
                    pickup_conflict_spin += 1
                    _mk = int(meta.get("ecbs_min_wave_k") or 4)
                    pickup_soft_claim = max(2, min(4, _mk // 2))
                    pickup_claim_cooldown = 0  # reclaim next wave immediately
                    _force_wait_tick("pickup_spin_corridor_miss")
                    now = _disperse_idle_agents(
                        pose,
                        steps_by,
                        now,
                        movers=set(pending_delivery.keys())
                        | set(assigned.keys()),
                        free=free,
                        blocked_plan=blocked_plan,
                        keep_clear=set(stations) if stations else set(),
                        loaded=ld_s,
                        dest=ds_s,
                        tid=td_s,
                        aggressive=True,
                    )
                    if pickup_conflict_spin >= 48:
                        # Escape endless corridor-miss (r46: 1k+ spins @ manh=1).
                        pickup_conflict_spin = 0
                        pickup_soft_claim = 0
                        pickup_spin_ban.clear()
                        pickup_claim_cooldown = 8
                        print(
                            f"[ECBS] pickup spin hard-reset "
                            f"-> joint ECBS @t={now}",
                            flush=True,
                        )
                    elif pickup_conflict_spin < 8 or pickup_conflict_spin % 25 == 0:
                        print(
                            f"[ECBS] pickup spin corridor-miss "
                            f"spin={pickup_conflict_spin} @t={now}",
                            flush=True,
                        )
                    if not pending_delivery:
                        continue  # avoid ECBS-fail k=0; retry claim next wave
                    assigned = {}
                    pickup_serial_done = True
                    paths = None
            else:
                # Do NOT pre-stage idle AGVs here — synchronizing them to the longest
                # pickup path was inflating sim_time (wave=1 @ ~40 ticks).
                # Cars still on an unload door are not walked to the pickup
                # now. Their leave is planned with the delivery joint below.
                door_hold = _all_unload_doors(set(blocked_plan))
                for a in list(assigned):
                    if (
                        pose[a][:2] in door_hold
                        and not loaded.get(a, False)
                        and pose[a][:2] != goals.get(a)
                    ):
                        pending_pickup[a] = assigned.pop(a)
                if not assigned:
                    paths = {n: [starts[n]] for n in names}
                    pickup_serial_done = True
                else:
                    wave_movers = set(assigned)
                    paths = _route_wave(starts, goals, wave_movers)
                for _skip in list(_PICKUP_SKIP):
                    _skip_task = assigned.pop(_skip, None)
                    _PICKUP_SKIP.discard(_skip)
                    if _skip_task is not None:
                        _requeue_task(_skip_task)
                if paths is not None and (
                    int(recent_joint_fail) >= 8 or int(pickup_stall_streak) > 0
                ):
                    # ST-first success: cool the fail counter so delivery_jam
                    # does not keep triggering full-fleet disperse.
                    recent_joint_fail = max(0, int(recent_joint_fail) - 3)
            if (
                assigned
                and paths is None
                and not pickup_serial_done
                and wave_planner == "astar"
            ):
                # Baseline must stay parallel: drop hardest AGVs and retry ST together.
                items = sorted(
                    assigned.items(),
                    key=lambda kv: (
                        -_manh(starts[kv[0]], goals[kv[0]]),
                        kv[0],
                    ),
                )
                kept = dict(assigned)
                for drop_n in range(1, max(1, len(items))):
                    for agv, task in items[:drop_n]:
                        if agv in kept:
                            _detach_later_from_bag(kept, task)
                    if not kept:
                        break
                    g2 = {n: starts[n] for n in names}
                    for a in kept:
                        g2[a] = goals[a]
                    paths = _plan_wave_paths(
                        grid,
                        starts,
                        g2,
                        movers=set(kept),
                        static=blocked_plan,
                        planner="astar",
                        joint_plan=plan,
                        weight=weight,
                        time_limit=min(25.0, max(12.0, float(time_limit))),
                        max_expansions=max_expansions,
                        plan_horizon=plan_horizon,
                        exec_horizon=exec_horizon,
                    )
                    if paths is not None:
                        assigned = kept
                        goals = g2
                        print(
                            f"[ECBS] pickup baseline ST ok after drop={drop_n} "
                            f"keep={len(assigned)}",
                            flush=True,
                        )
                        break
            if (
                assigned
                and paths is None
                and not pickup_serial_done
                and wave_planner == "astar"
            ):
                # Do NOT requeue forever (infinite loop). Fall back to joint ECBS
                # for this wave only — still multi-agent parallel, not serial freeze.
                print(
                    f"[ECBS] pickup baseline ST fail k={len(assigned)} "
                    f"-> wave ECBS fallback (keep parallel)",
                    flush=True,
                )
                paths = _wcbs_paths(starts, goals, set(assigned))
            if assigned and paths is None and not pickup_serial_done:
                # ECBS (or A*→ECBS) still failing: drop farthest agents and retry joint.
                items = sorted(
                    assigned.items(),
                    key=lambda kv: (
                        -_manh(starts[kv[0]], goals[kv[0]]),
                        kv[0],
                    ),
                )
                kept = dict(assigned)
                min_wave = 4 if len(items) >= 4 else 1
                for drop_n in range(1, max(1, len(items))):
                    if len(items) - drop_n < min_wave:
                        break
                    for agv, task in items[:drop_n]:
                        if agv in kept:
                            _detach_later_from_bag(kept, task)
                    if not kept:
                        break
                    g2 = {n: starts[n] for n in names}
                    for a in kept:
                        g2[a] = goals[a]
                    paths = _wcbs_paths(starts, g2, set(kept))
                    if paths is not None:
                        assigned = kept
                        goals = g2
                        print(
                            f"[ECBS] pickup windowed-ECBS ok after drop={drop_n} "
                            f"keep={len(assigned)}",
                            flush=True,
                        )
                        break
                # If ECBS still fails at ≥4, try prioritized ST on same floor.
                if paths is None and len(assigned) >= 4:
                    items2 = sorted(
                        assigned.items(),
                        key=lambda kv: (
                            -_manh(starts[kv[0]], goals[kv[0]]),
                            kv[0],
                        ),
                    )
                    keep4 = dict(items2[: max(4, (len(items2) + 1) // 2)])
                    g2 = {n: starts[n] for n in names}
                    for a in keep4:
                        g2[a] = goals[a]
                    paths = _prioritized_st_paths(
                        starts, g2, set(keep4), blocked_plan
                    )
                    if paths is not None and _st_paths_ok(paths, set(keep4), g2):
                        assigned = keep4
                        goals = g2
                        print(
                            f"[ECBS] pickup ST-floor after ECBS fail "
                            f"k={len(assigned)}",
                            flush=True,
                        )
                    else:
                        paths = None
            if paths is None and not pickup_serial_done:
                recent_joint_fail += 1
                # Prefer keep parallel large wave: peel to half (floor 4) instead
                # of serial A* — never stabilize on 1–2 vehicle waves.
                min_keep = int(meta.get("ecbs_min_wave_k") or 6)
                if len(assigned) >= 4:
                    items = sorted(
                        assigned.items(),
                        key=lambda kv: (
                            -_manh(starts[kv[0]], goals[kv[0]]),
                            kv[0],
                        ),
                    )
                    keep_n = max(4, min(len(items), max(min_keep // 2, (len(items) + 1) // 2)))
                    kept = dict(items[:keep_n])
                    for agv, task in items[keep_n:]:
                        _detach_later_from_bag(assigned, task)
                    g2 = {n: starts[n] for n in names}
                    for a in kept:
                        g2[a] = goals[a]
                    paths = _wcbs_paths(starts, g2, set(kept))
                    if paths is None:
                        paths = _prioritized_st_paths(starts, g2, set(kept), blocked_plan)
                        if paths is not None and not _st_paths_ok(
                            paths, set(kept), g2
                        ):
                            paths = None
                    if paths is not None:
                        assigned = kept
                        goals = g2
                        print(
                            f"[ECBS] pickup keep-parallel after fail "
                            f"k={len(assigned)} (min_k={min_keep})",
                            flush=True,
                        )
                if paths is None:
                    print(
                        f"[ECBS] pickup {wave_planner} fail k={len(assigned)} "
                        f"-> joint-evac+wave",
                        flush=True,
                    )
                    # r90: evacuate 1 idle to a park WHILE jointly ST-planning
                    # the pickup wave — do not wait for full serial disperse.
                    keep_cells: Set[Cell] = set()
                    for agv, task in assigned.items():
                        keep_cells.add(pose[agv][:2])
                        keep_cells.add(
                            _pk(task, pose[agv][:2], free, static)
                        )
                    ld0 = {n: False for n in names}
                    ds0 = {n: "" for n in names}
                    td0 = {n: "" for n in names}
                    for agv, task in pending_delivery.items():
                        ld0[agv] = True
                        ds0[agv] = str(task.get("destination") or "")
                        td0[agv] = str(task.get("task_id") or "")
                    s_je = {n: pose[n][:2] for n in names}
                    g_je = {n: s_je[n] for n in names}
                    for a, task in assigned.items():
                        g_je[a] = _pk(task, s_je[a], free, static)
                    joint_e = _joint_wave_with_evac(
                        s_je,
                        g_je,
                        set(assigned.keys()),
                        blocked_plan,
                        pose,
                        free,
                        keep_cells,
                        names,
                        n_evac=1,
                        stations=stations,
                    )
                    if joint_e is not None:
                        paths, mv_e, goals = joint_e
                        starts = s_je
                        pickup_extra_movers.clear()
                        pickup_extra_movers |= set(mv_e) - set(assigned.keys())
                        pickup_serial_done = False
                        pickup_stall_streak = 0
                        print(
                            f"[ECBS] pickup joint-evac+wave ok "
                            f"k={len(assigned)} evac={len(pickup_extra_movers)} "
                            f"@t={now}",
                            flush=True,
                        )
                    if paths is None:
                        # Fallback: serial disperse then retry (legacy).
                        now = _disperse_idle_agents(
                            pose,
                            steps_by,
                            now,
                            movers=set(assigned.keys()),
                            free=free,
                            blocked_plan=blocked_plan,
                            keep_clear=keep_cells,
                            loaded=ld0,
                            dest=ds0,
                            tid=td0,
                        )
                    # After disperse, retry a short classic ECBS for small waves.
                    if paths is None and len(assigned) <= 3:
                        g2 = {n: pose[n][:2] for n in names}
                        s2 = {n: pose[n][:2] for n in names}
                        for a, task in assigned.items():
                            g2[a] = _pk(task, s2[a], free, static)
                        paths = _ecbs(
                            grid,
                            s2,
                            g2,
                            weight=max(weight, 1.8),
                            time_limit=min(20.0, max(8.0, float(time_limit))),
                            max_expansions=max_expansions,
                            movers=set(assigned.keys()),
                            plan=wave_plan,
                        )
                        if (
                            paths is not None
                            and not _cell_paths_conflict(paths, names)
                            and (
                                not stations
                                or _paths_avoid_station_transit(
                                    paths, set(assigned.keys()), g2, stations
                                )
                            )
                        ):
                            starts = s2
                            goals = g2
                            pickup_serial_done = False
                            pickup_stall_streak = 0
                            print(
                                f"[ECBS] pickup classic-ECBS ok after disperse k={len(assigned)}",
                                    flush=True,
                                )
                            # fall through to joint apply below
                        else:
                            paths = None
                    if paths is None:
                        ranked_pk = sorted(
                            list(assigned.items()),
                            key=lambda kv: (
                                _manh(
                                    pose[kv[0]][:2],
                                    _pk(kv[1], pose[kv[0]][:2], free, static),
                                ),
                                kv[0],
                            ),
                        )
                        batch_pk = list(ranked_pk)
                        while len(batch_pk) >= 2 and paths is None:
                            s_pk = {n: pose[n][:2] for n in names}
                            g_pk = {n: s_pk[n] for n in names}
                            movers_pk = set()
                            for agv, task in batch_pk:
                                g_pk[agv] = _pk(task, s_pk[agv], free, static)
                                movers_pk.add(agv)
                            sp_pk = _turn_aware_cell_paths(
                                pose,
                                g_pk,
                                movers_pk,
                                set(blocked_plan),
                                prefer_outer_ring=len(movers_pk) >= 2,
                            )
                            if sp_pk is None or not _st_paths_ok(
                                sp_pk, movers_pk, g_pk
                            ):
                                sp_pk = _prioritized_st_paths(
                                    s_pk,
                                    g_pk,
                                    movers_pk,
                                    blocked_plan,
                                    hold_after=0,
                                )
                            if sp_pk is not None and _st_paths_ok(
                                sp_pk, movers_pk, g_pk
                            ):
                                paths = sp_pk
                                assigned = dict(batch_pk)
                                goals = g_pk
                                starts = s_pk
                                pickup_serial_done = False
                                print(
                                    f"[ECBS] pickup shrink-ST k={len(batch_pk)} "
                                    f"@t={now}",
                                    flush=True,
                                )
                                break
                            drop_agv, drop_task = batch_pk.pop()
                            pending_pickup[drop_agv] = drop_task
                            print(
                                f"[ECBS] pickup shrink {len(batch_pk) + 1}"
                                f"->{len(batch_pk)} drop={drop_agv} @t={now}",
                                flush=True,
                            )
                        if paths is None:
                            for agv, task in batch_pk:
                                pending_pickup[agv] = task
                            assigned = {}
                            pickup_serial_done = True
                            print(
                                f"[ECBS] pickup shrink stop "
                                f"defer={len(batch_pk)} @t={now}",
                                flush=True,
                            )
                        if not assigned:
                            if pending_delivery:
                                # Still need to retry carriers; skip stall accounting.
                                pickup_serial_done = True
                            else:
                                pickup_stall_streak += 1
                                print(
                                    f"[ECBS] FAIL pickup wave @t={now} "
                                    f"(drop wave, stall={pickup_stall_streak})",
                                    flush=True,
                                )
                                # Advance lightly — full-fleet disperse every stall
                                # burned tens of thousands of sim ticks (r17).
                                _force_wait_tick(f"pickup_stall_{pickup_stall_streak}")
                                if pickup_stall_streak in (1, 3, 6, 12, 24):
                                    inflight_station.clear()
                                    print(
                                        "[ECBS] pickup stall -> clear inflight_station",
                                        flush=True,
                                    )
                                if pickup_stall_streak % 4 == 0:
                                    # Never wipe cargo on disperse (r78: FALSE
                                    # frames + later force-load → pickup_cell at
                                    # unload hubs). Park only true idles.
                                    _ld_d = {
                                        n: (n in pending_delivery) for n in names
                                    }
                                    _ds_d = {
                                        n: (
                                            str(
                                                pending_delivery[n].get(
                                                    "destination"
                                                )
                                                or ""
                                            )
                                            if n in pending_delivery
                                            else ""
                                        )
                                        for n in names
                                    }
                                    _td_d = {
                                        n: (
                                            str(
                                                pending_delivery[n].get(
                                                    "task_id"
                                                )
                                                or ""
                                            )
                                            if n in pending_delivery
                                            else ""
                                        )
                                        for n in names
                                    }
                                    now = _disperse_idle_agents(
                                        pose,
                                        steps_by,
                                        now,
                                        movers=set(pending_delivery.keys())
                                        | set(assigned.keys()),
                                        free=free,
                                        blocked_plan=blocked_plan,
                                        keep_clear=set(stations)
                                        if stations
                                        else set(),
                                        loaded=_ld_d,
                                        dest=_ds_d,
                                        tid=_td_d,
                                        aggressive=False,
                                    )
                                pickup_soft_claim = max(4, min(8, 10 - pickup_stall_streak // 2))
                                # Never reorder surface FIFO (SH12: rotate-head
                                # skipped Horse-34/35 → surface_fifo/surface_head).
                                # Never mass-fail remaining queue (r17 abort → ratio 0.8).
                                if pickup_stall_streak >= 80:
                                    print(
                                        f"[ECBS] pickup stall soft-reset @t={now} "
                                        f"remaining_q={sum(len(v) for v in queues.values())}",
                                        flush=True,
                                    )
                                    inflight_station.clear()
                                    pickup_stall_streak = 20
                                    pickup_soft_claim = 4
                                    _force_wait_tick("pickup_stall_soft_reset")
                                continue
                        else:
                            pickup_serial_done = True
                            pickup_stall_streak = 0
                            pickup_soft_claim = 0
            elif paths is not None:
                recent_joint_fail = max(0, recent_joint_fail - 1)
                pickup_serial_done = False
                pickup_stall_streak = 0

        if not assigned and not pending_delivery:
            # Door cars were pulled out of this pickup apply. Still plan their
            # leave in the delivery joint below, even when nobody is carrying.
            _doors_hold = _all_unload_doors(set(blocked_plan))
            if not any(
                pose[n][:2] in _doors_hold
                and not loaded.get(n, False)
                and n not in _STICKY
                for n in names
            ):
                continue

        # Apply joint pickup paths when we did not already serial-move.
        if assigned and not pickup_serial_done:
            assert paths is not None
            apply_movers = set(assigned.keys()) | set(pickup_extra_movers)
            # Unique pads this wave + idle-seeded ST. Gate on trimmed cell
            # conflicts; pose expansion keeps idlers out of the mover sync loop.
            if not _st_paths_ok(paths, apply_movers, goals):
                ld0 = {n: False for n in names}
                ds0 = {n: "" for n in names}
                td0 = {n: "" for n in names}
                for agv, task in pending_delivery.items():
                    ld0[agv] = True
                    ds0[agv] = str(task.get("destination") or "")
                    td0[agv] = str(task.get("task_id") or "")
                timelines_try = _plan_paths_as_poses(pose, paths)
                applied = False
                # Never apply movers-only paths (r32: 120k collisions — ghost
                # through parked idlers). Evacuate then full-gate replan only.
                if _st_movers_only_ok(paths, set(assigned), goals) and _pose_movers_ok(
                    pose, timelines_try, set(assigned)
                ):
                    # Clear every cell on mover paths before replan/apply.
                    blocked_cells: Set[Cell] = set()
                    for a in assigned:
                        for c in paths.get(a) or []:
                            blocked_cells.add(c)
                        blocked_cells.add(goals[a])
                    now = _disperse_idle_agents(
                        pose,
                        steps_by,
                        now,
                        movers=set(assigned.keys()),
                        free=free,
                        blocked_plan=blocked_plan,
                        keep_clear=blocked_cells,
                        loaded=ld0,
                        dest=ds0,
                        tid=td0,
                        aggressive=True,
                    )
                    s_r = {n: pose[n][:2] for n in names}
                    g_r = {n: s_r[n] for n in names}
                    for a, tsk in assigned.items():
                        g_r[a] = _pk(tsk, s_r[a], free, static)
                    sp_r = _prioritized_st_paths(
                        s_r,
                        g_r,
                        set(assigned.keys()),
                        blocked_plan,
                        hold_after=0,
                    )
                    if (
                        sp_r is not None
                        and _st_paths_ok(sp_r, set(assigned.keys()), g_r)
                    ):
                        tl_r = _plan_paths_as_poses(pose, sp_r)
                        # Full fleet: a door car frozen on the corridor is not
                        # in `assigned`, and a movers-only check walks through it.
                        if not _pose_timelines_conflict(
                            pose, tl_r, pad_holds=True
                        ):
                            now = _apply_pose_timelines(
                                steps_by,
                                pose,
                                tl_r,
                                now,
                                loaded=ld0,
                                dest=ds0,
                                tid=td0,
                                rise_ok={},
)
                            applied = True
                            pickup_serial_done = True
                            pickup_conflict_spin = 0
                            goals.update(
                                {a: g_r[a] for a in assigned if a in g_r}
                            )
                            print(
                                f"[ECBS] pickup clear+ST apply "
                                f"k={len(assigned)} @t={now}",
                                flush=True,
                            )
                if not applied:
                    # 2) Disperse + fresh ST peel (≥4), movers-only gate.
                    print(
                        "[ECBS] pickup cell conflict -> shrink+parallel ST",
                        flush=True,
                    )
                    now = _disperse_idle_agents(
                        pose,
                        steps_by,
                        now,
                        movers=set(assigned.keys()),
                        free=free,
                        blocked_plan=blocked_plan,
                        keep_clear={pose[a][:2] for a in assigned}
                        | {goals[a] for a in assigned},
                        loaded=ld0,
                        dest=ds0,
                        tid=td0,
                        aggressive=True,
                    )
                    ranked = sorted(
                        assigned.items(),
                        key=lambda kv: (
                            _manh(
                                pose[kv[0]][:2],
                                _pk(
                                    kv[1], pose[kv[0]][:2], free, static
                                ),
                            ),
                            kv[0],
                        ),
                    )
                    peel_n = (
                        max(4, (len(ranked) + 1) // 2)
                        if len(ranked) >= 4
                        else len(ranked)
                    )
                    keep = dict(ranked[:peel_n])
                    for agv, task in ranked[len(keep) :]:
                        _requeue_task(task)
                        _clear_affinity(agv)
                    s_k = {n: pose[n][:2] for n in names}
                    g_k = {n: s_k[n] for n in names}
                    for a, tsk in keep.items():
                        g_k[a] = _pk(tsk, s_k[a], free, static)
                    sp_k = _prioritized_st_paths(
                        s_k,
                        g_k,
                        set(keep.keys()),
                        blocked_plan,
                        hold_after=0,
                    )
                    kept_ok: Dict[str, dict] = {}
                    if sp_k is not None and _st_paths_ok(
                        sp_k, set(keep.keys()), g_k
                    ):
                        timelines = _plan_paths_as_poses(pose, sp_k)
                        if not _pose_timelines_conflict(
                            pose, timelines, pad_holds=True
                        ):
                            now = _apply_pose_timelines(
                                steps_by,
                                pose,
                                timelines,
                                now,
                                loaded=ld0,
                                dest=ds0,
                                tid=td0,
                                rise_ok={},
)
                            kept_ok = dict(keep)
                            print(
                                f"[ECBS] pickup recover parallel-ST k={len(keep)} "
                                f"@t={now}",
                                flush=True,
                            )
                    if not kept_ok:
                        # 2b) Corridor escape: try parallel floor first when
                        # ecbs_min_wave_k demands large waves. Skip k=1 unless
                        # the wave is already tiny (r110 late wave=1 blew sim).
                        ranked_keep = list(keep.items())
                        _mk = int(meta.get("ecbs_min_wave_k") or 4)
                        try_ns: List[int] = []
                        if len(ranked_keep) >= 4:
                            try_ns.append(min(len(ranked_keep), max(4, _mk // 2)))
                            try_ns.append(min(len(ranked_keep), 4))
                        elif len(ranked_keep) > 1:
                            try_ns.append(len(ranked_keep))
                            try_ns.append(min(2, len(ranked_keep)))
                        if len(ranked_keep) <= 2:
                            try_ns.append(1)
                        _dedup: List[int] = []
                        for _n in try_ns:
                            if _n > 0 and _n not in _dedup:
                                _dedup.append(_n)
                        try_ns = _dedup
                        kept_serial: Dict[str, dict] = {}
                        tried: Set[str] = set()
                        for serial_n in try_ns:
                            for agv, task in ranked_keep[:serial_n]:
                                if agv in tried:
                                    continue
                                tried.add(agv)
                                goal = _pk(
                                    task, pose[agv][:2], free, static
                                )
                                now, ok = _jam_fleet_unload(
                                    pose,
                                    steps_by,
                                    now,
                                    agv,
                                    goal,
                                    blocked_plan,
                                    free,
                                    loaded=ld0,
                                    dest=ds0,
                                    tid=td0,
                                )
                                if ok:
                                    kept_serial[agv] = task
                            if kept_serial:
                                break
                        for agv, task in ranked_keep:
                            if agv in kept_serial:
                                continue
                            _requeue_task(task)
                            _clear_affinity(agv)
                        if kept_serial:
                            assigned = kept_serial
                            pickup_serial_done = True
                            pickup_conflict_spin = 0
                            print(
                                f"[ECBS] pickup conflict corridor "
                                f"k={len(kept_serial)} @t={now}",
                                flush=True,
                            )
                        else:
                            # Only requeue deferred pending not already handled.
                            done_ids = {
                                str(t.get("task_id") or "")
                                for _, t in ranked_keep
                            }
                            for agv, task in list(pending_pickup.items()):
                                tid_p = str(task.get("task_id") or "")
                                if tid_p not in done_ids:
                                    _requeue_task(task)
                                _clear_affinity(agv)
                            pending_pickup.clear()
                            assigned = {}
                            pickup_conflict_spin += 1
                            _mk = int(meta.get("ecbs_min_wave_k") or 4)
                            pickup_soft_claim = max(2, min(4, _mk // 2))
                            pickup_claim_cooldown = max(
                                pickup_claim_cooldown, 2
                            )
                            _repair_inflight()
                            _force_wait_tick("pickup_conflict_dump")
                            # Preserve carrier cargo flags (r78 pickup_cell storm).
                            _ld_d = {
                                n: (n in pending_delivery) for n in names
                            }
                            _ds_d = {
                                n: (
                                    str(
                                        pending_delivery[n].get("destination")
                                        or ""
                                    )
                                    if n in pending_delivery
                                    else ""
                                )
                                for n in names
                            }
                            _td_d = {
                                n: (
                                    str(
                                        pending_delivery[n].get("task_id")
                                        or ""
                                    )
                                    if n in pending_delivery
                                    else ""
                                )
                                for n in names
                            }
                            now = _disperse_idle_agents(
                                pose,
                                steps_by,
                                now,
                                movers=set(pending_delivery.keys())
                                | set(assigned.keys()),
                                free=free,
                                blocked_plan=blocked_plan,
                                keep_clear=set(stations) if stations else set(),
                                loaded=_ld_d,
                                dest=_ds_d,
                                tid=_td_d,
                                aggressive=True,
                            )
                            print(
                                f"[ECBS] pickup conflict dump "
                                f"spin={pickup_conflict_spin} "
                                f"cd={pickup_claim_cooldown} @t={now}",
                                flush=True,
                            )
                            pickup_serial_done = True
                    else:
                        assigned = kept_ok
                        pickup_serial_done = True
                        pickup_conflict_spin = 0
            else:
                timelines = _plan_paths_as_poses(pose, paths)
                apply_movers = set(assigned.keys()) | set(pickup_extra_movers)
                mover_pose = {a: pose[a] for a in apply_movers}
                mover_tl = {a: timelines.get(a) or [] for a in apply_movers}
                if _pose_timelines_conflict(
                    mover_pose, mover_tl, pad_holds=False
                ) or _pose_timelines_conflict(pose, timelines, pad_holds=True):
                    print(
                        "[ECBS] pickup pose conflict -> shrink+parallel ST",
                        flush=True,
                    )
                    ranked = sorted(
                        assigned.items(),
                        key=lambda kv: (
                            _manh(
                                pose[kv[0]][:2],
                                _pk(
                                    kv[1], pose[kv[0]][:2], free, static
                                ),
                            ),
                            kv[0],
                        ),
                    )
                    keep = dict(
                        ranked[
                            : (
                                max(4, (len(ranked) + 1) // 2)
                                if len(ranked) >= 4
                                else len(ranked)
                            )
                        ]
                    )
                    for agv, task in ranked[len(keep) :]:
                        pending_pickup[agv] = task
                    ld0 = {n: False for n in names}
                    ds0 = {n: "" for n in names}
                    td0 = {n: "" for n in names}
                    for agv, task in pending_delivery.items():
                        ld0[agv] = True
                        ds0[agv] = str(task.get("destination") or "")
                        td0[agv] = str(task.get("task_id") or "")
                    now = _disperse_idle_agents(
                        pose,
                        steps_by,
                        now,
                        movers=set(keep.keys()),
                        free=free,
                        blocked_plan=blocked_plan,
                        keep_clear={pose[a][:2] for a in keep}
                        | {
                            _pk(t, pose[a][:2], free, static)
                            for a, t in keep.items()
                        },
                        loaded=ld0,
                        dest=ds0,
                        tid=td0,
                        aggressive=True,
                    )
                    s_k = {n: pose[n][:2] for n in names}
                    g_k = {n: s_k[n] for n in names}
                    for a, tsk in keep.items():
                        g_k[a] = _pk(tsk, s_k[a], free, static)
                    sp_k = _prioritized_st_paths(
                        s_k,
                        g_k,
                        set(keep.keys()),
                        blocked_plan,
                        hold_after=0,
                    )
                    kept_ok2: Dict[str, dict] = {}
                    if sp_k is not None and _st_paths_ok(
                        sp_k, set(keep.keys()), g_k
                    ):
                        timelines2 = _plan_paths_as_poses(pose, sp_k)
                        if not _pose_timelines_conflict(
                            pose, timelines2, pad_holds=True
                        ):
                            now = _apply_pose_timelines(
                                steps_by,
                                pose,
                                timelines2,
                                now,
                                loaded=ld0,
                                dest=ds0,
                                tid=td0,
                                rise_ok={},
)
                            kept_ok2 = dict(keep)
                            print(
                                f"[ECBS] pickup pose recover parallel-ST "
                                f"k={len(keep)} @t={now}",
                                flush=True,
                            )
                    if not kept_ok2:
                        batch_k = sorted(
                            list(keep.items()),
                            key=lambda kv: (
                                _manh(
                                    pose[kv[0]][:2],
                                    _pk(kv[1], pose[kv[0]][:2], free, static),
                                ),
                                kv[0],
                            ),
                        )
                        while len(batch_k) >= 2 and not kept_ok2:
                            s_k2 = {n: pose[n][:2] for n in names}
                            g_k2 = {n: s_k2[n] for n in names}
                            movers_k2 = {a for a, _t in batch_k}
                            for a, tsk in batch_k:
                                g_k2[a] = _pk(tsk, s_k2[a], free, static)
                            sp2 = _prioritized_st_paths(
                                s_k2,
                                g_k2,
                                movers_k2,
                                blocked_plan,
                                hold_after=0,
                            )
                            if sp2 is not None and _st_paths_ok(
                                sp2, movers_k2, g_k2
                            ):
                                tl2 = _plan_paths_as_poses(pose, sp2)
                                if not _pose_timelines_conflict(
                                    pose, tl2, pad_holds=True
                                ):
                                    now = _apply_timelines_until_first_goal(
                                        steps_by,
                                        pose,
                                        tl2,
                                        g_k2,
                                        movers_k2,
                                        now,
                                        loaded=ld0,
                                        dest=ds0,
                                        tid=td0,
                                        rise_ok={},
                                    )
                                    kept_ok2 = dict(batch_k)
                                    print(
                                        f"[ECBS] pickup pose shrink-ST "
                                        f"k={len(batch_k)} @t={now}",
                                        flush=True,
                                    )
                                    break
                            drop_a, drop_t = batch_k.pop()
                            pending_pickup[drop_a] = drop_t
                            print(
                                f"[ECBS] pickup pose shrink "
                                f"{len(batch_k) + 1}->{len(batch_k)} "
                                f"drop={drop_a} @t={now}",
                                flush=True,
                            )
                        if not kept_ok2:
                            for agv, task in batch_k:
                                pending_pickup[agv] = task
                            pickup_conflict_spin += 1
                            pickup_claim_cooldown = max(
                                pickup_claim_cooldown,
                                min(4, 1 + pickup_conflict_spin // 2),
                            )
                            _force_wait_tick("pickup_pose_conflict")
                        else:
                            pickup_conflict_spin = 0
                    assigned = kept_ok2
                    pickup_serial_done = True
                else:
                    now = _apply_timelines_until_first_goal(
                        steps_by,
                        pose,
                        timelines,
                        goals,
                        set(assigned.keys()),
                        now,
                        loaded=loaded,
                        dest=dest,
                        tid=tid,
                        rise_ok={},
                    )

        # Commit pickup ONLY on the true station pad, and only FIFO head.
        committed: Dict[str, dict] = {}
        committed_stations: Set[str] = set()

        def _inflight_rank(agv_name: str) -> tuple:
            task = assigned[agv_name]
            tid_s = str(task.get("task_id") or "")
            st_name = _task_station(task)
            q_inf = inflight_station.get(st_name) or []
            try:
                return (st_name, q_inf.index(tid_s))
            except ValueError:
                return (st_name, 10**6)

        # First door arrival already ended this pickup slice. Agents still
        # short of the door stay in pending_pickup and resume next wave.

        for agv in sorted(assigned.keys(), key=_inflight_rank):
            task = assigned[agv]
            if agv in _PICKUP_SKIP:
                _PICKUP_SKIP.discard(agv)
                _requeue_task(task)
                continue
            true_pk = _pk(task, pose[agv][:2], free, static)
            st_name = _task_station(task)
            tid_s = str(task.get("task_id") or "")
            if not _is_station_head(task) or st_name in committed_stations:
                pending_pickup[agv] = task
                continue
            if pose[agv][:2] != true_pk:
                # Not at the door yet. Resume next wave instead of
                # walking this AGV alone while the fleet stands.
                pending_pickup[agv] = task
                continue
            # Pickup is the door beside the station, not the station center.
            # The center is a dead end. r101 forbade the whole leave-ring
            # (Ox rose on Rabbit's corridor); the door is the one inward cell.
            st_center = true_pickups.get(st_name)
            door = _pickup_approach(st_center) if st_center else true_pk
            px, py = int(pose[agv][0]), int(pose[agv][1])
            seq_last = (steps_by.get(agv) or [None])[-1]
            last_cell = (
                (int(seq_last["X"]), int(seq_last["Y"])) if seq_last else None
            )
            last_pitch = (
                int(seq_last.get("pitch") or 0) % 360 if seq_last else -1
            )
            # Next tick is the 1s dwell. It only counts if this tick is
            # already the door, and that dwell second does not turn.
            on_door = (px, py) == door and last_cell == door
            if (
                not on_door
                or not _is_station_head(task)
                or st_name in committed_stations
            ):
                pending_pickup[agv] = task
                continue
            # Refuse pickup while still carrying another task (off-pad clear
            # guards left ghost cargo → tid overwrite → FIFO skip).
            cur_tid = str(tid.get(agv) or "").strip()
            if bool(loaded.get(agv, False)) and cur_tid and cur_tid != tid_s:
                print(
                    f"[ECBS] defer pickup {tid_s}: still carrying {cur_tid} "
                    f"@t={now}",
                    flush=True,
                )
                _requeue_task(task)
                if agv not in pending_delivery:
                    pending_delivery[agv] = {
                        "task_id": cur_tid,
                        "destination": str(dest.get(agv) or ""),
                        "end_points": [
                            list(c) for c in sorted(_TID_DROPOFFS.get(cur_tid) or [])
                        ],
                    }
                continue
            prev_row = steps_by[agv][-1] if steps_by.get(agv) else None
            if prev_row is not None:
                prev_ld = str(prev_row.get("loaded", "")).lower() in (
                    "true",
                    "1",
                    "yes",
                )
                prev_tid = str(prev_row.get("task-id") or "").strip()
                if prev_ld and prev_tid and prev_tid != tid_s:
                    print(
                        f"[ECBS] defer pickup {tid_s}: traj still carrying "
                        f"{prev_tid} @t={now}",
                        flush=True,
                    )
                    _requeue_task(task)
                    if agv not in pending_delivery:
                        pending_delivery[agv] = {
                            "task_id": prev_tid,
                            "destination": str(prev_row.get("destination") or ""),
                            "end_points": [
                                list(c)
                                for c in sorted(_TID_DROPOFFS.get(prev_tid) or [])
                            ],
                        }
                        loaded[agv] = True
                        tid[agv] = prev_tid
                        dest[agv] = str(prev_row.get("destination") or "")
                    continue
            pose[agv] = (px, py, last_pitch)
            tid[agv] = tid_s
            loaded[agv] = True
            dest[agv] = str(task.get("destination") or "")
            if tid[agv]:
                picked_ids.add(tid[agv])
                q_inf = inflight_station.get(st_name) or []
                if q_inf and q_inf[0] == tid[agv]:
                    q_inf.pop(0)
                    if not q_inf:
                        inflight_station.pop(st_name, None)
                    else:
                        inflight_station[st_name] = q_inf
                # Pop queue head now that rising-edge is committed (handoff
                # no longer pops on approach bind).
                q_st = queues.get(st_name) or []
                if q_st and str(q_st[0].get("task_id") or "") == tid[agv]:
                    q_st.pop(0)
                    if not q_st:
                        queues.pop(st_name, None)
            work_count[agv] = int(work_count.get(agv, 0)) + 1
            committed[agv] = task
            committed_stations.add(st_name)
        if committed:
            # Rising edge on the next stay tick: door cell, pitch unchanged.
            rise_ok: Dict[str, Set[Cell]] = {}
            for agv, task in committed.items():
                st_n = _task_station(task)
                center = true_pickups.get(st_n)
                pad = (
                    _pickup_approach(center)
                    if center
                    else _pk(task, pose[agv][:2], free, static)
                )
                rise_ok[agv] = {pad}
            now = _fleet_hold_tick(
                steps_by,
                pose,
                names,
                now,
                loaded=loaded,
                dest=dest,
                tid=tid,
                rise_ok=rise_ok,
            )
            pickup_conflict_spin = 0
            pickup_soft_claim = 0
        assigned = committed

        door_now = _all_unload_doors(set(blocked_plan))
        on_door_idle = any(
            pose[n][:2] in door_now
            and not loaded.get(n, False)
            and n not in _STICKY
            and n not in pending_delivery
            for n in names
        )
        if not assigned and not pending_delivery and not on_door_idle:
            if pending_pickup:
                pickup_conflict_spin += 1
                pickup_soft_claim = max(4, min(8, 10 - pickup_conflict_spin))
                if pickup_conflict_spin >= 3:
                    # Break identical pending set: keep closest ≥4, requeue rest.
                    items = sorted(
                        pending_pickup.items(),
                        key=lambda kv: (
                            _manh(
                                pose[kv[0]][:2],
                                _pk(
                                    kv[1], pose[kv[0]][:2], free, static
                                ),
                            ),
                            kv[0],
                        ),
                    )
                    keep_n = (
                        max(4, min(len(items), int(meta.get("ecbs_min_wave_k") or 6)))
                        if len(items) >= 4
                        else max(1, len(items) // 2)
                    )
                    for agv, task in items[keep_n:]:
                        pending_pickup.pop(agv, None)
                        _requeue_task(task)
                        _clear_affinity(agv)
                    ld_s = {n: False for n in names}
                    ds_s = {n: "" for n in names}
                    td_s = {n: "" for n in names}
                    for agv, task in pending_delivery.items():
                        ld_s[agv] = True
                        ds_s[agv] = str(task.get("destination") or "")
                        td_s[agv] = str(task.get("task_id") or "")
                    now = _disperse_idle_agents(
                        pose,
                        steps_by,
                        now,
                        movers=set(pending_pickup.keys()),
                        free=free,
                        blocked_plan=blocked_plan,
                        keep_clear={pose[a][:2] for a in pending_pickup}
                        | {
                            _pk(t, pose[a][:2], free, static)
                            for a, t in pending_pickup.items()
                        },
                        loaded=ld_s,
                        dest=ds_s,
                        tid=td_s,
                        aggressive=True,
                    )
                    print(
                        f"[ECBS] pickup spin-break keep={len(pending_pickup)} "
                        f"requeued={max(0, len(items) - keep_n)} @t={now}",
                        flush=True,
                    )
                    pickup_conflict_spin = 0
                continue
            continue

        # ---- delivery ----
        starts = {n: pose[n][:2] for n in names}
        goals = {n: starts[n] for n in names}
        deliverable: Dict[str, dict] = {}
        reserved_goals: Set[Cell] = set()
        unique_movers: Set[str] = set()
        delivery_agents = dict(pending_delivery)
        for agv, task in assigned.items():
            if loaded.get(agv):
                delivery_agents[agv] = task
        for agv, task in sorted(delivery_agents.items()):
            # Unique pad per joint wave; NEVER use maze-wall end_points as goals.
            walk = _valid_unload_pads(task, free, static, prefer=starts[agv])
            if not walk:
                # Keep cargo — clearing off-pad writes premature_unload in traj.
                print(
                    f"[ECBS] no free unload pad for {task['task_id']} "
                    f"-> hold cargo @t={now}",
                    flush=True,
                )
                deliverable[agv] = task
                continue
            cands = [c for c in walk if c not in reserved_goals]
            if cands:
                goals[agv] = min(cands, key=lambda c: (_manh(starts[agv], c), c))
                reserved_goals.add(goals[agv])
                unique_movers.add(agv)
            else:
                # Share a true pad (ST time-multiplex); never unload on staging
                # neighbors — that caused task_carry violations.
                goals[agv] = min(walk, key=lambda c: (_manh(starts[agv], c), c))
            deliverable[agv] = task

        if not deliverable:
            print(f"[ECBS] no deliverable goals @t={now}", flush=True)

        # Carriers must stay loaded through every delivery apply/hold.
        for agv in list(pending_delivery.keys()) + list(deliverable.keys()):
            task_c = pending_delivery.get(agv) or deliverable.get(agv)
            t_fix = str(
                tid.get(agv)
                or ((task_c or {}).get("task_id") if task_c else "")
                or ""
            )
            d_fix = str(
                dest.get(agv)
                or ((task_c or {}).get("destination") if task_c else "")
                or ""
            )
            if agv in pending_delivery and task_c is not None:
                if not _heal_same_tid_cargo_wipe(
                    steps_by, agv, tid_s=t_fix, dest_s=d_fix
                ):
                    pending_delivery.pop(agv, None)
                    deliverable.pop(agv, None)
                    if t_fix and t_fix not in done_seen and t_fix not in failed_seen:
                        last = steps_by[agv][-1]
                        pads = set(_valid_unload_pads(task_c, free, static))
                        last_cell = (int(last["X"]), int(last["Y"]))
                        if last_cell in pads:
                            _count_done(t_fix)
                        else:
                            _requeue_task(task_c)
                    _STICKY.pop(agv, None)
                    _clear_affinity(agv)
                    continue
            last = steps_by[agv][-1]
            prev_ld = str(last.get("loaded", "")).lower() in ("true", "1", "yes")
            same = (int(last["X"]), int(last["Y"])) == pose[agv][:2]
            if not prev_ld and task_c is not None:
                # No invent (r100). Only commit+rise_ok may create False→True.
                pending_delivery.pop(agv, None)
                deliverable.pop(agv, None)
                if agv in assigned:
                    pending_pickup[agv] = task_c
                    assigned.pop(agv, None)
                elif t_fix and t_fix not in done_seen and t_fix not in failed_seen:
                    _requeue_task(task_c)
                _STICKY.pop(agv, None)
                _clear_affinity(agv)
                loaded[agv] = False
                dest[agv] = ""
                tid[agv] = ""
                continue
            loaded[agv] = True
            if t_fix and prev_ld and same:
                # Maintain annotation only — never flip FALSE→TRUE here.
                if str(last.get("task-id") or "").strip() == t_fix:
                    last["destination"] = d_fix
                    dest[agv] = d_fix
                    tid[agv] = t_fix
                else:
                    last["destination"] = d_fix
                    last["task-id"] = t_fix
                    dest[agv] = d_fix
                    tid[agv] = t_fix
        # Hub jam: drain with A*-follow 1-hop BEFORE joint ECBS (r28: recover
        # never reached a pad; joint fail counter spun to 700).
        if int(recent_joint_fail) >= 6 and deliverable:
            n_ok = 0
            ranked = sorted(
                deliverable.keys(),
                key=lambda a: (
                    _manh(
                        pose[a][:2],
                        _valid_unload_pads(
                            deliverable[a], free, static, prefer=pose[a][:2]
                        )[0]
                        if _valid_unload_pads(
                            deliverable[a], free, static, prefer=pose[a][:2]
                        )
                        else pose[a][:2],
                    ),
                    a,
                ),
            )
            print(
                f"[ECBS] jam unload try k={len(ranked)} fail={recent_joint_fail} "
                f"@t={now}",
                flush=True,
            )
            for agv in ranked:
                if not loaded.get(agv, False):
                    continue
                task = deliverable[agv]
                pads = _valid_unload_pads(
                    task, free, static, prefer=pose[agv][:2]
                )
                if not pads:
                    continue
                arrived = pose[agv][:2] in pads
                if not arrived:
                    now, ok = _jam_fleet_unload(
                        pose,
                        steps_by,
                        now,
                        agv,
                        pads[0],
                        blocked_plan,
                        free,
                        loaded=loaded,
                        dest=dest,
                        tid=tid,
                    )
                    arrived = bool(ok) or pose[agv][:2] in pads
                if not arrived:
                    for tgt in pads[:3]:
                        now, ok = _greedy_reach(
                            pose,
                            steps_by,
                            now,
                            agv,
                            tgt,
                            blocked_plan,
                            loaded=loaded,
                            dest=dest,
                            tid=tid,
                            max_steps=100,
                        )
                        if ok or pose[agv][:2] in pads:
                            arrived = True
                            break
                if arrived and pose[agv][:2] in pads:
                    # Dwell while loaded, then clear on a stay tick (validator
                    # rejects rewriting move-in as unload — r32 premature_unload).
                    now = _fleet_hold_tick(
                        steps_by,
                        pose,
                        names,
                        now,
                        loaded=loaded,
                        dest=dest,
                        tid=tid,
                        rise_ok={},
)
                    if pose[agv][:2] not in pads:
                        continue
                    loaded[agv] = False
                    dest[agv] = ""
                    tid[agv] = ""
                    now = _fleet_hold_tick(
                        steps_by,
                        pose,
                        names,
                        now,
                        loaded=loaded,
                        dest=dest,
                        tid=tid,
                        rise_ok={},
)
                    _count_done(str(task.get("task_id") or ""))
                    pending_delivery.pop(agv, None)
                    deliverable.pop(agv, None)
                    assigned.pop(agv, None)
                    _STICKY.pop(agv, None)
                    _clear_affinity(agv)
                    n_ok += 1
                    if n_ok >= 2:
                        break
            if n_ok:
                recent_joint_fail = max(0, int(recent_joint_fail) - 4 * n_ok)
                force_unload_miss_streak = 0
                print(
                    f"[ECBS] jam greedy-unload k={n_ok} left="
                    f"{len(pending_delivery)} @t={now}",
                    flush=True,
                )

        _step_off_buf: List[str] = []

        def _clear_after_unload(agv: str) -> None:
            _step_off_buf.append(agv)

        def _batch_step_off(agents: List[str]) -> None:
            """Do not walk off the pad here.

            A finished unloader stays until the next delivery plan. That plan
            is built first (pickup door or a parking cell), then run in the
            same joint timeline as the carriers. Applying the leave now would
            finish it before the next delivery starts.
            """
            del agents
            return

        def _flush_step_off() -> None:
            if _step_off_buf:
                agents = list(_step_off_buf)
                _step_off_buf.clear()
                _batch_step_off(agents)

        def _unload_arrivals(movers: Set[str]) -> None:
            nonlocal now
            cleared: List[str] = []
            for agv in sorted(movers):
                if agv not in deliverable:
                    continue
                ok_pads = pose[agv][:2]
                if not _carrier_on_stand(
                    deliverable[agv], ok_pads, free, static, blocked_plan
                ):
                    continue
                loaded[agv] = False
                dest[agv] = ""
                tid[agv] = ""
                _count_done(str(deliverable[agv]["task_id"]))
                pending_delivery.pop(agv, None)
                _STICKY.pop(agv, None)
                _clear_affinity(agv)
                cleared.append(agv)
            if cleared:
                # Falling edge on a shared stay tick (unload dwell).
                now = _fleet_hold_tick(
                    steps_by, pose, names, now, loaded=loaded, dest=dest, tid=tid,
                    rise_ok={},
)
                _batch_step_off(cleared)

        # All carriers in one wave (同出). Shared pads OK via vacate-after-goal.
        already = [
            agv
            for agv, task in list(deliverable.items())
            if loaded.get(agv, False)
            and _carrier_on_stand(
                task, pose[agv][:2], free, static, blocked_plan
            )
        ]
        if already:
            for agv in already:
                task = deliverable.get(agv)
                if task is None:
                    continue
                loaded[agv] = False
                dest[agv] = ""
                tid[agv] = ""
                _count_done(str(task.get("task_id") or ""))
                pending_delivery.pop(agv, None)
                _STICKY.pop(agv, None)
                deliverable.pop(agv, None)
                _clear_affinity(agv)
                print(
                    f"[PIPE] unload-done {agv} at={pose[agv][:2]} "
                    f"tid={task.get('task_id')}",
                    flush=True,
                )
            now = _fleet_hold_tick(
                steps_by, pose, names, now,
                loaded=loaded, dest=dest, tid=tid, rise_ok={},
            )
        active = set(deliverable.keys())
        # Plan the leave first, then run it in this same joint as delivery.
        # A car the distance claim did not pick still has to leave the door:
        # toward its next pickup if it already has one, otherwise a park cell.
        leave_agents: Set[str] = set()
        leave_doors = _all_unload_doors(set(blocked_plan))
        occupied_leave: Set[Cell] = {pose[n][:2] for n in names}
        occupied_leave.update(g for g in goals.values())
        for a in names:
            if a in active or a in _STICKY or loaded.get(a, False):
                continue
            cell = pose[a][:2]
            if cell not in leave_doors:
                continue
            task_l = pending_pickup.get(a) or assigned.get(a)
            goal_l: Optional[Cell] = None
            kind = "park"
            if task_l is not None:
                cand = _pk(task_l, cell, free, static)
                if cand != cell and cand not in occupied_leave:
                    goal_l = cand
                    kind = "pickup"
            if goal_l is None:
                goal_l = _park_off_doors(cell, leave_doors, occupied_leave, free)
                kind = "park"
            if goal_l is None or goal_l == cell:
                continue
            goals[a] = goal_l
            occupied_leave.add(goal_l)
            active.add(a)
            leave_agents.add(a)
            print(
                f"[PIPE] leave-with-delivery {a} {cell}->{goal_l} {kind}",
                flush=True,
            )
        paths = None
        joint_applied = False
        if not active:
            joint_applied = True  # nothing to deliver and nobody blocking a door
        # True last-mile or sustained fail: disperse then serial.
        # Do NOT treat "many carriers" alone as jam — that forced k=1 and
        # inflated sim_time under hub bursts (general congestion anti-pattern).
        delivery_jam = (
            not assigned
            and len(deliverable) >= 1
            and (
                int(recent_joint_fail) >= 4
                or int(no_progress_waves) >= 2
            )
        )
        last_mile = (
            (len(deliverable) <= 2 and not assigned and (total - done) <= 3)
            or delivery_jam
        )
        carriers_now = {a for a in active if a in deliverable}
        # Door cars leaving for the next pickup are in `active` but not
        # `deliverable`. Last-mile serial is only for carriers; a leave-only
        # wave must still take the joint so they reach the pickup door.
        if last_mile and carriers_now:
            keep_lm: Set[Cell] = set()
            for a in carriers_now:
                keep_lm.add(pose[a][:2])
                keep_lm.add(goals.get(a, pose[a][:2]))
                keep_lm.update(
                    _valid_unload_pads(deliverable[a], free, static)[:4]
                )
            now = _disperse_idle_agents(
                pose,
                steps_by,
                now,
                movers=set(active),
                free=free,
                blocked_plan=blocked_plan,
                keep_clear=keep_lm,
                loaded={n: bool(loaded.get(n)) for n in names},
                dest=dict(dest),
                tid=dict(tid),
                aggressive=True,
            )
            starts = {n: pose[n][:2] for n in names}
            tag = "jam" if delivery_jam and (total - done) > 3 else "last-mile"
            print(
                f"[RECOVER] delivery-{tag} disperse k={len(active)} "
                f"fail={recent_joint_fail} left={total - done} @t={now}",
                flush=True,
            )
        gmap = {n: starts[n] for n in names}
        for a in active:
            gmap[a] = goals[a]
        if active and not (last_mile and carriers_now):
            now = _yield_blockers_joint(
                pose,
                steps_by,
                now,
                movers=set(active),
                goals={a: goals[a] for a in active},
                static=set(blocked_plan),
                free=free,
                loaded=loaded,
                dest=dest,
                tid=tid,
            )
            starts = {n: pose[n][:2] for n in names}
            gmap = {n: starts[n] for n in names}
            for a in active:
                gmap[a] = goals[a]
            paths = _turn_aware_cell_paths(
                pose, gmap, active, set(blocked_plan), prefer_outer_ring=True
            )
            if paths is not None:
                moving_d = {
                    n
                    for n, seq in paths.items()
                    if len(seq) > 1 and any(c != seq[0] for c in seq[1:])
                } or set(active)
                g_chk = dict(gmap)
                for n in moving_d:
                    if n not in g_chk and paths.get(n):
                        g_chk[n] = paths[n][-1]
                if not _st_paths_ok(paths, moving_d, g_chk):
                    paths = None
            if paths is None:
                # Prefer prioritized ST (hold_after=0) over ECBS joint — SH03-style
                # concurrency with shared-pad vacate. ECBS joint fails often on hubs.
                paths = _prioritized_st_paths(
                    starts, gmap, active, blocked_plan, hold_after=1
                )
            if paths is not None and not _st_paths_ok(paths, active, gmap):
                paths = None
            if paths is None and leave_agents:
                # Far pickup goals made the joint fail, so the door cars never
                # moved. Retarget them to the nearest park and plan that leave
                # with the carriers. If the carriers still cannot fit, move
                # the door cars anyway — a sticky carrier is not a reason to
                # keep blocking the pad.
                occupied_p: Set[Cell] = {pose[n][:2] for n in names}
                n_park = 0
                for a in sorted(leave_agents):
                    if a not in active:
                        continue
                    cell = pose[a][:2]
                    park = _park_off_doors(cell, leave_doors, occupied_p, free)
                    if park is None or park == cell:
                        continue
                    goals[a] = park
                    gmap[a] = park
                    occupied_p.add(park)
                    n_park += 1
                if n_park:
                    print(f"[PIPE] leave-park retry n={n_park}", flush=True)
                    paths = _prioritized_st_paths(
                        starts, gmap, active, blocked_plan, hold_after=0
                    )
                    if paths is not None and not _st_paths_ok(paths, active, gmap):
                        paths = None
                if paths is None:
                    g_leave = {n: starts[n] for n in names}
                    movers_l = {a for a in leave_agents if a in active}
                    for a in movers_l:
                        g_leave[a] = goals.get(a, starts[a])
                    sp_l = _prioritized_st_paths(
                        starts, g_leave, movers_l, blocked_plan, hold_after=0
                    )
                    if sp_l is not None and _st_paths_ok(sp_l, movers_l, g_leave):
                        paths = sp_l
                        gmap = g_leave
                        active = set(movers_l)
                        print(
                            f"[PIPE] leave-only joint n={len(active)}",
                            flush=True,
                        )
            if paths is None and active:
                keep_rt: Set[Cell] = set()
                for a in active:
                    keep_rt.add(pose[a][:2])
                    keep_rt.add(goals.get(a, pose[a][:2]))
                now = _disperse_idle_agents(
                    pose,
                    steps_by,
                    now,
                    movers=set(active),
                    free=free,
                    blocked_plan=blocked_plan,
                    keep_clear=keep_rt,
                    loaded={n: bool(loaded.get(n)) for n in names},
                    dest=dict(dest),
                    tid=dict(tid),
                    aggressive=True,
                )
                starts = {n: pose[n][:2] for n in names}
                gmap = {n: starts[n] for n in names}
                for a in active:
                    gmap[a] = goals[a]
                paths = _prioritized_st_paths(
                    starts, gmap, active, blocked_plan, hold_after=1
                )
                if paths is not None and not _st_paths_ok(paths, active, gmap):
                    paths = None
        elif active and last_mile and carriers_now:
            # Skip joint; fall through to per-agent serial recover.
            paths = None
        if paths is not None and _st_paths_ok(paths, active, gmap):
            paths_v = _vacate_extend_after_goals(
                paths, active, gmap, blocked_plan, starts
            )
            timelines = _plan_paths_as_poses(pose, paths_v)
            if not _pose_movers_ok(pose, timelines, active):
                # Shared vacate collided — retry unique pads only.
                if unique_movers and unique_movers != active:
                    active = set(unique_movers) | set(leave_agents)
                    gmap = {n: starts[n] for n in names}
                    for a in active:
                        gmap[a] = goals[a]
                    paths = _prioritized_st_paths(
                        starts, gmap, active, blocked_plan, hold_after=1
                    )
                    if paths is not None and _st_paths_ok(paths, active, gmap):
                        paths_v = _vacate_extend_after_goals(
                            paths, active, gmap, blocked_plan, starts
                        )
                        timelines = _plan_paths_as_poses(pose, paths_v)
                    else:
                        paths = None
                else:
                    paths = None
            else:
                paths = paths_v
        if paths is not None and _st_paths_ok(paths, active, gmap):
            timelines = _plan_paths_as_poses(pose, paths)
            pad_u = {
                a: set(_valid_unload_pads(deliverable[a], free, static))
                for a in active
                if a in deliverable
            }

            def _on_unload_mid(agv: str) -> None:
                if agv not in deliverable or not loaded.get(agv, False):
                    return
                loaded[agv] = False
                dest[agv] = ""
                tid[agv] = ""
                _count_done(str(deliverable[agv]["task_id"]))
                pending_delivery.pop(agv, None)
                _STICKY.pop(agv, None)
                _clear_affinity(agv)

            if _pose_movers_ok(pose, timelines, active):
                # Cut on a delivery arrival, not on a short park/pickup leave.
                # The slice is at least long enough for each door car to step
                # off; it does not wait out the whole leave before delivery.
                cut_movers = {a for a in active if a in deliverable} or set(active)
                off_door: List[int] = []
                for a in leave_agents:
                    seq = timelines.get(a) or []
                    start_c = pose[a][:2]
                    for i, p in enumerate(seq):
                        if (int(p[0]), int(p[1])) != start_c:
                            off_door.append(i + 1)
                            break
                floor_steps = min(8, max(off_door) if off_door else 0)
                now_before = now
                now = _apply_timelines_until_first_goal(
                    steps_by,
                    pose,
                    timelines,
                    gmap,
                    cut_movers,
                    now,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                    pad_unload=pad_u,
                    on_unload=_on_unload_mid,
                    rise_ok={},
                    min_steps=floor_steps,
                )
                if now == now_before:
                    paths = None
                else:
                    _unload_arrivals(active)
                    _flush_step_off()
                    joint_applied = True
                    print(
                        f"[ECBS] delivery wave-ST k={len(active)} "
                        f"unique={len(unique_movers)} shared="
                        f"{max(0, len(active) - len(unique_movers))} @t={now}",
                        flush=True,
                    )

        if not joint_applied:
            print(
                f"[ECBS] delivery joint fail/skip k={len(deliverable)} "
                f"unique={len(unique_movers)} -> shrink",
                flush=True,
            )
            recent_joint_fail += 1

        need = sorted(
            a
            for a in deliverable
            if loaded.get(a, False) and a not in _STICKY
        )
        if not joint_applied and leave_agents:
            # The carrier joint failed and a sticky-wait would advance the
            # clock without moving the car on the unload door. Park that car
            # now so the next wave can pick it up.
            starts_l = {n: pose[n][:2] for n in names}
            goals_l = {n: starts_l[n] for n in names}
            occupied_l: Set[Cell] = set(starts_l.values())
            movers_l: Set[str] = set()
            for a in sorted(leave_agents):
                cell = pose[a][:2]
                park = _park_off_doors(cell, leave_doors, occupied_l, free)
                if park is None or park == cell:
                    continue
                goals_l[a] = park
                occupied_l.add(park)
                movers_l.add(a)
            if movers_l:
                sp_l = _prioritized_st_paths(
                    starts_l, goals_l, movers_l, blocked_plan, hold_after=0
                )
                if sp_l is not None and _st_paths_ok(sp_l, movers_l, goals_l):
                    timelines_l = _plan_paths_as_poses(pose, sp_l)
                    if _pose_movers_ok(pose, timelines_l, movers_l):
                        now_before = now
                        now = _apply_timelines_until_first_goal(
                            steps_by,
                            pose,
                            timelines_l,
                            goals_l,
                            movers_l,
                            now,
                            loaded=loaded,
                            dest=dest,
                            tid=tid,
                            rise_ok={},
                        )
                        if now != now_before:
                            joint_applied = True
                            print(
                                f"[PIPE] leave-unstick n={len(movers_l)} @t={now}",
                                flush=True,
                            )
            if not joint_applied:
                # Joint and park-ST both rejected the wave. Step each door
                # car one cell toward its leave goal so the same six cars
                # cannot sit on the pads until the wall clock expires.
                hopped = 0
                for a in sorted(leave_agents):
                    cell = pose[a][:2]
                    if cell not in leave_doors:
                        continue
                    goal_h = goals.get(a, cell)
                    if goal_h == cell:
                        goal_h = _park_off_doors(
                            cell,
                            leave_doors,
                            {pose[n][:2] for n in names},
                            free,
                        ) or cell
                    if goal_h == cell:
                        continue
                    blocked_h = set(blocked_plan)
                    for n in names:
                        if n != a:
                            blocked_h.add(pose[n][:2])
                    blocked_h.discard(cell)
                    blocked_h.discard(goal_h)
                    hop_path = _astar_cells(cell, goal_h, blocked_h)
                    if not hop_path or len(hop_path) < 2:
                        continue
                    now, ok_h = _one_hop_move(
                        pose,
                        steps_by,
                        now,
                        a,
                        hop_path[1],
                        blocked_plan,
                        loaded=loaded,
                        dest=dest,
                        tid=tid,
                        allow_cells={hop_path[1]},
                    )
                    if ok_h and pose[a][:2] != cell:
                        hopped += 1
                if hopped:
                    joint_applied = True
                    print(f"[PIPE] leave-hop n={hopped} @t={now}", flush=True)
        if not joint_applied and not need and _STICKY:
            _force_wait_tick("sticky-wait")
            joint_applied = True
        if need and not joint_applied:
            # Parallel ST before any per-agent serial (all planners).
            starts_b = {n: pose[n][:2] for n in names}
            goals_b = {n: starts_b[n] for n in names}
            for agv in need:
                pads = _valid_unload_pads(
                    deliverable[agv], free, static, prefer=starts_b[agv]
                )
                if not pads:
                    print(
                        f"[ECBS] no unload pads for {deliverable[agv].get('task_id')} "
                        f"— keep cargo @t={now}",
                        flush=True,
                    )
                    continue
                target = goals[agv] if goals.get(agv) in pads else pads[0]
                if target not in pads:
                    target = min(pads, key=lambda c: (_manh(starts_b[agv], c), c))
                goals_b[agv] = target
            movers_b = {a for a in need if loaded.get(a, False)}
            if movers_b:
                sp = _prioritized_st_paths(
                    starts_b, goals_b, movers_b, blocked_plan, hold_after=1
                )
                if _st_paths_ok(sp, movers_b, goals_b):
                    timelines = _plan_paths_as_poses(pose, sp)
                    if _pose_movers_ok(pose, timelines, movers_b):
                        now = _apply_timelines_until_first_goal(
                            steps_by,
                            pose,
                            timelines,
                            goals_b,
                            movers_b,
                            now,
                            loaded=loaded,
                            dest=dest,
                            tid=tid,
                            rise_ok={},
                        )
                        for agv in list(movers_b):
                            pads = set(
                                _valid_unload_pads(deliverable[agv], free, static)
                            )
                            if pose[agv][:2] in pads and loaded.get(agv, False):
                                loaded[agv] = False
                                dest[agv] = ""
                                tid[agv] = ""
                                _count_done(str(deliverable[agv]["task_id"]))
                                pending_delivery.pop(agv, None)
                                _STICKY.pop(agv, None)
                                _clear_after_unload(agv)
                        # Falling edge must be a stay tick — never rewrite move-in.
                        now = _fleet_hold_tick(
                            steps_by,
                            pose,
                            names,
                            now,
                            loaded=loaded,
                            dest=dest,
                            tid=tid,
                            rise_ok={},
)
                        _flush_step_off()
                        joint_applied = True
                        print(
                            f"[ECBS] delivery ST parallel k={len(movers_b)}",
                            flush=True,
                            )
            need = sorted(
                a
                for a in deliverable
                if loaded.get(a, False) and a not in _STICKY
            )

        if need and not joint_applied:
            # Joint success already returned the first arrival to the wave
            # loop. This branch is only the joint-fail fallback.
            if len(need) > 1:
                starts_b = {n: pose[n][:2] for n in names}
                goals_b = {n: starts_b[n] for n in names}
                ok_need = []
                for agv in need:
                    pads = _valid_unload_pads(
                        deliverable[agv], free, static, prefer=starts_b[agv]
                    )
                    if not pads:
                        _mark_failed(str(deliverable[agv]["task_id"]))
                        loaded[agv] = False
                        continue
                    target = goals[agv] if goals.get(agv) in pads else pads[0]
                    if target not in pads:
                        target = min(pads, key=lambda c: (_manh(starts_b[agv], c), c))
                    goals_b[agv] = target
                    ok_need.append(agv)
                if ok_need:
                    sp = _prioritized_st_paths(
                        starts_b, goals_b, set(ok_need), blocked_plan
                    )
                    if _st_paths_ok(sp, set(ok_need), goals_b):
                        timelines = _plan_paths_as_poses(pose, sp)
                        if _pose_movers_ok(pose, timelines, set(ok_need)):
                            now = _apply_pose_timelines(
                            steps_by,
                            pose,
                                timelines,
                                now,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                                rise_ok={},
)
                            for agv in list(ok_need):
                                pads = set(
                                    _valid_unload_pads(
                                    deliverable[agv], free, static
                                    )
                                )
                                if pose[agv][:2] in pads and loaded.get(agv, False):
                                    loaded[agv] = False
                                    dest[agv] = ""
                                    tid[agv] = ""
                                    _count_done(str(deliverable[agv]["task_id"]))
                                    pending_delivery.pop(agv, None)
                                    _STICKY.pop(agv, None)
                                    _clear_after_unload(agv)
                            now = _fleet_hold_tick(
                                steps_by, pose, names, now,
                                loaded=loaded, dest=dest, tid=tid,
                                rise_ok={},
)
                            _flush_step_off()
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f"[ECBS] delivery ST leftover-parallel "
                                f"k={len(ok_need)} left={len(need)}",
                                flush=True,
                            )

            if len(need) > 1:
                # P1: try disperse + full Prioritized once before shrinking.
                if int(recent_joint_fail) >= 1 or len(need) <= 2:
                    keep_cells: Set[Cell] = set()
                    for a in need:
                        keep_cells.add(pose[a][:2])
                        keep_cells.add(goals.get(a, pose[a][:2]))
                        keep_cells.update(
                            _valid_unload_pads(deliverable[a], free, static)[:4]
                        )
                    now = _disperse_idle_agents(
                    pose,
                    steps_by,
                        now,
                        movers=set(need),
                        free=free,
                        blocked_plan=blocked_plan,
                        keep_clear=keep_cells,
                        loaded={n: bool(loaded.get(n)) for n in names},
                        dest=dict(dest),
                        tid=dict(tid),
                    )
                    starts_r = {n: pose[n][:2] for n in names}
                    goals_r = {n: starts_r[n] for n in names}
                    for a in need:
                        pads = _valid_unload_pads(
                            deliverable[a], free, static, prefer=starts_r[a]
                        )
                        if pads:
                            goals_r[a] = min(
                                pads, key=lambda c: (_manh(starts_r[a], c), c)
                            )
                    sp_r = _prioritized_st_paths(
                        starts_r, goals_r, set(need), blocked_plan
                    )
                    if _st_paths_ok(sp_r, set(need), goals_r):
                        timelines = _plan_paths_as_poses(pose, sp_r)
                        if _pose_movers_ok(pose, timelines, set(need)):
                            now = _apply_pose_timelines(
                            steps_by,
                            pose,
                                timelines,
                                now,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                                rise_ok={},
)
                            for a in list(need):
                                pads = set(
                                    _valid_unload_pads(deliverable[a], free, static)
                                )
                                if pose[a][:2] in pads and loaded.get(a, False):
                                    loaded[a] = False
                                    dest[a] = ""
                                    tid[a] = ""
                                    _count_done(str(deliverable[a]["task_id"]))
                                    pending_delivery.pop(a, None)
                                    _STICKY.pop(a, None)
                                    _clear_affinity(a)
                                    _clear_after_unload(a)
                            now = _fleet_hold_tick(
                                steps_by, pose, names, now,
                                loaded=loaded, dest=dest, tid=tid,
                                rise_ok={},
)
                            _flush_step_off()
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f"[RECOVER] delivery disperse+prio k_left={len(need)} "
                                f"@t={now}",
                                flush=True,
                                )
                if len(need) > 1:
                    # Prefer one spacetime wave for ALL leftovers (parallel clock).
                    # Sequential _serial_move_to sums path lengths into makespan.
                    ranked = sorted(
                        need,
                        key=lambda a: (
                            _manh(
                                pose[a][:2],
                                goals.get(a, pose[a][:2]),
                            ),
                            a,
                        ),
                    )
                    jam_hard = (
                        int(recent_joint_fail) >= 8
                        and int(no_progress_waves) >= 4
                    )
                    # Prefer one spacetime wave for ALL leftovers (同出).
                    # Even under hub jam keep ≥4 movers when available —
                    # peeling to k=1 is the main sim_t blow-up (r110 endgame).
                    if jam_hard and len(ranked) > 4:
                        batch = ranked[: max(4, len(ranked) // 2)]
                    elif jam_hard:
                        batch = ranked[: max(1, min(4, len(ranked)))]
                    else:
                        batch = list(ranked)
                    for agv in ranked[len(batch) :]:
                        pending_delivery[agv] = deliverable[agv]
                        loaded[agv] = True
                        dest[agv] = str(deliverable[agv].get("destination") or "")
                        tid[agv] = str(deliverable[agv].get("task_id") or "")
                    starts_p = {n: pose[n][:2] for n in names}
                    goals_p = {n: starts_p[n] for n in names}
                    for a in batch:
                        pads = _valid_unload_pads(
                            deliverable[a], free, static, prefer=starts_p[a]
                        )
                        if not pads:
                            continue
                        tgt = goals[a] if goals.get(a) in pads else pads[0]
                        if tgt not in pads:
                            tgt = min(pads, key=lambda c: (_manh(starts_p[a], c), c))
                        goals_p[a] = tgt
                    movers_p = {a for a in batch if loaded.get(a, False)}
                    if movers_p:
                        _ks = set()
                        for _a in movers_p:
                            _ks.add(pose[_a][:2])
                            _ks.update(
                                _valid_unload_pads(deliverable[_a], free, static)[:6]
                            )
                        now = _disperse_idle_agents(
                            pose, steps_by, now, movers=set(movers_p), free=free,
                            blocked_plan=blocked_plan, keep_clear=_ks,
                            loaded={n: bool(loaded.get(n)) for n in names},
                            dest=dict(dest), tid=dict(tid),
                        )
                    sp_p = _prioritized_st_paths(
                        starts_p, goals_p, movers_p, blocked_plan
                    )
                    if _st_paths_ok(sp_p, movers_p, goals_p):
                        timelines = _plan_paths_as_poses(pose, sp_p)
                        if _pose_movers_ok(pose, timelines, movers_p):
                            now = _apply_pose_timelines(
                                steps_by,
                                pose,
                                timelines,
                                now,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                                rise_ok={},
)
                            for a in list(movers_p):
                                pads = set(
                                    _valid_unload_pads(deliverable[a], free, static)
                                )
                                if pose[a][:2] in pads and loaded.get(a, False):
                                    loaded[a] = False
                                    dest[a] = ""
                                    tid[a] = ""
                                    _count_done(str(deliverable[a]["task_id"]))
                                    pending_delivery.pop(a, None)
                                    _STICKY.pop(a, None)
                                    _clear_affinity(a)
                                    _clear_after_unload(a)
                            now = _fleet_hold_tick(
                                steps_by, pose, names, now,
                                loaded=loaded, dest=dest, tid=tid,
                                rise_ok={},
)
                            _flush_step_off()
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f"[ECBS] delivery batch-ST k={len(movers_p)} "
                                f"left={len(need)} jam={int(jam_hard)} @t={now}",
                                flush=True,
                            )
                        else:
                            # Pose conflict: peel but keep ≥4 movers when possible
                            # (r67 VALID; floor avoids k=1–2 sim blow-up).
                            half = (
                                max(4, len(batch) // 2)
                                if len(batch) >= 4
                                else max(1, len(batch) // 2)
                            )
                            need = list(batch[:half])
                            for _a in batch[half:]:
                                pending_delivery[_a] = deliverable[_a]
                                loaded[_a] = True
                                dest[_a] = str(
                                    deliverable[_a].get("destination") or ""
                                )
                                tid[_a] = str(
                                    deliverable[_a].get("task_id") or ""
                                )
                            if len(need) > 1:
                                starts_r2 = {n: pose[n][:2] for n in names}
                                goals_r2 = {n: starts_r2[n] for n in names}
                                for a in need:
                                    pads = _valid_unload_pads(
                                        deliverable[a],
                                        free,
                                        static,
                                        prefer=starts_r2[a],
                                    )
                                    if pads:
                                        goals_r2[a] = min(
                                            pads,
                                            key=lambda c: (
                                                _manh(starts_r2[a], c),
                                                c,
                                            ),
                                        )
                                sp_r2 = _prioritized_st_paths(
                                    starts_r2,
                                    goals_r2,
                                    set(need),
                                    blocked_plan,
                                )
                                if _st_paths_ok(sp_r2, set(need), goals_r2):
                                    timelines = _plan_paths_as_poses(pose, sp_r2)
                                    if _pose_movers_ok(
                                        pose, timelines, set(need)
                                    ):
                                        now = _apply_pose_timelines(
                                            steps_by,
                                            pose,
                                            timelines,
                                            now,
                                            loaded=loaded,
                                            dest=dest,
                                            tid=tid,
                                            rise_ok={},
)
                                        _unload_arrivals(set(need))
                                        _flush_step_off()
                                        need = sorted(
                                            a
                                            for a in need
                                            if loaded.get(a, False)
                                        )
                                        print(
                                            f"[ECBS] delivery floor-ST retry "
                                            f"k_left={len(need)} @t={now}",
                                            flush=True,
                                        )
                            print(
                                f"[ECBS] delivery batch-ST pose-conflict "
                                f"k={len(batch)} -> floor={half} @t={now}",
                                flush=True,
                            )
                    else:
                        need = batch
                        print(
                            f"[ECBS] delivery batch-ST fail k={len(batch)} "
                            f"jam={int(jam_hard)} -> shrink @t={now}",
                            flush=True,
                        )
                    if jam_hard:
                        recent_joint_fail += 1
            if need:
                # Before serial: clear blockers around the single (or few) targets.
                if last_mile or int(recent_joint_fail) >= 3:
                    keep_s: Set[Cell] = set()
                    for a in need:
                        keep_s.add(pose[a][:2])
                        keep_s.update(
                            _valid_unload_pads(deliverable[a], free, static)[:6]
                        )
                    now = _disperse_idle_agents(
                        pose,
                        steps_by,
                        now,
                        movers=set(need),
                        free=free,
                        blocked_plan=blocked_plan,
                        keep_clear=keep_s,
                        loaded={n: bool(loaded.get(n)) for n in names},
                        dest=dict(dest),
                        tid=dict(tid),
                    )
                for agv in list(need):
                    if not loaded.get(agv, False):
                        continue
                    pads = _valid_unload_pads(
                        deliverable[agv], free, static, prefer=pose[agv][:2]
                    )
                    if not pads:
                        print(
                            f"[ECBS] no unload pads for "
                            f"{deliverable[agv].get('task_id')} — keep cargo "
                            f"@t={now}",
                            flush=True,
                        )
                        continue
                # Fail by shrinking the wave, not by walking each AGV alone.
                # Drop the farthest carrier and replan the rest together.
                ranked = sorted(
                    [a for a in need if loaded.get(a, False)],
                    key=lambda a: (
                        _manh(pose[a][:2], goals.get(a, pose[a][:2])),
                        a,
                    ),
                )
                batch = list(ranked)
                yielded_for: Set[Tuple[str, ...]] = set()
                pad_rot: Dict[str, int] = {}
                while batch and not joint_applied:
                    sig = tuple(sorted(batch))
                    starts_p = {n: pose[n][:2] for n in names}
                    goals_p = {n: starts_p[n] for n in names}
                    movers_p: Set[str] = set()
                    occ_other = {
                        pose[n][:2] for n in names if n not in batch
                    }
                    for a in batch:
                        pads = _valid_unload_pads(
                            deliverable[a], free, static, prefer=starts_p[a]
                        )
                        if not pads:
                            continue
                        reachable = [
                            c
                            for c in pads
                            if _bfs_len(starts_p[a], c, blocked_plan) < 10**6
                        ]
                        use = reachable or list(pads)

                        def _pad_key(c: Cell, _a: str = a) -> Tuple[int, int, Cell]:
                            path = _astar_cells(
                                starts_p[_a], c, set(blocked_plan)
                            )
                            hits = (
                                10**6
                                if not path
                                else sum(
                                    1 for cell in path[1:] if cell in occ_other
                                )
                            )
                            return (hits, len(path or []), c)

                        use.sort(key=_pad_key)
                        ix = int(pad_rot.get(a, 0)) % len(use)
                        if ix == 0 and goals.get(a) in use:
                            # Keep the wave's unique pad unless it is crowded.
                            pref = goals[a]
                            if _pad_key(pref)[0] <= _pad_key(use[0])[0]:
                                tgt = pref
                            else:
                                tgt = use[0]
                        else:
                            tgt = use[ix]
                        goals_p[a] = tgt
                        goals[a] = tgt
                        movers_p.add(a)
                    if movers_p and sig not in yielded_for:
                        yielded_for.add(sig)
                        now = _yield_blockers_joint(
                            pose,
                            steps_by,
                            now,
                            movers=set(movers_p),
                            goals=goals_p,
                            static=set(blocked_plan),
                            free=free,
                            loaded=loaded,
                            dest=dest,
                            tid=tid,
                        )
                        continue
                    sp = None
                    if movers_p:
                        sp = _prioritized_st_paths(
                            starts_p,
                            goals_p,
                            movers_p,
                            blocked_plan,
                            hold_after=1,
                        )
                        if sp is not None and not _st_paths_ok(
                            sp, movers_p, goals_p
                        ):
                            sp = _turn_aware_cell_paths(
                                pose,
                                goals_p,
                                movers_p,
                                set(blocked_plan),
                                prefer_outer_ring=False,
                            )
                    if (
                        sp is not None
                        and movers_p
                        and _st_paths_ok(sp, movers_p, goals_p)
                    ):
                        timelines = _plan_paths_as_poses(pose, sp)
                        if _pose_movers_ok(pose, timelines, movers_p):
                            now = _apply_timelines_until_first_goal(
                                steps_by,
                                pose,
                                timelines,
                                goals_p,
                                movers_p,
                                now,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                                rise_ok={},
                            )
                            _unload_arrivals(movers_p)
                            _flush_step_off()
                            joint_applied = True
                            print(
                                f"[ECBS] delivery shrink-ST k={len(movers_p)} "
                                f"@t={now}",
                                flush=True,
                            )
                            break
                    if len(batch) <= 1:
                        a0 = batch[0] if batch else ""
                        pads0 = (
                            _valid_unload_pads(
                                deliverable[a0],
                                free,
                                static,
                                prefer=pose[a0][:2],
                            )
                            if a0 in deliverable
                            else []
                        )
                        if a0 and int(pad_rot.get(a0, 0)) + 1 < len(pads0):
                            pad_rot[a0] = int(pad_rot.get(a0, 0)) + 1
                            yielded_for.discard(sig)
                            print(
                                f"[ECBS] delivery shrink pad {a0} "
                                f"#{pad_rot[a0]} @t={now}",
                                flush=True,
                            )
                            continue
                        print(
                            f"[ECBS] delivery shrink stop k=1 "
                            f"defer={batch[0] if batch else '-'} @t={now}",
                            flush=True,
                        )
                        if a0 and a0 in pose:
                            goal0 = goals_p.get(a0) or pose[a0][:2]
                            if pose[a0][:2] != goal0:
                                now, ok_s = _serial_move_to(
                                    pose,
                                    steps_by,
                                    now,
                                    a0,
                                    goal0,
                                    blocked_plan,
                                    loaded=loaded,
                                    dest=dest,
                                    tid=tid,
                                )
                                if ok_s:
                                    _unload_arrivals({a0})
                                    _flush_step_off()
                                    joint_applied = True
                                    print(
                                        f"[ECBS] delivery shrink-serial {a0} "
                                        f"@t={now}",
                                        flush=True,
                                    )
                                    break
                        if not joint_applied:
                            _force_wait_tick("shrink_stop")
                        break
                    drop = batch.pop()
                    pending_delivery[drop] = deliverable[drop]
                    loaded[drop] = True
                    dest[drop] = str(deliverable[drop].get("destination") or "")
                    tid[drop] = str(deliverable[drop].get("task_id") or "")
                    print(
                        f"[ECBS] delivery shrink {len(batch) + 1}->{len(batch)} "
                        f"drop={drop} @t={now}",
                        flush=True,
                    )

        eff_core = _effective_joint_core()
        if eff_core != joint_core_mode and recent_joint_fail >= 6:
            print(
                f"[HIER] joint_core escalate {joint_core_mode}->{eff_core} "
                f"fail={recent_joint_fail}",
                flush=True,
            )
        # Persist still-loaded deliverable carriers (r72).
        for agv, task in list(deliverable.items()):
            if loaded.get(agv, False):
                pending_delivery[agv] = task
            elif agv in pending_delivery and not loaded.get(agv, False):
                pending_delivery.pop(agv, None)
        print(
            f"[PROG] t={now} done={done}/{total} "
            f"wall={time.perf_counter() - t0:.1f}s "
            f"planner={wave_planner} joint_core={eff_core} "
            f"wave={len(assigned)}"
            f"{'' if hybrid_mode == 'off' else f'+hybrid({hybrid_mode})'} "
            f"core={len(movers_ecbs)} astar={len(movers_astar)} "
            f"carry={len(pending_delivery)} fail={recent_joint_fail}",
            flush=True,
        )

        # P0 anti-spin recover
        # Stall = same sim-t AND same done. Recover `_force_wait_tick` advances
        # `now` so this counter alone never hits drop_thr — hub livelock is
        # broken by drop-on-force-unload-miss below (r25).
        if done == stall_done:
            done_stall_waves += 1
        else:
            done_stall_waves = 0
        if now == stall_t and done == stall_done:
            no_progress_waves += 1
        else:
            no_progress_waves = 0
            serial_recover_used = 0
            stall_t, stall_done = now, done
        # sticky-wait advances `now` by 1, so the same-t counter never fires
        # and loaded cars sit forever. Several waves with no new completion
        # is the same stall.
        if done_stall_waves >= 6 and (
            pending_delivery or pending_pickup or any(queues.values())
        ):
            no_progress_waves = max(int(no_progress_waves), 2)

        if no_progress_waves >= 2 and (
            pending_delivery or pending_pickup or any(queues.values())
        ):
            print(
                f"[RECOVER] no-progress waves={no_progress_waves} "
                f"t={now} done={done} carry={len(pending_delivery)} "
                f"serial_budget={serial_recover_used}/{serial_recover_budget}",
                flush=True,
            )
            if serial_recover_used >= serial_recover_budget:
                # Budget exhausted: deepen to hard instead of infinite serial.
                if gate is not None:
                    try:
                        gate.force_commit_at_least(
                            "hard",
                            reason=f"serial_budget_exhausted:{serial_recover_used}",
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"[RECOVER] WARN force hard failed: {exc}", flush=True)
                joint_core_override = "ecbs"
                serial_recover_used = 0
                no_progress_waves = 0
                print(
                    "[RECOVER] serial budget exhausted -> force hard/ECBS core",
                    flush=True,
                )
                continue
            serial_recover_used += 1
            if joint_core_mode == "prioritized":
                joint_core_override = "ecbs"
            _force_wait_tick("no_progress")
            # Force serial unload: try all carriers (SH03: hub jam needs >1).
            carriers = sorted(pending_delivery.keys())
            unloaded_any = False
            if carriers:
                keep_r: Set[Cell] = set()
                for agv in carriers:
                    keep_r.add(pose[agv][:2])
                    keep_r.update(
                        _valid_unload_pads(
                            pending_delivery[agv], free, static
                        )[:6]
                    )
                now = _disperse_idle_agents(
                    pose,
                    steps_by,
                    now,
                    movers=set(carriers),
                    free=free,
                    blocked_plan=blocked_plan,
                    keep_clear=keep_r,
                    loaded={n: (n in pending_delivery) for n in names},
                    dest={
                        n: str(pending_delivery[n].get("destination") or "")
                        if n in pending_delivery
                        else ""
                        for n in names
                    },
                    tid={
                        n: str(pending_delivery[n].get("task_id") or "")
                        if n in pending_delivery
                        else ""
                        for n in names
                    },
                    aggressive=True,
                )
            for agv in list(carriers):
                if agv not in pending_delivery:
                    continue
                task = pending_delivery[agv]
                pads = _valid_unload_pads(task, free, static, prefer=pose[agv][:2])
                if not pads:
                    _mark_failed(str(task.get("task_id") or ""))
                    pending_delivery.pop(agv, None)
                    _clear_affinity(agv)
                    continue
                # Aggressive: park EVERYONE else (incl. other carriers) to corners.
                now = _disperse_idle_agents(
                    pose,
                    steps_by,
                    now,
                    movers={agv},
                    free=free,
                    blocked_plan=blocked_plan,
                    keep_clear=set(pads[:6]) | {pose[agv][:2]},
                    loaded={n: (n in pending_delivery) for n in names},
                    dest={
                        n: str(pending_delivery[n].get("destination") or "")
                        if n in pending_delivery
                        else ""
                        for n in names
                    },
                    tid={
                        n: str(pending_delivery[n].get("task_id") or "")
                        if n in pending_delivery
                        else ""
                        for n in names
                    },
                    aggressive=True,
                )
                ld = {n: (n in pending_delivery) for n in names}
                ds = {
                    n: str(pending_delivery[n].get("destination") or "")
                    if n in pending_delivery
                    else ""
                    for n in names
                }
                td = {
                    n: str(pending_delivery[n].get("task_id") or "")
                    if n in pending_delivery
                    else ""
                    for n in names
                }
                arrived = False
                if pose[agv][:2] in pads:
                    arrived = True
                if not arrived:
                    for tgt in sorted(
                        pads, key=lambda c: (_manh(pose[agv][:2], c), c)
                    )[:4]:
                        now, ok = _greedy_reach(
                            pose,
                            steps_by,
                            now,
                            agv,
                            tgt,
                            blocked_plan,
                            loaded=ld,
                            dest=ds,
                            tid=td,
                            max_steps=80,
                        )
                        if ok and pose[agv][:2] in pads:
                            arrived = True
                            break
                for tgt in sorted(pads, key=lambda c: (_manh(pose[agv][:2], c), c)):
                    if arrived:
                        break
                    now, ok = _serial_move_to(
                        pose,
                        steps_by,
                        now,
                        agv,
                        tgt,
                        blocked_plan,
                        loaded=ld,
                        dest=ds,
                        tid=td,
                    )
                    if ok and pose[agv][:2] in pads:
                        arrived = True
                        break
                if not arrived:
                    # Soft path ignoring dynamic agents, then shove blockers off it.
                    soft = _astar_cells(pose[agv][:2], pads[0], set(blocked_plan))
                    if soft:
                        blockers = {
                            n
                            for n in names
                            if n != agv and pose[n][:2] in set(soft)
                        }
                        for other in sorted(blockers):
                            now = _disperse_idle_agents(
                                pose,
                                steps_by,
                                now,
                                movers={agv, other},
                                free=free,
                                blocked_plan=blocked_plan,
                                keep_clear=set(soft) | set(pads[:4]),
                                loaded=ld,
                                dest=ds,
                                tid=td,
                                aggressive=True,
                            )
                            # Park this blocker to nearest corner-ish free cell
                            corners = [
                                (1, 1),
                                (1, 20),
                                (20, 1),
                                (20, 20),
                                (1, 10),
                                (20, 10),
                            ]
                            occ = {pose[n][:2] for n in names if n != other}
                            cands = [
                                c
                                for c in free
                                if c not in occ
                                and c not in pads
                                and c not in set(soft)
                            ]
                            if not cands:
                                cands = [
                                    (x, y)
                                    for x in range(1, 21)
                                    for y in range(1, 21)
                                    if (x, y) not in blocked_plan
                                    and (x, y) not in occ
                                    and (x, y) not in pads
                                    and (x, y) not in set(soft)
                                ]
                            if not cands:
                                continue
                            park = min(
                                cands,
                                key=lambda c: (
                                    min(_manh(c, k) for k in corners),
                                    c,
                                ),
                            )
                            now, _ = _serial_move_to(
                                pose,
                                steps_by,
                                now,
                                other,
                                park,
                                blocked_plan,
                                loaded=ld,
                                dest=ds,
                                tid=td,
                            )
                        for tgt in sorted(
                            pads, key=lambda c: (_manh(pose[agv][:2], c), c)
                        ):
                            now, ok = _serial_move_to(
                                pose,
                                steps_by,
                                now,
                                agv,
                                tgt,
                                blocked_plan,
                                loaded=ld,
                                dest=ds,
                                tid=td,
                            )
                            if ok and pose[agv][:2] in pads:
                                arrived = True
                                break
                if arrived:
                    # Dwell loaded, then unload on a stay tick (never rewrite move).
                    ld = {n: bool(loaded.get(n)) for n in names}
                    ds = {n: str(dest.get(n, "") or "") for n in names}
                    td = {n: str(tid.get(n, "") or "") for n in names}
                    now = _fleet_hold_tick(
                        steps_by, pose, names, now, loaded=ld, dest=ds, tid=td,
                        rise_ok={},
)
                    loaded[agv] = False
                    dest[agv] = ""
                    tid[agv] = ""
                    ld[agv] = False
                    ds[agv] = ""
                    td[agv] = ""
                    now = _fleet_hold_tick(
                        steps_by, pose, names, now, loaded=ld, dest=ds, tid=td,
                        rise_ok={},
)
                    _count_done(str(task.get("task_id") or ""))
                    pending_delivery.pop(agv, None)
                    _STICKY.pop(agv, None)
                    _clear_affinity(agv)
                    unloaded_any = True
                    print(
                        f"[RECOVER] forced unload {agv} task={task.get('task_id')} "
                        f"@t={now}",
                        flush=True,
                    )
                    # One success per recover tick is enough to break the spin;
                    # remaining carriers retry next wave with lower contention.
                    break
            if unloaded_any:
                recent_joint_fail = max(0, int(recent_joint_fail) - 4)
                if recent_joint_fail < 3:
                    joint_core_override = None
                no_progress_waves = 0
                force_unload_miss_streak = 0
            else:
                # Keep pressure; do not wipe streak on failed recover.
                no_progress_waves = max(2, int(no_progress_waves))
                force_unload_miss_streak += 1
                print(
                    f"[RECOVER] force-unload failed carry={len(pending_delivery)} "
                    f"miss={force_unload_miss_streak} @t={now}",
                    flush=True,
                )
                # Hub livelock: drop only after repeated misses (r26 dropped
                # every wave → done stuck while fail tasks piled up).
                if (
                    force_unload_miss_streak >= 8
                    and int(recent_joint_fail) >= 8
                    and len(pending_delivery) >= 1
                ):
                    # Do NOT wipe cargo off-pad (premature_unload). Keep trying.
                    print(
                        f"[RECOVER] force-unload miss streak high "
                        f"carry={len(pending_delivery)} — keep cargo, retry "
                        f"@t={now}",
                        flush=True,
                    )
                    force_unload_miss_streak = 4
                    _force_wait_tick("keep_carrier")

        # Drop sooner when force-unload keeps failing (hub livelock).
        drop_thr = 4 if int(recent_joint_fail) >= 8 else (20 if (total - done) <= 2 else 10)
        if no_progress_waves >= drop_thr and pending_delivery and (total - done) > 2:
            # Last resort used to drop+wipe cargo (premature_unload). Keep cargo
            # and force another recover wait instead.
            print(
                f"[RECOVER] no-progress waves={no_progress_waves} "
                f"carry={len(pending_delivery)} — keep cargo @t={now}",
                flush=True,
            )
            no_progress_waves = max(2, drop_thr - 4)
            _force_wait_tick("keep_carrier")
        elif no_progress_waves >= drop_thr and pending_delivery and (total - done) <= 2:
            # Final-task hard push: another wait + disperse cycle, no drop yet.
            print(
                f"[RECOVER] final-task persist waves={no_progress_waves} "
                f"left={total - done} @t={now}",
                flush=True,
            )
            _force_wait_tick("final_persist")
            no_progress_waves = max(2, drop_thr - 4)

        if not pipeline:
            # non-pipeline: still continue waves until queues empty
            pass

    _dense_fill(steps_by, now)

    # Export legalize: split move_and_turn pitch violations (same as M0 path).
    try:
        by_t: Dict[int, List[dict]] = {}
        for n in steps_by:
            for r in steps_by[n]:
                by_t.setdefault(int(r["timestamp"]), []).append(dict(r))
        by_t = _legalize_engine_steps(by_t)
        steps_by = {n: [] for n in names}
        for t in sorted(by_t):
            for r in by_t[t]:
                steps_by.setdefault(str(r["name"]), []).append(r)
        for n in steps_by:
            steps_by[n].sort(key=lambda r: int(r["timestamp"]))
    except Exception as exc:  # noqa: BLE001
        print(f"[HIER] WARN traj legalize skipped: {exc}", flush=True)

    sid = str(meta["id"])
    swap_tag = "hier" if use_hierarchical else "base"
    traj = TRAJ / f"{sid}_ecbs_{plan}_w{weight}_k{max_active}_{swap_tag}.csv"
    rows = []
    for n in sorted(steps_by):
        rows.extend(steps_by[n])
    rows.sort(key=lambda r: (int(r["timestamp"]), r["name"]))
    # Conservative motion repair (hold unsafe steps) before write/validate.
    scrubbed_pickup_tids: Set[str] = set()
    try:
        from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
            repair_trajectory_rows,
        )

        rows, rep_stats = repair_trajectory_rows(
            # Walls only — do NOT pass station_pads into repair (r105: when the
            # planner still threaded foreign pads, repair held ≈600k steps and
            # collision-exploded). Station no-transit is enforced in A*/ST.
            rows,
            hard_walls=set(static),
            lo=1,
            hi=20,
        )
        if int(rep_stats.get("held_moves") or 0) > 0 or int(
            rep_stats.get("stripped_rises") or 0
        ) > 0:
            print(
                f"[HIER] traj repair held_moves={rep_stats.get('held_moves')} "
                f"applied={rep_stats.get('applied_moves')} "
                f"stripped_rises={rep_stats.get('stripped_rises')}",
                flush=True,
            )
        # Export-only scrub: episodes that rise off-pad and never visit a legal
        # pickup cell (r107 still had 4 hub phantoms bypassing write guards).
        try:
            from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
                _load_task_pickups,
                scrub_illegal_pickup_rising,
            )

            n_scrub, scrubbed_pickup_tids = scrub_illegal_pickup_rising(
                rows, _load_task_pickups(meta)
            )
            if n_scrub:
                print(
                    f"[HIER] scrubbed illegal pickup rising n={n_scrub} "
                    f"tids={sorted(scrubbed_pickup_tids)[:8]}",
                    flush=True,
                )
        except Exception as scrub_exc:  # noqa: BLE001
            print(f"[HIER] WARN pickup scrub skipped: {scrub_exc}", flush=True)
            scrubbed_pickup_tids = set()
        # Premature-unload safety net: scrub only if still present after
        # off-pad-clear guards (should be rare). Prefer legalize-free re-runs.
        try:
            from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
                _check_task_carry,
                _load_task_dropoffs,
            )

            for e in _check_task_carry(rows, _load_task_dropoffs(meta)):
                if str(e.get("kind") or "") == "premature_unload":
                    tid_p = str(e.get("task_id") or "").strip()
                    if tid_p:
                        scrubbed_pickup_tids.add(tid_p)
        except Exception as prem_exc:  # noqa: BLE001
            print(f"[HIER] WARN premature skip scan failed: {prem_exc}", flush=True)
        try:
            from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
                repair_action_dwell_rows,
            )

            n_dwell = repair_action_dwell_rows(rows)
            if n_dwell:
                print(f"[HIER] repaired action-dwell edges n={n_dwell}", flush=True)
        except Exception as dwell_exc:  # noqa: BLE001
            print(f"[HIER] WARN dwell repair skipped: {dwell_exc}", flush=True)
        # Repair can reintroduce move_and_turn; legalize pitch again.
        by_t2: Dict[int, List[dict]] = {}
        for r in rows:
            by_t2.setdefault(int(r["timestamp"]), []).append(dict(r))
        by_t2 = _legalize_engine_steps(by_t2)
        rows = []
        for t in sorted(by_t2):
            rows.extend(by_t2[t])
        rows.sort(key=lambda r: (int(r["timestamp"]), r["name"]))
    except Exception as exc:  # noqa: BLE001
        print(f"[HIER] WARN traj repair skipped: {exc}", flush=True)
    with traj.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRAJ_HEADER})

    # Wave done-counter can miss the final unload tick; reconcile from traj pads.
    try:
        from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
            _load_task_dropoffs,
        )

        drop_map = _load_task_dropoffs(meta)
        seen_unload: Set[str] = set()
        by_name: Dict[str, List[dict]] = {}
        for r in rows:
            by_name.setdefault(str(r.get("name") or ""), []).append(r)
        for _n, seq in by_name.items():
            seq = sorted(seq, key=lambda x: int(x.get("timestamp") or 0))
            for a, b in zip(seq, seq[1:]):
                if int(b.get("timestamp") or 0) != int(a.get("timestamp") or 0) + 1:
                    continue
                tid_a = str(a.get("task-id") or "").strip()
                la = str(a.get("loaded", "")).lower() in ("true", "1", "yes")
                lb = str(b.get("loaded", "")).lower() in ("true", "1", "yes")
                if not tid_a or not la or lb:
                    continue
                cell = (int(b["X"]), int(b["Y"]))
                if cell in (drop_map.get(tid_a) or set()):
                    seen_unload.add(tid_a)
        traj_done = len(seen_unload)
        if traj_done > int(done):
            print(
                f"[HIER] reconcile done {done}->{traj_done} from traj pad unloads",
                flush=True,
            )
            done = int(traj_done)
    except Exception as exc:  # noqa: BLE001
        print(f"[HIER] WARN done-reconcile skipped: {exc}", flush=True)

    # Pass all failed ids; analyze_fifo only skips those never actually picked.
    val = validate_hybrid_trajectory(
        meta, traj, skip_task_ids=set(failed_seen) | set(scrubbed_pickup_tids)
    )
    iss = val.get("issues") or {}
    if gate is not None:
        hier_stats.update(gate.stats_dict())
    rep = {
        "scenario_id": sid,
        "method": (
            f"hybrid_astar_{joint_core_mode}_v1"
            if use_hierarchical
            else f"windowed_{joint_core_mode}_rhcr_v3"
        ),
        "joint_core": joint_core_mode,
        "plan_horizon": int(plan_horizon),
        "exec_horizon": int(exec_horizon),
        "weight": weight,
        "max_active": max_active,
        "plan": plan,
        "pipeline": bool(pipeline),
        "turn_aware": bool(turn_aware),
        "tasks_total": total,
        "tasks_completed": done,
        "tasks_failed": failed,
        "completion_ratio": round(done / max(1, total), 4),
        "sim_time": now,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "trajectory": str(traj),
        "validate_ok": bool(val.get("ok")),
        "validate_summary": format_validation_summary(val),
        "n_collisions": int(iss.get("n_collisions") or 0),
        "n_swaps": int(iss.get("n_swaps") or 0),
        "n_hard_wall": int(iss.get("n_hard_wall") or 0),
        "n_illegal_motion": int(iss.get("n_illegal_motion") or 0),
        "n_fifo": int(iss.get("n_fifo_violations") or 0),
        "n_pickup_cell": int(iss.get("n_pickup_cell_violations") or 0),
        "use_swapnet": False,
        "hierarchical": dict(hier_stats),
    }
    (OUT / f"{sid}_ecbs.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: rep[k] for k in (
        "sim_time", "completion_ratio", "validate_ok", "tasks_completed",
        "tasks_failed", "wall_seconds",
    )}, indent=2), flush=True)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Windowed multi-AGV MAPD (rewritten ECBS pipeline)."
    )
    ap.add_argument("--slot", type=int, default=3)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument(
        "--max-active",
        type=int,
        default=0,
        help="每波最多几台车领任务；0=等于该场景 AGV 数量（默认）",
    )
    ap.add_argument(
        "--plan",
        choices=("per_agent", "joint", "eecbs", "wcbs", "ecbs"),
        default="joint",
    )
    ap.add_argument("--weight", type=float, default=1.5)
    ap.add_argument("--time-limit", type=float, default=35.0)
    ap.add_argument("--max-expansions", type=int, default=300_000)
    ap.add_argument("--allow-solo-fallback", action="store_true")
    ap.add_argument("--no-initial-park", action="store_true")
    ap.add_argument("--deliver-batch", type=int, default=0)
    ap.add_argument("--turn-aware", action="store_true", default=True)
    ap.add_argument("--pipeline", action="store_true", default=True)
    ap.add_argument("--hierarchical", action="store_true")
    ap.add_argument("--wavenet", action="store_true")
    ap.add_argument("--wave-hard-cap", type=int, default=4)
    ap.add_argument("--no-traffic-accel", action="store_true")
    ap.add_argument(
        "--plan-horizon",
        type=int,
        default=24,
        help="RHCR window horizon H (0=classic one-shot ECBS)",
    )
    ap.add_argument(
        "--exec-horizon",
        type=int,
        default=12,
        help="steps committed per window before replan",
    )
    ap.add_argument(
        "--joint-core",
        choices=("prioritized", "ecbs"),
        default=None,
        help="medium/hard joint core (default prioritized; ecbs keeps legacy)",
    )
    ap.add_argument(
        "--allow-mode-switch",
        action="store_true",
        help="Deprecated unless --legacy-wave-ecbs (old gate handoff).",
    )
    ap.add_argument(
        "--legacy-wave-ecbs",
        action="store_true",
        help="Use old SceneDifficulty / wave ECBS path instead of M0+RulePark.",
    )
    ap.add_argument(
        "--no-traffic-recovery",
        action="store_true",
        help="Disable rule park/semaphore recovery on the default M0 path.",
    )
    args = ap.parse_args()
    plan = "joint" if args.plan in ("eecbs", "wcbs", "ecbs") else args.plan
    rep = solve_ecbs(
        args.slot,
        max_tasks=args.max_tasks,
        max_active=args.max_active,
        weight=args.weight,
        time_limit=args.time_limit,
        max_expansions=args.max_expansions,
        plan=plan,
        allow_solo_fallback=bool(args.allow_solo_fallback),
        initial_park=not bool(args.no_initial_park),
        deliver_batch=int(args.deliver_batch),
        turn_aware=bool(args.turn_aware),
        pipeline=bool(args.pipeline),
        use_hierarchical=bool(args.hierarchical or args.wavenet),
        use_wavenet=bool(args.wavenet),
        wave_hard_cap=int(args.wave_hard_cap),
        use_traffic_accel=not bool(args.no_traffic_accel),
        plan_horizon=int(args.plan_horizon),
        exec_horizon=int(args.exec_horizon),
        joint_core=args.joint_core,
        allow_mode_switch=bool(args.allow_mode_switch) or None,
        legacy_wave_ecbs=bool(args.legacy_wave_ecbs),
        traffic_recovery=not bool(args.no_traffic_recovery),
    )
    ok = bool(rep.get("validate_ok")) and float(rep.get("completion_ratio") or 0) >= 0.999
    return 0 if ok and not rep.get("tasks_failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
