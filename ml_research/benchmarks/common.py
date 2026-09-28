"""Shared helpers for AGV allocation benchmarks."""
from __future__ import annotations

import contextlib
import csv
import io
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ml_research.common.data_utils import load_main_copy
from ml_research.common.paths import POSITION_CSV, RESULTS, ROOT

BENCH_OUT = RESULTS / "benchmarks"
BENCH_OUT.mkdir(parents=True, exist_ok=True)

# Dispatcher pair-cost distance: "manhattan" (legacy) or "astar" (static-obs shortest path).
# Path planning in Simulation / A_Star is untouched — this only affects allocation costs/features.
# Shape curriculum defaults to astar via set_pair_cost_mode("astar") at train entry.
_PAIR_COST_MODE = os.environ.get("PAIR_COST_MODE", "manhattan").strip().lower()
_PAIR_COST_MODE_LOGGED = False

# Rolling reservation window for spacetime A* (avoids full-trajectory obstacle blow-up).
DEFAULT_MOVING_OBSTACLE_HORIZON = 20
_MO_HORIZON_LOGGED = False


def get_pair_cost_mode() -> str:
    return os.environ.get("PAIR_COST_MODE", _PAIR_COST_MODE).strip().lower() or "manhattan"


def set_pair_cost_mode(mode: str) -> None:
    """Set PAIR_COST_MODE for this process (also exports env for child clarity)."""
    global _PAIR_COST_MODE, _PAIR_COST_MODE_LOGGED
    m = (mode or "manhattan").strip().lower()
    if m in ("a*", "bfs", "shortest", "shortest_path"):
        m = "astar"
    _PAIR_COST_MODE = m
    os.environ["PAIR_COST_MODE"] = m
    _PAIR_COST_MODE_LOGGED = False


def _obs_fingerprint(static_obs: Sequence) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((int(p[0]), int(p[1])) for p in static_obs))


def _bfs_dist_to_goal(
    goal: Tuple[int, int],
    static_obs: Sequence,
    *,
    width: int = 21,
    height: int = 21,
    lo: int = 1,
) -> Dict[Tuple[int, int], int]:
    """Unweighted shortest-path distances from every free cell to goal (static obstacles only).

    Equivalent to A* length on a uniform grid; goal cell is walkable even if listed in obs
    (matches Simulation A* allowing the destination station).
    """
    obs = {(int(p[0]), int(p[1])) for p in static_obs}
    gx, gy = int(goal[0]), int(goal[1])
    dist: Dict[Tuple[int, int], int] = {(gx, gy): 0}
    q: deque = deque([(gx, gy)])
    while q:
        x, y = q.popleft()
        d = dist[(x, y)]
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if not (lo <= nx <= width and lo <= ny <= height):
                continue
            if (nx, ny) in dist:
                continue
            if (nx, ny) in obs and (nx, ny) != (gx, gy):
                continue
            dist[(nx, ny)] = d + 1
            q.append((nx, ny))
    return dist


class StaticAstarDistanceCache:
    """Cache BFS distance maps keyed by static-obstacle set + goal (AGV poses vary)."""

    def __init__(self) -> None:
        self._obs_key: Optional[Tuple[Tuple[int, int], ...]] = None
        self._by_goal: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self._obs_key = None
        self._by_goal.clear()

    def distance(
        self,
        start,
        goal,
        static_obs: Sequence,
        *,
        width: int = 21,
        height: int = 21,
    ) -> float:
        key = _obs_fingerprint(static_obs)
        if key != self._obs_key:
            self._obs_key = key
            self._by_goal.clear()
        g = (int(goal[0]), int(goal[1]))
        s = (int(start[0]), int(start[1]))
        if s == g:
            self.hits += 1
            return 0.0
        if g not in self._by_goal:
            self.misses += 1
            self._by_goal[g] = _bfs_dist_to_goal(g, static_obs, width=width, height=height)
        else:
            self.hits += 1
        d = self._by_goal[g].get(s)
        if d is None:
            return float("inf")
        return float(d)


def _manhattan_distance(pos1, pos2) -> float:
    return float(abs(int(pos1[0]) - int(pos2[0])) + abs(int(pos1[1]) - int(pos2[1])))


_ASTAR_DIST_CACHE = StaticAstarDistanceCache()


def travel_distance(sim, pos1, pos2) -> float:
    """Allocation travel distance under PAIR_COST_MODE (astar | manhattan).

    Uses only static obstacles (+ extra_obstacles via patched get_static_obstacles).
    Does NOT call spacetime A* / reservation — path execution stays in main copy.

    When static BFS is unreachable (maze maps: outer parking → inner pickup), fall back
    to Manhattan so the dispatcher can still assign; spacetime A* validates paths.
    """
    global _PAIR_COST_MODE_LOGGED
    mode = get_pair_cost_mode()
    if mode == "astar":
        if not _PAIR_COST_MODE_LOGGED:
            print(
                "[PAIR_COST] mode=A* (static-obstacle shortest-path / BFS; "
                "Simulation path planner unchanged)",
                flush=True,
            )
            _PAIR_COST_MODE_LOGGED = True
        static_obs: Sequence = []
        if hasattr(sim, "env") and hasattr(sim.env, "get_static_obstacles"):
            static_obs = sim.env.get_static_obstacles() or []
        w = h = 21
        if hasattr(sim, "map_size") and sim.map_size:
            w = int(sim.map_size[0]) if not isinstance(sim.map_size, int) else int(sim.map_size)
            h = int(sim.map_size[1]) if not isinstance(sim.map_size, int) else int(sim.map_size)
        d = _ASTAR_DIST_CACHE.distance(pos1, pos2, static_obs, width=w, height=h)
        if d != float("inf"):
            return d
        # Maze / segmented maps: static graph may mark corridor unreachable while
        # spacetime A* can still route — do not starve assign with inf pair cost.
        if hasattr(sim, "manhattan_distance"):
            return float(sim.manhattan_distance(pos1, pos2))
        return _manhattan_distance(pos1, pos2)
    if hasattr(sim, "manhattan_distance"):
        return float(sim.manhattan_distance(pos1, pos2))
    return float(abs(int(pos1[0]) - int(pos2[0])) + abs(int(pos1[1]) - int(pos2[1])))

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


def get_mod(force_reload: bool = False):
    return load_main_copy(force_reload=force_reload)


def enrich_remaining_time(task_states: dict, all_tasks: dict) -> None:
    """Inject remaining_time into task_states (not present in get_task_state)."""
    rt_map = {}
    for station, tasks in all_tasks.items():
        for t in tasks:
            rt_map[t["task_id"]] = t.get("remaining_time")
    for station, queue in task_states.items():
        for t in queue:
            t["remaining_time"] = rt_map.get(t["task_id"])


def load_scenario(
    task_csv: Path,
    position_csv: Path = POSITION_CSV,
    force_reload: bool = False,
) -> Tuple[Any, Any, dict, dict, int]:
    """Return (mod, env, agv_states, task_states, n_tasks)."""
    mod = get_mod(force_reload=force_reload)
    task_csv = Path(task_csv)
    position_csv = Path(position_csv)
    all_tasks = mod.get_task_list(str(task_csv))
    start_points, end_points, agv_list = mod.get_object_position(str(position_csv))
    env = mod.ENV(list(start_points.values()), list(end_points.values()))
    agv_states = mod.get_agv_state(agv_list)
    task_states = mod.get_task_state(start_points, all_tasks, end_points)
    enrich_remaining_time(task_states, all_tasks)
    n_tasks = sum(len(v) for v in task_states.values())
    return mod, env, agv_states, task_states, n_tasks


def soft_endpoint_multiplier(sim, task_info: dict) -> float:
    """Same soft-cap / capacity logic as baseline greedy (scale cost, never hard-block here)."""
    end_pts = task_info.get("end_points", [])
    if not end_pts:
        return 1.0
    # Prefer Simulation.endpoint_cost_multiplier when congestion control is present
    if hasattr(sim, "endpoint_cost_multiplier"):
        return float(sim.endpoint_cost_multiplier(end_pts))
    parked = sim._count_endpoint_occupancy(end_pts)
    active = sim._count_active_unloaders(end_pts)
    n_slots = max(1, len(end_pts))
    soft_cap = len(sim.agvs)
    mult = 1.0
    if parked >= n_slots:
        mult *= 5.0
    if active >= soft_cap:
        return float("inf")
    if active >= n_slots:
        mult *= 1.0 + 0.04 * (active - n_slots + 1)
    return mult


def endpoint_occupancy_features(sim, task_info: dict) -> Tuple[float, float, float]:
    """Return (parked/n_slots, active/n_slots, soft_mult_finite) for RL features."""
    end_pts = task_info.get("end_points", [])
    if not end_pts:
        return 0.0, 0.0, 1.0
    n_slots = max(1, len(end_pts))
    parked = sim._count_endpoint_occupancy(end_pts)
    active = sim._count_active_unloaders(end_pts)
    mult = soft_endpoint_multiplier(sim, task_info)
    soft = 10.0 if mult == float("inf") else float(mult)
    return parked / n_slots, active / n_slots, soft


def is_urgent_task(task_info: dict) -> bool:
    return str(task_info.get("priority", "")).lower() == "urgent"


def m0_pair_cost(sim, agv, task_id: str, task_info: dict) -> float:
    """M0 greedy cost: travel_dist × numbers_before_urgent(0.7) × endpoint soft_cap.

    travel_dist = A* (static obstacles) when PAIR_COST_MODE=astar, else Manhattan.
    """
    if (agv.name, task_id) in sim.tried_tasks:
        return float("inf")
    if sim.assign_cooldown.get(agv.name, -1) > sim.time:
        return float("inf")

    cost = float(travel_distance(sim, agv.state[:2], task_info["pickup_point"]))
    if cost == float("inf"):
        return float("inf")
    if task_info.get("numbers_before_urgent", -1) >= 0:
        cost *= 0.7
    cost *= soft_endpoint_multiplier(sim, task_info)
    return cost


def enhanced_pair_cost(
    sim,
    agv,
    task_id: str,
    task_info: dict,
    urgency_mode: str = "m0_plus",
) -> float:
    """
    M0-aligned cost with optional enhancements for M1–M5.

    Distance term follows PAIR_COST_MODE (astar | manhattan); path planner unchanged.

    Modes:
      - m0: exact baseline
      - m0_plus: M0 + mild RT pull for Urgent / before-urgent (keeps FIFO soft weight)
      - rt_strong: stronger remaining_time weighting (Hungarian / RH)
      - default / rt_weighted: legacy aliases
    """
    if urgency_mode in ("default", "m0"):
        return m0_pair_cost(sim, agv, task_id, task_info)

    if (agv.name, task_id) in sim.tried_tasks:
        return float("inf")
    if sim.assign_cooldown.get(agv.name, -1) > sim.time:
        return float("inf")

    cost = float(travel_distance(sim, agv.state[:2], task_info["pickup_point"]))
    if cost == float("inf"):
        return float("inf")

    # Keep M0 FIFO soft weight first (critical for buried urgents)
    if task_info.get("numbers_before_urgent", -1) >= 0:
        cost *= 0.7

    rt = task_info.get("remaining_time")
    urgent = is_urgent_task(task_info)

    if urgency_mode in ("m0_plus", "rt_strong"):
        if urgent and rt is not None:
            # Milder RT pull — stay close to travel distance to cut detours
            scale = max(0.35, min(1.0, float(rt) / 260.0))
            if urgency_mode == "rt_strong":
                scale = max(0.25, min(1.0, float(rt) / 220.0))
            cost *= scale
        elif urgent:
            cost *= 0.65 if urgency_mode == "m0_plus" else 0.55
        elif rt is not None and urgency_mode == "rt_strong":
            # Soft pull only; avoid snatching far non-urgent tokens
            cost *= max(0.7, min(1.0, float(rt) / 400.0))
    elif urgency_mode == "rt_weighted":
        # Legacy: RT-dominant (kept for ablation)
        if rt is not None:
            cost *= max(0.05, float(rt) / 300.0)
        elif urgent:
            cost *= 0.4
        if task_info.get("numbers_before_urgent", -1) >= 0:
            cost *= 0.8

    cost *= soft_endpoint_multiplier(sim, task_info)
    return cost


def base_pair_cost(sim, agv, task_id: str, task_info: dict, urgency_mode: str = "m0_plus") -> float:
    """Backward-compatible alias → enhanced_pair_cost."""
    return enhanced_pair_cost(sim, agv, task_id, task_info, urgency_mode=urgency_mode)


def make_assigned(sim, agv, task_id: str):
    if task_id not in sim.surface_tasks:
        return []
    info = sim.surface_tasks[task_id]
    ends = list(info.get("end_points") or [])
    if hasattr(sim, "env"):
        ends = filter_end_points_not_blocked(ends, obstacle_cell_set(sim.env))
    return {
        "agv": agv.name,
        "task_id": task_id,
        "agv_start_point": agv.state,
        "pickup_point": info["pickup_point"],
        "end_points": ends,
        "destination": info["destination"],
        "priority": info["priority"],
        "pickup_name": info["pickup_name"],
        "remaining_time": info.get("remaining_time"),
        "numbers_before_urgent": info.get("numbers_before_urgent", -1),
    }


def maybe_evacuate_on_block(sim) -> bool:
    """Clear tried-pairs so rematch can proceed (path_fallback removed)."""
    tried = getattr(sim, "tried_tasks", None)
    if isinstance(tried, set) and tried:
        tried.clear()
        return True
    return False


def _steps_up_to_time(steps_dict: dict, max_t: int) -> dict:
    """Keep only executed ticks (planned future steps must not appear in traj CSV)."""
    max_t = int(max_t)
    out: Dict[int, List[dict]] = {}
    for t, rows in steps_dict.items():
        ti = int(t)
        if ti <= max_t:
            out[ti] = rows
    return out


def write_trajectory(path: Path, steps_dict: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
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


def analyze_trajectory_conflicts(steps_dict: dict) -> dict:
    """Post-hoc spacetime check on recorded steps.

    - Vertex / stay conflict: two AGVs share the same cell at the same timestamp
    - Edge / head-on swap: A@P1,B@P2 at t and A@P2,B@P1 at t+1 (P1!=P2)
    """
    by_t: Dict[int, Dict[str, Tuple[int, int]]] = {}
    for t, steps in steps_dict.items():
        by_t[int(t)] = {s["name"]: (int(s["X"]), int(s["Y"])) for s in steps}

    collisions: List[dict] = []
    swaps: List[dict] = []
    times = sorted(by_t.keys())
    for t in times:
        positions = by_t[t]
        names = list(positions.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if positions[a] == positions[b]:
                    collisions.append(
                        {
                            "t": t,
                            "agv_a": a,
                            "agv_b": b,
                            "cell": positions[a],
                        }
                    )
                if t + 1 in by_t and a in by_t[t + 1] and b in by_t[t + 1]:
                    pa, pb = positions[a], positions[b]
                    na, nb = by_t[t + 1][a], by_t[t + 1][b]
                    if pa == nb and pb == na and pa != pb:
                        swaps.append(
                            {
                                "t": t,
                                "agv_a": a,
                                "agv_b": b,
                                "cell_a": pa,
                                "cell_b": pb,
                            }
                        )
    n_coll = len(collisions)
    n_swap = len(swaps)
    return {
        "collisions": n_coll,
        "swaps": n_swap,
        "n_conflicts": n_coll + n_swap,
        "conflict_free": n_coll == 0 and n_swap == 0,
        "collision_events": collisions[:20],
        "swap_events": swaps[:20],
    }


def get_moving_obstacle_horizon() -> int:
    """Steps ahead to reserve other AGVs in spacetime A* (rolling window). 0 = disabled."""
    v = os.environ.get("MOVING_OBSTACLE_HORIZON", "").strip()
    if v != "":
        try:
            return max(0, int(v))
        except ValueError:
            pass
    return DEFAULT_MOVING_OBSTACLE_HORIZON


def patch_moving_obstacle_horizon(sim, horizon: Optional[int] = None) -> int:
    """Enable rolling moving_obstacle reservation for spacetime A* (main copy reads env).

    Sets MOVING_OBSTACLE_HORIZON so A_Star / plan_relocation only block other AGVs
    within plan_t0 + horizon steps (beyond that: no freeze-at-end over-blocking).

    Also wraps ``is_valid_path`` with the same horizon so A* results are not rejected
    by freeze-at-end occupancy beyond the planning window (second-wave starvation).
    """
    global _MO_HORIZON_LOGGED
    h = int(horizon if horizon is not None else get_moving_obstacle_horizon())
    sim._mo_horizon = h
    if h <= 0:
        os.environ["MOVING_OBSTACLE_HORIZON"] = "0"
        sim._mo_horizon_patched = True
        return 0
    os.environ["MOVING_OBSTACLE_HORIZON"] = str(h)
    if getattr(sim, "_mo_horizon_patched", False):
        return h
    sim._mo_horizon_patched = True

    # Align path validation with A* horizon (critical for assign after first wave)
    if not getattr(sim.env, "_mo_horizon_valid_patched", False):
        sim.env._mo_horizon_valid_patched = True
        orig_valid = sim.env.is_valid_path

        def is_valid_path_horizon(end_points, new_path, time, agv_name=None):
            if not new_path:
                return False
            if not any(p[:2] in end_points for p in new_path):
                return False
            hard = get_hard_wall_cells(sim.env)
            from simulation.engine import mo_occupied_cell, path_has_spacetime_conflict

            for state in new_path:
                cell = tuple(state[:2])
                # Hard maze walls are never valid unload/travel cells
                if cell in hard:
                    return False
            for i in range(1, len(new_path)):
                a = tuple(new_path[i - 1][:2])
                b = tuple(new_path[i][:2])
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1:
                    return False
            # Same occupancy semantics as A* / kin reject (no idle-vanish holes).
            if path_has_spacetime_conflict(
                new_path, sim.env.moving_obstacles, agv_name
            ):
                return False
            return True

        sim.env.is_valid_path = is_valid_path_horizon

    # Keep module constant in sync (A_Star reads it at call time via global).
    try:
        import simulation.engine as _eng

        _eng.MOVING_OBSTACLE_HORIZON = int(h)
    except Exception:  # noqa: BLE001
        pass

    if not _MO_HORIZON_LOGGED:
        print(
            f"[MO-HORIZON] rolling window={h} steps "
            f"(planned+floor+past-end always reserved; A_Star via MOVING_OBSTACLE_HORIZON; "
            f"0=legacy label only)",
            flush=True,
        )
        _MO_HORIZON_LOGGED = True
    return h



# --- execution shield (vertex/swap repair at each sim tick) ---
from ml_research.benchmarks.execution_shield import (  # noqa: E402
    _one_step_toward,
    _sanitize_appended_path,
    finalize_task_carry,
    patch_conflict_free_execution,
)


def urgent_completion_stats(steps_dict: dict, task_rt: Dict[str, Optional[float]]) -> dict:
    """urgent_on_time / urgent_late using last timestamp of each urgent task-id."""
    last_t: Dict[str, int] = {}
    for t, steps in steps_dict.items():
        for s in steps:
            tid = str(s.get("task-id", "") or "")
            if tid and tid in task_rt and task_rt[tid] is not None:
                last_t[tid] = max(last_t.get(tid, -1), int(s["timestamp"]))

    on_time = late = missing = 0
    for tid, rt in task_rt.items():
        if rt is None:
            continue
        if tid not in last_t:
            missing += 1
            continue
        if last_t[tid] <= float(rt):
            on_time += 1
        else:
            late += 1
    return {
        "urgent_on_time": on_time,
        "urgent_late": late,
        "urgent_missing": missing,
        "urgent_total": on_time + late + missing,
    }


def collect_task_rt(task_csv: Path) -> Dict[str, Optional[float]]:
    mod = get_mod()
    all_tasks = mod.get_task_list(str(task_csv))
    out = {}
    for station, tasks in all_tasks.items():
        for t in tasks:
            if t.get("priority") == "Urgent" or t.get("remaining_time") is not None:
                out[t["task_id"]] = t.get("remaining_time")
    return out


def _repair_sim_steps_upto(sim, max_t: int) -> None:
    """Optional post-hoc step XY repair.

    Disabled by default: writing repaired poses back into live ``path``/``steps``
    during checkpoints collapsed every AGV onto its spawn cell in the export CSV
    (loaded/task-id still flipped → false task_carry / fifo / display failures).

    Set ``AGV_TRAJ_REPAIR=1`` to enable. Even then, **never mutate ``agv.path``** —
    only step rows used for CSV export.
    """
    import os

    if str(os.environ.get("AGV_TRAJ_REPAIR", "0")).strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return
    try:
        from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
            repair_trajectory_rows,
        )

        rows = []
        for agv in sim.agvs:
            for s in agv.steps:
                ts = int(s.get("timestamp", -1))
                if ts < 0 or ts > int(max_t):
                    continue
                rows.append(
                    {
                        "timestamp": ts,
                        "name": str(s.get("name") or agv.name),
                        "X": int(s["X"]),
                        "Y": int(s["Y"]),
                        "pitch": int(s.get("pitch") or 0),
                        "loaded": s.get("loaded", "false"),
                        "destination": s.get("destination", ""),
                        "Emergency": s.get("Emergency", "false"),
                        "task-id": str(s.get("task-id") or ""),
                    }
                )
        if not rows:
            return
        hard_walls = get_hard_wall_cells(sim.env)
        lo, hi = _grid_bounds(sim)
        fixed, stats = repair_trajectory_rows(rows, hard_walls=hard_walls, lo=lo, hi=hi)
        if int(stats.get("held_moves", 0)) > 0:
            carry = getattr(sim, "_shield_task_carry_stats", None)
            if carry is None:
                carry = {}
                sim._shield_task_carry_stats = carry
            carry["traj_repair_held"] = int(carry.get("traj_repair_held", 0)) + int(
                stats["held_moves"]
            )
        by_name_ts = {(r["name"], int(r["timestamp"])): r for r in fixed}
        for agv in sim.agvs:
            new_steps = []
            for s in agv.steps:
                ts = int(s.get("timestamp", -1))
                if ts > int(max_t):
                    new_steps.append(s)
                    continue
                key = (str(s.get("name") or agv.name), ts)
                fr = by_name_ts.get(key)
                if fr is not None:
                    s["X"] = int(fr["X"])
                    s["Y"] = int(fr["Y"])
                    s["pitch"] = int(fr.get("pitch") or 0)
                new_steps.append(s)
            agv.steps = new_steps
        # Intentionally do NOT rewrite agv.path — that corrupts execution.
    except Exception as exc:
        import sys

        print(f"[SHIELD] traj_repair failed: {exc}", file=sys.stderr, flush=True)


def run_sim_loop(
    sim,
    max_time: int,
    label: str,
    progress_every: int = 50,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 50,
    wall_timeout: Optional[float] = None,
    should_stop: Optional[Callable[[Any], bool]] = None,
) -> Tuple[dict, bool]:
    """Advance simulation; quiet stdout, progress on stderr.

    Stops early on sim max_time, wall_timeout (seconds), or ``should_stop(sim)``.
    Any early stop yields forced=True (caller may inspect gate / leftover tasks).
    """
    import time as _time

    buf = io.StringIO()
    forced = False
    t0 = _time.perf_counter()
    n_tasks0 = sum(len(v) for v in sim.task_states.values())
    last_done = -1
    with contextlib.redirect_stdout(buf):
        while not sim.all_over():
            sim.time_forward()
            elapsed = _time.perf_counter() - t0
            left = sum(len(v) for v in sim.task_states.values())
            done = int(n_tasks0 - left)
            # Progress must go to stderr — stdout is redirected into buf.
            # Avoid per-task spam: tick cadence, or every 10 completions.
            if (
                sim.time % progress_every == 0
                or (done != last_done and (done % 10 == 0 or done == n_tasks0))
            ):
                print(
                    f"  [{label}] t={sim.time} done={done}/{n_tasks0} "
                    f"left={left} surface={len(sim.surface_tasks)} "
                    f"wall={elapsed:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                last_done = done
            if (
                checkpoint_path is not None
                and checkpoint_every > 0
                and sim.time % checkpoint_every == 0
            ):
                # Clip to executed ticks — planned future steps often still conflict
                # until CONFLICT_HOLD resolves them when that tick becomes current.
                write_trajectory(
                    checkpoint_path,
                    _steps_up_to_time(sim.steps_reorganize(), int(sim.time)),
                )
            if should_stop is not None and bool(should_stop(sim)):
                forced = True
                print(
                    f"  [{label}] should_stop t={sim.time} done={done}/{n_tasks0}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            if sim.time >= max_time:
                forced = True
                break
            if wall_timeout is not None and elapsed >= wall_timeout:
                forced = True
                print(
                    f"  [{label}] wall timeout {wall_timeout}s "
                    f"t={sim.time} done={done}/{n_tasks0}",
                    file=sys.stderr,
                    flush=True,
                )
                break
        # No execution-shield finalize; optional export-only XY repair if enabled.
        _repair_sim_steps_upto(sim, int(sim.time))
        steps = _steps_up_to_time(sim.steps_reorganize(), sim.time)
    return steps, forced


def patch_extra_obstacles(env, extra_obstacles: List[Tuple[int, int]]) -> None:
    """Monkey-patch ENV.get_static_obstacles to include extra static blocks."""
    if not extra_obstacles:
        return
    orig = env.get_static_obstacles
    extra = [tuple(p) for p in extra_obstacles]
    env._extra_obstacle_cells = set(extra)

    def _combined():
        return list(orig()) + extra

    env.get_static_obstacles = _combined


def obstacle_cell_set(env) -> set:
    """All static blocked cells (base stations + maze extra_obstacles)."""
    return {(int(p[0]), int(p[1])) for p in (env.get_static_obstacles() or [])}


def filter_end_points_not_blocked(
    end_points: Sequence, blocked: set
) -> List[Tuple[int, int]]:
    """Drop unload neighbors that sit on walls/stations footprints.

    Competition maps put unload slots on the 4-neighbors of a destination
    center. Maze ``extra_obstacles`` often cover the west neighbor — if that
    cell stays an end_point, planners treat it as a goal and drive into a wall.
    """
    out: List[Tuple[int, int]] = []
    seen = set()
    for e in end_points or []:
        if e is None:
            continue
        cell = (int(e[0]), int(e[1]))
        if cell in blocked or cell in seen:
            continue
        seen.add(cell)
        out.append(cell)
    return out


def sanitize_task_end_points(env, task_states: dict) -> int:
    """Rewrite task_states[*].end_points to exclude blocked cells. Returns #dropped."""
    blocked = obstacle_cell_set(env)
    dropped = 0
    for _station, queue in (task_states or {}).items():
        for t in queue:
            if not isinstance(t, dict):
                continue
            eps = list(t.get("end_points") or [])
            cleaned = filter_end_points_not_blocked(eps, blocked)
            dropped += max(0, len(eps) - len(cleaned))
            t["end_points"] = cleaned
    return dropped


def sanitize_sim_end_points(sim) -> int:
    """Sanitize surface / in-flight / queued task end_points on a live sim."""
    blocked = obstacle_cell_set(sim.env)
    dropped = 0

    def _clean_info(info: dict) -> None:
        nonlocal dropped
        if not isinstance(info, dict) or "end_points" not in info:
            return
        eps = list(info.get("end_points") or [])
        cleaned = filter_end_points_not_blocked(eps, blocked)
        dropped += max(0, len(eps) - len(cleaned))
        info["end_points"] = cleaned

    for _station, queue in (getattr(sim, "task_states", None) or {}).items():
        for t in queue:
            _clean_info(t)
    for info in (getattr(sim, "surface_tasks", None) or {}).values():
        _clean_info(info)
    for info in (getattr(sim, "_inflight_tasks", None) or {}).values():
        _clean_info(info)
    return dropped


def get_hard_wall_cells(env) -> set:
    """Editor maze walls (extra_obstacles) — AGVs must never occupy these cells."""
    hw = getattr(env, "_extra_obstacle_cells", None)
    if hw is not None:
        return set(hw)
    return set()


def path_on_hard_walls(path, hard_walls: set, end_points=None) -> bool:
    """True if any pose sits on a hard maze wall.

    Hard walls are never valid — even if a stale end_point lists them as unload
    slots (legacy 4-neighbor destinations overlapping shelves).
    """
    if not path or not hard_walls:
        return False
    for state in path:
        if tuple(state[:2]) in hard_walls:
            return True
    return False


def _grid_bounds(sim) -> Tuple[int, int]:
    w = h = 20
    if hasattr(sim, "map_size") and sim.map_size:
        if isinstance(sim.map_size, int):
            w = h = int(sim.map_size)
        else:
            w, h = int(sim.map_size[0]), int(sim.map_size[1])
    return w, h
