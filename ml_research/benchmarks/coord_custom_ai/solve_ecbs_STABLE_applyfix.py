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
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    """Export-only: split illegal move_and_turn by keeping prior pitch on move ticks.

    Does not rewrite XY (avoids introducing vertex/swap artifacts). Teleports from
    shield path-holes remain a separate engine issue.
    """
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
            if abs(ax - bx) + abs(ay - by) != 1:
                continue
            pa = int(a.get("pitch") or 0) % 360
            pb = int(b.get("pitch") or 0) % 360
            if pa != pb:
                b["pitch"] = pa

    out: dict = {}
    for _name, seq in by_name.items():
        for r in seq:
            out.setdefault(int(r["timestamp"]), []).append(r)
    for t in out:
        out[t].sort(key=lambda r: str(r["name"]))
    return out


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
) -> dict:
    """True easy-path: Simulation greedy + spacetime A* (+ optional SwapNet).

    Wave-batched 'astar' inside solve_ecbs is NOT M0 — scene=easy must use the
    continuous engine loop (assign-time promote, tick-level planning).
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

    print(
        f"[HIER] scene=easy -> engine M0"
        f"{'+SwapNet' if use_swapnet else ''} "
        f"tasks={total}",
        flush=True,
    )
    hier_stats = dict(hier_stats)
    hier_stats["last_planner"] = "m0_engine"
    hier_stats["last_reason"] = "scene_easy_m0_engine"
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
    patch_conflict_free_execution(sim)

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

    sid = str(meta["id"])
    traj = TRAJ / f"{sid}_m0_engine{'_swap' if use_swapnet else ''}.csv"
    traj.parent.mkdir(parents=True, exist_ok=True)
    max_time = int(meta.get("max_sim_time") or 500000)
    # Prefer explicit meta wall; treat "infinite" (>>1e8) as 30min safety cap.
    raw_wall = float(meta.get("wall_timeout") or 1800.0)
    wall_timeout = None if raw_wall <= 0 else (1800.0 if raw_wall > 1e8 else raw_wall)
    prog_every = int(meta.get("progress_every") or 100)
    print(
        f"[M0] start max_time={max_time} wall_timeout={wall_timeout} "
        f"progress_every={prog_every} swapnet={bool(use_swapnet)}",
        flush=True,
    )

    steps, forced = run_sim_loop(
        sim,
        max_time=max_time,
        label=f"{sid}/m0_engine",
        progress_every=max(1, prog_every),
        checkpoint_path=None,
        checkpoint_every=0,
        wall_timeout=wall_timeout,
    )
    steps = _legalize_engine_steps(steps)
    write_trajectory(traj, steps)
    left = sum(len(v) for v in sim.task_states.values())
    done = int(n_tasks - left)
    rs = getattr(sim, "_replan_stats", {}) or {}
    swap_stats["station_swap"] = int(rs.get("station_swap") or 0)

    val = validate_hybrid_trajectory(meta, traj)
    iss = val.get("issues") or {}
    conf = analyze_trajectory_conflicts(steps)
    rep = {
        "scenario_id": sid,
        "method": "m0_engine_swapnet" if swap_stats["enabled"] else "m0_engine",
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
        "use_swapnet": bool(swap_stats["enabled"]),
        "swapnet": dict(swap_stats),
        "forced_stop": bool(forced and left > 0),
        "hierarchical": hier_stats,
        "use_hierarchical": bool(use_hierarchical),
        "use_wavenet": bool(use_wavenet),
    }
    print(
        f"[HIER] M0-engine done={done}/{n_tasks} sim={rep['sim_time']} "
        f"wall={rep['wall_seconds']}s validate={rep['validate_summary']} "
        f"swap={swap_stats.get('station_swap', 0)}",
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


def _build_grid(static: Set[Cell], stations: Set[Cell], W: int = 20, H: int = 20) -> GridMap:
    grid = []
    for y in range(1, H + 1):
        row = []
        for x in range(1, W + 1):
            blocked = (x, y) in static or (x, y) in stations
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
    """
    s0 = start if start in free else _snap_free(start, free, static, prefer=goal)
    g0 = _snap_free(goal, free, static, prefer=start)
    comp = _component_from(s0, free)
    if not comp:
        return g0
    if g0 in comp:
        return g0
    return min(comp, key=lambda c: (_manh(c, g0), c))


def _pickup_goal(
    task: dict,
    start: Cell,
    free: Set[Cell],
    static: Set[Cell],
) -> Cell:
    """True station pickup cell when reachable; snap only for wall-island pads.

    Never invent a far alternate stand — loaded may rise only at this cell.
    """
    raw = tuple(task["pickup_point"])
    s0 = start if start in free else _snap_free(start, free, static, prefer=raw)
    if raw in free:
        comp = _component_from(s0, free)
        if raw in comp:
            return raw
    return _snap_reachable(s0, raw, free, static)


def _valid_unload_pads(
    task: dict,
    free: Set[Cell],
    static: Set[Cell],
    *,
    prefer: Optional[Cell] = None,
) -> List[Cell]:
    """Unload pads that are walkable (never maze walls / stations)."""
    raw = [tuple(e) for e in (task.get("end_points") or []) if e is not None]
    pads: List[Cell] = []
    seen: Set[Cell] = set()
    for e in raw:
        if e in static:
            continue
        c = e if e in free else _snap_free(e, free, static, prefer=prefer)
        if prefer is not None:
            c = _snap_reachable(prefer, c, free, static)
        if c in free and c not in seen and c not in static:
            pads.append(c)
            seen.add(c)
    if pads:
        return pads
    # Last resort: snap around destination name neighbors / raw ends
    for e in raw:
        c = _snap_free(e, free, static, prefer=prefer)
        if prefer is not None:
            c = _snap_reachable(prefer, c, free, static)
        if c in free and c not in seen and c not in static:
            pads.append(c)
            seen.add(c)
    return pads


def _dest_inflight_cap(
    task: dict,
    free: Set[Cell],
    static: Set[Cell],
    n_agvs: int,
) -> int:
    """Soft per-destination concurrency ≈ unload pads + staging neighbors."""
    pads = _valid_unload_pads(task, free, static)
    n_pads = len(pads) if pads else 2
    # Allow modest staging beyond pad count; still bound by fleet size.
    return max(3, min(int(n_pads) + 2, max(3, int(n_agvs) - 1)))


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
    hold_after: int = 8,
    seed_reserved: Optional[Dict[Tuple[int, Cell], str]] = None,
) -> Optional[Dict[str, List[Cell]]]:
    """M0-style baseline: priority spacetime A* — all movers get paths in parallel time.

    Higher-priority agents reserved first; lower-priority avoid their spacetime cells.
    ``seed_reserved`` lets a higher-priority ECBS core own spacetime first (hybrid).
    Non-movers are seeded as stationary reservations so plans cannot ghost-through
    idle AGVs (avoids false ``_cell_paths_conflict`` → serial collapse).
    Returns None only if some mover cannot be routed.
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
    order = sorted(movers, key=lambda n: (_manh(starts[n], goals[n]), n))
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
        for t in range(len(path), len(path) + int(hold_after)):
            reserved[(t, path[-1])] = n
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
    ECBS on all AGVs. Medium scenes force a tiny core (≤2).
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
        k_req = max(1, min(2, k_req, len(pool)))
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
    """Static A* treating ``blocked`` as permanent obstacles (others freeze)."""
    if start == goal:
        return [start]
    if goal in blocked and goal != start:
        return None
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


def _st_paths_ok(
    paths: Optional[Dict[str, List[Cell]]],
    movers: Set[str],
    goals: Optional[Dict[str, Cell]] = None,
) -> bool:
    if paths is None or not movers:
        return False
    check = paths
    if goals is not None:
        check = _trim_paths_to_goals(paths, goals, movers)
    # Shared pads: do not pad holds — ST already sequenced arrivals.
    return not _cell_paths_conflict(check, list(movers), pad_holds=False)


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
    if goal in blocked_plan:
        free_cells = {
            (x, y)
            for x in range(1, 21)
            for y in range(1, 21)
            if (x, y) not in blocked_plan
        }
        if not free_cells:
            return now, False
        goal = _snap_free(goal, free_cells, blocked_plan, prefer=pose[mover][:2])
        if goal in blocked_plan:
            return now, False
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
        cands = []
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx == 0 and dy == 0:
                    continue
                c = (ox + dx, oy + dy)
                if not (1 <= c[0] <= 20 and 1 <= c[1] <= 20):
                    continue
                if c in blocked_plan or c in keep_clear or c in occ:
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
        steps_by, pose, timelines, now, loaded=loaded, dest=dest, tid=tid
    )
    return now, True


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
            seq.append((x, y, need))
            pitch = need
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
) -> int:
    names = list(pose.keys())
    T = max((len(timelines.get(n) or []) for n in names), default=0)
    if T <= 0:
        return now
    for k in range(T):
        t = now + k + 1
        for n in names:
            seq = timelines.get(n) or []
            if k < len(seq):
                pose[n] = seq[k]
            steps_by[n].append(
                _hold(
                    n,
                    pose[n],
                    t,
                    loaded=bool(loaded.get(n, False)),
                    dest=dest.get(n, "") if loaded.get(n, False) else "",
                    tid=tid.get(n, "") if loaded.get(n, False) else "",
                )
            )
    return now + T


def _plan_paths_as_poses(
    pose: Dict[str, Pose],
    paths: Dict[str, List[Cell]],
) -> Dict[str, List[Pose]]:
    """Expand ECBS cell paths with *synchronized* turn inserts.

    Independent per-agent turn expansion desyncs spacetime and causes
    collisions; keep a global clock so every agent advances together.
    """
    names = list(pose.keys())
    cell_paths: Dict[str, List[Cell]] = {}
    for n in names:
        cells = list(paths.get(n) or [pose[n][:2]])
        if not cells:
            cells = [pose[n][:2]]
        if cells[0] != pose[n][:2]:
            cells = [pose[n][:2]] + cells
        cell_paths[n] = cells
    T = max((len(cell_paths[n]) for n in names), default=1)
    for n in names:
        while len(cell_paths[n]) < T:
            cell_paths[n].append(cell_paths[n][-1])

    cur: Dict[str, Pose] = {n: pose[n] for n in names}
    out: Dict[str, List[Pose]] = {n: [] for n in names}
    for i in range(T - 1):
        nxt_cell = {n: cell_paths[n][i + 1] for n in names}
        need_turn: Dict[str, int] = {}
        for n in names:
            x, y, pitch = cur[n]
            nx, ny = nxt_cell[n]
            if (nx, ny) == (x, y):
                continue
            need = _pitch_from_delta(nx - x, ny - y, pitch)
            if need != int(pitch) % 360:
                need_turn[n] = need
        if need_turn:
            for n in names:
                x, y, pitch = cur[n]
                if n in need_turn:
                    cur[n] = (x, y, need_turn[n])
                out[n].append(cur[n])
        for n in names:
            x, y, pitch = cur[n]
            nx, ny = nxt_cell[n]
            cur[n] = (nx, ny, pitch)
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
    queues.setdefault(st, []).insert(0, task)
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
) -> dict:
    # use_swapnet kept for easy M0-engine path
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

    total = sum(len(v) for v in queues.values())
    static_list = list(env.get_static_obstacles() or [])
    static = {(int(p[0]), int(p[1])) for p in static_list}
    map_json = meta.get("map_json")
    stations: Set[Cell] = set()
    if map_json and Path(map_json).exists():
        j = json.loads(Path(map_json).read_text(encoding="utf-8"))
        stations = {
                (int(p["x"]), int(p["y"]))
                for p in (j.get("pickups") or []) + (j.get("dropoffs") or [])
            }
    blocked_plan = set(static) | set(stations)
    free = {
        (x, y)
        for x in range(1, 21)
        for y in range(1, 21)
        if (x, y) not in blocked_plan
    }
    grid = _build_grid(static, stations)

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
        return _plan_joint_core_paths(
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
    # Sticky planner affinity until unload: "astar" | "joint"
    agv_affinity: Dict[str, str] = {}
    # Per-station in-flight queue order (assign-time promote, pickup-time FIFO).
    inflight_station: Dict[str, List[str]] = {}
    picked_ids: Set[str] = set()
    # Guard against infinite requeue when every pickup wave fails at the same t.
    pickup_stall_streak = 0
    # P0 anti-spin: same sim-t / done with no progress
    stall_t = -1
    stall_done = -1
    no_progress_waves = 0
    joint_core_override: Optional[str] = None

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
        tid_s = str(task.get("task_id") or "")
        return str(
            task.get("pickup_name")
            or task.get("start_point")
            or (tid_s.rsplit("-", 1)[0] if "-" in tid_s else tid_s or "UNK")
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
        )
        from ml_research.benchmarks.wave_net.policy import load_wave_policy, select_wave_agents

        hardness0 = map_hardness(
            extra_obstacles=list(meta.get("extra_obstacles") or []),
            static=static,
            free=free,
        )
        escalate_net = None
        for ck in (CKPT / "escalate_net_map_v1.pt", CKPT / "escalate_net_v1.pt"):
            if ck.exists():
                try:
                    if "map" in ck.name:
                        from ml_research.benchmarks.hier_coord.escalate_map import (
                            load_escalate_net_map,
                        )

                        escalate_net = load_escalate_net_map(ck)
                    else:
                        from ml_research.benchmarks.hier_coord.escalate import (
                            load_escalate_net,
                        )

                        escalate_net = load_escalate_net(ck)
                    hier_stats["escalate_ckpt"] = str(ck)
                    break
                except Exception as exc:  # noqa: BLE001
                    hier_stats["escalate_load_error"] = str(exc)
        wave_net, wave_meta = load_wave_policy(enabled=bool(use_wavenet))
        hier_stats["wave_meta"] = wave_meta
        scene_diff_net = None
        try:
            from ml_research.benchmarks.hier_coord.scene_difficulty.model import (
                SCENE_DIFF_CKPT,
                load_scene_difficulty_net,
            )

            if SCENE_DIFF_CKPT.exists():
                scene_diff_net = load_scene_difficulty_net(SCENE_DIFF_CKPT)
                hier_stats["scene_diff_ckpt"] = str(SCENE_DIFF_CKPT)
        except Exception as exc:  # noqa: BLE001
            hier_stats["scene_diff_load_error"] = str(exc)
        # Mode-switch: only when explicitly requested (meta/arg). Default OFF so
        # static easy demos (SH01) keep the M0+SwapNet short-circuit.
        if allow_mode_switch is None:
            allow_switch = bool(meta.get("allow_mode_switch", False))
        else:
            allow_switch = bool(allow_mode_switch)
        gate = LifelongGateTracker(
            map_hardness=float(hardness0),
            threshold=float(hierarchical_threshold),
            force=bool(hierarchical_force),
            hard_cap=int(wave_hard_cap),
            max_active=int(max_active),
            escalate_net=escalate_net,
            scene_diff_net=scene_diff_net,
            use_escalate_ai=True,
            use_map_obs=True,
            use_scene_diff=True,
            allow_mode_switch=bool(allow_switch),
        )
        hier_stats["hardness"] = float(hardness0)
        hier_stats["allow_mode_switch"] = bool(allow_switch)

        # Probe SceneDifficultyNet once: static easy (no mode switch) → true
        # engine M0(+SwapNet). Lifelong / switchable maps stay in the wave loop.
        gate.set_scene(static=static, stations=stations, pose=pose)
        probe = gate.decide(
            assigned={},
            queues=queues,
            recent_joint_fail=0,
        )
        if (
            not allow_switch
            and str(getattr(probe, "planner", "") or "") == "astar"
            and str(getattr(probe, "scene_label", "") or gate.last_scene_label)
            == "easy"
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
            return _solve_via_m0_engine(
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
            )
        elif allow_switch and str(getattr(probe, "planner", "") or "") == "astar":
            print(
                "[HIER] allow_mode_switch=1 → skip M0 short-circuit "
                f"(probe={probe.planner}/{gate.last_scene_label})",
                flush=True,
            )

    plan = "joint" if plan in ("eecbs", "wcbs", "ecbs") else plan
    print(
        f"[ECBS] SH slot={slot} agents={len(names)} tasks={total} "
        f"max_active={max_active} plan={plan} pipeline={pipeline} "
        f"turn_aware={turn_aware} hier={use_hierarchical} wavenet={use_wavenet} "
        f"joint_core={joint_core_mode}",
        flush=True,
    )

    while any(queues.values()) or pending_delivery or pending_pickup:
        wall_budget = float(meta.get("wall_timeout") or 1e9)
        if wall_budget < 1e8 and (time.perf_counter() - t0) > wall_budget:
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

        # ---- SceneDifficultyNet: choose planner BEFORE claim ----
        # A* (main-copy baseline): fill all free AGVs; no max_active / WaveNet k.
        # ECBS: claim_cap = suggested_k (selector concurrency window).
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
            # main copy: every idle AGV may claim; k-cap is ECBS-only
            claim_cap = len(names)
        elif wave_planner == "hybrid":
            # medium: claim full fleet, joint only on small core after split
            claim_cap = len(names)
        elif pre_gate is not None:
            claim_cap = max(1, min(int(max_active), int(pre_gate.suggested_k)))
        else:
            claim_cap = int(max_active)

        if gate is not None and pre_gate is not None:
                    print(
                f"[SCENE] before-claim label={gate.last_scene_label} "
                f"p_hard={gate.last_scene_p_hard:.3f} k={claim_cap} "
                f"pol={gate.last_k_policy} planner={wave_planner} "
                f"joint_core={joint_core_mode} "
                f"reason={pre_gate.reason}",
                        flush=True,
                    )

        assigned: Dict[str, dict] = {}
        # Resume incomplete pickups before claiming new surface heads.
        for agv, task in list(pending_pickup.items()):
            assigned[agv] = task
            pending_pickup.pop(agv, None)
        # Carrying / in-flight pickup AGVs are busy — do not claim new pickups.
        free_agvs = [
            n
            for n in names
            if n not in pending_delivery and n not in assigned
        ]
        # main-copy surface (navigation.py): assign head → pop → next head
        # immediately becomes surface. Same station can feed many AGVs in one
        # wave (SwapNet relies on this). used_stations-per-wave was WRONG.
        claim_room = (
            len(free_agvs)
            if (wave_planner == "astar" or gate is not None)
            else max(0, int(claim_cap) - len(pending_delivery))
        )
        picks = 0
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
            heads: List[Tuple[str, dict]] = []
            for cand in order:
                if queues.get(cand) and not (inflight_station.get(cand) or []):
                    heads.append((cand, queues[cand][0]))
            if not heads:
                break

            if wave_planner in ("astar", "hybrid") or gate is not None:
                # main-copy greedy: global min (manhattan × urgent soft weight)
                best = None
                best_cost = float("inf")
                for st, task in heads:
                    pk = tuple(task["pickup_point"])
                    urg = 0.7 if int(task.get("numbers_before_urgent", -1)) >= 0 else 1.0
                    for agv in free_agvs:
                        cost = float(_manh(pose[agv][:2], pk)) * urg
                        if cost < best_cost or (
                            cost == best_cost and best is not None and agv < best[0]
                        ):
                            best_cost = cost
                            best = (agv, st, task)
                if best is None:
                    break
                agv, st, task = best
            else:
                # ECBS window: round-robin among claimable surface heads
                task = None
                st = None
                for k in range(len(order)):
                    cand = order[(rr + k) % len(order)]
                    if queues.get(cand) and not (inflight_station.get(cand) or []):
                        st, task = cand, queues[cand][0]
                        rr = (rr + k + 1) % max(1, len(order))
                        break
                if not task or st is None:
                    # Retry after clearing stale inflight.
                    if any(queues.values()):
                        inflight_station.clear()
                        for k in range(len(order)):
                            cand = order[(rr + k) % len(order)]
                            if queues.get(cand):
                                st, task = cand, queues[cand][0]
                                rr = (rr + k + 1) % max(1, len(order))
                                break
                if not task or st is None:
                    break
                pk = tuple(task["pickup_point"])
                agv = min(free_agvs, key=lambda n: (_manh(pose[n][:2], pk), n))

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
                if int(recent_joint_fail) >= 2:
                    force_promote = {
                        n
                        for n, kind in agv_affinity.items()
                        if kind == "astar" and n in assigned
                    }
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
            # main-copy first: full-fleet spacetime A* (handles same-pad waits)
            st_paths = _prioritized_st_paths(
                starts_g, goals_g, movers_set, blocked_plan
            )
            if st_paths is not None:
                return st_paths

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
                if st2 is not None:
                    return st2
                jc = _wcbs_paths(starts_g, goals_g, me, tl=budget)
                if jc is not None:
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
        loaded = {n: False for n in names}
        dest = {n: "" for n in names}
        tid = {n: "" for n in names}
        for agv, task in pending_delivery.items():
            loaded[agv] = True
            dest[agv] = str(task.get("destination") or "")
            tid[agv] = str(task["task_id"])
            steps_by[agv][-1]["loaded"] = "TRUE"
            steps_by[agv][-1]["destination"] = dest[agv]
            steps_by[agv][-1]["task-id"] = tid[agv]

        pickup_serial_done = True
        paths = None
        if not assigned:
            # Delivery-retry wave only
            pass
        else:
            pickup_serial_done = False
            starts = {n: pose[n][:2] for n in names}
            goals = {n: starts[n] for n in names}
            # Only FIFO heads remain in assigned; go straight to true pad.
            for agv, task in assigned.items():
                goals[agv] = _pickup_goal(task, starts[agv], free, static)

            # Do NOT pre-stage idle AGVs here — synchronizing them to the longest
            # pickup path was inflating sim_time (wave=1 @ ~40 ticks).
            wave_movers = set(assigned)

            paths = _route_wave(starts, goals, wave_movers)
            if paths is None and wave_planner == "astar":
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
            if paths is None and wave_planner == "astar":
                # Do NOT requeue forever (infinite loop). Fall back to joint ECBS
                # for this wave only — still multi-agent parallel, not serial freeze.
                print(
                    f"[ECBS] pickup baseline ST fail k={len(assigned)} "
                    f"-> wave ECBS fallback (keep parallel)",
                    flush=True,
                )
                paths = _wcbs_paths(starts, goals, set(assigned))
            if paths is None:
                # ECBS (or A*→ECBS) still failing: drop farthest agents and retry joint.
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
            if paths is None:
                recent_joint_fail += 1
                print(
                    f"[ECBS] pickup {wave_planner} fail k={len(assigned)} -> serial A*",
                                flush=True,
                            )
                # Maze choke-point recovery: push idle AGVs off pickup corridors.
                keep_cells: Set[Cell] = set()
                for agv, task in assigned.items():
                    keep_cells.add(pose[agv][:2])
                    keep_cells.add(
                        _pickup_goal(task, pose[agv][:2], free, static)
                    )
                ld0 = {n: False for n in names}
                ds0 = {n: "" for n in names}
                td0 = {n: "" for n in names}
                # Preserve cargo state for carriers while dispersing idlers.
                for agv, task in pending_delivery.items():
                    ld0[agv] = True
                    ds0[agv] = str(task.get("destination") or "")
                    td0[agv] = str(task.get("task_id") or "")
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
                if len(assigned) <= 3:
                    g2 = {n: pose[n][:2] for n in names}
                    s2 = {n: pose[n][:2] for n in names}
                    for a, task in assigned.items():
                        g2[a] = _pickup_goal(task, s2[a], free, static)
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
                    if paths is not None and not _cell_paths_conflict(paths, names):
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
                    kept = {}
                    for agv, task in list(assigned.items()):
                        goal = _pickup_goal(task, pose[agv][:2], free, static)
                        now, ok = _serial_move_to(
                                pose,
                            steps_by,
                            now,
                            agv,
                            goal,
                            blocked_plan,
                            loaded=ld0,
                            dest=ds0,
                            tid=td0,
                        )
                        if not ok:
                            # One more disperse+retry for this single mover
                            now = _disperse_idle_agents(
                                pose,
                                    steps_by,
                                now,
                                movers={agv},
                                free=free,
                                blocked_plan=blocked_plan,
                                keep_clear={pose[agv][:2], goal},
                                loaded=ld0,
                                dest=ds0,
                                tid=td0,
                            )
                            now, ok = _serial_move_to(
                                pose,
                                steps_by,
                                now,
                                agv,
                                goal,
                                blocked_plan,
                                loaded=ld0,
                                dest=ds0,
                                tid=td0,
                            )
                        if not ok:
                            _requeue_task(task)
                            continue
                        kept[agv] = task
                    assigned = kept
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
                            # Flush only after many disperse retries (was 5: too eager).
                            if pickup_stall_streak >= 12:
                                dropped = 0
                                stations_flushed = 0
                                for st in list(order):
                                    q = queues.get(st) or []
                                    if not q:
                                        continue
                                    # Drop only the head task first (less destructive).
                                    task = q.pop(0)
                                    tid_drop = str(task.get("task_id") or "")
                                    if tid_drop:
                                        _mark_failed(tid_drop)
                                    dropped += 1
                                    if not q:
                                        del queues[st]
                                    stations_flushed += 1
                                    if stations_flushed >= 1:
                                        break
                                print(
                                    f"[ECBS] pickup stall -> drop-head n={dropped} "
                                    f"failed_n={len(failed)}",
                                    flush=True,
                                )
                            if pickup_stall_streak >= 50 or not any(queues.values()):
                                print(
                                    f"[ECBS] pickup stall abort @t={now} "
                                    f"remaining_q={sum(len(v) for v in queues.values())}",
                                    flush=True,
                                )
                                for st, q in list(queues.items()):
                                    for task in q:
                                        tid_drop = str(task.get("task_id") or "")
                                        if tid_drop:
                                            _mark_failed(tid_drop)
                                    queues[st] = []
                                queues = {k: v for k, v in queues.items() if v}
                                for agv, task in list(pending_delivery.items()):
                                    tid_drop = str(task.get("task_id") or "")
                                    if tid_drop:
                                        _mark_failed(tid_drop)
                                pending_delivery.clear()
                                break
                            continue
                    else:
                        pickup_serial_done = True
                        pickup_stall_streak = 0
            else:
                recent_joint_fail = max(0, recent_joint_fail - 1)
                pickup_serial_done = False
                pickup_stall_streak = 0

        if not assigned and not pending_delivery:
            continue

        # Apply joint pickup paths when we did not already serial-move.
        if assigned and not pickup_serial_done:
            assert paths is not None
            # Shared pads: use trimmed no-hold check (pad_holds=False). The old
            # full-fleet pad_holds=True check false-conflicts sequenced arrivals
            # and collapses into shrink+serial (sim_time blow-up). Idle seeding
            # in `_prioritized_st_paths` keeps this collision-safe.
            if not _st_paths_ok(paths, set(assigned), goals):
                print(
                    "[ECBS] pickup cell conflict -> shrink+serial recover",
                    flush=True,
                )
                ranked = sorted(
                    assigned.items(),
                    key=lambda kv: (
                        _manh(
                            pose[kv[0]][:2],
                            _pickup_goal(kv[1], pose[kv[0]][:2], free, static),
                        ),
                        kv[0],
                    ),
                )
                keep = dict(ranked[: min(3, len(ranked))])
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
                        _pickup_goal(t, pose[a][:2], free, static)
                        for a, t in keep.items()
                    },
                    loaded=ld0,
                    dest=ds0,
                    tid=td0,
                    aggressive=True,
                )
                kept_ok: Dict[str, dict] = {}
                for agv, task in keep.items():
                    goal = _pickup_goal(task, pose[agv][:2], free, static)
                    now, ok = _serial_move_to(
                        pose,
                        steps_by,
                        now,
                        agv,
                        goal,
                        blocked_plan,
                        loaded=ld0,
                        dest=ds0,
                        tid=td0,
                    )
                    if ok:
                        kept_ok[agv] = task
                    else:
                        pending_pickup[agv] = task
                assigned = kept_ok
                pickup_serial_done = True
            else:
                timelines = _plan_paths_as_poses(pose, paths)
                now = _apply_pose_timelines(
                    steps_by, pose, timelines, now, loaded=loaded, dest=dest, tid=tid
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

        # Batch-finish approaches (no per-AGV serial stacking).
        need_pad = [
            agv
            for agv in assigned
            if pose[agv][:2]
            != _pickup_goal(assigned[agv], pose[agv][:2], free, static)
        ]
        if need_pad:
            s2 = {n: pose[n][:2] for n in names}
            g2 = {n: s2[n] for n in names}
            for agv in need_pad:
                g2[agv] = _pickup_goal(assigned[agv], s2[agv], free, static)
            sp = _prioritized_st_paths(s2, g2, set(need_pad), blocked_plan)
            if _st_paths_ok(sp, set(need_pad), g2):
                timelines = _plan_paths_as_poses(pose, sp)
                if not _pose_timelines_conflict(pose, timelines):
                    now = _apply_pose_timelines(
                    steps_by,
                    pose,
                        timelines,
                        now,
                        loaded=loaded,
                        dest=dest,
                        tid=tid,
                        )

        for agv in sorted(assigned.keys(), key=_inflight_rank):
            task = assigned[agv]
            true_pk = _pickup_goal(task, pose[agv][:2], free, static)
            st_name = _task_station(task)
            tid_s = str(task.get("task_id") or "")
            if not _is_station_head(task) or st_name in committed_stations:
                pending_pickup[agv] = task
                continue
            if pose[agv][:2] != true_pk:
                # Single leftover only — avoid serial chain across the fleet.
                now, ok = _serial_move_to(
                                    pose,
                    steps_by,
                    now,
                    agv,
                    true_pk,
                    blocked_plan,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                )
            if pose[agv][:2] != true_pk or not _is_station_head(task):
                pending_pickup[agv] = task
                continue
            tid[agv] = tid_s
            loaded[agv] = True
            dest[agv] = str(task.get("destination") or "")
            steps_by[agv][-1]["loaded"] = "TRUE"
            steps_by[agv][-1]["destination"] = dest[agv]
            steps_by[agv][-1]["task-id"] = tid[agv]
            if tid[agv]:
                picked_ids.add(tid[agv])
                q_inf = inflight_station.get(st_name) or []
                if q_inf and q_inf[0] == tid[agv]:
                    q_inf.pop(0)
                    if not q_inf:
                        inflight_station.pop(st_name, None)
                    else:
                        inflight_station[st_name] = q_inf
            work_count[agv] = int(work_count.get(agv, 0)) + 1
            committed[agv] = task
            committed_stations.add(st_name)
        assigned = committed

        if not assigned and not pending_delivery:
            if pending_pickup:
                continue
            continue

        # ---- delivery ----
        starts = {n: pose[n][:2] for n in names}
        goals = {n: starts[n] for n in names}
        deliverable: Dict[str, dict] = {}
        reserved_goals: Set[Cell] = set()
        unique_movers: Set[str] = set()
        delivery_agents = dict(pending_delivery)
        delivery_agents.update(assigned)
        for agv, task in sorted(delivery_agents.items()):
            # Unique pad per joint wave; NEVER use maze-wall end_points as goals.
            walk = _valid_unload_pads(task, free, static, prefer=starts[agv])
            if not walk:
                _mark_failed(str(task["task_id"]))
                print(
                    f"[ECBS] no free unload pad for {task['task_id']} -> fail",
                    flush=True,
                    )
                pending_delivery.pop(agv, None)
                _clear_affinity(agv)
                loaded[agv] = False
                steps_by[agv][-1]["loaded"] = "FALSE"
                steps_by[agv][-1]["destination"] = ""
                steps_by[agv][-1]["task-id"] = ""
                continue
            cands = [c for c in walk if c not in reserved_goals]
            if not cands:
                # Expand to free neighbors of pads (staging cells) — map-agnostic.
                seen_e: Set[Cell] = set()
                for p in walk:
                    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                        c = (p[0] + dx, p[1] + dy)
                        if (
                            c in free
                            and c not in static
                            and c not in reserved_goals
                            and c not in seen_e
                        ):
                            cands.append(c)
                            seen_e.add(c)
            if cands:
                goals[agv] = min(cands, key=lambda c: (_manh(starts[agv], c), c))
                reserved_goals.add(goals[agv])
                unique_movers.add(agv)
            else:
                # Last resort: share a pad (ST will time-multiplex).
                goals[agv] = min(walk, key=lambda c: (_manh(starts[agv], c), c))
            deliverable[agv] = task

        if not deliverable:
            print(f"[ECBS] no deliverable goals @t={now}", flush=True)

        # Carriers must stay loaded through every delivery apply/hold.
        for agv in list(pending_delivery.keys()) + list(deliverable.keys()):
            loaded[agv] = True
            if tid.get(agv):
                steps_by[agv][-1]["loaded"] = "TRUE"
                steps_by[agv][-1]["destination"] = dest.get(agv, "")
                steps_by[agv][-1]["task-id"] = tid.get(agv, "")

        def _clear_after_unload(agv: str) -> None:
            return

        def _unload_arrivals(movers: Set[str]) -> None:
            for agv in sorted(movers):
                if agv not in deliverable:
                    continue
                ok_pads = set(_valid_unload_pads(deliverable[agv], free, static))
                ok_pads.add(goals[agv])
                if pose[agv][:2] not in ok_pads:
                    continue
                loaded[agv] = False
                dest[agv] = ""
                tid[agv] = ""
                steps_by[agv][-1]["loaded"] = "FALSE"
                steps_by[agv][-1]["destination"] = ""
                steps_by[agv][-1]["task-id"] = ""
                _count_done(str(deliverable[agv]["task_id"]))
                pending_delivery.pop(agv, None)
                _clear_affinity(agv)
                _clear_after_unload(agv)

        # Prefer unique-pad movers for the first joint wave; leftovers recover via ST.
        active = set(unique_movers) if unique_movers else set(deliverable.keys())
        paths = None
        joint_applied = False
        if not deliverable:
            joint_applied = True  # skip joint/serial delivery body
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
        if last_mile and active:
            keep_lm: Set[Cell] = set()
            for a in active:
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
        if active and not last_mile:
            gmap = {n: starts[n] for n in names}
            for a in active:
                gmap[a] = goals[a]
            paths = _route_wave(starts, gmap, active)
        elif active and last_mile:
            # Skip joint; fall through to per-agent serial recover.
            paths = None
        if paths is not None:
            timelines = _plan_paths_as_poses(pose, paths)
            if not _st_paths_ok(paths, active, gmap):
                if wave_planner == "astar":
                    print(
                        "[ECBS] delivery baseline cell conflict -> replan ST",
                        flush=True,
                    )
                else:
                    print(
                        f"[ECBS] delivery {wave_planner} cell conflict -> ST",
                        flush=True,
                    )
                paths = _prioritized_st_paths(
                    starts, gmap, active, blocked_plan
                )
                if not _st_paths_ok(paths, active, gmap):
                    paths = None
                if paths is not None:
                    timelines = _plan_paths_as_poses(pose, paths)
                    now = _apply_pose_timelines(
                        steps_by,
                        pose,
                        timelines,
                        now,
                        loaded=loaded,
                        dest=dest,
                        tid=tid,
                    )
                    joint_applied = True
                    _unload_arrivals(active)
            else:
                now = _apply_pose_timelines(
                    steps_by, pose, timelines, now, loaded=loaded, dest=dest, tid=tid
                )
                joint_applied = True
                _unload_arrivals(active)

        if not joint_applied:
            print(
                f"[ECBS] delivery joint fail/skip k={len(deliverable)} "
                f"unique={len(unique_movers)} -> per-agent",
                flush=True,
            )
            recent_joint_fail += 1

        need = sorted(a for a in deliverable if loaded.get(a, False))
        if need and not joint_applied:
            # Parallel ST before any per-agent serial (all planners).
            starts_b = {n: pose[n][:2] for n in names}
            goals_b = {n: starts_b[n] for n in names}
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
            movers_b = {a for a in need if loaded.get(a, False)}
            if movers_b:
                sp = _prioritized_st_paths(
                    starts_b, goals_b, movers_b, blocked_plan
                )
                if _st_paths_ok(sp, movers_b, goals_b):
                    timelines = _plan_paths_as_poses(pose, sp)
                    if not _pose_timelines_conflict(pose, timelines):
                        now = _apply_pose_timelines(
                            steps_by,
                            pose,
                            timelines,
                            now,
                            loaded=loaded,
                            dest=dest,
                            tid=tid,
                        )
                        for agv in list(movers_b):
                            pads = set(
                                _valid_unload_pads(deliverable[agv], free, static)
                            )
                            pads.add(goals_b[agv])
                            if pose[agv][:2] in pads:
                                loaded[agv] = False
                                dest[agv] = ""
                                tid[agv] = ""
                                steps_by[agv][-1]["loaded"] = "FALSE"
                                steps_by[agv][-1]["destination"] = ""
                                steps_by[agv][-1]["task-id"] = ""
                                _count_done(str(deliverable[agv]["task_id"]))
                                pending_delivery.pop(agv, None)
                                _clear_after_unload(agv)
                        joint_applied = True
                        print(
                            f"[ECBS] delivery ST parallel k={len(movers_b)}",
                            flush=True,
                            )
            need = sorted(a for a in deliverable if loaded.get(a, False))

        if need:
            if joint_applied:
                print(f"[ECBS] delivery leftover k={len(need)}", flush=True)
            # Prefer one more parallel ST for leftovers, else serial.
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
                        if not _pose_timelines_conflict(pose, timelines):
                            now = _apply_pose_timelines(
                            steps_by,
                            pose,
                                timelines,
                                now,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                                )
                            for agv in list(ok_need):
                                pads = set(
                                    _valid_unload_pads(
                                    deliverable[agv], free, static
                                    )
                                )
                                pads.add(goals_b[agv])
                                if pose[agv][:2] in pads:
                                    loaded[agv] = False
                                    dest[agv] = ""
                                    tid[agv] = ""
                                    steps_by[agv][-1]["loaded"] = "FALSE"
                                    steps_by[agv][-1]["destination"] = ""
                                    steps_by[agv][-1]["task-id"] = ""
                                    _count_done(str(deliverable[agv]["task_id"]))
                                    pending_delivery.pop(agv, None)
                                    _clear_after_unload(agv)
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
                        if not _pose_timelines_conflict(pose, timelines):
                            now = _apply_pose_timelines(
                            steps_by,
                            pose,
                                timelines,
                                now,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                                )
                            for a in list(need):
                                pads = set(
                                    _valid_unload_pads(deliverable[a], free, static)
                                )
                                pads.add(goals_r[a])
                                if pose[a][:2] in pads:
                                    loaded[a] = False
                                    dest[a] = ""
                                    tid[a] = ""
                                    steps_by[a][-1]["loaded"] = "FALSE"
                                    steps_by[a][-1]["destination"] = ""
                                    steps_by[a][-1]["task-id"] = ""
                                    _count_done(str(deliverable[a]["task_id"]))
                                    pending_delivery.pop(a, None)
                                    _clear_affinity(a)
                                    _clear_after_unload(a)
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
                        int(recent_joint_fail) >= 6
                        or int(no_progress_waves) >= 2
                    )
                    batch = ranked[:1] if jam_hard else list(ranked)
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
                    sp_p = _prioritized_st_paths(
                        starts_p, goals_p, movers_p, blocked_plan
                    )
                    if _st_paths_ok(sp_p, movers_p, goals_p):
                        timelines = _plan_paths_as_poses(pose, sp_p)
                        if not _pose_timelines_conflict(pose, timelines):
                            now = _apply_pose_timelines(
                                steps_by,
                                pose,
                                timelines,
                                now,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                            )
                            for a in list(movers_p):
                                pads = set(
                                    _valid_unload_pads(deliverable[a], free, static)
                                )
                                pads.add(goals_p[a])
                                if pose[a][:2] in pads:
                                    loaded[a] = False
                                    dest[a] = ""
                                    tid[a] = ""
                                    steps_by[a][-1]["loaded"] = "FALSE"
                                    steps_by[a][-1]["destination"] = ""
                                    steps_by[a][-1]["task-id"] = ""
                                    _count_done(str(deliverable[a]["task_id"]))
                                    pending_delivery.pop(a, None)
                                    _clear_affinity(a)
                                    _clear_after_unload(a)
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f"[ECBS] delivery batch-ST k={len(movers_p)} "
                                f"left={len(need)} jam={int(jam_hard)} @t={now}",
                                flush=True,
                            )
                        else:
                            need = batch
                            print(
                                f"[ECBS] delivery batch-ST pose-conflict "
                                f"k={len(batch)} -> serial @t={now}",
                                flush=True,
                            )
                    else:
                        need = batch
                        print(
                            f"[ECBS] delivery batch-ST fail k={len(batch)} "
                            f"jam={int(jam_hard)} -> serial @t={now}",
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
                        _mark_failed(str(deliverable[agv]["task_id"]))
                        pending_delivery.pop(agv, None)
                        _clear_affinity(agv)
                        loaded[agv] = False
                        steps_by[agv][-1]["loaded"] = "FALSE"
                        steps_by[agv][-1]["destination"] = ""
                        steps_by[agv][-1]["task-id"] = ""
                        continue
                    target = goals[agv] if goals.get(agv) in pads else pads[0]
                    if target not in pads:
                        target = min(pads, key=lambda c: (_manh(pose[agv][:2], c), c))
                    now, ok = _serial_move_to(
                    pose,
                            steps_by,
                            now,
                        agv,
                        target,
                        blocked_plan,
                            loaded=loaded,
                            dest=dest,
                            tid=tid,
                        )
                    arrived = bool(ok) and (
                        pose[agv][:2] in pads or pose[agv][:2] == target
                    )
                    if not arrived:
                        for tgt in sorted(
                            pads, key=lambda c: (_manh(pose[agv][:2], c), c)
                        ):
                            now, ok2 = _serial_move_to(
                                pose,
                                steps_by,
                                now,
                                agv,
                                tgt,
                                blocked_plan,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                            )
                            if ok2 and (
                                pose[agv][:2] in pads or pose[agv][:2] == tgt
                            ):
                                arrived = True
                                break
                    if arrived:
                        loaded[agv] = False
                        dest[agv] = ""
                        tid[agv] = ""
                        steps_by[agv][-1]["loaded"] = "FALSE"
                        steps_by[agv][-1]["destination"] = ""
                        steps_by[agv][-1]["task-id"] = ""
                        _count_done(str(deliverable[agv]["task_id"]))
                        pending_delivery.pop(agv, None)
                        _clear_affinity(agv)
                        _clear_after_unload(agv)
                    else:
                        tid_miss = str(deliverable[agv]["task_id"])
                        delivery_miss_count[tid_miss] = int(
                            delivery_miss_count.get(tid_miss, 0)
                        ) + 1
                        # Near end-of-run: allow more retries before hard-fail.
                        miss_cap = 24 if (total - done) <= 2 else 8
                        if delivery_miss_count[tid_miss] <= miss_cap:
                            pending_delivery[agv] = deliverable[agv]
                            loaded[agv] = True
                            dest[agv] = str(
                                deliverable[agv].get("destination") or ""
                            )
                            tid[agv] = tid_miss
                            steps_by[agv][-1]["loaded"] = "TRUE"
                            steps_by[agv][-1]["destination"] = dest[agv]
                            steps_by[agv][-1]["task-id"] = tid_miss
                            recent_joint_fail += 1
                        else:
                            _mark_failed(tid_miss)
                            pending_delivery.pop(agv, None)
                            _clear_affinity(agv)
                            loaded[agv] = False
                            steps_by[agv][-1]["loaded"] = "FALSE"
                            steps_by[agv][-1]["destination"] = ""
                            steps_by[agv][-1]["task-id"] = ""

        eff_core = _effective_joint_core()
        if eff_core != joint_core_mode and recent_joint_fail >= 6:
            print(
                f"[HIER] joint_core escalate {joint_core_mode}->{eff_core} "
                f"fail={recent_joint_fail}",
                flush=True,
            )
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
        if now == stall_t and done == stall_done:
            no_progress_waves += 1
        else:
            no_progress_waves = 0
            stall_t, stall_done = now, done

        if no_progress_waves >= 2 and (
            pending_delivery or pending_pickup or any(queues.values())
        ):
            print(
                f"[RECOVER] no-progress waves={no_progress_waves} "
                f"t={now} done={done} carry={len(pending_delivery)}",
                flush=True,
            )
            if joint_core_mode == "prioritized":
                joint_core_override = "ecbs"
            _force_wait_tick("no_progress")
            # Force serial unload: try all carriers one-by-one (hub jam needs >2).
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
                for tgt in sorted(pads, key=lambda c: (_manh(pose[agv][:2], c), c)):
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
                    _count_done(str(task.get("task_id") or ""))
                    pending_delivery.pop(agv, None)
                    _clear_affinity(agv)
                    # Clear loaded flags for trajectory consistency
                    if steps_by.get(agv):
                        steps_by[agv][-1]["loaded"] = "FALSE"
                        steps_by[agv][-1]["destination"] = ""
                        steps_by[agv][-1]["task-id"] = ""
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
            else:
                # Keep pressure; do not wipe streak on failed recover.
                no_progress_waves = max(2, int(no_progress_waves))
                print(
                    f"[RECOVER] force-unload failed carry={len(pending_delivery)} "
                    f"@t={now}",
                    flush=True,
                )

        # Drop sooner when force-unload keeps failing (hub livelock).
        drop_thr = 4 if int(recent_joint_fail) >= 8 else (20 if (total - done) <= 2 else 10)
        if no_progress_waves >= drop_thr and pending_delivery and (total - done) > 2:
            # Last resort: drop one stuck carrier to unblock fleet
            agv = sorted(pending_delivery.keys())[0]
            task = pending_delivery.pop(agv)
            _mark_failed(str(task.get("task_id") or ""))
            _clear_affinity(agv)
            print(
                f"[RECOVER] drop stuck carrier {agv} "
                f"task={task.get('task_id')} @t={now}",
                flush=True,
            )
            no_progress_waves = 0
            _force_wait_tick("drop_carrier")
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

    sid = str(meta["id"])
    swap_tag = "hier" if use_hierarchical else "base"
    traj = TRAJ / f"{sid}_ecbs_{plan}_w{weight}_k{max_active}_{swap_tag}.csv"
    rows = []
    for n in sorted(steps_by):
        rows.extend(steps_by[n])
    rows.sort(key=lambda r: (int(r["timestamp"]), r["name"]))
    with traj.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRAJ_HEADER})

    # Pass all failed ids; analyze_fifo only skips those never actually picked.
    val = validate_hybrid_trajectory(meta, traj, skip_task_ids=set(failed_seen))
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
    )
    ok = bool(rep.get("validate_ok")) and float(rep.get("completion_ratio") or 0) >= 0.999
    return 0 if ok and not rep.get("tasks_failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
