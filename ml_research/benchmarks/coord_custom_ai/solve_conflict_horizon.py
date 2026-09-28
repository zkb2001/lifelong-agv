"""Conflict-horizon staged MAPD (optimistic solo A* + priority yield).

Each **wave** is one replan cycle; **sim_t** (`now`) is simulation seconds.
Planning is **event-driven on stage need**: if an AGV finishes pickup/drop or
gets a new assignment and thus lacks a next-stage path, the current commit is
aborted and the next wave replans immediately (no long idle wait inside a
bulk commit). Motion follows competition rules: **turn-in-place = 1s**, then
**move = 1s**. When paths do not conflict, commit until the next stage event
or end of plans; on conflict only advance to T*-1 then yield/resume.

Offline windowed solver for SH_custom_*; writes trajectory + validation JSON.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ml_research.benchmarks.common import TRAJ_HEADER, load_scenario, patch_extra_obstacles
from ml_research.benchmarks.coord_custom_ai.llm_coord import DEFAULT_MODEL, LlmCoordinator
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    format_validation_summary,
    validate_hybrid_trajectory,
)
from ml_research.common.paths import RESULTS

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]
OUT = RESULTS / "coord_custom_ai" / "conflict_horizon"
TRAJ = OUT / "trajectories"


@dataclass(frozen=True)
class CHTuning:
    """Tunable heuristic knobs (offline Optuna / CLI overrides)."""

    recent_parks_k: int = 6
    idle_path_cap: int = 20
    dispatch_lambda: int = 1
    staging_max_d: int = 16
    # 0 = never freeze idle AGVs (max-effort staging)
    idle_hold_dist: int = 0

    @classmethod
    def from_dict(cls, raw: dict) -> CHTuning:
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: int(v) for k, v in raw.items() if k in fields})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


DEFAULT_CH_TUNING = CHTuning()


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


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
        "pitch": pose[2],
        "loaded": "TRUE" if loaded else "FALSE",
        "destination": dest,
        "Emergency": "FALSE",
        "task-id": tid,
    }


def _neighbors(c: Cell):
    x, y = c
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 1 <= nx <= 20 and 1 <= ny <= 20:
            yield (nx, ny)


def _parking_bays(static: Set[Cell], stations: Set[Cell]) -> List[Cell]:
    free = {
        (x, y)
        for x in range(1, 21)
        for y in range(1, 21)
        if (x, y) not in static and (x, y) not in stations
    }
    deg = {c: sum(1 for n in _neighbors(c) if n in free) for c in free}
    # Prefer low-degree / corner cells as yield parking
    return sorted(free, key=lambda c: (deg[c], -(c[0] + c[1]), -c[0]))


def _side_park_candidates(parks: List[Cell]) -> List[Cell]:
    """Perimeter / low-traffic parks for idle tier-2 (O4)."""
    side = [p for p in parks if p[0] in (1, 20) or p[1] in (1, 20)]
    return side if side else list(parks)


def _nearest_side_park(
    pos: Cell, parks: List[Cell], occupied: Set[Cell]
) -> Cell:
    for park in sorted(
        _side_park_candidates(parks),
        key=lambda p: (_manh(pos, p), p[0] in (1, 20), p),
    ):
        if park not in occupied and park != pos:
            return park
    for park in sorted(parks, key=lambda p: (_manh(pos, p), p)):
        if park not in occupied and park != pos:
            return park
    return pos


def _pick_staging_agvs(
    names: List[str],
    active: Dict[str, dict],
    assign_count: Dict[str, int],
    max_active: int,
    pose: Dict[str, Pose],
    queues: Dict[str, list],
    order: List[str],
    *,
    staging_max_d: int,
) -> Set[str]:
    """Idle AGVs allowed toward pickup staging (O4): ≤open slots, low assign_count."""
    free = [n for n in names if n not in active]
    slots = max(1, max_active - len(active))
    ranked = sorted(free, key=lambda n: (assign_count[n], n))
    out: Set[str] = set()
    for n in ranked[:slots]:
        pk = _nearest_queued_pickup(queues, order, pose[n][:2])
        if pk is None:
            continue
        if _manh(pose[n][:2], pk) <= staging_max_d:
            out.add(n)
    return out


def _pickup_corridor_load(
    pk: Cell,
    active: Dict[str, dict],
    pose: Dict[str, Pose],
) -> int:
    """Active AGVs already near this pickup (O7)."""
    load = 0
    for agv, info in active.items():
        if tuple(info.get("pickup") or ()) == pk:
            load += 3
        elif _manh(pose[agv][:2], pk) <= 4:
            load += 1
    return load


def _solo_task_cost(pos: Cell, task: dict, static: Set[Cell]) -> int:
    """A* cost for this AGV alone: pos→pickup + pickup→nearest unload."""
    pk = tuple(task["pickup_point"])
    c_pick = _astar_len(pos, pk, static, allow={pk})
    if c_pick >= 10**6:
        return 10**6
    ends = [
        tuple(e)
        for e in (task.get("end_points") or [])
        if tuple(e) not in static
    ]
    if not ends:
        ends = [tuple(e) for e in (task.get("end_points") or [])]
    if not ends:
        return c_pick
    allow = set(ends) | {pk}
    c_drop = min(_astar_len(pk, e, static, allow=allow) for e in ends)
    if c_drop >= 10**6:
        return 10**6
    return int(c_pick) + int(c_drop)


def _stations_with_inflight_pickup(active: Dict[str, dict]) -> Set[str]:
    """Stations that already have an AGV en-route to pickup (FIFO lock)."""
    busy: Set[str] = set()
    for info in active.values():
        if info.get("leg") != "to_pickup":
            continue
        st = str(info.get("station") or "")
        if not st:
            task = info.get("task") or {}
            st = str(
                task.get("start_point")
                or task.get("pickup_name")
                or task.get("station")
                or ""
            )
        if st:
            busy.add(st)
    return busy


def _best_queued_task_for_agv(
    pos: Cell,
    queues: Dict[str, list],
    order: List[str],
    static: Set[Cell],
    *,
    active: Optional[Dict[str, dict]] = None,
) -> Optional[Tuple[str, int, dict, int]]:
    """Pick FIFO head of some free station with min solo A*.

    A station is free only if its queue head is eligible and no other AGV is
    already en-route to pick from that station (prevents Tiger-3 before Tiger-2).
    """
    busy = _stations_with_inflight_pickup(active or {})
    best: Optional[Tuple[Tuple, str, int, dict, int]] = None
    for st in order:
        if st in busy:
            continue
        q = queues.get(st) or []
        if not q:
            continue
        # FIFO: only the station queue head is eligible
        i, task = 0, q[0]
        cost = _solo_task_cost(pos, task, static)
        urgent = 0 if str(task.get("priority") or "") == "Urgent" else 1
        key = (cost, urgent, st, str(task.get("task_id") or ""))
        if best is None or key < best[0]:
            best = (key, st, i, task, cost)
    if best is None:
        return None
    _, st, i, task, cost = best
    return st, i, task, cost


def _bind_task_to_agv(
    agv: str,
    task: dict,
    *,
    active: Dict[str, dict],
    static: Set[Cell],
    tid: Dict[str, str],
    dest: Dict[str, str],
    loaded: Dict[str, bool],
    assign_count: Dict[str, int],
    station: str = "",
) -> None:
    pk = tuple(task["pickup_point"])
    st = str(
        station
        or task.get("start_point")
        or task.get("pickup_name")
        or ""
    )
    assign_count[agv] += 1
    active[agv] = {
        "task": task,
        "leg": "to_pickup",
        "pickup": pk,
        "station": st,
        "ends": [
            tuple(e)
            for e in (task.get("end_points") or [])
            if tuple(e) not in static
        ],
        "urgent": str(task.get("priority") or "") == "Urgent",
    }
    if not active[agv]["ends"]:
        active[agv]["ends"] = [tuple(e) for e in (task.get("end_points") or [])]
    tid[agv] = str(task["task_id"])
    dest[agv] = str(task.get("destination") or "")
    loaded[agv] = False


def _assign_agv_by_astar(
    agv: str,
    *,
    pose: Dict[str, Pose],
    queues: Dict[str, list],
    order: List[str],
    static: Set[Cell],
    active: Dict[str, dict],
    max_active: int,
    tid: Dict[str, str],
    dest: Dict[str, str],
    loaded: Dict[str, bool],
    assign_count: Dict[str, int],
    now: int,
) -> bool:
    """Immediate dispatch: free AGV gets min solo-A* queued task (ignore other AGVs)."""
    if agv in active or len(active) >= max_active or not any(queues.values()):
        return False
    hit = _best_queued_task_for_agv(
        pose[agv][:2], queues, order, static, active=active
    )
    if hit is None:
        return False
    st, idx, task, cost = hit
    if cost >= 10**6:
        return False
    queues[st].pop(idx)
    if st in queues and not queues[st]:
        del queues[st]
        order[:] = [s for s in order if s in queues]
    _bind_task_to_agv(
        agv,
        task,
        active=active,
        static=static,
        tid=tid,
        dest=dest,
        loaded=loaded,
        assign_count=assign_count,
        station=st,
    )
    print(
        f"[CH] assign {task['task_id']} -> {agv} costA*={cost} t={now}",
        flush=True,
    )
    return True


def _fill_free_agvs_by_astar(
    names: List[str],
    *,
    pose: Dict[str, Pose],
    queues: Dict[str, list],
    order: List[str],
    static: Set[Cell],
    active: Dict[str, dict],
    max_active: int,
    tid: Dict[str, str],
    dest: Dict[str, str],
    loaded: Dict[str, bool],
    assign_count: Dict[str, int],
    now: int,
) -> int:
    """Assign every free AGV (under max_active) its current best solo-A* task."""
    n_new = 0
    free = sorted(
        [n for n in names if n not in active],
        key=lambda n: (assign_count[n], n),
    )
    for agv in free:
        if len(active) >= max_active:
            break
        if _assign_agv_by_astar(
            agv,
            pose=pose,
            queues=queues,
            order=order,
            static=static,
            active=active,
            max_active=max_active,
            tid=tid,
            dest=dest,
            loaded=loaded,
            assign_count=assign_count,
            now=now,
        ):
            n_new += 1
    return n_new


def _best_solo_path(
    start: Pose,
    goal: Cell,
    static: Set[Cell],
    *,
    allow: Optional[Set[Cell]] = None,
    extra_block: Optional[Set[Cell]] = None,
) -> Optional[List[Cell]]:
    """xy vs turn-cost BFS: pick shorter expanded timeline (O3)."""
    allow_set = set(allow or set()) | {goal, (start[0], start[1])}
    xy = _safe_path(
        (start[0], start[1]),
        goal,
        static,
        allow=allow,
        extra_block=extra_block,
    )
    blocked = (static | set(extra_block or set())) - allow_set
    turn = _turn_cost_bfs(start, goal, blocked, allow=allow_set)
    if not xy and not turn:
        return None
    if not xy:
        return turn
    if not turn:
        return xy
    if len(_expand_full_timeline(start, turn)) < len(
        _expand_full_timeline(start, xy)
    ):
        return turn
    return xy


def _prefix_to_cell(cells: List[Cell], target: Cell) -> List[Cell]:
    out: List[Cell] = []
    for c in cells:
        out.append(c)
        if c == target:
            break
    return out


def _bfs_static(start: Cell, goal: Cell, blocked: Set[Cell]) -> List[Cell]:
    """Unit-step path on static grid (ignore other AGVs)."""
    if start == goal:
        return [start]
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    q: deque = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in _neighbors(cur):
            if nxt in parent:
                continue
            if nxt in blocked and nxt != goal:
                continue
            parent[nxt] = cur
            q.append(nxt)
    if goal not in parent:
        return []
    rev: List[Cell] = []
    cur: Optional[Cell] = goal
    while cur is not None:
        rev.append(cur)
        cur = parent[cur]
    rev.reverse()
    return rev


def _path_hits_walls(cells: List[Cell], static: Set[Cell], allow: Set[Cell]) -> bool:
    for c in cells:
        if c in static and c not in allow:
            return True
    return False


def _safe_path(
    start: Cell,
    goal: Cell,
    static: Set[Cell],
    *,
    allow: Optional[Set[Cell]] = None,
    extra_block: Optional[Set[Cell]] = None,
) -> Optional[List[Cell]]:
    """BFS that never enters static walls (except allowed goals).

    ``extra_block`` (other AGVs) always wins over ``allow`` — never path onto
    an occupied cell, even if that cell is a station goal.
    """
    extra = set(extra_block or set()) - {start}
    # Station allow may open walls, but never clears live AGV cells.
    allow_set = (set(allow or set()) | {goal, start}) - extra
    if goal in extra:
        return None
    blocked = (set(static) | extra) - allow_set
    path = _bfs_static(start, goal, blocked)
    if not path:
        return None
    if _path_hits_walls(path, static, allow_set | {goal, start}):
        return None
    if any(c in extra for c in path[1:]):
        return None
    return path


def _pitch_to_delta(pitch: int) -> Tuple[int, int]:
    p = int(pitch) % 360
    if p == 0:
        return (1, 0)
    if p == 180:
        return (-1, 0)
    if p == 90:
        return (0, 1)
    if p == 270:
        return (0, -1)
    return (0, 0)


def _turn_cost_bfs(
    start: Pose,
    goal: Cell,
    blocked: Set[Cell],
    *,
    allow: Optional[Set[Cell]] = None,
) -> List[Cell]:
    """Shortest turn+move path (§2); returns xy cells from start to goal."""
    sx, sy, sp = int(start[0]), int(start[1]), int(start[2]) % 360
    gx, gy = int(goal[0]), int(goal[1])
    allow_set = set(allow or set()) | {(sx, sy), (gx, gy)}
    if (sx, sy) == (gx, gy):
        return [(sx, sy)]
    start_st: Pose = (sx, sy, sp)
    parent: Dict[Pose, Optional[Pose]] = {start_st: None}
    moved: Dict[Pose, bool] = {start_st: False}
    q: deque = deque([start_st])
    goal_st: Optional[Pose] = None
    while q:
        cur = q.popleft()
        x, y, pitch = cur
        if (x, y) == (gx, gy):
            goal_st = cur
            break
        for tp in (0, 90, 180, 270):
            if tp == pitch:
                continue
            ns: Pose = (x, y, tp)
            if ns not in parent:
                parent[ns] = cur
                moved[ns] = False
                q.append(ns)
        dx, dy = _pitch_to_delta(pitch)
        nx, ny = x + dx, y + dy
        if not (1 <= nx <= 20 and 1 <= ny <= 20):
            continue
        nc = (nx, ny)
        if nc in blocked and nc not in allow_set:
            continue
        ns = (nx, ny, pitch)
        if ns not in parent:
            parent[ns] = cur
            moved[ns] = True
            q.append(ns)
    if goal_st is None:
        return []
    chain: List[Pose] = []
    cur_p: Optional[Pose] = goal_st
    while cur_p is not None:
        chain.append(cur_p)
        cur_p = parent[cur_p]
    chain.reverse()
    cells: List[Cell] = [(sx, sy)]
    for st in chain[1:]:
        if moved.get(st, False):
            cells.append((st[0], st[1]))
    return cells


def _end_pose_after_cells(start: Pose, cells: List[Cell]) -> Pose:
    tl = _expand_full_timeline(start, cells)
    return tl[-1] if tl else start


def _tail_wait_streak(steps: List[dict]) -> int:
    """Consecutive hold seconds at trajectory tail (same x,y,pitch)."""
    if len(steps) < 2:
        return 0
    streak = 0
    for i in range(len(steps) - 1, 0, -1):
        a, b = steps[i - 1], steps[i]
        if (
            a["X"] == b["X"]
            and a["Y"] == b["Y"]
            and a["pitch"] == b["pitch"]
        ):
            streak += 1
        else:
            break
    return streak


def _sync_wait_streaks(
    steps_by: Dict[str, List[dict]], names: List[str], streak: Dict[str, int]
) -> None:
    for n in names:
        streak[n] = _tail_wait_streak(steps_by[n])


def _note_park(
    recent: Dict[str, deque], agv: str, cell: Cell, *, maxlen: int
) -> None:
    dq = recent.setdefault(agv, deque(maxlen=maxlen))
    if cell not in dq:
        dq.append(cell)


def _park_taboo(recent: Dict[str, deque], agv: str) -> Set[Cell]:
    return set(recent.get(agv, ()))


def _solo_path_engine(
    mod,
    assigned: dict,
    start: Pose,
    pickup: Cell,
    ends: List[Cell],
    static_list: list,
    *,
    now: int,
    wall: float = 12.0,
) -> Optional[List[Cell]]:
    """Full pickup→delivery as if alone; reject wall-piercing engine paths."""
    static = {(int(p[0]), int(p[1])) for p in static_list}
    allow = {pickup} | set(ends)
    agv_start = (start[0], start[1], now, start[2])
    path, _steps = mod.A_Star(
        {
            "agv": assigned["agv"],
            "task_id": str(assigned["task_id"]),
            "destination": assigned.get("destination") or "",
            "priority": assigned.get("priority") or "Normal",
            "agv_start_point": agv_start,
            "pickup_point": pickup,
            "end_points": ends,
        },
        agv_start,
        pickup,
        ends,
        static_list,
        {},
        wall_budget_s=wall,
        max_path_len=600,
        max_visited=200000,
    )
    if path:
        cells = [(int(p[0]), int(p[1])) for p in path]
        if not cells or cells[0] != (start[0], start[1]):
            cells = [(start[0], start[1])] + cells
        ok = _sanitize_cells(cells, static, allow | {(start[0], start[1])})
        if ok:
            return ok
    # Wall-safe BFS fallback; turn-cost when xy unreachable (O3)
    to_pk = _safe_path((start[0], start[1]), pickup, static, allow=allow)
    if not to_pk:
        allow_set = set(allow) | {(start[0], start[1]), pickup}
        to_pk = _turn_cost_bfs(start, pickup, static - allow_set, allow=allow_set)
    if not to_pk:
        return None
    if not ends:
        return to_pk
    goal_u = min(ends, key=lambda e: (_manh(pickup, e), e))
    to_drop = _safe_path(pickup, goal_u, static, allow=allow)
    if not to_drop:
        end_pose = _end_pose_after_cells(start, to_pk)
        allow_set = set(allow) | {(start[0], start[1]), pickup, goal_u}
        to_drop = _turn_cost_bfs(
            end_pose, goal_u, static - allow_set, allow=allow_set
        )
    if not to_drop:
        return to_pk
    return to_pk + to_drop[1:]


def _solo_to_goal(
    mod,
    name: str,
    start: Pose,
    goal: Cell,
    static_list: list,
    *,
    now: int,
    tid: str = "",
    dest: str = "",
    wall: float = 12.0,
    extra_block: Optional[Set[Cell]] = None,
    allow: Optional[Set[Cell]] = None,
) -> Optional[List[Cell]]:
    """Solo path via xy BFS; turn-cost if xy miss (O3 fallback)."""
    del mod, now, tid, dest, wall, name
    static = {(int(p[0]), int(p[1])) for p in static_list}
    path = _safe_path(
        (start[0], start[1]),
        goal,
        static,
        allow=allow,
        extra_block=extra_block,
    )
    if path:
        return path
    allow_set = set(allow or set()) | {goal, (start[0], start[1])}
    blocked = (static | set(extra_block or set())) - allow_set
    turn = _turn_cost_bfs(start, goal, blocked, allow=allow_set)
    if turn and not _path_hits_walls(turn, static, allow_set):
        return turn
    return None


def _sanitize_cells(
    cells: Optional[List[Cell]],
    static: Set[Cell],
    allow: Set[Cell],
) -> Optional[List[Cell]]:
    if not cells:
        return None
    if _path_hits_walls(cells, static, allow):
        return None
    return cells


def _first_conflict(
    traj: Dict[str, List[Cell]],
) -> Optional[Tuple[int, str, str, str]]:
    """Return (t, a, b, kind) for first vertex collision or swap. t is index in traj."""
    if len(traj) < 2:
        return None
    names = sorted(traj)
    T = max(len(traj[n]) for n in names)
    # pad conceptually with last cell
    def cell_at(n: str, t: int) -> Cell:
        p = traj[n]
        if not p:
            return (0, 0)
        return p[t] if t < len(p) else p[-1]

    for t in range(T):
        occ: Dict[Cell, str] = {}
        for n in names:
            c = cell_at(n, t)
            if c in occ:
                return (t, occ[c], n, "vertex")
            occ[c] = n
        if t + 1 < T:
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    ca0, cb0 = cell_at(a, t), cell_at(b, t)
                    ca1, cb1 = cell_at(a, t + 1), cell_at(b, t + 1)
                    if ca0 == cb1 and cb0 == ca1 and ca0 != cb0:
                        return (t + 1, a, b, "swap")
    return None


def _priority_key(
    name: str,
    *,
    loaded: bool,
    urgent: bool,
    dist_goal: int,
) -> Tuple:
    """Higher priority sorts first (low priority yields)."""
    return (
        0 if urgent else 1,
        0 if loaded else 1,
        dist_goal,
        name,
    )


# ---------------------------------------------------------------------------
# Micro-wave → macro-wave fusion (no time-loss merge; conflict → big wave first)
# ---------------------------------------------------------------------------


def _init_micro_waves(names: Sequence[str]) -> Dict[str, str]:
    """Each AGV starts as its own micro-wave (wave_id == name)."""
    return {n: n for n in names}


def _wave_members(wave_of: Dict[str, str], wid: str) -> List[str]:
    return sorted(n for n, w in wave_of.items() if w == wid)


def _wave_size(wave_of: Dict[str, str], wid: str) -> int:
    return sum(1 for w in wave_of.values() if w == wid)


def _paths_conflict_pair(
    traj: Dict[str, List[Cell]],
    a: str,
    b: str,
) -> bool:
    """True if a and b have vertex/swap conflict on their cell timelines."""
    pa = traj.get(a) or []
    pb = traj.get(b) or []
    if not pa or not pb:
        return False
    T = max(len(pa), len(pb))

    def at(p: List[Cell], t: int) -> Cell:
        return p[t] if t < len(p) else p[-1]

    for t in range(T):
        if at(pa, t) == at(pb, t):
            return True
        if t + 1 < T:
            ca0, cb0 = at(pa, t), at(pb, t)
            ca1, cb1 = at(pa, t + 1), at(pb, t + 1)
            if ca0 == cb1 and cb0 == ca1 and ca0 != cb0:
                return True
    return False


def _path_cells(traj: Dict[str, List[Cell]], n: str) -> Set[Cell]:
    return set(traj.get(n) or [])


def _streams_affinity(
    traj: Dict[str, List[Cell]],
    a: str,
    b: str,
) -> bool:
    """True if two agents share corridor cells (same traffic stream candidate)."""
    ca, cb = _path_cells(traj, a), _path_cells(traj, b)
    if not ca or not cb:
        return False
    if ca & cb:
        return True
    # Nearby endpoints: same local corridor neighborhood
    pa, pb = traj.get(a) or [], traj.get(b) or []
    ends = []
    if pa:
        ends.append(pa[0])
        ends.append(pa[-1])
    if pb:
        ends.append(pb[0])
        ends.append(pb[-1])
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            if _manh(ends[i], ends[j]) <= 2:
                # only count cross-agent pairs
                ai = ends[i] in ca
                bj = ends[j] in cb
                aj = ends[j] in ca
                bi = ends[i] in cb
                if (ai and bj) or (aj and bi):
                    return True
    return False


def _fuse_waves_no_loss(
    traj: Dict[str, List[Cell]],
    wave_of: Dict[str, str],
    movers: Sequence[str],
) -> Dict[str, str]:
    """Greedily fuse micro-waves whose paths do not conflict (zero wait penalty).

    Only fuse agents that share a traffic stream (spatial affinity). Fusion is
    lossless: joint commit equals solo timelines (no forced hold). Prefer
    absorbing into the larger wave.
    """
    out = dict(wave_of)
    active = [n for n in movers if n in traj and len(traj.get(n) or []) > 1]
    changed = True
    guard = 0
    while changed and guard < len(active) + 2:
        guard += 1
        changed = False
        wids = sorted(
            {out[n] for n in active},
            key=lambda w: (-_wave_size(out, w), w),
        )
        for i, wa in enumerate(wids):
            mem_a = _wave_members(out, wa)
            for wb in wids[i + 1 :]:
                mem_b = _wave_members(out, wb)
                # Must have stream affinity between the two groups
                affinity = False
                for a in mem_a:
                    for b in mem_b:
                        if _streams_affinity(traj, a, b):
                            affinity = True
                            break
                    if affinity:
                        break
                if not affinity:
                    continue
                clash = False
                for a in mem_a:
                    for b in mem_b:
                        if _paths_conflict_pair(traj, a, b):
                            clash = True
                            break
                    if clash:
                        break
                if clash:
                    continue
                sa, sb = len(mem_a), len(mem_b)
                if sa > sb or (sa == sb and wa <= wb):
                    keep, drop = wa, wb
                else:
                    keep, drop = wb, wa
                for n in _wave_members(out, drop):
                    out[n] = keep
                changed = True
                break
            if changed:
                break
    return out


def _try_absorb_into_wave(
    traj: Dict[str, List[Cell]],
    wave_of: Dict[str, str],
    target_wid: str,
    candidates: Sequence[str],
) -> Dict[str, str]:
    """Absorb candidate AGVs into ``target_wid`` if no path conflict with members."""
    out = dict(wave_of)
    members = _wave_members(out, target_wid)
    for c in sorted(candidates):
        if out.get(c) == target_wid:
            continue
        if any(_paths_conflict_pair(traj, c, m) for m in members):
            continue
        out[c] = target_wid
        members.append(c)
    return out


def _pick_keeper_by_wave(
    involved: Sequence[str],
    wave_of: Dict[str, str],
    *,
    rem_map: Dict[str, int],
    active: Dict[str, dict],
    loaded: Dict[str, bool],
) -> Tuple[str, List[str], str]:
    """Prefer the larger wave as keeper; within wave use priority_key.

    Returns (keeper, yielders, reason).
    """
    if not involved:
        raise ValueError("involved empty")
    # Wave sizes among involved
    sizes = {
        n: _wave_size(wave_of, wave_of.get(n, n)) for n in involved
    }
    max_sz = max(sizes.values())
    big = [n for n in involved if sizes[n] == max_sz]
    keeper = min(
        big,
        key=lambda n: (
            0 if n in active else 1,
            *_priority_key(
                n,
                loaded=loaded.get(n, False),
                urgent=bool((active.get(n) or {}).get("urgent")),
                dist_goal=int(rem_map.get(n, 10**6)),
            ),
        ),
    )
    yielders = [n for n in involved if n != keeper]
    k_wid = wave_of.get(keeper, keeper)
    reason = (
        f"wave_fuse size={sizes[keeper]} wid={k_wid} "
        f"vs sizes={{{', '.join(f'{n}:{sizes[n]}' for n in involved)}}}"
    )
    return keeper, yielders, reason


def _astar_len(
    start: Cell,
    goal: Cell,
    static: Set[Cell],
    *,
    allow: Optional[Set[Cell]] = None,
    extra_block: Optional[Set[Cell]] = None,
) -> int:
    """Remaining static A* length; inf-like if unreachable."""
    path = _safe_path(start, goal, static, allow=allow, extra_block=extra_block)
    if not path:
        return 10**6
    return max(0, len(path) - 1)


def _true_goal(info: Optional[dict], pos: Cell) -> Cell:
    if not info:
        return pos
    if info.get("leg") == "to_pickup":
        return tuple(info["pickup"])
    ends = [tuple(e) for e in (info.get("ends") or [])]
    if not ends:
        return tuple(info.get("pickup") or pos)
    return min(ends, key=lambda e: (_manh(pos, e), e))


def _llm_scene_brief(
    pickups: List[dict],
    dropoffs: List[dict],
    *,
    n_agv: int,
    max_active: int,
) -> str:
    pk = ", ".join(
        f"{p.get('name')}({p.get('x')},{p.get('y')})" for p in (pickups or [])
    )
    df = ", ".join(
        f"{p.get('name')}({p.get('x')},{p.get('y')})" for p in (dropoffs or [])
    )
    return (
        "## 本局地图\n"
        f"20×20 迷宫，{n_agv} 台 AGV，同时最多 {max_active} 台有任务；通道经常 1 格宽，对向无法侧向错车。\n"
        f"取货点: {pk}\n"
        f"卸货点: {df}\n"
        "空车去取货，满载去卸货。你只决定冲突时谁占用通道。"
    )


def _llm_agent_row(
    n: str,
    *,
    pose: Dict[str, Pose],
    active: Dict[str, dict],
    loaded: Dict[str, bool],
    tid: Dict[str, str],
    dest: Dict[str, str],
    yield_until: Dict[str, int],
    now: int,
    traj: Dict[str, List[Cell]],
    static: Set[Cell],
    rem_override: Optional[int] = None,
) -> dict:
    info = active.get(n)
    allow = set()
    if info:
        allow = set(info.get("ends") or []) | {tuple(info["pickup"])}
    goal = _true_goal(info, pose[n][:2])
    rem = (
        rem_override
        if rem_override is not None
        else _astar_len(pose[n][:2], goal, static, allow=allow or None)
    )
    path = (traj.get(n) or [pose[n][:2]])[:6]
    return {
        "name": n,
        "pos": [int(pose[n][0]), int(pose[n][1])],
        "pitch": int(pose[n][2]),
        "loaded": bool(loaded.get(n)),
        "urgent": bool((info or {}).get("urgent")),
        "leg": (info or {}).get("leg") or "idle",
        "task_id": tid.get(n) or "",
        "destination": dest.get(n) or "",
        "pickup": list(info["pickup"]) if info and info.get("pickup") else None,
        "goal": [int(goal[0]), int(goal[1])],
        "rem_astar": rem,
        "active": n in active,
        "yield_locked": now < int(yield_until.get(n, -1)),
        "path_next": [[int(c[0]), int(c[1])] for c in path],
    }


def _truncate_off_corridor(
    cells: List[Cell],
    forbidden: Set[Cell],
    *,
    min_steps: int = 1,
    max_steps: int = 8,
) -> List[Cell]:
    """Keep only enough of a yield path to leave the keeper corridor, then stop.

    Resume original task as soon as the conflict cells are clear.
    """
    if not cells or len(cells) < 2:
        return cells
    out = [cells[0]]
    for i, c in enumerate(cells[1:], start=1):
        out.append(c)
        if c not in forbidden and i >= min_steps:
            break
        if i >= max_steps:
            break
    return out


def _pick_resume_park(
    pos: Cell,
    goal: Cell,
    parks: List[Cell],
    static: Set[Cell],
    *,
    occupied: Set[Cell],
    forbidden: Set[Cell],
    allow: Optional[Set[Cell]] = None,
    taboo: Optional[Set[Cell]] = None,
) -> Optional[Tuple[Cell, List[Cell]]]:
    """Stage park that reduces remaining A* to ``goal``, then is reachable.

    Prefers parks with smaller remaining A* than current (closer to the task).
    Forbidden cells (keeper corridor) are avoided as destinations.
    ``taboo`` (O1): skip recent park cells to reduce ping-pong.
    """
    taboo = taboo or set()
    cur_rem = _astar_len(pos, goal, static, allow=allow)
    cur_m = _manh(pos, goal)
    candidates: List[Cell] = []
    for p in parks:
        if p in occupied or p in forbidden or p == pos:
            continue
        if _manh(p, goal) <= cur_m + 2:
            candidates.append(p)
    if len(candidates) < 16:
        rest = [
            p
            for p in parks
            if p not in occupied
            and p not in forbidden
            and p != pos
            and p not in candidates
        ]
        rest.sort(key=lambda p: (_manh(p, goal), _manh(pos, p)))
        candidates.extend(rest[: 16 - len(candidates)])
    ranked: List[Tuple[int, int, int, Cell, List[Cell]]] = []
    for park in candidates:
        path = _safe_path(pos, park, static, extra_block=occupied - {pos})
        if not path or len(path) < 2:
            continue
        rem = _astar_len(park, goal, static, allow=allow)
        # Progress toward task first; then short detour; O1 deprioritize taboo
        better = 0 if rem < cur_rem else (1 if rem == cur_rem else 2)
        taboo_pen = 1 if park in taboo else 0
        ranked.append((better, taboo_pen, rem, park, path))
    if not ranked:
        return None
    ranked.sort(key=lambda x: (x[0], x[1], x[2], len(x[4]), x[3]))
    _better, _taboo, rem, park, path = ranked[0]
    # Refuse a strictly worse park unless we have no improving option
    if _better == 2 and any(b == 0 for b, *_ in ranked):
        ranked = [r for r in ranked if r[0] == 0]
        ranked.sort(key=lambda x: (x[1], x[2], len(x[4]), x[3]))
        _better, _taboo, rem, park, path = ranked[0]
    return park, path


def _pick_resume_park_with_taboo(
    pos: Cell,
    goal: Cell,
    parks: List[Cell],
    static: Set[Cell],
    *,
    occupied: Set[Cell],
    forbidden: Set[Cell],
    allow: Optional[Set[Cell]] = None,
    taboo: Optional[Set[Cell]] = None,
) -> Optional[Tuple[Cell, List[Cell]]]:
    """O1 wrapper: prefer non-taboo parks; relax taboo if none reachable."""
    picked = _pick_resume_park(
        pos,
        goal,
        parks,
        static,
        occupied=occupied,
        forbidden=forbidden,
        allow=allow,
        taboo=taboo,
    )
    if picked or not taboo:
        return picked
    return _pick_resume_park(
        pos,
        goal,
        parks,
        static,
        occupied=occupied,
        forbidden=forbidden,
        allow=allow,
        taboo=set(),
    )


def _force_resume_cells(
    agv: str,
    pose: Dict[str, Pose],
    goal: Cell,
    static: Set[Cell],
    stations: Set[Cell],
    names: List[str],
    *,
    allow: Optional[Set[Cell]] = None,
    avoid: Optional[Set[Cell]] = None,
) -> Optional[List[Cell]]:
    """O2: one greedy/resume step when active AGV has waited too long."""
    here = pose[agv][:2]
    if here == goal:
        return None
    blocked = static | stations | {pose[n][:2] for n in names if n != agv}
    nxt = _greedy_step(
        here, goal, static, blocked, allow=allow, avoid=avoid or set()
    )
    if nxt is None and avoid:
        nxt = _greedy_step(here, goal, static, blocked, allow=allow)
    if nxt is not None:
        return [here, nxt]
    return None


def _greedy_step(
    pos: Cell,
    goal: Cell,
    static: Set[Cell],
    blocked: Set[Cell],
    *,
    allow: Optional[Set[Cell]] = None,
    avoid: Optional[Set[Cell]] = None,
) -> Optional[Cell]:
    """One 4-connected step that strictly decreases remaining A* if possible."""
    avoid = avoid or set()
    cur = _astar_len(pos, goal, static, allow=allow)
    best = None
    best_r = cur
    for nxt in _neighbors(pos):
        if nxt in blocked or nxt in static and nxt not in (allow or set()) | {goal}:
            continue
        if nxt in avoid:
            continue
        r = _astar_len(nxt, goal, static, allow=allow)
        if r < best_r:
            best_r = r
            best = nxt
    return best


def _evacuate_path(
    start: Cell,
    corridor: Set[Cell],
    static: Set[Cell],
    occupied: Set[Cell],
    *,
    opponent: Optional[Cell] = None,
    allow: Optional[Set[Cell]] = None,
    max_len: int = 4,
) -> Optional[List[Cell]]:
    """Leave keeper occupancy band. remA* may increase (swap / yield-miss)."""
    occ = set(occupied) - {start}
    # allow opens walls only; never step onto live AGVs
    allow_set = (set(allow or set()) | {start}) - occ
    blocked = (set(static) | occ) - allow_set
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    depth: Dict[Cell, int] = {start: 0}
    q: deque = deque([start])
    found: Optional[Cell] = None
    while q:
        cur = q.popleft()
        if cur != start and cur not in corridor and cur not in occ:
            found = cur
            break
        if depth[cur] >= max_len:
            continue
        for nxt in _neighbors(cur):
            if nxt in parent or nxt in blocked:
                continue
            parent[nxt] = cur
            depth[nxt] = depth[cur] + 1
            q.append(nxt)
    if found is None:
        step = _step_off_band(
            start, corridor, static, occupied, opponent=opponent, allow=allow
        )
        if step is None:
            return None
        return [start, step]
    rev: List[Cell] = []
    cur: Optional[Cell] = found
    while cur is not None:
        rev.append(cur)
        cur = parent[cur]
    rev.reverse()
    return rev if len(rev) >= 2 else None


def _step_off_band(
    here: Cell,
    corridor: Set[Cell],
    static: Set[Cell],
    occupied: Set[Cell],
    *,
    opponent: Optional[Cell] = None,
    allow: Optional[Set[Cell]] = None,
) -> Optional[Cell]:
    """One legal step that leaves the opponent / corridor if possible."""
    occ = set(occupied) - {here}
    allow_set = (set(allow or set()) | {here}) - occ
    blocked = (set(static) | occ) - allow_set
    ranked: List[Tuple[int, int, int, Cell]] = []
    for nxt in _neighbors(here):
        if nxt in blocked or nxt in occ:
            continue
        if opponent is not None and nxt == opponent:
            continue
        off = 0 if nxt not in corridor else 1
        away = -_manh(nxt, opponent) if opponent is not None else 0
        ranked.append((off, away, _manh(nxt, here), nxt))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][3]


def _flee_any_step(
    here: Cell,
    static: Set[Cell],
    occupied: Set[Cell],
    *,
    avoid: Optional[Set[Cell]] = None,
    opponent: Optional[Cell] = None,
) -> Optional[Cell]:
    """Last-resort one-step retreat for idle / yield-miss (ignore task goal)."""
    avoid = avoid or set()
    ranked: List[Tuple[int, int, int, Cell]] = []
    for nxt in _neighbors(here):
        if nxt in static or nxt in occupied or nxt in avoid:
            continue
        if opponent is not None and nxt == opponent:
            continue
        away = -_manh(nxt, opponent) if opponent is not None else 0
        ranked.append((0 if nxt not in avoid else 1, away, _manh(nxt, here), nxt))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][3]


def _egress_probe_cells(
    start: Cell,
    corridor: Set[Cell],
    static: Set[Cell],
    *,
    max_len: int = 3,
) -> Set[Cell]:
    """Cells a yielder may need to leave the corridor (ignoring dynamic agents)."""
    allow_set = {start}
    blocked = set(static) - allow_set
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    depth: Dict[Cell, int] = {start: 0}
    q: deque = deque([start])
    out: Set[Cell] = set()
    while q:
        cur = q.popleft()
        if cur != start:
            out.add(cur)
        if cur != start and cur not in corridor:
            continue
        if depth[cur] >= max_len:
            continue
        for nxt in _neighbors(cur):
            if nxt in parent or nxt in blocked:
                continue
            parent[nxt] = cur
            depth[nxt] = depth[cur] + 1
            q.append(nxt)
    out |= set(_neighbors(start))
    return out


def _agents_blocking_egress(
    yielder: str,
    *,
    keeper: str,
    pose: Dict[str, Pose],
    names: Sequence[str],
    corridor: Set[Cell],
    static: Set[Cell],
) -> List[str]:
    """Other AGVs sitting on yielder evacuate exits / corridor (multi-agent jam)."""
    probe = _egress_probe_cells(
        pose[yielder][:2], corridor, static, max_len=3
    )
    probe |= set(corridor)
    blockers: List[Tuple[int, str]] = []
    ypos = pose[yielder][:2]
    for n in names:
        if n == yielder or n == keeper:
            continue
        cell = pose[n][:2]
        if cell in probe:
            blockers.append((_manh(cell, ypos), n))
    blockers.sort()
    return [n for _, n in blockers]


def _expand_yield_group(
    keeper: str,
    yielders: List[str],
    *,
    pose: Dict[str, Pose],
    names: Sequence[str],
    corridor: Set[Cell],
    static: Set[Cell],
    max_extra: int = 4,
) -> List[str]:
    """Grow yielders: anyone on corridor or blocking evacuate egress joins."""
    group = list(yielders)
    seen = set(group) | {keeper}
    # Anyone already sitting on keeper corridor (except keeper)
    for n in names:
        if n in seen:
            continue
        if pose[n][:2] in corridor:
            group.append(n)
            seen.add(n)
    # Iteratively add egress blockers of current yielders
    guard = 0
    while guard < max_extra:
        guard += 1
        added = False
        for y in list(group):
            for b in _agents_blocking_egress(
                y,
                keeper=keeper,
                pose=pose,
                names=names,
                corridor=corridor,
                static=static,
            ):
                if b not in seen:
                    group.append(b)
                    seen.add(b)
                    added = True
                    if len(group) - len(yielders) >= max_extra:
                        break
            if len(group) - len(yielders) >= max_extra:
                break
        if not added:
            break
    return group


def _order_evacuators(
    yielders: Sequence[str],
    *,
    keeper: str,
    pose: Dict[str, Pose],
    corridor: Set[Cell],
) -> List[str]:
    """Clear outer blockers first (far from keeper / off corridor), then inner."""
    kpos = pose[keeper][:2]

    def key(n: str) -> Tuple:
        cell = pose[n][:2]
        on_c = 1 if cell in corridor else 0
        return (-_manh(cell, kpos), on_c, n)

    return sorted(yielders, key=key)


def _nearest_queued_pickup(
    queues: Dict[str, list],
    order: List[str],
    from_cell: Cell,
) -> Optional[Cell]:
    best: Optional[Cell] = None
    best_d = 10**9
    for st in order:
        for task in queues.get(st, []):
            pk = tuple(task["pickup_point"])
            d = _manh(from_cell, pk)
            if d < best_d:
                best_d = d
                best = pk
    return best


def _idle_staging_goal(
    pos: Cell,
    queues: Dict[str, list],
    order: List[str],
    parks: List[Cell],
    occupied: Set[Cell],
) -> Cell:
    """Target for task-free AGVs: nearest queued pickup, else a free park."""
    pk = _nearest_queued_pickup(queues, order, pos)
    if pk is not None and pk != pos:
        return pk
    for park in sorted(
        _side_park_candidates(parks), key=lambda p: (_manh(pos, p), p)
    ):
        if park not in occupied and park != pos:
            return park
    for park in sorted(parks, key=lambda p: (_manh(pos, p), p)):
        if park not in occupied and park != pos:
            return park
    return pos


def _idle_hold_far(
    name: str,
    pose: Dict[str, Pose],
    active: Dict[str, dict],
    traj: Dict[str, List[Cell]],
    *,
    dist: int = 0,
) -> bool:
    """Optionally hold idle AGV far from traffic. ``dist<=0`` disables (max effort)."""
    if dist <= 0 or not active:
        return False
    pos = pose[name][:2]
    if min(_manh(pos, pose[a][:2]) for a in active) <= dist:
        return False
    hot: Set[Cell] = set()
    for a in active:
        hot.update((traj.get(a) or [])[:14])
    return pos not in hot


def _poses_atomic(a: Pose, b: Pose) -> bool:
    """True if a→b is one legal wait/turn/move (no diagonal / teleport)."""
    ax, ay, ap = int(a[0]), int(a[1]), int(a[2]) % 360
    bx, by, bp = int(b[0]), int(b[1]), int(b[2]) % 360
    d = abs(ax - bx) + abs(ay - by)
    if d == 0:
        return True
    if d != 1:
        return False
    # move must keep pitch aligned with delta
    dx, dy = bx - ax, by - ay
    need = _pitch_from_delta(dx, dy, ap)
    return bp == need == ap

def _plan_toward_goal(
    mod,
    name: str,
    pose: Pose,
    goal: Cell,
    static_list: list,
    static: Set[Cell],
    blocked: Set[Cell],
    *,
    allow: Optional[Set[Cell]] = None,
) -> List[Cell]:
    """Solo path toward goal, or one greedy step — never freeze if progress exists."""
    here = pose[:2]
    if here == goal:
        return [here]
    cells = _solo_to_goal(
        mod,
        name,
        pose,
        goal,
        static_list,
        now=0,
        extra_block=blocked - {here},
        allow=allow,
    )
    if cells and len(cells) > 1:
        return cells
    nxt = _greedy_step(here, goal, static, blocked, allow=allow)
    if nxt is not None:
        return [here, nxt]
    return [here]


def _tick_conflicts(
    pose: Dict[str, Pose],
    prop: Dict[str, Cell],
) -> List[Tuple[str, str, str]]:
    """Vertex/swap conflicts among one-step proposals."""
    names = sorted(prop)
    out: List[Tuple[str, str, str]] = []
    for i, a in enumerate(names):
        pa = prop[a]
        ca = pose[a][:2]
        for b in names[i + 1 :]:
            pb = prop[b]
            cb = pose[b][:2]
            if pa == pb:
                out.append((a, b, "vertex"))
            elif pa == cb and pb == ca and pa != pb:
                out.append((a, b, "swap"))
    return out


def _propose_first_step(
    mod,
    name: str,
    pose: Dict[str, Pose],
    goal: Cell,
    static_list: list,
    static: Set[Cell],
    stations: Set[Cell],
    parks: List[Cell],
    names: List[str],
    *,
    allow: Optional[Set[Cell]] = None,
    move: bool = True,
) -> Tuple[Cell, List[Cell]]:
    """Next cell = first step on solo BFS/A* toward goal; ``move=False`` => hold."""
    here = pose[name][:2]
    if not move or here == goal:
        return here, [here]
    blocked = static | stations | {pose[n][:2] for n in names if n != name}
    path = _solo_to_goal(
        mod,
        name,
        pose[name],
        goal,
        static_list,
        now=0,
        extra_block=blocked - {here},
        allow=allow,
    )
    if path and len(path) >= 2:
        return path[1], path
    nxt = _greedy_step(here, goal, static, blocked, allow=allow)
    if nxt is not None:
        return nxt, [here, nxt]
    picked = _pick_resume_park(
        here,
        goal,
        parks,
        static,
        occupied={pose[n][:2] for n in names},
        forbidden=set(),
        allow=allow,
    )
    if picked:
        _park, ppath = picked
        if len(ppath) >= 2:
            return ppath[1], ppath
    return here, [here]


def _propose_effort_cell(
    name: str,
    pose: Dict[str, Pose],
    *,
    goal: Cell,
    static: Set[Cell],
    stations: Set[Cell],
    parks: List[Cell],
    names: List[str],
    allow: Optional[Set[Cell]] = None,
) -> Cell:
    """Best single-step move toward ``goal``; stay put only if no progress exists."""
    here = pose[name][:2]
    if here == goal:
        return here
    blocked = static | stations | {pose[n][:2] for n in names if n != name}
    nxt = _greedy_step(here, goal, static, blocked, allow=allow)
    if nxt is not None:
        return nxt
    picked = _pick_resume_park(
        here,
        goal,
        parks,
        static,
        occupied={pose[n][:2] for n in names},
        forbidden=set(),
        allow=allow,
    )
    if picked:
        _park, path = picked
        if len(path) >= 2:
            return path[1]
    return here


def _yield_alts(
    y: str,
    keeper: str,
    pose: Dict[str, Pose],
    prop: Dict[str, Cell],
    goal: Cell,
    static: Set[Cell],
    stations: Set[Cell],
    names: List[str],
    *,
    allow: Optional[Set[Cell]] = None,
) -> List[Cell]:
    here = pose[y][:2]
    keeper_next = prop[keeper]
    keeper_here = pose[keeper][:2]
    blocked = static | stations | {pose[n][:2] for n in names if n != y}
    opts: List[Cell] = []
    for nxt in _neighbors(here):
        if nxt in blocked:
            continue
        if nxt == keeper_next and keeper_next != keeper_here:
            continue
        if nxt == keeper_here and keeper_next == here:
            continue
        opts.append(nxt)
    opts.append(here)
    opts.sort(
        key=lambda c: (
            _astar_len(c, goal, static, allow=allow),
            _manh(c, goal),
            c,
        )
    )
    seen: Set[Cell] = set()
    uniq: List[Cell] = []
    for c in opts:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _resolve_one_second(
    pose: Dict[str, Pose],
    proposals: Dict[str, Cell],
    solo_paths: Dict[str, List[Cell]],
    names: List[str],
    active: Dict[str, dict],
    loaded: Dict[str, bool],
    static: Set[Cell],
    stations: Set[Cell],
    parks: List[Cell],
    pair_hits: Dict[Tuple[str, str], int],
    yield_until: Dict[str, int],
    now: int,
    coord: LlmCoordinator,
    *,
    use_llm: bool,
) -> Dict[str, Cell]:
    prop = dict(proposals)

    def _goal_of(n: str) -> Cell:
        return _true_goal(active.get(n), pose[n][:2])

    def _allow_of(n: str) -> Set[Cell]:
        info = active.get(n)
        if not info:
            return set()
        return set(info.get("ends") or []) | {tuple(info["pickup"])}

    def _rank(a: str, b: str) -> Tuple[str, str]:
        rem = {
            n: _astar_len(pose[n][:2], _goal_of(n), static, allow=_allow_of(n))
            for n in (a, b)
        }
        ranked = sorted(
            (a, b),
            key=lambda n: (
                1 if now < int(yield_until.get(n, -1)) else 0,
                *_priority_key(
                    n,
                    loaded=loaded.get(n, False),
                    urgent=bool((active.get(n) or {}).get("urgent")),
                    dist_goal=rem[n],
                ),
            ),
        )
        return ranked[0], ranked[1]

    for _ in range(96):
        confs = _tick_conflicts(pose, prop)
        if not confs:
            return prop
        a, b, kind = confs[0]
        pair = tuple(sorted((a, b)))
        pair_hits[pair] += 1
        stuck_thresh = 2 if kind == "swap" else 3
        if int(pair_hits[pair]) >= stuck_thresh or not use_llm:
            keeper, yielder = _rank(a, b)
            if int(pair_hits[pair]) >= stuck_thresh:
                rem = {
                    n: _astar_len(
                        pose[n][:2], _goal_of(n), static, allow=_allow_of(n)
                    )
                    for n in (a, b)
                }
                keeper = min((a, b), key=lambda n: (rem[n], n))
                yielder = b if keeper == a else a
        else:
            rem_map = {
                n: _astar_len(
                    pose[n][:2], _goal_of(n), static, allow=_allow_of(n)
                )
                for n in (a, b)
            }
            ranked = sorted(
                (a, b),
                key=lambda n: (
                    1 if now < int(yield_until.get(n, -1)) else 0,
                    *_priority_key(
                        n,
                        loaded=loaded.get(n, False),
                        urgent=bool((active.get(n) or {}).get("urgent")),
                        dist_goal=rem_map[n],
                    ),
                ),
            )
            snap = {
                "t": now,
                "kind": kind,
                "t_star": 1,
                "involved": [a, b],
                "agents": [
                    {
                        "name": n,
                        "pos": list(pose[n][:2]),
                        "pitch": int(pose[n][2]),
                        "loaded": bool(loaded.get(n)),
                        "urgent": bool((active.get(n) or {}).get("urgent")),
                        "leg": (active.get(n) or {}).get("leg") or "idle",
                        "task_id": "",
                        "destination": "",
                        "goal": list(_goal_of(n)),
                        "rem_astar": rem_map[n],
                        "active": n in active,
                    }
                    for n in (a, b)
                ],
            }
            dec = coord.decide(snap, ranked, now=now, pair_hits=int(pair_hits[pair]))
            keeper = dec.keeper
            yielder = dec.yielders[0] if dec.yielders else (
                b if keeper == a else a
            )
        goal_y = _goal_of(yielder)
        kp = solo_paths.get(keeper) or [pose[keeper][:2]]
        forbidden = set(kp[: min(6, len(kp))]) | {proposals[keeper]}
        alts = list(
            _yield_alts(
                yielder,
                keeper,
                pose,
                prop,
                goal_y,
                static,
                stations,
                names,
                allow=_allow_of(yielder) or None,
            )
        )
        if int(pair_hits[pair]) >= stuck_thresh:
            picked = _pick_resume_park(
                pose[yielder][:2],
                goal_y,
                parks,
                static,
                occupied={pose[n][:2] for n in names},
                forbidden=forbidden,
                allow=_allow_of(yielder) or None,
            )
            if picked:
                _park, ppath = picked
                if len(ppath) >= 2 and ppath[1] not in alts:
                    alts.insert(0, ppath[1])
        fixed = False
        for alt in alts:
            trial = dict(prop)
            trial[yielder] = alt
            if not _tick_conflicts(pose, trial):
                prop = trial
                fixed = True
                break
        if not fixed:
            prop[yielder] = pose[yielder][:2]
    return prop


def _required_pitch(dx: int, dy: int) -> int:
    if dx == 1:
        return 0
    if dx == -1:
        return 180
    if dy == 1:
        return 90
    if dy == -1:
        return 270
    raise ValueError(f"non-unit move ({dx}, {dy})")


def _expand_cell_path(start: Pose, cells: List[Cell]) -> List[Pose]:
    """Expand xy path into pose timeline: turn-in-place (1s) then move (1s)."""
    if not cells:
        return []
    out: List[Pose] = []
    cur = start
    for cell in cells:
        x1, y1 = int(cell[0]), int(cell[1])
        if (x1, y1) == (cur[0], cur[1]):
            continue
        dx, dy = x1 - cur[0], y1 - cur[1]
        tp = _required_pitch(dx, dy)
        if cur[2] != tp:
            cur = (cur[0], cur[1], tp)
            out.append(cur)
        cur = (x1, y1, tp)
        out.append(cur)
    return out


def _apply_one_tick(
    steps_by: Dict[str, List[dict]],
    pose: Dict[str, Pose],
    next_by: Dict[str, Cell],
    now: int,
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    on_cell=None,
) -> int:
    names = list(pose.keys())
    expanded: Dict[str, List[Pose]] = {
        n: _expand_cell_path(pose[n], [next_by[n]]) for n in names
    }
    T = max((len(expanded[n]) for n in names), default=0)
    for k in range(T):
        t = now + 1 + k
        for n in names:
            if k < len(expanded[n]):
                pose[n] = expanded[n][k]
            if on_cell is not None:
                on_cell(n, pose[n][:2], t)
            steps_by[n].append(
                _hold(
                    n,
                    pose[n],
                    t,
                    loaded=loaded.get(n, False),
                    dest=dest.get(n, ""),
                    tid=tid.get(n, ""),
                )
            )
    return now + T


def _expand_full_timeline(start: Pose, cells: List[Cell]) -> List[Pose]:
    """Pose per sim second (start + turn holds + moves)."""
    if not cells:
        return [start]
    path = list(cells)
    if path[0] != (start[0], start[1]):
        path = [(start[0], start[1])] + path
    steps = _expand_cell_path(start, path[1:])
    return [start] + steps


def _apply_timeline(
    steps_by: Dict[str, List[dict]],
    pose: Dict[str, Pose],
    timelines: Dict[str, List[Pose]],
    now: int,
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    until_idx: Optional[int] = None,
    on_cell=None,
    abort_check=None,
) -> int:
    """Apply pre-expanded pose timelines; ``until_idx`` is inclusive sim index.

    If ``abort_check()`` becomes true after a tick (e.g. an AGV needs a new
    stage path), stop committing and return so the caller can replan.
    """
    names = list(pose.keys())
    chunks: Dict[str, List[Pose]] = {}
    for n in names:
        tl = timelines.get(n) or [pose[n]]
        end = len(tl) if until_idx is None else min(len(tl), int(until_idx) + 1)
        chunks[n] = tl[1:end]
    T = max((len(chunks[n]) for n in names), default=0)
    applied = 0
    for k in range(T):
        t = now + 1 + k
        for n in names:
            if k < len(chunks[n]):
                pose[n] = chunks[n][k]
            if on_cell is not None:
                on_cell(n, pose[n][:2], t)
            steps_by[n].append(
                _hold(
                    n,
                    pose[n],
                    t,
                    loaded=loaded.get(n, False),
                    dest=dest.get(n, ""),
                    tid=tid.get(n, ""),
                )
            )
        applied = k + 1
        if abort_check is not None and abort_check():
            break
    return now + applied


BYSTANDER_MAX_STEPS = 16


def _clip_path_clear_of_agents(
    cells: List[Cell],
    mover: str,
    pose: Dict[str, Pose],
    names: List[str],
) -> List[Cell]:
    """Truncate a cell path before it would land on / cross another AGV."""
    if not cells:
        return cells
    occ = {pose[n][:2] for n in names if n != mover}
    out: List[Cell] = [cells[0]]
    for c in cells[1:]:
        if c in occ:
            break
        out.append(c)
    return out


def _advance_bystanders(
    steps_by: Dict[str, List[dict]],
    pose: Dict[str, Pose],
    timelines: Dict[str, List[Pose]],
    names: List[str],
    frozen: Set[str],
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    now: int,
    on_cell=None,
    abort_check=None,
    max_steps: int = BYSTANDER_MAX_STEPS,
    goal_of=None,
    allow_of=None,
    static: Optional[Set[Cell]] = None,
    stations: Optional[Set[Cell]] = None,
) -> int:
    """Keep non-involved AGVs on stage work while a conflict group resolves.

    Prefer remaining wave timelines; when exhausted, take one greedy A* step
    toward each mover's stage / staging goal so bystanders keep working during
    evacuate/yield instead of freezing until the jam clears.
    """
    movers = [n for n in names if n not in frozen]
    if not movers:
        return now
    cursor: Dict[str, int] = {}
    for n in movers:
        tl = timelines.get(n) or [pose[n]]
        idx = 0
        for i, p in enumerate(tl):
            if (p[0], p[1], p[2]) == pose[n]:
                idx = i
                break
            if p[:2] == pose[n][:2]:
                idx = i
        cursor[n] = idx + 1

    for _ in range(max_steps):
        if abort_check is not None and abort_check():
            break
        frozen_cells = {pose[f][:2] for f in frozen}
        others_now = {pose[m][:2] for m in names}
        cand: Dict[str, Pose] = {}
        for n in movers:
            tl = timelines.get(n) or [pose[n]]
            c = cursor[n]
            pick: Optional[Pose] = None
            used_tl = False
            if c < len(tl) and _poses_atomic(pose[n], tl[c]):
                pick = tl[c]
                used_tl = True
            if (
                (pick is None or pick[:2] == pose[n][:2])
                and goal_of is not None
                and static is not None
            ):
                g = goal_of(n)
                if g is not None and g != pose[n][:2]:
                    allow = allow_of(n) if allow_of is not None else set()
                    blocked = (
                        set(static)
                        | set(stations or set())
                        | (others_now - {pose[n][:2]})
                    )
                    step = _greedy_step(
                        pose[n][:2], g, static, blocked, allow=allow
                    )
                    if step is not None and step not in frozen_cells:
                        exp = _expand_full_timeline(
                            pose[n], [pose[n][:2], step]
                        )
                        if len(exp) >= 2 and _poses_atomic(pose[n], exp[1]):
                            pick = exp[1]
                            used_tl = False
            cand[n] = pick if pick is not None else pose[n]
            if not used_tl:
                # mark so cursor does not advance on stale timeline
                cursor[n] = max(cursor[n], len(tl))

        nxt: Dict[str, Pose] = dict(cand)
        for n in movers:
            if not _poses_atomic(pose[n], nxt[n]):
                nxt[n] = pose[n]
            cell = nxt[n][:2]
            if cell != pose[n][:2] and cell in frozen_cells:
                nxt[n] = pose[n]

        def _joint_ok(trial: Dict[str, Pose]) -> bool:
            cells = [pose[f][:2] for f in frozen] + [
                trial[n][:2] for n in movers
            ]
            if len(cells) != len(set(cells)):
                return False
            for i, a in enumerate(movers):
                for b in movers[i + 1 :]:
                    if (
                        trial[a][:2] == pose[b][:2]
                        and trial[b][:2] == pose[a][:2]
                        and trial[a][:2] != pose[a][:2]
                    ):
                        return False
            return True

        if not _joint_ok(nxt):
            order = sorted(
                movers,
                key=lambda n: 0 if nxt[n][:2] == pose[n][:2] else 1,
            )
            for n in reversed(order):
                if _joint_ok(nxt):
                    break
                nxt[n] = pose[n]

        if not _joint_ok(nxt):
            break

        moved = any(nxt[n] != pose[n] for n in movers)
        if not moved:
            break

        one_tl: Dict[str, List[Pose]] = {
            f: [pose[f], pose[f]] for f in frozen
        }
        for n in movers:
            one_tl[n] = [pose[n], nxt[n]]
            if nxt[n] != pose[n]:
                tl = timelines.get(n) or [pose[n]]
                c = cursor[n]
                if c < len(tl) and tl[c] == nxt[n]:
                    cursor[n] = c + 1

        now = _apply_timeline(
            steps_by,
            pose,
            one_tl,
            now,
            loaded=loaded,
            dest=dest,
            tid=tid,
            until_idx=1,
            on_cell=on_cell,
            abort_check=abort_check,
        )
    return now

def _apply_cells(
    steps_by: Dict[str, List[dict]],
    pose: Dict[str, Pose],
    cells_by: Dict[str, List[Cell]],
    now: int,
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
    until_idx: Optional[int] = None,
    on_cell=None,
) -> int:
    """Apply synchronized cell paths.

    ``until_idx``: max *original* path index to reach (inclusive), counting the
    leading start cell. After trimming the start, we apply ``until_idx`` steps.
    ``on_cell(name, cell, t)`` optional hook after each step (pick mid-path).
    """
    names = list(pose.keys())
    trimmed: Dict[str, List[Cell]] = {}
    for n in names:
        p = list(cells_by.get(n) or [pose[n][:2]])
        if len(p) >= 2 and p[0] == pose[n][:2]:
            p = p[1:]
        if not p:
            p = [pose[n][:2]]
        if until_idx is not None:
            p = p[: max(0, int(until_idx))]
            if not p:
                p = [pose[n][:2]]
        trimmed[n] = p
    expanded: Dict[str, List[Pose]] = {
        n: _expand_cell_path(pose[n], trimmed[n]) for n in names
    }
    T = max((len(expanded[n]) for n in names), default=0)
    for k in range(T):
        t = now + 1 + k
        for n in names:
            if k < len(expanded[n]):
                pose[n] = expanded[n][k]
            x, y = pose[n][0], pose[n][1]
            if on_cell is not None:
                on_cell(n, (x, y), t)
            steps_by[n].append(
                _hold(
                    n,
                    pose[n],
                    t,
                    loaded=loaded.get(n, False),
                    dest=dest.get(n, ""),
                    tid=tid.get(n, ""),
                )
            )
    return now + T


def solve_conflict_horizon(
    slot: int = 3,
    *,
    max_tasks: int = 0,
    max_active: int = 8,
    wall: float = 12.0,
    deadline: int = 2000,
    use_llm: bool = False,
    llm_model: str = "",
    llm_cooldown: int = 8,
    tuning: Optional[CHTuning] = None,
    quiet: bool = False,
) -> dict:
    tun = tuning or DEFAULT_CH_TUNING
    OUT.mkdir(parents=True, exist_ok=True)
    TRAJ.mkdir(parents=True, exist_ok=True)
    meta = load_custom_meta(slot, max_sim_time=300000, wall_timeout=1e9)
    assert meta, f"slot {slot} meta missing"

    mod, env, agv_states, task_states, _ = load_scenario(
        Path(meta["task_csv"]), Path(meta["position_csv"]), force_reload=True
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
    j = json.loads(Path(meta["map_json"]).read_text(encoding="utf-8"))
    stations = {
        (int(p["x"]), int(p["y"]))
        for p in (j.get("pickups") or []) + (j.get("dropoffs") or [])
    }
    parks = _parking_bays(static, stations)

    pose: Dict[str, Pose] = {}
    for name, agv in agv_states.items():
        st = agv["state"] if isinstance(agv, dict) else agv.state
        pose[name] = (int(st[0]), int(st[1]), int(st[3]) if len(st) > 3 else 90)

    names = sorted(pose)
    steps_by = {n: [_hold(n, pose[n], 0)] for n in names}
    now = 0
    done = 0
    failed: List[str] = []
    t0 = time.perf_counter()
    order = sorted(queues.keys())

    # Active assignments: agv -> task dict + leg
    active: Dict[str, dict] = {}
    loaded: Dict[str, bool] = {n: False for n in names}
    dest: Dict[str, str] = {n: "" for n in names}
    tid: Dict[str, str] = {n: "" for n in names}
    assign_count: Dict[str, int] = defaultdict(int)

    print(
        f"[CH] SH slot={slot} agents={len(names)} tasks={total} "
        f"max_active={max_active} parks={len(parks)}",
        flush=True,
    )

    stall = 0
    max_stall = 80
    wave = 0
    wave_of: Dict[str, str] = _init_micro_waves(names)
    pair_hits: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_geom: Dict[Tuple[str, str], Tuple] = {}
    pair_same: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_lock: Dict[Tuple[str, str], str] = {}
    yield_until: Dict[str, int] = {}
    recent_parks: Dict[str, deque] = {
        n: deque(maxlen=tun.recent_parks_k) for n in names
    }
    consecutive_wait: Dict[str, int] = {n: 0 for n in names}
    deferred_assign: Dict[str, Tuple[str, str]] = {}
    coord = LlmCoordinator(
        enabled=use_llm,
        model=llm_model or DEFAULT_MODEL,
        cooldown=llm_cooldown,
        min_hits=1,
        scene=_llm_scene_brief(
            list(j.get("pickups") or []),
            list(j.get("dropoffs") or []),
            n_agv=len(names),
            max_active=max_active,
        ),
    )
    print(
        f"[CH] llm={use_llm} model={coord.model} cooldown={llm_cooldown} deadline={deadline}",
        flush=True,
    )

    while (any(queues.values()) or active) and stall < max_stall and now < deadline:
        wave += 1
        # Reveal tasks assigned on a previous unload frame
        for n, (nt, nd) in list(deferred_assign.items()):
            deferred_assign.pop(n, None)
            if n in active and active[n].get("leg") == "to_pickup":
                tid[n] = nt
                dest[n] = nd
        # --- assign free AGVs: each free AGV gets min solo-A* task immediately ---
        _fill_free_agvs_by_astar(
            names,
            pose=pose,
            queues=queues,
            order=order,
            static=static,
            active=active,
            max_active=max_active,
            tid=tid,
            dest=dest,
            loaded=loaded,
            assign_count=assign_count,
            now=now,
        )

        if not active:
            break

        # --- Phase A: solo A* toward current stage goal ---
        traj: Dict[str, List[Cell]] = {n: [pose[n][:2]] for n in names}
        plan_ok = True
        conf = None
        for agv, info in list(active.items()):
            task = info["task"]
            allow_agv = set(info.get("ends") or []) | {tuple(info["pickup"])}
            if info["leg"] == "to_pickup":
                stage_goal = info["pickup"]
            else:
                ends = info["ends"] or [info["pickup"]]
                stage_goal = min(
                    ends, key=lambda e: (_manh(pose[agv][:2], e), e)
                )
            rem = _astar_len(
                pose[agv][:2], stage_goal, static, allow=allow_agv
            )
            # O2: only nudge during yield (not Phase A — 1-step plans kill bulk T* commits)
            if info["leg"] == "to_pickup":
                cells = _solo_path_engine(
                    mod,
                    {
                        "agv": agv,
                        "task_id": task["task_id"],
                        "destination": task.get("destination") or "",
                        "priority": task.get("priority") or "Normal",
                    },
                    pose[agv],
                    info["pickup"],
                    info["ends"] or [info["pickup"]],
                    static_list,
                    now=now,
                    wall=wall,
                )
                if cells is None:
                    cells = _solo_to_goal(
                        mod,
                        agv,
                        pose[agv],
                        info["pickup"],
                        static_list,
                        now=now,
                        tid=str(task["task_id"]),
                        wall=wall,
                    )
            else:
                ends = info["ends"] or [info["pickup"]]
                goal_u = min(ends, key=lambda e: (_manh(pose[agv][:2], e), e))
                cells = _best_solo_path(
                    pose[agv],
                    goal_u,
                    static,
                    allow=set(ends),
                )
                if not cells:
                    cells = _solo_to_goal(
                        mod,
                        agv,
                        pose[agv],
                        goal_u,
                        static_list,
                        now=now,
                        tid=str(task["task_id"]),
                        dest=dest.get(agv, ""),
                        wall=wall,
                        allow=set(ends),
                    )
            if not cells:
                plan_ok = False
                print(
                    f"[CH] solo miss {agv} {task['task_id']} leg={info['leg']}",
                    flush=True,
                )
                continue
            traj[agv] = cells

        occupied_now = {pose[n][:2] for n in names}
        for n in names:
            if n in active:
                continue
            if _idle_hold_far(
                n, pose, active, traj, dist=tun.idle_hold_dist
            ):
                traj[n] = [pose[n][:2]]
                continue
            sg = _idle_staging_goal(
                pose[n][:2], queues, order, parks, occupied_now
            )
            path = _plan_toward_goal(
                mod,
                n,
                pose[n],
                sg,
                static_list,
                static,
                static | stations | (occupied_now - {pose[n][:2]}),
            )
            if len(path) > tun.idle_path_cap:
                path = path[: tun.idle_path_cap]
            traj[n] = path

        def _allow_of_name(n: str) -> Set[Cell]:
            info = active.get(n)
            if not info:
                return set()
            return set(info.get("ends") or []) | {tuple(info["pickup"])}

        # Event-driven replan: abort commit when any AGV needs a next-stage path
        needs_stage_plan = False

        def _request_stage_replan(reason: str, t: int) -> None:
            nonlocal needs_stage_plan
            if not needs_stage_plan:
                print(f"[CH] stage-replan @t={t}: {reason}", flush=True)
            needs_stage_plan = True

        def _hook_progress(name: str, cell: Cell, t: int) -> None:
            nonlocal done
            if name in deferred_assign:
                nt, nd = deferred_assign.pop(name)
                if name in active and active[name].get("leg") == "to_pickup":
                    tid[name] = nt
                    dest[name] = nd
            info = active.get(name)
            if not info:
                return
            task = info["task"]
            if (
                info["leg"] == "to_pickup"
                and cell == info["pickup"]
                and not loaded.get(name)
            ):
                loaded[name] = True
                dest[name] = str(task.get("destination") or "")
                tid[name] = str(task["task_id"])
                info["leg"] = "to_drop"
                print(f"[CH] pick {task['task_id']} by {name} t={t}", flush=True)
                # 取货完成 → 缺 to_drop 路径，打断 commit 立刻重规划
                _request_stage_replan(f"{name} needs to_drop", t)
            elif (
                info["leg"] == "to_drop"
                and cell in set(info["ends"])
                and loaded.get(name)
            ):
                loaded[name] = False
                dest[name] = ""
                tid[name] = ""
                done += 1
                active.pop(name, None)
                print(
                    f"[CH] done={done}/{total} drop {task['task_id']} "
                    f"by {name} t={t} wall={time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
                before = set(active)
                _fill_free_agvs_by_astar(
                    names,
                    pose=pose,
                    queues=queues,
                    order=order,
                    static=static,
                    active=active,
                    max_active=max_active,
                    tid=tid,
                    dest=dest,
                    loaded=loaded,
                    assign_count=assign_count,
                    now=t,
                )
                # Keep unload frame clear in traj; show new task from next second
                if name in active:
                    deferred_assign[name] = (
                        tid.get(name, ""),
                        dest.get(name, ""),
                    )
                    tid[name] = ""
                    dest[name] = ""
                newly = set(active) - before
                if newly or name not in active:
                    who = ",".join(sorted(newly)) if newly else name
                    _request_stage_replan(f"assign/path needed ({who})", t)

        def _abort_if_need_stage() -> bool:
            return needs_stage_plan

        if not any(len(traj[a]) > 1 for a in active):
            stall += 1
            timelines = {n: [pose[n], pose[n]] for n in names}
            now = _apply_timeline(
                steps_by,
                pose,
                timelines,
                now,
                loaded=loaded,
                dest=dest,
                tid=tid,
                until_idx=1,
                abort_check=_abort_if_need_stage,
            )
            _sync_wait_streaks(steps_by, names, consecutive_wait)
            continue

        timelines = {
            n: _expand_full_timeline(pose[n], traj[n]) for n in names
        }
        # Each replan: restart as micro-waves, then lossless-fuse on current traj
        wave_of = _init_micro_waves(names)
        movers_now = [
            n
            for n in names
            if n in active or len(traj.get(n) or []) > 1
        ]
        wave_of = _fuse_waves_no_loss(traj, wave_of, movers_now)
        conf = _first_conflict(
            {n: [p[:2] for p in timelines[n]] for n in names}
        )
        if conf is None:
            now = _apply_timeline(
                steps_by,
                pose,
                timelines,
                now,
                loaded=loaded,
                dest=dest,
                tid=tid,
                on_cell=_hook_progress,
                abort_check=_abort_if_need_stage,
            )
            _sync_wait_streaks(steps_by, names, consecutive_wait)
            stall = 0
            # 阶段切换已打断：下一循环立刻为缺路径的车重规划
            continue
        else:
            t_star, a_hit, b_hit, kind = conf
            pair = tuple(sorted((a_hit, b_hit)))
            geom = (pose[a_hit][:2], pose[b_hit][:2], kind)
            if pair_geom.get(pair) == geom:
                pair_same[pair] += 1
            else:
                pair_geom[pair] = geom
                pair_same[pair] = 1
                pair_lock.pop(pair, None)
            pair_hits[pair] += 1
            stuck_thresh = 2 if (kind == "swap" or t_star <= 1) else 4
            face_swap = kind == "swap" and t_star <= 1
            chronic = int(pair_hits[pair]) >= stuck_thresh or int(pair_same[pair]) >= 3
            commit_rel = max(0, t_star - 1)
            print(
                f"[CH] wave={wave} first_{kind} t*={t_star} "
                f"{a_hit}<->{b_hit} -> commit T*-1={commit_rel} @sim={now} "
                f"pair_hits={pair_hits[pair]}",
                flush=True,
            )
            if commit_rel > 0:
                now = _apply_timeline(
                    steps_by,
                    pose,
                    timelines,
                    now,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                    until_idx=commit_rel,
                    on_cell=_hook_progress,
                    abort_check=_abort_if_need_stage,
                )
                _sync_wait_streaks(steps_by, names, consecutive_wait)
                if needs_stage_plan:
                    stall = 0
                    continue

            involved = [a_hit, b_hit]
            frozen_agents: Set[str] = set(involved)

            # Same fused wave but now conflict → split hit pair first, then re-fuse others
            if wave_of.get(a_hit) == wave_of.get(b_hit):
                old_wid = wave_of[a_hit]
                members = _wave_members(wave_of, old_wid)
                # a keeps old id; b becomes its own micro-wave; peers re-attach losslessly
                wave_of[b_hit] = b_hit
                for m in members:
                    if m in (a_hit, b_hit):
                        continue
                    # Prefer attach to a if conflict-free with a's group, else b, else stay micro
                    if not _paths_conflict_pair(traj, m, a_hit):
                        wave_of[m] = wave_of[a_hit]
                    elif not _paths_conflict_pair(traj, m, b_hit):
                        wave_of[m] = wave_of[b_hit]
                    else:
                        wave_of[m] = m
                print(
                    f"[CH] wave-split {old_wid}: "
                    f"{a_hit}(sz={_wave_size(wave_of, wave_of[a_hit])}) vs "
                    f"{b_hit}(sz={_wave_size(wave_of, wave_of[b_hit])})",
                    flush=True,
                )

            # Conflict between waves: first try absorb more traffic into larger side
            wa = wave_of.get(a_hit, a_hit)
            wb = wave_of.get(b_hit, b_hit)
            if _wave_size(wave_of, wa) >= _wave_size(wave_of, wb):
                prefer_wid, other_wid = wa, wb
            else:
                prefer_wid, other_wid = wb, wa
            # Only absorb idle/staging cars not on the opposing wave
            absorb_cands = [
                n
                for n in names
                if n not in involved
                and n not in active  # idle / staging only — don't steal job cars
                and len(traj.get(n) or []) > 1
                and wave_of.get(n) != prefer_wid
                and wave_of.get(n) != other_wid
            ]
            before_sz = _wave_size(wave_of, prefer_wid)
            wave_of = _try_absorb_into_wave(
                traj, wave_of, prefer_wid, absorb_cands
            )
            after_sz = _wave_size(wave_of, prefer_wid)
            if after_sz > before_sz:
                print(
                    f"[CH] wave-absorb wid={prefer_wid} "
                    f"{before_sz}->{after_sz} "
                    f"(other={other_wid})",
                    flush=True,
                )

            def _bystanders_go(max_steps: int = BYSTANDER_MAX_STEPS) -> None:
                """Conflict group is resolving; everyone else keeps stage work."""
                nonlocal now

                def _by_goal(n: str) -> Cell:
                    if n in active:
                        return _goal_of(n)
                    return _idle_staging_goal(
                        pose[n][:2], queues, order, parks, {pose[m][:2] for m in names}
                    )

                def _by_allow(n: str) -> Set[Cell]:
                    return _allow_of(n)

                now = _advance_bystanders(
                    steps_by,
                    pose,
                    timelines,
                    names,
                    frozen_agents,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                    now=now,
                    on_cell=_hook_progress,
                    abort_check=_abort_if_need_stage,
                    max_steps=max_steps,
                    goal_of=_by_goal,
                    allow_of=_by_allow,
                    static=static,
                    stations=stations,
                )
                _sync_wait_streaks(steps_by, names, consecutive_wait)

            def _goal_of(n: str) -> Cell:
                return _true_goal(active.get(n), pose[n][:2])

            def _allow_of(n: str) -> Set[Cell]:
                info = active.get(n)
                if not info:
                    return set()
                return set(info.get("ends") or []) | {tuple(info["pickup"])}

            rem_map = {
                n: _astar_len(pose[n][:2], _goal_of(n), static, allow=_allow_of(n))
                for n in involved
            }
            skip_llm = (
                not use_llm
                or chronic
                or face_swap
                or int(pair_same[pair]) >= 3
            )
            if pair in pair_lock and pair_lock[pair] in involved:
                keeper = pair_lock[pair]
                yielders = [n for n in involved if n != keeper]
                print(
                    f"[CH] priority keeper={keeper} yield={yielders} "
                    f"src=pair_lock same={pair_same[pair]}",
                    flush=True,
                )
            elif chronic or face_swap:
                # Deadlock: larger fused wave keeps corridor; smaller yields
                keeper, yielders, wr = _pick_keeper_by_wave(
                    involved,
                    wave_of,
                    rem_map=rem_map,
                    active=active,
                    loaded=loaded,
                )
                if int(pair_same[pair]) >= 3:
                    pair_lock[pair] = keeper
                print(
                    f"[CH] priority keeper={keeper} yield={yielders} "
                    f"src={'face_swap_evac' if face_swap else 'chronic'}_wave "
                    f"{wr}",
                    flush=True,
                )
            else:
                # Soft conflict: also prefer larger wave before LLM / ranked
                keeper, yielders, wr = _pick_keeper_by_wave(
                    involved,
                    wave_of,
                    rem_map=rem_map,
                    active=active,
                    loaded=loaded,
                )
                snap = {
                    "t": now,
                    "deadline": deadline,
                    "done": done,
                    "total": total,
                    "kind": kind,
                    "t_star": t_star,
                    "involved": involved,
                    "active": list(active),
                    "skip_llm": skip_llm,
                    "same_pose_streak": int(pair_same[pair]),
                    "wave_reason": wr,
                    "agents": [
                        _llm_agent_row(
                            n,
                            pose=pose,
                            active=active,
                            loaded=loaded,
                            tid=tid,
                            dest=dest,
                            yield_until=yield_until,
                            now=now,
                            traj=traj,
                            static=static,
                            rem_override=rem_map.get(n),
                        )
                        for n in names
                    ],
                }
                if not skip_llm:
                    # Seed ranked with wave-size preference for LLM fallback
                    ranked_wave = sorted(
                        involved,
                        key=lambda n: (
                            0 if n == keeper else 1,
                            1 if now < int(yield_until.get(n, -1)) else 0,
                            0 if n in active else 1,
                            *_priority_key(
                                n,
                                loaded=loaded.get(n, False),
                                urgent=bool(
                                    (active.get(n) or {}).get("urgent")
                                ),
                                dist_goal=rem_map[n],
                            ),
                        ),
                    )
                    dec = coord.decide(
                        snap,
                        ranked_wave,
                        now=now,
                        pair_hits=int(pair_hits[pair]),
                    )
                    # Honor LLM only if it keeps the larger-wave agent as keeper
                    # when sizes differ; otherwise stick with wave_fuse.
                    k_sz = _wave_size(wave_of, wave_of.get(keeper, keeper))
                    d_sz = _wave_size(
                        wave_of, wave_of.get(dec.keeper, dec.keeper)
                    )
                    if dec.keeper in involved and d_sz >= k_sz:
                        keeper = dec.keeper
                        yielders = [n for n in involved if n != keeper]
                        for n, role in dec.roles.items():
                            if role.act in (
                                "YIELD_PARK",
                                "EVACUATE",
                                "HOLD_GATE",
                            ):
                                yield_until[n] = now + int(role.ttl)
                            elif role.act in ("PUSH", "RESUME"):
                                yield_until.pop(n, None)
                        print(
                            f"[CH] priority keeper={keeper} yield={yielders} "
                            f"src={dec.source}+wave {dec.reason}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[CH] priority keeper={keeper} yield={yielders} "
                            f"src=wave_fuse (llm overruled) {wr}",
                            flush=True,
                        )
                else:
                    print(
                        f"[CH] priority keeper={keeper} yield={yielders} "
                        f"src=wave_fuse {wr}",
                        flush=True,
                    )

            kp = traj.get(keeper) or [pose[keeper][:2]]
            corridor = set(kp[: max(8, min(len(kp), t_star + 6))])
            corridor.add(pose[keeper][:2])
            corridor.add(pose[a_hit][:2])
            corridor.add(pose[b_hit][:2])
            # O5: yielder must not block keeper imminent expanded cells
            kexp_cells: List[Cell] = [pose[keeper][:2]]
            for c in kp[1:4]:
                if _manh(kexp_cells[-1], c) == 1:
                    kexp_cells.append(c)
                else:
                    break
            if len(kexp_cells) >= 2:
                for p in _expand_full_timeline(pose[keeper], kexp_cells)[1:4]:
                    corridor.add((p[0], p[1]))

            # Smaller wave yields as a group: same-wave members ON corridor only
            k_wid = wave_of.get(keeper, keeper)
            y_wids = {wave_of.get(y, y) for y in yielders}
            for n in names:
                if n == keeper or n in yielders:
                    continue
                if wave_of.get(n) == k_wid:
                    continue
                if wave_of.get(n) in y_wids and pose[n][:2] in corridor:
                    yielders.append(n)
            if len(yielders) > 1:
                print(
                    f"[CH] wave-yield group={yielders} "
                    f"(keeper_wave={k_wid} size={_wave_size(wave_of, k_wid)})",
                    flush=True,
                )

            def _apply_yield_path(
                y: str,
                cells: List[Cell],
                label: str,
                *,
                park_cell: Optional[Cell] = None,
                leave_band: bool = False,
            ) -> None:
                nonlocal now, stall
                short = list(cells)
                if not leave_band:
                    short = _truncate_off_corridor(
                        cells, corridor, min_steps=1, max_steps=3
                    )
                else:
                    if short and short[0] != pose[y][:2]:
                        short = [pose[y][:2]] + short
                    if len(short) > 5:
                        short = short[:5]
                short = _clip_path_clear_of_agents(short, y, pose, names)
                if len(short) < 2:
                    return
                ytl = _expand_full_timeline(pose[y], short)
                occ = {pose[n][:2] for n in names if n != y}
                clean: List[Pose] = [ytl[0]]
                for p in ytl[1:]:
                    if (p[0], p[1]) in occ and (p[0], p[1]) != pose[y][:2]:
                        break
                    if not _poses_atomic(clean[-1], p):
                        break
                    clean.append(p)
                if len(clean) < 2:
                    return
                # Apply yield path as one block (safe); bystanders advance after,
                # not mid-path — mid-tick interleave caused vertex collisions.
                one_tl = {m: [pose[m]] for m in names}
                one_tl[y] = clean
                now = _apply_timeline(
                    steps_by,
                    pose,
                    one_tl,
                    now,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                    on_cell=_hook_progress,
                    abort_check=_abort_if_need_stage,
                )
                _bystanders_go(max_steps=max(2, len(clean) - 1))
                _sync_wait_streaks(steps_by, names, consecutive_wait)
                if park_cell is not None:
                    _note_park(
                        recent_parks, y, park_cell, maxlen=tun.recent_parks_k
                    )
                stall = 0
                rem = _astar_len(
                    pose[y][:2], _goal_of(y), static, allow=_allow_of(y)
                )
                print(
                    f"[CH] {y} {label} stop@{pose[y][:2]} remA*={rem}",
                    flush=True,
                )

            def _try_evacuate(y: str) -> bool:
                occupied_now = {pose[n][:2] for n in names}
                opp = pose[keeper][:2]
                path = _evacuate_path(
                    pose[y][:2],
                    corridor,
                    static,
                    occupied_now,
                    opponent=opp,
                    allow=_allow_of(y),
                    max_len=4,
                )
                if path and len(path) >= 2:
                    _apply_yield_path(y, path, "evacuate", leave_band=True)
                    return True
                return False

            need_evac = face_swap or chronic
            if need_evac:
                # Multi-AGV jam: expand yielders to anyone on corridor / egress
                before_y = list(yielders)
                yielders = _expand_yield_group(
                    keeper,
                    yielders,
                    pose=pose,
                    names=names,
                    corridor=corridor,
                    static=static,
                    max_extra=4,
                )
                if yielders != before_y:
                    print(
                        f"[CH] yield-group expand {before_y} -> {yielders}",
                        flush=True,
                    )
                frozen_agents.update(yielders)
                frozen_agents.add(keeper)
                ordered = _order_evacuators(
                    yielders,
                    keeper=keeper,
                    pose=pose,
                    corridor=corridor,
                )
                print(
                    f"[CH] stuck/swap group keeper={keeper} "
                    f"evac_order={ordered}",
                    flush=True,
                )
                occupied = {pose[n][:2] for n in names}
                before = {n: pose[n][:2] for n in ([keeper] + list(yielders))}
                moved_any = False
                # Up to 2 passes: clear outer blockers, then retry inner
                for _pass in range(2):
                    for y in ordered:
                        if y == keeper:
                            continue
                        if _try_evacuate(y):
                            occupied = {pose[n][:2] for n in names}
                            moved_any = True
                            continue
                        flee = _flee_any_step(
                            pose[y][:2],
                            static,
                            occupied - {pose[y][:2]},
                            avoid=corridor,
                            opponent=pose[keeper][:2],
                        )
                        if flee is not None:
                            _apply_yield_path(
                                y,
                                [pose[y][:2], flee],
                                "idle-flee",
                                leave_band=True,
                            )
                            occupied = {pose[n][:2] for n in names}
                            moved_any = True
                        elif _pass == 0:
                            # Pull in whoever blocks this yielder's egress
                            extra = _agents_blocking_egress(
                                y,
                                keeper=keeper,
                                pose=pose,
                                names=names,
                                corridor=corridor,
                                static=static,
                            )
                            for b in extra:
                                if b not in yielders and b != keeper:
                                    yielders.append(b)
                                    print(
                                        f"[CH] egress-blocker {b} "
                                        f"blocks {y} -> force evacuate",
                                        flush=True,
                                    )
                            ordered = _order_evacuators(
                                yielders,
                                keeper=keeper,
                                pose=pose,
                                corridor=corridor,
                            )
                        else:
                            rem = _astar_len(
                                pose[y][:2],
                                _goal_of(y),
                                static,
                                allow=_allow_of(y),
                            )
                            print(
                                f"[CH] {y} evacuate miss remA*={rem}",
                                flush=True,
                            )
                kg = _goal_of(keeper)
                blocked = static | stations | {
                    pose[m][:2] for m in names if m != keeper
                }
                kpos = pose[keeper][:2]
                knxt = _greedy_step(
                    kpos, kg, static, blocked, allow=_allow_of(keeper)
                )
                if knxt is not None and knxt in {pose[y][:2] for y in yielders}:
                    knxt = None
                if knxt is not None:
                    one_tl = {m: [pose[m]] for m in names}
                    one_tl[keeper] = _expand_full_timeline(
                        pose[keeper], [kpos, knxt]
                    )
                    now = _apply_timeline(
                        steps_by,
                        pose,
                        one_tl,
                        now,
                        loaded=loaded,
                        dest=dest,
                        tid=tid,
                        on_cell=_hook_progress,
                        abort_check=_abort_if_need_stage,
                    )
                    _sync_wait_streaks(steps_by, names, consecutive_wait)
                    _bystanders_go(max_steps=1)
                    print(
                        f"[CH] keeper-push {keeper} {knxt} remA*="
                        f"{_astar_len(pose[keeper][:2], kg, static, allow=_allow_of(keeper))}",
                        flush=True,
                    )
                    moved_any = True
                after = {
                    n: pose[n][:2]
                    for n in ([keeper] + list(yielders))
                }
                progressed = any(
                    before.get(n) != after.get(n) for n in before
                )
                if progressed:
                    pair_hits[pair] = 0
                    pair_same[pair] = 0
                    stall = 0
                else:
                    stall += 1
                    # only freeze the stuck group; others keep moving
                    hold_set = set([keeper] + list(yielders))
                    hold_tl = {n: [pose[n]] for n in names}
                    for n in hold_set:
                        hold_tl[n] = [pose[n], pose[n]]
                    now = _apply_timeline(
                        steps_by,
                        pose,
                        hold_tl,
                        now,
                        loaded=loaded,
                        dest=dest,
                        tid=tid,
                        until_idx=1,
                    )
                    _sync_wait_streaks(steps_by, names, consecutive_wait)
                _bystanders_go()
                if needs_stage_plan:
                    stall = 0
                continue

            occupied = {pose[n][:2] for n in names}
            moved_yield = False
            for y in yielders:
                goal = _goal_of(y)
                here = pose[y][:2]
                idle_y = y not in active
                if idle_y and _try_evacuate(y):
                    occupied = {pose[n][:2] for n in names}
                    moved_yield = True
                    continue
                if here not in corridor:
                    blocked = static | stations | {
                        pose[x][:2] for x in names if x != y
                    }
                    nxt = _greedy_step(
                        here,
                        goal,
                        static,
                        blocked,
                        allow=_allow_of(y),
                        avoid=corridor,
                    )
                    if nxt is not None:
                        _apply_yield_path(y, [here, nxt], "resume-step")
                        occupied = {pose[n][:2] for n in names}
                        moved_yield = True
                        continue
                picked = _pick_resume_park_with_taboo(
                    here,
                    goal,
                    parks,
                    static,
                    occupied=occupied,
                    forbidden=corridor,
                    allow=_allow_of(y),
                    taboo=_park_taboo(recent_parks, y),
                )
                if picked:
                    park, path = picked
                    _apply_yield_path(
                        y, path, f"yield->park {park}", park_cell=park
                    )
                    occupied = {pose[n][:2] for n in names}
                    moved_yield = True
                    continue
                if _try_evacuate(y):
                    occupied = {pose[n][:2] for n in names}
                    moved_yield = True
                    continue
                blocked = static | stations | {
                    pose[x][:2] for x in names if x != y
                }
                nxt = _greedy_step(
                    here, goal, static, blocked, allow=_allow_of(y), avoid=set()
                )
                if nxt is None:
                    nxt = _greedy_step(
                        here, goal, static, blocked, allow=_allow_of(y)
                    )
                if nxt is not None:
                    _apply_yield_path(y, [here, nxt], f"yield-step {nxt}")
                    occupied = {pose[n][:2] for n in names}
                    moved_yield = True
                else:
                    flee = _flee_any_step(
                        here,
                        static,
                        occupied - {here},
                        avoid=corridor,
                        opponent=pose[keeper][:2],
                    )
                    if flee is not None:
                        _apply_yield_path(y, [here, flee], "idle-flee")
                        occupied = {pose[n][:2] for n in names}
                        moved_yield = True
                    else:
                        rem = _astar_len(
                            here, goal, static, allow=_allow_of(y)
                        )
                        print(
                            f"[CH] {y} yield miss -> wait (remA*={rem})",
                            flush=True,
                        )
                        stall += 1

            kg = _goal_of(keeper)
            blocked = static | stations | {
                pose[x][:2] for x in names if x != keeper
            }
            kpos = pose[keeper][:2]
            knxt = _greedy_step(
                kpos, kg, static, blocked, allow=_allow_of(keeper)
            )
            if knxt is not None and knxt in {pose[y][:2] for y in yielders}:
                knxt = None
            if knxt is not None:
                one_tl = {m: [pose[m]] for m in names}
                one_tl[keeper] = _expand_full_timeline(
                    pose[keeper], [kpos, knxt]
                )
                now = _apply_timeline(
                    steps_by,
                    pose,
                    one_tl,
                    now,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                    on_cell=_hook_progress,
                    abort_check=_abort_if_need_stage,
                )
                _sync_wait_streaks(steps_by, names, consecutive_wait)
                stall = 0
                print(f"[CH] keeper {keeper} resume-step {knxt}", flush=True)
                _bystanders_go(max_steps=1)
            elif not moved_yield:
                stall += 1
                hold_tl = {n: [pose[n]] for n in names}
                for n in involved:
                    hold_tl[n] = [pose[n], pose[n]]
                now = _apply_timeline(
                    steps_by,
                    pose,
                    hold_tl,
                    now,
                    loaded=loaded,
                    dest=dest,
                    tid=tid,
                    until_idx=1,
                )
                _sync_wait_streaks(steps_by, names, consecutive_wait)
                _bystanders_go(max_steps=1)

            # Catch-up: bystanders may still have leftover stage steps
            _bystanders_go(max_steps=BYSTANDER_MAX_STEPS)
            if needs_stage_plan:
                stall = 0
                continue

        for agv, info in list(active.items()):
            task = info["task"]
            pos_xy = pose[agv][:2]
            if (
                info["leg"] == "to_pickup"
                and pos_xy == info["pickup"]
                and not loaded[agv]
            ):
                now += 1
                loaded[agv] = True
                dest[agv] = str(task.get("destination") or "")
                tid[agv] = str(task["task_id"])
                info["leg"] = "to_drop"
                for n in names:
                    steps_by[n].append(
                        _hold(
                            n, pose[n], now, loaded=loaded[n], dest=dest[n], tid=tid[n]
                        )
                    )
                stall = 0
                print(f"[CH] pick {task['task_id']} by {agv} t={now}", flush=True)
            elif (
                info["leg"] == "to_drop"
                and pos_xy in set(info["ends"])
                and loaded[agv]
            ):
                now += 1
                loaded[agv] = False
                dest[agv] = ""
                tid[agv] = ""
                done += 1
                active.pop(agv, None)
                for n in names:
                    steps_by[n].append(
                        _hold(
                            n, pose[n], now, loaded=loaded[n], dest=dest[n], tid=tid[n]
                        )
                    )
                stall = 0
                print(
                    f"[CH] done={done}/{total} drop {task['task_id']} "
                    f"by {agv} t={now} wall={time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
                _fill_free_agvs_by_astar(
                    names,
                    pose=pose,
                    queues=queues,
                    order=order,
                    static=static,
                    active=active,
                    max_active=max_active,
                    tid=tid,
                    dest=dest,
                    loaded=loaded,
                    assign_count=assign_count,
                    now=now,
                )
                if agv in active:
                    deferred_assign[agv] = (
                        tid.get(agv, ""),
                        dest.get(agv, ""),
                    )
                    tid[agv] = ""
                    dest[agv] = ""
                # Same-t reassign must not rewrite the unload traj row already
                # appended above; blank tid/dest until the next written second.
                if agv in active:
                    # next apply/_hold will show the new task
                    pass

        if not plan_ok and conf is None:
            stall += 1

        if wave % 20 == 0:
            print(
                f"[CH] wave={wave} active={len(active)} done={done}/{total} "
                f"t={now} stall={stall}",
                flush=True,
            )

    # dense fill (first write wins on duplicate t unless later is a true unload)
    for name in steps_by:
        by: Dict[int, dict] = {}
        for s in steps_by[name]:
            t = int(s["timestamp"])
            if t not in by:
                by[t] = s
                continue
            prev = by[t]
            prev_loaded = str(prev.get("loaded", "")).lower() in (
                "true",
                "1",
                "yes",
            )
            new_loaded = str(s.get("loaded", "")).lower() in (
                "true",
                "1",
                "yes",
            )
            prev_tid = str(prev.get("task-id") or "").strip()
            new_tid = str(s.get("task-id") or "").strip()
            # unload-clear must not be overwritten by same-t reassign
            if (not prev_loaded and not prev_tid) and new_tid and not new_loaded:
                continue
            if prev_loaded and (not new_loaded) and not new_tid:
                by[t] = s
                continue
            by[t] = s
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

    sid = str(meta["id"])
    traj_path = TRAJ / f"{sid}_conflict_horizon_k{max_active}.csv"
    rows = []
    for n in sorted(steps_by):
        rows.extend(steps_by[n])
    rows.sort(key=lambda r: (int(r["timestamp"]), r["name"]))
    with traj_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRAJ_HEADER})

    val = validate_hybrid_trajectory(meta, traj_path)
    iss = val.get("issues") or {}
    rep = {
        "scenario_id": sid,
        "method": "conflict_horizon_solo_astar_llm_yield" if use_llm else "conflict_horizon_solo_astar_priority_yield",
        "max_active": max_active,
        "deadline": deadline,
        "llm": coord.stats(),
        "tasks_total": total,
        "tasks_completed": done,
        "tasks_failed": failed,
        "completion_ratio": round(done / max(1, total), 4),
        "sim_time": now,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "trajectory": str(traj_path),
        "validate_ok": bool(val.get("ok")),
        "validate_summary": format_validation_summary(val),
        "n_collisions": int(iss.get("n_collisions") or 0),
        "n_swaps": int(iss.get("n_swaps") or 0),
        "n_hard_wall": int(iss.get("n_hard_wall") or 0),
        "n_illegal_motion": int(iss.get("n_illegal_motion") or 0),
        "n_fifo": int(iss.get("n_fifo_violations") or 0),
        "stall_exit": stall >= max_stall,
        "deadline_exit": now >= deadline,
        "within_deadline": bool(done >= total and now <= deadline),
        "tuning": tun.to_dict(),
    }
    if use_llm:
        (OUT / f"{sid}_llm_coord.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in coord.log),
            encoding="utf-8",
        )
    if not quiet:
        (OUT / f"{sid}_conflict_horizon.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(rep, indent=2, ensure_ascii=False), flush=True)
    return rep


def _tuning_from_namespace(args: argparse.Namespace) -> CHTuning:
    base = DEFAULT_CH_TUNING
    overrides: dict = {}
    for field in dataclasses.fields(CHTuning):
        val = getattr(args, field.name, None)
        if val is not None:
            overrides[field.name] = int(val)
    return dataclasses.replace(base, **overrides) if overrides else base


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", type=int, default=3)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--max-active", type=int, default=8)
    ap.add_argument("--wall", type=float, default=12.0)
    ap.add_argument("--deadline", type=int, default=2000)
    ap.add_argument("--llm", action="store_true", help="Use local Ollama for yield/resume roles")
    ap.add_argument("--llm-model", default="", help="Local Ollama model (default deepseek-r1:latest)")
    ap.add_argument("--llm-cooldown", type=int, default=8)
    ap.add_argument(
        "--recent-parks-k",
        type=int,
        default=None,
        help="O1 parking taboo window (default 6)",
    )
    ap.add_argument(
        "--idle-path-cap",
        type=int,
        default=None,
        help="Max idle AGV path cells per wave (default 8)",
    )
    ap.add_argument(
        "--dispatch-lambda",
        type=int,
        default=None,
        help="O7 pickup congestion penalty (default 1)",
    )
    ap.add_argument(
        "--staging-max-d",
        type=int,
        default=None,
        help="Max manhattan for idle pickup staging (default 16)",
    )
    ap.add_argument(
        "--idle-hold-dist",
        type=int,
        default=None,
        help="Hold idle AGV when farther than this from active (default 10)",
    )
    args = ap.parse_args()
    rep = solve_conflict_horizon(
        args.slot,
        max_tasks=args.max_tasks,
        max_active=args.max_active,
        wall=args.wall,
        deadline=args.deadline,
        use_llm=bool(args.llm),
        llm_model=args.llm_model,
        llm_cooldown=args.llm_cooldown,
        tuning=_tuning_from_namespace(args),
    )
    ok = (
        bool(rep.get("validate_ok"))
        and float(rep.get("completion_ratio") or 0) >= 0.999
        and not rep.get("tasks_failed")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
