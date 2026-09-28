"""Keep an overflow unloader parked, still loaded, until a later delivery.

A dropoff only has as many parallel stands as approach cells reachable from the
road. The extra carrier stays out of the current joint plan. They are not
counted done, and the next wave does not clear their cargo. One waiting
carrier may join a later delivery, onto a stand nobody is standing on.
A carrier already on their own stand is unloaded there. An empty car left on
an unload stand, because distance assignment picked someone else, is driven
to a pickup stand in its own plan, before the delivery joint. That plan
failing does not abort the run. A pickup whose only door is blocked is
requeued instead of aborting. An empty car blocking the only door of a
delivery stand steps one cell aside.
"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]
_NBR = ((1, 0), (-1, 0), (0, 1), (0, -1))
_ANCHORS = ((6, 1), (10, 10), (2, 2), (19, 2), (10, 16))

_DROPOFFS: Set[Cell] = set()
_PICKUPS: Set[Cell] = set()
_DROP_NAME: Dict[Cell, str] = {}
_PICK_NAME: Dict[Cell, str] = {}
# agv -> cargo still on the car, waiting for a free stand of ``station``
_STICKY: Dict[str, Dict[str, object]] = {}
# Pickup whose stand is walled off by a waiting car. Do not load that task.
_SKIP_LOAD: Set[str] = set()
_ORIG: Dict[str, object] = {}
_OVERLAP_PENDING = False


def set_stations_from_map(map_json: Path) -> None:
    data = json.loads(Path(map_json).read_text(encoding="utf-8"))
    drops: Set[Cell] = set()
    picks: Set[Cell] = set()
    names: Dict[Cell, str] = {}
    pick_names: Dict[Cell, str] = {}
    for p in data.get("pickups") or []:
        cell = (int(p["x"]), int(p["y"]))
        picks.add(cell)
        if p.get("name"):
            pick_names[cell] = str(p["name"])
    for p in data.get("dropoffs") or []:
        cell = (int(p["x"]), int(p["y"]))
        drops.add(cell)
        if p.get("name"):
            names[cell] = str(p["name"])
    global _DROPOFFS, _PICKUPS, _DROP_NAME, _PICK_NAME
    _DROPOFFS = drops
    _PICKUPS = picks
    _DROP_NAME = names
    _PICK_NAME = pick_names


def _in_grid(c: Cell) -> bool:
    return 1 <= c[0] <= 20 and 1 <= c[1] <= 20


def _bfs(start: Cell, goal: Cell, blocked: Set[Cell]) -> Optional[int]:
    if start == goal:
        return 0
    from collections import deque

    q = deque([(start, 0)])
    seen = {start}
    while q:
        (x, y), dist = q.popleft()
        for dx, dy in _NBR:
            n = (x + dx, y + dy)
            if n in seen or not _in_grid(n):
                continue
            if n in blocked and n != goal:
                continue
            if n == goal:
                return dist + 1
            seen.add(n)
            q.append((n, dist + 1))
    return None


def _dropoff_of(cell: Cell) -> Optional[Cell]:
    for dx, dy in _NBR:
        s = (cell[0] + dx, cell[1] + dy)
        if s in _DROPOFFS:
            return s
    return None


def _road_pads(station: Cell, blocked: Set[Cell]) -> Set[Cell]:
    """Approach cells of ``station`` that the road can actually reach."""
    found: Set[Cell] = set()
    for anchor in _ANCHORS:
        if not _in_grid(anchor) or anchor in blocked:
            continue
        pads = {
            (station[0] + dx, station[1] + dy)
            for dx, dy in _NBR
            if _in_grid((station[0] + dx, station[1] + dy))
            and (station[0] + dx, station[1] + dy) not in blocked
            and _bfs(anchor, (station[0] + dx, station[1] + dy), blocked) is not None
        }
        if pads:
            found |= pads
    return found


def _solver_frame():
    f = sys._getframe(1)
    for _ in range(12):
        if f is None:
            return None
        if f.f_code.co_name == "solve_ecbs":
            return f
        f = f.f_back
    return None


def _restore_cargo(loaded, dest, tid) -> None:
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


def _remember_cargo(agv: str, loaded, dest, tid, station: Optional[Cell]) -> None:
    cargo = _STICKY.get(agv) or {"dest": "", "tid": "", "station": station}
    if station is not None:
        cargo["station"] = station
    if isinstance(loaded, dict) and loaded.get(agv) and isinstance(tid, dict):
        tid_s = str(tid.get(agv) or "")
        if tid_s:
            cargo["tid"] = tid_s
            cargo["dest"] = str(dest.get(agv) or "") if isinstance(dest, dict) else ""
    _STICKY[agv] = cargo


def _detach_new_tasks() -> None:
    """Put tasks just assigned to a waiting carrier back on the queue."""
    frame = _solver_frame()
    if frame is None:
        return
    loc = frame.f_locals
    assigned = loc.get("assigned")
    queues = loc.get("queues")
    order = loc.get("order")
    if not isinstance(assigned, dict):
        return
    for agv in list(_STICKY):
        task = assigned.get(agv)
        if not isinstance(task, dict):
            continue
        assigned.pop(agv, None)
        tid_s = str(task.get("task_id") or "")
        st = tid_s.rsplit("-", 1)[0] if "-" in tid_s else ""
        if st and isinstance(queues, dict):
            queues.setdefault(st, []).insert(0, task)
            if isinstance(order, list) and st not in order:
                order.append(st)
        print(
            f"[PIPE] unload-keep {agv} requeue {tid_s} "
            f"(still carrying {_STICKY[agv].get('tid')})",
            flush=True,
        )


def _requeue_assigned(agv: str) -> bool:
    """Put one just-assigned task back. Safe only before the load loop."""
    frame = _solver_frame()
    if frame is None:
        return False
    loc = frame.f_locals
    assigned = loc.get("assigned")
    queues = loc.get("queues")
    order = loc.get("order")
    if not isinstance(assigned, dict):
        return False
    task = assigned.get(agv)
    if not isinstance(task, dict):
        return False
    assigned.pop(agv, None)
    tid_s = str(task.get("task_id") or "")
    st = tid_s.rsplit("-", 1)[0] if "-" in tid_s else ""
    if st and isinstance(queues, dict):
        queues.setdefault(st, []).insert(0, task)
        if isinstance(order, list) and st not in order:
            order.append(st)
    return True


def _on_own_pad(agv: str, cell: Cell, blocked: Set[Cell]) -> bool:
    station = _STICKY.get(agv, {}).get("station")
    if not isinstance(station, tuple):
        station = _dropoff_of(cell)
    if not isinstance(station, tuple):
        return False
    return cell in _road_pads(station, blocked)


def _dist(start: Cell, goal: Cell, blocked: Set[Cell]) -> int:
    """Road distance. Zero is a real distance; unreachable is large."""
    dist = _bfs(start, goal, blocked)
    if dist is None:
        return 10**9
    return dist


def _is_delivery_call(goals) -> bool:
    """Pickup goals sit beside pickups. Only a dropoff approach is a delivery."""
    if not isinstance(goals, dict):
        return False
    for goal in goals.values():
        if isinstance(goal, tuple) and _dropoff_of(goal) is not None:
            return True
    return False


def _bodies(pose) -> Dict[Cell, str]:
    standing: Dict[Cell, str] = {}
    if not isinstance(pose, dict):
        return standing
    for agv, p in pose.items():
        if isinstance(p, tuple) and len(p) >= 2:
            standing[(int(p[0]), int(p[1]))] = str(agv)
    return standing


def _seal_unload(steps_by, agv: str, cell: Cell) -> bool:
    """Clear cargo on the last written row, which must already be this pad.

    The overlap writer pads a missing timeline with the old cargo. Flipping
    that last row keeps the True-to-False step on the stand, before any
    later leave plan can walk the car off it.
    """
    if not isinstance(steps_by, dict):
        return False
    seq = steps_by.get(agv)
    if not seq:
        return False
    last = seq[-1]
    try:
        here = (int(last["X"]), int(last["Y"]))
    except (KeyError, TypeError, ValueError):
        return False
    if here != cell:
        return False
    last["loaded"] = "FALSE"
    last["destination"] = ""
    last["task-id"] = ""
    return True


def _door_blocker(
    start: Cell,
    goal: Cell,
    standing: Dict[Cell, str],
    keep: Set[str],
    blocked: Set[Cell],
    skip: Set[str],
) -> Optional[str]:
    """Idle whose departure opens ``goal``. Loaded cars are not eligible."""
    others = {
        cell: who
        for cell, who in standing.items()
        if who not in keep and who not in _STICKY and who not in skip
    }
    obstacles = blocked | set(others)
    opened = _bfs(start, goal, obstacles) is not None
    occupant = others.get(goal)
    if opened and occupant is None:
        return None
    ranked: List[Tuple[int, str, Cell]] = []
    for cell, who in others.items():
        if opened and who != occupant:
            continue
        if not opened and _bfs(start, goal, obstacles - {cell}) is None:
            continue
        ranked.append((abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]), who, cell))
    if not ranked:
        return None
    ranked.sort()
    return ranked[0][1]


def _yield_step(cell: Cell, goal: Cell, blocked: Set[Cell], occupied: Set[Cell]) -> Optional[Cell]:
    """One step off a door, away from the stand the delivery car needs."""
    opts: List[Tuple[int, Cell]] = []
    for dx, dy in _NBR:
        nxt = (cell[0] + dx, cell[1] + dy)
        if not _in_grid(nxt) or nxt in blocked or nxt in occupied or nxt == goal or nxt == cell:
            continue
        away = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
        opts.append((away, nxt))
    if not opts:
        return None
    opts.sort(key=lambda item: (-item[0], item[1]))
    return opts[0][1]


def _as_pickup_station(cell: Cell) -> Optional[Cell]:
    """Map a queued pickup point back to the station, not the stand beside it."""
    if cell in _PICKUPS:
        return cell
    for dx, dy in _NBR:
        station = (cell[0] + dx, cell[1] + dy)
        if station in _PICKUPS:
            return station
    return None


def _queued_pickups(loc) -> List[Cell]:
    """Pickup stations of tasks still waiting, front of each queue first."""
    queues = loc.get("queues") if isinstance(loc, dict) else None
    order = loc.get("order") if isinstance(loc, dict) else None
    if not isinstance(queues, dict):
        return []
    keys = list(order) if isinstance(order, list) and order else list(queues)
    found: List[Cell] = []
    for st in keys:
        bucket = queues.get(st) or []
        if not bucket or not isinstance(bucket[0], dict):
            continue
        pt = bucket[0].get("pickup_point")
        if not pt or len(pt) < 2:
            continue
        cell = (int(pt[0]), int(pt[1]))
        station = _as_pickup_station(cell)
        if station is not None and station not in found:
            found.append(station)
    return found


def _nearest_stand(start: Cell, pads: List[Cell], reserved: Set[Cell], blocked: Set[Cell]) -> Optional[Cell]:
    """Closest free stand at least two steps away, so a pocket door is not a goal."""
    best: Optional[Tuple[int, Cell]] = None
    obstacles = set(blocked) | (set(reserved) - {start})
    for pad in pads:
        if pad in reserved or pad == start:
            continue
        dist = _dist(start, pad, obstacles)
        if dist < 2 or dist >= 10**9:
            continue
        if best is None or (dist, pad) < best:
            best = (dist, pad)
    return None if best is None else best[1]


def _release_finished(pose, movers: Set[str], blocked: Set[Cell]) -> None:
    """Unload a car already standing on its own stand. Leave the counter alone."""
    frame = _solver_frame()
    if frame is None or not isinstance(pose, dict):
        return
    loc = frame.f_locals
    loaded = loc.get("loaded")
    dest = loc.get("dest")
    tid = loc.get("tid")
    steps_by = loc.get("steps_by")
    for agv in list(pose):
        if agv in movers:
            continue
        raw = pose.get(agv)
        if not isinstance(raw, tuple) or len(raw) < 2:
            continue
        cell = (int(raw[0]), int(raw[1]))
        cargo = _STICKY.get(agv)
        station = cargo.get("station") if isinstance(cargo, dict) else None
        if not isinstance(station, tuple):
            station = _dropoff_of(cell)
        if not isinstance(station, tuple) or cell not in _road_pads(station, blocked):
            continue
        loaded_flag = bool(isinstance(loaded, dict) and loaded.get(agv))
        if cargo is None and not loaded_flag:
            continue
        if cargo is None and isinstance(dest, dict):
            want = _DROP_NAME.get(station, "")
            have = dest.get(agv)
            if have not in ("", None) and have != want:
                continue
        if not _seal_unload(steps_by, agv, cell):
            continue
        tid_s = ""
        if isinstance(cargo, dict):
            tid_s = str(cargo.get("tid") or "")
        elif isinstance(tid, dict):
            tid_s = str(tid.get(agv) or "")
        if isinstance(loaded, dict):
            loaded[agv] = False
        if isinstance(dest, dict):
            dest[agv] = ""
        if isinstance(tid, dict):
            tid[agv] = ""
        _STICKY.pop(agv, None)
        print(
            f"[PIPE] unload-done {agv} at={cell} tid={tid_s or '?'} "
            f"(on stand, cargo cleared)",
            flush=True,
        )


def _assign_pickup_goals(pose, movers: Set[str], goals, blocked: Set[Cell], loaded) -> Dict[str, Cell]:
    """Empty cars sitting on unload stands, not chosen this wave, go to a pickup."""
    if not isinstance(pose, dict):
        return {}
    candidates: List[str] = []
    for agv, raw in pose.items():
        if agv in movers or agv in _STICKY:
            continue
        if isinstance(loaded, dict) and loaded.get(agv):
            continue
        if not isinstance(raw, tuple) or len(raw) < 2:
            continue
        cell = (int(raw[0]), int(raw[1]))
        station = _dropoff_of(cell)
        if station is None or cell not in _road_pads(station, blocked):
            continue
        candidates.append(agv)
    if not candidates:
        return {}
    frame = _solver_frame()
    loc = frame.f_locals if frame is not None else {}
    preferred = _queued_pickups(loc)
    pref_set = set(preferred)
    pref_pads: List[Cell] = []
    other_pads: List[Cell] = []
    seen: Set[Cell] = set()
    stations = preferred + [pk for pk in sorted(_PICKUPS) if pk not in pref_set]
    for pk in stations:
        if pk not in _PICKUPS:
            continue
        bucket = pref_pads if pk in pref_set else other_pads
        for pad in sorted(_road_pads(pk, blocked)):
            if pad in seen or _dropoff_of(pad) is not None:
                continue
            if abs(pad[0] - pk[0]) + abs(pad[1] - pk[1]) != 1:
                continue
            seen.add(pad)
            bucket.append(pad)
    reserved: Set[Cell] = set()
    for agv, raw in pose.items():
        if agv in candidates or not isinstance(raw, tuple) or len(raw) < 2:
            continue
        reserved.add((int(raw[0]), int(raw[1])))
    if isinstance(goals, dict):
        for agv in movers:
            goal = goals.get(agv)
            if isinstance(goal, tuple) and len(goal) >= 2:
                reserved.add((int(goal[0]), int(goal[1])))
    out: Dict[str, Cell] = {}
    ordered = sorted(
        candidates,
        key=lambda agv: (int(pose[agv][0]), int(pose[agv][1]), agv),
    )
    for agv in ordered:
        start = (int(pose[agv][0]), int(pose[agv][1]))
        choice = _nearest_stand(start, pref_pads, reserved, blocked)
        if choice is None:
            choice = _nearest_stand(start, other_pads, reserved, blocked)
        if choice is None:
            print(
                f"[PIPE] pad-vacate-retry {agv} off={start} (no free pickup stand)",
                flush=True,
            )
            continue
        out[agv] = choice
        reserved.add(choice)
        station = next((pk for pk in _PICKUPS if choice in _road_pads(pk, blocked)), None)
        label = _PICK_NAME.get(station, "") if isinstance(station, tuple) else ""
        print(
            f"[PIPE] pad-vacate {agv} off={start} to={choice}"
            + (f" pickup={label}" if label else ""),
            flush=True,
        )
    return out


def _sequence_vacate(parts: List[Tuple[str, list]], snap: Dict[str, tuple]) -> Dict[str, list]:
    """Play one leave after another so a failed group plan still clears the stands."""
    lengths = [len(seq) for _agv, seq in parts if seq]
    total = sum(lengths)
    out: Dict[str, list] = {}
    cursor = 0
    for agv, seq in parts:
        if not seq:
            continue
        before = cursor
        after = total - cursor - len(seq)
        start = snap[agv]
        end = seq[-1]
        out[agv] = [start] * before + list(seq) + [end] * after
        cursor += len(seq)
    return out


def _ends_on(seq, goal: Cell) -> bool:
    if not seq:
        return False
    last = seq[-1]
    return isinstance(last, tuple) and len(last) >= 2 and (int(last[0]), int(last[1])) == goal


def _drive_vacate(pose, goals: Dict[str, Cell], static, kwargs) -> Optional[dict]:
    """Separate joint for empty cars. Failure returns None and does not abort."""
    if not goals or "joint" not in _ORIG:
        return None
    scale = kwargs.get("tmax_scale", 4.0)
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = 4.0
    if scale < 4.0:
        scale = 4.0
    joint = _ORIG["joint"]

    def _planned(want: Dict[str, Cell], world):
        for outer in (True, False):
            tl = joint(
                world,
                dict(want),
                set(want),
                static,
                tmax_scale=scale,
                prefer_outer_ring=outer,
            )
            if isinstance(tl, dict) and all(_ends_on(tl.get(agv), goal) for agv, goal in want.items()):
                return {agv: list(tl[agv]) for agv in want}
        return None

    group = _planned(goals, pose)
    if group is not None:
        return group
    snap = {agv: pose[agv] for agv in goals if agv in pose}
    parts: List[Tuple[str, list]] = []
    try:
        for agv, goal in goals.items():
            one = _planned({agv: goal}, pose)
            seq = None if one is None else one.get(agv)
            if not seq:
                print(
                    f"[PIPE] pad-vacate-retry {agv} to={goal} (plan failed, stays for next tick)",
                    flush=True,
                )
                continue
            parts.append((agv, list(seq)))
            pose[agv] = seq[-1]
        if not parts:
            return None
        return _sequence_vacate(parts, snap)
    finally:
        for agv, raw in snap.items():
            pose[agv] = raw


def _commit_vacate(pose, vac_tl: dict) -> bool:
    """Write the leave into the trajectory now, so the next joint sees free stands.

    ``now`` is a plain int in the champion frame. On Python 3.12 a locals write
    only sticks if it is copied back onto the fast local.
    """
    frame = _solver_frame()
    if frame is None or not vac_tl or "hold" not in _ORIG:
        return False
    loc = frame.f_locals
    steps_by = loc.get("steps_by")
    now = loc.get("now")
    if not isinstance(steps_by, dict) or not isinstance(now, int):
        return False
    loaded = loc.get("loaded") if isinstance(loc.get("loaded"), dict) else {}
    dest = loc.get("dest") if isinstance(loc.get("dest"), dict) else {}
    tid = loc.get("tid") if isinstance(loc.get("tid"), dict) else {}
    names = list(pose.keys())
    if any(n not in steps_by for n in names):
        return False
    width = max((len(seq) for seq in vac_tl.values() if seq), default=0)
    if width <= 0:
        return False
    hold = _ORIG["hold"]
    for t_i in range(width):
        now += 1
        for n in names:
            seq = vac_tl.get(n)
            if seq and t_i < len(seq):
                pose[n] = seq[t_i]
            elif seq:
                pose[n] = seq[-1]
            steps_by[n].append(
                hold(
                    n,
                    pose[n],
                    now,
                    loaded=bool(loaded.get(n, False)),
                    dest=str(dest.get(n, "") or ""),
                    tid=str(tid.get(n, "") or ""),
                )
            )
    loc["now"] = now
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))
    return True


def install(mod) -> None:
    """Patch the loaded champion so overflow unloaders keep their cargo."""
    if _ORIG:
        return
    _ORIG["joint"] = mod._plan_turn_aware_joint
    _ORIG["cargo"] = mod._apply_pose_timelines_cargo
    _ORIG["overlap"] = mod._build_overlap_delivery_timelines
    _ORIG["pickup"] = mod._pickup_load_agv
    _ORIG["apply"] = mod._apply_pose_timelines
    _ORIG["paths"] = mod._apply_paths
    _ORIG["goals"] = mod._distinct_goals_near
    _ORIG["clear"] = mod._clear_blocking_idles
    _ORIG["hold"] = mod._hold

    def _plan_turn_aware_joint(pose, goals, movers, static, *args, **kwargs):
        movers = set(movers or ())
        blocked = set(static) | set(_DROPOFFS) | set(_PICKUPS)
        _release_finished(pose, movers, blocked)
        loaded_now = None
        fr0 = _solver_frame()
        if fr0 is not None and isinstance(fr0.f_locals.get("loaded"), dict):
            loaded_now = fr0.f_locals["loaded"]
        vac_goals = _assign_pickup_goals(pose, movers, goals, blocked, loaded_now)
        vac_tl = _drive_vacate(pose, vac_goals, static, kwargs)
        if isinstance(vac_tl, dict) and vac_tl:
            if _commit_vacate(pose, vac_tl):
                print(
                    f"[PIPE] pad-vacate-ok n={len(vac_tl)} "
                    + " ".join(f"{agv}={pose[agv][:2]}" for agv in sorted(vac_tl)),
                    flush=True,
                )
            else:
                print(
                    "[PIPE] pad-vacate-retry (could not write the leave, cars stay)",
                    flush=True,
                )

        hold: Set[str] = set()
        keep: Set[str] = set()
        used: Dict[Cell, Set[Cell]] = {}
        standing = _bodies(pose)
        frame = _solver_frame()
        caller_goals = frame.f_locals.get("goals") if frame is not None else None
        pickup_call = goals is caller_goals and not _is_delivery_call(goals)
        _SKIP_LOAD.clear()
        for cell, who in list(standing.items()):
            station = _dropoff_of(cell)
            if station is None or who in movers:
                continue
            if cell not in _road_pads(station, blocked):
                continue
            used.setdefault(station, set()).add(cell)

        ranked = sorted(movers, key=lambda a: a)
        for agv in ranked:
            start = pose[agv][:2]
            goal = goals.get(agv, start)
            station = _dropoff_of(goal)
            if agv in _STICKY:
                hold.add(agv)
                continue
            if goal == start or station is None:
                # Other cars count. A sticky unloader sitting in a one-cell
                # door makes that pickup impossible; planning it aborts the run.
                obstacles = blocked | {c for c, who in standing.items() if who != agv}
                if _bfs(start, goal, obstacles) is None and goal != start:
                    hold.add(agv)
                    if pickup_call and _requeue_assigned(agv):
                        _SKIP_LOAD.add(agv)
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
                loc = {}
                fr = _solver_frame()
                if fr is not None:
                    loc = fr.f_locals
                _remember_cargo(
                    agv,
                    loc.get("loaded"),
                    loc.get("dest"),
                    loc.get("tid"),
                    station,
                )
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

        injected: List[str] = []
        saved_goal: Dict[str, object] = {}
        # Pickup reuses the solver goals dict, so identity is not enough.
        # A delivery call is one whose goals touch a dropoff. Leave uses
        # another dict and never enters here. At most one waiting carrier
        # joins, and only onto a pad nobody is standing on.
        for cell, who in standing.items():
            station = _dropoff_of(cell)
            if station is None or who in keep:
                continue
            if cell not in _road_pads(station, blocked):
                continue
            used.setdefault(station, set()).add(cell)
        if goals is caller_goals and _is_delivery_call(goals) and _STICKY:
            best: Optional[Tuple[int, Cell, str]] = None
            for agv, cargo in list(_STICKY.items()):
                station = cargo.get("station")
                if not isinstance(station, tuple) or agv not in pose:
                    continue
                start = pose[agv][:2]
                pads = _road_pads(station, blocked)
                taken = used.setdefault(station, set())
                # Already on a legal stand: stay. Cargo apply records the unload.
                if start in pads and start not in taken:
                    taken.add(start)
                    continue
                if start in pads:
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
                pad = min(free, key=lambda p: (_dist(start, p, blocked), p))
                dist = _dist(start, pad, blocked)
                if dist >= 10**9:
                    continue
                if best is None or (dist, agv) < (best[0], best[2]):
                    best = (dist, pad, agv)
            if best is not None:
                _dist_b, pad, agv = best
                cargo = _STICKY.get(agv) or {}
                station = cargo.get("station")
                saved_goal[agv] = goals.get(agv, None)
                goals[agv] = pad
                keep.add(agv)
                hold.discard(agv)
                if isinstance(station, tuple):
                    used.setdefault(station, set()).add(pad)
                injected.append(agv)
                print(
                    f"[PIPE] unload-join {agv} pad={pad} "
                    f"tid={cargo.get('tid') or '?'}",
                    flush=True,
                )

        yielded: List[str] = []
        if _is_delivery_call(goals) and keep:
            occupied = set(standing)
            carrying: Set[str] = set()
            if frame is not None and isinstance(frame.f_locals.get("loaded"), dict):
                carrying = {n for n, flag in frame.f_locals["loaded"].items() if flag}
            for agv in list(keep):
                if agv not in pose:
                    continue
                goal = goals.get(agv)
                if not isinstance(goal, tuple) or _dropoff_of(goal) is None:
                    continue
                start = pose[agv][:2]
                who = _door_blocker(start, goal, standing, keep, blocked, carrying)
                if who is None or who not in pose or who in yielded:
                    continue
                cell = pose[who][:2]
                step = _yield_step(cell, goal, blocked, occupied - {cell})
                if step is None:
                    continue
                saved_goal[who] = goals.get(who, None)
                goals[who] = step
                keep.add(who)
                occupied.discard(cell)
                occupied.add(step)
                standing = dict(standing)
                standing.pop(cell, None)
                standing[step] = who
                yielded.append(who)
                print(
                    f"[PIPE] door-yield {who} off={cell} step={step} "
                    f"open={goal} for {agv}",
                    flush=True,
                )

        if not keep:
            # Nobody can move. An empty plan is success; None aborts the run.
            return {}

        static_use = static
        if yielded:
            drop = {pose[a][:2] for a in yielded if a in pose}
            static_use = set(static) - drop

        def _undo(names: List[str]) -> None:
            for name in names:
                prev = saved_goal.get(name, None)
                if prev is None:
                    goals.pop(name, None)
                else:
                    goals[name] = prev
                keep.discard(name)

        tl = _ORIG["joint"](pose, goals, keep, static_use, *args, **kwargs)
        if tl is None and (injected or yielded):
            _undo(injected)
            _undo(yielded)
            if injected:
                print(
                    f"[PIPE] unload-join rollback {injected} (joint failed)",
                    flush=True,
                )
            if yielded:
                print(
                    f"[PIPE] door-yield rollback {yielded} (joint failed)",
                    flush=True,
                )
            if not keep:
                return {}
            return _ORIG["joint"](pose, goals, keep, static, *args, **kwargs)

        if isinstance(tl, dict) and injected and frame is not None:
            batch = frame.f_locals.get("batch")
            for agv in injected:
                seq = tl.get(agv) or []
                goal = goals.get(agv)
                hit = any(isinstance(p, tuple) and p[:2] == goal for p in seq)
                if not hit:
                    tl.pop(agv, None)
                    prev = saved_goal.get(agv, None)
                    if prev is None:
                        goals.pop(agv, None)
                    else:
                        goals[agv] = prev
                    continue
                cargo = _STICKY.get(agv) or {}
                if isinstance(batch, dict) and agv not in batch:
                    batch[agv] = {
                        "task_id": str(cargo.get("tid") or ""),
                        "destination": str(cargo.get("dest") or ""),
                        "end_points": [list(goal)] if isinstance(goal, tuple) else [],
                    }
        if isinstance(tl, dict) and yielded:
            for name in yielded:
                prev = saved_goal.get(name, None)
                if prev is None:
                    goals.pop(name, None)
                else:
                    goals[name] = prev
        return tl

    def _build_overlap_delivery_timelines(*args, **kwargs):
        global _OVERLAP_PENDING
        _OVERLAP_PENDING = True
        return _ORIG["overlap"](*args, **kwargs)

    def _apply_pose_timelines_cargo(steps_by, pose, timelines, cargo_tl, now):
        global _OVERLAP_PENDING
        pending = _OVERLAP_PENDING
        _OVERLAP_PENDING = False
        frame = _solver_frame()
        loc = frame.f_locals if frame is not None else {}
        result = _ORIG["cargo"](steps_by, pose, timelines, cargo_tl, now)
        now2, loaded, dest, tid = result
        if not pending:
            _restore_cargo(loaded, dest, tid)
            return (now2, loaded, dest, tid)
        blocked = set(_DROPOFFS) | set(_PICKUPS)
        batch = loc.get("batch") if isinstance(loc, dict) else None
        for agv in list(_STICKY):
            cell = pose[agv][:2] if agv in pose else None
            on_pad = isinstance(cell, tuple) and _on_own_pad(agv, cell, blocked)
            already_clear = isinstance(loaded, dict) and not loaded.get(agv, True)
            if on_pad and (already_clear or _seal_unload(steps_by, agv, cell)):
                if isinstance(loaded, dict):
                    loaded[agv] = False
                if isinstance(dest, dict):
                    dest[agv] = ""
                if isinstance(tid, dict):
                    tid[agv] = ""
                cargo = _STICKY.pop(agv, {})
                if isinstance(batch, dict) and agv not in batch:
                    batch[agv] = {
                        "task_id": str(cargo.get("tid") or ""),
                        "destination": str(cargo.get("dest") or ""),
                        "end_points": [list(cell)],
                    }
                print(
                    f"[PIPE] unload-done {agv} at={cell} tid={cargo.get('tid')}",
                    flush=True,
                )
                continue
            _remember_cargo(agv, loaded, dest, tid, _STICKY[agv].get("station"))  # type: ignore[arg-type]
            if isinstance(batch, dict) and agv in batch:
                batch.pop(agv, None)
                print(
                    f"[PIPE] unload-keep {agv} tid={_STICKY[agv].get('tid')} "
                    f"stay={cell} (not counted this wave)",
                    flush=True,
                )
        return (now2, loaded, dest, tid)

    def _apply_pose_timelines(steps_by, pose, timelines, now, *args, **kwargs):
        _restore_cargo(kwargs.get("loaded"), kwargs.get("dest"), kwargs.get("tid"))
        return _ORIG["apply"](steps_by, pose, timelines, now, *args, **kwargs)

    def _apply_paths(steps_by, pose, paths, now, *args, **kwargs):
        _restore_cargo(kwargs.get("loaded"), kwargs.get("dest"), kwargs.get("tid"))
        return _ORIG["paths"](steps_by, pose, paths, now, *args, **kwargs)

    def _pickup_load_agv(agv, task, *args, **kwargs):
        loaded = kwargs.get("loaded")
        dest = kwargs.get("dest")
        tid = kwargs.get("tid")
        _restore_cargo(loaded, dest, tid)
        if agv in _SKIP_LOAD:
            _SKIP_LOAD.discard(agv)
        elif agv not in _STICKY:
            return _ORIG["pickup"](agv, task, *args, **kwargs)
        # Caller is iterating assigned. Do not pop it here. Keep the old cargo
        # and still emit the sync tick the loader would have written.
        now = kwargs.get("now")
        if now is None and args:
            now = args[-1]
        steps_by = kwargs.get("steps_by")
        pose = kwargs.get("pose")
        names = kwargs.get("names")
        if not isinstance(now, int) or not isinstance(names, list):
            return _ORIG["pickup"](agv, task, *args, **kwargs)
        now += 1
        _restore_cargo(loaded, dest, tid)
        for n in names:
            steps_by[n].append(
                _ORIG["hold"](
                    n,
                    pose[n],
                    now,
                    loaded=bool(loaded.get(n, False)),
                    dest=str(dest.get(n, "") if isinstance(dest, dict) else ""),
                    tid=str(tid.get(n, "") if isinstance(tid, dict) else ""),
                )
            )
        return now

    def _distinct_goals_near(*args, **kwargs):
        _detach_new_tasks()
        return _ORIG["goals"](*args, **kwargs)

    def _clear_blocking_idles(*args, **kwargs):
        _detach_new_tasks()
        return _ORIG["clear"](*args, **kwargs)

    mod._plan_turn_aware_joint = _plan_turn_aware_joint
    mod._build_overlap_delivery_timelines = _build_overlap_delivery_timelines
    mod._apply_pose_timelines_cargo = _apply_pose_timelines_cargo
    mod._apply_pose_timelines = _apply_pose_timelines
    mod._apply_paths = _apply_paths
    mod._pickup_load_agv = _pickup_load_agv
    mod._distinct_goals_near = _distinct_goals_near
    mod._clear_blocking_idles = _clear_blocking_idles
