"""SH_custom_* lifelong MAPD via windowed multi-agent ECBS (Weighted-CBS).

One-shot ECBS cannot eat 100 pickup–delivery pairs at once. We roll:
  assign ≤K free AGVs → joint plan to pickups → joint plan to unloads → repeat.

Default ``plan=joint`` (aliases: eecbs / wcbs / ecbs): all active AGVs are
planned **together**. Duplicate-station tasks get distinct free cells adjacent
to the same station so K>6 still means many cars working in parallel — not a
single-car A* loop.

``plan=per_agent``: prioritized one-by-one (still multi-task waves).

Grid cells in pymapf are (row, col) = (y-1, x-1) for warehouse (x,y) in 1..20.
"""
from __future__ import annotations
import argparse, csv, heapq, json, time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from pymapf import Agent, GridMap, MAPFProblem, solve
from ml_research.benchmarks.common import TRAJ_HEADER, load_scenario, patch_extra_obstacles
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import format_validation_summary, validate_hybrid_trajectory
from ml_research.common.paths import RESULTS; Cell = Tuple[(int, int)]; Pose = Tuple[(int, int, int)]

OUT = RESULTS / "coord_custom_ai" / "ecbs"

TRAJ = OUT / "trajectories"
def _wh_to_rc(x: "int", y: "int") -> "Cell":
    return (y - 1, x - 1)

def _rc_to_wh(r: "int", c: "int") -> "Cell":
    return (c + 1, r + 1)

def _manh(a: "Cell", b: "Cell") -> "int":
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def _pitch_from_delta(dx: "int", dy: "int", prev: "int") -> "int":
    if dx == 1:
        return 0
    elif dx == -1:
        return 180
    elif dy == 1:
        return 90
    elif dy == -1:
        return 270
    return int(prev) % 360

def _hold(name: "str", pose: "Pose", t: "int", *, loaded: "bool", dest: "str", tid: "str") -> "dict":
    return {"timestamp": t, "name": name, "X": pose[0], "Y": pose[1], "pitch": pose[2], "loaded": "FALSE", "destination": dest, "Emergency": "FALSE", "task-id": tid}

def _build_grid(static: "Set[Cell]", stations: "Set[Cell]", W: "int"=20, H: "int"=20) -> "GridMap":
    grid = []
    for y in range(1, H + 1):
        row = []
        for x in range(1, W + 1):
            if not (x, y) in static:
                (x, y) in static
            blocked = (x, y) in stations
            row.append(0)
        grid.append(row)
    return GridMap(grid)

def _unload_cell(end_xy: "Cell", free: "Set[Cell]", prefer: "Cell") -> "Cell":
    x, y = end_xy
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        pass
    nbrs = [(x + dx, y + dy)]; dx = dx; dy = dy
    if not nbrs:
        return prefer
    
    return min(nbrs, key=(lambda c: (_manh(c, prefer), c)))
    
    dy = None; dx = None

def _rematch_pipeline_next(unload: "Dict[str, Cell]", next_task: "Dict[str, dict]", next_pick: "Dict[str, Cell]", static: "Set[Cell]", *, arrive: "Optional[Dict[str, int]]", free: "Optional[Set[Cell]]", include_deliver: "bool") -> "Tuple[Dict[str, dict], Dict[str, Cell], int]":
    import itertools; ag = [a for a in sorted(next_task) if not a in unload]; a = pk
    if len(ag) <= 1:
        return (next_task, next_pick, 1_000_000_000)
    items = [(next_task[a], next_pick[a]) for a in ag]; a = static; best = None
    for perm in itertools.permutations(range(len(ag))):
        costs = []
        used_p = set()
        ok = True
        for i, a in enumerate(ag):
            task, pk = items[perm[i]]
            if pk in used_p:
                ok = False
                break
            used_p.add(pk)
            d = _bfs_len(unload[a], pk, static)
            if d >= 100_000:
                ok = False
                break
            elif not arrive:
                arrive
            t0 = int({}.get(a, 0) or 0)
            cost = t0 + d * 2 + 4
            if include_deliver and free is None:
                if not task.get("end_points"):
                    task.get("end_points")
                ends = [tuple(e) for e in [] if tuple(e) in free]
                e = None
                if ends:
                    d2 = min((_bfs_len(pk, e, static) for e in ends))
                    cost += d2 * 2 + 4
            costs.append(cost)
        if not ok or costs:
            continue
        mspan = max(costs)
        total = sum(costs)
        if best is None:
            continue
        for i in enumerate(ag):
            a = ()
            task, pk = items[perm[i]]
            tcopy = dict(task)
            tcopy["_pipe_pick"] = pk
            new_t[a] = tcopy
            new_p[a] = pk
        best = (mspan, total, new_t, new_p)
    if best is not None:
        pass
    dict(next_pick)
    
    a = dict(next_task); a = None
    e = None

def _retarget_unload_toward_next(starts: "Dict[str, Cell]", batch: "Dict[str, dict]", goals: "Dict[str, Cell]", next_pick: "Dict[str, Cell]", free: "Set[Cell]", static: "Set[Cell]", avoid: "Optional[Set[Cell]]"=None) -> "Dict[str, Cell]":
    reserved = {starts[n] for n in starts if not n not in batch}; n = pk; avoid = set(avoid or ()); out = dict(goals); order = sorted(batch.keys(), key=(lambda a: (1, _bfs_len(starts[a], goals.get(a, starts[a]), static), a)))
    for agv in order:
        task = batch[agv]
        if not task.get("end_points"):
            task.get("end_points")
        ends = [tuple(e) for e in [] if tuple(e) in free]
        e = None
        if not ends:
            continue
        c = cores
        if not [c for c in ends if not c not in reserved]:
            [c for c in ends if not c not in reserved]
        cands = list(ends)
        cores = list(avoid)
        forced_far = False
        if avoid:
            cleared = [c for c in cands if not c not in avoid]
            c = agv
            if cleared:
                cands = cleared
            elif cores:
                forced_far = True
        pk = next_pick.get(agv)
        if forced_far:
            choice = max(cands, key=(lambda c: (min((_manh(c, e) for e in cores), default=0),
    -_bfs_len(starts[agv], c, static),
    
    c)))
        elif pk is not None:
            choice = min(cands, key=(lambda c: (_bfs_len(starts[agv], c, static), _manh(starts[agv], c), c)))
        else:
            choice = min(cands, key=(lambda c: (_bfs_len(starts[agv], c, static) + 2 * _bfs_len(c, pk, static), _bfs_len(c, pk, static), _bfs_len(starts[agv], c, static), c)))
        out[agv] = choice
        reserved.add(choice)
    return out
    next_pick
    n = goals
    starts
    e = None
    
    c = None; c = None

def _path_unit_steps(path: "List[Cell]") -> "bool":
    for a, b in zip(path, path[1:]):
        if not _manh(a, b) > 1:
            pass
    return False; return True

def _station_of_approach(pk: "Cell", stations: "Set[Cell]") -> "Cell":
    for s in stations:
        if not _manh(s, pk) == 1:
            pass
    
    return s; return pk

def _adj_free(center: "Cell", free: "Set[Cell]") -> "List[Cell]":
    x, y = center
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        pass
    out = [(x + dx, y + dy)]; dx = dx; dy = dy; out.sort(key=(lambda c: (_manh(c, center), c)))
    return out
    
    dy = None; dx = None

def _lr_adj_free(center: "Cell", free: "Set[Cell]") -> "List[Cell]":
    x, y = center; out = [(x + dx, y) for dx in (1, -1) if not (x + dx, y) in free]; dx = None; out.sort()
    return out
    
    dx = None

def _pickup_stand_cands(primary: "Cell", station: "Cell", free: "Set[Cell]", reserved: "Set[Cell]") -> "List[Cell]":
    cands = []
    if primary in free and primary not in reserved:
        cands.append(primary)
    for c in _lr_adj_free(station, free):
        if not c not in reserved:
            continue
        elif not c not in cands:
            continue
        cands.append(c)
    return cands

def _distinct_goals_near(assigned: "Dict[str, dict]", starts: "Dict[str, Cell]", free: "Set[Cell]", stations: "Set[Cell]", *, primary_of) -> "Optional[Dict[str, Cell]]":
    reserved = {starts[n] for n in starts if not n not in assigned}; n = primary; goals = {}; order = sorted(assigned.keys(), key=(lambda n: (_manh(starts[n], primary_of(assigned[n])), n)))
    for agv in order:
        primary = primary_of(assigned[agv])
        station = _station_of_approach(primary, stations)
        cands = _pickup_stand_cands(primary, station, free, reserved)
        if not cands:
            return None
        goal = min(cands, key=(lambda c: (1, 0, _manh(starts[agv], c), c)))
        goals[agv] = goal
        reserved.add(goal)
    return goals
    primary_of
    n = starts

def _ecbs(grid: "GridMap", starts_wh: "Dict[str, Cell]", goals_wh: "Dict[str, Cell]", *, weight: "float", time_limit: "float", max_expansions: "int", movers: "Optional[Set[str]]", plan: "str") -> "Optional[Dict[str, List[Cell]]]":
    movers = set(movers) if movers is None else set(starts_wh); w = grid.width; h = grid.height
    for r in range(h):
        c = None
    base = [[1 for c in range(w)]]; r = r; c = c
    for name, cell in starts_wh.items():
        if name in movers:
            continue
        r, c = _wh_to_rc(*cell)
        if not 0 <= r < h:
            continue
        else:
            continue
        if not 0 <= c < w:
            continue
        else:
            continue
        base[r][c] = 1
    agents = []
    for name in sorted(movers):
        sr, sc = _wh_to_rc(*starts_wh[name])
        gr, gc = _wh_to_rc(*goals_wh[name])
        if 0 <= sr < h and 0 <= sc < w:
            pass
        
    
    base[sr][sc] = 0
    
    g2 = GridMap(base)
    if not g2.is_free((gr, gc)):
        return None
    
    agents.append(Agent(name=name, start=(sr, sc), goal=(gr, gc)))
    if not agents:
        n = None
        return {n: [starts_wh[n]] for n in starts_wh}
    g2 = GridMap(base)
    
    problem = MAPFProblem(g2, agents); priority = sorted(movers, key=(lambda n: (_manh(starts_wh[n], goals_wh[n]), n)))
    
    plan_l = str(plan).strip().lower()
    if plan_l in ("joint", "eecbs", "wcbs", "ecbs"):
        sol = solve(problem, "wcbs", weight=float(weight), time_limit=float(time_limit), max_expansions=int(max_expansions))
    
    elif plan_l == "lacam":
        sol = solve(problem, "lacam")
    else:
        sol = solve(problem, "prioritized", priority=priority)
    if not sol is None and sol.paths:
        return None
    out = {}
    for name, path_rc in sol.paths.items():
        for r, c in path_rc:
            pass
        c = c
        r = r
        out[name] = [_rc_to_wh(r, c)]
    for name in starts_wh:
        if not name not in out:
            continue
        out[name] = [starts_wh[name]]
    if not all((_path_unit_steps(p) for p in out.values())):
        return None
    return out
    
    c = None
    goals_wh
    c = None; r = starts_wh; n = None; c = None; r = None

def _repark_others(grid: "GridMap", pose: "Dict[str, Pose]", worker: "str", free: "Set[Cell]", keep_clear: "Set[Cell]", *, weight: "float", time_limit: "float", max_expansions: "int", plan: "str") -> "Optional[Dict[str, List[Cell]]]":
    others = [n for n in pose if not n != worker]; n = None; park_cells = []; reserved_p = {pose[worker][:2]} | set(keep_clear)
    for idle in others:
        park = _nearby_park(pose[idle][:2], free, reserved_p)
        if park is not None:
            break
        park_cells.append(park)
        reserved_p.add(park)
    starts_r = {n: pose[n][:2] for n in pose}; n = None; goals_r = {n: starts_r[n] for n in pose}; n = None
    for idle, park in zip(others, park_cells):
        goals_r[idle] = park
    if not others:
        n = None
        return {n: [starts_r[n]] for n in starts_r}
    paths = _ecbs(grid, starts_r, goals_r, weight=weight, time_limit=max(60.0, time_limit), max_expansions=max_expansions, movers=set(others), plan=plan)
    if paths is not None:
        return None
    elif not all((_path_unit_steps(p) for p in paths.values())):
        return None
    return paths
    
    n = None; n = None
    
    n = None; n = None

def _repark_others_serial(mod, pose: "Dict[str, Pose]", names: "List[str]", worker: "str", free: "Set[Cell]", keep_clear: "Set[Cell]", static_list: "list", now: "int", *, steps_by: "Dict[str, List[dict]]", loaded: "Dict[str, bool]", dest: "Dict[str, str]", tid: "Dict[str, str]") -> "int":
    others = [n for n in names if not n != worker]; n = None; n = {pose[worker][:2]} | set(keep_clear); reserved_p = ##ERROR## | {pose[n][:2] for n in names}; parks = {}
    for idle in others:
        park = _nearby_park(pose[idle][:2], free, reserved_p)
        if park is not None:
            continue
        parks[idle] = park
        reserved_p.add(park)
    for idle, park in parks.items():
        if pose[idle][:2] == park:
            continue
        cells = _astar_move(mod, pose, names, idle, park, static_list, now, tid="", dest="", wall=45.0)
        if not cells:
            continue
        paths = {n: [pose[n][:2]] for n in names}
        n = None
        paths[idle] = cells
        now = _apply_paths(steps_by, pose, paths, now, loaded=loaded, dest=dest, tid=tid)
    return now
    
    n = None; n = None; n = None

def _astar_move(mod, pose: "Dict[str, Pose]", names: "List[str]", agv: "str", goal: "Cell", static_list: "list", now: "int", *, tid: "str", dest: "str", wall: "float") -> "Optional[List[Cell]]":
    block = [pose[n][:2] for n in names if not n != agv]; n = None; path, _steps = mod.plan_relocation({"agv": agv, "task_id": tid, "destination": dest, "priority": "Normal"}, [now, pose[agv][2]], goal, list(static_list) + block, {}, wall_budget_s=wall, max_path_len=500, max_visited=150_000)
    if not path:
        return None
    cells = [(int(p[0]), int(p[1])) for p in path]; p = None
    if cells and cells[0] != pose[agv][:2]:
        cells = [pose[agv][:2]] + cells
    return cells
    
    n = None; p = None

def _nearest_park(from_cell: "Cell", free: "Set[Cell]", forbidden: "Set[Cell]") -> "Optional[Cell]":
    return _nearby_park(from_cell, free, forbidden)

def _clear_blocking_idles(grid: "GridMap", pose: "Dict[str, Pose]", blocked_cells: "Set[Cell]", movers_needed: "Set[str]", free: "Set[Cell]", *, weight: "float", time_limit: "float", max_expansions: "int", plan: "str") -> "Optional[Dict[str, List[Cell]]]":
    starts = {n: pose[n][:2] for n in pose}; n = None; goals = {n: starts[n] for n in pose}; n = None; clear_movers = set(); reserved = set(starts.values()) - set(blocked_cells)
    for name, cell in starts.items():
        if name in movers_needed:
            continue
        elif not cell in blocked_cells:
            continue
        park = _nearest_park(cell, free, reserved | set(blocked_cells))
        if park is not None:
            return None
        goals[name] = park
        reserved.add(park)
        clear_movers.add(name)
    if not clear_movers:
        n = None
        return {n: [starts[n]] for n in starts}
    
    return _ecbs(grid, starts, goals, weight=weight, time_limit=time_limit, max_expansions=max_expansions, movers=clear_movers, plan=plan)
    
    n = None; n = None; n = None

def _pad_paths(paths: "Dict[str, List[Cell]]", names: "List[str]", starts: "Dict[str, Cell]") -> "Dict[str, List[Cell]]":
    for n in names:
        if n not in paths and paths[n]:
            continue
        paths[n] = [starts[n]]
    T = max((len(p) for p in paths.values()))
    for n in names:
        p = paths[n]
        if not len(p) < T:
            continue
        p.append(p[-1])
        if len(p) < T:
            pass
    
    return paths

def _required_pitch(dx: "int", dy: "int") -> "int":
    if dx == 1:
        return 0
    elif dx == -1:
        return 180
    elif dy == 1:
        return 90
    elif dy == -1:
        return 270
    raise ValueError(f"non-unit move ({dx}, {dy})")

def _delta_of_pitch(pitch: "int") -> "Tuple[int, int]":
    p = int(pitch) % 360
    if p == 0:
        return (1, 0)
    elif p == 90:
        return (0, 1)
    elif p == 180:
        return (-1, 0)
    elif p == 270:
        return (0, -1)
    return (0, 0)

def _turn_aware_st_astar(start: "Pose", goal: "Cell", static: "Set[Cell]", reserved_v: "Dict[Tuple[int, Cell], str]", reserved_e: "Set[Tuple[int, Cell, Cell]]", who: "str", *, tmax: "int", max_expansions: "int") -> "Optional[List[Pose]]":
    sx = int(start[0]); sy = int(start[1]); sp = int(start[2]) % 360
    if (sx, sy) == goal:
        return []
    def cell_free(t: "int", cell: "Cell") -> "bool":
        if cell in static and cell != goal and cell != (sx, sy):
            return False
        elif not 1 <= cell[0] <= 20 or 1 <= cell[1] <= 20:
            return False
        other = reserved_v.get((t, cell))
        return other is None or other == who
    
    def edge_free(t: "int", c0: "Cell", c1: "Cell") -> "bool":
        if c0 == c1:
            return True
        return (t, c1, c0) not in reserved_e
    
    start_st = (sx, sy, sp, 0); open_h = []; heapq.heappush(open_h, (_manh((sx, sy), goal), 0, start_st))
    
    parent = {start_st: None}; g_best = {start_st: 0}; closed = set(); found = None; exp = 0
    if open_h:
        while exp < max_expansions:
            _f, _, st = heapq.heappop(open_h)
            if st in closed:
                continue
            closed.add(st)
            exp += 1
            x, y, pitch, t = st
            if (x, y) == goal:
                found = st
                break
            elif t >= tmax:
                continue
            t1 = t + 1
            nxt_states = []
            if cell_free(t1, (x, y)):
                nxt_states.append((x, y, pitch))
            if cell_free(t1, (x, y)):
                for tp in (0, 90, 180, 270):
                    if not tp != pitch:
                        continue
                    nxt_states.append((x, y, tp))
            dx, dy = _delta_of_pitch(pitch)
            ny = y + dy
            nx = x + dx
            if cell_free(t1, (nx, ny)) and edge_free(t1, (x, y), (nx, ny)):
                nxt_states.append((nx, ny, pitch))
            for nx, ny, np in nxt_states:
                nst = (nx, ny, np, t1)
                if nst in closed:
                    continue
                ng = t1
                if ng >= g_best.get(nst, 1_000_000_000):
                    continue
                g_best[nst] = ng
                parent[nst] = st
                heapq.heappush(open_h, (ng + _manh((nx, ny), goal), exp, nst))
            if open_h:
                pass
    elif found is not None:
        return None
    chain = []; cur = found
    while cur is None:
        chain.append(cur)
        cur = parent[cur]
    
    chain.reverse()
    for x, y, pitch, _t in chain[1:]:
        pass
    _t = _t; pitch = pitch; y = y; x = x
    return [(x, y, pitch)]
    sy
    _t = reserved_v; pitch = sx; y = who; x = reserved_e

def _plan_turn_aware_joint(pose: "Dict[str, Pose]", goals: "Dict[str, Cell]", movers: "Set[str]", static: "Set[Cell]", *, tmax_scale: "float", prefer_outer_ring: "bool") -> "Optional[Dict[str, List[Pose]]]":
    names = list(pose.keys()); movers = set(movers)
    if not movers:
        n = timelines
        return {n: [] for n in names}
    def _ring_via(start: "Cell", goal: "Cell") -> "List[Tuple[Cell, Cell]]":
        out = []
        for y_ring in (2, 19):
            w1 = (start[0], y_ring)
            w2 = (goal[0], y_ring)
            cost = abs(start[1] - y_ring) + abs(goal[1] - y_ring) + abs(start[0] - goal[0])
            out.append((w1, w2, cost))
        for x_ring in (2, 19):
            w1 = (x_ring, start[1])
            w2 = (x_ring, goal[1])
            cost = abs(start[0] - x_ring) + abs(goal[0] - x_ring) + abs(start[1] - goal[1])
            out.append((w1, w2, cost + 5))
        
        out.sort(key=(lambda x: (x[2], x[0], x[1])))
        for a, b, _ in out[:4]:
            pass
        _ = _; b = b; a = a
        return [(a, b)]
        
        _ = None; b = None; a = None
    
    def _path_agent(n: "str", reserved_v: "Dict[Tuple[int, Cell], str]", reserved_e: "Set[Tuple[int, Cell, Cell]]", horizon: "int", *, force_ring: "Optional[Tuple[Cell, Cell]]") -> "Optional[List[Pose]]":
        start = pose[n]; goal = goals[n]; direct = _turn_aware_st_astar(start, goal, static, reserved_v, reserved_e, n, tmax=horizon)
        if direct is not None:
            direct = _turn_aware_st_astar(start, goal, static, reserved_v, reserved_e, n, tmax=horizon * 2)
        best = direct; best_score = 1_000_000_000
        if prefer_outer_ring and force_ring is not None:
            return best
        def _try_via(w1: "Cell", w2: "Cell") -> "Optional[List[Pose]]":
            if w1 in static or w2 in static:
                return None
            elif not 1 <= w1[0] <= 20 or 1 <= w1[1] <= 20:
                return None
            elif not 1 <= w2[0] <= 20 or 1 <= w2[1] <= 20:
                return None
            rv = dict(reserved_v); re = set(reserved_e); segs = []; cur = start
            for gp in (w1, w2, goal):
                if cur[:2] == gp:
                    continue
                bud = max(50, int(_bfs_len(cur[:2], gp, static) * 5) + 50)
                seg = _turn_aware_st_astar(cur, gp, static, rv, re, n, tmax=bud)
                if seg is not None:
                    seg = _turn_aware_st_astar(cur, gp, static, rv, re, n, tmax=bud * 2)
                if seg is not None:
                    return None
                t0 = len(segs)
                prev = cur[:2]
                for i, p in enumerate(seg):
                    t = t0 + i + 1
                    c = (p[0], p[1])
                    rv[(t, c)] = n
                    if c != prev:
                        re.add((t, prev, c))
                    prev = c
                segs.extend(seg)
                cur = segs[-1]
            if segs:
                return segs
        
        vias = _ring_via(start[:2], goal)
        for pair in vias:
            if pair is not None:
                continue
            w1, w2 = pair
            segs = _try_via(w1, w2)
            if segs is not None:
                continue
            score = len(segs) - 0
            if not score < best_score:
                continue
            best = segs
            best_score = score
        return best
    
    reserved_v = {}; reserved_e = set(); base_h = 40
    for n in movers:
        base_h = max(base_h, int(_bfs_len(pose[n][:2], goals[n], static) * tmax_scale) + 30)
    horizon = max(base_h, 120)
    for n in names:
        c = pose[n][:2]
        if n in movers:
            reserved_v[(0, c)] = n
            continue
        for t in range(0, horizon * 2 + 64):
            reserved_v[(t, c)] = n
    reserved_v0 = dict(reserved_v); base = sorted(movers, key=(lambda n: (_bfs_len(pose[n][:2], goals[n], static), n)))
    
    orders = [("direct_short", list(base),
    None), ("direct_long", list(reversed(base)),
    None)]
    if prefer_outer_ring:
        for y_ring in (2, 19):
            orders.append((f"highway_y{y_ring}_short", list(base), y_ring))
            orders.append((f"highway_y{y_ring}_long",
    
    list(reversed(base)), y_ring))
    best_tl = None; best_T = 1_000_000_000
    for _tag, order, y_hw in orders:
        reserved_v = dict(reserved_v0)
        reserved_e = set()
        n = None
        timelines = {n: [] for n in names}
        ok = True
        for n in order:
            force = None
            if y_hw is None:
                s = pose[n][:2]
                g = goals[n]
                force = ((s[0], y_hw), (g[0], y_hw))
            path = _path_agent(n, reserved_v, reserved_e, horizon, force_ring=force)
            if path is not None and force is None:
                path = _path_agent(n, reserved_v, reserved_e, horizon)
            if path is not None:
                ok = False
                break
            timelines[n] = path
            prev = pose[n][:2]
            for i, p in enumerate(path):
                t = i + 1
                c = (p[0], p[1])
                reserved_v[(t, c)] = n
                if c != prev:
                    reserved_e.add((t, prev, c))
                prev = c
            fin_t = len(path)
            final = pose[n][:2]
            hold_to = max(horizon * 2 + 64, fin_t + 8)
            for t in range(fin_t + 1, hold_to + 1):
                reserved_v[(t, final)] = n
        if not ok:
            continue
        Tmax = max((len(timelines[n]) for n in movers), default=0)
        for n in movers:
            last = pose[timelines[n][-1] if timelines[n] else n]
            if not len(timelines[n]) < Tmax:
                continue
            timelines[n].append(last)
            if len(timelines[n]) < Tmax:
                pass
        prev_c = {n: pose[n][:2] for n in names}
        n = None
        conflict = False
        for t in range(Tmax):
            cur_c = {n: prev_c[timelines[n][t][:2] if t < len(timelines[n]) else n] for n in names}
            n = None
            occ = {}
            for n, c in cur_c.items():
                if c in occ:
                    conflict = True
                    break
                occ[c] = n
            if conflict:
                break
            for n, c in cur_c.items():
                p = prev_c[n]
                if c == p:
                    continue
                for m, mc in cur_c.items():
                    if not m != n:
                        continue
                    elif not mc == p:
                        continue
                    conflict = prev_c[m] == c or True
                if not conflict:
                    pass
            prev_c = cur_c
        if conflict:
            continue
        if not Tmax < best_T:
            continue
        best_T = Tmax
        best_tl = timelines
        print(f"[PIPE] leave-plan {_tag} T={Tmax}", flush=True)
    if best_tl is not None:
        return None
    for n in names:
        if n not in movers:
            best_tl[n] = []
        last = pose[best_tl[n][-1] if best_tl[n] else n]
        if not len(best_tl[n]) < best_T:
            continue
        best_tl[n].append(last)
        if len(best_tl[n]) < best_T:
            pass
    
    return best_tl
    _ring_via
    n = prefer_outer_ring
    static
    n = goals
    pose
    n = None; n = None

def _apply_pose_timelines(steps_by: "Dict[str, List[dict]]", pose: "Dict[str, Pose]", timelines: "Dict[str, List[Pose]]", now: "int", *, loaded: "Dict[str, bool]", dest: "Dict[str, str]", tid: "Dict[str, str]") -> "int":
    names = list(pose.keys()); T = max((len([]) for n in names), default=0)
    for t_i in range(T):
        now += 1
        for n in names:
            if not timelines.get(n):
                timelines.get(n)
            seq = []
            if t_i < len(seq):
                pose[n] = seq[t_i]
            steps_by[n].append(_hold(n, pose[n], now, loaded=loaded.get(n, False), dest=dest.get(n, ""), tid=tid.get(n, "")))
    return now

def _expand_cell_paths_to_pose_timelines(pose0: "Dict[str, Pose]", paths: "Dict[str, List[Cell]]") -> "Dict[str, List[Pose]]":
    pose = {n: pose0[n] for n in pose0}; n = trimmed; names = list(pose.keys()); trimmed = {}
    for n in names:
        if not paths.get(n):
            paths.get(n)
        p = list([pose[n][:2]])
        if len(p) >= 2 and p[0] == pose[n][:2]:
            p = p[1:]
        if not p:
            p = [pose[n][:2]]
        trimmed[n] = p
    T_cells = max((len(trimmed[n]) for n in names)); timelines = {n: [] for n in names}; n = None
    for step in range(T_cells):
        nxt = {n: trimmed[n][-1] for n in names}
        n = None
        for _guard in range(4):
            need = []
            for n in names:
                cur = pose[n]
                goal_c = nxt[n]
                if goal_c == (cur[0], cur[1]):
                    continue
                dy = goal_c[1] - cur[1]
                dx = goal_c[0] - cur[0]
                if abs(dx) + abs(dy) != 1:
                    continue
                tp = _required_pitch(dx, dy)
                if not int(cur[2]) % 360 != tp:
                    continue
                need.append((n, tp))
            if not need:
                break
            for n, tp in need:
                x, y, _ = pose[n]
                pose[n] = (x, y, tp)
            for n in names:
                timelines[n].append(pose[n])
        for n in names:
            x, y = nxt[n]
            pitch = pose[n][2]
            if (x, y) != (pose[n][0], pose[n][1]):
                dy = y - pose[n][1]
                dx = x - pose[n][0]
                if abs(dx) + abs(dy) == 1:
                    pitch = _required_pitch(dx, dy)
            pose[n] = (x, y,
                
                pitch)
        for n in names:
            timelines[n].append(pose[n])
    return timelines
    
    n = None; n = None; n = None

def _apply_pose_timelines_cargo(steps_by: "Dict[str, List[dict]]", pose: "Dict[str, Pose]", timelines: "Dict[str, List[Pose]]", cargo_tl: "Dict[str, List[Tuple[bool, str, str]]]", now: "int") -> "Tuple[int, Dict[str, bool], Dict[str, str], Dict[str, str]]":
    names = list(pose.keys()); T = max((len([]) for n in names), default=0); loaded = {}; dest = {}; tid = {}
    for n in names:
        if not cargo_tl.get(n):
            cargo_tl.get(n)
        cq = []
        if cq:
            loaded[n] = cq[0][0]
            dest[n] = cq[0][1]
            tid[n] = cq[0][2]
            continue
        loaded[n],
            
            dest[n], tid[n] = (False, "", "")
    
    for t_i in range(T):
        now += 1
        for n in names:
            if not timelines.get(n):
                timelines.get(n)
            seq = []
            if t_i < len(seq):
                pose[n] = seq[t_i]
            if not cargo_tl.get(n):
                cargo_tl.get(n)
            cq = []
            if t_i < len(cq):
                loaded[n], dest[n], tid[n] = cq[t_i]
            steps_by[n].append(_hold(n, pose[n], now, loaded=loaded[n], dest=dest[n], tid=tid[n]))
    return (now, loaded, dest, tid)

def _apply_cargo_chunked_with_idle_nudge(steps_by: "Dict[str, List[dict]]", pose: "Dict[str, Pose]", timelines: "Dict[str, List[Pose]]", cargo_tl: "Dict[str, List[Tuple[bool, str, str]]]", now: "int", *, idle_movers: "List[str]", names: "List[str]", free: "Set[Cell]", static: "Set[Cell]", chunk: "int", forbidden: "Optional[Set[Cell]]") -> "Tuple[int, Dict[str, bool], Dict[str, str], Dict[str, str]]":
    names_all = list(timelines.keys()); T = max((len([]) for n in names_all), default=0); loaded = {n: False for n in names}; n = cur; dest = {n: "" for n in names}; n = timelines; tid = {n: "" for n in names}; n = None; t0 = 0
    while t0 < T:
        t1 = min(T, t0 + chunk)
        sub_tl = {[n]: [][t0:t1] for n in names_all if not timelines.get(n)}
        n = None
        sub_cg = {[n]: [][t0:t1] for n in names_all if not cargo_tl.get(n)}
        n = None
        now, loaded, dest, tid = _apply_pose_timelines_cargo(steps_by, pose, sub_tl, sub_cg, now)
        t0 = t1
        if not t0 >= T or idle_movers:
            continue
        elif not forbidden:
            forbidden
        c = {}
        n = {c for c in ()}
        reserved = {} | {pose[n][:2] for n in names if not n not in idle_movers}
        ring = _wave_wait_ring(free, reserved)
        cell_goals = {}
        for idle in idle_movers:
            if idle not in names:
                continue
            cur = pose[idle][:2]
            cell_goals[idle] = nxt
    min(opts, key=(lambda c: (_manh(cur, c), c))); n = [c for c in ring]; n = None; n = None; n = None; n = None; c = None; n = None
    
    c = _send_idles_wave_wait(steps_by, pose, names, list(idle_movers), free, static, now, loaded=loaded, dest=dest, tid=tid, reserved=reserved, max_movers=len(idle_movers))

def _try_via_roll_suffix(names: "List[str]", batch: "Dict[str, dict]", next_task: "Dict[str, dict]", next_pick: "Dict[str, Cell]", tl_v: "Dict[str, List[Pose]]", cargo_v: "Dict[str, List[Tuple[bool, str, str]]]", static: "Set[Cell]", free: "Set[Cell]", queues: "Dict[str, List[dict]]", *, k_early: "int", gap_min: "int") -> "Optional[Tuple[int, List[str], Dict[str, List[Pose]], Dict[str, List[Tuple[bool, str, str]]], Dict[str, List[Pose]], Dict[str, List[Tuple[bool, str, str]]], Dict[str, dict], Dict[str, Cell]]]":
    arr_via = {}
    for a, g in next_pick.items():
        if not tl_v.get(a):
            tl_v.get(a)
        seq = []
        if not cargo_v.get(a):
            cargo_v.get(a)
        cq = []
        for i, p in enumerate(seq):
            if (p[0], p[1]) != g:
                continue
            elif not i < len(cq):
                continue
            elif cq[i][0]:
                pass
            arr_via[a] = i + 1
            tmax_e
    all_un_t = 0
    for a in batch:
        if not cargo_v.get(a):
            cargo_v.get(a)
        cq = []
        ut = next((i + 1 for ld, _, _ in enumerate(cq)), None)
        if ut is not None:
            return None
        all_un_t = max(all_un_t, ut)
    sorted_via = sorted(arr_via, key=(lambda a: (arr_via[a], a)))
    if len(sorted_via) < max(2, k_early):
        return None
    gap_via = arr_via[sorted_via[-1]] - arr_via[sorted_via[0]]
    if gap_via < gap_min:
        return None
    T_via = max((len([]) for a in batch), default=0)
    
    t_cut = max(arr_via[sorted_via[k_early - 1]], all_un_t)
    if t_cut > T_via - 20:
        return None
    a = td; early_names = [a for a in sorted_via if not arr_via[a] <= t_cut][:k_early]
    if len(early_names) < k_early:
        return None
    def _pose_at(n: "str", cut: "int") -> "Pose":
        if not tl_v.get(n):
            tl_v.get(n)
        seq = list([])
        if not seq:
            return (1, 1, 0)
        while len(seq) < cut:
            seq.append(seq[-1])
        return seq[cut - 1]
    
    n = suffix_len; pose_cut = {n: _pose_at(n, t_cut) for n in names}; early = [a for a in early_names if a in next_task]; a = reserved_v
    if len(early) < k_early:
        print(f"[DBG] early<k names={early_names} got={early}", flush=True)
        return None
    
    late = [a for a in next_pick if not a not in early]
    
    a = reserved_e
    if not late:
        print("[DBG] no late", flush=True)
        return None
    
    print(f"[DBG] early={early} late={late} t_cut={t_cut} suffix={T_via - t_cut}", flush=True); suffix_len = T_via - t_cut; reserved_v = {}; reserved_e = set()
    for n in names:
        c0 = pose_cut[n][:2]
        reserved_v[(0, c0)] = n
    n = None; post_tl = {n: [] for n in names}; post_cg = {n: [] for n in names}; n = post_tl; seed_extra = {}; seed_pick = {}; tmax_e = suffix_len + 80; a = pose_cut
    a = {pose_cut[a][:2] for a in early}; early_cells = path | {next_pick[a] for a in early if not a in next_pick}; late_pose = dict(pose_cut); park_T = 0
    for n in sorted(late):
        drop = pose_cut[n][:2]
        for c in free:
            x = c
        cands = [c]
        c = c
        x = x
        if not cands:
            post_tl[n] = []
            continue
        park = min(cands, key=(lambda c: if not c[0] in (1, 2, 19, 20):
    pass; (1, _bfs_len(drop, c, static), c)))
        for t, c in reserved_v.items():
            who = None
        sh_v = {(t, c): who}
        c = c
        t = t
        who = who
        sh_e = set(reserved_e)
        sh_v[(0, drop)] = n
        bud = max(20, int(_bfs_len(drop, park, static) * 6) + 15)
        path_p = _turn_aware_st_astar(pose_cut[n], park, static, sh_v, sh_e, n, tmax=bud)
        if path_p is None and len(path_p) > 10:
            continue
        prev = drop
        for i, p in enumerate(path_p):
            t = i + 1
            c = (p[0], p[1])
            reserved_v[(t, c)] = n
            if c != prev:
                reserved_e.add((t,
    
    prev, c))
            prev = c
        late_pose[n] = path_p[-1]
        post_tl[n] = list(path_p)
        post_cg[n] = [(False, "", "")] * len(path_p)
        park_T = max(park_T, len(path_p))
        for t in range(len(path_p) + 1, suffix_len + 90):
            reserved_v[(t, park)] = n
        print(f"[DBG] late-park {n} {drop}->{park} len={len(path_p)}", flush=True)
    used_cells = {late_pose[n][:2] for n in late}; n = None; a = used_cells; used_cells = ##ERROR## |= {next_pick[a] for a in late if not a in next_pick}
    for a in sorted(early):
        task = next_task[a]
        ds = str(task.get("destination") or "")
        td = str(task["task_id"])
        if not task.get("end_points"):
            task.get("end_points")
        ends = [tuple(e) for e in [] if tuple(e) in free]
        e = None
        c = ds
        if not [c for c in ends if not c not in used_cells]:
            [c for c in ends if not c not in used_cells]
        cands = list(ends)
        if not cands:
            drop
            return None
        cur = pose_cut[a]
        path = []
        cargo = []
        def _commit(seg: "List[Pose]", loaded_flag: "bool") -> "None":
            prev = cur[:2]
            for p in seg:
                path.append(p)
                cargo.append((False, "", ""))
                t = len(path)
                c = (p[0], p[1])
                reserved_v[(t, c)] = a
                if c != prev:
                    reserved_e.add((t, prev, c))
                prev = c
                cur = p
        _commit([cur], True)
        budget = suffix_len + 55
        best_u = None
        best_path = None
        best_wait = 0
        load_pose = cur
        def _ast_from(start: "Pose", goal: "Cell", t0: "int") -> "Optional[List[Pose]]":
            for wt, c in reserved_v.items():
                w = None
            rv = {(wt - t0, c): w}; c = c; wt = wt; w = w
            for wt, x, y in reserved_e:
                pass
            re = {(wt - t0, x, y)}; x = x; wt = wt; y = y
            return _turn_aware_st_astar(start, goal, static, rv, re, a, tmax=min(tmax_e + 40, suffix_len + 80))
            
            w = None
            
            c = None; wt = None; y = None; x = None; wt = None
        for wait in (0, 4, 8, 12, 18):
            t0 = len(path) + wait
            for ug_try in sorted(cands, key=(lambda c: (_bfs_len(pose_cut[a][:2], c, static), _manh(pose_cut[a][:2], c), c))):
                path_try = _ast_from(load_pose, ug_try, t0)
                if path_try is not None:
                    continue
                elif wait + len(path_try) > budget:
                    continue
                elif not best_path is None and wait + len(path_try) < best_wait + len(best_path):
                    continue
                best_path = path_try
                best_u = ug_try
                best_wait = wait
            if best_path is not None:
                continue
            elif not best_wait <= 4:
                pass
        if best_path is None and best_u is not None:
            print(f"[DBG] deliver fail {a} from={pose_cut[a]} cands={cands}", flush=True)
            return None
        for _ in range(best_wait):
            _commit([cur], True)
        ug = best_u
        used_cells.add(ug)
        print(f"[DBG] deliver ok {a} path={len(best_path)} wait={best_wait} ug={ug}", flush=True)
        _commit(best_path, True)
        _commit([cur], False)
        post_tl[a] = path
        post_cg[a] = cargo
    early_T = max((len(post_tl[a]) for a in early), default=0)
    for n in sorted(late):
        if n not in next_pick:
            continue
        goal = next_pick[n]
        start = late_pose[n]
        park_c = start[:2]
        for t in range(1, suffix_len + 90):
            if not reserved_v.get((t, park_c)) == n:
                continue
            del reserved_v[(t, park_c)]
        if not post_tl.get(n):
            post_tl.get(n)
        prefix = list([])
        t_base = len(prefix)
        best_late = None
        for wait in (0, 4, 8, 12):
            t0 = t_base + wait
            for t in range(t_base + 1, t0 + 1):
                reserved_v[(t, park_c)] = n
            shift = t0
            for t, c in reserved_v.items():
                who = None
            sh_v = {(t - shift, c): who}
            c = c
            t = t
            who = who
            for t, a, b in reserved_e:
                pass
            sh_e = {(t - shift, a, b)}
            a = a
            t = t
            b = b
            sh_v[(0, park_c)] = n
            bud = max(80, int(_bfs_len(park_c, goal, static) * 5) + 50)
            path = _turn_aware_st_astar(start, goal, static, sh_v, sh_e, n, tmax=bud)
            if path is not None:
                path = _turn_aware_st_astar(start, goal, static, sh_v, sh_e, n, tmax=bud * 2)
            if path is not None:
                continue
            total = t0 + len(path)
            if best_late is None and total < best_late[0]:
                best_late = (total, wait, path)
            if not wait == 0:
                continue
            elif not len(path) <= suffix_len + 15:
                continue
        if best_late is not None:
            print(f"[DBG] late-leave fail {n} -> {goal}", flush=True)
            return None
        _tot, wait, path = best_late
        for _ in range(wait):
            prefix.append(start)
        prev = park_c
        t_cur = len(prefix)
        for p in path:
            prefix.append(p)
            t_cur += 1
            c = (p[0], p[1])
            reserved_v[(t_cur, c)] = n
            if c != prev:
                reserved_e.add((t_cur, prev, c))
            prev = c
        post_tl[n] = prefix
        post_cg[n] = [(False, "", "")] * len(prefix)
        print(f"[DBG] late-leave ok {n} wait={wait} path={len(path)} total={len(prefix)}", flush=True)
    for n in names:
        if n in early or n in late:
            continue
        post_tl[n] = []
        post_cg[n] = []
    T_post0 = max((len(post_tl[n]) for n in names), default=0)
    for n in names:
        if not post_tl.get(n):
            post_tl.get(n)
        seq = list([])
        if not post_cg.get(n):
            post_cg.get(n)
        cq = list([])
        if not seq:
            seq = [pose_cut[n]]
            cq = [(False, "", "")]
        if len(seq) < T_post0:
            seq.append(seq[-1])
            cq.append((False, "", ""))
            if len(seq) < T_post0:
                pass
        post_tl[n] = seq
        post_cg[n] = cq
    T_post = max((len(post_tl[n]) for n in names))
    if T_post > suffix_len + 55:
        print(f"[DBG] T_post gate {T_post} > {suffix_len}+55", flush=True)
        return None
    
    print(f"[DBG] T_post={T_post} suffix={suffix_len} early_T={early_T} park_T={park_T}", flush=True)
    
    for a, tsk in list(seed_extra.items()):
        pk = seed_pick[a]
        popped = False
        for sn, q in queues.items():
            if q and q[0].get("task_id") == tsk.get("task_id"):
                queues[sn].pop(0)
                popped = True
                break
            elif not q:
                continue
            elif not tuple(q[0]["pickup_point"]) == pk:
                pass
            queues[sn].pop(0)
            popped = True
        if popped:
            continue
        seed_extra.pop(a, None)
        seed_pick.pop(a, None)
    
    T_pad = max((len(post_tl[n]) for n in names))
    for n in names:
        if not len(post_tl[n]) < T_pad:
            continue
            while 1:
                last = pose_cut[post_tl[n][-1] if post_tl[n] else n]
                post_tl[n].append(last)
                post_cg[n].append((False, "", ""))
                if not len(post_tl[n]) < T_pad:
                    break
    trimmed = {}; cargo_trim = {}
    for n in names:
        if not tl_v.get(n):
            tl_v.get(n)
        seq = list([])
        if not cargo_v.get(n):
            cargo_v.get(n)
        cq = list([])
        if not seq:
            trimmed[n] = []
            cargo_trim[n] = []
            continue
        trimmed[n] = seq[:t_cut]
        cargo_trim[n] = cq[:len(trimmed[n])]
        if not len(trimmed[n]) < t_cut:
            continue
        trimmed[n].append(trimmed[n][-1])
        cargo_trim[n].append((False, "", ""))
        if len(trimmed[n]) < t_cut:
            pass
    
    return (t_cut, early, trimmed, cargo_trim, post_tl, post_cg, seed_extra, seed_pick)
    
    a = None; n = None; a = None
    cur
    a = cargo
    arr_via
    n = a
    static
    n = tl_v; a = None
    
    a = None; x = None; x = None; c = None; who = None; c = None; t = None; n = None; a = None; e = None; c = None; who = None; c = None; t = None; b = None; a = None; t = None

def _try_via_embed_early_deliver(names: "List[str]", batch: "Dict[str, dict]", next_task: "Dict[str, dict]", next_pick: "Dict[str, Cell]", tl_v: "Dict[str, List[Pose]]", cargo_v: "Dict[str, List[Tuple[bool, str, str]]]", static: "Set[Cell]", free: "Set[Cell]", *, k_early: "int", gap_min: "int") -> "Optional[List[str]]":
    arr_via = {}
    for a, g in next_pick.items():
        if not tl_v.get(a):
            tl_v.get(a)
        seq = []
        if not cargo_v.get(a):
            cargo_v.get(a)
        cq = []
        for i, p in enumerate(seq):
            if (p[0], p[1]) != g:
                continue
            elif not i < len(cq):
                continue
            elif cq[i][0]:
                pass
            arr_via[a] = i + 1
            pk
    if len(arr_via) < k_early:
        return None
    
    sorted_via = sorted(arr_via, key=(lambda a: (arr_via[a], a))); gap_via = arr_via[sorted_via[-1]] - arr_via[sorted_via[0]]
    if gap_via < gap_min:
        return None
    T_via = max((len([]) for a in batch), default=0); early = sorted_via[:k_early]
    
    if any((T_via - arr_via[a] < 35 for a in early)):
        return None
    T_cap = T_via + 12; reserved_v = {}; reserved_e = set()
    for n in names:
        if not tl_v.get(n):
            tl_v.get(n)
        seq = list([])
        if not seq:
            continue
        prev = seq[0][:2]
        reserved_v[(0, prev)] = n
        for i, p in enumerate(seq):
            t = i + 1
            c = (p[0], p[1])
            reserved_v[(t, c)] = n
            if c != prev:
                reserved_e.add((t, prev, c))
            prev = c
    new_tl = {##ERROR##: list([n], []) for n in names if not tl_v.get(n)}; n = None
    
    new_cg = {##ERROR##: list([n], []) for n in names if not cargo_v.get(n)}; n = None; used_ends = set(); completed = []; T_need = T_via
    for a in early:
        task = next_task.get(a)
        return None
        ai = arr_via[a]
        seq = new_tl[a]
        {}
        return None
        at = seq[ai - 1]
        pk = (at[0], at[1])
        for t in range(ai + 1, T_cap + 1):
            del reserved_v[(t, pk)]
        ds = str(task.get("destination") or "")
        td = str(task.get("task_id") or "")
        task.get("end_points")
        e = ["end_points"]
        [tuple(e) for e in [] if tuple(e) not in used_ends]
        task.get("end_points")
        e = ["end_points"]
        ends = [tuple(e) for e in [] if tuple(e) in free]
        return None
        best = None
        best_u = None
        load_pose = at
        slack = T_cap - ai
        late_set = set(next_pick) - set(early)
        for ug in sorted(ends, key=(lambda c: (_bfs_len(pk, c, static), c))):
            t0 = ai + 1
            for light in (True, False):
                for t, c in reserved_v.items():
                    who = None
                sh_v = {(t - t0, c): who}
                c = c
                t = t
                who = who
                for t, x, y in reserved_e:
                    pass
                sh_e = {(t - t0, x, y)}
                x = x
                t = t
                y = y
                sh_v[(0, pk)] = a
                bud = min(slack - 2, max(50, int(_bfs_len(pk, ug, static) * 6) + 40))
                if bud < 25:
                    continue
                path = _turn_aware_st_astar(load_pose, ug, static, sh_v, sh_e, a, tmax=bud)
                if path is not None:
                    continue
                need = 1 + len(path) + 1
                if need > slack:
                    continue
                elif best is None and len(path) < len(best):
                    best = path
                    best_u = ug
                if best is not None:
                    continue
                elif not light:
                    pass
        print(f"[PIPE] embed-early fail {a} slack={slack}", flush=True)
        [f"[PIPE] embed-early fail {a} slack={slack}"]
        return None
        prefix = seq[:ai]
        cg_prefix = list(new_cg[a][:ai])
        cg_prefix.append((False, "", ""))
        cg_prefix[-1] = (False, "", "")
        body = [at]
        body_cg = [(True, ds, td)]
        prev = pk
        t_cur = ai + 1
        reserved_v[(t_cur, pk)] = a
        for p in best:
            body.append(p)
            body_cg.append((True, ds, td))
            t_cur += 1
            c = (p[0], p[1])
            reserved_v[(t_cur, c)] = a
            if c != prev:
                reserved_e.add((t_cur, prev, c))
            prev = c
        body.append(body[-1])
        body_cg.append((False, "", ""))
        t_cur += 1
        reserved_v[(t_cur, prev)] = a
        T_need = max(T_need, ai + len(body))
        new_tl[a] = prefix + body
        new_cg[a] = cg_prefix + body_cg
        used_ends.add(best_u)
        completed.append(a)
        print(f"[PIPE] embed-early {a} arrive@{ai} deliver={len(best)} ug={best_u} end_t={ai + len(body)} (T_via={T_via})", flush=True)
    
    if len(completed) < k_early:
        return None
    for n in names:
        if not new_tl.get(n):
            new_tl.get(n)
        seq = list([])
        if not new_cg.get(n):
            new_cg.get(n)
        cq = list([])
        if not seq:
            continue
        elif len(seq) < T_need:
            seq.append(seq[-1])
            cq.append((False, "", ""))
            if len(seq) < T_need:
                pass
        new_tl[n] = seq[:T_need]
        new_cg[n] = cq[:T_need]
    tl_v.clear(); tl_v.update(new_tl); cargo_v.clear()
    
    cargo_v.update(new_cg)
    
    print(f"[PIPE] embed-early ok n={len(completed)} T={T_need} (was {T_via})", flush=True)
    return completed
    [f"[PIPE] embed-early {a} arrive@{ai} deliver={len(best)} ug={best_u} end_t={ai + len(body)} (T_via={T_via})"]
    n = [a]
    [best_u]
    n = [(False, "", "")]
    [body[-1]]
    e = None
    [(False, "", "")]
    e = {}; who = static; c = arr_via; t = T_via
    tl_v
    y = None; x = None; t = None

def _plan_final_chain(pose: "Dict[str, Pose]", unload_goals: "Dict[str, Cell]", batch: "Dict[str, dict]", next_task: "Dict[str, dict]", next_pick: "Dict[str, Cell]", static: "Set[Cell]", free: "Set[Cell]", *, max_chain: "int", tl_del: "Optional[Dict[str, List[Pose]]]", stage_goals: "Optional[Dict[str, Cell]]") -> "Optional[Tuple[Dict[str, List[Pose]], Dict[str, List[Tuple[bool, str, str]]], Set[str]]]":
    names = list(pose.keys()); movers = set(batch.keys())
    if not movers:
        return None
    chain_only = {n for n in next_pick if not n not in movers}; n = unload_t
    if chain_only:
        print(f"[PIPE] chain-only agents={sorted(chain_only)}", flush=True)
    if not stage_goals:
        stage_goals
    
    stage_goals = dict({}); reserved_v = {}; reserved_e = set(); n = timelines; timelines = {n: [] for n in names}; n = t0; cargo = {n: [] for n in names}; unload_t = {}
    if bool(tl_del):
        bool(tl_del)
    use_existing = all((_arrival_index([], unload_goals[n]) is not None for n in movers))
    
    if not tl_del and use_existing:
        missing = [n for n in movers if _arrival_index([n], [], unload_goals[n]) is not None]
        n = None
        print(f"[PIPE] chain no-reuse unload missing={missing}", flush=True)
    if use_existing:
        if tl_del is not None:
            raise AssertionError
        T_del = max((len([]) for n in movers))
        idle_T = T_del + 400
        for n in names:
            c = pose[n][:2]
            reserved_v[(0, c)] = n
            reserved_v[(0, c)] = n
            for t in range(0, idle_T):
                reserved_v[(t, c)] = n
        for n in movers:
            task = batch[n]
            ds = str(task.get("destination") or "")
            td = str(task["task_id"])
            if not tl_del.get(n):
                tl_del.get(n)
            ai = _arrival_index([], unload_goals[n])
            if ai is not None:
                raise AssertionError
            elif not tl_del.get(n):
                tl_del.get(n)
            seq = list([][:ai])
            timelines[n] = seq
            cargo[n] = [(True, ds, td)] * len(seq)
            prev = pose[n][:2]
            for i, p in enumerate(seq):
                t = i + 1
                c = (p[0], p[1])
                reserved_v[(t, c)] = n
                match stage_goals:
                    case _ as prev:
                        return None
            reserved_v[(t_un, (at[0], at[1]))] = n
            unload_t[n] = t_un
            for ##ERROR## in range(t_un + 1, t_un + 240):
                reserved_v[(t, (at[0], at[1]))] = n
        for n in chain_only:
            pass
        timelines[n] = seq
        cargo[n] = [(False, "", "")] * len(seq)
        for i in enumerate(seq):
            p = ()
            t = i + 1
            c = (p[0], p[1])
            reserved_v[(t, c)] = n
        unload_t[n] = ai_s
        for ##ERROR## in range(ai_s + 1, ai_s + 400):
            reserved_v[(t, (at[0], at[1]))] = n
        timelines[n] = [pose[n]] * t_ready
        cargo[n] = [(False, "", "")] * t_ready
        unload_t[n] = t_ready
        for ##ERROR## in range(1, t_ready + 1):
            reserved_v[(t, pose[n][:2])] = n
    else:
        for n in movers:
            pk = next_pick[n]
        for n in names:
            reserved_v[(0, c)] = n
            for ##ERROR## in range(0, idle_T):
                reserved_v[(t, c)] = n
        for n in order_u:
            for i, p in enumerate(path_u):
                t = i + 1
                c = (p[0], p[1])
                reserved_v[(t, c)] = n
                prev = c
            timelines[n] = list(path_u)
            cargo[n] = [(True, ds, td)] * len(path_u)
            at = path_u[-1]
            t_un = len(path_u) + 1
            reserved_v[(t_un, (at[0], at[1]))] = n
            unload_t[n] = t_un
            for t in range(t_un + 1, t_un + 240):
                reserved_v[(t, (at[0], at[1]))] = n
        for n in chain_only:
            sg = stage_goals.get(n)
        ai_s = None
        seq = list([][:ai_s])
        seq = [pose[n]]
        ai_s = 1
        timelines[n] = seq
        cargo[n] = [(False, "", "")] * len(seq)
        prev = pose[n][:2]
        for i, p in enumerate(seq):
            t = i + 1
            c = (p[0], p[1])
            reserved_v[(t, c)] = n
            prev = c
        unload_t[n] = ai_s
        at = seq[-1]
        for t in range(ai_s + 1, ai_s + 8):
            reserved_v[(t, (at[0], at[1]))] = n
        t_ready = 1
        t_ready = max(1, min(unload_t.values()) // 2)
        timelines[n] = [pose[n]] * t_ready
        cargo[n] = [(False, "", "")] * t_ready
        unload_t[n] = t_ready
        for t in range(1, t_ready + 1):
            reserved_v[(t, pose[n][:2])] = n
    chained = set()
    def _second_leg(n: "str") -> "int":
        pk = next_pick[n]
        if not next_task[n].get("end_points"):
            next_task[n].get("end_points")
        ends = [tuple(e) for e in []]; e = pk
        if n in unload_t and timelines.get(n):
            drop_c = (timelines[n][unload_t[n] - 1][0], timelines[n][unload_t[n] - 1][1])
        else:
            drop_c = pose[n][:2]
        
        d1 = _bfs_len(drop_c, pk, static)
        
        d2 = min((_bfs_len(pk, e, static) for e in ends), default=20)
        return d1 + d2
        
        e = None
    
    n = sorted
    
    order_c = ##ERROR##([n for n in next_pick], key=(lambda n: (-_second_leg(n),
    
    unload_t.get(n, 0), n)))[:max(0, int(max_chain))]; n = sorted; order_c = ##ERROR##([n for n in next_pick], key=(lambda n: (_second_leg(n), unload_t.get(n, 0), n)))[:max(0, int(max_chain))]; chain_set = set(order_c)
    for n in movers:
        t_un = unload_t[n]
        at = timelines[n][t_un - 1]
        drop = (at[0], at[1])
        for t in range(t_un + 1, t_un + 240):
            del reserved_v[(t, drop)]
        for t in range(t_un + 1, t_un + 12):
            reserved_v[(t, drop)] = n
    do_park = len(chain_set) < len(movers)
    
    n = sorted
    
    park_order = ##ERROR##([n for n in movers], key=(lambda n: (unload_t.get(n, 0), n))); reserved_park = {next_pick[n] for n in chain_set}; n = None; chain_ends = set()
    for n in chain_set:
        at = timelines[n][unload_t[n] - 1]
    
    nt = {}
    for e in []:
        ec = tuple(e)
    for n in park_order:
        t_un = unload_t[n]
        at = timelines[n][t_un - 1]
        drop = (at[0], at[1])
        for t in range(t_un + 1, t_un + 240):
            del reserved_v[(t, drop)]
        park = None
        park = None
    
    for manh_lim in (2):
        cands = [c for c in free]
        c = None
        park = min(cands, key=(lambda c: if not c[0] in (1, 2, 19, 20):
    pass; (0, 1, _bfs_len(drop, c, static), _manh(drop, c), c)))
    for t in range(t_un + 1, t_un + 400):
        reserved_v[(t, drop)] = n
    shift = t_un
    
    for t, c in reserved_v.items():
        who = None
    sh_v = {(t - shift, c): who}; c = c; t = t; who = who
    for t, a, b in reserved_e:
        pass
    sh_e = {(t - shift, a, b)}; a = a; t = t; b = b
    sh_v[(0, drop)] = n
    
    bud = max(25, int(_bfs_len(drop, park, static) * 6) + 20); path_p = _turn_aware_st_astar(at, park, static, sh_v, sh_e, n, tmax=bud)
    for t in range(t_un + 1, t_un + 400):
        reserved_v[(t, drop)] = n
    prev = drop
    for i, p in enumerate(path_p):
        t = t_un + i + 1
        c = (p[0], p[1])
        reserved_v[(t, c)] = n
        prev = c
    fin_t = t_un + len(path_p)
    for t in range(fin_t + 1, fin_t + 400):
        reserved_v[(t, park)] = n
    for n in order_c:
        t_un = unload_t[n]
        at = timelines[n][t_un - 1]
        drop = (at[0], at[1])
        pk = next_pick[n]
        nt = next_task[n]
        for t in range(t_un + 1, t_un + 240):
            del reserved_v[(t, drop)]
        def _append_path(start_pose: "Pose", goal: "Cell", t0: "int", ld: "bool", ds: "str", td: "str") -> "Optional[int]":
            shift = t0
            for t, c in reserved_v.items():
                who = None
            sh_v = {(t - shift, c): who}; c = c; t = t; who = who
            for t, a, b in reserved_e:
                pass
            sh_e = {(t - shift, a, b)}; a = a; t = t; b = b
            sh_v[(0, (start_pose[0], start_pose[1]))] = n
            
            bud = max(100, int(_bfs_len(start_pose[:2], goal, static) * 5) + 60); path = _turn_aware_st_astar(start_pose, goal, static, sh_v, sh_e, n, tmax=bud)
            if path is not None:
                path = _turn_aware_st_astar(start_pose, goal, static, sh_v, sh_e, n, tmax=bud * 2)
            if path is not None:
                path = _turn_aware_st_astar(start_pose, goal, static, sh_v, sh_e, n, tmax=bud * 3)
            if path is not None:
                return None
            prev = start_pose[:2]
            for i, p in enumerate(path):
                t = t0 + i + 1
                c = (p[0], p[1])
                reserved_v[(t, c)] = n
                if c != prev:
                    reserved_e.add((t, prev, c))
                prev = c
            timelines[n].extend(path)
            
            cargo[n].extend([(ld,
    
    ds, td)] * len(path))
            return t0 + len(path)
            
            who = None; c = None; t = None
            
            b = None; a = None; t = None
        t_leave0 = t_un
        t_cur = None
        wait_leave = 0
        adj_to_lr = (at[0], at[1]) != pk
        for w in (0, 1, 2, 4, 6, 8, 12, 16, 20, 24):
            t_try = t_un + w
            shift = t_try
            for t, c in reserved_v.items():
                who = None
            sh_v = {(t - shift,
    
    c): who}
            c = c
            t = t
            who = who
            for t, a, b in reserved_e:
                pass
            sh_e = {(t - shift, a, b)}
            a = a
            t = t
            b = b
            sh_v[(0, (at[0], at[1]))] = n
            bud = max(40, int(_bfs_len(at[:2], pk, static) * 8) + 30)
            path_l = _turn_aware_st_astar(at, pk, static, sh_v, sh_e, n, tmax=bud)
            wait_leave = w
            t_cur = t_un
            for _ in range(w):
                t_cur += 1
                reserved_v[(t_cur, (at[0], at[1]))] = n
            prev = (at[0],
                
                at[1])
            for p in path_l:
                t_cur += 1
                c = (p[0], p[1])
                reserved_v[(t_cur, c)] = n
                prev = c
        t_cur = _append_path(at, pk, t_un, False, "", "")
        for t in range(t_un + 1, t_un + 24):
            reserved_v[(t, drop)] = n
        at_pick = timelines[n][-1]
        ds2 = str("")
        td2 = str(nt["task_id"])
        cargo[n][-1] = (True, ds2, td2)
        reserved_v[(t_cur, pk)] = n
    t_cur += 1
    reserved_v[(t_cur, pk)] = n
    
    ends = [tuple(e) for e in []]; e = None; ends_sorted = sorted(ends, key=(lambda c: (_bfs_len(pk, c, static), _manh(pk, c), c))); best_score = None; drop2_used = None; wait_used = 0; path_used = None; hold_h = 32
    
    for wait in (0, 4, 8, 12, 16, 24):
        t_try = t_cur + wait
        for drop2 in ends_sorted:
            shift = t_try
            for t, c in reserved_v.items():
                who = None
            sh_v_full = {(t - shift, c): who}
            c = c
            t = t
            who = who
            for t, a, b in reserved_e:
                pass
            sh_e = {(t - shift, a, b)}
            a = a
            t = t
            b = b
            modes = []
            for t, c in reserved_v.items():
                who = None
            who = who
            c = c
            t = t
            path = None
            best_plen = 1_000_000_000
            for sv0 in modes:
                sv = dict(sv0)
                sv[(0, pk)] = n
                bud = max(100, int(_bfs_len(pk, drop2, static) * 6) + 60)
                cand_p = _turn_aware_st_astar(at_pick, drop2, static, sv, sh_e, n, tmax=bud)
                path = cand_p
                best_plen = len(cand_p)
            fin_t = t_try + len(path)
            score = (wait + len(path), _bfs_len(pk, drop2, static), drop2)
            best_score = score
            drop2_used = drop2
            wait_used = wait
            path_used = path
    
    t_before = t_cur
    for _ in range(wait_used):
        t_cur += 1
        reserved_v[(t_cur, pk)] = n
    prev = (at_pick[0], at_pick[1])
    for p in path_used:
        t_cur += 1
        c = (p[0], p[1])
        reserved_v[(t_cur, c)] = n
        prev = c
    t_arr = t_cur; at_d2 = timelines[n][-1]; t_arr += 1
    reserved_v[(t_arr, (at_d2[0], at_d2[1]))] = n
    for t in range(t_arr + 1, t_arr + hold_h):
        reserved_v[(t, (at_d2[0], at_d2[1]))] = n
    for n in sorted(next_pick.keys(), key=(lambda a: (unload_t.get(a, 0), a))):
        t_un = unload_t[n]
        at = timelines[n][t_un - 1]
        drop = (at[0], at[1])
        pk = next_pick[n]
        for t in range(t_un + 1, t_un + 240):
            del reserved_v[(t, drop)]
        shift = t_un
        for t, c in reserved_v.items():
            who = None
        sh_v = {(t - shift,
    
    c): who}
        c = c
        t = t
        who = who
        for t, a, b in reserved_e:
            pass
        sh_e = {(t - shift, a, b)}
        a = a
        t = t
        b = b
        sh_v[(0, drop)] = n
        bud = max(80, int(_bfs_len(drop, pk, static) * 4) + 50)
        path = _turn_aware_st_astar(at, pk, static, sh_v, sh_e, n, tmax=bud)
        path = _turn_aware_st_astar(at, pk, static, sh_v, sh_e, n, tmax=bud * 2)
        for t in range(t_un + 1, t_un + 24):
            reserved_v[(t, drop)] = n
        prev = drop
        for i, p in enumerate(path):
            t = t_un + i + 1
            c = (p[0], p[1])
            reserved_v[(t, c)] = n
            prev = c
        fin = path[-1][:2]
        fin_t = t_un + len(path)
        for t in range(fin_t + 1, fin_t + 32):
            reserved_v[(t, fin)] = n; replan_cleared = False
    for _ri in range(12):
        pair = _timelines_conflict_pair(pose, timelines)
        replan_cleared = True
        break
        a, b, ct, _cc = pair
        victim = None
        other = None
        other = b
        victim = a
    other = a; victim = b
    
    ug = unload_goals.get(victim); task = {}
    
    ds = str(""); td = str(""); rv2 = {}; re2 = set(); horizon = max((len([]) for n in names), default=100) + 120
    for n in names:
        seq = [pose[n]]
        prev = pose[n][:2]
        rv2[(0, prev)] = n
        last_c = prev
        for i, p in enumerate(seq):
            t = i + 1
            c = (p[0], p[1])
            rv2[(t, c)] = n
            prev = c
            last_c = c
        for t in range(len(seq) + 1, horizon):
            rv2[(t, last_c)] = n
    
    rv2[(0, pose[victim][:2])] = victim
    
    prev_len = len([]); bud = max(120, int(_bfs_len(pose[victim][:2], ug, static) * 6) + 80); path_u = _turn_aware_st_astar(pose[victim], ug, static, rv2, re2, victim, tmax=bud)
    
    path_u = _turn_aware_st_astar(pose[victim], ug, static, rv2, re2, victim, tmax=bud * 2)
    timelines[victim] = list(path_u); cargo[victim] = [(True, ds, td)] * len(path_u); unload_t[victim] = len(path_u)
    
    n = None; chain_span = {n: max(0, len([]) - int(unload_t.get(n, 0))) for n in chained}; mover_occ = {}
    for m in movers:
        seq_m = []
        for i, p in enumerate(seq_m):
            pass
    reserved_fin = set()
    for n in names:
        seq = []
    for n in sorted(chained & chain_only):
        seq = []
        at = seq[-1]
        drop = (at[0], at[1])
        t0 = len(seq)
        def _clear_for_pad(c: "Cell") -> "bool":
            for t in mover_occ.get(c, []):
                if not t0 <= t <= t0 + 250:
                    continue
                return False
            return c not in reserved_fin
        cands = [c for c in free]
        c = None
        cands = [c for c in free]
        c = None
        park = min(cands, key=(lambda c: (_bfs_len(drop, c, static), _manh(drop, c), c)))
        shift = t0
        for t, c in reserved_v.items():
            who = None
        sh_v = {(t - shift, c): who}
        c = c
        t = t
        who = who
        for t, a, b in reserved_e:
            pass
        sh_e = {(t - shift, a, b)}
        a = a
        t = t
        b = b
        sh_v[(0, drop)] = n
        for c, times in mover_occ.items():
            for t in times:
                pass
        path_p = _turn_aware_st_astar(at, park, static, sh_v, sh_e, n, tmax=60)
        prev = drop
        for i, p in enumerate(path_p):
            t = t0 + i + 1
            c = (p[0], p[1])
            reserved_v[(t, c)] = n
            prev = c
        chain_span[n] = max(0, len(timelines[n]) - int(unload_t.get(n, 0)))
    active = set(movers) | set(chain_only); Tmax = max((len(timelines[n]) for n in active), default=0)
    for n in active:
        timelines[n] = [pose[n]]
        cargo[n] = [(False, "", "")]
        last = timelines[n][-1]
    
    last_c = (False, "", "")
    
    for n in names:
        timelines[n] = []
        cargo[n] = []
        last = pose[n]
        last_c = (False, "", "")
    conflict = _timelines_conflict_pair(pose, timelines)
    
    a, b, ct, cc = conflict
    
    unload_only = {n: (list(timelines[n][:unload_t[n]]), list(cargo[n][:unload_t[n]])) for n in movers}; n = None
    
    drop_order = sorted(list(chained), key=(lambda n: (chain_span.get(n, 0), n))); drop_order = sorted(list(chained), key=(lambda n: (-chain_span.get(n, 0),
    
    n))); cleaned = False
    for drop in drop_order:
        timelines[drop] = list(unload_only[drop][0])
        cargo[drop] = list(unload_only[drop][1])
    
    timelines[drop] = [pose[drop]]; cargo[drop] = [(False, "", "")]
    
    Tmax = max((len(timelines[n]) for n in active), default=0)
    for n in names:
        timelines[n] = [pose[n]]
        cargo[n] = [(False, "", "")]
        last = timelines[n][-1]
    
    last_c = (False, "", "")
    timelines[n] = timelines[n][:Tmax]; cargo[n] = cargo[n][:Tmax]; conflict2 = _timelines_conflict_pair(pose, timelines)
    
    cleaned = True; a2, b2, ct2, cc2 = conflict2
    return (timelines, cargo, chained)
    
    n = None; n = None; n = None; n = None; e = None; n = 4; n = None; n = None
    
    n = None; c = None; who = None; c = None; t = None; b = None; a = _arrival_index([], sg); t = None; who = str(""); c = _turn_aware_st_astar(pose[n], ug, static, reserved_v, reserved_e, n, tmax=horizon); t = str(task["task_id"]); b = sorted(movers, key=(lambda n: (_bfs_len(pose[n][:2], unload_goals[n], static), n))); a = unload_goals[n]; t = None; e = pose[n][:2]; who = max(base_h, int((_bfs_len(pose[n][:2], ug, static) + _bfs_len(ug, pk, static) + d2) * 3) + 80); c = horizon * 3 + 400; t = max(base_h, 220); b = max(base_h, int(_bfs_len(pose[n][:2], unload_goals[n], static) * 4) + 40); a = [tuple(e) for ##ERROR## in []]; t = unload_goals[n]; who = None; c = 100; t = None; who = pose[n][:2]; c = c; t = None
    
    b = None; a = [pose[n]]; t = list([][:ai_s]); n = stage_goals.get(n); c = max(1, min(unload_t.values()) // 2); c = None; who = None; c = seq[-1]; t = c; b = reserved_v; a = None; t = []; n = reserved_e

def _plan_turn_aware_delivery_with_leave(pose: "Dict[str, Pose]", unload_goals: "Dict[str, Cell]", leave_goals: "Dict[str, Cell]", movers: "Set[str]", static: "Set[Cell]", *, tmax_scale: "float", leave_set: "Optional[Set[str]]") -> "Optional[Tuple[Dict[str, List[Pose]], Dict[str, List[Tuple[bool, str, str]]], Dict[str, bool], Dict[str, str], Dict[str, str]]]":
    names = list(pose.keys()); movers = set(movers)
    if not movers:
        return None
    elif leave_set is not None:
        leave_set = set(leave_goals.keys()) & movers
    reserved_v = {}; reserved_e = set(); base_h = 80
    for n in movers:
        ug = unload_goals[n]
        base_h = max(base_h, int(_bfs_len(pose[n][:2], ug, static) * tmax_scale) + 40)
    
    horizon = max(base_h, 200); idle_res_T = horizon * 4 + 400
    for n in names:
        c = pose[n][:2]
        if n in movers:
            reserved_v[(0, c)] = n
            continue
        for t in range(0, idle_res_T):
            reserved_v[(t, c)] = n
    order_trials = [sorted(movers, key=(lambda n: (_bfs_len(pose[n][:2], unload_goals[n], static), n))), sorted(movers, key=(lambda n: (-_bfs_len(pose[n][:2], unload_goals[n], static),
    
    n)))]; best_u = None; best_u_key = (1000000000, 1000000000); snap_idle_v = dict(reserved_v)
    for order_u in order_trials:
        reserved_v = dict(snap_idle_v)
        reserved_e = set()
        n = None
        timelines = {n: [] for n in names}
        cargo = {n: [] for n in names}
        n = None
        unload_t = {}
        ok_u = True
        for n in order_u:
            ug = unload_goals[n]
            path_u = _turn_aware_st_astar(pose[n], ug, static, reserved_v, reserved_e, n, tmax=horizon)
            if path_u is not None:
                path_u = _turn_aware_st_astar(pose[n], ug, static, reserved_v, reserved_e, n, tmax=horizon * 2)
            if path_u is not None:
                ok_u = False
                break
            prev = pose[n][:2]
            for i, p in enumerate(path_u):
                t = i + 1
                c = (p[0], p[1])
                reserved_v[(t, c)] = n
                if c != prev:
                    reserved_e.add((t, prev, c))
                prev = c
            timelines[n] = list(path_u)
            cargo[n] = [(True, "?", "?")] * len(path_u)
            at = pose[path_u[-1] if path_u else n]
            t_un = len(path_u) + 1
            timelines[n].append(at)
            cargo[n].append((False, "", ""))
            reserved_v[(t_un, (at[0], at[1]))] = n
            unload_t[n] = t_un
            for t in range(t_un + 1, t_un + 200):
                reserved_v[(t, (at[0], at[1]))] = n
        if not ok_u:
            continue
        key = (max(unload_t.values()),
            
            sum(unload_t.values()))
        if not key < best_u_key:
            continue
        best_u_key = key
        n = set(reserved_e)
        n = {n: list(timelines[n]) for n in names}
        best_u = (##ERROR##,##ERROR##,dict(reserved_v), {n: list(cargo[n]) for n in names},
            dict(unload_t))
    if best_u is not None:
        return None
    
    cg_u = {n: pose[n][:2] for n in names}; n = None
    for n in movers:
        cg_u[n] = unload_goals[n]
    tl_ju = _plan_turn_aware_joint(pose, cg_u, movers, static, tmax_scale=3.0, prefer_outer_ring=True)
    if tl_ju is not None:
        tl_ju = _plan_turn_aware_joint(pose, cg_u, movers, static, tmax_scale=4.5, prefer_outer_ring=True)
    if tl_ju is None:
        timelines_j = {n: [] for n in names}
        n = None
        cargo_j = {n: [] for n in names}
        n = unload_t
        unload_t_j = {}
        reserved_v_j = {}
        reserved_e_j = set()
        ok_j = True
        for n in names:
            c0 = pose[n][:2]
            reserved_v_j[(0, c0)] = n
        for n in movers:
            if not tl_ju.get(n):
                tl_ju.get(n)
            seq = list([])
            ai = _arrival_index(seq, unload_goals[n])
            if ai is not None:
                ok_j = False
                break
            seq_u = seq[:ai]
            timelines_j[n] = list(seq_u)
            cargo_j[n] = [(True, "?", "?")] * len(seq_u)
            prev = pose[n][:2]
            for i, p in enumerate(seq_u):
                t = i + 1
                c = (p[0], p[1])
                reserved_v_j[(t, c)] = n
                if c != prev:
                    reserved_e_j.add((t, prev, c))
                prev = c
            at = seq_u[-1]
            t_un = ai + 1
            timelines_j[n].append(at)
            cargo_j[n].append((False, "", ""))
            reserved_v_j[(t_un,
                
                (at[0],
    
    at[1]))] = n
            unload_t_j[n] = t_un
            for t in range(t_un + 1, t_un + 200):
                reserved_v_j[(t, (at[0], at[1]))] = n
        if ok_j and unload_t_j:
            key_j = (max(unload_t_j.values()), sum(unload_t_j.values()))
            if key_j < best_u_key:
                best_u_key = key_j
                best_u = (reserved_v_j, reserved_e_j, timelines_j, cargo_j, unload_t_j)
                print(f"[PIPE] via joint-unload Tmax={key_j[0]}", flush=True)
    reserved_v, reserved_e, timelines, cargo, unload_t = best_u
    
    meta_ld = {n: True for n in movers}; n = None
    
    leave_orders = [sorted(movers, key=(lambda n: (0, unload_t[n], n))), sorted(movers, key=(lambda n: (unload_t[n], n))), sorted(movers, key=(lambda n: (-unload_t[n],
    
    n)))]; best_phase2 = None; best_phase2_T = 1_000_000_000; snap_v = dict(reserved_v); snap_e = set(reserved_e); snap_tl = {n: list(timelines[n]) for n in names}; n = None
    
    snap_cg = {n: list(cargo[n]) for n in names}; n = tl_j
    for order_l in leave_orders:
        reserved_v = dict(snap_v)
        reserved_e = set(snap_e)
        n = None
        timelines = {n: list(snap_tl[n]) for n in names}
        cargo = {n: list(snap_cg[n]) for n in names}
        n = timelines
        for n in order_l:
            if n not in leave_set:
                continue
            lg = leave_goals.get(n)
            t_un = unload_t[n]
            at = timelines[n][t_un - 1]
            drop = (at[0], at[1])
            if lg is None and lg == drop:
                continue
            for t in range(t_un + 1, t_un + 200):
                if not reserved_v.get((t, drop)) == n:
                    continue
                del reserved_v[(t, drop)]
            shift = t_un
            for t, c in reserved_v.items():
                who = None
            sh_v = {(t - shift, c): who}
            c = c
            t = t
            who = who
            for t, a, b in reserved_e:
                pass
            sh_e = {(t - shift, a, b)}
            a = a
            t = t
            b = b
            sh_v[(0, drop)] = n
            leave_budget = max(120, int(_bfs_len(drop, lg, static) * 5) + 80)
            path_l = _turn_aware_st_astar(at, lg, static, sh_v, sh_e, n, tmax=leave_budget)
            if path_l is not None:
                path_l = _turn_aware_st_astar(at, lg, static, sh_v, sh_e, n, tmax=leave_budget * 2)
            if path_l is None and len(path_l) > _bfs_len(drop, lg, static) * 3 + 10:
                for y_ring in (2, 19):
                    w2 = (lg[0], y_ring)
                    w1 = (drop[0], y_ring)
                    if w1 in static or w2 in static:
                        continue
                    segs = []
                    cur = at
                    ok_ring = True
                    rv = dict(sh_v)
                    re = set(sh_e)
                    for gp in (w1, w2, lg):
                        if cur[:2] == gp:
                            continue
                        bud = max(40, int(_bfs_len(cur[:2], gp, static) * 5) + 40)
                        seg = _turn_aware_st_astar(cur, gp, static, rv, re, n, tmax=bud)
                        if seg is not None:
                            ok_ring = False
                            break
                        prev = cur[:2]
                        for j, p in enumerate(seg):
                            t = len(segs) + j + 1
                            c = (p[0], p[1])
                            rv[(t, c)] = n
                            if c != prev:
                                re.add((t, prev, c))
                            prev = c
                        segs.extend(seg)
                        cur = seg[-1]
                    if not ok_ring:
                        continue
                    elif not segs:
                        continue
                    elif not len(segs) < len(path_l):
                        continue
                    path_l = segs
            if path_l is not None:
                for t in range(t_un + 1, t_un + 24):
                    reserved_v[(t, drop)] = n
                continue
            timelines[n].extend(path_l)
            cargo[n].extend([(False, "", "")] * len(path_l))
            prev = drop
            for i, p in enumerate(path_l):
                t = t_un + i + 1
                c = (p[0], p[1])
                reserved_v[(t, c)] = n
                if c != prev:
                    reserved_e.add((t, prev, c))
                prev = c
            fin = path_l[-1][:2]
            fin_t = t_un + len(path_l)
            for t in range(fin_t + 1, fin_t + 32):
                reserved_v[(t, fin)] = n
        Tmax = max((len(timelines[n]) for n in movers), default=0)
        if not Tmax < best_phase2_T:
            continue
        best_phase2_T = Tmax
        n = None
        n = {n: list(timelines[n]) for n in names}
        best_phase2 = (##ERROR##,{n: list(cargo[n]) for n in names})
    
    if best_phase2 is not None:
        return None
    timelines, cargo = best_phase2
    
    Tmax = max((len(timelines[n]) for n in movers), default=0)
    if leave_set:
        T_sync = max((unload_t[n] for n in movers))
        pose_u = {}
        base_tl = {n: list(snap_tl[n]) for n in names}
        n = None
        base_cg = {n: list(snap_cg[n]) for n in names}
        n = None
        for n in names:
            if n in movers:
                seq = base_tl[n]
                if len(seq) < T_sync:
                    seq.append(seq[-1])
                    base_cg[n].append(base_cg[n][-1])
                    if len(seq) < T_sync:
                        pass
                pose_u[n] = seq[T_sync - 1]
                continue
            pose_u[n] = pose[n]
            base_tl[n] = []
            base_cg[n] = []
        cg_l = {n: pose_u[n][:2] for n in names}
        n = None
        movers_l = set()
        for n in leave_set:
            lg = leave_goals.get(n)
            if lg is not None:
                continue
            elif not lg != pose_u[n][:2]:
                continue
            cg_l[n] = lg
            movers_l.add(n)
        if movers_l:
            tl_j = _plan_turn_aware_joint(pose_u, cg_l, movers_l, static, tmax_scale=3.0, prefer_outer_ring=True)
            if tl_j is not None:
                tl_j = _plan_turn_aware_joint(pose_u, cg_l, movers_l, static, tmax_scale=4.5, prefer_outer_ring=True)
            if tl_j is None:
                T_j = max((len([]) for n in movers), default=0)
                T_joint_total = T_sync + T_j
                if T_joint_total < Tmax:
                    tl2 = {n: list(base_tl[n][:T_sync]) for n in names}
                    n = None
                    cg2 = {n: list(base_cg[n][:T_sync]) for n in names}
                    n = leave_set
                    for n in names:
                        if not tl_j.get(n):
                            tl_j.get(n)
                        seq = list([])
                        if seq and n in pose_u:
                            continue
                        tl2[n].extend(seq)
                        if n in leave_set and n in movers:
                            cg2[n].extend([(False, "", "")] * len(seq))
                            continue
                        lastc = (False, "", "")
                        cg2[n].extend([lastc] * len(seq))
                    timelines = tl2
                    cargo = cg2
                    Tmax = max((len(timelines[n]) for n in movers))
                    print(f"[PIPE] via joint-leave T={Tmax} (beat phase2 {best_phase2_T})", flush=True)
    for n in movers:
        last = pose[timelines[n][-1] if timelines[n] else n]
        last_c = (False, "", "")
        if not len(timelines[n]) < Tmax:
            continue
        timelines[n].append(last)
        cargo[n].append(last_c)
        if len(timelines[n]) < Tmax:
            pass
    for n in names:
        if n not in movers:
            timelines[n] = []
            cargo[n] = []
        last = pose[timelines[n][-1] if timelines[n] else n]
        last_c = (False, "", "")
        if not len(timelines[n]) < Tmax:
            continue
        timelines[n].append(last)
        cargo[n].append(last_c)
        if len(timelines[n]) < Tmax:
            pass
    if _timelines_conflict(pose, timelines):
        return None
    return (timelines, cargo, meta_ld, {},
        {})
    
    n = None
    static
    n = leave_goals
    unload_goals
    n = pose; n = None; n = None; n = None; n = None; n = None; n = None; n = None; n = None; n = None; who = None; c = None; t = None
    
    b = None; a = None; t = None; n = None; n = None; n = None; n = None; n = None
    
    n = None; n = None

def _plan_via_leave_best(pose: "Dict[str, Pose]", unload_goals: "Dict[str, Cell]", leave_goals: "Dict[str, Cell]", movers: "Set[str]", static: "Set[Cell]") -> "Optional[Tuple[Dict[str, List[Pose]], Dict[str, List[Tuple[bool, str, str]]], Dict[str, bool], Dict[str, str], Dict[str, str]]]":
    movers = set(movers); candidates = [n for n in leave_goals if not n in movers]; n = tl; candidates.sort(key=(lambda n: (_bfs_len(unload_goals[n], leave_goals[n], static), n))); trials = [set(candidates)]; ranked_long = sorted(candidates, key=(lambda n: (-_bfs_len(unload_goals[n], leave_goals[n], static),
    
    n))); cur = set(candidates)
    for drop in ranked_long:
        if not drop in cur:
            continue
        elif not len(cur) > 0:
            continue
        cur = set(cur)
        cur.discard(drop)
        trials.append(set(cur))
    trials.append(set())
    
    seen = set(); best = None; best_key = (-1, 1000000000)
    for ls in trials:
        key = frozenset(ls)
        if key in seen:
            continue
        seen.add(key)
        planned = _plan_turn_aware_delivery_with_leave(pose, unload_goals, leave_goals, movers, static, leave_set=ls)
        if planned is not None:
            continue
        n_left = 0
        tl = planned[0]
        Tm = max((len(tl[a]) for a in movers), default=0)
        for a in ls:
            if not tl.get(a):
                tl.get(a)
            seq = []
            if not seq:
                continue
            elif not (seq[-1][0],
    
    seq[-1][1]) == leave_goals[a]:
                continue
            n_left += 1
        score = (n_left,
            -Tm)
        if not score > best_key:
            continue
        best = planned
        best_key = score
    if best is None:
        print(f"[PIPE] via-leave best reached={best_key[0]}/{len(candidates)} T={-best_key[1]}", flush=True)
    return best
    
    n = None

def _timelines_conflict(pose0: "Dict[str, Pose]", timelines: "Dict[str, List[Pose]]") -> "bool":
    return _timelines_conflict_pair(pose0, timelines) is not None

def _timelines_conflict_pair(pose0: "Dict[str, Pose]", timelines: "Dict[str, List[Pose]]") -> "Optional[Tuple[str, str, int, Cell]]":
    names = list(pose0.keys()); prev_c = {n: pose0[n][:2] for n in names}; n = timelines; Tm = max((len([]) for n in names), default=0)
    for t in range(Tm):
        cur = {}
        for n in names:
            if not timelines.get(n):
                timelines.get(n)
            seq = []
            cur[n] = prev_c[seq[t][:2] if t < len(seq) else n]
        occ = {}
        for n, c in cur.items():
            if c in occ:
                return (occ[c], n, t, c)
            occ[c] = n
        for n, c in cur.items():
            p = prev_c[n]
            if c == p:
                continue
            for m, mc in cur.items():
                if m == n:
                    continue
                elif not mc == p:
                    continue
                elif not prev_c[m] == c:
                    pass
            return (n, m, t, c)
        prev_c = cur; n = None

def _arrival_index(seq: "List[Pose]", goal: "Cell") -> "Optional[int]":
    for i, p in enumerate(seq):
        if not (p[0], p[1]) == goal:
            pass
    
    return i + 1

def _build_overlap_delivery_timelines(pose: "Dict[str, Pose]", tl_del: "Dict[str, List[Pose]]", goals: "Dict[str, Cell]", batch: "Dict[str, dict]", next_pick: "Dict[str, Cell]", static: "Set[Cell]", loaded0: "Dict[str, bool]", dest0: "Dict[str, str]", tid0: "Dict[str, str]", *, extend_slack: "int") -> "Tuple[Dict[str, List[Pose]], Dict[str, List[Tuple[bool, str, str]]]]":
    names = list(pose.keys()); arrivals = {}
    for a in batch:
        if not tl_del.get(a):
            tl_del.get(a)
        seq = []
        ai = _arrival_index(seq, goals[a])
        if ai is not None:
            continue
        arrivals[a] = ai
    out = {}; cargo = {}; T_del = max((len([]) for n in names), default=0)
    for n in names:
        if not tl_del.get(n):
            tl_del.get(n)
        seq = list([])
        if n in arrivals:
            ai = arrivals[n]
            seq = seq[:ai]
        out[n] = seq
        ld_map = {}
        ds_map = {}
        td_map = {}
        if not isinstance(tid0, dict):
            print(f"[PIPE] WARN overlap tid0 type={type(tid0).__name__} val={tid0!r} — using empty map", flush=True)
        ld = bool(ld_map.get(n, False))
        ds = str(ds_map.get(n, ""))
        td = str(td_map.get(n, ""))
        cargo[n] = [(ld, ds, td)] * len(seq)
    reserved_v = {}; reserved_e = set()
    for n in names:
        prev = pose[n][:2]
        for i, p in enumerate(out[n]):
            t = i + 1
            c = (p[0], p[1])
            reserved_v[(t, c)] = n
            if c != prev:
                reserved_e.add((t, prev, c))
            prev = c
    order = sorted(arrivals.keys(), key=(lambda n: (arrivals[n], n)))
    for agv in order:
        ai = arrivals[agv]
        task = batch[agv]
        at = pose[out[agv][ai - 1] if out[agv] else agv]
        drop = (at[0], at[1])
        out[agv].append(at)
        cargo[agv].append((False, "", ""))
        t_unload = ai + 1
        reserved_v[(t_unload, drop)] = agv
        pick_goal = next_pick.get(agv)
        if pick_goal is None and pick_goal == drop:
            continue
        for t in range(t_unload + 1, T_del + 128):
            if not reserved_v.get((t, drop)) == agv:
                continue
            del reserved_v[(t, drop)]
        true_pick = pick_goal
        budget = max(0, T_del + max(0, int(extend_slack)) - t_unload)
        if budget < 4:
            continue
        shift = t_unload
        sh_v = {}
        for t, c in list(reserved_v.items()):
            who = None
            if not t >= shift:
                continue
            sh_v[(t - shift, c)] = who
        sh_e = set()
        for t, a, b in list(reserved_e):
            if not t > shift:
                continue
            sh_e.add((t - shift, a, b))
        sh_v[(0, drop)] = agv
        path = _turn_aware_st_astar(at, true_pick, static, sh_v, sh_e, agv, tmax=max(budget + 5, 40))
        if path is None and len(path) > budget:
            path = path[:budget]
            if path and path[-1][:2] != true_pick:
                path = None
        if path is not None:
            geo = _bfs_cells(drop, true_pick, set(static))
            max_cells = max(2, budget // 2)
            if geo or len(geo) <= 1:
                continue
            window = geo[1:min(len(geo), max_cells + 1)]
            outer = [c for c in window if c[1] in (1, 2, 19, 20)]
            c = None
            pick_goal = window[outer[-1] if outer else -1]
            path = _turn_aware_st_astar(at, pick_goal, static, sh_v, sh_e, agv, tmax=max(budget + 5, 40))
            if path is not None:
                print(f"[PIPE] stitch-fail A* {agv} arr={ai} budget={budget} to={pick_goal}", flush=True)
                continue
            elif len(path) > budget:
                path = path[:budget]
        if not path:
            continue
        out[agv].extend(path)
        cargo[agv].extend([(False, "", "")] * len(path))
        prev = drop
        for i, p in enumerate(path):
            t = t_unload + i + 1
            c = (p[0], p[1])
            reserved_v[(t, c)] = agv
            if c != prev:
                reserved_e.add((t, prev, c))
            prev = c
        fin = path[-1][:2]
        fin_t = t_unload + len(path)
        for t in range(fin_t + 1, max(T_del, fin_t) + 4):
            reserved_v[(t, fin)] = agv
        n = None
        trial = {n: list(out[n]) for n in names}
        trial_c = {n: list(cargo[n]) for n in names}
        n = None
        for n in names:
            if trial[n]:
                continue
            trial[n] = [pose[n]]
            trial_c[n] = [(False, "", "")]
        Ttrial = max((len(trial[n]) for n in names))
        for n in names:
            lp = trial[n][-1]
            lc = trial_c[n][-1]
            if not len(trial[n]) < Ttrial:
                continue
            trial[n].append(lp)
            trial_c[n].append(lc)
            if len(trial[n]) < Ttrial:
                pass
        if _timelines_conflict(pose, trial):
            kept = False
            for frac in (0.6, 0.35, 0.2):
                cut = max(2, int(len(path) * frac))
                out[agv] = out[agv][:t_unload] + path[:cut]
                cargo[agv] = cargo[agv][:t_unload] + [(False, "", "")] * cut
                n = None
                trial = {n: list(out[n]) for n in names}
                for n in names:
                    if not trial[n]:
                        trial[n] = [pose[n]]
                    lp = trial[n][-1]
                    if not len(trial[n]) < Ttrial:
                        continue
                    trial[n].append(lp)
                    if len(trial[n]) < Ttrial:
                        pass
                if _timelines_conflict(pose, trial):
                    continue
                print(f"[PIPE] stitch-ok-short {agv} steps={cut}/{len(path)} end={path[cut - 1][:2]}", flush=True)
                kept = True
                reserved_v.clear()
                reserved_e.clear()
                for n in names:
                    prev = pose[n][:2]
                    for i, p in enumerate(out[n]):
                        t = i + 1
                        c = (p[0], p[1])
                        reserved_v[(t, c)] = n
                        if c != prev:
                            reserved_e.add((t, prev, c))
                        prev = c
            if kept:
                continue
            print(f"[PIPE] stitch-revert conflict {agv} len={len(path)}", flush=True)
            out[agv] = out[agv][:t_unload]
            cargo[agv] = cargo[agv][:t_unload]
            reserved_v.clear()
            reserved_e.clear()
            for n in names:
                prev = pose[n][:2]
                for i, p in enumerate(out[n]):
                    t = i + 1
                    c = (p[0],
                        
                        p[1])
                    reserved_v[(t,
                        
                        c)] = n
                    if c != prev:
                        reserved_e.add((t, prev, c))
                    prev = c
            continue
        print(f"[PIPE] stitch-ok {agv} steps={len(path)} end={path[-1][:2]}", flush=True)
    cap = T_del + 1 + max(0, int(extend_slack))
    for n in names:
        if not len(out[n]) > cap:
            continue
        out[n] = out[n][:cap]
        cargo[n] = cargo[n][:cap]
    T2 = max((len(out[n]) for n in names), default=0)
    for n in names:
        if not out[n]:
            out[n] = [pose[n]]
            cargo[n] = [(bool(loaded0.get(n, False)), str(dest0.get(n, "")),
    
    str(tid0.get(n, "")))]
        last_p = out[n][-1]
        last_c = (False, "", "")
        if not len(out[n]) < T2:
            continue
        out[n].append(last_p)
        cargo[n].append(last_c)
        if len(out[n]) < T2:
            pass
    
    return (out, cargo)
    
    c = None
    
    n = None; n = None; n = None

def _embed_idle_patrol_motion(pose: "Dict[str, Pose]", names: "List[str]", spare: "List[str]", timelines: "Dict[str, List[Pose]]", cargo: "Dict[str, List[Tuple[bool, str, str]]]", free: "Set[Cell]", static: "Set[Cell]", *, forbidden: "Optional[Set[Cell]]") -> "None":
    pass

def _embed_spare_hub_motion(pose: "Dict[str, Pose]", names: "List[str]", spare: "List[str]", timelines: "Dict[str, List[Pose]]", cargo: "Dict[str, List[Tuple[bool, str, str]]]", free: "Set[Cell]", static: "Set[Cell]", *, forbidden: "Optional[Set[Cell]]") -> "None":
    _embed_idle_patrol_motion(pose, names, spare, timelines, cargo, free, static, forbidden=forbidden)

def _maybe_embed_pipeline_idles(pose: "Dict[str, Pose]", names: "List[str]", idle_park_movers: "List[str]", timelines: "Dict[str, List[Pose]]", cargo: "Dict[str, List[Tuple[bool, str, str]]]", free: "Set[Cell]", static: "Set[Cell]", batch: "Dict[str, dict]", goals: "Dict[str, Cell]", *, extra_forbidden: "Optional[Set[Cell]]") -> "None":
    if not idle_park_movers:
        return None
    forbidden = set((goals.get(a, pose[a][:2]) for a in batch)) | set(extra_forbidden or ()); spare = []
    for n in idle_park_movers:
        if n not in names:
            continue
        match extra_forbidden:
            case 2 as moves if moves <= 1:
                return None
            case _:
                return None

def _bfs_cells(start: "Cell", goal: "Cell", blocked: "Set[Cell]") -> "List[Cell]":
    if start == goal:
        return [start]
    parent = {start: None}; q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not 1 <= nxt[0] <= 20 or 1 <= nxt[1] <= 20:
                pass
            elif nxt in parent:
                continue
            elif nxt in blocked and nxt != goal:
                continue
            parent[nxt] = cur
            q.append(nxt)
    if goal not in parent:
        return []
    rev = []; cur_o = goal
    while cur_o is None:
        rev.append(cur_o)
        cur_o = parent[cur_o]
    
    rev.reverse()
    return rev

def _shortcut_path(path: "List[Cell]", static: "Set[Cell]", *, extra_block: "Optional[Set[Cell]]") -> "List[Cell]":
    if path and len(path) < 3:
        return path
    goal = path[-1]; start = path[0]
    if not extra_block:
        extra_block
    blocked = set(static) | set(set()); blocked.discard(start); blocked.discard(goal); short = _bfs_cells(start, goal, blocked)
    if short and _path_unit_steps(short):
        match extra_block:
            case _:
                return short
    return path

def _paths_conflict(paths: "Dict[str, List[Cell]]") -> "bool":
    if len(paths) < 2:
        return False
    names = sorted(paths); T = max((len(paths[n]) for n in names))
    def cell_at(n: "str", t: "int") -> "Cell":
        p = paths[n]
        if t < len(p):
            return p[t]
        
        return p[-1]
    
    for t in range(T):
        occ = {}
        for n in names:
            c = cell_at(n, t)
            if c in occ:
                return True
            occ[c] = n
        if not t + 1 < T:
            continue
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                cb0 = cell_at(b, t)
                ca0 = cell_at(a, t)
                cb1 = cell_at(b, t + 1)
                ca1 = cell_at(a, t + 1)
                if not ca0 == cb1:
                    continue
                elif not cb0 == ca1:
                    continue
                elif not ca0 != cb0:
                    pass
                paths
            return True
    return False

def _tighten_paths(paths: "Dict[str, List[Cell]]", movers: "Set[str]", static: "Set[Cell]") -> "Dict[str, List[Cell]]":
    for n, p in paths.items():
        pass
    p = p; n = n; out = {n: list(p)}
    def _pad(d: "Dict[str, List[Cell]]") -> "Dict[str, List[Cell]]":
        if not d:
            return d
        T = max((len(pp) for pp in d.values()))
        for n, pp in d.items():
            pass
        pad = {n: list(pp)}; n = n; pp = pp
        for n in pad:
            if not len(pad[n]) < T:
                continue
            pad[n].append(pad[n][-1])
            if len(pad[n]) < T:
                pass
        
        return pad
        
        pp = None; n = None
    
    for _ in range(2):
        order = sorted(movers, key=(lambda n: if not out.get(n):
    out.get(n); (-len([]),
    
    n)))
        progressed = False
        for name in order:
            if not out.get(name):
                out.get(name)
            p = []
            if len(p) < 3:
                continue
            extra = set()
            for m, mp in out.items():
                if not m == name or mp:
                    continue
                extra.add(mp[0])
                extra.add(mp[-1])
            for extra_block in (extra, set()):
                short = _shortcut_path(p, static, extra_block=extra_block)
                if short is p or len(short) >= len(p):
                    continue
                trial = dict(out)
                trial[name] = short
                if _paths_conflict(_pad(trial)):
                    pass
                out[name] = short
                progressed = True
        if progressed:
            pass
    order = sorted(movers, key=(lambda n: if not out.get(n):
    out.get(n); (-len([]),
    
    n)))
    for name in order:
        if not out.get(name):
            out.get(name)
        p = []
        if len(p) < 3:
            continue
        goal = p[-1]
        start = p[0]
        others = [out[m] for m in out if not out[m]]
        m = None
        tmax = max(len(p) + 8, max((len(o) for o in others), default=0) + 8)
        short = _spacetime_bfs(start, goal, static, others, tmax=tmax)
        if not short:
            continue
        elif not _path_unit_steps(short):
            continue
        elif not len(short) < len(p):
            continue
        trial = dict(out)
        trial[name] = short
        if _paths_conflict(_pad(trial)):
            continue
        out[name] = short
    
    return out
    out
    p = None; n = None
    
    m = None

def _spacetime_bfs(start: "Cell", goal: "Cell", static: "Set[Cell]", other_paths: "List[List[Cell]]", *, tmax: "int") -> "List[Cell]":
    def other_at(op: "List[Cell]", t: "int") -> "Cell":
        if not op:
            return start
        elif t < len(op):
            return op[t]
        
        return op[-1]
    
    def blocked(cell: "Cell", t: "int", prev: "Cell") -> "bool":
        if cell in static and cell != goal and cell != start:
            return True
        for op in other_paths:
            if other_at(op, t) == cell:
                return True
            elif not t > 0:
                continue
            elif not other_at(op, t) == prev:
                continue
            elif not other_at(op, t - 1) == cell:
                pass
        return True; return False
    
    parent = {}; q = deque()
    parent[(start, 0)] = None
    
    q.append((start, 0)); found = None
    while q:
        cur, t = q.popleft()
        if cur == goal:
            found = (cur, t)
            break
        elif t >= tmax:
            continue
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dx, cur[1] + dy)
            if not 1 <= nxt[0] <= 20 or 1 <= nxt[1] <= 20:
                pass
            t1 = t + 1
            if (nxt, t1) in parent:
                continue
            elif blocked(nxt, t1, cur):
                continue
            parent[(nxt, t1)] = (cur, t)
            q.append((nxt, t1))
    if found is not None:
        return []
    rev = []; node = found
    while node is None:
        rev.append(node[0])
        node = parent[node]
    
    rev.reverse()
    return rev

def _prioritized_st_paths(starts: "Dict[str, Cell]", goals: "Dict[str, Cell]", movers: "Set[str]", static: "Set[Cell]") -> "Optional[Dict[str, List[Cell]]]":
    order = sorted(movers, key=(lambda n: (_manh(starts[n], goals[n]), n))); out = {n: [starts[n]] for n in starts}; n = goals; reserved = []
    for name in order:
        g = goals[name]
        s = starts[name]
        if s == g:
            out[name] = [s]
            reserved.append(out[name])
            continue
        span = _manh(s, g) + 40 + sum((max(0, len(p) - 1) for p in reserved)) // 2
        path = _spacetime_bfs(s, g, static, reserved, tmax=max(span, 80))
        if path and path[-1] != g:
            return None
        out[name] = path
        reserved.append(path)
    for name in starts:
        if not name not in movers:
            continue
        out[name] = [starts[name]]
    if not all((_path_unit_steps(p) for p in out.values())):
        return None
    return out
    starts
    n = None

def _max_detour_ratio(paths: "Dict[str, List[Cell]]", starts: "Dict[str, Cell]", goals: "Dict[str, Cell]", movers: "Set[str]") -> "float":
    worst = 1.0
    for n in movers:
        if not paths.get(n):
            paths.get(n)
        p = []
        if len(p) < 2:
            continue
        manh = _manh(starts[n], goals[n])
        if manh <= 0:
            continue
        worst = max(worst, (len(p) - 1) / manh)
    return worst

def _bfs_len(start: "Cell", goal: "Cell", static: "Set[Cell]") -> "int":
    if isinstance(start, list):
        start = tuple(start)
    if isinstance(goal, list):
        goal = tuple(goal)
    if start == goal:
        return 0
    blocked = set(static); blocked.discard(start); blocked.discard(goal); path = _bfs_cells(start, goal, blocked)
    if not path:
        return 1_000_000
    return len(path) - 1

def _chunk_list(items: "List", size: "int") -> "List[List]":
    if size <= 0:
        return [list(items)]
    i = None
    return [list(items[i:i + size]) for i in range(0, len(items), size)]
    
    i = None

_WAVE_WAIT_HUB: "Cell" = (6, 1)

_WAVE_WAIT_MAX = 2
def _wave_wait_ring(free: "Set[Cell]", forbidden: "Set[Cell]") -> "Set[Cell]":
    ring = set(_wave_wait_cells(8, free, forbidden))
    for x in range(3, 11):
        c = (x, 1)
        if not c in free:
            continue
        elif not c not in forbidden:
            continue
        ring.add(c)
    return ring

def _wave_wait_cells(n: "int", free: "Set[Cell]", forbidden: "Set[Cell]") -> "List[Cell]":
    if n <= 0:
        return []
    hx, hy = _WAVE_WAIT_HUB; ring = [_WAVE_WAIT_HUB, (hx - 1, hy), (hx + 1, hy), (hx - 2, hy), (hx + 2, hy), (hx, hy + 1), (hx - 1, hy + 1), (hx + 1, hy + 1), (hx + 2, hy + 1), (hx - 2, hy + 1)]; out = []
    for c in ring:
        if c in free and c not in forbidden and c not in out:
            out.append(c)
        if not len(out) >= n:
            pass
    
    return out
    
    extra = sorted((c for c in free), key=(lambda c: (_manh(c, _WAVE_WAIT_HUB), c)))
    for c in extra:
        out.append(c)
        if not len(out) >= n:
            pass
    
    return out

def _wave_wait_cell(free: "Set[Cell]", forbidden: "Set[Cell]") -> "Optional[Cell]":
    cells = _wave_wait_cells(1, free, forbidden)
    if cells:
        return cells[0]

def _send_idles_wave_wait(pose: "Dict[str, Pose]", steps_by: "Dict[str, List[dict]]", names: "List[str]", agvs: "List[str]", free: "Set[Cell]", static: "Set[Cell]", now: "int", *, loaded: "Dict[str, bool]", dest: "Dict[str, str]", tid: "Dict[str, str]", reserved: "Optional[Set[Cell]]", max_movers: "int") -> "int":
    reserved = set(reserved or ()); n = reserved; reserved = ##ERROR## |= {pose[n][:2] for n in names}; ring = _wave_wait_ring(free, reserved); n = None; movers = [n for n in agvs if pose[n][:2] not in ring][:max(0, int(max_movers))]
    if not movers:
        return now
    cells = _wave_wait_cells(len(movers), free, reserved)
    if not cells:
        return now
    cg = {n: pose[n][:2] for n in names}; n = None
    for agv, cell in zip(movers, cells):
        cg[agv] = cell
    tl = _plan_turn_aware_joint(pose, cg, set(movers), static, tmax_scale=2.5, prefer_outer_ring=True)
    if tl is not None:
        tl = _plan_turn_aware_joint(pose, cg, set(movers), static, tmax_scale=4.0)
    if tl is not None:
        for agv, cell in zip(movers, cells):
            path_cells = _bfs_cells(pose[agv][:2], cell, static)
            if path_cells:
                pass
            solo[agv] = path_cells
    now = _apply_pose_timelines(steps_by, pose, tl, now, loaded=loaded, dest=dest, tid=tid); print(f"[PIPE] wave-wait {len(movers)} cells={cells} t={now}", flush=True)
    {m: [pose[m][:2]] for ##ERROR## in names}
    n = None; n = None; n = None; m = _apply_paths(steps_by, pose, solo, now, loaded=loaded, dest=dest, tid=tid)

def _pickup_load_agv(agv: "str", task: "dict", *, loaded: "Dict[str, bool]", dest: "Dict[str, str]", tid: "Dict[str, str]", steps_by: "Dict[str, List[dict]]", pose: "Dict[str, Pose]", names: "List[str]", now: "int") -> "int":
    loaded[agv] = True; dest[agv] = str(task.get("destination") or ""); tid[agv] = str(task["task_id"])
    
    now += 1
    for n in names:
        steps_by[n].append(_hold(n, pose[n], now, loaded=loaded[n], dest=dest[n], tid=tid[n]))
    return now

def _move_agvs_to_cells(pose: "Dict[str, Pose]", steps_by: "Dict[str, List[dict]]", names: "List[str]", movers: "List[str]", cell_goals: "Dict[str, Cell]", free: "Set[Cell]", static: "Set[Cell]", now: "int", *, loaded: "Dict[str, bool]", dest: "Dict[str, str]", tid: "Dict[str, str]") -> "int":
    todo = [m for m in movers if pose[m][:2] != cell_goals[m]]; m = None
    if not todo:
        return now
    cg = {n: pose[n][:2] for n in names}; n = None
    for m in todo:
        g = cell_goals[m]
        cg[m] = g
    tl = _plan_turn_aware_joint(pose, cg, set(todo), static, tmax_scale=2.5, prefer_outer_ring=True)
    if tl is None:
        now = _apply_pose_timelines(steps_by, pose, tl, now, loaded=loaded, dest=dest, tid=tid)
        return now
    
    for m in todo:
        cells = _bfs_cells(pose[m][:2], cell_goals[m], static)
        if cells and len(cells) < 2:
            continue
        solo = {n: [pose[n][:2]] for n in names}
        n = None
        solo[m] = cells
        now = _apply_paths(steps_by, pose, solo, now, loaded=loaded, dest=dest, tid=tid)
    return now
    
    m = None; n = None; n = None

def _delivery_stagger_cargo() -> "Tuple[int, bool]":
    pending = set(batch.keys()); rounds = 0
    while pending and rounds < 140:
        rounds += 1
        at_drop = {a for a in pending if not pose[a][:2] == goals[a]}
        a = None
        for a in list(at_drop):
            now += 1
            loaded[a] = False
            dest[a] = ""
            tid[a] = ""
            for n in names:
                steps_by[n].append(_hold(n, pose[n], now, loaded=loaded[n], dest=dest[n], tid=tid[n]))
            pending.discard(a)
        if not pending:
            break
        still = {a for a in pending if not pose[a][:2] != goals[a]}
        a = None
        if not still:
            break
        sub_goals = {n: pose[n][:2] for n in names}
        n = None
        for a in still:
            sub_goals[a] = goals[a]
        tl = _plan_turn_aware_joint(pose, sub_goals, still, static)
        if tl is not None:
            tl = {##ERROR##: list([n], [pose[n]]) for n in names if not tl_plan.get(n)}
            n = None
            t_before = now
            now = _apply_timelines_until_first_goal(steps_by, pose, tl, goals, still, now, loaded=loaded, dest=dest, tid=tid)
    
    ok = not pending
    if ok:
        print(f"[PIPE] delivery-stagger unloaded={len(batch)} t={now}", flush=True)
    return (now, ok)
    
    a = None
    
    a = None
    
    n = None; n = None

def _pickup_stagger_turn_aware() -> "Tuple[int, bool]":
    if not hub_movers:
        hub_movers
    hub_movers = []
    if not hub_goals:
        hub_goals
    hub_goals = {}; pending = set(assigned.keys()); rounds = 0
    while pending:
        match hub_goals:
            case 120 as rounds if pose[a][:2] == sub_goals[a] and pose[a][:2] != sub_goals[a] and pose[h][:2] == hub_goals.get(h):
                return (now, False)
        t_before = now
        if still:
            now = _apply_timelines_until_first_goal(steps_by, pose, tl, sub_goals, still, now, loaded=loaded, dest=dest, tid=tid)
        elif hub_still:
            now = _apply_pose_timelines(steps_by, pose, tl, now, loaded=loaded, dest=dest, tid=tid)
        if still:
            pass
    _apply_pose_timelines(steps_by, pose, tl, now, loaded=loaded, dest=dest, tid=tid)
    
    a = _plan_turn_aware_joint(pose, sub_goals, movers, static)
    still | hub_still
    
    a = _pickup_load_agv(a, assigned[a], loaded=loaded, dest=dest, tid=tid, steps_by=steps_by, pose=pose, names=names, now=now); a = not pending

def _pickup_stagger_paths() -> "Tuple[int, bool]":
    pending = set(assigned.keys()); rounds = 0
    while pending and rounds < 120:
        rounds += 1
        at_pick = {a for a in pending if not pose[a][:2] == goals[a]}
        a = None
        for a in list(at_pick):
            now = _pickup_load_agv(a, assigned[a], loaded=loaded, dest=dest, tid=tid, steps_by=steps_by, pose=pose, names=names, now=now)
            pending.discard(a)
        if not pending:
            break
        still = {a for a in pending if not pose[a][:2] != goals[a]}
        a = None
        if not still:
            for a in list(pending):
                now = _pickup_load_agv(a, assigned[a], loaded=loaded, dest=dest, tid=tid, steps_by=steps_by, pose=pose, names=names, now=now)
            pending.clear()
            break
        t_before = now
        now = _apply_paths_until_first_goal(steps_by, pose, paths, goals, still, now, loaded=loaded, dest=dest, tid=tid)
        if now == t_before:
            now = _apply_paths(steps_by, pose, paths, now, loaded=loaded, dest=dest, tid=tid)
    ok = not pending
    if ok:
        print(f"[PIPE] pickup-stagger(paths) loaded={len(assigned)} t={now}", flush=True)
    return (now, ok)
    
    a = None; a = None

def _nearby_park(from_cell: "Cell", free: "Set[Cell]", forbidden: "Set[Cell]") -> "Optional[Cell]":
    if from_cell in free and from_cell not in forbidden:
        return from_cell
    cands = [c for c in free if not c not in forbidden]; c = from_cell
    if not cands:
        return None
    return min(cands, key=(lambda c: (_manh(from_cell, c), 0, c) if c[0] in (1, 20) or c[1] in (1, 20) else (##ERROR##,1, c)))
    
    c = None

def _apply_paths(steps_by: "Dict[str, List[dict]]", pose: "Dict[str, Pose]", paths: "Dict[str, List[Cell]]", now: "int", *, loaded: "Dict[str, bool]", dest: "Dict[str, str]", tid: "Dict[str, str]") -> "int":
    names = list(pose.keys()); trimmed = {}
    for n in names:
        if not paths.get(n):
            paths.get(n)
        p = list([pose[n][:2]])
        if len(p) >= 2 and p[0] == pose[n][:2]:
            p = p[1:]
        if not p:
            p = [pose[n][:2]]
        trimmed[n] = p
    T_cells = max((len(trimmed[n]) for n in names))
    
    def _emit() -> "None":
        now += 1
        for n in names:
            steps_by[n].append(_hold(n, pose[n], now, loaded=loaded.get(n, False), dest=dest.get(n, ""), tid=tid.get(n, "")))
    
    for step in range(T_cells):
        nxt = {n: trimmed[n][-1] for n in names}
        n = None
        for _guard in range(4):
            need = []
            for n in names:
                cur = pose[n]
                goal_c = nxt[n]
                if goal_c == (cur[0], cur[1]):
                    continue
                dy = goal_c[1] - cur[1]
                dx = goal_c[0] - cur[0]
                if abs(dx) + abs(dy) != 1:
                    continue
                tp = _required_pitch(dx, dy)
                if not int(cur[2]) % 360 != tp:
                    continue
                need.append((n, tp))
            if not need:
                break
            for n, tp in need:
                x, y, _ = pose[n]
                pose[n] = (x, y, tp)
            _emit()
        for n in names:
            x, y = nxt[n]
            pitch = pose[n][2]
            if (x, y) != (pose[n][0],
    
    pose[n][1]):
                dy = y - pose[n][1]
                dx = x - pose[n][0]
                if abs(dx) + abs(dy) == 1:
                    pitch = _required_pitch(dx, dy)
            pose[n] = (x, y, pitch)
        _emit()
    return now
    
    n = None

def _stitch_post_unload_stages(timelines: "Dict[str, List[Pose]]", pose0: "Dict[str, Pose]", arrivals: "Dict[str, int]", stage_goals: "Dict[str, Cell]", static: "Set[Cell]") -> "Dict[str, List[Pose]]":
    if not stage_goals and arrivals:
        return timelines
    names = list(timelines.keys()); Tmax = max((len([]) for n in names), default=0)
    if Tmax <= 0:
        return timelines
    reserved_v = {}; reserved_e = set()
    for n in names:
        prev = pose0[n][:2]
        if not timelines.get(n):
            timelines.get(n)
        seq = []
        for i, p in enumerate(seq):
            t = i + 1
            c = (p[0], p[1])
            reserved_v[(t, c)] = n
            if c != prev:
                reserved_e.add((t, prev, c))
            prev = c
    
    n = None; out = {##ERROR##: list([n], []) for n in names if not timelines.get(n)}; order = sorted(stage_goals.keys(), key=(lambda n: (arrivals.get(n, Tmax), n)))
    for agv in order:
        t_a = int(arrivals.get(agv, 0))
        goal = stage_goals[agv]
        at = out[agv][t_a - 1]
    
    at = pose0[out[agv][-1] if out[agv] else agv]; t_leave = t_a + 1; drop = (at[0], at[1])
    for t in range(t_a + 1, Tmax + 64):
        if not reserved_v.get((t, drop)) == agv:
            continue
        del reserved_v[(t, drop)]
    
    out[agv] = out[agv][:t_a]
    
    while len(out[agv]) < t_leave:
        out[agv].append(at)
    
    shift = t_leave; sh_v = {}
    for t, c in list(reserved_v.items()):
        who = None
        if t < shift:
            continue
        sh_v[(t - shift, c)] = who
    sh_e = set()
    for t, a, b in list(reserved_e):
        if t < shift:
            continue
        sh_e.add((t - shift, a, b))
    sh_v[(0, drop)] = agv
    
    path = _turn_aware_st_astar(at, goal, static, sh_v, sh_e, agv, tmax=max(80, _bfs_len(drop, goal, static) * 3 + 40))
    if path is not None:
        if len(out[agv]) < Tmax:
            out[agv].append(at)
            if len(out[agv]) < Tmax:
                pass
    out[agv].extend(path); prev = drop
    for i, p in enumerate(path):
        t = t_leave + i + 1
        c = (p[0], p[1])
        reserved_v[(t, c)] = agv
        if c != prev:
            reserved_e.add((t, prev, c))
        prev = c
    
    fin = drop; fin_t = t_leave + len(path)
    for t in range(fin_t + 1, max(Tmax, fin_t) + 8):
        reserved_v[(t, fin)] = agv
    T2 = max((len(out[n]) for n in names), default=0)
    for n in names:
        if not out[n]:
            continue
        last = out[n][-1]
        if not len(out[n]) < T2:
            continue
        out[n].append(last)
        if len(out[n]) < T2:
            pass
    
    return out
    
    n = None

def _outer_leave_near_pick(lr_pick: "Cell", free: "Set[Cell]", reserved: "Set[Cell]") -> "Cell":
    if lr_pick in free and lr_pick not in reserved:
        x, y = lr_pick
        rim = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = (x + dx, y + dy)
            if c not in free or c in reserved:
                continue
            elif not c[0] in (1, 2, 19, 20) and c[1] in (1, 2, 19, 20):
                continue
            rim.append(c)
        if rim:
            return min(rim, key=(lambda c: (_manh(c, lr_pick), c)))
    park = _outer_staging(lr_pick, free, reserved)
    if park is None:
        return park
    
    return lr_pick

def _leave_from_unload(lr_pick: "Cell", unload: "Cell", free: "Set[Cell]", reserved: "Set[Cell]", static: "Set[Cell]") -> "Cell":
    x, y = lr_pick; cands = []
    if lr_pick in free and lr_pick not in reserved:
        cands.append(lr_pick)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        c = (x + dx, y + dy)
        if c not in free or c in reserved:
            continue
        elif not c[0] in (1, 2, 19, 20) and c[1] in (1, 2, 19, 20) and _manh(c, lr_pick) <= 2:
            continue
        cands.append(c)
    for c in free:
        if c in reserved or c in cands:
            continue
        elif _manh(c, lr_pick) > 5:
            continue
        elif not c[0] in (1, 2, 19, 20) and c[1] in (1, 2, 19, 20):
            continue
        cands.append(c)
    if not cands:
        return _outer_leave_near_pick(lr_pick, free, reserved)
    
    return min(cands, key=(lambda c: (_bfs_len(unload, c, static), _bfs_len(c, lr_pick, static), _manh(c, lr_pick), c)))

def _outer_staging(near: "Cell", free: "Set[Cell]", forbidden: "Set[Cell]") -> "Optional[Cell]":
    cands = [c for c in free if not c not in forbidden]; c = near
    if not cands:
        return None
    outer = [c for c in cands if c[1] in (1, 2, 19, 20)]; c = None; pool = outer or cands
    return min(pool, key=(lambda c: (_manh(c, near), c)))
    
    c = None; c = None

def _apply_timelines_until_first_goal(steps_by: "Dict[str, List[dict]]", pose: "Dict[str, Pose]", timelines: "Dict[str, List[Pose]]", goals: "Dict[str, Cell]", movers: "Set[str]", now: "int", *, loaded: "Dict[str, bool]", dest: "Dict[str, str]", tid: "Dict[str, str]") -> "int":
    arrive = {}
    for n in movers:
        if not timelines.get(n):
            timelines.get(n)
        seq = []
        g = goals.get(n)
        if g is not None:
            continue
        elif pose[n][:2] == g:
            arrive[n] = 0
            continue
        for i, p in enumerate(seq):
            if not (p[0], p[1]) == g:
                pass
            arrive[n] = i + 1
    if not arrive:
        return _apply_pose_timelines(steps_by, pose, timelines, now, loaded=loaded, dest=dest, tid=tid)
    positive = [t for t in arrive.values() if not t > 0]; t = None
    if not positive:
        return now
    t_cut = min(positive)
    
    names = list(pose.keys()); trimmed = {}
    for n in names:
        if not timelines.get(n):
            timelines.get(n)
        seq = list([])
        if not seq:
            trimmed[n] = []
            continue
        trimmed[n] = seq[:t_cut]
        if not len(trimmed[n]) < t_cut:
            continue
        trimmed[n].append(pose[trimmed[n][-1] if trimmed[n] else n])
        if len(trimmed[n]) < t_cut:
            pass
    
    return _apply_pose_timelines(steps_by, pose, trimmed, now, loaded=loaded, dest=dest, tid=tid)
    
    t = None

def _apply_paths_until_first_goal(steps_by: "Dict[str, List[dict]]", pose: "Dict[str, Pose]", paths: "Dict[str, List[Cell]]", goals: "Dict[str, Cell]", movers: "Set[str]", now: "int", *, loaded: "Dict[str, bool]", dest: "Dict[str, str]", tid: "Dict[str, str]") -> "int":
    names = list(pose.keys()); trimmed = {}
    for n in names:
        if not paths.get(n):
            paths.get(n)
        p = list([pose[n][:2]])
        if len(p) >= 2 and p[0] == pose[n][:2]:
            p = p[1:]
        if not p:
            p = [pose[n][:2]]
        trimmed[n] = p
    T_cells = max((len(trimmed[n]) for n in names))
    def _emit() -> "None":
        now += 1
        for n in names:
            steps_by[n].append(_hold(n, pose[n], now, loaded=loaded.get(n, False), dest=dest.get(n, ""), tid=tid.get(n, "")))
    
    for step in range(T_cells):
        nxt = {n: trimmed[n][-1] for n in names}
        n = None
        for _guard in range(4):
            need = []
            for n in names:
                cur = pose[n]
                goal_c = nxt[n]
                if goal_c == (cur[0], cur[1]):
                    continue
                dy = goal_c[1] - cur[1]
                dx = goal_c[0] - cur[0]
                if abs(dx) + abs(dy) != 1:
                    continue
                tp = _required_pitch(dx, dy)
                if not int(cur[2]) % 360 != tp:
                    continue
                need.append((n, tp))
            if not need:
                break
            for n, tp in need:
                x, y, _ = pose[n]
                pose[n] = (x, y, tp)
            _emit()
        for n in names:
            x, y = nxt[n]
            pitch = pose[n][2]
            if (x, y) != (pose[n][0],
    
    pose[n][1]):
                dy = y - pose[n][1]
                dx = x - pose[n][0]
                if abs(dx) + abs(dy) == 1:
                    pitch = _required_pitch(dx, dy)
            pose[n] = (x, y, pitch)
        _emit()
        if not any((pose[n][:2] == goals.get(n) for n in movers)):
            continue
    
    return now
    
    n = trimmed

def _run_pipeline() -> "Tuple[int, int, List[str], Dict[str, int]]":
    pass

def solve_ecbs(slot: "int"=3, *, meta: "Optional[dict]", max_tasks: "int", max_active: "int", weight: "float", time_limit: "float", max_expansions: "int", plan: "str", allow_solo_fallback: "bool", initial_park: "bool", deliver_batch: "int", turn_aware: "bool", pipeline: "bool", task_csv: "Optional[Path]") -> "dict":
    with traj.open("w", newline="", encoding="utf-8") as f:
        pass
    
    for r in rows:
        k = w.writerow
        ##ERROR##({k: r.get(k, "") for k in TRAJ_HEADER})
    w.writeheader()(None, None, None)
    while 1:
        val.get("issues")
        t = rows.sort(key=(lambda r: (int(r["timestamp"]), r["name"])))
        t = print(f"[ECBS] done={done}/{total} wave={len(assigned)} batches={len(batches)} t={now} wall={time.perf_counter() - t0:.1f}s", flush=True)
        q = filled.append(r)
        n = None
        steps_by[n].append(_hold(n, pose[n], now, loaded=loaded[n], dest=dest[n], tid=tid[n]))
        k = None
        v = None
        k = isinstance(paths, dict)(f"print[ECBS] delivery batch {bi + 1}/{len(batches)} n={len(batch)}{[goals[a] for a in sorted(batch)]}", flush=True)
        {"__serial_ta__": True}
        p = None
        done += 1
        p = None
        now += 1
        v = {n: [(False, "", "")] * len([]) for ##ERROR## in names}
        _plan_turn_aware_joint(pose, cg_1, {agv}, static, tmax_scale=5.0)
        v = {m: pose[m][:2] for ##ERROR## in names}
        v = print(f"[PIPE] corner-evac {n}->{park} t={now}", flush=True)
        v = _apply_paths(steps_by, pose, {n: [pose[n][:2]] for ##ERROR## in names}, now, loaded=loaded, dest=dest, tid=tid)
        _ecbs({n: pose[n][:2] for ##ERROR## in names}, grid, {n: pose[n][:2] for ##ERROR## in names}, weight=max(weight, 2.0), time_limit=float(time_limit) * 2.0, max_expansions=max_expansions, movers={agv}, plan="per_agent")
        y = None
        x = goals.get(agv)
        n = {n: [pose[n][:2]] for ##ERROR## in names}
        _ecbs(grid, starts, goals, weight=weight, time_limit=float(time_limit) * 2.0, max_expansions=max_expansions, movers=set(batch) | stage_movers, plan="per_agent")
        n = _ecbs(grid, starts, goals, weight=max(weight, 2.0), time_limit=float(time_limit) * 3.0, max_expansions=max_expansions * 2, movers=set(batch) | stage_movers, plan=plan)
        _ecbs(grid, starts, goals, weight=weight, time_limit=float(time_limit) * (1.0 + 0.25 * max(0, len(batch) - 2)), max_expansions=max_expansions, movers=set(batch) | stage_movers, plan=plan)
        n = {n: pose[n][:2] for ##ERROR## in names}
        _clear_blocking_idles(grid, pose, ul_cells, set(batch), free, weight=weight, time_limit=time_limit, max_expansions=max_expansions, plan=plan)
        n = None if any(queues.values()) or pipeline_seed else None
        {"__ta__": True}
        n = _apply_pose_timelines(steps_by, pose, tl_d, now, loaded=loaded, dest=dest, tid=tid)
        {"__ta__": True, "__overlap__": True}
        n = _apply_pose_timelines(steps_by, pose, tl_1, now, loaded=loaded, dest=dest, tid=tid) if next_pick else left += 1
        _plan_turn_aware_joint(pose, cg_1, {a}, static, tmax_scale=4.0)
        n = {n: pose[n][:2] for ##ERROR## in names}
        left += 1
        n = None
        n = sum((1 for a, g in leave_goals.items()))
        _apply_pose_timelines(steps_by, pose, tl_l, now, loaded=loaded, dest=dest, tid=tid)
        n = _plan_turn_aware_joint(pose, cg_l, set(leave_goals), static, tmax_scale=4.0, prefer_outer_ring=True)
        _plan_turn_aware_joint(pose, cg_l, set(leave_goals), static, tmax_scale=2.5, prefer_outer_ring=True)
        n = {n: pose[n][:2] for ##ERROR## in names}
        _apply_pose_timelines(steps_by, pose, tl_c, now, loaded=loaded, dest=dest, tid=tid)
        n = _plan_turn_aware_joint(pose, cg, blockers, static)
        _nearby_park(pose[n][:2], free, reserved_c)
        t = None
        {pose[a][:2] for ##ERROR## in names}
        n = {n for ##ERROR## in names}
        set(leave_goals.values())
        n = _apply_pose_timelines(steps_by, pose, tl_p, now, loaded=loaded, dest=dest, tid=tid)
        _plan_turn_aware_joint(pose, cg_p, movers_p, static, tmax_scale=3.0)
        a = min(corners, key=(lambda c: (c in reserved_c, _manh(pose[n][:2], c), c)))
        _outer_staging(pose[n][:2], free, reserved_c)
        e = None
        set()
        i = [c for ##ERROR## in free]
        {n: pose[n][:2] for ##ERROR## in names}
        t = [n for ##ERROR## in names]
        {a: next_pick[a] for a in next_pick}
        n = None
        _build_overlap_delivery_timelines(pose, tl_d, goals, batch, next_pick, static, loaded, dest, tid, extend_slack=55)
        task = {n: str("") for ##ERROR## in names}
        agv = {n: n in batch for ##ERROR## in names}
        n = {str(batch[n]["task_id"]): ""}
        n = {}
        n = {"__ta__": True, "__via__": True}
        n = [a]
        []
        n = a
        g
        a = None
        done += 1
        n = None
        now += 1
        n = (dag)
        n = []
        _apply_pose_timelines(steps_by, pose, tl_l, now, loaded=loaded, dest=dest, tid=tid)
        a = _plan_turn_aware_joint(pose, cg_l, movers_l, static, tmax_scale=4.5, prefer_outer_ring=True)
        _plan_turn_aware_joint(pose, cg_l, movers_l, static, tmax_scale=3.0, prefer_outer_ring=True)
        n = {}
        set(left)
        n = {n: pose[n][:2] for ##ERROR## in names}
        _apply_pose_timelines(steps_by, pose, tl_p, now, loaded=loaded, dest=dest, tid=tid)
        a = _plan_turn_aware_joint(pose, cg_p, movers_p, static, tmax_scale=2.0)
        n = a
        g
        n = print(f"[PIPE] door-overflow counted {sorted(door_via_complete)} done={done}", flush=True)
        a = print(f"[PIPE] embed-complete {a} done={done}", flush=True)
        a = None
        list([])
        n = None
        {}
        n = {}
        max(int(sorted_arr[i]), int(all_un_t))
        n = None
        T_via
        n = sorted(arr_e.values())
        n = None
        n = list([])
        goals.get(idle)
        n = None
        n = cg_path[:T_via]
        path[:T_via]
        n = None
        list(seq[:ai]) if not ai is None and seq else [(True, ds, td)] * len(path)
        n = str(tsk["task_id"])
        str("")
        n = [(True, ds, td)] * len(path)
        list(deliv)
        n = str(tsk["task_id"])
        str("")
        n = None
        max(80, int(_bfs_len(pose[agv][:2], drop, static) * 5) + 40)
        a = c
        n = None
        n = (agv)
        n = None
        set(door_done_via)
        n = None
        list([])
        n = list([])
        n = set(batch) | set(door_pick_goal)
        cg_path[:T_via]
        n = path[:T_via]
        {(t - shift4, a, b)}
        a = a
        t
        a = b
        {(t - shift4, c): who}
        n = c
        t
        n = who
        n = max(40, int(_bfs_len(drop2, g4, static) * 5) + 20)
        min(c4, key=(lambda c: (_bfs_len(drop2, c, static), c)))
        e = _pickup_stand_cands(pk4, st4, free, set())
        _station_of_approach(pk4, stations)
        c = tuple(t4["pickup_point"])
        q4[0]
        c = None
        queues.get(sn_u)
        n = ""
        tid_u.rsplit("-", 1)[0]
        a = str("")
        a = cg_try
        path_try
        s = list(cg_path) + [(True, ds, td)] * len(deliv)
        list(path) + list(deliv)
        a = 35
        _turn_aware_st_astar(at_pick, drop2, static, sh_v, sh_e, agv, tmax=bud)
        a = max(100, int(_bfs_len(stand, drop2, static) * 5) + 50)
        {(t - shift, a, b)}
        n = a
        t
        a = b
        {(t - shift,
    
    c): who}
        a = c
        t
        a = who
        e = ()
        c = c
        c = pose[a][:2]
        a = set()
        {}
        agv = sorted(ends, key=(lambda c: (_bfs_len(stand, c, static), c)))
        [tuple(e) for ##ERROR## in []]
        n = list(tl_d.get(agv), [])
        n = [(False, "", "")]
        a = cargo_raw.get(a)
        n = None
        tsk.get("_pipe_pick") if lr is None else tuple(lr)
        e = None
        a = {"__ta__": True, "__overlap__": True, "__chain__": True}
        sum((1 for a, g in leave_goals.items()))
        n = _apply_pose_timelines(steps_by, pose, tl_l, now, loaded=loaded, dest=dest, tid=tid)
        _plan_turn_aware_joint(pose, cg_l, set(leave_goals), static, tmax_scale=4.0, prefer_outer_ring=True)
        a = _plan_turn_aware_joint(pose, cg_l, set(leave_goals), static, tmax_scale=2.5, prefer_outer_ring=True)
        {n: pose[n][:2] for ##ERROR## in names}
        a = {a: next_pick[a] for a in seed_rest}
        print(f"[PIPE] chain extra={len(chained)}/{len(chain_task)} t={now} T={T_ch} done={done}", flush=True)
        n = None
        n = chain_task
        T_ch <= 170
        n = len(chained) >= 2
        rem_q <= 2
        n = len(chained) >= 3
        len(chained) >= 4
        e = len(chained) >= 5
        T_ch <= T_del + 90 + 35 * max(1, len(chained))
        c = len(chained) >= 1
        print(f"[PIPE] chain fail mc={mc_try}", flush=True)
        a = print(f"[PIPE] chain-gate final={is_final} next={len(next_task)} done={done} rem_q={rem_q}", flush=True)
        rem_q == 0
        a = (max(costs), sum(costs)) if staged_idles and any((a in staged_idles for a in best_asg[2])) else _
        d2b
        n = d1b
        min((_bfs_len(lr_c, e, static) for e in ends), default=25)
        a = None
        goals[agv]
        a = ()
        (agv)
        a = None
        {}
        a = True
        []
        e = None
        _ = None
        tsk = sorted(batch.keys())
        agv = sorted(set(batch.keys()) | set(staged_idles))
        len(items)
        a = [(next_task[a], next_pick[a]) for ##ERROR## in sorted(next_task)]
        None
        a = None
        [a for ##ERROR## in next_task](sorted, key=(lambda n: if not tl_d.get(n):
    tl_d.get(n); (_arrival_index([], goals[n]) or 1_000_000_000, n)))
        a = max((len([]) for a in batch), default=80)
        t = nt2
        a = np2
        a = {starts[n] for ##ERROR## in names}(f"print("[PIPE] delivery TA ok after ECBS-park idles", flush=True)print{{a: pose[a] for a in sorted(batch)}}{{a: goals[a] for a in sorted(batch)}}{{n: pose[n][:2] for n in names if not n not in batch}}", flush=True)(f"print[ECBS] delivery batch {bi + 1}/{len(batches)} turn-aware n={len(batch)}{[goals[a] for a in sorted(batch)]}", flush=True)
        {n: starts[n] for ##ERROR## in names}
        n = _plan_turn_aware_joint(pose, goals_i, movers_p, static, tmax_scale=2.5)
        _nearby_park(pose[n][:2], free, reserved_c)
        a = None
        set()
        a = {n: pose[n][:2] for ##ERROR## in names}
        {goals[a] for ##ERROR## in batch}
        _ = {goals[a] for ##ERROR## in batch}
        ld = None
        [tuple(e) for ##ERROR## in []]
        e = batch[victim]
        who = None
        c = c
        t = ()
        pose[idle][:2]
        b = max(60, int(_bfs_len(pose[idle][:2], sg, static) * 5) + 40)
        a = _turn_aware_st_astar(pose[idle], sg, static, {(0, pose[idle][:2]): idle}, set(), idle, tmax=bud * 2)
        t = _turn_aware_st_astar(pose[idle], sg, static, rv_s, re_s, idle, tmax=bud)
        pose[idle][:2]
        who = None
        c = None
        t = c
        b = "door-wait"
        a = None
        t = "parallel"
        pre_next_task.get(_owner)
        s = None
        cands[0] if tail_door else _outer_leave_near_pick(cands[0], free, used_stage)
        w = tuple(pre_next_pick[agv_n])
        c = _pickup_stand_cands(pk, station, free, used_stage)
        t = _station_of_approach(pk, stations)
        pre_next_task[agv_n]
        g = "door-wait"
        a = None
        pk
        n = min(cands, key=(lambda c: (_manh(c, pk), c)))
        _pickup_stand_cands(pk, station, free, used_stage | {pk})
        i = _outer_leave_near_pick(pk, free, used_stage)
        _station_of_approach(pk, stations)
        i = None
        s = tuple(task["pickup_point"])
        len(pre_next_task) >= 4
        s = rem_tail == 2
        done == 6
        n = done >= 6
        rem_tail == 0 if idle_park_movers else 0 < len(pre_next_task) <= 2
        a = None
        a = None
        n = None
        g = None
        a = None
        n = None
        n = None
        n = None
        a = None
        n = None
        n = None
        c = None
        a = None
        n = None
        n = None
        a = None
        n = None
        n = None
        n = None
        n = None
        n = None
        n = None
        n = None
        e = None
        n = None
        c = None
        a = None
        a = None
        c = None
        m = None
        m = None
        n = None
        _ = None
        a = None
        e = None
        s = None
        k = None

def main() -> "int":
    ap = argparse.ArgumentParser(description="Windowed multi-AGV MAPD. Default plan=joint (WCBS≈EECBS): all active cars planned together. max_active = parallel workers."); ap.add_argument("--slot", type=int, default=3); ap.add_argument("--max-tasks", type=int, default=0); ap.add_argument("--max-active", type=int, default=8); ap.add_argument("--plan", choices=("per_agent", "joint", "eecbs", "wcbs", "ecbs"), default="joint", help="joint/eecbs=多车联合 WCBS; per_agent=逐车优先规划"); ap.add_argument("--weight", type=float, default=1.5); ap.add_argument("--time-limit", type=float, default=35.0)
    
    ap.add_argument("--max-expansions", type=int, default=300_000)
    
    ap.add_argument("--allow-solo-fallback", action="store_true", help="允许缩到 1 车 + A*（默认关闭，强制多车联合）"); ap.add_argument("--no-initial-park", action="store_true", help="跳过开局 idle 靠边停车（可省 ~30–40 sim_t）"); ap.add_argument("--deliver-batch", type=int, default=0, help="运货子波大小；0=整波一起送（推荐配合较小 max_active）"); ap.add_argument("--turn-aware", action="store_true", help="转向感知联合规划 (x,y,pitch,t) 优先时空A*，避免事后同步转向栅栏"); ap.add_argument("--pipeline", action="store_true", help="流水线：卸完立刻派下一单 + 外圈 staging 等候，按最早到达事件截断推进")
    
    args = ap.parse_args()
    
    plan = args.plan
    
    rep = solve_ecbs(args.slot, max_tasks=args.max_tasks, max_active=args.max_active, weight=args.weight, time_limit=args.time_limit, max_expansions=args.max_expansions, plan=plan, allow_solo_fallback=bool(args.allow_solo_fallback), initial_park=not bool(args.no_initial_park), deliver_batch=int(args.deliver_batch), turn_aware=bool(args.turn_aware), pipeline=bool(args.pipeline))
    if bool(rep.get("validate_ok")):
        bool(rep.get("validate_ok"))
        if float(rep.get("completion_ratio") or 0) >= 0.999:
            float(rep.get("completion_ratio") or 0) >= 0.999
    
    ok = not rep.get("tasks_failed")
    if ok:
        return 0
    
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
