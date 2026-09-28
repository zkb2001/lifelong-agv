# Source Generated with Decompyle++
# File: solve_ecbs.cpython-312.pyc (Python 3.12)

'''SH_custom_* lifelong MAPD via windowed multi-agent ECBS (Weighted-CBS).

One-shot ECBS cannot eat 100 pickup鈥揹elivery pairs at once. We roll:
  assign 鈮 free AGVs 鈫?joint plan to pickups 鈫?joint plan to unloads 鈫?repeat.

Default ``plan=joint`` (aliases: eecbs / wcbs / ecbs): all active AGVs are
planned **together**. Duplicate-station tasks get distinct free cells adjacent
to the same station so K>6 still means many cars working in parallel 鈥?not a
single-car A* loop.

``plan=per_agent``: prioritized one-by-one (still multi-task waves).

Grid cells in pymapf are (row, col) = (y-1, x-1) for warehouse (x,y) in 1..20.
'''
from __future__ import annotations
import argparse
import csv
import heapq
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from pymapf import Agent, GridMap, MAPFProblem, solve
from ml_research.benchmarks.common import TRAJ_HEADER, load_scenario, patch_extra_obstacles
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import format_validation_summary, validate_hybrid_trajectory
from ml_research.common.paths import RESULTS
Cell = Tuple[(int, int)]
Pose = Tuple[(int, int, int)]
OUT = RESULTS / 'coord_custom_ai' / 'ecbs'
TRAJ = OUT / 'trajectories'

def _wh_to_rc(x = None, y = None):
    return (y - 1, x - 1)


def _rc_to_wh(r = None, c = None):
    return (c + 1, r + 1)


def _manh(a = None, b = None):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _pitch_from_delta(dx = None, dy = None, prev = None):
    if dx == 1:
        return 0
    if dx == -1:
        return 180
    if dy == 1:
        return 90
    if dy == -1:
        return 270
    return int(prev) % 360


def _hold(name = None, pose = None, t = None, *, loaded, dest, tid):
    return {
        'timestamp': t,
        'name': name,
        'X': pose[0],
        'Y': pose[1],
        'pitch': pose[2],
        'loaded': 'TRUE' if loaded else 'FALSE',
        'destination': dest,
        'Emergency': 'FALSE',
        'task-id': tid }


def _build_grid(static = None, stations = None, W = None, H = (20, 20)):
    grid = []
    for y in range(1, H + 1):
        row = []
        for x in range(1, W + 1):
            if not (x, y) in static:
                (x, y) in static
            blocked = (x, y) in stations
            row.append(1 if blocked else 0)
        grid.append(row)
    return GridMap(grid)


def _unload_cell(end_xy = None, free = None, prefer = None):
    pass
# WARNING: Decompyle incomplete


def _rematch_pipeline_next(unload = None, next_task = None, next_pick = None, static = None, *, arrive, free, include_deliver):
    '''Reassign claimed next tasks to minimize leave-makespan proxy.

    Cost per agent 鈮?arrive[a] + 2路BFS(unload[a], pick)[+ 2路BFS(pick, unload2)].
    n鈮? 鈫?full permutation search.
    '''
    pass
# WARNING: Decompyle incomplete


def _retarget_unload_toward_next(starts, batch, goals = None, next_pick = None, free = None, static = (None,), avoid = ('starts', 'Dict[str, Cell]', 'batch', 'Dict[str, dict]', 'goals', 'Dict[str, Cell]', 'next_pick', 'Dict[str, Cell]', 'free', 'Set[Cell]', 'static', 'Set[Cell]', 'avoid', 'Optional[Set[Cell]]', 'return', 'Dict[str, Cell]')):
    """Prefer unload cells that also sit closer to each agent's next pick."""
    pass
# WARNING: Decompyle incomplete


def _path_unit_steps(path = None):
    for a, b in zip(path, path[1:]):
        if not _manh(a, b) > 1:
            continue
        zip(path, path[1:])
        return False
    return True


def _station_of_approach(pk = None, stations = None):
    for s in stations:
        if not _manh(s, pk) == 1:
            continue
        
        return stations, s
    return pk


def _adj_free(center = None, free = None):
    pass
# WARNING: Decompyle incomplete


def _lr_adj_free(center = None, free = None):
    '''Pickup stands: only directly left/right of the station (same y, x卤1).'''
    (x, y) = center
# WARNING: Decompyle incomplete


def _pickup_stand_cands(primary = None, station = None, free = None, reserved = ('primary', 'Cell', 'station', 'Cell', 'free', 'Set[Cell]', 'reserved', 'Set[Cell]', 'return', 'List[Cell]')):
    '''Legal pickup interaction cells: station left/right only.'''
    cands = []
    if primary in free and primary not in reserved:
        cands.append(primary)
    for c in _lr_adj_free(station, free):
        if not c not in reserved:
            continue
        if not c not in cands:
            continue
        cands.append(c)
    return cands


def _distinct_goals_near(assigned = None, starts = None, free = None, stations = ('assigned', 'Dict[str, dict]', 'starts', 'Dict[str, Cell]', 'free', 'Set[Cell]', 'stations', 'Set[Cell]', 'return', 'Optional[Dict[str, Cell]]'), *, primary_of):
    '''Unique free pickup goals 鈥?station left/right only (no above/below).'''
    pass
# WARNING: Decompyle incomplete


def _ecbs(grid = None, starts_wh = None, goals_wh = None, *, weight, time_limit, max_expansions, movers, plan):
    '''Multi-agent path finding for ``movers``.

    ``plan`` ``joint`` / ``eecbs`` / ``wcbs`` / ``ecbs``: Weighted-CBS on all
    movers together (pymapf has no separate EECBS binary). ``per_agent``:
    prioritized sequential. Non-movers are frozen obstacles.
    '''
    pass
# WARNING: Decompyle incomplete


def _repark_others(grid = None, pose = None, worker = None, free = None, keep_clear = {
    'plan': 'joint' }, *, weight, time_limit, max_expansions, plan):
    pass
# WARNING: Decompyle incomplete


def _repark_others_serial(mod, pose, names, worker, free = None, keep_clear = None, static_list = None, now = ('pose', 'Dict[str, Pose]', 'names', 'List[str]', 'worker', 'str', 'free', 'Set[Cell]', 'keep_clear', 'Set[Cell]', 'static_list', 'list', 'now', 'int', 'steps_by', 'Dict[str, List[dict]]', 'loaded', 'Dict[str, bool]', 'dest', 'Dict[str, str]', 'tid', 'Dict[str, str]', 'return', 'int'), *, steps_by, loaded, dest, tid):
    '''Move idles one-by-one with spacetime A* (safe when ECBS repark fails).'''
    pass
# WARNING: Decompyle incomplete


def _astar_move(mod, pose, names, agv = None, goal = None, static_list = None, now = ('pose', 'Dict[str, Pose]', 'names', 'List[str]', 'agv', 'str', 'goal', 'Cell', 'static_list', 'list', 'now', 'int', 'tid', 'str', 'dest', 'str', 'wall', 'float', 'return', 'Optional[List[Cell]]'), *, tid, dest, wall):
    pass
# WARNING: Decompyle incomplete


def _nearest_park(from_cell = None, free = None, forbidden = None):
    return _nearby_park(from_cell, free, forbidden)


def _clear_blocking_idles(grid = None, pose = None, blocked_cells = None, movers_needed = None, free = {
    'plan': 'joint' }, *, weight, time_limit, max_expansions, plan):
    '''Move idles sitting on ``blocked_cells`` to nearby parks (ECBS).'''
    pass
# WARNING: Decompyle incomplete


def _pad_paths(paths = None, names = None, starts = None):
    '''Pad all paths to same length with waits at end.'''
    for n in names:
        if n not in paths and paths[n]:
            continue
        paths[n] = [
            starts[n]]
    T = (lambda .0: pass# WARNING: Decompyle incomplete
)(paths.values()())
    for n in names:
        p = paths[n]
        if not len(p) < T:
            continue
        p.append(p[-1])
        if len(p) < T:
            continue
    continue
    return paths


def _required_pitch(dx = None, dy = None):
    if dx == 1:
        return 0
    if dx == -1:
        return 180
    if dy == 1:
        return 90
    if dy == -1:
        return 270
    raise ValueError(f'''non-unit move ({dx}, {dy})''')


def _delta_of_pitch(pitch = None):
    p = int(pitch) % 360
    if p == 0:
        return (1, 0)
    if p == 90:
        return (0, 1)
    if p == 180:
        return (-1, 0)
    if p == 270:
        return (0, -1)
    return (0, 0)


def _turn_aware_st_astar(start = None, goal = None, static = None, reserved_v = None, reserved_e = {
    'max_expansions': 250000 }, who = ('start', 'Pose', 'goal', 'Cell', 'static', 'Set[Cell]', 'reserved_v', 'Dict[Tuple[int, Cell], str]', 'reserved_e', 'Set[Tuple[int, Cell, Cell]]', 'who', 'str', 'tmax', 'int', 'max_expansions', 'int', 'return', 'Optional[List[Pose]]'), *, tmax, max_expansions):
    '''Earliest arrival in (x,y,pitch,t); actions are wait / turn / forward move.'''
    pass
# WARNING: Decompyle incomplete


def _plan_turn_aware_joint(pose = None, goals = None, movers = None, static = None, *, tmax_scale, prefer_outer_ring):
    '''Prioritized turn-aware spacetime plans; non-movers freeze as obstacles.'''
    pass
# WARNING: Decompyle incomplete


def _apply_pose_timelines(steps_by = None, pose = None, timelines = None, now = ('steps_by', 'Dict[str, List[dict]]', 'pose', 'Dict[str, Pose]', 'timelines', 'Dict[str, List[Pose]]', 'now', 'int', 'loaded', 'Dict[str, bool]', 'dest', 'Dict[str, str]', 'tid', 'Dict[str, str]', 'return', 'int'), *, loaded, dest, tid):
    '''Emit synchronized rows from per-agent pose timelines (already legal).'''
    pass
# WARNING: Decompyle incomplete


def _expand_cell_paths_to_pose_timelines(pose0 = None, paths = None):
    '''Dry-run competition turn鈫抦ove expand; return per-agent pose timelines.'''
    pass
# WARNING: Decompyle incomplete


def _apply_pose_timelines_cargo(steps_by, pose = None, timelines = None, cargo_tl = None, now = ('steps_by', 'Dict[str, List[dict]]', 'pose', 'Dict[str, Pose]', 'timelines', 'Dict[str, List[Pose]]', 'cargo_tl', 'Dict[str, List[Tuple[bool, str, str]]]', 'now', 'int', 'return', 'Tuple[int, Dict[str, bool], Dict[str, str], Dict[str, str]]')):
    '''Like ``_apply_pose_timelines`` but per-step loaded/dest/task-id.'''
    pass
# WARNING: Decompyle incomplete


def _apply_cargo_chunked_with_idle_nudge(steps_by = None, pose = None, timelines = None, cargo_tl = None, now = {
    'chunk': 18,
    'forbidden': None }, *, idle_movers, names, free, static, chunk, forbidden):
    '''Apply cargo timelines in chunks; nudge spare idles on door ring between.'''
    pass
# WARNING: Decompyle incomplete


def _try_via_roll_suffix(names, batch, next_task, next_pick = None, tl_v = None, cargo_v = None, static = None, free = {
    'k_early': 2,
    'gap_min': 40 }, queues = ('names', 'List[str]', 'batch', 'Dict[str, dict]', 'next_task', 'Dict[str, dict]', 'next_pick', 'Dict[str, Cell]', 'tl_v', 'Dict[str, List[Pose]]', 'cargo_v', 'Dict[str, List[Tuple[bool, str, str]]]', 'static', 'Set[Cell]', 'free', 'Set[Cell]', 'queues', 'Dict[str, List[dict]]', 'k_early', 'int', 'gap_min', 'int', 'return', 'Optional[Tuple[int, List[str], Dict[str, List[Pose]], Dict[str, List[Tuple[bool, str, str]]], Dict[str, List[Pose]], Dict[str, List[Tuple[bool, str, str]]], Dict[str, dict], Dict[str, Cell]]]'), *, k_early, gap_min):
    '''Cut via-leave at early arrivals; late keep original suffix; early deliver in parallel.

    Returns
    -------
    t_cut, early, trimmed, cargo_trim, post_tl, post_cargo, seed_extra, seed_pick
    or None if roll is not beneficial / not feasible.
    '''
    pass
# WARNING: Decompyle incomplete


def _try_via_embed_early_deliver(names, batch, next_task = None, next_pick = None, tl_v = None, cargo_v = None, static = {
    'k_early': 2,
    'gap_min': 45 }, free = ('names', 'List[str]', 'batch', 'Dict[str, dict]', 'next_task', 'Dict[str, dict]', 'next_pick', 'Dict[str, Cell]', 'tl_v', 'Dict[str, List[Pose]]', 'cargo_v', 'Dict[str, List[Tuple[bool, str, str]]]', 'static', 'Set[Cell]', 'free', 'Set[Cell]', 'k_early', 'int', 'gap_min', 'int', 'return', 'Optional[List[str]]'), *, k_early, gap_min):
    """Embed early agents' next-task deliver inside existing via timeline.

    Unlike via-roll, does NOT cut late agents 鈥?wall T stays T_via. Returns the
    list of agents whose next task was completed in-window (caller must bump
    done and drop next_*), or None.
    """
    pass
# WARNING: Decompyle incomplete


def _plan_final_chain(pose, unload_goals = None, batch = None, next_task = None, next_pick = None, static = {
    'max_chain': 99,
    'tl_del': None,
    'stage_goals': None }, free = ('pose', 'Dict[str, Pose]', 'unload_goals', 'Dict[str, Cell]', 'batch', 'Dict[str, dict]', 'next_task', 'Dict[str, dict]', 'next_pick', 'Dict[str, Cell]', 'static', 'Set[Cell]', 'free', 'Set[Cell]', 'max_chain', 'int', 'tl_del', 'Optional[Dict[str, List[Pose]]]', 'stage_goals', 'Optional[Dict[str, Cell]]', 'return', 'Optional[Tuple[Dict[str, List[Pose]], Dict[str, List[Tuple[bool, str, str]]], Set[str]]]'), *, max_chain, tl_del, stage_goals):
    '''Unload current batch, then chain leave鈫抣oad鈫抎eliver for final next tasks.

    If ``tl_del`` is provided, reuse those unload approaches (same as delivery plan)
    instead of replanning 鈥?usually shorter/more consistent.

    ``stage_goals``: preposition targets for chain-only idles; with ``tl_del``,
    reuse their staging approach as a prefix (parallel with unload).
    '''
    pass
# WARNING: Decompyle incomplete


def _plan_turn_aware_delivery_with_leave(pose = None, unload_goals = None, leave_goals = None, movers = None, static = {
    'tmax_scale': 4,
    'leave_set': None }, *, tmax_scale, leave_set):
    '''Two-phase TA: (1) all unload approaches (2) leaves after unload times.

    Leaves never steal corridor from unfinished deliveries.
    '''
    pass
# WARNING: Decompyle incomplete


def _plan_via_leave_best(pose, unload_goals = None, leave_goals = None, movers = None, static = ('pose', 'Dict[str, Pose]', 'unload_goals', 'Dict[str, Cell]', 'leave_goals', 'Dict[str, Cell]', 'movers', 'Set[str]', 'static', 'Set[Cell]', 'return', 'Optional[Tuple[Dict[str, List[Pose]], Dict[str, List[Tuple[bool, str, str]]], Dict[str, bool], Dict[str, str], Dict[str, str]]]')):
    '''Try full leave set, then drop farthest leavers until conflict-free.'''
    pass
# WARNING: Decompyle incomplete


def _timelines_conflict(pose0 = None, timelines = None):
    return _timelines_conflict_pair(pose0, timelines) is not None


def _timelines_conflict_pair(pose0 = None, timelines = None):
    '''Return (a, b, t, cell) for first vertex/swap conflict, else None.'''
    pass
# WARNING: Decompyle incomplete


def _arrival_index(seq = None, goal = None):
    '''1-based step index when cell first equals goal; None if never.'''
    for i, p in enumerate(seq):
        if not (p[0], p[1]) == goal:
            continue
        
        return enumerate(seq), i + 1


def _build_overlap_delivery_timelines(pose, tl_del, goals, batch = None, next_pick = None, static = None, loaded0 = None, dest0 = {
    'extend_slack': 0 }, tid0 = ('pose', 'Dict[str, Pose]', 'tl_del', 'Dict[str, List[Pose]]', 'goals', 'Dict[str, Cell]', 'batch', 'Dict[str, dict]', 'next_pick', 'Dict[str, Cell]', 'static', 'Set[Cell]', 'loaded0', 'Dict[str, bool]', 'dest0', 'Dict[str, str]', 'tid0', 'Dict[str, str]', 'extend_slack', 'int', 'return', 'Tuple[Dict[str, List[Pose]], Dict[str, List[Tuple[bool, str, str]]]]'), *, extend_slack):
    '''Unload at first drop arrival, then stitch early finishers to next pick.

    Cargo flips to unloaded only while standing on the drop cell (validate-safe).
    '''
    pass
# WARNING: Decompyle incomplete


def _embed_idle_patrol_motion(pose, names = None, spare = None, timelines = None, cargo = None, free = {
    'forbidden': None }, static = ('pose', 'Dict[str, Pose]', 'names', 'List[str]', 'spare', 'List[str]', 'timelines', 'Dict[str, List[Pose]]', 'cargo', 'Dict[str, List[Tuple[bool, str, str]]]', 'free', 'Set[Cell]', 'static', 'Set[Cell]', 'forbidden', 'Optional[Set[Cell]]', 'return', 'None'), *, forbidden):
    '''Cycle spare idles along door ring without spacetime conflicts.'''
    pass
# WARNING: Decompyle incomplete


def _embed_spare_hub_motion(pose, names = None, spare = None, timelines = None, cargo = None, free = {
    'forbidden': None }, static = ('pose', 'Dict[str, Pose]', 'names', 'List[str]', 'spare', 'List[str]', 'timelines', 'Dict[str, List[Pose]]', 'cargo', 'Dict[str, List[Tuple[bool, str, str]]]', 'free', 'Set[Cell]', 'static', 'Set[Cell]', 'forbidden', 'Optional[Set[Cell]]', 'return', 'None'), *, forbidden):
    '''Give spare idles motion during a long batch timeline (anti freeze).'''
    _embed_idle_patrol_motion(pose, names, spare, timelines, cargo, free, static, forbidden = forbidden)


def _maybe_embed_pipeline_idles(pose, names, idle_park_movers, timelines = None, cargo = None, free = None, static = None, batch = {
    'extra_forbidden': None }, goals = ('pose', 'Dict[str, Pose]', 'names', 'List[str]', 'idle_park_movers', 'List[str]', 'timelines', 'Dict[str, List[Pose]]', 'cargo', 'Dict[str, List[Tuple[bool, str, str]]]', 'free', 'Set[Cell]', 'static', 'Set[Cell]', 'batch', 'Dict[str, dict]', 'goals', 'Dict[str, Cell]', 'extra_forbidden', 'Optional[Set[Cell]]', 'return', 'None'), *, extra_forbidden):
    '''Patrol only idles not already moving in the joint timeline.'''
    pass
# WARNING: Decompyle incomplete


def _bfs_cells(start = None, goal = None, blocked = None):
    '''Unit-step shortest path; ``blocked`` cells are impassable except goal.'''
    if start == goal:
        return [
            start]
    parent = {
        None: None }
    q = deque([
        start])
    if q:
        cur = q.popleft()
        if cur == goal:
            pass
        else:
            (x, y) = cur
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (x + dx, y + dy)
                if  <= 1, nxt[0] or 1, nxt[0] <= 20:
                    pass
                else:
                    ((1, 0), (-1, 0), (0, 1), (0, -1))
            if not  <= 1, nxt[1] or 1, nxt[1] <= 20:
                pass
            else:
                ((1, 0), (-1, 0), (0, 1), (0, -1))
        if nxt in parent:
            continue
        if nxt in blocked and nxt != goal:
            continue
        q.append(nxt)
        continue
        if q:
            continue
    if goal not in parent:
        return []
    None = None
    cur_o = goal
# WARNING: Decompyle incomplete


def _shortcut_path(path = None, static = None, *, extra_block):
    '''Replace path with static shortest if strictly shorter.'''
    if path or len(path) < 3:
        return path
    goal = path[-1]
    start = None[0]
    if not extra_block:
        extra_block
    blocked = set(static) | set(set())
    blocked.discard(start)
    blocked.discard(goal)
    short = _bfs_cells(start, goal, blocked)
    if short and _path_unit_steps(short) and len(short) < len(path):
        return short


def _paths_conflict(paths = None):
    '''Vertex / swap conflict on padded cell timelines.'''
    pass
# WARNING: Decompyle incomplete


def _tighten_paths(paths = None, movers = None, static = None):
    '''Shorten mover paths via BFS / spacetime when the joint timeline stays safe.'''
    pass
# WARNING: Decompyle incomplete


def _spacetime_bfs(start = None, goal = None, static = None, other_paths = ('start', 'Cell', 'goal', 'Cell', 'static', 'Set[Cell]', 'other_paths', 'List[List[Cell]]', 'tmax', 'int', 'return', 'List[Cell]'), *, tmax):
    '''Earliest-arrival path avoiding vertex/swap conflicts with others.'''
    pass
# WARNING: Decompyle incomplete


def _prioritized_st_paths(starts = None, goals = None, movers = None, static = ('starts', 'Dict[str, Cell]', 'goals', 'Dict[str, Cell]', 'movers', 'Set[str]', 'static', 'Set[Cell]', 'return', 'Optional[Dict[str, List[Cell]]]')):
    '''Short-Manhattan-first prioritized spacetime A* (cuts WCBS spaghetti).'''
    pass
# WARNING: Decompyle incomplete


def _max_detour_ratio(paths = None, starts = None, goals = None, movers = ('paths', 'Dict[str, List[Cell]]', 'starts', 'Dict[str, Cell]', 'goals', 'Dict[str, Cell]', 'movers', 'Set[str]', 'return', 'float')):
    worst = 1
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


def _bfs_len(start = None, goal = None, static = None):
    if isinstance(start, list):
        start = tuple(start)
    if isinstance(goal, list):
        goal = tuple(goal)
    if start == goal:
        return 0
    blocked = set(static)
    blocked.discard(start)
    blocked.discard(goal)
    path = _bfs_cells(start, goal, blocked)
    if not path:
        return 1000000
    return len(path) - 1


def _chunk_list(items = None, size = None):
    if size <= 0:
        return [
            list(items)]
# WARNING: Decompyle incomplete

_WAVE_WAIT_HUB: 'Cell' = (6, 1)
_WAVE_WAIT_MAX = 2

def _wave_wait_ring(free = None, forbidden = None):
    '''All acceptable door-staging cells (not only the first two slots).'''
    ring = set(_wave_wait_cells(8, free, forbidden))
    for x in range(3, 11):
        c = (x, 1)
        if not c in free:
            continue
        if not c not in forbidden:
            continue
        ring.add(c)
    return ring


def _wave_wait_cells(n = None, free = None, forbidden = None):
    '''Distinct staging cells near door hub (6,1); prefer same row y=1.'''
    pass
# WARNING: Decompyle incomplete


def _wave_wait_cell(free = None, forbidden = None):
    '''Preferred staging for next-wave / spare idles (user hub at (6,1)).'''
    cells = _wave_wait_cells(1, free, forbidden)
    if cells:
        return cells[0]


def _send_idles_wave_wait(pose, steps_by = None, names = None, agvs = None, free = None, static = {
    'reserved': None,
    'max_movers': _WAVE_WAIT_MAX }, now = ('pose', 'Dict[str, Pose]', 'steps_by', 'Dict[str, List[dict]]', 'names', 'List[str]', 'agvs', 'List[str]', 'free', 'Set[Cell]', 'static', 'Set[Cell]', 'now', 'int', 'loaded', 'Dict[str, bool]', 'dest', 'Dict[str, str]', 'tid', 'Dict[str, str]', 'reserved', 'Optional[Set[Cell]]', 'max_movers', 'int', 'return', 'int'), *, loaded, dest, tid, reserved, max_movers):
    '''Move up to max_movers spare idles to distinct wave-wait cells near (6,1).'''
    if not reserved:
        reserved
    reserved = set(())
# WARNING: Decompyle incomplete


def _pickup_load_agv(agv = None, task = None, *, loaded, dest, tid, steps_by, pose, names, now):
    loaded[agv] = True
    if not task.get('destination'):
        task.get('destination')
    dest[agv] = str('')
    tid[agv] = str(task['task_id'])
    now += 1
    for n in names:
        steps_by[n].append(_hold(n, pose[n], now, loaded = loaded[n], dest = dest[n], tid = tid[n]))
    return now


def _move_agvs_to_cells(pose, steps_by, names, movers, cell_goals = None, free = None, static = None, now = ('pose', 'Dict[str, Pose]', 'steps_by', 'Dict[str, List[dict]]', 'names', 'List[str]', 'movers', 'List[str]', 'cell_goals', 'Dict[str, Cell]', 'free', 'Set[Cell]', 'static', 'Set[Cell]', 'now', 'int', 'loaded', 'Dict[str, bool]', 'dest', 'Dict[str, str]', 'tid', 'Dict[str, str]', 'return', 'int'), *, loaded, dest, tid):
    '''Move listed AGVs to explicit goals (BFS fallback); never block on peers.'''
    pass
# WARNING: Decompyle incomplete


def _delivery_stagger_cargo(*, pose, steps_by, names, batch, goals, tl_plan, static, now, loaded, dest, tid):
    '''Enter-wave delivery: unload each AGV on arrival, no wait for peers.'''
    pending = set(batch.keys())
    rounds = 0
# WARNING: Decompyle incomplete


def _pickup_stagger_turn_aware(*, pose, steps_by, names, assigned, goals, static, now, loaded, dest, tid, set_pickup_goals, hub_movers, hub_goals):
    '''Arrive-at-pick 鈫?load immediately; spare idles 鈫?door hub in parallel.'''
    if not hub_movers:
        hub_movers
    hub_movers = []
    if not hub_goals:
        hub_goals
    hub_goals = { }
    pending = set(assigned.keys())
    rounds = 0
# WARNING: Decompyle incomplete


def _pickup_stagger_paths(*, pose, steps_by, names, assigned, paths, goals, now, loaded, dest, tid):
    pending = set(assigned.keys())
    rounds = 0
# WARNING: Decompyle incomplete


def _nearby_park(from_cell = None, free = None, forbidden = None):
    '''Prefer stay-put, then 1-hop, then nearest free 鈥?never chase far corners.'''
    pass
# WARNING: Decompyle incomplete


def _apply_paths(steps_by = None, pose = None, paths = None, now = ('steps_by', 'Dict[str, List[dict]]', 'pose', 'Dict[str, Pose]', 'paths', 'Dict[str, List[Cell]]', 'now', 'int', 'loaded', 'Dict[str, bool]', 'dest', 'Dict[str, str]', 'tid', 'Dict[str, str]', 'return', 'int'), *, loaded, dest, tid):
    '''Apply cell paths with competition motion: turn 1s then move 1s.

    Agents stay synchronized by cell index: for each step, everyone who needs
    a turn rotates first (others wait), then everyone moves. This preserves
    ECBS cell reservations while emitting legal pitch timelines.
    '''
    pass
# WARNING: Decompyle incomplete


def _stitch_post_unload_stages(timelines, pose0 = None, arrivals = None, stage_goals = None, static = ('timelines', 'Dict[str, List[Pose]]', 'pose0', 'Dict[str, Pose]', 'arrivals', 'Dict[str, int]', 'stage_goals', 'Dict[str, Cell]', 'static', 'Set[Cell]', 'return', 'Dict[str, List[Pose]]')):
    """After delivery arrivals, replace goal-holds with paths to outer stages.

    Keeps other agents' spacetime reservations so early finishers leave without
    a full replan (avoids event-cut thrashing).
    """
    pass
# WARNING: Decompyle incomplete


def _outer_leave_near_pick(lr_pick = None, free = None, reserved = None):
    '''446-style: stage leave on outer ring beside LR pick (x=1/20 or y=1/20).'''
    pass
# WARNING: Decompyle incomplete


def _leave_from_unload(lr_pick, unload = None, free = None, reserved = None, static = ('lr_pick', 'Cell', 'unload', 'Cell', 'free', 'Set[Cell]', 'reserved', 'Set[Cell]', 'static', 'Set[Cell]', 'return', 'Cell')):
    '''Outer/LR leave near stand, minimizing unload鈫抣eave (compress via T).'''
    pass
# WARNING: Decompyle incomplete


def _outer_staging(near = None, free = None, forbidden = None):
    '''Park on outer ring near ``near`` (outside dense inner corridors).'''
    pass
# WARNING: Decompyle incomplete


def _apply_timelines_until_first_goal(steps_by, pose, timelines = None, goals = None, movers = None, now = ('steps_by', 'Dict[str, List[dict]]', 'pose', 'Dict[str, Pose]', 'timelines', 'Dict[str, List[Pose]]', 'goals', 'Dict[str, Cell]', 'movers', 'Set[str]', 'now', 'int', 'loaded', 'Dict[str, bool]', 'dest', 'Dict[str, str]', 'tid', 'Dict[str, str]', 'return', 'int'), *, loaded, dest, tid):
    '''Advance until the earliest mover reaches its goal (pipeline event cut).'''
    arrive = { }
# WARNING: Decompyle incomplete


def _apply_paths_until_first_goal(steps_by, pose, paths = None, goals = None, movers = None, now = ('steps_by', 'Dict[str, List[dict]]', 'pose', 'Dict[str, Pose]', 'paths', 'Dict[str, List[Cell]]', 'goals', 'Dict[str, Cell]', 'movers', 'Set[str]', 'now', 'int', 'loaded', 'Dict[str, bool]', 'dest', 'Dict[str, str]', 'tid', 'Dict[str, str]', 'return', 'int'), *, loaded, dest, tid):
    '''Like ``_apply_paths`` but stop when any mover first lands on its goal.'''
    pass
# WARNING: Decompyle incomplete


def _run_pipeline(*, names, pose, steps_by, queues, order, static, free, stations, grid, max_active, total, weight, time_limit, max_expansions, plan, now):
    '''Lifelong pipeline: finish 鈫?assign next; outer staging; event cuts.'''
    pass
# WARNING: Decompyle incomplete


def solve_ecbs(slot = None, *, meta, max_tasks, max_active, weight, time_limit, max_expansions, plan, allow_solo_fallback, initial_park, deliver_batch, turn_aware, pipeline, task_csv):
    pass
# WARNING: Decompyle incomplete


def main():
    ap = argparse.ArgumentParser(description = 'Windowed multi-AGV MAPD. Default plan=joint (WCBS鈮圗ECBS): all active cars planned together. max_active = parallel workers.')
    ap.add_argument('--slot', type = int, default = 3)
    ap.add_argument('--max-tasks', type = int, default = 0)
    ap.add_argument('--max-active', type = int, default = 8)
    ap.add_argument('--plan', choices = ('per_agent', 'joint', 'eecbs', 'wcbs', 'ecbs'), default = 'joint', help = 'joint/eecbs=澶氳溅鑱斿悎 WCBS; per_agent=閫愯溅浼樺厛瑙勫垝')
    ap.add_argument('--weight', type = float, default = 1.5)
    ap.add_argument('--time-limit', type = float, default = 35)
    ap.add_argument('--max-expansions', type = int, default = 300000)
    ap.add_argument('--allow-solo-fallback', action = 'store_true', help = '鍏佽缂╁埌 1 杞?+ A*锛堥粯璁ゅ叧闂紝寮哄埗澶氳溅鑱斿悎锛?)
    ap.add_argument('--no-initial-park', action = 'store_true', help = '璺宠繃寮€灞€ idle 闈犺竟鍋滆溅锛堝彲鐪?~30鈥?0 sim_t锛?)
    ap.add_argument('--deliver-batch', type = int, default = 0, help = '杩愯揣瀛愭尝澶у皬锛?=鏁存尝涓€璧烽€侊紙鎺ㄨ崘閰嶅悎杈冨皬 max_active锛?)
    ap.add_argument('--turn-aware', action = 'store_true', help = '杞悜鎰熺煡鑱斿悎瑙勫垝 (x,y,pitch,t) 浼樺厛鏃剁┖A*锛岄伩鍏嶄簨鍚庡悓姝ヨ浆鍚戞爡鏍?)
    ap.add_argument('--pipeline', action = 'store_true', help = '娴佹按绾匡細鍗稿畬绔嬪埢娲句笅涓€鍗?+ 澶栧湀 staging 绛夊€欙紝鎸夋渶鏃╁埌杈句簨浠舵埅鏂帹杩?)
    args = ap.parse_args()
    plan = 'joint' if args.plan in ('eecbs', 'wcbs', 'ecbs') else args.plan
    rep = solve_ecbs(args.slot, max_tasks = args.max_tasks, max_active = args.max_active, weight = args.weight, time_limit = args.time_limit, max_expansions = args.max_expansions, plan = plan, allow_solo_fallback = bool(args.allow_solo_fallback), initial_park = not bool(args.no_initial_park), deliver_batch = int(args.deliver_batch), turn_aware = bool(args.turn_aware), pipeline = bool(args.pipeline))
    if bool(rep.get('validate_ok')):
        bool(rep.get('validate_ok'))
        if not rep.get('completion_ratio'):
            rep.get('completion_ratio')
        if float(0) >= 0.999:
            float(0) >= 0.999
    ok = not rep.get('tasks_failed')
    if ok:
        return 0

if __name__ == '__main__':
    raise SystemExit(main())
