"""Grid motion helpers (vendored; no PIBT / ECBS dependency)."""
from __future__ import annotations

import heapq
from collections import deque
from typing import Callable, Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]

_NEI4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def bfs_dist_field(
    goal: Cell,
    blocked: Set[Cell],
    lo: int = 1,
    hi: int = 20,
) -> Dict[Cell, int]:
    if goal in blocked:
        return {}
    dist: Dict[Cell, int] = {goal: 0}
    q: deque = deque([goal])
    while q:
        cur = q.popleft()
        d = dist[cur]
        x, y = cur
        for dx, dy in _NEI4:
            nxt = (x + dx, y + dy)
            if not (lo <= nxt[0] <= hi and lo <= nxt[1] <= hi):
                continue
            if nxt in blocked or nxt in dist:
                continue
            dist[nxt] = d + 1
            q.append(nxt)
    return dist


def pitch_for_step(cur: Cell, nxt: Cell) -> int:
    dx = int(nxt[0]) - int(cur[0])
    dy = int(nxt[1]) - int(cur[1])
    if dx == 1 and dy == 0:
        return 0
    if dx == -1 and dy == 0:
        return 180
    if dx == 0 and dy == 1:
        return 90
    if dx == 0 and dy == -1:
        return 270
    return -1


def turn_pitch_chain(from_pitch: int, to_pitch: int) -> List[int]:
    """In-place reorient; any angle change finishes in one tick (1s)."""
    a = int(from_pitch) % 360
    b = int(to_pitch) % 360
    if a == b:
        return []
    return [b]


def turn_cost_ticks(from_pitch: int, to_pitch: int) -> int:
    """Seconds/ticks to face ``to_pitch``: 0 if already facing, else 1 (any Δθ)."""
    a = int(from_pitch) % 360
    b = int(to_pitch) % 360
    return 0 if a == b else 1


def turn_aware_search(
    start: Cell,
    goal: Cell,
    blocked: Set[Cell],
    start_pitch: int,
    *,
    lo: int = 1,
    hi: int = 20,
    reserved: Optional[Set[Cell]] = None,
    move_cost: Optional[Callable[[Cell, Cell], float]] = None,
    max_expand: int = 4000,
) -> Tuple[float, Cell, List[Cell]]:
    """Dijkstra on (cell, heading); edge = turn_ticks + move.

    Returns ``(total_cost, first_step, cell_path_after_start)``.
    If unreachable, returns ``(inf, start, [])``.
    """
    if start == goal:
        return 0.0, start, []
    if start in blocked or goal in blocked:
        return float("inf"), start, []

    pitch0 = int(start_pitch) % 360
    if pitch0 not in (0, 90, 180, 270):
        # Snap to nearest cardinal for search state.
        pitch0 = int(round(pitch0 / 90.0) * 90) % 360

    hard = set(blocked)
    if reserved:
        hard |= {c for c in reserved if c != start}

    def _ok(c: Cell) -> bool:
        return lo <= c[0] <= hi and lo <= c[1] <= hi and c not in hard

    # state: (x, y, pitch)
    dist: Dict[Tuple[int, int, int], float] = {(start[0], start[1], pitch0): 0.0}
    prev: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {
        (start[0], start[1], pitch0): None
    }
    pq: List[Tuple[float, int, int, int]] = [(0.0, start[0], start[1], pitch0)]
    goal_state: Optional[Tuple[int, int, int]] = None
    expands = 0
    while pq and expands < max_expand:
        cost, x, y, pitch = heapq.heappop(pq)
        expands += 1
        st = (x, y, pitch)
        if cost > dist.get(st, 1e18):
            continue
        if (x, y) == goal:
            goal_state = st
            break
        cur = (x, y)
        for dx, dy in _NEI4:
            nxt = (x + dx, y + dy)
            if not _ok(nxt) and nxt != goal:
                continue
            if nxt in hard and nxt != goal:
                continue
            if not (lo <= nxt[0] <= hi and lo <= nxt[1] <= hi):
                continue
            need = pitch_for_step(cur, nxt)
            if need < 0:
                continue
            tcost = float(turn_cost_ticks(pitch, need))
            mcost = float(move_cost(cur, nxt)) if move_cost is not None else 1.0
            nd = cost + tcost + mcost
            nst = (nxt[0], nxt[1], need)
            if nd < dist.get(nst, 1e18):
                dist[nst] = nd
                prev[nst] = st
                heapq.heappush(pq, (nd, nxt[0], nxt[1], need))

    if goal_state is None:
        # Fallback: pick neighbor with least turn+step that reduces geometric dist.
        best = start
        best_score = 1e18
        gmanh = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        for dx, dy in _NEI4:
            nxt = (start[0] + dx, start[1] + dy)
            if not (lo <= nxt[0] <= hi and lo <= nxt[1] <= hi):
                continue
            if nxt in hard:
                continue
            need = pitch_for_step(start, nxt)
            if need < 0:
                continue
            m = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
            score = float(turn_cost_ticks(pitch0, need)) + 1.0 + float(m)
            if m < gmanh and score < best_score:
                best_score = score
                best = nxt
        return float("inf"), best, ([best] if best != start else [])

    # Reconstruct cell path (collapse heading chain).
    cells: List[Cell] = []
    cur_s: Optional[Tuple[int, int, int]] = goal_state
    while cur_s is not None:
        cells.append((cur_s[0], cur_s[1]))
        cur_s = prev.get(cur_s)
    cells.reverse()
    # cells[0] is start
    path = cells[1:]
    first = path[0] if path else start
    return float(dist[goal_state]), first, path


def turn_aware_next(
    start: Cell,
    goal: Cell,
    blocked: Set[Cell],
    start_pitch: int,
    *,
    lo: int = 1,
    hi: int = 20,
    reserved: Optional[Set[Cell]] = None,
    move_cost: Optional[Callable[[Cell, Cell], float]] = None,
) -> Cell:
    """First cell of turn-aware shortest path (or ``start`` if none)."""
    _cost, first, _path = turn_aware_search(
        start,
        goal,
        blocked,
        start_pitch,
        lo=lo,
        hi=hi,
        reserved=reserved,
        move_cost=move_cost,
    )
    return first
