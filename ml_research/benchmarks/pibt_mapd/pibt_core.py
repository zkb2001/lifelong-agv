"""Self-contained PIBT one-step (vertex + swap free).

Independent of ``pibt_fallback.py`` / engine A* handoff.
Reference shape: Okumura et al. PIBT.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

Cell = Tuple[int, int]

_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))


def _manh(a: Cell, b: Cell) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _neighbors(cell: Cell, blocked: Set[Cell], lo: int = 1, hi: int = 20) -> List[Cell]:
    x, y = int(cell[0]), int(cell[1])
    out: List[Cell] = []
    for dx, dy in _DIRS:
        nx, ny = x + dx, y + dy
        if dx == 0 and dy == 0:
            out.append((x, y))
            continue
        if not (lo <= nx <= hi and lo <= ny <= hi):
            continue
        if (nx, ny) in blocked:
            continue
        out.append((nx, ny))
    return out


def bfs_dist_field(
    goal: Cell,
    blocked: Set[Cell],
    lo: int = 1,
    hi: int = 20,
) -> Dict[Cell, int]:
    """Shortest-path distances to ``goal`` on the free grid (obstacle-aware)."""
    if goal in blocked:
        return {}
    dist: Dict[Cell, int] = {goal: 0}
    q: deque = deque([goal])
    while q:
        cur = q.popleft()
        d = dist[cur]
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not (lo <= nxt[0] <= hi and lo <= nxt[1] <= hi):
                continue
            if nxt in blocked or nxt in dist:
                continue
            dist[nxt] = d + 1
            q.append(nxt)
    return dist


def pibt_one_step(
    *,
    names: Sequence[str],
    pos: Dict[str, Cell],
    goals: Dict[str, Cell],
    priority: Dict[str, float],
    blocked: Set[Cell],
    fixed: Optional[Set[str]] = None,
    dist_fields: Optional[Dict[Cell, Dict[Cell, int]]] = None,
    lo: int = 1,
    hi: int = 20,
) -> Dict[str, Cell]:
    """Next cell for every agent; no vertex collision and no edge swap.

    ``fixed`` agents stay put (must turn / dwell); their cells are reserved
    before the PIBT search so movers do not plan into a turn-in-place body.
    ``dist_fields`` maps goal -> BFS distance map (preferred over Manhattan).
    """
    fixed = set(fixed or ())
    decided: Dict[str, Cell] = {}
    occupied: Set[Cell] = set()
    from_cell = {n: pos[n] for n in names}
    fields = dist_fields if dist_fields is not None else {}

    for name in names:
        if name in fixed:
            decided[name] = pos[name]
            occupied.add(pos[name])

    order = sorted(
        (n for n in names if n not in fixed),
        key=lambda n: (-float(priority.get(n, 0.0)), n),
    )

    def occupied_swap_ok(agent: str, nxt: Cell) -> bool:
        if nxt in occupied:
            return False
        for other, o_to in decided.items():
            if o_to == from_cell[agent] and from_cell[other] == nxt:
                return False
        return True

    def goal_dist(cell: Cell, goal: Cell) -> int:
        field = fields.get(goal)
        if field is not None and cell in field:
            return int(field[cell])
        if field is not None:
            return 10_000 + _manh(cell, goal)
        return _manh(cell, goal)

    def prefer_order(agent: str) -> List[Cell]:
        cur = pos[agent]
        g = goals.get(agent, cur)
        cands = _neighbors(cur, blocked, lo, hi)
        noise = abs(hash((agent, int(priority.get(agent, 0.0) * 1000)))) % 5
        cands.sort(
            key=lambda c: (
                goal_dist(c, g),
                0 if c != cur else 1,
                (hash((agent, c)) + noise) % 11,
                c,
            )
        )
        return cands

    def dfs(agent: str, stack: Set[str]) -> bool:
        if agent in decided:
            return True
        if agent in stack:
            return False
        stack.add(agent)
        for nxt in prefer_order(agent):
            blocker = None
            for other, op in pos.items():
                if other == agent or other in decided:
                    continue
                if op == nxt:
                    blocker = other
                    break
            if blocker is not None:
                if blocker in fixed:
                    continue
                if float(priority.get(blocker, 0.0)) > float(priority.get(agent, 0.0)):
                    continue
                if not dfs(blocker, stack):
                    continue
            if not occupied_swap_ok(agent, nxt):
                continue
            decided[agent] = nxt
            occupied.add(nxt)
            stack.discard(agent)
            return True
        cur = pos[agent]
        if occupied_swap_ok(agent, cur):
            decided[agent] = cur
            occupied.add(cur)
            stack.discard(agent)
            return True
        stack.discard(agent)
        return False

    for name in order:
        if name not in decided:
            dfs(name, set())
    for name in names:
        if name not in decided:
            decided[name] = pos[name]
    return decided


def pitch_for_step(cur: Cell, nxt: Cell) -> int:
    """Facing required to step from ``cur`` to ``nxt`` (stay → keep caller pitch)."""
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
    """In-place pitches ≤90°/tick ending at ``to_pitch`` (180° → two ticks)."""
    a = int(from_pitch) % 360
    b = int(to_pitch) % 360
    if a == b:
        return []
    diff = (b - a + 360) % 360
    if diff == 180:
        return [(a + 90) % 360, b]
    return [b]
