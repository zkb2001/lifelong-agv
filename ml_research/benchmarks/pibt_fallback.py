"""PIBT-style one-step / short-horizon fallback (LaCAM low-level successor).

Used when spacetime A* fails: produce a collision-free path toward the goal
*immediately* without parking / escape transition.

Reference: Okumura et al. PIBT / LaCAM (see Bilibili BV1Sb421H7x1 / LaCAM*).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Cell = Tuple[int, int]
Pose = Tuple[int, int, int, int]  # x, y, t, pitch

_PITCH = {(1, 0): 0, (-1, 0): 180, (0, 1): 90, (0, -1): 270}
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))  # move + wait


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(cell: Cell, static: Set[Cell], lo: int = 1, hi: int = 20) -> List[Cell]:
    x, y = cell
    out: List[Cell] = []
    for dx, dy in _DIRS:
        nx, ny = x + dx, y + dy
        if dx == 0 and dy == 0:
            out.append((x, y))
            continue
        if not (lo <= nx <= hi and lo <= ny <= hi):
            continue
        if (nx, ny) in static:
            continue
        out.append((nx, ny))
    return out


def pibt_one_step(
    *,
    names: Sequence[str],
    pos: Dict[str, Cell],
    goals: Dict[str, Cell],
    priority: Dict[str, float],
    static: Set[Cell],
    lo: int = 1,
    hi: int = 20,
) -> Dict[str, Cell]:
    """Compute next cells for all agents (vertex + swap free)."""
    order = sorted(names, key=lambda n: (-float(priority.get(n, 0.0)), n))
    decided: Dict[str, Cell] = {}
    occupied: Set[Cell] = set()
    # parent for swap check: from->to
    from_cell = dict(pos)

    def occupied_swap_ok(agent: str, nxt: Cell) -> bool:
        if nxt in occupied:
            return False
        # swap: someone decided to move into our current cell from nxt
        for other, o_to in decided.items():
            if o_to == from_cell[agent] and from_cell[other] == nxt:
                return False
        return True

    def prefer_order(agent: str) -> List[Cell]:
        cur = pos[agent]
        g = goals.get(agent, cur)
        cands = _neighbors(cur, static, lo, hi)
        # closer to goal first; wait last among equal
        cands.sort(key=lambda c: (_manh(c, g), 0 if c != cur else 1, c))
        return cands

    def dfs(agent: str, stack: Set[str]) -> bool:
        if agent in decided:
            return True
        if agent in stack:
            return False
        stack.add(agent)
        for nxt in prefer_order(agent):
            # another undecided agent currently sitting on nxt with lower prio → push
            blocker = None
            for other, op in pos.items():
                if other == agent or other in decided:
                    continue
                if op == nxt:
                    blocker = other
                    break
            if blocker is not None:
                # priority inheritance: try to move blocker first
                if float(priority.get(blocker, 0.0)) > float(priority.get(agent, 0.0)):
                    # cannot push higher priority
                    if not occupied_swap_ok(agent, nxt):
                        continue
                    # higher-prio occupies nxt and stays undecided — skip
                    continue
                if not dfs(blocker, stack):
                    continue
                # blocker moved; re-check
            if not occupied_swap_ok(agent, nxt):
                continue
            decided[agent] = nxt
            occupied.add(nxt)
            stack.discard(agent)
            return True
        # forced wait if free
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
    # fill any missing with wait
    for name in names:
        if name not in decided:
            decided[name] = pos[name]
    return decided


def _pose_at(traj: Sequence[Pose], t: int, default: Pose) -> Pose:
    if not traj:
        return default
    if t < len(traj):
        return tuple(traj[t])  # type: ignore[return-value]
    last = traj[-1]
    return (int(last[0]), int(last[1]), t, int(last[3]) % 360)


def plan_pibt_to_goal(
    *,
    ego: str,
    start: Pose,
    goal: Cell,
    static_obstacles: Iterable[Cell],
    moving_obstacles: Dict[str, List[Pose]],
    all_names: Sequence[str],
    max_steps: int = 120,
    lo: int = 1,
    hi: int = 20,
) -> List[Pose]:
    """Roll PIBT until ``ego`` reaches ``goal`` (or ``max_steps``).

    Other agents follow reserved trajectories when available, else stay / PIBT
    with goal=current (yield to ego).
    """
    static = {(int(x), int(y)) for x, y in static_obstacles}
    t0 = int(start[2])
    pos: Dict[str, Cell] = {ego: (int(start[0]), int(start[1]))}
    for name in all_names:
        if name == ego:
            continue
        traj = moving_obstacles.get(name) or []
        if not traj:
            continue
        p = _pose_at(traj, t0, (int(traj[0][0]), int(traj[0][1]), t0, int(traj[0][3])))
        pos[name] = (int(p[0]), int(p[1]))

    names = list(pos.keys())

    path: List[Pose] = [tuple(start)]  # type: ignore[list-item]
    pitch = int(start[3]) % 360

    for k in range(1, max_steps + 1):
        t = t0 + k
        goals: Dict[str, Cell] = {ego: (int(goal[0]), int(goal[1]))}
        priority: Dict[str, float] = {ego: 1e6}
        # refresh others from reservations at time t-1 (current)
        for name in names:
            if name == ego:
                continue
            traj = moving_obstacles.get(name) or []
            if traj:
                p = _pose_at(traj, t - 1, path[0])
                pos[name] = (int(p[0]), int(p[1]))
                # soft goal: follow reserved next cell if any
                p2 = _pose_at(traj, t, p)
                goals[name] = (int(p2[0]), int(p2[1]))
                priority[name] = 1.0
            else:
                goals[name] = pos[name]
                priority[name] = 0.0

        nxt = pibt_one_step(
            names=names,
            pos=pos,
            goals=goals,
            priority=priority,
            static=static,
            lo=lo,
            hi=hi,
        )
        ex, ey = nxt[ego]
        dx, dy = ex - pos[ego][0], ey - pos[ego][1]
        if (dx, dy) in _PITCH:
            need = _PITCH[(dx, dy)]
            if pitch != need:
                # In-place turn time: one 90° tick per step (180° = two ticks).
                # Do not advance other agents on a turn-only ego tick — their
                # PIBT move assumed ego translated this frame.
                diff = (need - pitch) % 360
                step = 90 if diff <= 180 else -90
                pitch = (pitch + step) % 360
                path.append((pos[ego][0], pos[ego][1], t, pitch))
                continue
        pos[ego] = (ex, ey)
        for name in names:
            if name != ego:
                pos[name] = nxt.get(name, pos[name])
        path.append((ex, ey, t, pitch))
        if (ex, ey) == (int(goal[0]), int(goal[1])):
            return path

    return []


def plan_pibt_assign(
    *,
    ego: str,
    start: Pose,
    pickup: Cell,
    end_points: Sequence[Cell],
    static_obstacles: Iterable[Cell],
    moving_obstacles: Dict[str, List[Pose]],
    all_names: Sequence[str],
    max_steps: int = 160,
) -> List[Pose]:
    """PIBT: start → pickup → nearest end pad (assign-time full leg)."""
    to_pick = plan_pibt_to_goal(
        ego=ego,
        start=start,
        goal=(int(pickup[0]), int(pickup[1])),
        static_obstacles=static_obstacles,
        moving_obstacles=moving_obstacles,
        all_names=all_names,
        max_steps=max_steps,
    )
    if not to_pick:
        return []
    # wait one tick on pickup (load)
    last = to_pick[-1]
    wait = (int(last[0]), int(last[1]), int(last[2]) + 1, int(last[3]) % 360)
    to_pick = list(to_pick) + [wait]

    if not end_points:
        return to_pick
    # choose nearest pad from pickup
    pads = [(int(e[0]), int(e[1])) for e in end_points]
    pad = min(pads, key=lambda c: _manh(c, (wait[0], wait[1])))
    to_drop = plan_pibt_to_goal(
        ego=ego,
        start=wait,
        goal=pad,
        static_obstacles=static_obstacles,
        moving_obstacles=moving_obstacles,
        all_names=all_names,
        max_steps=max_steps,
    )
    if not to_drop:
        return []
    # drop wait tick
    dlast = to_drop[-1]
    to_drop = list(to_drop) + [
        (int(dlast[0]), int(dlast[1]), int(dlast[2]) + 1, int(dlast[3]) % 360)
    ]
    return to_pick + to_drop[1:]
