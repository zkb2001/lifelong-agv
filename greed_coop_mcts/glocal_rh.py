"""GLocal-RH: rolling-horizon committed paths + window repair + staging.

Inspired by RHCR (windowed conflict resolution) on top of the GLocal stage
protocol. Each AGV commits an H-step path; conflicts inside the window are
resolved by detour replan of lower-priority agents (not permanent WAIT walls).
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from greed_coop_mcts.rule_obs import DIR_DELTA, GRID, bfs_field, in_arena
from greed_coop_mcts.stage_protocol import CommStats, IntentSketch

Cell = Tuple[int, int]
# Space-time reservation: (t, x, y) for vertex; edge via (t, x1,y1,x2,y2)


@dataclass
class PathCommit:
    agent: str
    cells: List[Cell]  # future cells t+1 .. t+len
    goal: Cell
    priority: float = 0.0

    def next_cell(self, cur: Cell) -> Cell:
        if not self.cells:
            return cur
        return self.cells[0]

    def pop_step(self) -> Optional[Cell]:
        if not self.cells:
            return None
        return self.cells.pop(0)

    def remaining(self) -> int:
        return len(self.cells)


@dataclass
class RHStats:
    n_commits: int = 0
    n_window_repairs: int = 0
    n_restage: int = 0
    sum_committed_horizon: int = 0
    n_horizon_samples: int = 0
    n_staging: int = 0

    def avg_committed_horizon(self) -> float:
        if self.n_horizon_samples <= 0:
            return 0.0
        return float(self.sum_committed_horizon) / float(self.n_horizon_samples)


def _neighbors(cell: Cell, blocked: Set[Cell]) -> List[Cell]:
    x, y = cell
    out: List[Cell] = []
    for dx, dy in DIR_DELTA:
        nxt = (x + dx, y + dy)
        if not in_arena(nxt):
            continue
        if nxt in blocked:
            continue
        out.append(nxt)
    return out


def build_reservation(
    commits: Dict[str, PathCommit],
    *,
    exclude: Optional[str] = None,
    horizon: int = 8,
) -> Tuple[Dict[int, Set[Cell]], Dict[int, Set[Tuple[Cell, Cell]]]]:
    """Vertex and edge reservations by relative time index 0..horizon-1.

    Time 0 = next tick (first committed cell).
    """
    verts: Dict[int, Set[Cell]] = defaultdict(set)
    edges: Dict[int, Set[Tuple[Cell, Cell]]] = defaultdict(set)
    for name, pc in commits.items():
        if exclude is not None and name == exclude:
            continue
        prev: Optional[Cell] = None
        for t, c in enumerate(pc.cells[:horizon]):
            verts[t].add(c)
            if prev is not None:
                edges[t].add((prev, c))
            prev = c
        # Hold last cell for remaining horizon (stationary body forecast)
        if pc.cells:
            last = pc.cells[min(len(pc.cells), horizon) - 1]
            for t in range(len(pc.cells), horizon):
                verts[t].add(last)
    return verts, edges


def spacetime_bfs(
    start: Cell,
    goal: Cell,
    blocked: Set[Cell],
    *,
    verts: Dict[int, Set[Cell]],
    edges: Dict[int, Set[Tuple[Cell, Cell]]],
    horizon: int = 8,
    allow_goal_wait: bool = True,
) -> List[Cell]:
    """BFS in (cell, time) toward goal, avoiding reserved vertices/edges.

    Returns list of cells for t+1.. (length ≤ horizon). Empty if start==goal
    and we choose to wait, or unreachable — caller may fall back.
    """
    if start == goal:
        return [start] * min(2, horizon) if allow_goal_wait else []

    # Soft spatial field for progress heuristic / early exit
    field = bfs_field(goal, blocked)
    if start not in field and goal not in blocked:
        # disconnected from goal under static map
        return []

    # State: (x, y, t) ; t=0 is current (not yet moved)
    q: deque = deque()
    q.append((start[0], start[1], 0))
    prev: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {
        (start[0], start[1], 0): None
    }
    best_goal: Optional[Tuple[int, int, int]] = None
    best_prog: Optional[Tuple[int, int, int]] = None
    best_prog_d = field.get(start, 10_000)

    while q:
        x, y, t = q.popleft()
        if t >= horizon:
            continue
        cur = (x, y)
        dcur = field.get(cur, 10_000)
        if dcur < best_prog_d:
            best_prog_d = dcur
            best_prog = (x, y, t)
        if cur == goal:
            best_goal = (x, y, t)
            break

        nt = t + 1
        # Candidates: wait + 4-neighbors
        cands = [cur] + _neighbors(cur, blocked)
        for nxt in cands:
            ti = nt - 1
            # Vertex reserved by another agent at this timestep
            if nxt in verts.get(ti, ()):
                continue
            # Edge / swap: someone moving from nxt → cur at same time
            if (nxt, cur) in edges.get(ti, ()):
                continue
            st = (nxt[0], nxt[1], nt)
            if st in prev:
                continue
            prev[st] = (x, y, t)
            q.append(st)

    end = best_goal or best_prog
    if end is None or end[2] == 0:
        # one greedy spatial step fallback
        if start not in field:
            return [start]
        best = start
        bd = field.get(start, 10_000)
        for nxt in _neighbors(start, blocked):
            if nxt in verts.get(0, ()):
                continue
            d = field.get(nxt, 10_000)
            if d < bd:
                bd = d
                best = nxt
        return [best] if best != start else [start]

    # Reconstruct path of cells at times 1..end_t
    chain: List[Tuple[int, int, int]] = []
    cur_st: Optional[Tuple[int, int, int]] = end
    while cur_st is not None:
        chain.append(cur_st)
        cur_st = prev.get(cur_st)
    chain.reverse()
    # Drop t=0 (current)
    cells = [(s[0], s[1]) for s in chain[1:]]
    if not cells:
        return [start]
    return cells


def plan_committed(
    *,
    names: List[str],
    positions: Dict[str, Cell],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    priority: Dict[str, float],
    existing: Dict[str, PathCommit],
    horizon: int = 8,
    replan: Optional[Set[str]] = None,
) -> Dict[str, PathCommit]:
    """Priority-ordered spacetime planning into PathCommits."""
    replan = set(replan or ())
    order = sorted(
        names,
        key=lambda n: (-float(priority.get(n, 0.0)), n),
    )
    out: Dict[str, PathCommit] = {}
    # Seed with existing commits for agents not replanning
    for n in names:
        if n in force_stay:
            out[n] = PathCommit(
                agent=n,
                cells=[positions[n]] * horizon,
                goal=goals.get(n, positions[n]),
                priority=float(priority.get(n, 0.0)),
            )
            continue
        if (
            n not in replan
            and n in existing
            and existing[n].cells
            and existing[n].goal == goals.get(n, existing[n].goal)
        ):
            pc = existing[n]
            cells = list(pc.cells)
            # If we are already on the planned head, drop it (desync safety).
            if cells and cells[0] == positions[n]:
                cells = cells[1:]
            if cells:
                out[n] = PathCommit(
                    agent=n,
                    cells=cells,
                    goal=pc.goal,
                    priority=float(priority.get(n, 0.0)),
                )

    for n in order:
        if n in out and n not in replan and out[n].cells:
            continue
        if n in force_stay:
            continue
        verts, edges = build_reservation(out, exclude=n, horizon=horizon)
        # Also reserve current bodies of not-yet-planned lower priority as soft
        # at t=0 only for agents still to plan — skip; they plan later.
        g = goals.get(n, positions[n])
        path = spacetime_bfs(
            positions[n],
            g,
            blocked,
            verts=verts,
            edges=edges,
            horizon=horizon,
        )
        out[n] = PathCommit(
            agent=n,
            cells=path,
            goal=g,
            priority=float(priority.get(n, 0.0)),
        )
    return out


def window_repair(
    commits: Dict[str, PathCommit],
    *,
    positions: Dict[str, Cell],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    window: int = 8,
) -> Tuple[Dict[str, PathCommit], int]:
    """Detect conflicts in window; replan lower-priority agents with detours."""
    names = list(commits.keys())
    n_repairs = 0
    # Multi-pass repair (up to 3)
    for _pass in range(3):
        conflicted: Set[str] = set()
        # Vertex conflicts
        for t in range(window):
            at: Dict[Cell, List[str]] = defaultdict(list)
            for n, pc in commits.items():
                if n in force_stay:
                    continue
                if t < len(pc.cells):
                    c = pc.cells[t]
                elif pc.cells:
                    c = pc.cells[-1]
                else:
                    c = positions[n]
                at[c].append(n)
            for cell, ags in at.items():
                if len(ags) < 2:
                    continue
                ranked = sorted(
                    ags, key=lambda a: (-float(commits[a].priority), a)
                )
                for y in ranked[1:]:
                    conflicted.add(y)
        # Swap conflicts at t=0
        for i, a in enumerate(names):
            ca = commits[a].cells
            if not ca:
                continue
            pa = positions[a]
            for b in names[i + 1 :]:
                cb = commits[b].cells
                if not cb:
                    continue
                pb = positions[b]
                if ca[0] == pb and cb[0] == pa and pa != pb:
                    if float(commits[a].priority) >= float(commits[b].priority):
                        conflicted.add(b)
                    else:
                        conflicted.add(a)
        if not conflicted:
            break
        # Replan conflicted (lowest first) against fixed higher commits
        for n in sorted(
            conflicted, key=lambda a: (float(commits[a].priority), a)
        ):
            if n in force_stay:
                continue
            fixed = {
                m: pc
                for m, pc in commits.items()
                if m != n and m not in conflicted
            }
            # Also keep higher-priority conflicted as soft fixed
            for m in conflicted:
                if m == n:
                    continue
                if float(commits[m].priority) > float(commits[n].priority):
                    fixed[m] = commits[m]
            verts, edges = build_reservation(fixed, horizon=window)
            path = spacetime_bfs(
                positions[n],
                goals.get(n, positions[n]),
                blocked,
                verts=verts,
                edges=edges,
                horizon=window,
            )
            # If still colliding at step 0, force wait then spatial step
            if path and path[0] in verts.get(0, ()):
                path = [positions[n]] + [
                    c for c in path if c != path[0]
                ][: window - 1]
            if not path:
                path = [positions[n]]
            commits[n] = PathCommit(
                agent=n,
                cells=path,
                goal=goals.get(n, positions[n]),
                priority=float(commits[n].priority),
            )
            n_repairs += 1
    return commits, n_repairs


def commits_to_desired(
    commits: Dict[str, PathCommit],
    positions: Dict[str, Cell],
    force_stay: Set[str],
) -> Dict[str, Cell]:
    desired: Dict[str, Cell] = {}
    for n, pc in commits.items():
        if n in force_stay:
            desired[n] = positions[n]
            continue
        desired[n] = pc.next_cell(positions[n])
    return desired


def advance_commits_after_move(
    commits: Dict[str, PathCommit],
    positions: Dict[str, Cell],
    desired: Dict[str, Cell],
) -> None:
    """Pop committed head when agent successfully targeted that cell."""
    for n, pc in commits.items():
        if not pc.cells:
            continue
        # After _apply_desired, if we wanted cells[0], drop it regardless of
        # whether turn delayed the move (path stays valid as intent).
        want = desired.get(n)
        if want == pc.cells[0]:
            # Only pop if agent is now on that cell OR still turning from prev
            if positions[n] == pc.cells[0] or positions[n] != want:
                # If moved onto it, pop; if still at old (turning), keep
                if positions[n] == pc.cells[0]:
                    pc.cells.pop(0)


def staging_goals(
    *,
    carrying: Dict[str, dict],
    positions: Dict[str, Cell],
    unload_pads_of: Dict[str, List[Cell]],
    blocked: Set[Cell],
    primary_goals: Dict[str, Cell],
    max_per_dest: int = 2,
) -> Tuple[Dict[str, Cell], int]:
    """Limit simultaneous approachers per destination; others stage mid-path.

    Staging cell = point on BFS path ~ halfway / ≥3 from pad (not parking exile).
    """
    goals = dict(primary_goals)
    n_staging = 0
    by_dest: Dict[str, List[str]] = defaultdict(list)
    for n, task in carrying.items():
        dest = str(task.get("destination") or "")
        by_dest[dest].append(n)
    for dest, group in by_dest.items():
        pads = []
        for n in group:
            pads = unload_pads_of.get(n) or pads
        if not pads:
            continue

        def _rank(n: str) -> Tuple:
            return (
                min(
                    abs(positions[n][0] - p[0]) + abs(positions[n][1] - p[1])
                    for p in pads
                ),
                n,
            )

        ranked = sorted(group, key=_rank)
        winners = set(ranked[: max(1, int(max_per_dest))])
        for n in ranked[len(winners) :]:
            pad = min(
                pads,
                key=lambda p: abs(positions[n][0] - p[0])
                + abs(positions[n][1] - p[1]),
            )
            field = bfs_field(pad, blocked)
            cur = positions[n]
            if cur not in field:
                continue
            # Walk toward pad until dist≈max(3, d/2)
            target_d = max(3, field[cur] // 2)
            pos = cur
            stage = cur
            guard = 0
            while pos in field and field[pos] > target_d and guard < 40:
                guard += 1
                best = None
                bd = field[pos]
                for nxt in _neighbors(pos, blocked):
                    d = field.get(nxt)
                    if d is not None and d < bd:
                        bd = d
                        best = nxt
                if best is None:
                    break
                pos = best
                stage = pos
            # Ensure not on pad
            if stage in pads or (
                abs(stage[0] - pad[0]) + abs(stage[1] - pad[1]) < 2
            ):
                # step back: use neighbor of cur away-ish
                for nxt in _neighbors(cur, blocked):
                    if nxt not in pads:
                        stage = nxt
                        break
            goals[n] = stage
            n_staging += 1
    return goals, n_staging


def sketches_from_commits(
    commits: Dict[str, PathCommit],
) -> Dict[str, IntentSketch]:
    return {
        n: IntentSketch(
            agent=n,
            cells=list(pc.cells),
            priority=float(pc.priority),
        )
        for n, pc in commits.items()
    }
