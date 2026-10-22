"""RHCR-style windowed LaCAM* (plan H, commit E, replan) for lifelong MAPD."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ml_research.benchmarks.lacam_mapd.lacam_core import lacam_star_plan
from ml_research.benchmarks.pibt_mapd.pibt_core import bfs_dist_field

Cell = Tuple[int, int]


def _window_goal(
    cur: Cell,
    goal: Cell,
    blocked: Set[Cell],
    horizon: int,
    field_cache: Dict[Cell, Dict[Cell, int]],
) -> Cell:
    if cur == goal:
        return cur
    if goal not in field_cache:
        field_cache[goal] = bfs_dist_field(goal, blocked)
    field = field_cache[goal]
    if cur not in field:
        return goal
    # Walk greedily toward goal for up to ``horizon`` steps on the distance field.
    pos = cur
    for _ in range(max(1, int(horizon))):
        if pos == goal:
            break
        best = pos
        best_d = field.get(pos, 10**9)
        x, y = pos
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt in blocked:
                continue
            d = field.get(nxt)
            if d is not None and d < best_d:
                best_d = d
                best = nxt
        if best == pos:
            break
        pos = best
    return pos


def windowed_lacam_paths(
    *,
    names: Sequence[str],
    pos: Dict[str, Cell],
    goals: Dict[str, Cell],
    priority: Dict[str, float],
    blocked: Set[Cell],
    fixed: Optional[Set[str]] = None,
    dist_fields: Optional[Dict[Cell, Dict[Cell, int]]] = None,
    plan_horizon: int = 16,
    exec_horizon: int = 8,
    max_rounds: int = 64,
    time_limit_s: float = 2.0,
    wall_deadline: Optional[float] = None,
) -> Optional[Dict[str, List[Cell]]]:
    """Return per-agent cell paths from ``pos`` toward ``goals``, or None."""
    fixed_s = set(fixed or ())
    names_l = list(names)
    H = max(4, int(plan_horizon))
    E = max(1, min(int(exec_horizon), H))
    cur = {n: pos[n] for n in names_l}
    committed: Dict[str, List[Cell]] = {n: [pos[n]] for n in names_l}
    field_cache: Dict[Cell, Dict[Cell, int]] = dict(dist_fields or {})
    t_end = time.perf_counter() + float(time_limit_s)

    for rnd in range(max(1, int(max_rounds))):
        if wall_deadline is not None and time.perf_counter() >= wall_deadline:
            break
        if all(cur[n] == goals.get(n, cur[n]) for n in names_l):
            break

        win_goals = {
            n: (
                cur[n]
                if n in fixed_s
                else _window_goal(cur[n], goals.get(n, cur[n]), blocked, H, field_cache)
            )
            for n in names_l
        }
        # Unique window goals (injective).
        used: Set[Cell] = {win_goals[n] for n in fixed_s}
        for n in sorted(names_l, key=lambda a: (-float(priority.get(a, 0.0)), a)):
            if n in fixed_s:
                continue
            g = win_goals[n]
            if g not in used and g not in blocked:
                used.add(g)
                continue
            # Nearby alternate.
            x, y = g
            alt = None
            for rad in range(0, 6):
                for dx in range(-rad, rad + 1):
                    for dy in range(-rad, rad + 1):
                        if abs(dx) + abs(dy) != rad:
                            continue
                        c = (x + dx, y + dy)
                        if c in blocked or c in used:
                            continue
                        if not (1 <= c[0] <= 20 and 1 <= c[1] <= 20):
                            continue
                        alt = c
                        break
                    if alt:
                        break
                if alt:
                    break
            win_goals[n] = alt if alt is not None else cur[n]
            used.add(win_goals[n])

        remain = max(0.05, t_end - time.perf_counter())
        if wall_deadline is not None:
            remain = min(remain, max(0.05, wall_deadline - time.perf_counter()))
        win_tl = min(0.35, remain * 0.4) if rnd > 0 else min(0.5, remain * 0.5)

        fields = {g: field_cache.setdefault(g, bfs_dist_field(g, blocked)) for g in set(win_goals.values())}
        plan = lacam_star_plan(
            names=names_l,
            pos=cur,
            goals=win_goals,
            priority=priority,
            blocked=blocked,
            fixed=fixed_s,
            dist_fields=fields,
            time_limit_s=win_tl,
            max_nodes=12_000,
            improve=False,
            require_solution=True,
        )
        solved = len(plan) >= 2 and all(plan[-1][n] == win_goals[n] for n in names_l)
        if not solved:
            # Escalate once to true goals.
            remain = max(0.05, t_end - time.perf_counter())
            fields2 = {
                g: field_cache.setdefault(g, bfs_dist_field(g, blocked))
                for g in set(goals.values())
            }
            plan = lacam_star_plan(
                names=names_l,
                pos=cur,
                goals=goals,
                priority=priority,
                blocked=blocked,
                fixed=fixed_s,
                dist_fields=fields2,
                time_limit_s=min(1.0, remain * 0.8),
                max_nodes=40_000,
                improve=False,
                require_solution=True,
            )
            solved = len(plan) >= 2 and all(
                plan[-1][n] == goals.get(n, cur[n]) for n in names_l
            )
            if not solved:
                if rnd == 0:
                    return None
                break

        # Commit up to E steps from the plan (plan[0] is current).
        body = plan[1:]
        step_n = min(E, len(body))
        if step_n <= 0:
            break
        for i in range(step_n):
            cfg = body[i]
            for n in names_l:
                committed[n].append(cfg[n])
                cur[n] = cfg[n]

    # Align lengths
    T = max(len(committed[n]) for n in names_l)
    for n in names_l:
        while len(committed[n]) < T:
            committed[n].append(committed[n][-1])
    return committed
