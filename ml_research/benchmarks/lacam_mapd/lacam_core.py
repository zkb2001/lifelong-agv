"""LaCAM / LaCAM* configuration-space search (PIBT as low-level generator).

Reference: Okumura, LaCAM (AAAI'23) / LaCAM* (IJCAI'23).

Completeness (MAPF, finite graph): with unconstrained search budget the high-level
tree enumerates successors via constrained PIBT regenerations until a goal
configuration is found (when one exists). Callers that need completeness must
pass ``require_solution=True`` and a sufficient ``time_limit_s`` / ``max_nodes``.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from ml_research.benchmarks.pibt_mapd.pibt_core import pibt_one_step

Cell = Tuple[int, int]
Config = Tuple[Cell, ...]


@dataclass(order=True)
class _PQ:
    f: int
    g: int
    seq: int
    node_id: int = field(compare=False)


@dataclass
class _Node:
    config: Config
    g: int
    parent: Optional[int]
    constraints: frozenset  # {(agent_idx, cell), ...} forbidden for NEXT step


def _h(
    config: Config,
    names: Sequence[str],
    goals: Dict[str, Cell],
    dist_fields: Optional[Dict[Cell, Dict[Cell, int]]],
) -> int:
    total = 0
    fields = dist_fields or {}
    for i, name in enumerate(names):
        g = goals.get(name, config[i])
        fld = fields.get(g)
        if fld is not None and config[i] in fld:
            total += int(fld[config[i]])
        else:
            total += abs(config[i][0] - g[0]) + abs(config[i][1] - g[1])
    return total


def _goal_reached(
    config: Config, names: Sequence[str], goals: Dict[str, Cell]
) -> bool:
    return all(config[i] == goals.get(names[i], config[i]) for i in range(len(names)))


def _generate(
    *,
    names: Sequence[str],
    config: Config,
    goals: Dict[str, Cell],
    priority: Dict[str, float],
    blocked: Set[Cell],
    fixed: Set[str],
    dist_fields: Optional[Dict[Cell, Dict[Cell, int]]],
    constraints: frozenset,
) -> Optional[Config]:
    pos = {names[i]: config[i] for i in range(len(names))}
    forbid: Dict[str, Set[Cell]] = {}
    for ai, cell in constraints:
        if 0 <= int(ai) < len(names):
            forbid.setdefault(names[int(ai)], set()).add((int(cell[0]), int(cell[1])))

    desired = pibt_one_step(
        names=names,
        pos=pos,
        goals=goals,
        priority=priority,
        blocked=blocked,
        fixed=fixed,
        dist_fields=dist_fields,
        forbid=forbid,
    )
    # Safety: fixed agents must stay; forbidden next cells force stay if possible.
    occupied = {desired[n] for n in names}
    for name in names:
        if name in fixed:
            desired[name] = pos[name]
    occupied = {desired[n] for n in names}
    for name, bad in forbid.items():
        if name in fixed:
            continue
        nxt = desired[name]
        if nxt not in bad:
            continue
        cur = pos[name]
        occupied.discard(nxt)
        if cur in occupied:
            return None
        desired[name] = cur
        occupied.add(cur)
    return tuple(desired[n] for n in names)  # type: ignore[return-value]


def _reconstruct(nodes: List[_Node], nid: int, names: Sequence[str]) -> List[Dict[str, Cell]]:
    ids: List[int] = []
    cur: Optional[int] = nid
    while cur is not None:
        ids.append(cur)
        cur = nodes[cur].parent
    ids.reverse()
    path: List[Dict[str, Cell]] = []
    for i in ids:
        cfg = nodes[i].config
        path.append({names[j]: cfg[j] for j in range(len(names))})
    return path


def lacam_star_plan(
    *,
    names: Sequence[str],
    pos: Dict[str, Cell],
    goals: Dict[str, Cell],
    priority: Dict[str, float],
    blocked: Set[Cell],
    fixed: Optional[Set[str]] = None,
    dist_fields: Optional[Dict[Cell, Dict[Cell, int]]] = None,
    time_limit_s: float = 1.0,
    max_nodes: int = 50_000,
    improve: bool = False,
    require_solution: bool = True,
) -> List[Dict[str, Cell]]:
    """Return a collision-free configuration path ``[start, ..., goal]``.

    With ``require_solution=True`` (default), never returns a one-step PIBT
    fallback: either a full goal path or ``[start]`` if search exhausts.
    ``improve=True`` continues after the first solution (LaCAM*).
    """
    fixed_s = set(fixed or ())
    names_l = list(names)
    t_end = time.perf_counter() + float(time_limit_s)
    start: Config = tuple(pos[n] for n in names_l)  # type: ignore[misc]

    if _goal_reached(start, names_l, goals):
        return [{n: pos[n] for n in names_l}]

    nodes: List[_Node] = [
        _Node(config=start, g=0, parent=None, constraints=frozenset())
    ]
    open_h: List[_PQ] = []
    seq = 0
    heapq.heappush(
        open_h, _PQ(f=_h(start, names_l, goals, dist_fields), g=0, seq=seq, node_id=0)
    )
    seq += 1

    best_ids: Optional[List[int]] = None
    best_g = 10**18
    seen_edge: Set[Tuple[Config, Config]] = set()
    expansions = 0
    gen_count: Dict[int, int] = {}

    while open_h and expansions < int(max_nodes) and time.perf_counter() < t_end:
        item = heapq.heappop(open_h)
        nid = item.node_id
        node = nodes[nid]
        expansions += 1

        if _goal_reached(node.config, names_l, goals):
            if node.g < best_g:
                best_g = node.g
                best_ids = []
                cur: Optional[int] = nid
                while cur is not None:
                    best_ids.append(cur)
                    cur = nodes[cur].parent
                best_ids.reverse()
            if not improve:
                break
            continue

        succ = _generate(
            names=names_l,
            config=node.config,
            goals=goals,
            priority=priority,
            blocked=blocked,
            fixed=fixed_s,
            dist_fields=dist_fields,
            constraints=node.constraints,
        )
        if succ is None:
            continue

        edge = (node.config, succ)
        if edge in seen_edge:
            # LaCAM: regenerate with an added constraint on one mover.
            gen_count[nid] = gen_count.get(nid, 0) + 1
            if gen_count[nid] > 4 * len(names_l):
                continue
            movers = [
                i
                for i, name in enumerate(names_l)
                if name not in fixed_s and succ[i] != node.config[i]
            ]
            if not movers:
                movers = [
                    i
                    for i, name in enumerate(names_l)
                    if name not in fixed_s
                    and node.config[i] != goals.get(name, node.config[i])
                ]
            added = False
            for i in movers:
                cell = succ[i]
                if (i, cell) in node.constraints:
                    continue
                new_c = set(node.constraints)
                new_c.add((i, cell))
                nodes[nid] = _Node(
                    config=node.config,
                    g=node.g,
                    parent=node.parent,
                    constraints=frozenset(new_c),
                )
                hh = _h(node.config, names_l, goals, dist_fields)
                heapq.heappush(
                    open_h, _PQ(f=node.g + hh, g=node.g, seq=seq, node_id=nid)
                )
                seq += 1
                added = True
                break
            if not added:
                continue
            continue
        seen_edge.add(edge)

        if succ == node.config:
            # All-stay: constrain a non-goal agent away from staying.
            for i, name in enumerate(names_l):
                if name in fixed_s:
                    continue
                if node.config[i] == goals.get(name, node.config[i]):
                    continue
                cell = node.config[i]
                if (i, cell) in node.constraints:
                    continue
                new_c = set(node.constraints)
                new_c.add((i, cell))
                nodes[nid] = _Node(
                    config=node.config,
                    g=node.g,
                    parent=node.parent,
                    constraints=frozenset(new_c),
                )
                hh = _h(node.config, names_l, goals, dist_fields)
                heapq.heappush(
                    open_h, _PQ(f=node.g + hh, g=node.g, seq=seq, node_id=nid)
                )
                seq += 1
                break
            continue

        child = _Node(config=succ, g=node.g + 1, parent=nid, constraints=frozenset())
        cid = len(nodes)
        nodes.append(child)
        hh = _h(succ, names_l, goals, dist_fields)
        heapq.heappush(open_h, _PQ(f=child.g + hh, g=child.g, seq=seq, node_id=cid))
        seq += 1

    if best_ids:
        return _reconstruct(nodes, best_ids[-1], names_l)

    if require_solution:
        # Incomplete: caller should raise budget / change goals.
        return [{n: pos[n] for n in names_l}]

    nxt = pibt_one_step(
        names=names_l,
        pos=pos,
        goals=goals,
        priority=priority,
        blocked=blocked,
        fixed=fixed_s,
        dist_fields=dist_fields,
    )
    return [dict(pos), nxt]


def lacam_complete_plan(
    *,
    names: Sequence[str],
    pos: Dict[str, Cell],
    goals: Dict[str, Cell],
    priority: Dict[str, float],
    blocked: Set[Cell],
    fixed: Optional[Set[str]] = None,
    dist_fields: Optional[Dict[Cell, Dict[Cell, int]]] = None,
    wall_deadline: Optional[float] = None,
    base_time_s: float = 0.25,
    base_nodes: int = 8_000,
    improve_s: float = 0.05,
) -> Tuple[List[Dict[str, Cell]], bool]:
    """Adaptive-budget LaCAM until a full goal path is found (or wall exhausted).

    Returns ``(path, solved)``. Path always starts at ``pos``.
    """
    # Few escalating attempts; total wall per call stays well below ECBS scale.
    budgets = [
        (float(base_time_s), int(base_nodes)),
        (min(0.8, float(base_time_s) * 3), int(base_nodes) * 3),
        (min(1.5, float(base_time_s) * 8), int(base_nodes) * 8),
    ]
    plan: List[Dict[str, Cell]] = [{n: pos[n] for n in names}]
    for tlim, nlim in budgets:
        if wall_deadline is not None:
            remain = float(wall_deadline) - time.perf_counter()
            if remain <= 0.02:
                break
            tlim = min(float(tlim), max(0.05, remain * 0.5))
        plan = lacam_star_plan(
            names=names,
            pos=pos,
            goals=goals,
            priority=priority,
            blocked=blocked,
            fixed=fixed,
            dist_fields=dist_fields,
            time_limit_s=tlim,
            max_nodes=nlim,
            improve=False,
            require_solution=True,
        )
        if len(plan) >= 1 and _goal_reached(
            tuple(plan[-1][n] for n in names),  # type: ignore[misc]
            names,
            goals,
        ):
            # Optional short LaCAM* improve after first solution.
            if improve_s > 0:
                if wall_deadline is not None:
                    remain = float(wall_deadline) - time.perf_counter()
                    if remain < improve_s:
                        return plan, True
                better = lacam_star_plan(
                    names=names,
                    pos=pos,
                    goals=goals,
                    priority=priority,
                    blocked=blocked,
                    fixed=fixed,
                    dist_fields=dist_fields,
                    time_limit_s=min(float(improve_s), 0.1),
                    max_nodes=max(2000, nlim // 2),
                    improve=True,
                    require_solution=True,
                )
                if len(better) >= 2 and _goal_reached(
                    tuple(better[-1][n] for n in names),  # type: ignore[misc]
                    names,
                    goals,
                ):
                    if len(better) < len(plan):
                        plan = better
            return plan, True
    return plan, False


def lacam_star_next(
    *,
    names: Sequence[str],
    pos: Dict[str, Cell],
    goals: Dict[str, Cell],
    priority: Dict[str, float],
    blocked: Set[Cell],
    fixed: Optional[Set[str]] = None,
    dist_fields: Optional[Dict[Cell, Dict[Cell, int]]] = None,
    time_limit_s: float = 0.5,
    max_nodes: int = 20_000,
    plan_cache: Optional[List[Dict[str, Cell]]] = None,
    wall_deadline: Optional[float] = None,
) -> Tuple[Dict[str, Cell], List[Dict[str, Cell]], bool]:
    """Return ``(next_config, remaining_plan_incl_next, solved)``."""
    if plan_cache and len(plan_cache) >= 2:
        cur = plan_cache[0]
        if all(cur.get(n) == pos.get(n) for n in names):
            rest = plan_cache[1:]
            return rest[0], rest, True

    plan, solved = lacam_complete_plan(
        names=names,
        pos=pos,
        goals=goals,
        priority=priority,
        blocked=blocked,
        fixed=fixed,
        dist_fields=dist_fields,
        wall_deadline=wall_deadline,
        base_time_s=max(0.1, float(time_limit_s)),
        base_nodes=int(max_nodes),
        improve_s=0.05,
    )
    if len(plan) < 2:
        # Still must move something: one constrained PIBT step (progress attempt).
        nxt = pibt_one_step(
            names=names,
            pos=pos,
            goals=goals,
            priority=priority,
            blocked=blocked,
            fixed=fixed,
            dist_fields=dist_fields,
        )
        return nxt, [dict(pos), nxt], False
    return plan[1], plan[1:], solved
