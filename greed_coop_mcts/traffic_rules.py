"""Auto-learned warehouse traffic rules (NOT TrafficNet).

Labels come from frequency cartesian OD coverage (artery / parking) plus
geometric one-way / yield / no-wait teachers. AGVs pick temporary micro-tasks
when the fleet stalls: park, clear junction, hold for pad, wait, or deliver.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

Cell = Tuple[int, int]


class TempTask(str, Enum):
    DELIVER = "deliver"  # pursue primary goal
    WAIT = "wait"  # dwell in place (soft force_stay)
    PARK = "park"  # move to yield/parking pocket
    CLEAR = "clear"  # leave junction / artery choke → parking
    HOLD_PAD = "hold_pad"  # wait in parking until unload pad free


@dataclass
class TrafficMaps:
    artery: np.ndarray  # (H,W) float
    parking: np.ndarray
    walkable: np.ndarray
    direction: np.ndarray  # (H,W) int preferred dir 0E1N2W3S / -1
    yield_mask: np.ndarray
    nowait: np.ndarray
    map_id: str = ""


_DIR4 = ((1, 0), (0, 1), (-1, 0), (0, -1))  # E N W S


def _degree(walk: np.ndarray, x: int, y: int) -> int:
    H, W = walk.shape
    d = 0
    for dx, dy in _DIR4:
        nx, ny = x + dx, y + dy
        if 0 <= nx < W and 0 <= ny < H and walk[ny, nx] > 0.5:
            d += 1
    return d


def _geom_direction(walk: np.ndarray) -> np.ndarray:
    H, W = walk.shape
    pref = np.full((H, W), -1, dtype=np.int64)
    for y in range(1, min(20, H - 1) + 1):
        for x in range(1, min(20, W - 1) + 1):
            if walk[y, x] < 0.5:
                continue
            d = 0 if (y % 2 == 0) else 2
            if _degree(walk, x, y) >= 3:
                d = 1 if (x % 2 == 0) else 3
            dx, dy = _DIR4[d]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H and walk[ny, nx] > 0.5):
                for alt in range(4):
                    adx, ady = _DIR4[alt]
                    ax, ay = x + adx, y + ady
                    if 0 <= ax < W and 0 <= ay < H and walk[ay, ax] > 0.5:
                        d = alt
                        break
            pref[y, x] = d
    return pref


def _geom_yield_nowait(
    walk: np.ndarray, artery: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = walk.shape
    yld = np.zeros((H, W), dtype=np.float32)
    nowait = np.zeros((H, W), dtype=np.float32)
    for y in range(1, min(20, H - 1) + 1):
        for x in range(1, min(20, W - 1) + 1):
            if walk[y, x] < 0.5:
                continue
            deg = _degree(walk, x, y)
            # No-wait only at true choke junctions that also sit on arteries.
            if deg >= 3 and float(artery[y, x]) >= 0.5:
                nowait[y, x] = 1.0
                continue
            near = False
            for dx, dy in _DIR4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and artery[ny, nx] >= 0.5:
                    near = True
                    break
            if near and float(artery[y, x]) < 0.5 and deg <= 2:
                yld[y, x] = 1.0
            elif deg == 1 and near:
                yld[y, x] = 1.0
    return yld, nowait


def learn_traffic_maps(meta: dict) -> TrafficMaps:
    """Frequency-learned artery/parking + geometric one-way/yield (no TrafficNet)."""
    map_id = str(meta.get("id") or meta.get("source_slot") or "")
    artery = parking = walk = None
    try:
        from ml_research.benchmarks.lane_traffic_ai.map_labels import (
            compute_labels_from_meta,
            resolve_map_id,
        )

        mid = resolve_map_id(map_id, meta) or map_id
        lab = compute_labels_from_meta(meta)
        artery = np.asarray(lab["artery"], dtype=np.float32)
        parking = np.asarray(lab["parking"], dtype=np.float32)
        walk = np.asarray(lab.get("walkable"), dtype=np.float32)
        map_id = mid
        print(
            f"[TRAFFIC-RULES] freq labels map={map_id} "
            f"artery={(artery >= 0.5).sum()} parking={(parking >= 0.5).sum()}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[TRAFFIC-RULES] freq fallback ({exc})", flush=True)

    if walk is None or float(np.sum(walk)) <= 0:
        walk = np.ones((21, 21), dtype=np.float32)
        walk[0, :] = 0
        walk[:, 0] = 0
        for p in meta.get("extra_obstacles") or []:
            if isinstance(p, dict):
                x, y = int(p["x"]), int(p["y"])
            else:
                x, y = int(p[0]), int(p[1])
            if 0 <= x < 21 and 0 <= y < 21:
                walk[y, x] = 0.0
        # crude artery: even rows/cols
        artery = np.zeros_like(walk)
        for y in range(1, 21):
            for x in range(1, 21):
                if walk[y, x] < 0.5:
                    continue
                if x % 4 == 0 or y % 4 == 0:
                    artery[y, x] = 1.0
        parking = ((walk > 0.5) & (artery < 0.5)).astype(np.float32)

    if artery is None:
        artery = np.zeros_like(walk)
    if parking is None:
        parking = ((walk > 0.5) & (artery < 0.5)).astype(np.float32)

    direction = _geom_direction(walk)
    yield_mask, nowait = _geom_yield_nowait(walk, artery)
    # Prefer freq parking; merge with geometric yield pockets
    parking = np.maximum(parking, yield_mask * (walk > 0.5).astype(np.float32))
    return TrafficMaps(
        artery=artery,
        parking=parking,
        walkable=walk,
        direction=direction,
        yield_mask=yield_mask,
        nowait=nowait,
        map_id=map_id,
    )


def _at(mask: np.ndarray, cell: Cell) -> float:
    x, y = int(cell[0]), int(cell[1])
    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
        return float(mask[y, x])
    return 0.0


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearest_parking(
    maps: TrafficMaps,
    cell: Cell,
    *,
    occupied: Set[Cell],
    blocked: Set[Cell],
    prefer_near: Optional[Cell] = None,
    prefer_far_from: Optional[Sequence[Cell]] = None,
    min_pad_dist: int = 3,
) -> Optional[Cell]:
    H, W = maps.parking.shape
    pads = list(prefer_far_from or ())
    cands: List[Cell] = []
    for y in range(H):
        for x in range(W):
            if maps.parking[y, x] < 0.5:
                continue
            c = (x, y)
            if c in blocked or c in occupied:
                continue
            if maps.walkable[y, x] < 0.5:
                continue
            if pads and min(_manh(c, p) for p in pads) < int(min_pad_dist):
                continue
            cands.append(c)
    if not cands and pads:
        # relax pad distance if nothing left
        return nearest_parking(
            maps,
            cell,
            occupied=occupied,
            blocked=blocked,
            prefer_near=prefer_near,
            prefer_far_from=None,
            min_pad_dist=0,
        )
    if not cands:
        return None
    anchor = prefer_near or cell

    def _key(c: Cell):
        far = -min((_manh(c, p) for p in pads), default=0)
        return (far if pads else 0, _manh(c, anchor), _manh(c, cell), c)

    return min(cands, key=_key)


def artery_waypoint(
    maps: TrafficMaps,
    cell: Cell,
    goal: Cell,
    *,
    blocked: Set[Cell],
) -> Cell:
    """Prefer a neighbor that follows learned one-way / stays on artery toward goal."""
    x, y = int(cell[0]), int(cell[1])
    if cell == goal:
        return cell
    best = cell
    best_score = 1e18
    pref = int(_at(maps.direction, cell))
    for i, (dx, dy) in enumerate(_DIR4):
        nxt = (x + dx, y + dy)
        if nxt in blocked:
            continue
        if _at(maps.walkable, nxt) < 0.5:
            continue
        score = float(_manh(nxt, goal))
        if _at(maps.artery, nxt) >= 0.5:
            score -= 0.35
        if pref >= 0 and i == pref:
            score -= 0.5
        # discourage stepping onto nowait if not improving much
        if _at(maps.nowait, nxt) >= 0.5 and _manh(nxt, goal) >= _manh(cell, goal):
            score += 0.8
        if score < best_score:
            best_score = score
            best = nxt
    return best


def pad_contested(
    pads: Sequence[Cell],
    positions: Dict[str, Cell],
    carriers: Sequence[str],
) -> bool:
    pad_set = set(pads)
    on_pad = sum(1 for ag in carriers if positions.get(ag) in pad_set)
    # also count agents adjacent to pads (approaching)
    near = 0
    for ag in carriers:
        p = positions.get(ag)
        if p is None:
            continue
        if p in pad_set:
            continue
        if any(_manh(p, pad) <= 2 for pad in pads):
            near += 1
    return (on_pad + near) >= 2


def assign_temp_tasks(
    maps: TrafficMaps,
    *,
    names: Sequence[str],
    positions: Dict[str, Cell],
    primary_goals: Dict[str, Cell],
    carrying: Set[str],
    assigned: Set[str],
    force_stay: Set[str],
    blocked: Set[Cell],
    unload_pads_of: Dict[str, List[Cell]],
    stall: int,
    wait_pri: Dict[str, float],
    stall_on: int = 40,
    max_active_unloaders: int = 2,
    sticky_winners: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[TempTask, Optional[Cell]]]:
    """Per-AGV temporary task from learned traffic rules.

    At most ``max_active_unloaders`` carriers pursue unload pads globally;
    others HOLD_PAD in parking far from pads (prevents approach thrashing).
    """
    out: Dict[str, Tuple[TempTask, Optional[Cell]]] = {
        n: (TempTask.DELIVER, primary_goals.get(n)) for n in names
    }
    occupied = set(positions.values())
    all_pads: List[Cell] = []
    for pads in unload_pads_of.values():
        all_pads.extend(pads)
    all_pads = list(set(all_pads))
    sticky = sticky_winners if sticky_winners is not None else {}

    carriers = [n for n in carrying if n not in force_stay and unload_pads_of.get(n)]
    if carriers:
        def _rank(n: str) -> Tuple:
            pads = unload_pads_of[n]
            return (
                0 if sticky.get("global") == n else 1,
                min(_manh(positions[n], p) for p in pads),
                -float(wait_pri.get(n, 0.0)),
                n,
            )

        ranked = sorted(carriers, key=_rank)
        winners = ranked[: max(1, int(max_active_unloaders))]
        if winners:
            sticky["global"] = winners[0]
        for n in ranked[len(winners) :]:
            park = nearest_parking(
                maps,
                positions[n],
                occupied=occupied - {positions[n]},
                blocked=blocked,
                prefer_far_from=all_pads or unload_pads_of.get(n),
                min_pad_dist=4,
            )
            if park is None:
                out[n] = (TempTask.WAIT, positions[n])
            else:
                out[n] = (TempTask.HOLD_PAD, park)
                occupied.add(park)

        # Same-pad group: even among winners, only one per pad set
        by_dest: Dict[str, List[str]] = {}
        for n in winners:
            pads = unload_pads_of.get(n) or []
            key = "|".join(f"{a},{b}" for a, b in sorted(pads))
            by_dest.setdefault(key, []).append(n)
        for key, group in by_dest.items():
            if len(group) < 2:
                continue
            pads = unload_pads_of[group[0]]
            g_ranked = sorted(
                group,
                key=lambda n: (
                    min(_manh(positions[n], p) for p in pads),
                    -float(wait_pri.get(n, 0.0)),
                    n,
                ),
            )
            for n in g_ranked[1:]:
                park = nearest_parking(
                    maps,
                    positions[n],
                    occupied=occupied - {positions[n]},
                    blocked=blocked,
                    prefer_far_from=pads,
                    min_pad_dist=4,
                )
                out[n] = (
                    (TempTask.HOLD_PAD, park)
                    if park is not None
                    else (TempTask.WAIT, positions[n])
                )
                if park is not None:
                    occupied.add(park)

    if stall < int(stall_on):
        return out

    for n in names:
        if n in force_stay:
            continue
        if out[n][0] != TempTask.DELIVER:
            continue
        cell = positions[n]
        if _at(maps.nowait, cell) >= 0.5 and float(wait_pri.get(n, 0)) > 30:
            park = nearest_parking(
                maps,
                cell,
                occupied=occupied - {cell},
                blocked=blocked,
                prefer_far_from=all_pads,
                min_pad_dist=3,
            )
            if park is not None and park != cell:
                out[n] = (TempTask.CLEAR, park)
                occupied.add(park)
                continue

        if n not in carrying and n not in assigned:
            if _at(maps.artery, cell) >= 0.5 or _at(maps.nowait, cell) >= 0.5:
                park = nearest_parking(
                    maps,
                    cell,
                    occupied=occupied - {cell},
                    blocked=blocked,
                    prefer_far_from=all_pads,
                )
                if park is not None:
                    out[n] = (TempTask.PARK, park)
                    occupied.add(park)
                else:
                    out[n] = (TempTask.WAIT, cell)
                continue

        goal = primary_goals.get(n, cell)
        if goal != cell and n in carrying:
            wp = artery_waypoint(maps, cell, goal, blocked=blocked)
            if wp != cell:
                out[n] = (TempTask.DELIVER, wp)

    return out


def prefer_dir_action(
    maps: TrafficMaps,
    cell: Cell,
    goal: Optional[Cell],
    others: Sequence[Cell],
    blocked: Set[Cell],
) -> Optional[Cell]:
    """Suggested next cell under traffic rules (for greedy ticks)."""
    if goal is None:
        return None
    return artery_waypoint(maps, cell, goal, blocked=blocked | set(others))
