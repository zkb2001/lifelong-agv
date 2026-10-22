"""Region soft-capacity admit for RRSC-v2 parallel dispatch.

Geometric tile regions (no narrow-corridor special casing). Soft caps limit
how many stage goals may land in one region; other regions stay fully parallel.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from greed_coop_mcts.stage_net import KIND_CLEAR, KIND_FINAL, StageChoice, region_id
from greed_coop_mcts.stage_protocol import StageRole  # noqa: F401 — used in protect_roles

Cell = Tuple[int, int]
Region = Tuple[int, int]


def count_free_by_region(
    blocked: Set[Cell], *, tile: int = 5, grid: int = 21
) -> Dict[Region, int]:
    free: Dict[Region, int] = defaultdict(int)
    for x in range(1, grid):
        for y in range(1, grid):
            if (x, y) in blocked:
                continue
            free[region_id((x, y), tile=tile)] += 1
    return dict(free)


def region_caps(
    free_by_region: Dict[Region, int], *, k: int = 8, min_cap: int = 2, max_cap: int = 5
) -> Dict[Region, int]:
    """Soft capacity from free cells: denser free → higher parallel admit."""
    caps: Dict[Region, int] = {}
    for rid, nfree in free_by_region.items():
        caps[rid] = int(max(min_cap, min(max_cap, nfree // max(1, k))))
    return caps


def occupancy_from_goals(
    goals: Dict[str, Cell], *, tile: int = 5
) -> Dict[Region, int]:
    occ: Dict[Region, int] = defaultdict(int)
    for g in goals.values():
        occ[region_id(g, tile=tile)] += 1
    return dict(occ)


def apply_region_admit(
    choices: Dict[str, StageChoice],
    *,
    caps: Dict[Region, int],
    paths: Dict[str, List[Cell]],
    tile: int = 5,
    protect_roles: Optional[Set[StageRole]] = None,
) -> Dict[str, StageChoice]:
    """Soft-cap *work* FINAL targets per region; never clog arteries with idle.

    - TO_DROP / TO_PICK keep FINAL (completion-critical).
    - IDLE / YIELD / HOLD home parks do **not** compete for work caps and are
      never demoted onto path midpoints (that caused SH01 endgame livelock).
    - Only non-protected work-bound FINAL (if any) may be demoted to CLEAR,
      preferring a CLEAR candidate already on the path near the agent.
    """
    protect = protect_roles or {StageRole.TO_DROP, StageRole.TO_PICK}
    # Homes / yield do not consume work-region capacity.
    skip_cap_roles = {
        StageRole.IDLE,
        StageRole.YIELD,
        StageRole.HOLD,
        StageRole.CLEAR,
    }
    order = sorted(
        choices.keys(), key=lambda n: (-float(choices[n].priority), n)
    )
    occ: Dict[Region, int] = defaultdict(int)
    out: Dict[str, StageChoice] = {}
    for n in order:
        ch = choices[n]
        cell = ch.cell
        rid = region_id(cell, tile=tile)
        cap = int(caps.get(rid, 3))
        protected = ch.role in protect
        counts_toward_cap = (
            ch.kind == KIND_FINAL and ch.role not in skip_cap_roles
        )
        if (
            (not protected)
            and counts_toward_cap
            and occ[rid] >= cap
        ):
            path = paths.get(n) or []
            # Prefer a nearby cell on the path (early prefix), not the midpoint
            # of a long home route that often sits on shared arteries.
            if len(path) >= 3:
                alt = path[min(2, len(path) - 1)]
            elif path:
                alt = path[0]
            else:
                alt = ch.cell
            out[n] = StageChoice(
                agent=n,
                kind=KIND_CLEAR,
                cell=alt,
                role=StageRole.YIELD,
                priority=ch.priority,
                logits=ch.logits,
            )
            occ[region_id(alt, tile=tile)] += 1
        else:
            out[n] = ch
            if counts_toward_cap or protected:
                occ[rid] += 1
    return out


def fleet_hard_cap_default() -> int:
    """Safety ceiling; region claim soft-caps add parallelism without overload."""
    return 8


def claim_region_ok(
    door: Cell,
    *,
    region_occ: Dict[Region, int],
    caps: Dict[Region, int],
    tile: int = 5,
) -> bool:
    """True if assigning another work FINAL into door's region is admissible."""
    rid = region_id(door, tile=tile)
    return int(region_occ.get(rid, 0)) < int(caps.get(rid, 3))
