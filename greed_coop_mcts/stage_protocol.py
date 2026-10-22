"""GLocal Stage Protocol — messages between global AI and per-AGV local AI.

Downlink: StageContract / ConflictAlert / Residual
Uplink:   IntentSketch / RiskVote
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]


class StageRole(str, Enum):
    TO_PICK = "to_pick"
    TO_DROP = "to_drop"
    YIELD = "yield"
    HOLD = "hold"
    CLEAR = "clear"
    IDLE = "idle"


@dataclass
class StageContract:
    """Global → AGV stage assignment (slow timescale)."""

    agent: str
    role: StageRole
    waypoints: List[Cell]
    goal: Cell
    priority: float = 0.0
    ttl: int = 16
    exclusive_pad: Optional[Cell] = None
    issued_t: int = 0

    def active_waypoint(self, pos: Cell) -> Cell:
        """First waypoint not yet reached; else final goal."""
        for wp in self.waypoints:
            if pos != wp:
                return wp
        return self.goal if self.goal else (self.waypoints[-1] if self.waypoints else pos)

    def advance(self, pos: Cell) -> None:
        while self.waypoints and self.waypoints[0] == pos:
            self.waypoints.pop(0)

    def expired(self, t: int) -> bool:
        return int(t) - int(self.issued_t) >= int(self.ttl)


@dataclass
class IntentSketch:
    """AGV → global short-horizon occupancy forecast."""

    agent: str
    cells: List[Cell]  # future cells at t+1 .. t+H (may include waits)
    priority: float = 0.0
    blocked_streak: int = 0


@dataclass
class RiskVote:
    agent: str
    stuck: bool = False
    head_on: bool = False
    waiting_pad: bool = False
    request_restage: bool = False


@dataclass
class ConflictAlert:
    """IC-MAPF style targeted alert."""

    cells: Set[Cell] = field(default_factory=set)
    yielders: Set[str] = field(default_factory=set)
    holders: Set[str] = field(default_factory=set)


@dataclass
class Residual:
    """Cloud residual override for a single AGV (next cell)."""

    agent: str
    next_cell: Cell
    reason: str = ""


@dataclass
class CommStats:
    n_restage: int = 0
    n_residual: int = 0
    n_conflict_alerts: int = 0
    sum_path_horizon: int = 0
    n_local_plans: int = 0

    def avg_path_horizon(self) -> float:
        if self.n_local_plans <= 0:
            return 0.0
        return float(self.sum_path_horizon) / float(self.n_local_plans)


def detect_sketch_conflicts(
    sketches: Dict[str, IntentSketch],
    *,
    positions: Dict[str, Cell],
) -> ConflictAlert:
    """Find vertex / swap conflicts in short-horizon sketches."""
    alert = ConflictAlert()
    names = list(sketches.keys())
    # Horizon-aligned vertex conflicts
    max_h = max((len(s.cells) for s in sketches.values()), default=0)
    for h in range(max_h):
        at: Dict[Cell, List[str]] = {}
        for n in names:
            cells = sketches[n].cells
            if h < len(cells):
                c = cells[h]
            else:
                c = cells[-1] if cells else positions.get(n, (0, 0))
            at.setdefault(c, []).append(n)
        for cell, ags in at.items():
            if len(ags) < 2:
                continue
            alert.cells.add(cell)
            ranked = sorted(
                ags, key=lambda a: (-float(sketches[a].priority), a)
            )
            alert.holders.add(ranked[0])
            for y in ranked[1:]:
                alert.yielders.add(y)

    # Immediate swap: A wants B's cell, B wants A's
    for i, a in enumerate(names):
        ca = sketches[a].cells
        if not ca:
            continue
        pa = positions[a]
        for b in names[i + 1 :]:
            cb = sketches[b].cells
            if not cb:
                continue
            pb = positions[b]
            if ca[0] == pb and cb[0] == pa and pa != pb:
                alert.cells.add(pa)
                alert.cells.add(pb)
                if float(sketches[a].priority) >= float(sketches[b].priority):
                    alert.holders.add(a)
                    alert.yielders.add(b)
                else:
                    alert.holders.add(b)
                    alert.yielders.add(a)
    return alert


def arbitrate_intents(
    sketches: Dict[str, IntentSketch],
    *,
    positions: Dict[str, Cell],
    force_stay: Set[str],
) -> Tuple[Dict[str, Cell], List[Residual], ConflictAlert]:
    """Resolve conflicts: keep high priority, force yielders to WAIT (no illegal swap).

    Returns desired next cells, residual overrides, and the alert used.
    """
    alert = detect_sketch_conflicts(sketches, positions=positions)
    desired: Dict[str, Cell] = {}
    residuals: List[Residual] = []

    for n, sk in sketches.items():
        if n in force_stay:
            desired[n] = positions[n]
            continue
        if not sk.cells:
            desired[n] = positions[n]
            continue
        nxt = sk.cells[0]
        if n in alert.yielders:
            # Stay put — residual override
            desired[n] = positions[n]
            residuals.append(
                Residual(agent=n, next_cell=positions[n], reason="yield_conflict")
            )
        else:
            desired[n] = nxt

    # Second pass: prevent two non-yielders claiming same next cell
    claim: Dict[Cell, str] = {}
    order = sorted(
        desired.keys(),
        key=lambda a: (
            -float(sketches[a].priority) if a in sketches else 0.0,
            a,
        ),
    )
    for n in order:
        if n in force_stay:
            continue
        nxt = desired[n]
        cur = positions[n]
        if nxt == cur:
            continue
        # Block swap even if both somehow still moving
        holder_pos = {positions[m]: m for m in positions}
        victim = holder_pos.get(nxt)
        if victim is not None and victim != n:
            v_want = desired.get(victim, positions[victim])
            if v_want == cur:
                desired[n] = cur
                residuals.append(
                    Residual(agent=n, next_cell=cur, reason="block_swap")
                )
                continue
            if victim not in claim and desired.get(victim) == positions[victim]:
                # Stationary body — cannot enter
                desired[n] = cur
                residuals.append(
                    Residual(agent=n, next_cell=cur, reason="body_block")
                )
                continue
        if nxt in claim:
            desired[n] = cur
            residuals.append(
                Residual(agent=n, next_cell=cur, reason="vertex_claim")
            )
            continue
        claim[nxt] = n

    return desired, residuals, alert
