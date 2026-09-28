"""Traffic / follow acceleration for hard-wave batches (TrafficNet-lite).

When many agents share a dropoff (or pickup), force a shared outer-ring
highway and leader→follower planning order so joint waves finish with less
idle sync / head-on wait — the bulk of the 'useless' ticks in long same_drop
runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]


@dataclass
class WaveAccelPlan:
    """Hints consumed by ECBS turn-aware joint planner during escalated waves."""

    active: bool = False
    reason: str = ""
    convoy_mode: bool = False
    prefer_outer_ring: bool = True
    shared_highway_y: Optional[int] = None  # 2 or 19
    follow_order: List[str] = field(default_factory=list)  # leader first
    skip_idle_evac: bool = False
    hold_slack: int = 2  # shorter goal hold → less follower block
    dest_cell: Optional[Cell] = None


def _dominant_dest_cell(assigned: Dict[str, dict]) -> Optional[Cell]:
    """Pick a representative unload cell for the wave (mode of first end_point)."""
    counts: Dict[Cell, int] = {}
    for t in assigned.values():
        eps = t.get("end_points") or []
        if not eps:
            continue
        try:
            c = (int(eps[0][0]), int(eps[0][1]))
        except Exception:  # noqa: BLE001
            continue
        counts[c] = counts.get(c, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]


def pick_shared_highway_y(
    pose: Dict[str, Pose],
    goals: Dict[str, Cell],
    movers: Set[str],
) -> int:
    """Choose bottom (2) or top (19) ring that most agents already sit nearer to."""
    if not movers:
        return 2
    score2 = 0
    score19 = 0
    for n in movers:
        y = pose[n][1]
        gy = goals.get(n, pose[n][:2])[1]
        score2 += abs(y - 2) + abs(gy - 2)
        score19 += abs(y - 19) + abs(gy - 19)
    return 2 if score2 <= score19 else 19


def build_wave_accel(
    assigned: Dict[str, dict],
    pose: Dict[str, Pose],
    *,
    escalate: bool,
    dropoff_score: float = 0.0,
    bfs_len: Optional[Callable[[Cell, Cell, Set[Cell]], int]] = None,
    static: Optional[Set[Cell]] = None,
    goals: Optional[Dict[str, Cell]] = None,
) -> WaveAccelPlan:
    """Build accel plan for this wave. No-op when not escalated / no hotspot."""
    if not escalate or not assigned:
        return WaveAccelPlan(active=False, reason="off")

    hotspot = float(dropoff_score) >= 0.35 or len(assigned) >= 3
    if not hotspot:
        return WaveAccelPlan(
            active=True,
            reason="escalate_light",
            convoy_mode=False,
            prefer_outer_ring=True,
            follow_order=sorted(assigned.keys()),
            skip_idle_evac=True,
            hold_slack=4,
        )

    dest = _dominant_dest_cell(assigned)
    gmap: Dict[str, Cell] = {}
    if goals:
        gmap = {a: goals[a] for a in assigned if a in goals}
    else:
        for a, t in assigned.items():
            eps = t.get("end_points") or []
            if eps:
                gmap[a] = (int(eps[0][0]), int(eps[0][1]))
            elif dest is not None:
                gmap[a] = dest
            else:
                gmap[a] = pose[a][:2]

    movers = set(assigned.keys())
    hwy = pick_shared_highway_y(pose, gmap, movers)

    def _eta(n: str) -> float:
        g = gmap.get(n, pose[n][:2])
        if bfs_len is not None and static is not None:
            return float(bfs_len(pose[n][:2], g, static))
        return float(abs(pose[n][0] - g[0]) + abs(pose[n][1] - g[1]))

    # Leader = closest to unload (clears bay first); followers trail on same ring.
    order = sorted(movers, key=lambda n: (_eta(n), n))

    return WaveAccelPlan(
        active=True,
        reason="hotspot_convoy",
        convoy_mode=True,
        prefer_outer_ring=True,
        shared_highway_y=int(hwy),
        follow_order=order,
        skip_idle_evac=True,
        hold_slack=2,
        dest_cell=dest,
    )
