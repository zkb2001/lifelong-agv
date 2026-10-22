"""GGO-style guidance graph: directed edge weights inside existing A*/ST search.

Paper: Zhang, Li et al., IJCAI 2024 *Guidance Graph Optimization*.
Runtime only multiplies move/wait costs — not a third planner.

Default is **gain-gated**: inflate only on maps where A/B beat vanilla Hier.
Unproven maps are a no-op (identical to Hier). Bypass with AGV_GGO_FORCE=1.
"""
from __future__ import annotations

import os
from typing import Optional, Set, Tuple

Cell = Tuple[int, int]

# Measured sim < Hier (100/100 VALID). Do not add a slot until an A/B win.
PROVEN_GAIN_SLOTS: Set[int] = {2, 3, 9, 12}
# Two GGO policies both raised sim on maze-like maps. Stay off.
# Raised sim or maze (n_narrow>=50): ring GGO stays off (= Hier).
GGO_HARM_SLOTS: Set[int] = {5, 6, 7, 8, 10, 11, 14, 17, 18, 19}
# Unproven maps with this many degree-≤2 cells: ring highways fight the maze.
_MAZE_NARROW = 80

_ENABLED: Optional[bool] = None
_ACTIVE = False
_AGAINST = 1.22
_NARROW = 1.12
_WAIT = 1.0
_loaded = False
_static: Set[Cell] = set()
_narrow: Set[Cell] = set()
_slot: Optional[int] = None
_RING = (2, 19)
_CW = False
_CROWD_ONLY = False
_CROWD_R = 3


def _truthy(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() not in ("0", "false", "off", "no", "")


def ggo_enabled() -> bool:
    """Master switch (env / meta). Does not mean inflate is live."""
    global _ENABLED
    if _ENABLED is not None:
        return bool(_ENABLED)
    v = os.environ.get("AGV_GGO")
    if v is None:
        return False
    return str(v).strip().lower() not in ("0", "false", "off", "no")


def ggo_active() -> bool:
    """True only when edge weights actually differ from 1.0."""
    return bool(ggo_enabled() and _ACTIVE)


def ggo_spacetime_enabled() -> bool:
    """Joint ST search: only always-on proven rings. Crowd-only M0 must not
    switch BFS → Dijkstra (that blew up SH08 after handoff).
    """
    return bool(ggo_active() and not _CROWD_ONLY)


def ggo_harm_slots() -> Set[int]:
    raw = os.environ.get("AGV_GGO_HARM")
    if raw:
        out: Set[int] = set()
        for p in raw.replace(";", ",").split(","):
            p = p.strip()
            if p.isdigit():
                out.add(int(p))
        return out
    return set(GGO_HARM_SLOTS)


def set_ggo_enabled(on: bool) -> None:
    global _ENABLED
    _ENABLED = bool(on)


def proven_gain_slots() -> Set[int]:
    raw = os.environ.get("AGV_GGO_SLOTS")
    if raw:
        out: Set[int] = set()
        for p in raw.replace(";", ",").split(","):
            p = p.strip()
            if p.isdigit():
                out.add(int(p))
        return out
    return set(PROVEN_GAIN_SLOTS)


def slot_allowed(slot: Optional[int]) -> bool:
    """Master GGO may always attach. Proven slots use always-on ring;
    others use crowd-only (no empty-map detours). REQUIRE_GAIN=1 keeps
    unproven fully off (old gated deploy). Default 0 = crowd-only on all.
    """
    if _truthy("AGV_GGO_FORCE"):
        return True
    if slot is None:
        return ggo_enabled()
    if int(slot) in proven_gain_slots():
        return True
    if int(slot) in ggo_harm_slots() and not _truthy("AGV_GGO_FORCE"):
        return False
    if _truthy("AGV_GGO_REQUIRE_GAIN", "0"):
        return False
    return True


def _free(c: Cell) -> bool:
    x, y = int(c[0]), int(c[1])
    if not (1 <= x <= 20 and 1 <= y <= 20):
        return False
    return (x, y) not in _static


def _rebuild_narrow() -> None:
    global _narrow
    cells: Set[Cell] = set()
    for x in range(1, 21):
        for y in range(1, 21):
            c = (x, y)
            if not _free(c):
                continue
            deg = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if _free((x + dx, y + dy)):
                    deg += 1
            if deg <= 2:
                cells.add(c)
    _narrow = cells


def configure_ggo(*, static_obs, slot: Optional[int] = None) -> None:
    global _static, _slot, _loaded, _AGAINST, _WAIT, _NARROW, _CW, _ACTIVE, _CROWD_ONLY
    _static = {tuple(c) for c in (static_obs or [])}  # type: ignore[misc]
    _slot = None if slot is None else int(slot)
    _loaded = True
    _ACTIVE = False
    proven = _slot is not None and int(_slot) in proven_gain_slots()
    try:
        _WAIT = float(os.environ.get("AGV_GGO_WAIT", "1.0"))
    except ValueError:
        _WAIT = 1.0
    cw_env = os.environ.get("AGV_GGO_CW")
    if cw_env is None:
        # SH12 always-on; SH13/SH20 crowd A/B only beat Hier clockwise.
        _CW = _slot in (12, 13, 20)
    else:
        _CW = str(cw_env).strip().lower() in ("1", "true", "yes", "on")
    if proven:
        _CROWD_ONLY = False
        try:
            _AGAINST = float(os.environ.get("AGV_GGO_AGAINST", "1.12" if _CW else "1.22"))
        except ValueError:
            _AGAINST = 1.12 if _CW else 1.22
        try:
            _NARROW = float(os.environ.get("AGV_GGO_NARROW", "1.0" if _CW else "1.12"))
        except ValueError:
            _NARROW = 1.0 if _CW else 1.12
    else:
        # Unproven: never rewrite empty shortest paths (SH08 +7349 was always-on).
        _CROWD_ONLY = not _truthy("AGV_GGO_ALWAYS", "0")
        try:
            _AGAINST = float(os.environ.get("AGV_GGO_AGAINST", "1.15"))
        except ValueError:
            _AGAINST = 1.15
        try:
            _NARROW = float(os.environ.get("AGV_GGO_NARROW", "1.0"))
        except ValueError:
            _NARROW = 1.0
    if os.environ.get("AGV_GGO_CROWD_ONLY") is not None:
        _CROWD_ONLY = _truthy("AGV_GGO_CROWD_ONLY")
    _rebuild_narrow()
    maze = (not proven) and len(_narrow) >= int(os.environ.get("AGV_GGO_MAZE_NARROW", str(_MAZE_NARROW)))
    if maze and not _truthy("AGV_GGO_FORCE"):
        _ACTIVE = False
        print(
            f"[GGO] OFF maze slot={_slot} n_narrow={len(_narrow)} "
            f"(>= {os.environ.get('AGV_GGO_MAZE_NARROW', _MAZE_NARROW)}; ring hurts sim)",
            flush=True,
        )
        return
    if _slot is not None and int(_slot) in ggo_harm_slots() and not _truthy("AGV_GGO_FORCE"):
        _ACTIVE = False
        _CROWD_ONLY = True
        print(
            f"[GGO] OFF harm-slot={_slot} (prior A/B raised sim; "
            f"harm={sorted(ggo_harm_slots())})",
            flush=True,
        )
        return
    allow = slot_allowed(_slot)
    _ACTIVE = bool(ggo_enabled() and allow)
    if ggo_enabled() and not allow:
        print(
            f"[GGO] OFF unproven slot={_slot} (require_gain; "
            f"proven={sorted(proven_gain_slots())})",
            flush=True,
        )
    elif _ACTIVE:
        way = "CW" if _CW else "CCW"
        mode = "crowd" if _CROWD_ONLY else "always"
        print(
            f"[GGO] ring {way} {mode} slot={_slot} against={_AGAINST} "
            f"narrow={_NARROW} wait={_WAIT} n_narrow={len(_narrow)}",
            flush=True,
        )


def _along_corridor(a: Cell, b: Cell) -> bool:
    """Cohen parity only for 1-wide cells (not open floor)."""
    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])
    dx, dy = bx - ax, by - ay
    if dx == 1 and dy == 0:
        return (ay % 2) == 0
    if dx == -1 and dy == 0:
        return (ay % 2) == 1
    if dy == 1 and dx == 0:
        return (ax % 2) == 0
    if dy == -1 and dx == 0:
        return (ax % 2) == 1
    return True


def _ring_inflate(a: Cell, b: Cell) -> Optional[float]:
    """CCW outer ring: y=2 +x, x=19 +y, y=19 -x, x=2 -y."""
    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])
    dx, dy = bx - ax, by - ay
    lo, hi = _RING
    east, north = (dx == -1, dy == -1) if _CW else (dx == 1, dy == 1)
    if ay == lo and by == lo and dy == 0:
        return 1.0 if east else float(_AGAINST)
    if ay == hi and by == hi and dy == 0:
        return 1.0 if (not east) else float(_AGAINST)
    if ax == hi and bx == hi and dx == 0:
        return 1.0 if north else float(_AGAINST)
    if ax == lo and bx == lo and dx == 0:
        return 1.0 if (not north) else float(_AGAINST)
    return None


def _pose_xy(pos) -> Optional[Cell]:
    if pos is None:
        return None
    if isinstance(pos, dict):
        p = pos.get("pos") or pos.get("cell")
        return _pose_xy(p)
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        try:
            return (int(pos[0]), int(pos[1]))
        except (TypeError, ValueError):
            return None
    return None


def _others_near(
    cell: Cell,
    moving,
    who: Optional[str],
    t: Optional[int],
) -> bool:
    if not moving:
        return False
    cx, cy = int(cell[0]), int(cell[1])
    r = int(_CROWD_R)
    tt = 0 if t is None else int(t)
    for name, traj in moving.items():
        if who is not None and str(name) == str(who):
            continue
        if not traj:
            continue
        p = None
        if isinstance(traj, dict):
            p = _pose_xy(traj)
        elif isinstance(traj, (list, tuple)):
            if traj and isinstance(traj[0], (list, tuple)):
                idx = tt if 0 <= tt < len(traj) else -1
                p = _pose_xy(traj[idx])
            else:
                p = _pose_xy(traj)
        else:
            p = _pose_xy(traj)
        if p is None:
            continue
        if abs(p[0] - cx) + abs(p[1] - cy) <= r:
            return True
    return False


def ggo_step_inflate(
    cur: Cell,
    nxt: Cell,
    moving=None,
    who=None,
    t=None,
    *,
    force: bool = False,
) -> float:
    """Cost multiplier for one wait or cardinal step. 1.0 if GGO off/gated.

    ``force=True``: spacetime / joint search (already multi-agent).
    Crowd-only M0: inflate only if another AGV is nearby (empty A* unchanged).
    """
    if not ggo_active():
        return 1.0
    if cur == nxt:
        return float(_WAIT)
    if _CROWD_ONLY and not force:
        ncell = (int(nxt[0]), int(nxt[1]))
        if not _others_near(ncell, moving, None if who is None else str(who), t):
            return 1.0
    ring = _ring_inflate(cur, nxt)
    if ring is not None:
        return ring
    c = (int(cur[0]), int(cur[1]))
    n = (int(nxt[0]), int(nxt[1]))
    if _NARROW > 1.0 + 1e-9 and c in _narrow and n in _narrow and not _along_corridor(c, n):
        return float(_NARROW)
    return 1.0
