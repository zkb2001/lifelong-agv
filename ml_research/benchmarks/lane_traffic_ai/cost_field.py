"""Convert network traffic rules into edge step costs (signal + direction + zones)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .config import (
    BLOCK,
    DIR_DELTA,
    GRID,
    W_AGAINST_DIR,
    W_ARTERY_REBATE,
    W_CONSTRUCTION_BLOCK,
    W_CONSTRUCTION_SOFT,
    W_CROSS_ZONE,
    W_NOWAIT_WAIT,
    W_RED_PHASE,
    W_SIDE_EXTRA,
    W_URGENCY_SIDE,
    W_WAIT_BASE,
    W_YIELD_WAIT,
)
from .features import zone_id

Cell = Tuple[int, int]


def rules_to_numpy(out: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Detach first batch item to numpy."""

    def _first(t: torch.Tensor) -> np.ndarray:
        if t.dim() >= 3 and t.shape[0] <= 8:
            return t[0].detach().cpu().numpy()
        return t.detach().cpu().numpy()

    dir_idx = out["dir"][0].detach().cpu().numpy() if out["dir"].dim() == 3 else out["dir"].detach().cpu().numpy()
    phase = out["phase"][0].detach().cpu().numpy() if out["phase"].dim() == 3 else out["phase"].detach().cpu().numpy()
    zone = out["zone"][0].detach().cpu().numpy() if out["zone"].dim() == 3 else out["zone"].detach().cpu().numpy()
    rules: Dict[str, np.ndarray] = {
        "dir": dir_idx.astype(np.int64),
        "phase": phase.astype(np.int64),
        "zone": zone.astype(np.int64),
    }
    if "yield" in out:
        rules["yield"] = _first(out["yield"]).astype(np.float32)
    if "artery" in out:
        rules["artery"] = _first(out["artery"]).astype(np.float32)
    if "nowait" in out:
        rules["nowait"] = _first(out["nowait"]).astype(np.float32)
    if "construction" in out:
        # (B,4,H,W) → (H,W,4) for edge indexing
        c = out["construction"]
        if c.dim() == 4:
            rules["construction"] = c[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        else:
            rules["construction"] = c.detach().cpu().numpy().astype(np.float32)
    if "bidir" in out:
        rules["bidir"] = _first(out["bidir"]).astype(np.float32)
    return rules


def wait_cost(
    x: int,
    y: int,
    rules: Dict[str, np.ndarray],
    *,
    grid: int = GRID,
    w_base: float = W_WAIT_BASE,
    w_yield: float = W_YIELD_WAIT,
    w_nowait: float = W_NOWAIT_WAIT,
) -> float:
    """Relative wait penalty at cell (x,y). Lower on yield/parking; high on nowait."""
    if not (0 <= x < grid and 0 <= y < grid):
        return w_base
    nowait = rules.get("nowait")
    if nowait is not None and float(nowait[y, x]) > 0.5:
        return float(w_nowait)
    yld = rules.get("yield")
    if yld is not None and float(yld[y, x]) > 0.5:
        return float(w_yield)
    return float(w_base)


def step_cost(
    x: int,
    y: int,
    nx: int,
    ny: int,
    rules: Dict[str, np.ndarray],
    *,
    w_against: float = W_AGAINST_DIR,
    w_red: float = W_RED_PHASE,
    w_zone: float = W_CROSS_ZONE,
    w_artery_rebate: float = W_ARTERY_REBATE,
    w_side: float = W_SIDE_EXTRA,
    w_urgency_side: float = W_URGENCY_SIDE,
    block: int = BLOCK,
    grid: int = GRID,
    soft_against: bool = True,
) -> float:
    """Cost of moving (x,y) → (nx,ny) under traffic rules.

    Against-dir is **soft** (finite) unless ``soft_against=False`` or construction BLOCK.
    """
    cost = 1.0
    dx, dy = nx - x, ny - y
    move_dir = None
    for i, (adx, ady) in enumerate(DIR_DELTA):
        if (dx, dy) == (adx, ady):
            move_dir = i
            break
    if move_dir is None:
        return 1e6

    # Temporary closed / construction edges
    closed = rules.get("construction")
    if closed is not None and 0 <= y < grid and 0 <= x < grid:
        if closed.ndim == 3 and closed.shape[-1] >= 4:
            flag = float(closed[y, x, move_dir])
        elif closed.ndim == 2:
            flag = float(closed[y, x])
        else:
            flag = 0.0
        if flag >= 0.99:
            return float(W_CONSTRUCTION_BLOCK)
        if flag > 0.05:
            cost += W_CONSTRUCTION_SOFT * flag

    # bidir≈1 → free two-way (near bare unit step, keep construction only)
    bidir = rules.get("bidir")
    bidir_free = False
    if bidir is not None and 0 <= y < grid and 0 <= x < grid:
        bidir_free = float(bidir[y, x]) >= 0.6

    if not bidir_free:
        pref = int(rules["dir"][y, x]) if 0 <= y < grid and 0 <= x < grid else -1
        if pref >= 0 and move_dir != pref:
            opp = (pref + 2) % 4
            against = w_against * (2.0 if move_dir == opp else 1.0)
            if not soft_against:
                return float(W_CONSTRUCTION_BLOCK)
            cost += against

        jy, jx = y // block, x // block
        phase = rules["phase"]
        if 0 <= jy < phase.shape[0] and 0 <= jx < phase.shape[1]:
            ph = int(phase[jy, jx])
            if ph == 0 and move_dir in (0, 2):
                cost += w_red
            if ph == 1 and move_dir in (1, 3):
                cost += w_red

        z0 = zone_id(x, y)
        z1 = zone_id(nx, ny)
        zmap = rules.get("zone")
        if zmap is not None and 0 <= y < grid and 0 <= x < grid:
            pref_z = int(zmap[y, x])
            if z1 != pref_z and z0 == pref_z:
                cost += 0.5 * w_zone
        if z0 != z1:
            cost += 0.25 * w_zone

        artery = rules.get("artery")
        a = 0.5
        if artery is not None and 0 <= y < grid and 0 <= x < grid:
            a = float(np.clip(artery[y, x], 0.0, 1.0))
        urgency = float(rules.get("urgency", 0.0) or 0.0)
        cost -= w_artery_rebate * a
        cost += w_side * (1.0 - a) * (1.0 + w_urgency_side * urgency)

        if artery is not None and 0 <= ny < grid and 0 <= nx < grid:
            a1 = float(np.clip(artery[ny, nx], 0.0, 1.0))
            if a >= 0.7 and a1 < 0.4:
                cost += 0.25 * (1.0 + urgency)

    return float(cost)


def build_cost_tensor(rules: Dict[str, np.ndarray], walkable: np.ndarray) -> np.ndarray:
    """(H,W,4) step cost to each neighbor; inf if not walkable."""
    H, W = walkable.shape
    costs = np.full((H, W, 4), np.inf, dtype=np.float64)
    for y in range(H):
        for x in range(W):
            if walkable[y, x] < 0.5:
                continue
            for d, (dx, dy) in enumerate(DIR_DELTA):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                if walkable[ny, nx] < 0.5:
                    continue
                costs[y, x, d] = step_cost(x, y, nx, ny, rules)
    return costs


def apply_construction_disturbance(
    rules: Dict[str, np.ndarray],
    closed: np.ndarray,
    *,
    merge: bool = True,
) -> Dict[str, np.ndarray]:
    """Write / merge a sparse (H,W,4) construction mask into rules (ablation hook)."""
    out = dict(rules)
    c = np.asarray(closed, dtype=np.float32)
    if merge and "construction" in out and out["construction"] is not None:
        out["construction"] = np.maximum(out["construction"].astype(np.float32), c)
    else:
        out["construction"] = c
    return out
