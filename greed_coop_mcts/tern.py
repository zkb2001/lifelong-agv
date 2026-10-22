"""Traffic Edge Residual Net (TERN): OnlineGGO/TrafficFlow-style Δcost.

Outputs bounded (H,W,4) residual edge costs on top of rule-align + SoftGuidance
+ TrafficFlow analytic costs. Fail-safe: missing ckpt / stall / NaN → Δ=0.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from greed_coop_mcts.paths import CKPT
from greed_coop_mcts.rule_obs import DIR_DELTA, GRID, N_RULE_CH, Pose

Cell = Tuple[int, int]

TERN_CKPT = CKPT / "tern_v1.pt"
ARCH_VERSION = "tern_v1"
# SoftGuidance adds occ (1) + edge (4) → extra 5 channels when available.
N_TERN_EXTRA = 5
N_TERN_CH = N_RULE_CH + N_TERN_EXTRA
DELTA_MAX = 0.75
EDGE_LO = 1.0
EDGE_HI = 3.5


class TrafficEdgeResidualNet(nn.Module):
    """Small CNN: state (+ soft traffic) → (B,4,H,W) residual logits."""

    def __init__(
        self,
        channels: int = N_TERN_CH,
        width: int = 48,
        n_dirs: int = 4,
        delta_max: float = DELTA_MAX,
    ):
        super().__init__()
        self.delta_max = float(delta_max)
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, n_dirs, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,C,H,W) → residual (B,4,H,W) in [-delta_max, delta_max]."""
        return self.delta_max * torch.tanh(self.net(x))


def pack_tern_state(
    rule_state: np.ndarray,
    *,
    occ: Optional[np.ndarray] = None,
    edge: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Concatenate rule_obs (C,H,W) with SoftGuidance occ/edge → (C+5,H,W)."""
    c, h, w = rule_state.shape
    assert h == GRID and w == GRID
    extra = np.zeros((N_TERN_EXTRA, h, w), dtype=np.float32)
    if occ is not None:
        extra[0] = np.asarray(occ, dtype=np.float32)
    if edge is not None:
        e = np.asarray(edge, dtype=np.float32)
        if e.shape == (h, w, 4):
            for di in range(4):
                extra[1 + di] = e[:, :, di]
        elif e.shape == (4, h, w):
            extra[1:5] = e
    return np.concatenate([rule_state.astype(np.float32), extra], axis=0)


def traffic_flow_edge_bonus(
    cur: Cell,
    nxt: Cell,
    *,
    flow_fwd: Optional[np.ndarray],
    flow_rev: Optional[np.ndarray],
    vertex_load: Optional[np.ndarray],
    lambda_tf: float = 0.35,
) -> float:
    """TrafficFlow-style contraflow × vertex congestion bonus (≥0)."""
    if flow_fwd is None or lambda_tf <= 0:
        return 0.0
    x, y = cur
    nx, ny = nxt
    if not (0 <= y < GRID and 0 <= x < GRID):
        return 0.0
    di = -1
    for i, (dx, dy) in enumerate(DIR_DELTA):
        if (x + dx, y + dy) == (nx, ny):
            di = i
            break
    if di < 0:
        return 0.0
    fwd = float(flow_fwd[y, x, di])
    rev = 0.0
    if flow_rev is not None and 0 <= ny < GRID and 0 <= nx < GRID:
        rev = float(flow_rev[ny, nx, (di + 2) % 4])
    # c_e = f_fwd * f_rev (contraflow)
    c_e = fwd * rev
    p_v = 0.0
    if vertex_load is not None and 0 <= ny < GRID and 0 <= nx < GRID:
        n = float(vertex_load[ny, nx])
        # p_v ≈ ceil((n-1)/2) style soft share
        p_v = max(0.0, 0.5 * max(0.0, n - 1.0))
    return float(lambda_tf) * (c_e + 0.25 * p_v)


def accumulate_path_flows(
    paths: Dict[str, Sequence[Cell]],
    *,
    horizon: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flow_fwd (H,W,4), flow_rev same, vertex_load (H,W) from paths."""
    fwd = np.zeros((GRID, GRID, 4), dtype=np.float32)
    vload = np.zeros((GRID, GRID), dtype=np.float32)
    for path in paths.values():
        cells = list(path)[:horizon]
        for c in cells:
            if 0 <= c[1] < GRID and 0 <= c[0] < GRID:
                vload[c[1], c[0]] += 1.0
        for a, b in zip(cells, cells[1:]):
            x, y = a
            if not (0 <= y < GRID and 0 <= x < GRID):
                continue
            for di, (dx, dy) in enumerate(DIR_DELTA):
                if (x + dx, y + dy) == b:
                    fwd[y, x, di] += 1.0
                    break
    # Normalize softly
    if fwd.max() > 0:
        fwd = fwd / float(fwd.max())
    if vload.max() > 0:
        vload = vload / float(vload.max())
    return fwd, fwd, vload


def traffic_flow_cost_target(
    *,
    walk: np.ndarray,
    flow_fwd: np.ndarray,
    vertex_load: np.ndarray,
    heat: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Supervised (H,W,4) target costs from TrafficFlow analytic."""
    costs = np.full((GRID, GRID, 4), 1.0, dtype=np.float32)
    heat = heat if heat is not None else np.zeros((GRID, GRID), dtype=np.float32)
    for y in range(GRID):
        for x in range(GRID):
            if walk[y, x] < 0.5:
                costs[y, x, :] = 3.0
                continue
            for di, (dx, dy) in enumerate(DIR_DELTA):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < GRID and 0 <= ny < GRID) or walk[ny, nx] < 0.5:
                    costs[y, x, di] = 3.0
                    continue
                fwd = float(flow_fwd[y, x, di])
                rev = float(flow_fwd[ny, nx, (di + 2) % 4])
                c_e = fwd * rev
                p_v = 0.5 * max(0.0, float(vertex_load[ny, nx]) * 8.0 - 1.0)
                base = 1.0 + 1.2 * c_e + 0.35 * max(0.0, p_v) + 0.4 * float(heat[ny, nx])
                costs[y, x, di] = float(np.clip(base, EDGE_LO, EDGE_HI))
    return costs


class TERNRuntime:
    """Predict Δcost field and query directed edge residual."""

    def __init__(
        self,
        net: Optional[TrafficEdgeResidualNet] = None,
        *,
        device: str = "cpu",
        use_net: bool = True,
        delta_max: float = DELTA_MAX,
    ):
        self.device = device
        self.delta_max = float(delta_max)
        self.delta: Optional[np.ndarray] = None  # (H,W,4)
        self.flow_fwd: Optional[np.ndarray] = None
        self.flow_rev: Optional[np.ndarray] = None
        self.vertex_load: Optional[np.ndarray] = None
        self.n_infer = 0
        self.n_fallback = 0
        self.net = net
        if use_net and self.net is None:
            self.net = load_tern(device=device)

    def update_flows(self, paths: Dict[str, Sequence[Cell]], *, horizon: int = 16) -> None:
        fwd, rev, vload = accumulate_path_flows(paths, horizon=horizon)
        self.flow_fwd = fwd
        self.flow_rev = rev
        self.vertex_load = vload

    def infer_delta(
        self,
        rule_state: np.ndarray,
        *,
        occ: Optional[np.ndarray] = None,
        edge: Optional[np.ndarray] = None,
        stall: int = 0,
    ) -> np.ndarray:
        self.n_infer += 1
        if self.net is None or stall >= 40:
            self.n_fallback += 1
            self.delta = np.zeros((GRID, GRID, 4), dtype=np.float32)
            return self.delta
        state = pack_tern_state(rule_state, occ=occ, edge=edge)
        self.net.eval()
        with torch.no_grad():
            x = torch.as_tensor(state[None, ...], dtype=torch.float32, device=self.device)
            d = self.net(x)[0].cpu().numpy()  # (4,H,W)
        if not np.isfinite(d).all():
            self.n_fallback += 1
            self.delta = np.zeros((GRID, GRID, 4), dtype=np.float32)
            return self.delta
        self.delta = np.transpose(d, (1, 2, 0)).astype(np.float32)
        return self.delta

    def residual(self, cur: Cell, nxt: Cell) -> float:
        if self.delta is None:
            return 0.0
        x, y = cur
        if not (0 <= y < GRID and 0 <= x < GRID):
            return 0.0
        for di, (dx, dy) in enumerate(DIR_DELTA):
            if (x + dx, y + dy) == nxt:
                return float(np.clip(self.delta[y, x, di], -self.delta_max, self.delta_max))
        return 0.0

    def traffic_bonus(self, cur: Cell, nxt: Cell, *, lambda_tf: float = 0.35) -> float:
        return traffic_flow_edge_bonus(
            cur,
            nxt,
            flow_fwd=self.flow_fwd,
            flow_rev=self.flow_rev,
            vertex_load=self.vertex_load,
            lambda_tf=lambda_tf,
        )


def load_tern(
    ckpt: Optional[Path] = None, device: str = "cpu"
) -> Optional[TrafficEdgeResidualNet]:
    path = Path(ckpt) if ckpt else TERN_CKPT
    net = TrafficEdgeResidualNet()
    if not path.exists():
        return None
    blob = torch.load(path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def save_tern(
    net: TrafficEdgeResidualNet,
    path: Optional[Path] = None,
    *,
    meta: Optional[dict] = None,
) -> Path:
    path = Path(path) if path else TERN_CKPT
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": net.state_dict(),
        "arch": ARCH_VERSION,
        "channels": N_TERN_CH,
        "meta": meta or {},
    }
    torch.save(blob, path)
    return path
