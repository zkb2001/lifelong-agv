"""Adaptive Same-Destination Conflict Residual (ASDCR / IP-CGPN-lite).

Paper contribution (map-id free, LEAP-aligned):
  When 2–3 *loaded* AGVs share an unload cell and are near the pad, add a
  tiny residual so closer carriers rank above farther ones. Floods (≥4
  near-pad) are left to PGCP dest-gates — residuals there fought the holder
  lock and exploded SH11.

Never crosses loaded ≻ assigned ≻ idle; never rewrites StageNet kinds.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from greed_coop_mcts.paths import CKPT
from greed_coop_mcts.rule_obs import GRID, Pose
from greed_coop_mcts.stage_net import AGENT_FEAT, _agent_feat, _manh, region_id

# Pairwise feat dim used by optional net; keep local if stage_net is lean.
PAIR_FEAT = 6

Cell = Tuple[int, int]

CGPN_CKPT = CKPT / "cgpn_v1.pt"
ARCH_VERSION = "asdcr_v3"
N_REL = 3
REL_I_BEFORE, REL_J_BEFORE, REL_NONE = 0, 1, 2
DEFAULT_CONF_TAU = 0.55
DEFAULT_ALPHA = 16.0  # tiny residual; must not fight PGCP holder boost (3e5)
MIN_N = 14  # skip ≤12 fleets (gates + SH15); mid/large maps only
MIN_CONFLICT = 0.0
NEAR_PAD = 8  # only rank carriers already near the contested pad
# Rank cliques of size 2–3; floods (≥4 near-pad) stay under PGCP (SH11).
MAX_CLIQUE = 3
MIN_CLIQUE = 2


class ConflictGraphPriorityNet(nn.Module):
    def __init__(
        self,
        agent_dim: int = AGENT_FEAT,
        pair_dim: int = PAIR_FEAT,
        hidden: int = 64,
    ):
        super().__init__()
        self.agent_enc = nn.Sequential(
            nn.Linear(agent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + pair_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, N_REL),
        )

    def forward(self, agent: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        z = self.agent_enc(agent)
        b, n, h = z.shape
        zi = z.unsqueeze(2).expand(b, n, n, h)
        zj = z.unsqueeze(1).expand(b, n, n, h)
        return self.edge_mlp(torch.cat([zi, zj, pair], dim=-1))


def _phase_of(n: str, loaded: Set[str], assigned: Set[str]) -> int:
    if n in loaded:
        return 2
    if n in assigned:
        return 1
    return 0


def _would_cycle(adj: Dict[int, Set[int]], src: int, dst: int) -> bool:
    stack = [dst]
    seen = {dst}
    while stack:
        u = stack.pop()
        if u == src:
            return True
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)).astype(
        np.float32
    )


def intra_phase_residuals(
    logits: np.ndarray,
    *,
    names: Sequence[str],
    loaded: Set[str],
    assigned: Set[str],
    conf_tau: float = DEFAULT_CONF_TAU,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[Dict[str, float], float, int]:
    """Build per-phase DAGs; return residual scores in [0, alpha]."""
    n = int(logits.shape[0])
    if n == 0:
        return {}, 0.0, 0
    probs = _softmax_np(logits.reshape(-1, N_REL)).reshape(n, n, N_REL)
    phases = [_phase_of(names[i], loaded, assigned) for i in range(n)]
    residuals = {names[i]: 0.0 for i in range(n)}
    max_conf = 0.0
    n_edges = 0

    for ph in (2, 1, 0):
        idxs = [i for i in range(n) if phases[i] == ph]
        if len(idxs) < 2:
            continue
        cands: List[Tuple[float, int, int]] = []
        for a, i in enumerate(idxs):
            for j in idxs:
                if i == j:
                    continue
                p = probs[i, j]
                if p[REL_I_BEFORE] >= p[REL_J_BEFORE] and p[REL_I_BEFORE] >= p[REL_NONE]:
                    conf = float(p[REL_I_BEFORE])
                    cands.append((conf, i, j))
                    max_conf = max(max_conf, conf)
                elif p[REL_J_BEFORE] >= p[REL_NONE]:
                    conf = float(p[REL_J_BEFORE])
                    cands.append((conf, j, i))
                    max_conf = max(max_conf, conf)
        cands.sort(key=lambda t: -t[0])
        adj: Dict[int, Set[int]] = {i: set() for i in idxs}
        for conf, hi, lo in cands:
            if conf < conf_tau:
                continue
            if _would_cycle(adj, hi, lo):
                continue
            adj[hi].add(lo)
            n_edges += 1
        indeg = {i: 0 for i in idxs}
        for u in idxs:
            for v in adj[u]:
                indeg[v] += 1
        queue = [i for i in idxs if indeg[i] == 0]
        order: List[int] = []
        while queue:
            queue.sort()
            u = queue.pop(0)
            order.append(u)
            for v in sorted(adj[u]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        for i in idxs:
            if i not in order:
                order.append(i)
        m = len(order)
        for rank, i in enumerate(order):
            # rank0 highest → residual α
            score = 1.0 - float(rank) / float(max(1, m - 1)) if m > 1 else 0.5
            residuals[names[i]] = float(alpha) * float(score)
    return residuals, float(max_conf), int(n_edges)


def build_agent_matrix(
    *,
    names: Sequence[str],
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    loaded: Set[str],
    assigned: Set[str],
    blocked_streak: Dict[str, int],
    region_occ: Dict[Tuple[int, int], int],
    conflict_heat: Optional[np.ndarray],
) -> np.ndarray:
    rows = []
    for n in names:
        cur = poses[n][:2]
        final = goals.get(n, cur)
        rid = region_id(final)
        rocc = float(region_occ.get(rid, 0))
        heat = 0.0
        if (
            conflict_heat is not None
            and 0 <= cur[1] < GRID
            and 0 <= cur[0] < GRID
        ):
            heat = float(conflict_heat[cur[1], cur[0]])
        rows.append(
            _agent_feat(
                cur=cur,
                final=final,
                loaded=n in loaded,
                assigned=n in assigned,
                streak=int(blocked_streak.get(n, 0)),
                region_occ=rocc,
                heat=heat,
                v2=False,
            )
        )
    return np.stack(rows).astype(np.float32)


class CGPNRuntime:
    def __init__(
        self,
        net: Optional[ConflictGraphPriorityNet] = None,
        *,
        device: str = "cpu",
        use_net: bool = True,
        conf_tau: float = DEFAULT_CONF_TAU,
        alpha: float = DEFAULT_ALPHA,
        min_n: int = MIN_N,
        min_conflict: float = MIN_CONFLICT,
    ):
        self.device = device
        self.conf_tau = float(conf_tau)
        self.alpha = float(alpha)
        self.min_n = int(min_n)
        self.min_conflict = float(min_conflict)
        self.n_calls = 0
        self.n_fallback = 0
        self.n_apply = 0
        self.n_edges_total = 0
        self.net = net
        if use_net and self.net is None:
            self.net = load_cgpn(device=device)

    def score(
        self,
        *,
        names: List[str],
        poses: Dict[str, Pose],
        goals: Dict[str, Cell],
        paths: Dict[str, List[Cell]],
        loaded: Set[str],
        assigned: Set[str],
        blocked_streak: Dict[str, int],
        region_occ: Dict[Tuple[int, int], int],
        conflict_heat: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Same-dest loaded clique residual, or {} if no true conflict."""
        del paths, assigned, blocked_streak, region_occ, conflict_heat
        self.n_calls += 1
        if not names or len(names) < self.min_n:
            self.n_fallback += 1
            return {}
        residuals = _same_dest_residuals(
            names=names,
            poses=poses,
            goals=goals,
            loaded=loaded,
            alpha=float(self.alpha),
        )
        if not residuals:
            self.n_fallback += 1
            return {}
        self.n_apply += 1
        self.n_edges_total += len(residuals)
        return residuals


def _same_dest_residuals(
    *,
    names: Sequence[str],
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    loaded: Set[str],
    alpha: float,
) -> Dict[str, float]:
    """Closer-to-pad first for small near-pad same-unload cliques (2–3).

    Far-away carriers ignored. Floods (>MAX_CLIQUE) skipped so PGCP holder
    / pad-egress stay authoritative under dense pad contention.
    """
    by_pad: Dict[Cell, List[str]] = defaultdict(list)
    for n in names:
        if n not in loaded or n not in poses:
            continue
        g = goals.get(n)
        if g is None:
            continue
        if _manh(poses[n][:2], g) > int(NEAR_PAD):
            continue
        by_pad[g].append(n)
    out: Dict[str, float] = {}
    for pad, group in by_pad.items():
        m = len(group)
        if m < int(MIN_CLIQUE) or m > int(MAX_CLIQUE):
            continue
        ranked = sorted(
            group,
            key=lambda a: (_manh(poses[a][:2], pad), a),
        )
        for rank, a in enumerate(ranked):
            score = 1.0 - float(rank) / float(max(1, m - 1)) if m > 1 else 0.5
            out[a] = float(alpha) * float(score)
    return out


def _geo_phase_residuals(
    *,
    names: Sequence[str],
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    loaded: Set[str],
    assigned: Set[str],
    alpha: float,
) -> Dict[str, float]:
    """Legacy phase-wide residual (kept for ablation / training labels)."""
    out: Dict[str, float] = {}
    for ph in (2, 1, 0):
        idxs = [
            i
            for i, n in enumerate(names)
            if _phase_of(n, loaded, assigned) == ph
        ]
        if len(idxs) < 2:
            continue
        ranked = sorted(
            idxs,
            key=lambda i: (
                _manh(poses[names[i]][:2], goals.get(names[i], poses[names[i]][:2])),
                names[i],
            ),
        )
        m = len(ranked)
        for rank, i in enumerate(ranked):
            score = 1.0 - float(rank) / float(max(1, m - 1)) if m > 1 else 0.5
            out[names[i]] = float(alpha) * float(score)
    return out


def load_cgpn(
    ckpt: Optional[Path] = None, device: str = "cpu"
) -> Optional[ConflictGraphPriorityNet]:
    path = Path(ckpt) if ckpt else CGPN_CKPT
    net = ConflictGraphPriorityNet()
    if not path.exists():
        return None
    blob = torch.load(path, map_location=device, weights_only=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def save_cgpn(
    net: ConflictGraphPriorityNet,
    path: Optional[Path] = None,
    *,
    meta: Optional[dict] = None,
) -> Path:
    path = Path(path) if path else CGPN_CKPT
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "arch": ARCH_VERSION,
            "agent_feat": AGENT_FEAT,
            "pair_feat": PAIR_FEAT,
            "meta": meta or {},
        },
        path,
    )
    return path


def labels_from_total_order(
    order: Sequence[str],
    *,
    loaded: Set[str],
    assigned: Set[str],
) -> np.ndarray:
    """Pairwise labels only within the same phase; cross-phase = none."""
    rank = {n: i for i, n in enumerate(order)}
    names = list(order)
    n = len(names)
    lab = np.full((n, n), REL_NONE, dtype=np.int64)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                continue
            if _phase_of(ni, loaded, assigned) != _phase_of(nj, loaded, assigned):
                continue
            if rank[ni] < rank[nj]:
                lab[i, j] = REL_I_BEFORE
            else:
                lab[i, j] = REL_J_BEFORE
    return lab
