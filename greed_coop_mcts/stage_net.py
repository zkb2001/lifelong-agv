"""Stage selection + priority AI for RRSC-v2.

For each AGV, score stage candidates (FINAL / APPROACH / CLEAR / HOLD) and
emit a scalar priority. Road-rights paths fill candidate geometry; the net
only chooses among them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from greed_coop_mcts.paths import CKPT
from greed_coop_mcts.rule_obs import DIR_DELTA, GRID, Pose
from greed_coop_mcts.stage_protocol import StageRole

Cell = Tuple[int, int]

STAGE_CKPT = CKPT / "stage_selector_v1.pt"
N_CAND = 4  # FINAL, APPROACH_A, APPROACH_B, CLEAR
# Candidate kinds
KIND_FINAL, KIND_APP_A, KIND_APP_B, KIND_CLEAR = 0, 1, 2, 3
KIND_HOLD = 4  # used only when forced; maps to CLEAR slot if needed

AGENT_FEAT = 8
CAND_FEAT = 10


@dataclass
class StageChoice:
    agent: str
    kind: int
    cell: Cell
    role: StageRole
    priority: float
    logits: Optional[np.ndarray] = None


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def region_id(cell: Cell, *, tile: int = 5) -> Tuple[int, int]:
    """Coarse geometric region (not narrow-corridor specific)."""
    x, y = int(cell[0]), int(cell[1])
    return (max(0, x) // tile, max(0, y) // tile)


def build_candidates(
    *,
    cur: Cell,
    final: Cell,
    path: Sequence[Cell],
    blocked: Set[Cell],
    stay_penalty: Optional[np.ndarray],
    occupied: Set[Cell],
) -> List[Tuple[int, Cell]]:
    """Return up to N_CAND (kind, cell) pairs: FINAL, mid-path, CLEAR."""
    out: List[Tuple[int, Cell]] = []
    out.append((KIND_FINAL, final))

    full = list(path)
    if not full or full[-1] != final:
        full = full + [final]
    if len(full) >= 3:
        i1 = max(0, len(full) // 3 - 1)
        i2 = max(0, (2 * len(full)) // 3 - 1)
        out.append((KIND_APP_A, full[i1]))
        out.append((KIND_APP_B, full[i2]))
    elif len(full) >= 1:
        out.append((KIND_APP_A, full[0]))
        out.append((KIND_APP_B, full[-1]))
    else:
        out.append((KIND_APP_A, final))
        out.append((KIND_APP_B, final))

    # CLEAR: neighbor / ring with low stay penalty, not occupied, not final.
    best = cur
    best_score = 1e18
    for rad in (1, 2, 3):
        for dx in range(-rad, rad + 1):
            for dy in range(-rad, rad + 1):
                if abs(dx) + abs(dy) != rad:
                    continue
                cand = (cur[0] + dx, cur[1] + dy)
                if cand in blocked or cand in occupied or cand == final:
                    continue
                if not (1 <= cand[0] < GRID and 1 <= cand[1] < GRID):
                    continue
                stay = (
                    float(stay_penalty[cand[1], cand[0]])
                    if stay_penalty is not None
                    else 0.0
                )
                score = stay + 0.05 * _manh(cand, final)
                if score < best_score:
                    best_score = score
                    best = cand
        if best != cur:
            break
    out.append((KIND_CLEAR, best))

    # Pad / truncate to N_CAND unique-ish slots
    while len(out) < N_CAND:
        out.append((KIND_FINAL, final))
    return out[:N_CAND]


def _agent_feat(
    *,
    cur: Cell,
    final: Cell,
    loaded: bool,
    assigned: bool,
    streak: int,
    region_occ: float,
    heat: float,
) -> np.ndarray:
    return np.asarray(
        [
            cur[0] / float(GRID),
            cur[1] / float(GRID),
            final[0] / float(GRID),
            final[1] / float(GRID),
            _manh(cur, final) / float(GRID * 2),
            1.0 if loaded else (0.5 if assigned else 0.0),
            min(1.0, float(streak) / 40.0),
            min(1.0, float(region_occ) / 6.0) + 0.25 * heat,
        ],
        dtype=np.float32,
    )


def _cand_feat(
    *,
    cur: Cell,
    final: Cell,
    kind: int,
    cell: Cell,
    stay_penalty: Optional[np.ndarray],
    occupied: Set[Cell],
    edge_cost: float,
    region_occ: float,
) -> np.ndarray:
    stay = (
        float(stay_penalty[cell[1], cell[0]])
        if stay_penalty is not None and 0 <= cell[1] < GRID and 0 <= cell[0] < GRID
        else 0.0
    )
    return np.asarray(
        [
            cell[0] / float(GRID),
            cell[1] / float(GRID),
            _manh(cur, cell) / float(GRID * 2),
            _manh(cell, final) / float(GRID * 2),
            float(kind) / 4.0,
            1.0 if kind == KIND_FINAL else 0.0,
            1.0 if kind == KIND_CLEAR else 0.0,
            min(2.0, stay),
            1.0 if cell in occupied else 0.0,
            min(1.0, edge_cost / 3.0) + 0.15 * min(1.0, region_occ / 6.0),
        ],
        dtype=np.float32,
    )


class StageSelectorNet(nn.Module):
    """Score K candidates + scalar priority from agent context."""

    def __init__(
        self,
        agent_dim: int = AGENT_FEAT,
        cand_dim: int = CAND_FEAT,
        hidden: int = 64,
        n_cand: int = N_CAND,
    ):
        super().__init__()
        self.n_cand = int(n_cand)
        self.agent_enc = nn.Sequential(
            nn.Linear(agent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.cand_enc = nn.Sequential(
            nn.Linear(cand_dim + hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.pri_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self, agent: torch.Tensor, cands: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """agent (B,A), cands (B,K,C), mask (B,K) → logits (B,K), pri (B,)."""
        h = self.agent_enc(agent)  # B,H
        b, k, _ = cands.shape
        h_exp = h.unsqueeze(1).expand(-1, k, -1)
        logits = self.cand_enc(torch.cat([cands, h_exp], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(mask < 0.5, -1e9)
        pri = self.pri_head(h).squeeze(-1)
        return logits, pri


def save_stage_net(net: StageSelectorNet, path: Path | None = None, **meta) -> Path:
    path = Path(path or STAGE_CKPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "meta": meta}, path)
    return path


def load_stage_net(
    path: Path | None = None, *, device: str = "cpu"
) -> Optional[StageSelectorNet]:
    path = Path(path or STAGE_CKPT)
    if not path.is_file():
        return None
    blob = torch.load(path, map_location=device, weights_only=False)
    net = StageSelectorNet().to(device)
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net


def teacher_kind(
    *,
    cur: Cell,
    final: Cell,
    kinds: Sequence[int],
    cells: Sequence[Cell],
    region_occ: float,
    streak: int,
    heat: float,
    loaded: bool,
) -> int:
    """Heuristic IL label: FINAL when clear; APPROACH/CLEAR under congestion."""
    d = _manh(cur, final)
    # Near goal or recovering from stall → always FINAL (completion-critical).
    if d <= 5 or streak >= 18:
        for i, k in enumerate(kinds):
            if k == KIND_FINAL:
                return i
        return 0
    # Loaded agents never CLEAR in teacher labels (completion-critical).
    congested = region_occ >= 4.0 or heat >= 0.7
    if not congested or loaded:
        for i, k in enumerate(kinds):
            if k == KIND_FINAL:
                return i
        return 0
    if streak >= 12:
        for i, k in enumerate(kinds):
            if k == KIND_CLEAR:
                return i
    for prefer in (KIND_APP_B, KIND_APP_A):
        for i, k in enumerate(kinds):
            if k == prefer:
                return i
    return 0


def teacher_priority(
    *,
    loaded: bool,
    assigned: bool,
    streak: int,
    dist_final: int,
    heat: float,
) -> float:
    base = 1e6 if loaded else (1e5 if assigned else 1.0)
    # Congested / near-goal boost (RL-RH-PP lite)
    base += 500.0 * heat + 50.0 * max(0, 20 - dist_final) + 20.0 * min(streak, 30)
    if not loaded and not assigned:
        base = float(wait := 1.0 + streak)  # noqa: F841 — idle low
        base = 10.0 + float(streak) + 100.0 * heat
    return float(base)


class StageSelector:
    """Runtime wrapper: build candidates, score with net or teacher."""

    def __init__(
        self,
        net: Optional[StageSelectorNet] = None,
        *,
        device: str = "cpu",
        use_net: bool = True,
        use_cgpn: bool = False,
    ):
        self.device = device
        self.net = net
        if use_net and self.net is None:
            self.net = load_stage_net(device=device)
        self.cgpn = None
        self.n_cgpn_priority = 0
        if use_cgpn:
            try:
                from greed_coop_mcts.cgpn import CGPNRuntime

                # SDCR: geometry same-dest residual (no fragile IL net required).
                self.cgpn = CGPNRuntime(device=device, use_net=False)
            except Exception:  # noqa: BLE001
                self.cgpn = None
        self.n_choices = 0
        self.n_final = 0
        self.n_clear = 0
        self.n_approach = 0

    def choose_batch(
        self,
        *,
        names: List[str],
        poses: Dict[str, Pose],
        goals: Dict[str, Cell],
        paths: Dict[str, List[Cell]],
        blocked: Set[Cell],
        loaded: Set[str],
        assigned: Set[str],
        blocked_streak: Dict[str, int],
        stay_penalty: Optional[np.ndarray],
        conflict_heat: Optional[np.ndarray],
        region_occ: Dict[Tuple[int, int], int],
        edge_cost_fn=None,
    ) -> Dict[str, StageChoice]:
        occupied = {poses[n][:2] for n in names}
        agent_rows = []
        cand_rows = []
        masks = []
        meta: List[Tuple[str, List[Tuple[int, Cell]]]] = []

        for n in names:
            cur = poses[n][:2]
            final = goals.get(n, cur)
            path = paths.get(n) or []
            cands = build_candidates(
                cur=cur,
                final=final,
                path=path,
                blocked=blocked,
                stay_penalty=stay_penalty,
                occupied=occupied - {cur},
            )
            rid = region_id(final)
            rocc = float(region_occ.get(rid, 0))
            heat = 0.0
            if conflict_heat is not None and 0 <= cur[1] < GRID and 0 <= cur[0] < GRID:
                heat = float(conflict_heat[cur[1], cur[0]])
            af = _agent_feat(
                cur=cur,
                final=final,
                loaded=n in loaded,
                assigned=n in assigned,
                streak=int(blocked_streak.get(n, 0)),
                region_occ=rocc,
                heat=heat,
            )
            cfeats = []
            mask = []
            for kind, cell in cands:
                ec = 1.0
                if edge_cost_fn is not None and cell != cur:
                    # approximate from cur toward cell first step cost
                    step = cell
                    for dx, dy in DIR_DELTA:
                        nxt = (cur[0] + dx, cur[1] + dy)
                        if _manh(nxt, cell) < _manh(cur, cell):
                            step = nxt
                            break
                    try:
                        ec = float(edge_cost_fn(cur, step))
                    except Exception:  # noqa: BLE001
                        ec = 1.0
                cfeats.append(
                    _cand_feat(
                        cur=cur,
                        final=final,
                        kind=kind,
                        cell=cell,
                        stay_penalty=stay_penalty,
                        occupied=occupied - {cur},
                        edge_cost=ec,
                        region_occ=rocc,
                    )
                )
                mask.append(1.0)
            while len(cfeats) < N_CAND:
                cfeats.append(np.zeros(CAND_FEAT, dtype=np.float32))
                mask.append(0.0)
            agent_rows.append(af)
            cand_rows.append(np.stack(cfeats[:N_CAND]))
            masks.append(np.asarray(mask[:N_CAND], dtype=np.float32))
            meta.append((n, cands[:N_CAND]))

        choices: Dict[str, StageChoice] = {}
        if self.net is not None:
            with torch.no_grad():
                a = torch.as_tensor(
                    np.stack(agent_rows), dtype=torch.float32, device=self.device
                )
                c = torch.as_tensor(
                    np.stack(cand_rows), dtype=torch.float32, device=self.device
                )
                m = torch.as_tensor(
                    np.stack(masks), dtype=torch.float32, device=self.device
                )
                logits, pri = self.net(a, c, m)
                # Argmax is deterministic under PYTHONHASHSEED=0 + single-thread
                # CPU; teacher near-tie override previously regressed SH15 (+17).
                idx = logits.argmax(dim=-1).cpu().numpy()
                pri_np = pri.cpu().numpy()
                log_np = logits.cpu().numpy()
            for i, (n, cands) in enumerate(meta):
                j = int(idx[i])
                j = max(0, min(j, len(cands) - 1))
                kind, cell = cands[j]
                role = self._role_for(n, kind, loaded, assigned)
                # Blend net priority with role floor so carriers stay on top.
                p = float(pri_np[i])
                if n in loaded:
                    p = 1e6 + p
                elif n in assigned:
                    p = 1e5 + p
                else:
                    p = 10.0 + p
                choices[n] = StageChoice(
                    agent=n,
                    kind=kind,
                    cell=cell,
                    role=role,
                    priority=p,
                    logits=log_np[i],
                )
        else:
            for i, (n, cands) in enumerate(meta):
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
                kinds = [k for k, _ in cands]
                cells = [c for _, c in cands]
                j = teacher_kind(
                    cur=cur,
                    final=final,
                    kinds=kinds,
                    cells=cells,
                    region_occ=rocc,
                    streak=int(blocked_streak.get(n, 0)),
                    heat=heat,
                    loaded=n in loaded,
                )
                kind, cell = cands[j]
                role = self._role_for(n, kind, loaded, assigned)
                p = teacher_priority(
                    loaded=n in loaded,
                    assigned=n in assigned,
                    streak=int(blocked_streak.get(n, 0)),
                    dist_final=_manh(cur, final),
                    heat=heat,
                )
                choices[n] = StageChoice(
                    agent=n, kind=kind, cell=cell, role=role, priority=p
                )

        # IP-CGPN: additive intra-phase residual only (never replaces StageNet).
        if self.cgpn is not None and names:
            try:
                pri_map = self.cgpn.score(
                    names=names,
                    poses=poses,
                    goals=goals,
                    paths=paths,
                    loaded=loaded,
                    assigned=assigned,
                    blocked_streak=blocked_streak,
                    region_occ=region_occ,
                    conflict_heat=conflict_heat,
                )
                if pri_map:
                    for n, ch in choices.items():
                        if n in pri_map:
                            ch.priority = float(ch.priority) + float(pri_map[n])
                            self.n_cgpn_priority += 1
            except Exception:  # noqa: BLE001
                pass

        for ch in choices.values():
            self.n_choices += 1
            if ch.kind == KIND_FINAL:
                self.n_final += 1
            elif ch.kind == KIND_CLEAR:
                self.n_clear += 1
            else:
                self.n_approach += 1
        return choices

    @staticmethod
    def _role_for(
        n: str, kind: int, loaded: Set[str], assigned: Set[str]
    ) -> StageRole:
        if kind == KIND_CLEAR:
            return StageRole.CLEAR if n in loaded or n in assigned else StageRole.YIELD
        if kind == KIND_HOLD:
            return StageRole.HOLD
        if n in loaded:
            return StageRole.TO_DROP
        if n in assigned:
            return StageRole.TO_PICK
        return StageRole.IDLE
