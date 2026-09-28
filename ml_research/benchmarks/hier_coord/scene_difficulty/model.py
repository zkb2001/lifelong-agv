"""SceneDifficultyNet: map+task CNN → easy/hard (+ k policy hint)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_research.benchmarks.hier_coord.obs import N_HIER_CHANNELS
from ml_research.common.paths import CKPT

SCENE_DIFF_CKPT = CKPT / "scene_difficulty_v1.pt"
# 0=easy, 1=hard
CLASS_NAMES = ("easy", "hard")


class SceneDifficultyNet(nn.Module):
    """CNN over hier map obs → 2-class logits (easy/hard).

    Optional scalar feats (queue depth, n_tasks, …) concatenated after GAP.
    """

    FEAT_DIM = 8

    def __init__(self, in_ch: int = N_HIER_CHANNELS, feat_dim: int = FEAT_DIM, hidden: int = 96):
        super().__init__()
        self.in_ch = int(in_ch)
        self.feat_dim = int(feat_dim)
        self.decision_threshold = 0.35
        self.easy_max = 0.12
        self.enc = nn.Sequential(
            nn.Conv2d(self.in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        )
        self.head = nn.Sequential(
            nn.Linear(96 + self.feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        # auxiliary completion-ratio head (helps ranking hard vs near-easy)
        self.cr_head = nn.Sequential(
            nn.Linear(96 + self.feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def encode(self, maps: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        if maps.dim() == 3:
            maps = maps.unsqueeze(0)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        return torch.cat([self.enc(maps), feats], dim=-1)

    def forward(self, maps: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(maps, feats))

    def forward_with_cr(
        self, maps: torch.Tensor, feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(maps, feats)
        return self.head(z), self.cr_head(z).squeeze(-1)

    def predict(
        self,
        maps: Union[torch.Tensor, np.ndarray],
        feats: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[str, float, int]:
        """Return (class_name, p_hard, suggested_k_policy) where policy: 0=shrink ok, 1=full_k."""
        self.eval()
        if isinstance(maps, np.ndarray):
            maps = torch.from_numpy(maps.astype(np.float32))
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats.astype(np.float32))
        with torch.no_grad():
            logits = self.forward(maps, feats)
            prob = F.softmax(logits, dim=-1)[0]
            p_hard = float(prob[1].item())
            cls = 1 if p_hard >= 0.5 else 0
        return CLASS_NAMES[cls], p_hard, cls


def _norm_log_count(x: float, *, ref: float = 100.0) -> float:
    import math

    return float(min(1.0, math.log1p(max(0.0, x)) / math.log1p(ref)))


def burst_from_task_rows(tasks) -> tuple[int, int, float, float]:
    """Max same-pickup / same-dropoff counts and shares in a task list."""
    from collections import Counter

    rows = list(tasks or [])
    if not rows:
        return 0, 0, 0.0, 0.0
    picks: Counter = Counter()
    drops: Counter = Counter()
    for t in rows:
        pk = str(t.get("start_point") or t.get("pickup_name") or "").strip()
        dr = str(t.get("end_point") or t.get("destination") or "").strip()
        if pk:
            picks[pk] += 1
        if dr:
            drops[dr] += 1
    n = max(1, len(rows))
    pk_n = int(picks.most_common(1)[0][1]) if picks else 0
    dr_n = int(drops.most_common(1)[0][1]) if drops else 0
    return pk_n, dr_n, float(pk_n / n), float(dr_n / n)


# Near-horizon: gate decides THIS wave's planner, so feats/obs look at ~2*fleet
# tasks — not the entire lifelong backlog (which caused cold-start false-hard).
SCENE_HORIZON_MULT = 2


def scene_horizon_n(
    n_agvs: int,
    n_available: int,
    *,
    mult: int = SCENE_HORIZON_MULT,
) -> int:
    """How many pending tasks the classifier should see."""
    na = max(1, int(n_agvs))
    avail = max(0, int(n_available))
    return int(min(avail, max(na, int(mult) * na)))


def take_horizon_task_rows(tasks, n_agvs: int, *, mult: int = SCENE_HORIZON_MULT) -> list:
    """FIFO head of a flat task list, capped to near-horizon."""
    rows = list(tasks or [])
    h = scene_horizon_n(int(n_agvs), len(rows), mult=mult)
    return rows[:h]


def truncate_queues_horizon(
    queues: Optional[dict],
    n_agvs: int,
    *,
    assigned: Optional[dict] = None,
    mult: int = SCENE_HORIZON_MULT,
) -> dict:
    """Round-robin peek of queues so total ≈ horizon (train == deploy)."""
    assigned_n = len(assigned or {})
    full = sum(len(v or []) for v in (queues or {}).values())
    budget = max(0, scene_horizon_n(int(n_agvs), assigned_n + full, mult=mult) - assigned_n)
    if not queues or budget <= 0:
        return {}
    keys = list(queues.keys())
    out = {k: [] for k in keys}
    taken = 0
    idx = 0
    while taken < budget:
        progressed = False
        for k in keys:
            q = queues.get(k) or []
            if idx < len(q):
                out[k].append(q[idx])
                taken += 1
                progressed = True
                if taken >= budget:
                    break
        if not progressed:
            break
        idx += 1
    return {k: v for k, v in out.items() if v}


def burst_from_queues(
    queues: Optional[dict],
    *,
    assigned: Optional[dict] = None,
    peek: int = 64,
    n_agvs: Optional[int] = None,
) -> tuple[int, int, float, float]:
    """Burst stats from near-horizon queues (+ current assigned wave)."""
    rows = []
    for t in (assigned or {}).values():
        rows.append(t)
    if queues:
        if n_agvs is not None:
            hq = truncate_queues_horizon(queues, int(n_agvs), assigned=assigned)
            for q in hq.values():
                rows.extend(q)
        else:
            per = max(1, peek // max(1, len(queues)))
            n_peek = 0
            for q in queues.values():
                for t in (q or [])[:per]:
                    rows.append(t)
                    n_peek += 1
                    if n_peek >= peek:
                        break
                if n_peek >= peek:
                    break
    return burst_from_task_rows(rows)


def build_scene_feats(
    *,
    n_tasks: int,
    n_agvs: int,
    n_obstacles: int,
    queue_depth: int,
    same_pickup_burst: int = 0,
    same_dropoff_burst: int = 0,
    pickup_share: float = 0.0,
    dropoff_share: float = 0.0,
) -> np.ndarray:
    """Scale-invariant 8-d feat vector (train == deploy), near-horizon semantics.

    Indices
    -------
    0 horizon task count (log)
    1 fleet fraction
    2 obstacle density
    3 load ratio = horizon / n_agvs (capped; NOT full lifelong backlog)
    4/5 pickup/dropoff burst counts
    6/7 pickup/dropoff shares within horizon
    """
    qd = int(queue_depth if queue_depth > 0 else n_tasks)
    nt = max(1, int(n_tasks if n_tasks > 0 else qd))
    na = max(1, int(n_agvs))
    pk = int(same_pickup_burst)
    dr = int(same_dropoff_burst)
    pk_sh = float(pickup_share if pickup_share > 0 else pk / nt)
    dr_sh = float(dropoff_share if dropoff_share > 0 else dr / nt)
    load = min(1.0, float(qd) / float(na) / float(SCENE_HORIZON_MULT))
    return np.asarray(
        [
            _norm_log_count(float(nt), ref=float(max(16, SCENE_HORIZON_MULT * 50))),
            min(1.0, float(n_agvs) / 50.0),
            min(1.0, float(n_obstacles) / 200.0),
            float(load),
            min(1.0, float(pk) / 40.0),
            min(1.0, float(dr) / 40.0),
            min(1.0, pk_sh),
            min(1.0, dr_sh),
        ],
        dtype=np.float32,
    )


def build_scene_feats_live(
    *,
    assigned: Optional[dict],
    queues: Optional[dict],
    n_agvs: int,
    n_obstacles: int,
    dropoff_score: float = 0.0,
    pickup_score: float = 0.0,
) -> np.ndarray:
    """Features for gate inference (near-horizon; matches refreshed training data)."""
    del dropoff_score, pickup_score  # reserved for future; keep API stable
    pk_n, dr_n, pk_sh, dr_sh = burst_from_queues(
        queues, assigned=assigned, n_agvs=int(n_agvs)
    )
    full = sum(len(v) for v in (queues or {}).values()) + len(assigned or {})
    hz = scene_horizon_n(int(n_agvs), int(full))
    return build_scene_feats(
        n_tasks=hz,
        n_agvs=n_agvs,
        n_obstacles=n_obstacles,
        queue_depth=hz,
        same_pickup_burst=pk_n,
        same_dropoff_burst=dr_n,
        pickup_share=pk_sh,
        dropoff_share=dr_sh,
    )


def load_scene_difficulty_net(
    ckpt: Optional[Path] = None, device: str = "cpu"
) -> SceneDifficultyNet:
    path = Path(ckpt) if ckpt else SCENE_DIFF_CKPT
    net = SceneDifficultyNet()
    if path.exists():
        blob = torch.load(path, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        net.load_state_dict(state, strict=False)
        if isinstance(blob, dict):
            net.decision_threshold = float(blob.get("decision_threshold", 0.35))
            net.easy_max = float(blob.get("easy_max", 0.12))
    net.to(device)
    net.eval()
    return net


def save_scene_difficulty_net(
    net: SceneDifficultyNet, path: Optional[Path] = None, **meta
) -> Path:
    path = Path(path or SCENE_DIFF_CKPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "in_ch": net.in_ch,
            "feat_dim": net.feat_dim,
            "version": "v2_near_horizon_wave",
            "classes": CLASS_NAMES,
            "horizon_mult": SCENE_HORIZON_MULT,
            **meta,
        },
        path,
    )
    return path


def decide_scene_difficulty(
    maps: np.ndarray,
    feats: np.ndarray,
    net: Optional[SceneDifficultyNet],
    *,
    threshold: Optional[float] = None,
    easy_max: Optional[float] = None,
) -> Tuple[bool, str, float, str]:
    """Return (is_hard, source, p_hard, k_policy).

    Balanced single-threshold rule (no fail-closed easy_max / mid band):
    - p_hard >= threshold → hard / full_k
    - else → easy / shrink_ok
    """
    del easy_max  # kept for API compat; unused under balanced policy
    if net is None:
        return True, "scene_net_missing", 1.0, "full_k"
    thr = float(
        threshold
        if threshold is not None
        else getattr(net, "decision_threshold", 0.5)
    )
    _name, p_hard, _cls = net.predict(maps, feats)
    if p_hard >= thr:
        return True, "scene_hard", float(p_hard), "full_k"
    return False, "scene_easy", float(p_hard), "shrink_ok"

