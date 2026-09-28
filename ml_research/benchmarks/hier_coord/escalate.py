"""EscalateNet — upper-layer AI: whether to enable hard wave mode this wave."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ml_research.common.paths import CKPT

ESCALATE_CKPT = CKPT / "escalate_net_v1.pt"
ESCALATE_FEAT_DIM = 10
ESCALATE_FEAT_NAMES = (
    "map_hardness",
    "dropoff_score",
    "pickup_score",
    "fail_h",
    "n_assigned_n",
    "queue_depth_n",
    "top_share",
    "assigned_top_n_n",
    "was_escalated",
    "calm_streak_n",
)


@dataclass
class EscalateContext:
    map_hardness: float
    dropoff_score: float
    pickup_score: float = 0.0
    recent_joint_fail: int = 0
    n_assigned: int = 0
    queue_depth: int = 0
    top_share: float = 0.0
    assigned_top_n: int = 0
    was_escalated: bool = False
    calm_streak: int = 0


def build_escalate_features(ctx: EscalateContext) -> np.ndarray:
    fail_h = min(1.0, 0.18 * float(max(0, ctx.recent_joint_fail)))
    feat = np.array(
        [
            float(ctx.map_hardness),
            float(ctx.dropoff_score),
            float(ctx.pickup_score),
            fail_h,
            min(1.0, float(ctx.n_assigned) / 8.0),
            min(1.0, float(ctx.queue_depth) / 40.0),
            float(ctx.top_share),
            min(1.0, float(ctx.assigned_top_n) / 6.0),
            1.0 if ctx.was_escalated else 0.0,
            min(1.0, float(ctx.calm_streak) / 4.0),
        ],
        dtype=np.float32,
    )
    assert feat.shape[0] == ESCALATE_FEAT_DIM
    return feat


class EscalateNet(nn.Module):
    """MLP: features → logit (P(hard-wave))."""

    def __init__(self, feat_dim: int = ESCALATE_FEAT_DIM, hidden: int = 32):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def prob(self, x: Union[torch.Tensor, np.ndarray]) -> float:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(self.forward(x)).item())


def load_escalate_net(
    ckpt: Optional[Path] = None,
    device: str = "cpu",
) -> EscalateNet:
    path = Path(ckpt) if ckpt else ESCALATE_CKPT
    net = EscalateNet()
    if path.exists():
        blob = torch.load(path, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def save_escalate_net(net: EscalateNet, path: Optional[Path] = None, **meta) -> Path:
    path = Path(path or ESCALATE_CKPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "feat_dim": net.feat_dim,
            "version": "v1_escalate",
            **meta,
        },
        path,
    )
    return path


def rule_escalate_prob(ctx: EscalateContext, *, threshold: float = 0.08) -> float:
    """Soft teacher target in [0,1] for BC."""
    runtime = max(float(ctx.dropoff_score), float(ctx.pickup_score))
    fail_h = min(1.0, 0.18 * float(max(0, ctx.recent_joint_fail)))
    score = max(float(ctx.map_hardness), runtime, fail_h)
    if ctx.assigned_top_n >= 3:
        score = max(score, 0.7)
    if ctx.was_escalated and score < threshold and ctx.calm_streak < 2:
        score = max(score, 0.55)  # hysteresis soft label
    return float(min(1.0, score / max(threshold, 1e-3) * 0.5))


def decide_escalate_ai(
    ctx: EscalateContext,
    net: Optional[EscalateNet],
    *,
    threshold: float = 0.48,
    rule_threshold: float = 0.08,
    dropoff_escalate_score: float = 0.45,
) -> tuple[bool, str, float]:
    """Return (escalate, source, confidence).

    Blends rule hard triggers with EscalateNet when available.
    """
    # Hard safety triggers always on
    if ctx.map_hardness >= rule_threshold:
        return True, "rule_map", 1.0
    if max(ctx.dropoff_score, ctx.pickup_score) >= dropoff_escalate_score:
        return True, "rule_hotspot", 1.0
    if ctx.recent_joint_fail > 0:
        return True, "rule_fail", 1.0
    if ctx.assigned_top_n >= 3:
        return True, "rule_assigned_burst", 1.0

    if net is None:
        # Hysteresis without net
        if ctx.was_escalated and ctx.calm_streak < 2:
            return True, "rule_hysteresis", 0.6
        return False, "rule_easy", 0.0

    p = net.prob(build_escalate_features(ctx))
    if ctx.was_escalated and ctx.calm_streak < 2 and p >= threshold * 0.7:
        return True, "net_hysteresis", p
    if p >= threshold:
        return True, "net", p
    if ctx.was_escalated and ctx.calm_streak < 2:
        return True, "rule_hysteresis", p
    return False, "net_easy", p
