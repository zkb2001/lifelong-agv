"""EscalateNetMap — CNN over map+AGV+task channels (+ scalar context)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ml_research.common.paths import CKPT

from .escalate import (
    ESCALATE_FEAT_DIM,
    EscalateContext,
    EscalateNet,
    build_escalate_features,
    rule_escalate_prob,
)
from .obs import N_HIER_CHANNELS

ESCALATE_MAP_CKPT = CKPT / "escalate_net_map_v1.pt"


class EscalateNetMap(nn.Module):
    """Map CNN + scalar MLP → escalate logit."""

    def __init__(
        self,
        in_ch: int = N_HIER_CHANNELS,
        feat_dim: int = ESCALATE_FEAT_DIM,
        hidden: int = 64,
        map_dim: int = 64,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.feat_dim = int(feat_dim)
        self.enc = nn.Sequential(
            nn.Conv2d(self.in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, map_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(map_dim + self.feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, maps: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """maps: (B,C,H,W); feats: (B,F) → logits (B,)"""
        if maps.dim() == 3:
            maps = maps.unsqueeze(0)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        h = self.enc(maps)
        return self.head(torch.cat([h, feats], dim=-1)).squeeze(-1)

    def prob(
        self,
        maps: Union[torch.Tensor, np.ndarray],
        feats: Union[torch.Tensor, np.ndarray],
    ) -> float:
        self.eval()
        if isinstance(maps, np.ndarray):
            maps = torch.from_numpy(maps.astype(np.float32))
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats.astype(np.float32))
        with torch.no_grad():
            return float(torch.sigmoid(self.forward(maps, feats)).item())


def load_escalate_net_map(
    ckpt: Optional[Path] = None,
    device: str = "cpu",
) -> EscalateNetMap:
    path = Path(ckpt) if ckpt else ESCALATE_MAP_CKPT
    net = EscalateNetMap()
    if path.exists():
        blob = torch.load(path, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def save_escalate_net_map(
    net: EscalateNetMap, path: Optional[Path] = None, **meta
) -> Path:
    path = Path(path or ESCALATE_MAP_CKPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "in_ch": net.in_ch,
            "feat_dim": net.feat_dim,
            "version": "v1_escalate_map",
            **meta,
        },
        path,
    )
    return path


def decide_escalate_map_ai(
    ctx: EscalateContext,
    maps: np.ndarray,
    net: Optional[EscalateNetMap],
    *,
    threshold: float = 0.48,
    dropoff_escalate_score: float = 0.45,
) -> tuple[bool, str, float]:
    """Escalate decision using map+task CNN.

    Hotspot / joint-fail remain hard triggers. Map topology is left to the net
    (no scalar map_hardness short-circuit) so the CNN can learn real difficulty.
    """
    if max(ctx.dropoff_score, ctx.pickup_score) >= dropoff_escalate_score:
        return True, "rule_hotspot", 1.0
    if ctx.recent_joint_fail > 0:
        return True, "rule_fail", 1.0
    if ctx.assigned_top_n >= 3:
        return True, "rule_assigned_burst", 1.0

    if net is None:
        # Fallback teacher without map net
        p = rule_escalate_prob(ctx)
        if ctx.was_escalated and ctx.calm_streak < 2:
            return True, "rule_hysteresis", p
        return p >= 0.55, "rule_scalar", p

    feats = build_escalate_features(ctx)
    p = net.prob(maps, feats)
    if ctx.was_escalated and ctx.calm_streak < 2 and p >= threshold * 0.7:
        return True, "mapnet_hysteresis", p
    if p >= threshold:
        return True, "mapnet", p
    if ctx.was_escalated and ctx.calm_streak < 2:
        return True, "rule_hysteresis", p
    return False, "mapnet_easy", p


# Keep old EscalateNet importable from this module for typing convenience.
__all__ = [
    "EscalateNetMap",
    "ESCALATE_MAP_CKPT",
    "load_escalate_net_map",
    "save_escalate_net_map",
    "decide_escalate_map_ai",
    "EscalateNet",
]
