"""Compact CostTracer policy-value network (~100k params)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from greed_coop_mcts.obs import (
    N_ACTIONS,
    N_CHANNELS,
    OBS_SIZE,
)
from greed_coop_mcts.paths import COSTTRACER_CKPT


class CostTracerNet(nn.Module):
    """Spatial encoder + policy/value heads (MATS-LP CostTracer style)."""

    def __init__(
        self,
        channels: int = N_CHANNELS,
        obs_size: int = OBS_SIZE,
        filters: int = 32,
        mlp: int = 64,
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()
        self.obs_size = int(obs_size)
        self.enc = nn.Sequential(
            nn.Conv2d(channels, filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        flat = filters * obs_size * obs_size
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, mlp),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(mlp, n_actions)
        self.value = nn.Sequential(
            nn.Linear(mlp, mlp // 2),
            nn.ReLU(inplace=True),
            nn.Linear(mlp // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B,C,H,W) → logits (B,A), value (B,) in roughly [-1,1]."""
        h = self.trunk(self.enc(x))
        logits = self.policy(h)
        if mask is not None:
            logits = logits.masked_fill(mask < 0.5, -1e9)
        v = torch.tanh(self.value(h).squeeze(-1))
        return logits, v

    def prob_value(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, float]:
        self.eval()
        if isinstance(obs, np.ndarray):
            x = torch.from_numpy(obs.astype(np.float32))
        else:
            x = obs.float()
        if x.dim() == 3:
            x = x.unsqueeze(0)
        m = None
        if mask is not None:
            m = (
                torch.from_numpy(mask.astype(np.float32))
                if isinstance(mask, np.ndarray)
                else mask.float()
            )
            if m.dim() == 1:
                m = m.unsqueeze(0)
            m = m.to(device)
        x = x.to(device)
        with torch.no_grad():
            logits, v = self.forward(x, m)
            p = F.softmax(logits, dim=-1)[0].cpu().numpy()
            return p.astype(np.float32), float(v[0].item())


def load_cost_tracer(
    ckpt: Optional[Path] = None, device: str = "cpu"
) -> CostTracerNet:
    path = Path(ckpt) if ckpt else COSTTRACER_CKPT
    net = CostTracerNet()
    if path.exists():
        blob = torch.load(path, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def save_cost_tracer(
    net: CostTracerNet, path: Optional[Path] = None, **meta
) -> Path:
    path = Path(path or COSTTRACER_CKPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "version": "greed_coop_cost_tracer_v1",
            "obs_size": OBS_SIZE,
            "n_actions": N_ACTIONS,
            **meta,
        },
        path,
    )
    return path
