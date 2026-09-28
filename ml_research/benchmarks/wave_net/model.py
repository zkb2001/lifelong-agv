"""WaveNet — scores AGVs for inclusion in the current joint planning wave."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ml_research.common.paths import CKPT

from .features import FEAT_DIM

WAVE_CKPT = CKPT / "wave_net_v1.pt"


class WaveNet(nn.Module):
    """MLP: FEAT_DIM → 48 → 48 → 1 (logit). Higher score = prefer in-wave."""

    def __init__(self, feat_dim: int = FEAT_DIM, hidden: int = 48):
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

    def score(self, x: Union[torch.Tensor, np.ndarray]) -> float:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return float(self.forward(x).item())


def load_wave_net(
    ckpt: Optional[Path] = None,
    device: str = "cpu",
) -> WaveNet:
    path = Path(ckpt) if ckpt else WAVE_CKPT
    net = WaveNet(feat_dim=FEAT_DIM)
    if path.exists():
        blob = torch.load(path, map_location=device, weights_only=False)
        fd = int(blob.get("feat_dim", FEAT_DIM)) if isinstance(blob, dict) else FEAT_DIM
        if fd != FEAT_DIM:
            net = WaveNet(feat_dim=fd)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def save_wave_net(net: WaveNet, path: Optional[Path] = None, **meta) -> Path:
    path = Path(path or WAVE_CKPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": net.state_dict(),
        "feat_dim": net.feat_dim,
        "version": "v1_wave_select",
        **meta,
    }
    torch.save(payload, path)
    return path
