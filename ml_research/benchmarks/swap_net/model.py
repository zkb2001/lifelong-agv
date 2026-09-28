"""SwapNet v3 — binary gate for TPTS station task swap."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ml_research.common.paths import CKPT

from .features import FEAT_DIM

SWAP_CKPT = CKPT / "swap_net.pt"
SWAP_CKPT_V3 = CKPT / "swap_net_v3.pt"


class SwapNet(nn.Module):
    """MLP: FEAT_DIM → 48 → 48 → 1 (logit)."""

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

    def prob(self, x: Union[torch.Tensor, np.ndarray]) -> float:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(self.forward(x)).item())

    def decide(
        self,
        x: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
    ) -> bool:
        return self.prob(x) >= float(threshold)


def load_swap_net(
    ckpt: Optional[Path] = None,
    device: str = "cpu",
) -> SwapNet:
    path = Path(ckpt) if ckpt else SWAP_CKPT_V3
    if not path.exists():
        path = SWAP_CKPT
    net = SwapNet(feat_dim=FEAT_DIM)
    if path.exists():
        blob = torch.load(path, map_location=device, weights_only=False)
        fd = int(blob.get("feat_dim", FEAT_DIM)) if isinstance(blob, dict) else FEAT_DIM
        if fd != FEAT_DIM:
            net = SwapNet(feat_dim=fd)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def save_swap_net(net: SwapNet, path: Optional[Path] = None, **meta) -> Path:
    path = Path(path or SWAP_CKPT_V3)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": net.state_dict(),
        "feat_dim": net.feat_dim,
        "version": "v3_urgent",
        **meta,
    }
    torch.save(payload, path)
    return path
