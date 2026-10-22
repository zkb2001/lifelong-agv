"""Global map → per-cell one-way traffic rules (CNN + Transformer).

Input: whole-map channels (walkable, AGV poses/headings, deliver/pickup goals,
conflict heat). Output: one direction per walkable cell (E/N/W/S).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from greed_coop_mcts.paths import CKPT
from greed_coop_mcts.rule_obs import GRID, N_RULE_CH

RULE_CKPT = CKPT / "global_rule_transformer_v1.pt"
N_DIRS = 4


class GlobalRuleNet(nn.Module):
    """Encode global grid, refine with Transformer tokens, decode dir logits."""

    def __init__(
        self,
        channels: int = N_RULE_CH,
        grid: int = GRID,
        width: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_dirs: int = N_DIRS,
    ):
        super().__init__()
        self.grid = int(grid)
        self.stem = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, d_model, 3, stride=2, padding=1),  # ~11x11
            nn.GELU(),
        )
        # tokens from coarse map
        self.tok_norm = nn.LayerNorm(d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(d_model, width, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, n_dirs, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B,C,H,W) → logits (B,4,H,W)."""
        b, c, h, w = x.shape
        feat = self.stem(x)  # (B,D,h',w')
        _, d, hh, ww = feat.shape
        tok = feat.flatten(2).transpose(1, 2)  # (B, h'*w', D)
        tok = self.tok_norm(tok)
        tok = self.transformer(tok)
        feat2 = tok.transpose(1, 2).reshape(b, d, hh, ww)
        logits = self.up(feat2)
        if logits.shape[-2:] != (h, w):
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits

    def predict_dirs(
        self,
        state: Union[np.ndarray, torch.Tensor],
        walkable: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """Return (H,W) int dirs; -1 on non-walkable."""
        self.eval()
        if isinstance(state, np.ndarray):
            x = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        else:
            x = state.float()
            if x.dim() == 3:
                x = x.unsqueeze(0)
        x = x.to(device)
        with torch.no_grad():
            logits = self.forward(x)[0]  # (4,H,W)
            dirs = logits.argmax(dim=0).cpu().numpy().astype(np.int64)
        dirs = np.where(walkable > 0.5, dirs, -1)
        return dirs


def load_rule_net(ckpt: Optional[Path] = None, device: str = "cpu") -> GlobalRuleNet:
    path = Path(ckpt) if ckpt else RULE_CKPT
    net = GlobalRuleNet()
    if path.exists():
        blob = torch.load(path, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def save_rule_net(net: GlobalRuleNet, path: Optional[Path] = None, **meta) -> Path:
    path = Path(path or RULE_CKPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "version": "global_rule_transformer_v1",
            "channels": N_RULE_CH,
            "grid": GRID,
            "arch": "cnn_transformer_rule_field",
            **meta,
        },
        path,
    )
    return path
