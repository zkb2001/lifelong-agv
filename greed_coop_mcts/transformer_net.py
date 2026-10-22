"""SCRIMP-style CNN + Transformer communication policy-value net.

Each AGV encodes a local FOV with a CNN, then a Transformer encoder mixes
messages across agents (scalable communication). Policy / value heads are
per-agent. Inspired by SCRIMP (AAMAS 2023) and MATS-LP neural priors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from greed_coop_mcts.obs import N_ACTIONS, OBS_SIZE
from greed_coop_mcts.paths import CKPT

# Richer FOV channels for Transformer training.
XFMR_CHANNELS = 4  # agents | obstacles | cost-to-go | goal-presence
TRANSFORMER_CKPT = CKPT / "mapd_transformer_v1.pt"


class SpatialEncoder(nn.Module):
    def __init__(self, channels: int = XFMR_CHANNELS, filters: int = 64, d_model: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(filters, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*N, C, H, W) or (B, C, H, W)
        h = self.conv(x).flatten(1)
        return self.proj(h)


class MapdTransformer(nn.Module):
    """Multi-agent FOV encoder + Transformer communication + π/V heads."""

    def __init__(
        self,
        channels: int = XFMR_CHANNELS,
        obs_size: int = OBS_SIZE,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn: int = 256,
        n_actions: int = N_ACTIONS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = int(channels)
        self.obs_size = int(obs_size)
        self.d_model = int(d_model)
        self.enc = SpatialEncoder(channels=channels, filters=64, d_model=d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.comm = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.policy = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward_team(
        self,
        obs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        agent_pad: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """obs: (B, N, C, H, W); mask: (B, N, A); agent_pad: (B, N) 1=valid.

        Returns logits (B, N, A), values (B, N).
        """
        b, n, c, h, w = obs.shape
        flat = obs.reshape(b * n, c, h, w)
        tok = self.enc(flat).reshape(b, n, self.d_model)
        key_pad = None
        if agent_pad is not None:
            # True where we should IGNORE (PyTorch convention)
            key_pad = agent_pad < 0.5
        mixed = self.comm(tok, src_key_padding_mask=key_pad)
        logits = self.policy(mixed)
        values = torch.tanh(self.value(mixed).squeeze(-1))
        if mask is not None:
            logits = logits.masked_fill(mask < 0.5, -1e9)
        return logits, values

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-agent / batched FOV: x (B,C,H,W) → logits (B,A), v (B,)."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        b = x.shape[0]
        obs = x.unsqueeze(1)  # (B,1,C,H,W)
        m = None
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            m = mask.unsqueeze(1)  # (B,1,A)
        logits, values = self.forward_team(obs, m)
        return logits[:, 0], values[:, 0]

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

    def team_actions(
        self,
        obs_n: np.ndarray,
        mask_n: np.ndarray,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """obs_n (N,C,H,W), mask_n (N,A) → action indices (N,), values (N,)."""
        self.eval()
        x = torch.from_numpy(obs_n.astype(np.float32)).unsqueeze(0).to(device)
        m = torch.from_numpy(mask_n.astype(np.float32)).unsqueeze(0).to(device)
        pad = torch.ones(1, obs_n.shape[0], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, values = self.forward_team(x, m, pad)
            acts = logits[0].argmax(dim=-1).cpu().numpy()
            vs = values[0].cpu().numpy()
        return acts.astype(np.int64), vs.astype(np.float32)


def load_transformer(
    ckpt: Optional[Path] = None, device: str = "cpu"
) -> MapdTransformer:
    path = Path(ckpt) if ckpt else TRANSFORMER_CKPT
    net = MapdTransformer()
    if path.exists():
        blob = torch.load(path, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def save_transformer(
    net: MapdTransformer, path: Optional[Path] = None, **meta
) -> Path:
    path = Path(path or TRANSFORMER_CKPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "version": "mapd_transformer_v1",
            "channels": XFMR_CHANNELS,
            "obs_size": OBS_SIZE,
            "n_actions": N_ACTIONS,
            "arch": "cnn_transformer_comm",
            **meta,
        },
        path,
    )
    return path
