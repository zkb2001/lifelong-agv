"""WaveNetMap — per-AGV score using shared map CNN embedding + agent features."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ml_research.common.paths import CKPT

from ml_research.benchmarks.wave_net.features import FEAT_DIM
from ml_research.benchmarks.hier_coord.obs import N_HIER_CHANNELS

WAVE_MAP_CKPT = CKPT / "wave_net_map_v1.pt"
WAVE_MAP_V2_CKPT = CKPT / "wave_net_map_v2.pt"


def _build_map_enc(in_ch: int, mid_ch: int, map_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, mid_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(mid_ch, map_dim),
        nn.ReLU(inplace=True),
    )


def _build_head_v1(map_dim: int, feat_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(map_dim + feat_dim, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, 1),
    )


def _build_head_v2(map_dim: int, feat_dim: int, hidden: int, dropout: float) -> nn.Sequential:
    drop = float(dropout)
    return nn.Sequential(
        nn.Linear(map_dim + feat_dim, hidden),
        nn.LayerNorm(hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(drop) if drop > 0 else nn.Identity(),
        nn.Linear(hidden, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(drop) if drop > 0 else nn.Identity(),
        nn.Linear(hidden, 1),
    )


class WaveNetMap(nn.Module):
    def __init__(
        self,
        in_ch: int = N_HIER_CHANNELS,
        feat_dim: int = FEAT_DIM,
        map_dim: int = 32,
        hidden: int = 64,
        dropout: float = 0.0,
        *,
        mid_ch: int = 48,
        head_version: str = "v2",
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.feat_dim = int(feat_dim)
        self.map_dim = int(map_dim)
        self.hidden = int(hidden)
        self.mid_ch = int(mid_ch)
        self.head_version = str(head_version)
        self.map_enc = _build_map_enc(self.in_ch, self.mid_ch, self.map_dim)
        if self.head_version == "v1":
            self.head = _build_head_v1(self.map_dim, self.feat_dim, self.hidden)
        else:
            self.head = _build_head_v2(self.map_dim, self.feat_dim, self.hidden, dropout)

    def encode_map(self, maps: torch.Tensor) -> torch.Tensor:
        if maps.dim() == 3:
            maps = maps.unsqueeze(0)
        return self.map_enc(maps)

    def forward(
        self, maps: torch.Tensor, feats: torch.Tensor, *, map_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if map_emb is None:
            map_emb = self.encode_map(maps)
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        if map_emb.shape[0] == 1 and feats.shape[0] > 1:
            map_emb = map_emb.expand(feats.shape[0], -1)
        return self.head(torch.cat([map_emb, feats], dim=-1)).squeeze(-1)

    def score(
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
            return float(self.forward(maps, feats).item())


def _infer_arch(state: dict) -> dict:
    """Infer WaveNetMap constructor kwargs from a state_dict."""
    mid_ch = int(state["map_enc.2.weight"].shape[0]) if "map_enc.2.weight" in state else 32
    map_dim = int(state["map_enc.6.weight"].shape[0]) if "map_enc.6.weight" in state else 32
    # LayerNorm present => v2 head
    if "head.1.weight" in state and state["head.1.weight"].dim() == 1:
        head_version = "v2"
        hidden = int(state["head.0.weight"].shape[0])
        feat_dim = int(state["head.0.weight"].shape[1] - map_dim)
    else:
        head_version = "v1"
        hidden = int(state["head.0.weight"].shape[0]) if "head.0.weight" in state else 64
        feat_dim = (
            int(state["head.0.weight"].shape[1] - map_dim) if "head.0.weight" in state else FEAT_DIM
        )
    return {
        "mid_ch": mid_ch,
        "map_dim": map_dim,
        "hidden": hidden,
        "feat_dim": feat_dim,
        "head_version": head_version,
        "dropout": 0.0,
    }


def load_wave_net_map(ckpt: Optional[Path] = None, device: str = "cpu") -> WaveNetMap:
    path = Path(ckpt) if ckpt else (
        WAVE_MAP_V2_CKPT if WAVE_MAP_V2_CKPT.exists() else WAVE_MAP_CKPT
    )
    blob = torch.load(path, map_location=device, weights_only=False) if path.exists() else {}
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    if not isinstance(state, dict) or not state:
        net = WaveNetMap()
        net.to(device)
        net.eval()
        return net

    kwargs = _infer_arch(state)
    if isinstance(blob, dict):
        kwargs["in_ch"] = int(blob.get("in_ch", N_HIER_CHANNELS))
        if "map_dim" in blob:
            kwargs["map_dim"] = int(blob["map_dim"])
        if "hidden" in blob:
            kwargs["hidden"] = int(blob["hidden"])
        if "feat_dim" in blob:
            kwargs["feat_dim"] = int(blob["feat_dim"])
        ver = str(blob.get("version") or "")
        if "v2" in ver:
            kwargs["head_version"] = "v2"
            kwargs.setdefault("mid_ch", 48)

    net = WaveNetMap(**kwargs)
    net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def save_wave_net_map(net: WaveNetMap, path: Optional[Path] = None, **meta) -> Path:
    path = Path(path or WAVE_MAP_CKPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "in_ch": net.in_ch,
            "feat_dim": net.feat_dim,
            "map_dim": getattr(net, "map_dim", 32),
            "hidden": getattr(net, "hidden", 64),
            "mid_ch": getattr(net, "mid_ch", 48),
            "head_version": getattr(net, "head_version", "v2"),
            "version": meta.pop("version", "v1_wave_map"),
            **meta,
        },
        path,
    )
    return path
