"""ParkArterySemaphoreNet v5 + legacy TrafficRuleNet v4."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .config import (
    BLOCK,
    GRID,
    HIDDEN,
    N_DIRS,
    N_MAP_CHANNELS,
    N_MAP_CHANNELS_V4,
    N_SEM_FEATURES,
    N_ZONES,
)


class ParkArterySemaphoreNet(nn.Module):
    """Map CNN → parking / artery / connectivity fields + semaphore MLP.

    Tag: park_artery_sem_v1. Designed for failure-recovery (not lane dir/phase).
    """

    def __init__(
        self,
        in_ch: int = N_MAP_CHANNELS,
        hidden: int = HIDDEN,
        grid: int = GRID,
        n_sem_feat: int = N_SEM_FEATURES,
    ):
        super().__init__()
        self.grid = int(grid)
        self.n_sem_feat = int(n_sem_feat)
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.parking_head = nn.Conv2d(hidden, 1, 1)
        self.artery_head = nn.Conv2d(hidden, 1, 1)
        self.connectivity_head = nn.Conv2d(hidden, 1, 1)
        self.sem_mlp = nn.Sequential(
            nn.Linear(hidden + n_sem_feat, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.enc(x)

    def forward(
        self,
        x: torch.Tensor,
        sem_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.encode(x)
        out: Dict[str, torch.Tensor] = {
            "parking_logits": self.parking_head(h),
            "artery_logits": self.artery_head(h),
            "connectivity_logits": self.connectivity_head(h),
            "value": self.value_head(h).squeeze(-1),
            "h": h,
        }
        pooled = h.mean(dim=(2, 3))  # (B, hidden)
        if sem_feat is None:
            b = pooled.shape[0]
            sem_feat = pooled.new_zeros((b, self.n_sem_feat))
        else:
            if sem_feat.dim() == 1:
                sem_feat = sem_feat.unsqueeze(0)
            if sem_feat.shape[-1] != self.n_sem_feat:
                fixed = pooled.new_zeros((sem_feat.shape[0], self.n_sem_feat))
                n = min(self.n_sem_feat, sem_feat.shape[-1])
                fixed[:, :n] = sem_feat[:, :n]
                sem_feat = fixed
        out["semaphore_logits"] = self.sem_mlp(torch.cat([pooled, sem_feat], dim=-1)).squeeze(-1)
        return out

    @torch.no_grad()
    def act(
        self,
        x: torch.Tensor,
        sem_feat: Optional[torch.Tensor] = None,
        *,
        deterministic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        out = self.forward(x, sem_feat=sem_feat)
        out["parking"] = torch.sigmoid(out["parking_logits"][:, 0])
        out["artery"] = torch.sigmoid(out["artery_logits"][:, 0])
        out["connectivity"] = torch.sigmoid(out["connectivity_logits"][:, 0])
        out["semaphore"] = torch.sigmoid(out["semaphore_logits"])
        return out


class TrafficRuleNet(nn.Module):
    """Legacy environment-centric traffic rules (lane_traffic_rules_v4)."""

    def __init__(
        self,
        in_ch: int = N_MAP_CHANNELS_V4,
        hidden: int = HIDDEN,
        grid: int = GRID,
        block: int = BLOCK,
        *,
        with_parking_head: bool = False,
    ):
        super().__init__()
        self.grid = grid
        self.block = block
        self.with_parking_head = bool(with_parking_head)
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dir_head = nn.Conv2d(hidden, N_DIRS, 1)
        self.zone_head = nn.Conv2d(hidden, N_ZONES, 1)
        self.yield_head = nn.Conv2d(hidden, 1, 1)
        self.artery_head = nn.Conv2d(hidden, 1, 1)
        self.bidir_head = nn.Conv2d(hidden, 1, 1)
        self.nowait_head = nn.Conv2d(hidden, 1, 1)
        self.construction_head = nn.Conv2d(hidden, N_DIRS, 1)
        if self.with_parking_head:
            self.parking_head = nn.Conv2d(hidden, 1, 1)
        jy = max(1, grid // block)
        jx = max(1, grid // block)
        self.jy, self.jx = jy, jx
        self.phase_pool = nn.AdaptiveAvgPool2d((jy, jx))
        self.phase_head = nn.Conv2d(hidden, 2, 1)
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        h = self.enc(x)
        out = {
            "dir_logits": self.dir_head(h),
            "phase_logits": self.phase_head(self.phase_pool(h)),
            "zone_logits": self.zone_head(h),
            "yield_logits": self.yield_head(h),
            "artery_logits": self.artery_head(h),
            "bidir_logits": self.bidir_head(h),
            "nowait_logits": self.nowait_head(h),
            "construction_logits": self.construction_head(h),
            "value": self.value_head(h).squeeze(-1),
        }
        if self.with_parking_head:
            out["parking_logits"] = self.parking_head(h)
        return out

    @torch.no_grad()
    def act(self, x: torch.Tensor, *, deterministic: bool = True) -> Dict[str, torch.Tensor]:
        out = self.forward(x)
        out["dir"] = out["dir_logits"].argmax(dim=1)
        out["phase"] = out["phase_logits"].argmax(dim=1)
        out["zone"] = out["zone_logits"].argmax(dim=1)
        yld = (torch.sigmoid(out["yield_logits"][:, 0]) > 0.5).to(torch.float32)
        if "parking_logits" in out:
            park = (torch.sigmoid(out["parking_logits"][:, 0]) > 0.5).to(torch.float32)
            out["parking"] = park
            yld = torch.maximum(yld, park)
        out["yield"] = yld
        out["artery"] = torch.sigmoid(out["artery_logits"][:, 0])
        out["bidir"] = torch.sigmoid(out["bidir_logits"][:, 0])
        out["nowait"] = (torch.sigmoid(out["nowait_logits"][:, 0]) > 0.5).to(torch.float32)
        out["construction"] = torch.zeros_like(out["construction_logits"])
        return out
