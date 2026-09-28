"""Shared neural modules."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """CNN over local occupancy + MLP over kinematic / goal features."""

    def __init__(self, patch_size: int = 7, n_extra: int = 8, n_actions: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 + n_extra, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, patch: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        # patch: (B, 1, H, W)
        h = self.conv(patch).flatten(1)
        return self.fc(torch.cat([h, extra], dim=1))


class HeuristicNet(nn.Module):
    """Predict remaining path cost (neural heuristic)."""

    def __init__(self, in_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ConflictNet(nn.Module):
    """Predict whether a local spatio-temporal window will contain a conflict."""

    def __init__(self, patch_size: int = 7, n_extra: int = 6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 + n_extra, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, patch: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        h = self.conv(patch).flatten(1)
        return self.fc(torch.cat([h, extra], dim=1)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int = 5):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.pi = nn.Linear(128, n_actions)
        self.v = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.pi(h), self.v(h).squeeze(-1)
