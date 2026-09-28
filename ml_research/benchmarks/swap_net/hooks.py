"""Attach SwapNet v3 policy to simulation."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ml_research.benchmarks.station_eta import ensure_station_eta_hooks
from ml_research.benchmarks.swap_net.model import SwapNet, load_swap_net
from ml_research.benchmarks.swap_net.policy import make_swap_policy


def attach_swap_net(
    sim,
    *,
    ckpt: Optional[Path] = None,
    threshold_normal: float = 0.48,
    threshold_urgent: float = 0.40,
    margin: float = 4.0,
) -> SwapNet:
    """Install station_eta hooks + urgent-aware SwapNet gate on ``sim``."""
    ensure_station_eta_hooks(sim, margin=margin)
    net = load_swap_net(ckpt)
    sim._swap_policy = make_swap_policy(
        net,
        threshold_normal=threshold_normal,
        threshold_urgent=threshold_urgent,
    )
    sim._swapnet_net = net
    return net
