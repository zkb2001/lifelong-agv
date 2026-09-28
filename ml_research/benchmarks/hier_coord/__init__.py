"""Hierarchical coordination: EscalateNet (upper) + WaveNet hooks.

TrafficNet-lite / neural edge shaping is kept as unused library code only;
the main ECBS pipeline does not call it.
"""
from __future__ import annotations

from .escalate import (
    ESCALATE_FEAT_DIM,
    EscalateNet,
    build_escalate_features,
    load_escalate_net,
    save_escalate_net,
    decide_escalate_ai,
)

__all__ = [
    "ESCALATE_FEAT_DIM",
    "EscalateNet",
    "build_escalate_features",
    "decide_escalate_ai",
    "load_escalate_net",
    "save_escalate_net",
]
