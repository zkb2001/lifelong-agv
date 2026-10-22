"""greed_coop_mcts — independent per-AGV greed–cooperation neural MCTS MAPD.

Separate from ECBS / Hier / A* pipelines under ``ml_research/benchmarks``.
"""
from __future__ import annotations

from greed_coop_mcts.network import (
    COSTTRACER_CKPT,
    CostTracerNet,
    load_cost_tracer,
    save_cost_tracer,
)

__all__ = [
    "COSTTRACER_CKPT",
    "CostTracerNet",
    "load_cost_tracer",
    "save_cost_tracer",
    "solve",
]


def __getattr__(name: str):
    if name == "solve":
        from greed_coop_mcts.solve import solve as _fn

        return _fn
    raise AttributeError(name)
