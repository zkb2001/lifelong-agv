"""mats_local — CostTracer-style policy-value + local neural MCTS for MAPD.

Standalone lifelong MAPD line (no Hier/ECBS hook).
"""
from __future__ import annotations

from ml_research.benchmarks.mats_local.network import (
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
    "solve_standalone",
]


def __getattr__(name: str):
    if name == "solve_standalone":
        from ml_research.benchmarks.mats_local.solve_standalone import (
            solve_standalone as _fn,
        )

        return _fn
    raise AttributeError(name)
