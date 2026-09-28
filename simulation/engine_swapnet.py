"""Engine fork: station ETA / SwapNet swap tick before assign."""
from __future__ import annotations

import simulation.engine as _engine
from ml_research.benchmarks.station_eta import station_eta_swap_tick

# Re-export engine API for importlib loaders (run_swapnet_ab).
for _name in dir(_engine):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_engine, _name)

Simulation = _engine.Simulation
ASTAR_WALL_BUDGET_S = getattr(_engine, "ASTAR_WALL_BUDGET_S", 0.35)


def _pre_assign_swap(self):
    """Inject SwapNet without replacing time_forward (keeps horizon replan)."""
    period = max(1, int(getattr(self, "_swapnet_period", 2) or 2))
    if getattr(self, "_swapnet_enabled", False) and (int(self.time) % period == 0):
        station_eta_swap_tick(self)


Simulation._pre_assign_hook = _pre_assign_swap
