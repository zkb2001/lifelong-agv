"""Regression: unload completion must track the latest loaded streak only."""
from __future__ import annotations

from types import SimpleNamespace


def _step(t, x, y, loaded, tid, dest=""):
    return {
        "timestamp": t,
        "name": "A",
        "X": x,
        "Y": y,
        "pitch": 90,
        "loaded": "TRUE" if loaded else "FALSE",
        "destination": dest,
        "Emergency": "FALSE",
        "task-id": tid,
    }


def test_historical_unload_does_not_clear_active_recarry():
    """Horse-6 unloaded once, then re-picked; still loaded → not completed."""
    from simulation.engine import Simulation

    # Minimal stub: only need the method + pad lookup.
    sim = object.__new__(Simulation)
    pads = {(5, 4)}
    sim._task_unload_pads = lambda tid: set(pads)  # type: ignore[method-assign]
    sim._is_unload_interaction_cell = lambda pos, tid=None: pos in pads  # type: ignore

    agv = SimpleNamespace(
        steps=[
            _step(86, 2, 14, True, "Horse-6", "Beijing"),
            _step(103, 5, 4, False, "", ""),  # first unload on pad
            _step(125, 2, 14, True, "Horse-6", "Beijing"),  # re-carry
            _step(160, 18, 10, True, "Horse-6", "Beijing"),
        ]
    )
    assert sim._has_completed_unload_for_task(agv, "Horse-6") is False


def test_latest_streak_unload_on_pad_counts():
    from simulation.engine import Simulation

    sim = object.__new__(Simulation)
    pads = {(5, 4)}
    sim._task_unload_pads = lambda tid: set(pads)  # type: ignore[method-assign]
    sim._is_unload_interaction_cell = lambda pos, tid=None: pos in pads  # type: ignore

    agv = SimpleNamespace(
        steps=[
            _step(86, 2, 14, True, "Horse-6", "Beijing"),
            _step(103, 5, 4, False, "", ""),
            _step(125, 2, 14, True, "Horse-6", "Beijing"),
            _step(200, 5, 4, False, "", ""),  # second unload
        ]
    )
    assert sim._has_completed_unload_for_task(agv, "Horse-6") is True


if __name__ == "__main__":
    test_historical_unload_does_not_clear_active_recarry()
    test_latest_streak_unload_on_pad_counts()
    print("ok")
