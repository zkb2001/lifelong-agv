"""Unit checks for dest-pad wave claim rules."""
from __future__ import annotations

from ml_research.benchmarks.coord_custom_ai.solve_ecbs import (
    _wave_claim_budget,
)


def _task(dest: str, pads):
    return {
        "destination": dest,
        "end_points": list(pads),
        "task_id": f"t-{dest}",
        "pickup_point": (2, 2),
    }


def test_no_double_count_same_hub():
    free = {(5, 5), (5, 6), (5, 7), (10, 10)}
    static = set()
    # Two surface tasks → same HubA with 3 pads ⇒ room 3, not 6.
    queues = {
        "S1": [_task("HubA", [(5, 5), (5, 6), (5, 7)])],
        "S2": [_task("HubA", [(5, 5), (5, 6), (5, 7)])],
    }
    cap, info = _wave_claim_budget(
        queues=queues,
        order=["S1", "S2"],
        free=free,
        static=static,
        bags=({},),
        n_free_agvs=8,
    )
    assert info["by_dest"]["HubA"]["pads"] == 3
    assert info["total_room"] == 3
    assert cap == 3


def test_subtract_inflight():
    free = {(5, 5), (5, 6), (5, 7)}
    static = set()
    queues = {"S1": [_task("HubA", [(5, 5), (5, 6), (5, 7)])]}
    pending = {"A1": _task("HubA", [(5, 5), (5, 6), (5, 7)])}
    cap, info = _wave_claim_budget(
        queues=queues,
        order=["S1"],
        free=free,
        static=static,
        bags=(pending,),
        n_free_agvs=8,
    )
    assert info["by_dest"]["HubA"]["room"] == 2
    assert cap == 2


def test_obstacle_pads_excluded():
    free = {(5, 5), (5, 6)}
    static = {(5, 7)}
    queues = {"S1": [_task("HubA", [(5, 5), (5, 6), (5, 7)])]}
    cap, info = _wave_claim_budget(
        queues=queues,
        order=["S1"],
        free=free,
        static=static,
        bags=({},),
        n_free_agvs=8,
    )
    assert info["by_dest"]["HubA"]["pads"] == 2
    assert cap == 2


def test_two_hubs_sum():
    free = {(5, 5), (5, 6), (8, 8), (8, 9)}
    static = set()
    queues = {
        "S1": [_task("HubA", [(5, 5), (5, 6)])],
        "S2": [_task("HubB", [(8, 8), (8, 9)])],
    }
    cap, info = _wave_claim_budget(
        queues=queues,
        order=["S1", "S2"],
        free=free,
        static=static,
        bags=({},),
        n_free_agvs=8,
    )
    assert info["total_room"] == 4
    assert cap == 4


def test_min_free_agvs():
    free = {(5, 5), (5, 6), (5, 7)}
    static = set()
    queues = {"S1": [_task("HubA", [(5, 5), (5, 6), (5, 7)])]}
    cap, _ = _wave_claim_budget(
        queues=queues,
        order=["S1"],
        free=free,
        static=static,
        bags=({},),
        n_free_agvs=2,
    )
    assert cap == 2


if __name__ == "__main__":
    test_no_double_count_same_hub()
    test_subtract_inflight()
    test_obstacle_pads_excluded()
    test_two_hubs_sum()
    test_min_free_agvs()
    print("ok")
