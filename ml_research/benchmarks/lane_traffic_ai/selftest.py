"""Unit self-tests for rule park/artery recovery (+ optional net shapes)."""
from __future__ import annotations

import numpy as np

from .config import GRID, N_MAP_CHANNELS, PARKING_THRESH
from .features import bfs_dist_field, build_park_artery_channels
from .net import ParkArterySemaphoreNet
from .policy import ParkArteryPolicy
from .recovery import select_parking_cell
from .rule_policy import RuleParkArteryPolicy
from .teacher import teacher_park_artery_from_channels
from .train_bc import _empty_walk_channels, make_synthetic_batch


def test_shapes() -> None:
    ch = _empty_walk_channels(failed_pos=[(5, 5)], unloads=[(15, 10)], seed=1)
    assert ch.shape == (N_MAP_CHANNELS, GRID, GRID), ch.shape
    lab = teacher_park_artery_from_channels(
        ch, goals=[(15, 10)], sem_pos=(5, 5), sem_goal=(15, 10)
    )
    assert lab["parking"].shape == (GRID, GRID)
    assert lab["artery"].shape == (GRID, GRID)
    assert lab["connectivity"].shape == (GRID, GRID)
    assert lab["sem_feat"].shape == (8,)
    net = ParkArterySemaphoreNet()
    import torch

    x = torch.as_tensor(ch).unsqueeze(0)
    feat = torch.as_tensor(lab["sem_feat"]).unsqueeze(0)
    out = net(x, sem_feat=feat)
    assert out["parking_logits"].shape == (1, 1, GRID, GRID)
    assert out["semaphore_logits"].shape == (1,)
    print("[selftest] shapes OK")


def test_select_parking_argmin() -> None:
    walk = np.zeros((GRID, GRID), dtype=np.float32)
    walk[1:21, 1:21] = 1.0
    parking = np.zeros((GRID, GRID), dtype=np.float32)
    parking[3, 3] = 1.0
    parking[12, 14] = 1.0
    unload = (15, 15)
    dist = bfs_dist_field(walk, [unload])
    assert dist[12, 14] < dist[3, 3]
    goal = select_parking_cell(
        parking=parking,
        unload=unload,
        walkable=walk,
        occupied=set(),
    )
    assert goal == (14, 12), goal
    print("[selftest] select_parking OK", goal)


def test_semaphore_blocks_when_path_unclear() -> None:
    artery = np.zeros((GRID, GRID), dtype=np.float32)
    parking = np.zeros((GRID, GRID), dtype=np.float32)
    walk = np.zeros((GRID, GRID), dtype=np.float32)
    walk[1:21, 1:21] = 1.0
    artery[1:21, 10] = 1.0
    parking[4, 4] = 1.0
    pol = RuleParkArteryPolicy(
        artery=artery, parking=parking, walkable=walk, map_id="test"
    )
    ch = _empty_walk_channels(failed_pos=[(4, 4)], unloads=[(16, 16)], seed=2)
    ch[3, 4:17, :] = 1.0
    ch[3] *= ch[1]
    fields = pol.infer(channels=ch, sem_pos=(4, 4), sem_goal=(16, 16), path_clear=0.0)
    allow = pol.semaphore_allow(
        fields, pos=(4, 4), goal=(16, 16), pad_free=1.0, path_clear=0.0
    )
    assert allow is False
    allow2 = pol.semaphore_allow(
        fields, pos=(4, 4), goal=(16, 16), pad_free=1.0, path_clear=1.0
    )
    assert isinstance(allow2, bool)
    print("[selftest] rule semaphore API OK", allow, allow2)


def test_bc_batch() -> None:
    x, y = make_synthetic_batch(4, seed=3)
    assert x.shape[0] == 4 and x.shape[1] == N_MAP_CHANNELS
    assert y["parking"].shape[0] == 4
    print("[selftest] bc batch OK")


def test_rule_policy_from_labels() -> None:
    from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
    from .map_labels import load_map_labels

    meta = load_custom_meta(1)
    assert meta, "SH01 meta missing"
    labels = load_map_labels(str(meta["id"]), meta=meta, compute_if_missing=True)
    pol = RuleParkArteryPolicy(
        artery=labels["artery"],
        parking=labels["parking"],
        walkable=labels.get("walkable"),
        map_id=str(meta["id"]),
    )
    fields = pol.infer(channels=_empty_walk_channels(unloads=[(10, 10)], seed=4))
    assert float(fields["parking"].sum()) > 0
    assert float(fields["artery"].sum()) > 0
    assert fields.get("rule") is True
    print(
        f"[selftest] rule policy OK artery={(fields['artery']>=0.5).sum()} "
        f"parking={(fields['parking']>=0.5).sum()}"
    )


def test_semaphore_from_parking_to_station() -> None:
    """Parking cell must get connectivity; station snaps to walkable pad."""
    from .teacher import teacher_connectivity_field, teacher_semaphore_label

    walk = np.zeros((GRID, GRID), dtype=np.float32)
    walk[1:21, 1:21] = 1.0
    # station blocked like Beijing
    station = (6, 4)
    walk[station[1], station[0]] = 0.0
    artery = np.zeros((GRID, GRID), dtype=np.float32)
    artery[1:21, 6] = 1.0
    artery[3, 1:21] = 1.0
    parking = np.zeros((GRID, GRID), dtype=np.float32)
    parking[2, 5:9] = 1.0
    walk[2, 5:9] = 1.0
    pos = (6, 2)
    conn = teacher_connectivity_field(
        walk, artery, [station], parking=parking
    )
    assert float(conn[pos[1], pos[0]]) > 0.15, float(conn[pos[1], pos[0]])
    # pad (6,3) must be reachable goal after snap
    assert float(conn[3, 6]) > 0.15
    lab = teacher_semaphore_label(
        connectivity=conn,
        pos=pos,
        goal=(6, 3),
        pad_free=1.0,
        path_clear=1.0,
        congestion=0.0,
    )
    assert float(lab) >= 0.5, lab
    print("[selftest] semaphore from parking OK", float(conn[pos[1], pos[0]]), lab)


def test_resolve_unload_pads_snaps_station() -> None:
    from .recovery import resolve_unload_pads, snap_walkable, sim_walkable

    class _Env:
        def get_static_obstacles(self):
            return [(6, 4)]

    class _Sim:
        env = _Env()

    sim = _Sim()
    walk = sim_walkable(sim)
    assert float(walk[4, 6]) < 0.5
    primary, pads, station = resolve_unload_pads(
        sim,
        {"unload_point": (6, 4), "end_points": [(7, 4), (5, 4), (6, 5), (6, 3)]},
    )
    assert station == (6, 4)
    assert primary is not None and float(walk[primary[1], primary[0]]) > 0.5
    assert all(float(walk[p[1], p[0]]) > 0.5 for p in pads)
    assert snap_walkable(walk, (6, 4)) in ((6, 3), (6, 5), (5, 4), (7, 4))
    print("[selftest] resolve unload pads OK", primary, len(pads))


def test_freq_labels() -> None:
    from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
    from .freq_probe import (
        cartesian_delivery_frequency,
        labels_from_frequency,
        stations_dests_from_meta,
        walkable_and_obstacles,
    )

    meta = load_custom_meta(1)
    assert meta, "SH01 meta missing"
    pickups, dropoffs = stations_dests_from_meta(meta)
    assert pickups and dropoffs
    walk, _ = walkable_and_obstacles(meta)
    freq = cartesian_delivery_frequency(meta, pickups, dropoffs)
    assert float(freq.sum()) > 0
    lab = labels_from_frequency(freq, walk, pickups=pickups, dropoffs=dropoffs)
    assert float(lab["artery"].sum()) > 0
    print(
        f"[selftest] freq labels OK artery={(lab['artery']>=0.5).sum()} "
        f"parking={(lab['parking']>=0.5).sum()} freq_sum={freq.sum():.0f}"
    )


def main() -> int:
    test_shapes()
    test_select_parking_argmin()
    test_semaphore_blocks_when_path_unclear()
    test_bc_batch()
    test_rule_policy_from_labels()
    test_semaphore_from_parking_to_station()
    test_resolve_unload_pads_snaps_station()
    test_freq_labels()
    _ = build_park_artery_channels
    _ = ParkArteryPolicy
    _ = PARKING_THRESH
    print("[selftest] ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
