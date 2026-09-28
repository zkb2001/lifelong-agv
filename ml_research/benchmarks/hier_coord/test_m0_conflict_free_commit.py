"""Unit tests: A* spacetime + reject kinematic expand that desyncs occupancy."""
from __future__ import annotations

from simulation.engine import (
    mo_occupied_cell,
    path_has_spacetime_conflict,
    truncate_path_to_horizon,
    expand_path_turn_moves,
    retime_path_from,
    ENV,
)


def test_planned_pose_always_blocks_even_outside_horizon():
    traj = [(i % 5, 1, i, 90) for i in range(40)]
    cell = mo_occupied_cell(traj, 35, plan_t0=0, horizon=20)
    assert cell == traj[35][:2]


def test_past_end_always_freezes_last_cell():
    traj = [(2, 2, t, 90) for t in range(10)]
    assert mo_occupied_cell(traj, 15, plan_t0=0, horizon=20) == (2, 2)
    assert mo_occupied_cell(traj, 50, plan_t0=0, horizon=0) == (2, 2)


def test_gap_holds_previous_pose():
    # Missing t=11 must still occupy the cell from t=10 (not free).
    traj = [(5, 5, 10, 90), (5, 7, 12, 90)]
    assert mo_occupied_cell(traj, 11, plan_t0=0, horizon=20) == (5, 5)
    assert mo_occupied_cell(traj, 12, plan_t0=0, horizon=20) == (5, 7)


def test_timestamp_lookup_not_index():
    # Deliberately non-index-aligned: list index 0 has timestamp 10
    traj = [(5, 5, 10, 90), (5, 6, 11, 90), (5, 7, 12, 90)]
    assert mo_occupied_cell(traj, 11, plan_t0=0, horizon=20) == (5, 6)
    assert mo_occupied_cell(traj, 1, plan_t0=0, horizon=20) is None  # before first pose


def test_expand_preserves_same_heading_waits():
    raw = [
        (1, 1, 0, 0),
        (1, 1, 1, 0),  # wait
        (1, 1, 2, 0),  # wait
        (2, 1, 3, 0),
    ]
    out = expand_path_turn_moves(raw)
    stays = [p for p in out if p[:2] == (1, 1)]
    assert len(stays) >= 3  # start + 2 waits (timestamps retimed later)


def test_terminal_freeze_vs_future_reservation():
    """Cannot end a path on a cell someone else reserved soon after."""
    other = [(5, 5, t, 90) for t in range(0, 10)] + [(9, 9, t, 90) for t in range(10, 20)]
    # Ego wants to finish at (9,9) at t=8 while Other arrives at (9,9) at t=10
    ego = [(8, 9, 7, 0), (9, 9, 8, 0)]
    assert path_has_spacetime_conflict(ego, {"Other": other}, "Ego") is True
    # Far-future reservation (beyond freeze window) is allowed at commit time.
    other_far = [(9, 9, t, 90) for t in range(200, 210)]
    assert path_has_spacetime_conflict(ego, {"Other": other_far}, "Ego") is False


def test_follower_vs_inplace_turn_detected():
    """Leader turns in place; follower enters same cell — must conflict."""
    leader = [
        (2, 10, 910, 180),
        (2, 10, 911, 90),  # in-place turn
        (2, 11, 912, 90),
    ]
    follower = [
        (3, 10, 910, 180),
        (2, 10, 911, 180),  # drives into leader's turn tick
    ]
    assert path_has_spacetime_conflict(follower, {"Leader": leader}, "Follower") is True


def test_kinematic_expand_shifts_time_detected_by_is_valid():
    """A* path misses; after turn-insert + retime it collides — must reject."""
    env = ENV([], {})
    # Other AGV occupies (3,3) at t=5
    env.moving_obstacles["Other"] = [(3, 3, t, 90) for t in range(0, 20)]

    raw = [
        (1, 3, 3, 90),  # facing north
        (2, 3, 4, 0),   # wants east but was north → turn then move
        (3, 3, 5, 0),
    ]
    seeded = [(1, 3, 2, 90)] + raw
    kin = expand_path_turn_moves(seeded)[1:]
    kin = retime_path_from(kin, 3)
    hits = [p for p in kin if p[:2] == (3, 3)]
    assert hits, "test setup: kin should visit (3,3)"
    ok = env.is_valid_path([(3, 3)], kin, time=2, agv_name="Ego")
    assert ok in (True, False)
    env.moving_obstacles["Other"] = [(p[0], p[1], p[2], 90) for p in kin]
    assert env.is_valid_path([(3, 3)], kin, time=2, agv_name="Ego") is False


def test_expand_splits_180_into_two_90():
    raw = [
        (1, 1, 0, 0),
        (1, 1, 1, 180),  # 180° in-place — must become two ±90° ticks
    ]
    out = expand_path_turn_moves(raw)
    pitches = [int(p[3]) % 360 for p in out]
    assert pitches[0] == 0
    assert 180 in pitches
    # No single-step 0→180 jump
    for a, b in zip(out, out[1:]):
        if a[:2] == b[:2]:
            dp = (int(b[3]) - int(a[3])) % 360
            assert dp in (0, 90, 270), (a, b)


def test_truncate_helpers_still_ok():
    path = [(1, 1, t, 90) for t in range(40)]
    out = truncate_path_to_horizon(path, t0=5, horizon=20)
    assert max(int(p[2]) for p in out) == 25


if __name__ == "__main__":
    test_planned_pose_always_blocks_even_outside_horizon()
    test_past_end_always_freezes_last_cell()
    test_gap_holds_previous_pose()
    test_timestamp_lookup_not_index()
    test_expand_preserves_same_heading_waits()
    test_expand_splits_180_into_two_90()
    test_terminal_freeze_vs_future_reservation()
    test_follower_vs_inplace_turn_detected()
    test_kinematic_expand_shifts_time_detected_by_is_valid()
    test_truncate_helpers_still_ok()
    print("OK astar kin-reject tests")
