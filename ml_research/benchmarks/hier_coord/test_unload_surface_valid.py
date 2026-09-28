"""Unit tests for unload / surface VALID checks."""
from __future__ import annotations

from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    _check_illegal_motion,
    _check_pickup_unload_dwell,
    _check_task_carry,
    _count_task_carry_kinds,
    format_validation_summary,
)


def test_premature_vs_foreign_unload():
    # Task A drops at (2,2); task B drops at (5,5)
    drops = {"A-1": {(2, 2), (2, 3)}, "B-1": {(5, 5), (5, 6)}}
    rows = [
        # Premature: clear loaded off any pad
        {"timestamp": 0, "name": "X", "X": 1, "Y": 1, "loaded": "TRUE", "task-id": "A-1", "destination": "Beijing"},
        {"timestamp": 1, "name": "X", "X": 1, "Y": 1, "loaded": "FALSE", "task-id": "", "destination": ""},
        # Foreign: clear on B's pad while carrying A
        {"timestamp": 10, "name": "Y", "X": 5, "Y": 4, "loaded": "TRUE", "task-id": "A-1", "destination": "Beijing"},
        {"timestamp": 11, "name": "Y", "X": 5, "Y": 5, "loaded": "FALSE", "task-id": "", "destination": ""},
        # Legal unload on own pad
        {"timestamp": 20, "name": "Z", "X": 2, "Y": 1, "loaded": "TRUE", "task-id": "A-1", "destination": "Beijing"},
        {"timestamp": 21, "name": "Z", "X": 2, "Y": 2, "loaded": "FALSE", "task-id": "", "destination": ""},
    ]
    issues = _check_task_carry(rows, drops)
    kinds = {e["kind"] for e in issues}
    assert "premature_unload" in kinds
    assert "foreign_unload" in kinds
    assert len(issues) == 2
    ck = _count_task_carry_kinds(issues)
    assert ck["premature_unload"] == 1
    assert ck["foreign_unload"] == 1


def test_summary_labels_surface_and_unload():
    rep = {
        "ok": False,
        "issues": {
            "n_hard_wall": 0,
            "n_illegal_motion": 0,
            "n_pickup_no_dwell": 3,
            "n_unload_no_dwell": 2,
            "n_pickup_unload_dwell": 5,
            "n_collisions": 0,
            "n_swaps": 0,
            "n_premature_unload": 2,
            "n_foreign_unload": 1,
            "n_task_carry_violations": 3,
            "n_pickup_cell_violations": 0,
            "n_fifo_violations": 4,
            "n_display_mismatch": 5,
        },
    }
    s = format_validation_summary(rep)
    assert "premature_unload=2" in s
    assert "foreign_unload=1" in s
    assert "surface_fifo=4" in s
    assert "surface_head=5" in s
    assert "pickup_dwell=3" in s
    assert "unload_dwell=2" in s
    assert "action_dwell=" not in s
    # task_carry suppressed when premature+foreign cover it
    assert "task_carry=" not in s


def test_pickup_unload_dwell_and_turn_gt_90():
    # Pickup / unload while translating: illegal dwell
    rows = [
        {"timestamp": 0, "name": "A", "X": 1, "Y": 1, "pitch": 0, "loaded": "FALSE"},
        {"timestamp": 1, "name": "A", "X": 2, "Y": 1, "pitch": 0, "loaded": "TRUE", "task-id": "T1"},
        {"timestamp": 2, "name": "A", "X": 2, "Y": 1, "pitch": 0, "loaded": "TRUE", "task-id": "T1"},
        {"timestamp": 3, "name": "A", "X": 3, "Y": 1, "pitch": 0, "loaded": "FALSE"},
    ]
    dwell = _check_pickup_unload_dwell(rows)
    kinds = {e["kind"] for e in dwell}
    assert "pickup_no_dwell" in kinds
    assert "unload_no_dwell" in kinds

    # 180° in one tick is legal; a non-cardinal pitch change is not.
    turn_rows = [
        {"timestamp": 0, "name": "B", "X": 5, "Y": 5, "pitch": 0, "loaded": "FALSE"},
        {"timestamp": 1, "name": "B", "X": 5, "Y": 5, "pitch": 180, "loaded": "FALSE"},
    ]
    motion = _check_illegal_motion(turn_rows, 1, 20)
    assert not any(e.get("kind") == "turn_gt_90" for e in motion)
    bad_rows = [
        {"timestamp": 0, "name": "B", "X": 5, "Y": 5, "pitch": 0, "loaded": "FALSE"},
        {"timestamp": 1, "name": "B", "X": 5, "Y": 5, "pitch": 45, "loaded": "FALSE"},
    ]
    bad = _check_illegal_motion(bad_rows, 1, 20)
    assert any(e.get("kind") == "turn_gt_90" for e in bad)


if __name__ == "__main__":
    test_premature_vs_foreign_unload()
    test_summary_labels_surface_and_unload()
    test_pickup_unload_dwell_and_turn_gt_90()
    print("OK unload/surface valid tests")
