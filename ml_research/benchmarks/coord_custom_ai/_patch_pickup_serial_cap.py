"""Cap pickup serial recover to 1 AGV to stop makespan stacking."""
from __future__ import annotations

from pathlib import Path

SOLVE = Path(__file__).resolve().parent / "solve_ecbs.py"


def main() -> None:
    text = SOLVE.read_text(encoding="utf-8")

    # Cap both shrink+serial keep sizes to 1.
    old_keep = (
        "_keep_n = max(2, min(int(wave_hard_cap) if int(wave_hard_cap) > 0 else 4, len(ranked)))"
    )
    new_keep = (
        "_keep_n = 1  # was hard_cap; multi-serial stacked SH02 sim_t to 55k"
    )
    n = text.count(old_keep)
    if n < 1:
        raise SystemExit(f"keep_n anchor count={n}")
    text = text.replace(old_keep, new_keep)

    # After pickup fail -> serial A*: only move 1, defer rest.
    old = '''                if paths is None:
                    kept = {}
                    for agv, task in list(assigned.items()):
                        goal = _pickup_goal(task, pose[agv][:2], free, static)
                        now, ok = _serial_move_to(
'''
    # Need more context - read if unique
    if text.count(old) != 1:
        raise SystemExit(f"pickup serial loop anchor count={text.count(old)}")

    # Replace the whole for-loop block that serials all assigned
    start = text.find(old)
    # Find end: pickup_serial_done = True after this loop - search forward
    chunk = text[start : start + 2500]
    # Look for pattern ending with pickup_serial_done = True after the for loop
    marker = "                    pickup_serial_done = True"
    # There may be multiple - find within this fail branch
    # Simpler approach: change the for to only first item
    old_for = '''                if paths is None:
                    kept = {}
                    for agv, task in list(assigned.items()):
                        goal = _pickup_goal(task, pose[agv][:2], free, static)
                        now, ok = _serial_move_to(
'''
    new_for = '''                if paths is None:
                    kept = {}
                    # Serial at most 1 pickup; defer the rest (anti-makespan).
                    _items = list(assigned.items())
                    for agv, task in _items[1:]:
                        pending_pickup[agv] = task
                    for agv, task in _items[:1]:
                        goal = _pickup_goal(task, pose[agv][:2], free, static)
                        now, ok = _serial_move_to(
'''
    if old_for not in text:
        raise SystemExit("pickup serial for-anchor not found")
    text = text.replace(old_for, new_for, 1)

    SOLVE.write_text(text, encoding="utf-8")
    print(f"ok keep_n_replaced={n} size={SOLVE.stat().st_size}")


if __name__ == "__main__":
    main()
