"""Remove always-on corridor clear; restore exact golden _serial_move_to."""
from __future__ import annotations

import ast
from pathlib import Path

TARGET = Path(__file__).with_name("solve_ecbs.py")


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    old = '''    if pose[mover][:2] == goal:
        return now, True

    def _manhattan_corridor(src: Cell, dst: Cell) -> Set[Cell]:
        """Cells on one axis-aligned Manhattan route (horizontal then vertical)."""
        cells: Set[Cell] = set()
        x, y = int(src[0]), int(src[1])
        tx, ty = int(dst[0]), int(dst[1])
        while x != tx:
            cells.add((x, y))
            x += 1 if tx > x else -1
        while y != ty:
            cells.add((x, y))
            y += 1 if ty > y else -1
        cells.add((tx, ty))
        return cells

    def _try_path() -> Optional[List[Cell]]:
'''
    new = '''    if pose[mover][:2] == goal:
        return now, True

    def _try_path() -> Optional[List[Cell]]:
'''
    if text.count(old) != 1:
        raise SystemExit(f"corridor def count={text.count(old)}")
    text = text.replace(old, new, 1)

    # Remove depth-0 corridor eviction block
    old2 = '''    # Depth-0: nudge idlers off the short corridor so A* does not take a
    # maze-long detour around a parked fleet (k=1 / serial anti-pattern).
    if _depth == 0:
        corridor = _manhattan_corridor(pose[mover][:2], goal)
        keep0 = set(corridor) | {goal, pose[mover][:2]}
        for other in list(names):
            if other == mover:
                continue
            if pose[other][:2] not in corridor:
                continue
            park = _park_spot(other, keep0)
            if park is None:
                continue
            now, _ = _serial_move_to(
                pose,
                steps_by,
                now,
                other,
                park,
                blocked_plan,
                loaded=loaded,
                dest=dest,
                tid=tid,
                _depth=_depth + 1,
            )

    path = _try_path()
'''
    new2 = '''    path = _try_path()
'''
    if text.count(old2) != 1:
        raise SystemExit(f"corridor block count={text.count(old2)}")
    text = text.replace(old2, new2, 1)

    assert "_manhattan_corridor" not in text
    ast.parse(text)
    TARGET.write_text(text, encoding="utf-8")
    print("fully restored serial_move_to", TARGET.stat().st_size)


if __name__ == "__main__":
    main()
