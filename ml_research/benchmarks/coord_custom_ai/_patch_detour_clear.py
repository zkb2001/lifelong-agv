"""Only clear corridor when serial A* path is a large detour vs Manhattan."""
from __future__ import annotations

import ast
from pathlib import Path

TARGET = Path(__file__).with_name("solve_ecbs.py")


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    old = '''    path = _try_path()
    if path is None and _depth < 6:
        # 1) Evict anyone sitting on the goal
'''
    new = '''    path = _try_path()
    # If the frozen-fleet path is a big detour, park corridor blockers once
    # and retry — but only when it actually helps (avoid always-on disperse).
    if (
        path is not None
        and _depth == 0
        and len(path) - 1 > _manh(pose[mover][:2], goal) + 10
    ):
        sx, sy = pose[mover][:2]
        tx, ty = int(goal[0]), int(goal[1])
        corridor: Set[Cell] = set()
        x, y = sx, sy
        while x != tx:
            corridor.add((x, y))
            x += 1 if tx > x else -1
        while y != ty:
            corridor.add((x, y))
            y += 1 if ty > y else -1
        corridor.add((tx, ty))
        keep0 = set(corridor) | {goal, pose[mover][:2]}
        moved_any = False
        for other in list(names):
            if other == mover:
                continue
            if pose[other][:2] not in corridor:
                continue
            park = _park_spot(other, keep0)
            if park is None:
                continue
            now, ok = _serial_move_to(
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
            moved_any = moved_any or bool(ok)
        if moved_any:
            path2 = _try_path()
            if path2 is not None and len(path2) < len(path):
                path = path2

    if path is None and _depth < 6:
        # 1) Evict anyone sitting on the goal
'''
    if text.count(old) != 1:
        raise SystemExit(f"count={text.count(old)}")
    text = text.replace(old, new, 1)
    # Ensure this sits AFTER _park_spot is defined
    if text.find("def _park_spot") > text.find("park = _park_spot(other, keep0)"):
        raise SystemExit("park_spot used before def — reorder needed")
    ast.parse(text)
    TARGET.write_text(text, encoding="utf-8")
    print("conditional detour clear ok", TARGET.stat().st_size)


if __name__ == "__main__":
    main()
