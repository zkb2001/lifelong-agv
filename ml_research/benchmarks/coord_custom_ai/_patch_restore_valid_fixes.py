"""Restore VALIDITY fixes onto full wave solve_ecbs.py (turn ±90 + local unload pads)."""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).with_name("solve_ecbs.py")


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    orig = text

    if "_turn_pitches_toward" not in text:
        old_pitch = '''def _pitch_from_delta(dx: int, dy: int, prev: int) -> int:
    if dx == 1:
        return 0
    if dx == -1:
        return 180
    if dy == 1:
        return 90
    if dy == -1:
        return 270
    return int(prev) % 360


def _hold('''
        new_pitch = '''def _pitch_from_delta(dx: int, dy: int, prev: int) -> int:
    if dx == 1:
        return 0
    if dx == -1:
        return 180
    if dy == 1:
        return 90
    if dy == -1:
        return 270
    return int(prev) % 360


def _turn_pitches_toward(pitch: int, need: int) -> List[int]:
    """In-place ±90° turns from ``pitch`` to ``need`` (no 180° single step)."""
    cur = int(pitch) % 360
    need = int(need) % 360
    out: List[int] = []
    guard = 0
    while cur != need and guard < 4:
        guard += 1
        cw = (cur + 90) % 360
        ccw = (cur - 90) % 360
        # Prefer the shorter direction; on 180° ties prefer CW.
        d_cw = (need - cur) % 360
        d_ccw = (cur - need) % 360
        cur = cw if d_cw <= d_ccw else ccw
        out.append(cur)
    return out


def _hold('''
        if old_pitch not in text:
            raise SystemExit("pitch/hold anchor not found")
        text = text.replace(old_pitch, new_pitch, 1)
        print("patched: _turn_pitches_toward")
    else:
        print("skip: _turn_pitches_toward already present")

    old_cell = '''def _cell_path_to_pose_seq(start: Pose, cells: List[Cell]) -> List[Pose]:
    """Expand a cell path into turn-safe pose timeline (no move+turn same step)."""
    if not cells:
        return []
    seq: List[Pose] = []
    x, y, pitch = int(start[0]), int(start[1]), int(start[2]) % 360
    # Align to first cell if needed
    if cells[0] != (x, y):
        cells = [(x, y)] + list(cells)
    for i in range(1, len(cells)):
        nx, ny = cells[i]
        if (nx, ny) == (x, y):
            seq.append((x, y, pitch))
            continue
        need = _pitch_from_delta(nx - x, ny - y, pitch)
        if need != pitch:
            seq.append((x, y, need))
            pitch = need
        seq.append((nx, ny, pitch))
        x, y = nx, ny
    return seq'''
    new_cell = '''def _cell_path_to_pose_seq(start: Pose, cells: List[Cell]) -> List[Pose]:
    """Expand a cell path into turn-safe pose timeline (no move+turn / turn>90)."""
    if not cells:
        return []
    seq: List[Pose] = []
    x, y, pitch = int(start[0]), int(start[1]), int(start[2]) % 360
    # Align to first cell if needed
    if cells[0] != (x, y):
        cells = [(x, y)] + list(cells)
    for i in range(1, len(cells)):
        nx, ny = cells[i]
        if (nx, ny) == (x, y):
            seq.append((x, y, pitch))
            continue
        need = _pitch_from_delta(nx - x, ny - y, pitch)
        if need != pitch:
            for p in _turn_pitches_toward(pitch, need):
                seq.append((x, y, p))
                pitch = p
        seq.append((nx, ny, pitch))
        x, y = nx, ny
    return seq'''
    if old_cell in text:
        text = text.replace(old_cell, new_cell, 1)
        print("patched: _cell_path_to_pose_seq")
    elif "no move+turn / turn>90" in text:
        print("skip: _cell_path_to_pose_seq already patched")
    else:
        raise SystemExit("_cell_path_to_pose_seq anchor not found")

    old_exp = '''        need_turn: Dict[str, int] = {}
        for n in names:
            x, y, pitch = cur[n]
            nx, ny = nxt_cell[n]
            if (nx, ny) == (x, y):
                continue
            need = _pitch_from_delta(nx - x, ny - y, pitch)
            if need != int(pitch) % 360:
                need_turn[n] = need
        if need_turn:
            for n in names:
                x, y, pitch = cur[n]
                if n in need_turn:
                    cur[n] = (x, y, need_turn[n])
                out[n].append(cur[n])
        for n in names:
            x, y, pitch = cur[n]
            nx, ny = nxt_cell[n]
            cur[n] = (nx, ny, pitch)
            out[n].append(cur[n])'''
    new_exp = '''        need_turn: Dict[str, int] = {}
        for n in names:
            x, y, pitch = cur[n]
            nx, ny = nxt_cell[n]
            if (nx, ny) == (x, y):
                continue
            need = _pitch_from_delta(nx - x, ny - y, pitch)
            if need != int(pitch) % 360:
                need_turn[n] = need
        # Competition kinematics: only ±90° in-place per tick (180° = two ticks).
        while need_turn:
            progressed = False
            for n in names:
                x, y, pitch = cur[n]
                if n in need_turn:
                    steps = _turn_pitches_toward(pitch, need_turn[n])
                    if steps:
                        cur[n] = (x, y, steps[0])
                        progressed = True
                        if int(cur[n][2]) % 360 == int(need_turn[n]) % 360:
                            need_turn.pop(n, None)
                out[n].append(cur[n])
            if not progressed:
                break
        for n in names:
            x, y, pitch = cur[n]
            nx, ny = nxt_cell[n]
            cur[n] = (nx, ny, pitch)
            out[n].append(cur[n])'''
    if old_exp in text:
        text = text.replace(old_exp, new_exp, 1)
        print("patched: sync turn expander")
    elif "Competition kinematics: only ±90" in text or "Competition kinematics: only" in text:
        print("skip: sync turn expander already patched")
    else:
        raise SystemExit("sync expander anchor not found")

    old_pads = '''def _valid_unload_pads(
    task: dict,
    free: Set[Cell],
    static: Set[Cell],
    *,
    prefer: Optional[Cell] = None,
) -> List[Cell]:
    """Unload pads that are walkable (never maze walls / stations)."""
    raw = [tuple(e) for e in (task.get("end_points") or []) if e is not None]
    pads: List[Cell] = []
    seen: Set[Cell] = set()
    for e in raw:
        if e in static:
            continue
        c = e if e in free else _snap_free(e, free, static, prefer=prefer)
        if prefer is not None:
            c = _snap_reachable(prefer, c, free, static)
        if c in free and c not in seen and c not in static:
            pads.append(c)
            seen.add(c)
    if pads:
        return pads
    # Last resort: snap around destination name neighbors / raw ends
    for e in raw:
        c = _snap_free(e, free, static, prefer=prefer)
        if prefer is not None:
            c = _snap_reachable(prefer, c, free, static)
        if c in free and c not in seen and c not in static:
            pads.append(c)
            seen.add(c)
    return pads'''
    new_pads = '''def _valid_unload_pads(
    task: dict,
    free: Set[Cell],
    static: Set[Cell],
    *,
    prefer: Optional[Cell] = None,
) -> List[Cell]:
    """Unload pads that are walkable (never maze walls / stations).

    Only return the task's end_points (and their immediate 4-neighbors if the
    raw ends are blocked). Do NOT BFS-snap to distant free cells — that made
    SH10 unload off the validator's dropoff ring → premature_unload.
    """
    del prefer  # kept for API compat; snapping-to-prefer caused false pads
    raw = [tuple(e) for e in (task.get("end_points") or []) if e is not None]
    pads: List[Cell] = []
    seen: Set[Cell] = set()

    def _try_add(c: Cell) -> None:
        if c in free and c not in static and c not in seen:
            pads.append(c)
            seen.add(c)

    for e in raw:
        _try_add((int(e[0]), int(e[1])))
    if pads:
        return pads
    # Raw ends blocked: allow immediate 4-neighbors only (matches validate ring).
    for e in raw:
        x, y = int(e[0]), int(e[1])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            _try_add((x + dx, y + dy))
    return pads'''
    if old_pads in text:
        text = text.replace(old_pads, new_pads, 1)
        print("patched: _valid_unload_pads")
    elif "Do NOT BFS-snap to distant free cells" in text:
        print("skip: _valid_unload_pads already patched")
    else:
        raise SystemExit("_valid_unload_pads anchor not found")

    if text == orig:
        print("no changes written")
        return
    bak = TARGET.with_suffix(".py.bak_pre_valid_fixes")
    if not bak.exists():
        bak.write_text(orig, encoding="utf-8")
        print(f"backup: {bak.name}")
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET} bytes={TARGET.stat().st_size}")


if __name__ == "__main__":
    main()
