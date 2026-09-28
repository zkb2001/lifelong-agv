def _apply_paths(
    steps_by: Dict[str, List[dict]],
    pose: Dict[str, Pose],
    paths: Dict[str, List[Cell]],
    now: int,
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
) -> int:
    """Apply cell paths with legal turn/move and ECBS cell-index sync.

    Agents share a cell-step index (WCBS reservations). Within a step:
    - agents that need a turn may turn while others move (if cells stay free);
    - mutual swaps are applied atomically in one tick.
    """
    names = list(pose.keys())
    trimmed: Dict[str, List[Cell]] = {}
    for n in names:
        p = list(paths.get(n) or [pose[n][:2]])
        if len(p) >= 2 and p[0] == pose[n][:2]:
            p = p[1:]
        if not p:
            p = [pose[n][:2]]
        trimmed[n] = p
    T_cells = max(len(trimmed[n]) for n in names)

    def _emit() -> None:
        nonlocal now
        now += 1
        for n in names:
            steps_by[n].append(
                _hold(
                    n,
                    pose[n],
                    now,
                    loaded=loaded.get(n, False),
                    dest=dest.get(n, ""),
                    tid=tid.get(n, ""),
                )
            )

    for step in range(T_cells):
        nxt: Dict[str, Cell] = {
            n: (trimmed[n][step] if step < len(trimmed[n]) else trimmed[n][-1])
            for n in names
        }
        if all(nxt[n] == (pose[n][0], pose[n][1]) for n in names):
            continue

        for _micro in range(64):
            if all(nxt[n] == (pose[n][0], pose[n][1]) for n in names):
                break

            want: Dict[str, Tuple[str, Pose]] = {}
            for n in names:
                cur = pose[n]
                g = nxt[n]
                if g == (cur[0], cur[1]):
                    want[n] = ("wait", cur)
                    continue
                dx, dy = g[0] - cur[0], g[1] - cur[1]
                if abs(dx) + abs(dy) != 1:
                    want[n] = ("wait", cur)
                    continue
                tp = _required_pitch(dx, dy)
                if int(cur[2]) % 360 != tp:
                    want[n] = ("turn", (cur[0], cur[1], tp))
                else:
                    want[n] = ("move", (g[0], g[1], tp))

            accepted: Dict[str, Pose] = {n: pose[n] for n in names}
            kind_acc: Dict[str, str] = {n: "wait" for n in names}

            # 1) Atomic swaps (both must move same tick)
            used: Set[str] = set()
            for i, a in enumerate(names):
                if a in used or want[a][0] != "move":
                    continue
                ca0 = (pose[a][0], pose[a][1])
                ca1 = (want[a][1][0], want[a][1][1])
                for b in names[i + 1 :]:
                    if b in used or want[b][0] != "move":
                        continue
                    cb0 = (pose[b][0], pose[b][1])
                    cb1 = (want[b][1][0], want[b][1][1])
                    if ca0 == cb1 and cb0 == ca1:
                        accepted[a] = want[a][1]
                        accepted[b] = want[b][1]
                        kind_acc[a] = kind_acc[b] = "move"
                        used.add(a)
                        used.add(b)
                        break

            # 2) Stayers (wait/turn) claim current cells
            occ: Dict[Cell, str] = {}
            for n in names:
                if n in used:
                    c = (accepted[n][0], accepted[n][1])
                    occ[c] = n
                    continue
                if want[n][0] in ("wait", "turn"):
                    accepted[n] = want[n][1]
                    kind_acc[n] = want[n][0]
                    occ[(accepted[n][0], accepted[n][1])] = n
                    used.add(n)

            # 3) Remaining movers into free cells
            for n in names:
                if n in used or want[n][0] != "move":
                    continue
                c_new = (want[n][1][0], want[n][1][1])
                if c_new in occ:
                    # stay
                    c_old = (pose[n][0], pose[n][1])
                    occ.setdefault(c_old, n)
                    continue
                accepted[n] = want[n][1]
                kind_acc[n] = "move"
                occ[c_new] = n
                used.add(n)

            # Ensure every agent occupies something in occ
            for n in names:
                c = (accepted[n][0], accepted[n][1])
                occ.setdefault(c, n)

            if all(accepted[n] == pose[n] for n in names):
                # No progress: force classic turn-then-move for one turner
                turned = False
                for n in names:
                    if want[n][0] == "turn":
                        pose[n] = want[n][1]
                        turned = True
                if turned:
                    _emit()
                    continue
                # Force all ready moves simultaneously (ECBS atomic)
                for n in names:
                    if want[n][0] == "move":
                        pose[n] = want[n][1]
                _emit()
                continue

            for n in names:
                pose[n] = accepted[n]
            _emit()
        else:
            raise RuntimeError(f"apply_paths cell-step stall @step={step}")

    return now


def solve_ecbs(