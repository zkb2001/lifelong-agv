def _tighten_paths(
    paths: Dict[str, List[Cell]],
    movers: Set[str],
    static: Set[Cell],
) -> Dict[str, List[Cell]]:
    """Shorten mover paths via BFS / spacetime when the joint timeline stays safe."""
    out = {n: list(p) for n, p in paths.items()}

    def _pad(d: Dict[str, List[Cell]]) -> Dict[str, List[Cell]]:
        if not d:
            return d
        T = max(len(pp) for pp in d.values())
        pad = {n: list(pp) for n, pp in d.items()}
        for n in pad:
            while len(pad[n]) < T:
                pad[n].append(pad[n][-1])
        return pad

    # Pass 1-2: static BFS shortcut (longest first)
    for _ in range(2):
        order = sorted(movers, key=lambda n: (-len(out.get(n) or []), n))
        progressed = False
        for name in order:
            p = out.get(name) or []
            if len(p) < 3:
                continue
            extra = set()
            for m, mp in out.items():
                if m == name or not mp:
                    continue
                extra.add(mp[0])
                extra.add(mp[-1])
            for extra_block in (extra, set()):
                short = _shortcut_path(p, static, extra_block=extra_block)
                if short is p or len(short) >= len(p):
                    continue
                trial = dict(out)
                trial[name] = short
                if not _paths_conflict(_pad(trial)):
                    out[name] = short
                    progressed = True
                    break
        if not progressed:
            break

    # Pass 3: spacetime A* toward goal in others' reservation
    order = sorted(movers, key=lambda n: (-len(out.get(n) or []), n))
    for name in order:
        p = out.get(name) or []
        if len(p) < 3:
            continue
        start, goal = p[0], p[-1]
        others = [out[m] for m in out if m != name and out[m]]
        tmax = max(len(p) + 8, max((len(o) for o in others), default=0) + 8)
        short = _spacetime_bfs(start, goal, static, others, tmax=tmax)
        if short and _path_unit_steps(short) and len(short) < len(p):
            trial = dict(out)
            trial[name] = short
            if not _paths_conflict(_pad(trial)):
                out[name] = short
    return out


def _spacetime_bfs(
    start: Cell,
    goal: Cell,
    static: Set[Cell],
    other_paths: List[List[Cell]],
    *,
    tmax: int,
) -> List[Cell]:
    """Earliest-arrival path avoiding vertex/swap conflicts with others."""

    def other_at(op: List[Cell], t: int) -> Cell:
        if not op:
            return start
        return op[t] if t < len(op) else op[-1]

    def blocked(cell: Cell, t: int, prev: Cell) -> bool:
        if cell in static and cell != goal and cell != start:
            return True
        for op in other_paths:
            if other_at(op, t) == cell:
                return True
            if t > 0 and other_at(op, t) == prev and other_at(op, t - 1) == cell:
                return True
        return False

    parent: Dict[Tuple[Cell, int], Optional[Tuple[Cell, int]]] = {}
    q: deque = deque()
    parent[(start, 0)] = None
    q.append((start, 0))
    found: Optional[Tuple[Cell, int]] = None
    while q:
        cur, t = q.popleft()
        if cur == goal:
            found = (cur, t)
            break
        if t >= tmax:
            continue
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                continue
            t1 = t + 1
            if (nxt, t1) in parent:
                continue
            if blocked(nxt, t1, cur):
                continue
            parent[(nxt, t1)] = (cur, t)
            q.append((nxt, t1))
    if found is None:
        return []
    rev: List[Cell] = []
    node: Optional[Tuple[Cell, int]] = found
    while node is not None:
        rev.append(node[0])
        node = parent[node]
    rev.reverse()
    # Drop consecutive duplicate waits at the end only — keep mid-path waits
    return rev


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
    """Apply cell paths with competition motion: turn 1s then move 1s.

    Agents stay synchronized by cell index: for each step, everyone who needs
    a turn rotates first (others wait), then everyone moves. This preserves
    ECBS cell reservations while emitting legal pitch timelines.
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
        nxt: Dict[str, Cell] = {}
        for n in names:
            p = trimmed[n]
            nxt[n] = p[step] if step < len(p) else p[-1]

        # Skip ECBS all-wait ticks (everyone stays) — safe relative compression
        if all(nxt[n] == (pose[n][0], pose[n][1]) for n in names):
            continue

        # Turn phase
        for _guard in range(4):
            need: List[Tuple[str, int]] = []
            for n in names:
                cur = pose[n]
                goal_c = nxt[n]
                if goal_c == (cur[0], cur[1]):
                    continue
                dx, dy = goal_c[0] - cur[0], goal_c[1] - cur[1]
                if abs(dx) + abs(dy) != 1:
                    continue
                tp = _required_pitch(dx, dy)
                if int(cur[2]) % 360 != tp:
                    need.append((n, tp))
            if not need:
                break
            for n, tp in need:
                x, y, _ = pose[n]
                pose[n] = (x, y, tp)
            _emit()

        # Move phase
        for n in names:
            x, y = nxt[n]
            pitch = pose[n][2]
            if (x, y) != (pose[n][0], pose[n][1]):
                dx, dy = x - pose[n][0], y - pose[n][1]
                if abs(dx) + abs(dy) == 1:
                    pitch = _required_pitch(dx, dy)
            pose[n] = (x, y, pitch)
        _emit()

    return now


def solve_ecbs(