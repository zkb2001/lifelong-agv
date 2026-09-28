"""Patch solve_ecbs.py: free-cell snap + ST delivery fallback + continue on fail."""
from __future__ import annotations

from pathlib import Path

p = Path(__file__).resolve().parents[1] / "solve_ecbs.py"
text = p.read_text(encoding="utf-8")

helper = '''

def _snap_free(cell: Cell, free: Set[Cell], static: Set[Cell], prefer: Optional[Cell] = None) -> Cell:
    """Map a possibly-blocked/station cell onto a nearby free cell."""
    c = (int(cell[0]), int(cell[1]))
    if c in free:
        return c
    x, y = c
    cands = [
        (x + dx, y + dy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))
        if (x + dx, y + dy) in free
    ]
    if not cands:
        cands = list(free)
    if not cands:
        return c
    anchor = prefer or c
    return min(cands, key=lambda q: (_manh(q, anchor), q))


def _spacetime_bfs(
    start: Cell,
    goal: Cell,
    static: Set[Cell],
    reserved: Dict[Tuple[int, Cell], str],
    who: str,
    *,
    tmax: int = 240,
) -> Optional[List[Cell]]:
    if start == goal:
        return [start]
    blocked = set(static)
    parent: Dict[Tuple[int, Cell], Optional[Tuple[int, Cell]]] = {(0, start): None}
    q = deque([(0, start)])
    found = None
    while q:
        t, cur = q.popleft()
        if cur == goal:
            found = (t, cur)
            break
        if t >= tmax:
            continue
        t1 = t + 1
        x, y = cur
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                continue
            if nxt in blocked and nxt != goal:
                continue
            occ = reserved.get((t1, nxt))
            if occ is not None and occ != who:
                continue
            key = (t1, nxt)
            if key in parent:
                continue
            parent[key] = (t, cur)
            q.append(key)
    if found is None:
        return None
    rev: List[Cell] = []
    curk: Optional[Tuple[int, Cell]] = found
    while curk is not None:
        rev.append(curk[1])
        curk = parent[curk]
    rev.reverse()
    return rev


def _prioritized_st_paths(
    starts: Dict[str, Cell],
    goals: Dict[str, Cell],
    movers: Set[str],
    static: Set[Cell],
) -> Optional[Dict[str, List[Cell]]]:
    reserved: Dict[Tuple[int, Cell], str] = {}
    out: Dict[str, List[Cell]] = {n: [starts[n]] for n in starts}
    order = sorted(movers, key=lambda n: (_manh(starts[n], goals[n]), n))
    for n in order:
        path = _spacetime_bfs(starts[n], goals[n], static, reserved, n)
        if path is None:
            return None
        out[n] = path
        for i, c in enumerate(path):
            reserved[(i, c)] = n
        for t in range(len(path), len(path) + 8):
            reserved[(t, path[-1])] = n
    T = max((len(out[n]) for n in movers), default=1)
    for n in starts:
        while len(out[n]) < T:
            out[n].append(out[n][-1])
    return out

'''

if "_snap_free" not in text:
    needle = "def _ecbs("
    if needle not in text:
        raise SystemExit("no _ecbs")
    text = text.replace(needle, helper + "\n\ndef _ecbs(", 1)

# pickup goals
text = text.replace(
    "goals[agv] = tuple(task[\"pickup_point\"])",
    "goals[agv] = _snap_free(tuple(task[\"pickup_point\"]), free, static, prefer=starts[agv])",
)

text = text.replace(
    """        if paths is None:
            for agv, task in assigned.items():
                failed.append(str(task["task_id"]))
                _requeue(queues, order, task)
            print(f"[ECBS] FAIL pickup wave @t={now}", flush=True)
            break
""",
    """        if paths is None:
            for agv, task in assigned.items():
                failed.append(str(task["task_id"]))
            print(f"[ECBS] FAIL pickup wave @t={now} (drop wave, continue)", flush=True)
            continue
""",
)

old_del_start = "        # ---- delivery ----"
idx = text.find(old_del_start)
if idx < 0:
    raise SystemExit("delivery marker missing")
# find next print done=
idx2 = text.find('        print(\n            f"[ECBS] done=', idx)
if idx2 < 0:
    idx2 = text.find('print(\n            f"[ECBS] done=', idx)
if idx2 < 0:
    # try single-line
    idx2 = text.find('[ECBS] done={done}/{total}', idx)
    # back up to start of print statement
    idx2 = text.rfind("        print(", idx, idx2)
if idx2 < 0:
    raise SystemExit("done print missing")

new_del = '''        # ---- delivery ----
        starts = {n: pose[n][:2] for n in names}
        goals = {n: starts[n] for n in names}
        deliverable: Dict[str, dict] = {}
        for agv, task in assigned.items():
            ends = [tuple(e) for e in (task.get("end_points") or [])]
            ends = [_snap_free(c, free, static, prefer=starts[agv]) for c in ends]
            ends = list(dict.fromkeys(ends))
            if not ends:
                failed.append(str(task["task_id"]))
                loaded[agv] = False
                continue
            goals[agv] = min(ends, key=lambda c: (_manh(starts[agv], c), c))
            deliverable[agv] = task

        if not deliverable:
            print(f"[ECBS] no deliverable goals @t={now}", flush=True)
            continue

        active = set(deliverable.keys())
        paths = _ecbs(
            grid,
            starts,
            goals,
            weight=weight,
            time_limit=time_limit,
            max_expansions=max_expansions,
            movers=active,
            plan=plan,
        )
        if paths is None:
            items = list(deliverable.items())
            for k_try in range(len(items), 0, -1):
                keep = dict(items[:k_try])
                g2 = {n: starts[n] for n in names}
                for a, _tsk in keep.items():
                    g2[a] = goals[a]
                paths = _ecbs(
                    grid,
                    starts,
                    g2,
                    weight=weight,
                    time_limit=float(time_limit) * 1.5,
                    max_expansions=max_expansions,
                    movers=set(keep),
                    plan=plan,
                )
                if paths is not None:
                    deliverable = keep
                    goals = g2
                    break
        if paths is None:
            paths = _prioritized_st_paths(starts, goals, set(deliverable), static)
        if paths is None:
            paths = {n: [starts[n]] for n in names}
            for agv in list(deliverable):
                solo_goals = {n: starts[n] for n in names}
                solo_goals[agv] = goals[agv]
                sp = _prioritized_st_paths(starts, solo_goals, {agv}, static)
                if sp is None:
                    failed.append(str(deliverable[agv]["task_id"]))
                    loaded[agv] = False
                    steps_by[agv][-1]["loaded"] = "FALSE"
                    steps_by[agv][-1]["destination"] = ""
                    steps_by[agv][-1]["task-id"] = ""
                    print(
                        f"[ECBS] FAIL delivery {deliverable[agv]['task_id']} @t={now}",
                        flush=True,
                    )
                    del deliverable[agv]
                    continue
                paths[agv] = sp[agv]
            if not deliverable:
                continue

        timelines = _plan_paths_as_poses(pose, paths)
        now = _apply_pose_timelines(
            steps_by, pose, timelines, now, loaded=loaded, dest=dest, tid=tid
        )
        for agv in list(deliverable):
            tid_s = str(deliverable[agv]["task_id"])
            if tid_s in failed:
                continue
            loaded[agv] = False
            steps_by[agv][-1]["loaded"] = "FALSE"
            steps_by[agv][-1]["destination"] = ""
            steps_by[agv][-1]["task-id"] = ""
            done += 1

'''

text = text[:idx] + new_del + text[idx2:]
p.write_text(text, encoding="utf-8")
print("patched", p.stat().st_size)
