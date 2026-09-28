"""SH_custom_* lifelong MAPD via windowed ECBS (pymapf WeightedCBS).

One-shot ECBS cannot eat 100 pickup–delivery pairs at once. We roll:
  assign ≤K free AGVs → ECBS to pickups (idles goal=stay) → ECBS to unloads → repeat.

Grid cells in pymapf are (row, col) = (y-1, x-1) for warehouse (x,y) in 1..20.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pymapf import Agent, GridMap, MAPFProblem, solve

from ml_research.benchmarks.common import TRAJ_HEADER, load_scenario, patch_extra_obstacles
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    format_validation_summary,
    validate_hybrid_trajectory,
)
from ml_research.common.paths import RESULTS

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]
OUT = RESULTS / "coord_custom_ai" / "ecbs"
TRAJ = OUT / "trajectories"


def _wh_to_rc(x: int, y: int) -> Cell:
    return (y - 1, x - 1)


def _rc_to_wh(r: int, c: int) -> Cell:
    return (c + 1, r + 1)


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _pitch_from_delta(dx: int, dy: int, prev: int) -> int:
    if dx == 1:
        return 0
    if dx == -1:
        return 180
    if dy == 1:
        return 90
    if dy == -1:
        return 270
    return int(prev) % 360


def _hold(name: str, pose: Pose, t: int, *, loaded: bool = False, dest: str = "", tid: str = "") -> dict:
    return {
        "timestamp": t,
        "name": name,
        "X": pose[0],
        "Y": pose[1],
        "pitch": pose[2],
        "loaded": "TRUE" if loaded else "FALSE",
        "destination": dest,
        "Emergency": "FALSE",
        "task-id": tid,
    }


def _build_grid(static: Set[Cell], stations: Set[Cell], W: int = 20, H: int = 20) -> GridMap:
    grid = []
    for y in range(1, H + 1):
        row = []
        for x in range(1, W + 1):
            blocked = (x, y) in static or (x, y) in stations
            row.append(1 if blocked else 0)
        grid.append(row)
    return GridMap(grid)


def _unload_cell(end_xy: Cell, free: Set[Cell], prefer: Cell) -> Cell:
    x, y = end_xy
    nbrs = [
        (x + dx, y + dy)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        if (x + dx, y + dy) in free
    ]
    if not nbrs:
        return prefer
    return min(nbrs, key=lambda c: (_manh(c, prefer), c))


def _ecbs(
    grid: GridMap,
    starts_wh: Dict[str, Cell],
    goals_wh: Dict[str, Cell],
    *,
    weight: float,
    time_limit: float,
    max_expansions: int,
) -> Optional[Dict[str, List[Cell]]]:
    agents = []
    for name in sorted(starts_wh):
        sr, sc = _wh_to_rc(*starts_wh[name])
        gr, gc = _wh_to_rc(*goals_wh[name])
        if not grid.is_free((sr, sc)) or not grid.is_free((gr, gc)):
            return None
        agents.append(Agent(name=name, start=(sr, sc), goal=(gr, gc)))
    problem = MAPFProblem(grid, agents)
    sol = solve(
        problem,
        "wcbs",
        weight=float(weight),
        time_limit=float(time_limit),
        max_expansions=int(max_expansions),
    )
    if sol is None or not sol.paths:
        return None
    out: Dict[str, List[Cell]] = {}
    for name, path_rc in sol.paths.items():
        out[name] = [_rc_to_wh(r, c) for r, c in path_rc]
    return out


def _pad_paths(paths: Dict[str, List[Cell]], names: List[str], starts: Dict[str, Cell]) -> Dict[str, List[Cell]]:
    """Pad all paths to same length with waits at end."""
    for n in names:
        if n not in paths or not paths[n]:
            paths[n] = [starts[n]]
    T = max(len(p) for p in paths.values())
    for n in names:
        p = paths[n]
        while len(p) < T:
            p.append(p[-1])
    return paths


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
    names = list(pose.keys())
    paths = _pad_paths(paths, names, {n: pose[n][:2] for n in names})
    T = len(next(iter(paths.values())))
    for k in range(T):
        t = now + k
        for n in names:
            x, y = paths[n][k]
            prev = pose[n]
            if k == 0:
                pitch = prev[2]
            else:
                px, py = paths[n][k - 1]
                pitch = _pitch_from_delta(x - px, y - py, prev[2])
            # turn-only: if moved diagonally illegally shouldn't happen; if wait keep pitch
            if (x, y) == (prev[0], prev[1]) and k > 0:
                pitch = prev[2]
            pose[n] = (x, y, pitch)
            steps_by[n].append(
                _hold(
                    n,
                    pose[n],
                    t,
                    loaded=loaded.get(n, False),
                    dest=dest.get(n, ""),
                    tid=tid.get(n, ""),
                )
            )
    return now + max(0, T - 1)


def solve_ecbs(
    slot: int = 3,
    *,
    max_tasks: int = 0,
    max_active: int = 4,
    weight: float = 1.5,
    time_limit: float = 20.0,
    max_expansions: int = 200_000,
) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    TRAJ.mkdir(parents=True, exist_ok=True)
    meta = load_custom_meta(slot, max_sim_time=300000, wall_timeout=1e9)
    assert meta, f"slot {slot} meta missing"

    mod, env, agv_states, task_states, _ = load_scenario(
        Path(meta["task_csv"]), Path(meta["position_csv"]), force_reload=True
    )
    if meta.get("extra_obstacles"):
        patch_extra_obstacles(env, list(meta["extra_obstacles"]))

    queues = {n: [dict(t) for t in q] for n, q in task_states.items() if q}
    if max_tasks > 0:
        kept, nq = 0, {k: [] for k in queues}
        while kept < max_tasks:
            prog = False
            for n in list(queues):
                if queues[n] and kept < max_tasks:
                    nq[n].append(queues[n].pop(0))
                    kept += 1
                    prog = True
            if not prog:
                break
        queues = {k: v for k, v in nq.items() if v}

    total = sum(len(v) for v in queues.values())
    static_list = list(env.get_static_obstacles() or [])
    static = {(int(p[0]), int(p[1])) for p in static_list}
    j = json.loads(Path(meta["map_json"]).read_text(encoding="utf-8"))
    stations = {(int(p["x"]), int(p["y"])) for p in (j.get("pickups") or []) + (j.get("dropoffs") or [])}
    free = {(x, y) for x in range(1, 21) for y in range(1, 21) if (x, y) not in static | stations}
    grid = _build_grid(static, stations)

    pose: Dict[str, Pose] = {}
    for name, agv in agv_states.items():
        st = agv["state"] if isinstance(agv, dict) else agv.state
        pose[name] = (int(st[0]), int(st[1]), int(st[3]) if len(st) > 3 else 90)

    names = sorted(pose)
    steps_by = {n: [_hold(n, pose[n], 0)] for n in names}
    now = 0
    done = 0
    failed: List[str] = []
    t0 = time.perf_counter()
    order = sorted(queues.keys())
    rr = 0

    print(
        f"[ECBS] SH slot={slot} agents={len(names)} tasks={total} "
        f"max_active={max_active} w={weight}",
        flush=True,
    )

    while any(queues.values()):
        # --- assign wave ---
        assigned: Dict[str, dict] = {}
        free_agvs = [n for n in names if n not in assigned]
        picks = 0
        for _ in range(len(order) * 4):
            if picks >= max_active or not free_agvs:
                break
            st = None
            task = None
            for k in range(len(order)):
                cand = order[(rr + k) % len(order)]
                if queues.get(cand):
                    st, task = cand, queues[cand][0]
                    rr = (rr + k + 1) % max(1, len(order))
                    break
            if not task:
                break
            # nearest free AGV to pickup
            pk = tuple(task["pickup_point"])
            agv = min(free_agvs, key=lambda n: (_manh(pose[n][:2], pk), n))
            queues[st].pop(0)
            if st in queues and not queues[st]:
                del queues[st]
            assigned[agv] = task
            free_agvs.remove(agv)
            picks += 1

        if not assigned:
            break

        # --- phase pickup ---
        starts = {n: pose[n][:2] for n in names}
        goals = {n: starts[n] for n in names}
        for agv, task in assigned.items():
            goals[agv] = tuple(task["pickup_point"])

        paths = None
        k_try = len(assigned)
        while k_try >= 1 and paths is None:
            # if shrinking: only first k_try workers move; others stay; rest of assigned go back to queues
            if k_try < len(assigned):
                # requeue surplus
                keep = dict(list(assigned.items())[:k_try])
                for agv, task in list(assigned.items())[k_try:]:
                    st = task.get("pickup_name") or task.get("start_point")
                    # recover station from task_id prefix
                    st = str(task["task_id"]).rsplit("-", 1)[0]
                    queues.setdefault(st, []).insert(0, task)
                    if st not in order:
                        order.append(st)
                assigned = keep
                goals = {n: starts[n] for n in names}
                for agv, task in assigned.items():
                    goals[agv] = tuple(task["pickup_point"])
            paths = _ecbs(
                grid,
                starts,
                goals,
                weight=weight,
                time_limit=time_limit,
                max_expansions=max_expansions,
            )
            if paths is None:
                k_try -= 1

        if paths is None:
            for agv, task in assigned.items():
                failed.append(str(task["task_id"]))
                print(f"[ECBS] FAIL pickup wave {task['task_id']}", flush=True)
            break

        loaded = {n: False for n in names}
        dest = {n: "" for n in names}
        tid = {n: "" for n in names}
        for agv, task in assigned.items():
            tid[agv] = str(task["task_id"])
        now = _apply_paths(steps_by, pose, paths, now, loaded=loaded, dest=dest, tid=tid)
        for agv, task in assigned.items():
            loaded[agv] = True
            dest[agv] = str(task.get("destination") or "")
            # mark arrive pickup (already at goal)
            steps_by[agv][-1]["loaded"] = "TRUE"
            steps_by[agv][-1]["destination"] = dest[agv]
            steps_by[agv][-1]["task-id"] = tid[agv]

        # --- phase delivery ---
        starts = {n: pose[n][:2] for n in names}
        goals = {n: starts[n] for n in names}
        for agv, task in assigned.items():
            ends = [tuple(e) for e in (task.get("end_points") or [])]
            if not ends:
                # fallback: destination name → station cell neighbors
                failed.append(str(task["task_id"]))
                continue
            # end_points already walkable unload cells from engine
            goals[agv] = min(ends, key=lambda c: (_manh(starts[agv], c), c))

        paths = _ecbs(
            grid,
            starts,
            goals,
            weight=weight,
            time_limit=time_limit,
            max_expansions=max_expansions,
        )
        if paths is None:
            # retry one-by-one for deliveries
            paths = {n: [starts[n]] for n in names}
            ok_all = True
            for agv in list(assigned):
                solo_goals = {n: starts[n] for n in names}
                solo_goals[agv] = goals[agv]
                sp = _ecbs(
                    grid,
                    starts,
                    solo_goals,
                    weight=weight,
                    time_limit=time_limit,
                    max_expansions=max_expansions,
                )
                if sp is None:
                    failed.append(str(assigned[agv]["task_id"]))
                    ok_all = False
                    break
                paths[agv] = sp[agv]
            if not ok_all:
                print(f"[ECBS] FAIL delivery wave @t={now}", flush=True)
                break

        now = _apply_paths(steps_by, pose, paths, now, loaded=loaded, dest=dest, tid=tid)
        for agv in assigned:
            # drop
            loaded[agv] = False
            steps_by[agv][-1]["loaded"] = "FALSE"
            steps_by[agv][-1]["destination"] = ""
            steps_by[agv][-1]["task-id"] = ""
            done += 1

        print(
            f"[ECBS] done={done}/{total} wave={len(assigned)} t={now} "
            f"wall={time.perf_counter()-t0:.1f}s",
            flush=True,
        )

    # fill missing timestamps per agent for dense traj
    for name in steps_by:
        by = {int(s["timestamp"]): s for s in steps_by[name]}
        last = by[min(by)]
        filled = []
        for t in range(0, now + 1):
            if t in by:
                last = by[t]
                filled.append(dict(last))
            else:
                r = dict(last)
                r["timestamp"] = t
                filled.append(r)
        steps_by[name] = filled

    sid = str(meta["id"])
    traj = TRAJ / f"{sid}_ecbs_w{weight}_k{max_active}.csv"
    rows = []
    for n in sorted(steps_by):
        rows.extend(steps_by[n])
    rows.sort(key=lambda r: (int(r["timestamp"]), r["name"]))
    with traj.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRAJ_HEADER})

    val = validate_hybrid_trajectory(meta, traj)
    iss = val.get("issues") or {}
    rep = {
        "scenario_id": sid,
        "method": "windowed_ecbs_wcbs",
        "weight": weight,
        "max_active": max_active,
        "tasks_total": total,
        "tasks_completed": done,
        "tasks_failed": failed,
        "completion_ratio": round(done / max(1, total), 4),
        "sim_time": now,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "trajectory": str(traj),
        "validate_ok": bool(val.get("ok")),
        "validate_summary": format_validation_summary(val),
        "n_collisions": int(iss.get("n_collisions") or 0),
        "n_swaps": int(iss.get("n_swaps") or 0),
        "n_hard_wall": int(iss.get("n_hard_wall") or 0),
        "n_illegal_motion": int(iss.get("n_illegal_motion") or 0),
        "n_fifo": int(iss.get("n_fifo_violations") or 0),
    }
    (OUT / f"{sid}_ecbs.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(rep, indent=2, ensure_ascii=False), flush=True)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slot", type=int, default=3)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--max-active", type=int, default=4)
    ap.add_argument("--weight", type=float, default=1.5)
    ap.add_argument("--time-limit", type=float, default=25.0)
    ap.add_argument("--max-expansions", type=int, default=250000)
    args = ap.parse_args()
    rep = solve_ecbs(
        args.slot,
        max_tasks=args.max_tasks,
        max_active=args.max_active,
        weight=args.weight,
        time_limit=args.time_limit,
        max_expansions=args.max_expansions,
    )
    ok = (
        bool(rep.get("validate_ok"))
        and float(rep.get("completion_ratio") or 0) >= 0.999
        and not rep.get("tasks_failed")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
