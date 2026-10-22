"""Standalone lifelong MAPD: CostTracer prior + greed–coop neural MCTS.

Lifelong MAPD shell (same as PIBT-MAPD):
  - FIFO surface heads (never reorder queues)
  - one approacher per pickup station
  - pickup on approach door + 1s no-turn dwell
  - unload on legal dropoff neighbor + 1s dwell
  - turn completes in 1s (incl. 180°); no same-tick move+turn

Motion: each AGV is an agent. Crowded steps use NeuralLocalMCTS with an
explicit greed–cooperation leaf reward (α progress − β conflict + γ yield);
distant agents follow CostTracer/teacher greedily (MATS-LP style).
Does NOT call solve_ecbs / Hier.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ml_research.benchmarks.common import TRAJ_HEADER, load_scenario, patch_extra_obstacles
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.mats_local.mcts import (
    LocalAgent,
    MCTSConfig,
    NeuralLocalMCTS,
)
from ml_research.benchmarks.mats_local.network import CostTracerNet, load_cost_tracer
from ml_research.benchmarks.mats_local.obs import apply_action
from ml_research.benchmarks.mats_local.runtime import pick_ego_crowded
from ml_research.benchmarks.mats_local.teacher import greedy_action
from ml_research.benchmarks.pibt_mapd.pibt_core import (
    bfs_dist_field,
    pitch_for_step,
    turn_pitch_chain,
)
from ml_research.common.paths import RESULTS

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]

OUT = RESULTS / "mats_local" / "standalone"
TRAJ = OUT / "trajectories"

_HUB13 = "Beijing"
_HUB13_PADS = {(7, 4), (6, 5)}

_NET_CACHE: Dict[str, Optional[CostTracerNet]] = {}


def _pickup_approach(cell: Cell) -> Cell:
    x, y = int(cell[0]), int(cell[1])
    if x <= 1:
        return (x + 1, y)
    if x >= 20:
        return (x - 1, y)
    return (x, y)


def _neighbors4(cell: Cell) -> List[Cell]:
    x, y = int(cell[0]), int(cell[1])
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _manh(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _restrict_hub13(dest: str, pads: List[Cell]) -> List[Cell]:
    if str(dest) != _HUB13:
        return pads
    kept = [p for p in pads if p in _HUB13_PADS]
    return kept if kept else list(_HUB13_PADS)


def _get_net(device: str) -> Optional[CostTracerNet]:
    key = str(device)
    if key not in _NET_CACHE:
        try:
            from ml_research.benchmarks.mats_local.network import COSTTRACER_CKPT

            _NET_CACHE[key] = (
                load_cost_tracer(device=device) if COSTTRACER_CKPT.exists() else None
            )
        except Exception:  # noqa: BLE001
            _NET_CACHE[key] = None
    return _NET_CACHE[key]


def _need_mcts(pos: Dict[str, Cell], active: List[str], radius: int = 3) -> bool:
    cells = [pos[n] for n in active]
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            if _manh(a, b) <= radius:
                return True
    return False


def _adaptive_greed_coop(
    stall: int,
    greed_w: float,
    coop_w: float,
    yield_w: float,
) -> Tuple[float, float, float]:
    """Shift toward greed under prolonged stall to break yield livelock."""
    if stall < 80:
        return greed_w, coop_w, yield_w
    if stall < 200:
        # Mild: less yielding, slightly more greed.
        return (
            min(0.85, greed_w + 0.15),
            max(0.10, coop_w - 0.10),
            max(0.0, yield_w - 0.05),
        )
    # Hard stall: almost pure personal progress + conflict hard-stop only.
    return (
        min(0.90, greed_w + 0.30),
        max(0.08, coop_w - 0.20),
        0.0,
    )


def _mats_desired(
    *,
    names: List[str],
    pose: Dict[str, Pose],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    net: Optional[CostTracerNet],
    num_simulations: int,
    k_proximal: int,
    device: str,
    seed: int,
    greed_w: float = 0.55,
    coop_w: float = 0.35,
    yield_w: float = 0.10,
    force_mcts: bool = False,
) -> Dict[str, Cell]:
    """One-step desired cells via greed–coop neural MCTS (crowded) + greedy."""
    pos = {n: pose[n][:2] for n in names}
    desired: Dict[str, Cell] = {}
    active = [n for n in names if n not in force_stay]
    agents = [
        LocalAgent(name=n, pos=pos[n], goal=goals.get(n)) for n in active
    ]

    acts: Dict[str, str] = {}
    if agents and (force_mcts or _need_mcts(pos, active)):
        ordered = pick_ego_crowded(agents)
        planner = NeuralLocalMCTS(
            net=net,
            cfg=MCTSConfig(
                num_simulations=int(num_simulations),
                horizon=4,
                k_proximal=int(k_proximal),
                greed_w=float(greed_w),
                coop_w=float(coop_w),
                yield_w=float(yield_w),
                seed=int(seed) % 997,
            ),
            device=device,
        )
        acts = planner.plan(ordered, list(blocked))

    for n in names:
        if n in force_stay:
            desired[n] = pos[n]
            continue
        others = [pos[m] for m in names if m != n]
        if n in acts:
            desired[n] = apply_action(pos[n], acts[n], blocked)
        else:
            ga = greedy_action(pos[n], goals.get(n), others, blocked)
            desired[n] = apply_action(pos[n], ga, blocked)
    return desired


def solve_standalone(
    slot: int,
    *,
    meta: Optional[dict] = None,
    task_csv: Optional[Path] = None,
    max_sim_time: int = 200000,
    wall_timeout: float = 1800.0,
    progress_every: int = 500,
    num_simulations: int = 24,
    k_proximal: int = 3,
    greed_w: float = 0.55,
    coop_w: float = 0.35,
    yield_w: float = 0.10,
    device: str = "cpu",
) -> dict:
    t0 = time.perf_counter()
    if meta is None:
        meta = load_custom_meta(
            int(slot),
            max_sim_time=int(max_sim_time),
            wall_timeout=float(wall_timeout),
            n_tasks=100,
        )
        assert meta, f"missing meta for slot {slot}"
    meta = dict(meta)
    sid = str(meta.get("id") or f"SH{int(slot):02d}_mats_standalone")
    meta["id"] = sid

    tcsv = Path(task_csv) if task_csv else Path(meta["task_csv"])
    meta["task_csv"] = str(tcsv.resolve())
    mod, env, agv_states, task_states, _ = load_scenario(
        tcsv, Path(meta["position_csv"]), force_reload=True
    )
    del mod, task_states
    if meta.get("extra_obstacles"):
        patch_extra_obstacles(env, list(meta["extra_obstacles"]))

    static_list = list(env.get_static_obstacles() or [])
    static_all = {(int(p[0]), int(p[1])) for p in static_list}

    pickups: Dict[str, Cell] = {}
    dropoffs: Dict[str, Cell] = {}
    jpath = Path(meta.get("map_json") or "")
    if jpath.is_file():
        j = json.loads(jpath.read_text(encoding="utf-8"))
        for p in j.get("pickups") or []:
            pickups[str(p["name"])] = (int(p["x"]), int(p["y"]))
        for p in j.get("dropoffs") or []:
            dropoffs[str(p["name"])] = (int(p["x"]), int(p["y"]))

    stations = set(pickups.values()) | set(dropoffs.values())
    walls = set(static_all) - set(stations)
    blocked = set(walls) | set(stations)
    free = {
        (x, y)
        for x in range(1, 21)
        for y in range(1, 21)
        if (x, y) not in blocked
    }
    dist_cache: Dict[Cell, Dict[Cell, int]] = {}

    def _field(goal: Cell) -> Dict[Cell, int]:
        if goal not in dist_cache:
            dist_cache[goal] = bfs_dist_field(goal, blocked)
        return dist_cache[goal]

    def _spath(a: Cell, b: Cell) -> int:
        field = _field(b)
        if a in field:
            return int(field[a])
        return 10_000 + _manh(a, b)

    def unload_pads(dest_name: str) -> List[Cell]:
        center = dropoffs.get(str(dest_name))
        if not center:
            return []
        pads = [
            c
            for c in _neighbors4(center)
            if 1 <= c[0] <= 20 and 1 <= c[1] <= 20 and c not in blocked
        ]
        return _restrict_hub13(dest_name, pads)

    queues: Dict[str, List[dict]] = defaultdict(list)
    with tcsv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            st = str(row.get("start_point") or "").strip()
            tid_s = str(row.get("task_id") or "").strip()
            dest_s = str(row.get("end_point") or "").strip()
            if not st or not tid_s:
                continue
            queues[st].append(
                {
                    "task_id": tid_s,
                    "start_point": st,
                    "destination": dest_s,
                    "end_point": dest_s,
                    "priority": row.get("priority") or "Normal",
                }
            )
    total = sum(len(v) for v in queues.values())

    pose: Dict[str, Pose] = {}
    home: Dict[str, Cell] = {}
    for name, agv in agv_states.items():
        st = agv["state"] if isinstance(agv, dict) else agv.state
        pose[name] = (int(st[0]), int(st[1]), int(st[3]) if len(st) > 3 else 90)
        home[name] = (int(st[0]), int(st[1]))
    names = sorted(pose)

    assigned: Dict[str, dict] = {}
    carrying: Dict[str, dict] = {}
    loaded: Dict[str, bool] = {n: False for n in names}
    dest: Dict[str, str] = {n: "" for n in names}
    tid: Dict[str, str] = {n: "" for n in names}
    wait_pri: Dict[str, float] = {n: 0.0 for n in names}
    approach_station: Dict[str, str] = {}
    force_stay: Set[str] = set()
    escape_goal: Dict[str, Cell] = {}
    escape_ttl: Dict[str, int] = {}
    face_commit: Dict[str, Tuple[int, int]] = {}

    steps: Dict[str, List[dict]] = {n: [] for n in names}
    now = 0
    done = 0
    done_seen: Set[str] = set()
    net = _get_net(device)
    n_mcts_calls = 0

    def _row(n: str, t: int) -> dict:
        x, y, p = pose[n]
        return {
            "timestamp": t,
            "name": n,
            "X": x,
            "Y": y,
            "pitch": int(p) % 360,
            "loaded": "TRUE" if loaded[n] else "FALSE",
            "destination": dest[n] if loaded[n] else "",
            "Emergency": "FALSE",
            "task-id": tid[n] if loaded[n] else "",
        }

    def _write_tick(t: int) -> None:
        for n in names:
            steps[n].append(_row(n, t))

    def _last_cell(n: str) -> Optional[Cell]:
        seq = steps[n]
        if not seq:
            return None
        r = seq[-1]
        return (int(r["X"]), int(r["Y"]))

    def _last_pitch(n: str) -> int:
        seq = steps[n]
        if not seq:
            return int(pose[n][2]) % 360
        return int(seq[-1].get("pitch") or 0) % 360

    _write_tick(0)

    def _claim() -> None:
        free_agvs = [
            n
            for n in names
            if n not in assigned and n not in carrying and not loaded[n]
        ]
        for st, agv in list(approach_station.items()):
            if agv not in assigned:
                approach_station.pop(st, None)
        occupied_stations = set(approach_station.keys())
        for st in sorted(queues.keys()):
            if not free_agvs:
                break
            if not queues.get(st):
                continue
            if st in occupied_stations:
                continue
            pad = pickups.get(st)
            if not pad:
                continue
            door = _pickup_approach(pad)
            if door in blocked:
                continue
            task = queues[st][0]
            agv = min(free_agvs, key=lambda n: (_spath(pose[n][:2], door), n))
            assigned[agv] = dict(task)
            approach_station[st] = agv
            free_agvs.remove(agv)

    def _goals_and_priority() -> Dict[str, Cell]:
        goals: Dict[str, Cell] = {}
        hot: Set[Cell] = set()
        for pad in pickups.values():
            hot.add(_pickup_approach(pad))
        for dname in dropoffs:
            hot.update(unload_pads(dname))

        for n in names:
            cell = pose[n][:2]
            if n in escape_goal and escape_ttl.get(n, 0) > 0:
                goals[n] = escape_goal[n]
                continue
            if n in carrying:
                task = carrying[n]
                pads = unload_pads(str(task.get("destination") or ""))
                goals[n] = (
                    min(pads, key=lambda c: (_spath(cell, c), c)) if pads else cell
                )
            elif n in assigned:
                task = assigned[n]
                st = str(task.get("start_point") or "")
                pad = pickups.get(st)
                goals[n] = _pickup_approach(pad) if pad else cell
            else:
                park = home.get(n, cell)
                if cell in hot:
                    cands = [
                        c for c in _neighbors4(cell) if c in free and c not in hot
                    ]
                    goals[n] = (
                        min(cands, key=lambda c: (_manh(c, park), c))
                        if cands
                        else park
                    )
                else:
                    goals[n] = park
        return goals

    def _assign_escape() -> None:
        occupied = {pose[n][:2] for n in names}
        free_cells = sorted(c for c in free if c not in occupied)
        if not free_cells:
            return
        for n in names:
            if n in force_stay:
                continue
            if n in assigned or n in carrying:
                if wait_pri.get(n, 0) < 500:
                    continue
            cell = pose[n][:2]
            far = max(free_cells, key=lambda c: (_spath(cell, c), hash((n, c, now))))
            escape_goal[n] = far
            escape_ttl[n] = 40
            face_commit.pop(n, None)

    def _try_pickup_rise() -> None:
        for agv, task in list(assigned.items()):
            st = str(task.get("start_point") or "")
            pad = pickups.get(st)
            if not pad:
                continue
            door = _pickup_approach(pad)
            cell = pose[agv][:2]
            if cell != door:
                continue
            last = _last_cell(agv)
            if last != door:
                continue
            if _last_pitch(agv) != int(pose[agv][2]) % 360:
                continue
            q = queues.get(st) or []
            if not q or str(q[0].get("task_id") or "") != str(task.get("task_id") or ""):
                continue
            q.pop(0)
            if not q:
                queues.pop(st, None)
            else:
                queues[st] = q
            tid_s = str(task.get("task_id") or "")
            pose[agv] = (door[0], door[1], _last_pitch(agv))
            loaded[agv] = True
            dest[agv] = str(task.get("destination") or "")
            tid[agv] = tid_s
            carrying[agv] = dict(task)
            assigned.pop(agv, None)
            approach_station.pop(st, None)
            wait_pri[agv] = 0.0
            force_stay.add(agv)

    def _try_unload() -> None:
        nonlocal done
        for agv, task in list(carrying.items()):
            pads = set(unload_pads(str(task.get("destination") or "")))
            cell = pose[agv][:2]
            if cell not in pads:
                continue
            last = _last_cell(agv)
            if last != cell:
                continue
            if _last_pitch(agv) != int(pose[agv][2]) % 360:
                continue
            tid_s = str(task.get("task_id") or "")
            pose[agv] = (cell[0], cell[1], _last_pitch(agv))
            loaded[agv] = False
            dest[agv] = ""
            tid[agv] = ""
            carrying.pop(agv, None)
            wait_pri[agv] = 0.0
            force_stay.add(agv)
            if tid_s and tid_s not in done_seen:
                done_seen.add(tid_s)
                done += 1

    def _apply_desired(desired: Dict[str, Cell]) -> bool:
        progressed = False
        new_pose = dict(pose)
        turned: Set[str] = set()

        for n in list(face_commit.keys()):
            pitch_c, ttl = face_commit[n]
            ttl -= 1
            if ttl <= 0:
                face_commit.pop(n, None)
            else:
                face_commit[n] = (pitch_c, ttl)

        for n in names:
            if n in force_stay:
                wait_pri[n] = float(wait_pri[n]) + 1.0
                continue
            cur = pose[n][:2]
            pitch = int(pose[n][2]) % 360
            want = desired.get(n, cur)
            if want == cur:
                wait_pri[n] = float(wait_pri[n]) + 1.0
                continue
            need = pitch_for_step(cur, want)
            if need < 0:
                wait_pri[n] = float(wait_pri[n]) + 1.0
                continue
            if pitch != need:
                if n in face_commit and face_commit[n][0] != need:
                    wait_pri[n] = float(wait_pri[n]) + 1.0
                    continue
                chain = turn_pitch_chain(pitch, need)
                if chain:
                    new_pose[n] = (cur[0], cur[1], int(chain[0]) % 360)
                    turned.add(n)
                    face_commit[n] = (need, 12)
                    progressed = True
                    wait_pri[n] = 0.0
                continue

        movers: Dict[str, Cell] = {}
        for n in names:
            if n in force_stay or n in turned:
                continue
            cur = pose[n][:2]
            pitch = int(new_pose[n][2]) % 360
            want = desired.get(n, cur)
            if want == cur:
                continue
            need = pitch_for_step(cur, want)
            if need >= 0 and pitch == need:
                movers[n] = want

        occ_owner: Dict[Cell, str] = {pose[n][:2]: n for n in names}
        final_move: Dict[str, Cell] = {}
        order = sorted(
            movers.keys(),
            key=lambda n: (
                -float(1e6 if n in carrying else 1e5 if n in assigned else 0)
                - wait_pri[n],
                n,
            ),
        )
        for n in order:
            nxt = movers[n]
            cur = pose[n][:2]
            if nxt in blocked:
                continue
            holder = occ_owner.get(nxt)
            if holder is not None and holder not in final_move:
                continue
            if holder is not None and final_move.get(holder) == cur:
                continue
            if occ_owner.get(cur) == n:
                del occ_owner[cur]
            occ_owner[nxt] = n
            final_move[n] = nxt

        for n, nxt in final_move.items():
            pitch = int(new_pose[n][2]) % 360
            new_pose[n] = (nxt[0], nxt[1], pitch)
            progressed = True
            wait_pri[n] = 0.0
            face_commit.pop(n, None)

        by_cell: Dict[Cell, List[str]] = {}
        for n in names:
            by_cell.setdefault(new_pose[n][:2], []).append(n)
        for cell, ags in by_cell.items():
            if len(ags) <= 1:
                continue
            natives = [n for n in ags if pose[n][:2] == cell]
            if natives:
                keep = sorted(
                    natives,
                    key=lambda n: (
                        -float(1e6 if n in carrying else 1e5 if n in assigned else 0)
                        - wait_pri[n],
                        n,
                    ),
                )[0]
            else:
                keep = sorted(
                    ags,
                    key=lambda n: (
                        -float(1e6 if n in carrying else 1e5 if n in assigned else 0)
                        - wait_pri[n],
                        n,
                    ),
                )[0]
            for n in ags:
                if n != keep:
                    new_pose[n] = pose[n]

        for n in names:
            pose[n] = new_pose[n]
        return progressed

    stall = 0
    last_done = 0
    while done < total and now < int(max_sim_time):
        if time.perf_counter() - t0 >= float(wall_timeout):
            break

        force_stay.clear()
        for n in list(escape_ttl.keys()):
            escape_ttl[n] = int(escape_ttl[n]) - 1
            if escape_ttl[n] <= 0:
                escape_ttl.pop(n, None)
                escape_goal.pop(n, None)

        _claim()
        _try_pickup_rise()
        _try_unload()
        if done >= total:
            now += 1
            _write_tick(now)
            break

        for agv, task in list(assigned.items()):
            st = str(task.get("start_point") or "")
            pad = pickups.get(st)
            if not pad:
                continue
            door = _pickup_approach(pad)
            if pose[agv][:2] == door:
                force_stay.add(agv)
        for agv in list(carrying.keys()):
            pads = set(unload_pads(str(carrying[agv].get("destination") or "")))
            if pose[agv][:2] in pads:
                force_stay.add(agv)

        if stall > 0 and stall % 200 == 0:
            _assign_escape()

        goals = _goals_and_priority()
        active = [n for n in names if n not in force_stay]
        pos_now = {n: pose[n][:2] for n in names}
        # Stride MCTS: every tick when fresh progress; every 3rd tick under stall
        # (greedy CostTracer fills gaps) — speeds wall clock without killing coop.
        crowded = _need_mcts(pos_now, active)
        stride = 1 if stall < 40 else 3
        run_mcts = crowded and (now % stride == 0)
        gw, cw, yw = _adaptive_greed_coop(stall, greed_w, coop_w, yield_w)
        if run_mcts:
            n_mcts_calls += 1
            desired = _mats_desired(
                names=names,
                pose=pose,
                goals=goals,
                blocked=blocked,
                force_stay=force_stay,
                net=net,
                num_simulations=num_simulations,
                k_proximal=k_proximal,
                device=device,
                seed=now,
                greed_w=gw,
                coop_w=cw,
                yield_w=yw,
                force_mcts=True,
            )
        else:
            # Pure per-AGV greed (distant-agent MATS-LP policy) for non-MCTS ticks.
            desired = {}
            for n in names:
                if n in force_stay:
                    desired[n] = pose[n][:2]
                    continue
                others = [pose[m][:2] for m in names if m != n]
                ga = greedy_action(
                    pose[n][:2], goals.get(n), others, blocked
                )
                desired[n] = apply_action(pose[n][:2], ga, blocked)
        for n in force_stay:
            desired[n] = pose[n][:2]

        _apply_desired(desired)
        now += 1
        _write_tick(now)

        if done != last_done:
            last_done = done
            stall = 0
            escape_goal.clear()
            escape_ttl.clear()
            face_commit.clear()
        else:
            stall += 1
        if stall >= 80000:
            print(
                f"[mats_standalone] stall break t={now} done={done}/{total}",
                flush=True,
            )
            break

        if progress_every > 0 and (
            now % int(progress_every) == 0 or done == total
        ):
            print(
                f"  [{sid}/mats] t={now} done={done}/{total} "
                f"carry={len(carrying)} approach={len(assigned)} "
                f"mcts={n_mcts_calls} wall={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    _try_unload()
    _try_pickup_rise()

    OUT.mkdir(parents=True, exist_ok=True)
    TRAJ.mkdir(parents=True, exist_ok=True)
    traj = TRAJ / f"{sid}_mats.csv"
    rows: List[dict] = []
    for n in names:
        rows.extend(steps[n])
    rows.sort(key=lambda r: (int(r["timestamp"]), r["name"]))
    with traj.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in TRAJ_HEADER})

    from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
        format_validation_summary,
        validate_hybrid_trajectory,
    )

    val = validate_hybrid_trajectory(meta, traj)
    iss = val.get("issues") or {}
    wall = round(time.perf_counter() - t0, 2)
    rep = {
        "scenario_id": sid,
        "method": "mats_greed_coop_mcts_v1",
        "slot": int(slot),
        "tasks_total": total,
        "tasks_completed": done,
        "completion_ratio": round(done / max(1, total), 4),
        "sim_time": now,
        "wall_seconds": wall,
        "trajectory": str(traj),
        "validate_ok": bool(val.get("ok")),
        "validate_summary": format_validation_summary(val),
        "n_collisions": int(iss.get("n_collisions") or 0),
        "n_swaps": int(iss.get("n_swaps") or 0),
        "n_illegal_motion": int(iss.get("n_illegal_motion") or 0),
        "n_fifo": int(iss.get("n_fifo_violations") or 0),
        "n_display_mismatch": int(iss.get("n_display_mismatch") or 0),
        "n_mcts_calls": int(n_mcts_calls),
        "planner": "greed_coop_neural_mcts",
        "greed_w": float(greed_w),
        "coop_w": float(coop_w),
        "yield_w": float(yield_w),
        "net": bool(net is not None),
    }
    (OUT / f"{sid}_mats.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[mats_standalone] done={done}/{total} sim={now} wall={wall}s "
        f"valid={rep['validate_ok']} mcts={n_mcts_calls} {rep['validate_summary']}",
        flush=True,
    )
    return rep


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Standalone CostTracer + greed–coop neural MCTS MAPD"
    )
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--wall-timeout", type=float, default=1800.0)
    ap.add_argument("--max-sim-time", type=int, default=200000)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument(
        "--greed",
        type=float,
        default=0.55,
        help="weight on personal progress (greed)",
    )
    ap.add_argument(
        "--coop",
        type=float,
        default=0.35,
        help="weight on conflict penalty (cooperation)",
    )
    ap.add_argument(
        "--yield-w",
        type=float,
        default=0.10,
        help="weight on soft yield bonus when blocking a closer agent",
    )
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
    solve_standalone(
        int(args.slot),
        wall_timeout=float(args.wall_timeout),
        max_sim_time=int(args.max_sim_time),
        num_simulations=int(args.sims),
        k_proximal=int(args.k),
        greed_w=float(args.greed),
        coop_w=float(args.coop),
        yield_w=float(args.yield_w),
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
