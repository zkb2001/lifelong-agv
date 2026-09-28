"""Collect abundant BC demos by static sampling + M2-like teacher ranking."""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from ml_research.benchmarks.common import is_urgent_task, load_scenario
from ml_research.common.paths import POSITION_CSV, RESULTS, TASK_CSV

from .features import AGV_DIM, MAX_AGV, MAX_TASK, TASK_DIM, fleet_density

OUT = RESULTS / "rl_rh_pp"
OUT.mkdir(parents=True, exist_ok=True)


def _flatten_tasks(task_states: dict) -> List[Tuple[str, dict]]:
    out = []
    for station, q in (task_states or {}).items():
        for t in q:
            tid = str(t.get("task_id") or t.get("id") or f"{station}-{len(out)}")
            info = dict(t)
            out.append((tid, info))
    return out


class _FakeSim:
    def __init__(self, agvs, surface_tasks):
        self.agvs = agvs
        self.surface_tasks = surface_tasks


class _FakeAgv:
    def __init__(self, name, xy):
        self.name = name
        self.state = (int(xy[0]), int(xy[1]), 0, 90)
        self.current_task = None


def _teacher_key(info: dict, agvs: list) -> tuple:
    urgent = 0 if is_urgent_task(info) else 1
    rt = info.get("remaining_time")
    rt_key = float(rt) if rt is not None else 1e9
    before = info.get("numbers_before_urgent", -1)
    before_key = before if before is not None and before >= 0 else 1e6
    pk = tuple(info["pickup_point"])
    min_c = min(
        (abs(int(a.state[0]) - pk[0]) + abs(int(a.state[1]) - pk[1]) for a in agvs),
        default=99,
    )
    if before is not None and before >= 0:
        min_c *= 0.7
    return (urgent, before_key, rt_key, min_c)


def collect_snapshots(*, max_snaps: int = 500, max_time: int = 0, stride: int = 1) -> Dict:
    del max_time, stride
    _mod, _env, agv_states, task_states, n_tasks = load_scenario(TASK_CSV, POSITION_CSV)
    all_tasks = _flatten_tasks(task_states)
    agv_names = list(agv_states.keys())
    init_xy = {}
    for name, agv in agv_states.items():
        st = agv["state"] if isinstance(agv, dict) else agv.state
        init_xy[name] = (int(st[0]), int(st[1]))

    rng = np.random.default_rng(42)
    agv_fs, agv_ms, task_fs, task_ms, teachers = [], [], [], [], []

    for _ in range(int(max_snaps)):
        agvs = []
        for name in agv_names[:MAX_AGV]:
            x0, y0 = init_xy[name]
            x = int(np.clip(x0 + int(rng.integers(-3, 4)), 1, 20))
            y = int(np.clip(y0 + int(rng.integers(-3, 4)), 1, 20))
            agvs.append(_FakeAgv(name, (x, y)))

        k = int(rng.integers(4, min(MAX_TASK, max(5, len(all_tasks))) + 1))
        idxs = rng.choice(len(all_tasks), size=min(k, len(all_tasks)), replace=False)
        surface = {all_tasks[i][0]: all_tasks[i][1] for i in idxs}
        sim = _FakeSim(agvs, surface)

        af = np.zeros((MAX_AGV, AGV_DIM), np.float32)
        am = np.zeros((MAX_AGV,), np.bool_)
        for i, a in enumerate(agvs[:MAX_AGV]):
            x, y = int(a.state[0]), int(a.state[1])
            af[i] = [x / 20, y / 20, fleet_density(sim, (x, y)), 0, 1, 0, 1, 0, 0, 0, 0, 0]
            am[i] = True

        tids = list(surface.keys())[:MAX_TASK]
        tf = np.zeros((MAX_TASK, TASK_DIM), np.float32)
        tm = np.zeros((MAX_TASK,), np.bool_)
        for i, tid in enumerate(tids):
            info = surface[tid]
            pk = tuple(info["pickup_point"])
            ends = info.get("end_points") or (
                [tuple(info["unload_point"])] if info.get("unload_point") else [pk]
            )
            ek = tuple(ends[0])
            urgent = 1.0 if is_urgent_task(info) else 0.0
            rt = info.get("remaining_time")
            rt_n = float(rt) / 500.0 if rt is not None else 2.0
            before = info.get("numbers_before_urgent", -1)
            before_n = float(before) / 10.0 if before is not None and before >= 0 else 1.0
            dens_p = fleet_density(sim, pk)
            dens_e = fleet_density(sim, ek)
            min_c = min(
                (abs(int(a.state[0]) - pk[0]) + abs(int(a.state[1]) - pk[1]) for a in agvs),
                default=99,
            )
            trip = abs(pk[0] - ek[0]) + abs(pk[1] - ek[1])
            tf[i] = [
                pk[0] / 20,
                pk[1] / 20,
                ek[0] / 20,
                ek[1] / 20,
                urgent,
                rt_n,
                before_n,
                dens_p,
                dens_e,
                min(min_c, 80) / 80,
                trip / 40,
                1,
                0,
                0,
            ]
            tm[i] = True

        order = sorted(tids, key=lambda tid: _teacher_key(surface[tid], agvs))
        id2i = {tid: i for i, tid in enumerate(tids)}
        teacher = np.full((MAX_TASK,), -1, np.int64)
        for j, tid in enumerate(order):
            teacher[j] = id2i[tid]

        agv_fs.append(af)
        agv_ms.append(am)
        task_fs.append(tf)
        task_ms.append(tm)
        teachers.append(teacher)

    path = OUT / "bc_demos.npz"
    np.savez_compressed(
        path,
        agv_f=np.asarray(agv_fs, np.float32),
        agv_m=np.asarray(agv_ms, np.bool_),
        task_f=np.asarray(task_fs, np.float32),
        task_m=np.asarray(task_ms, np.bool_),
        teacher=np.asarray(teachers, np.int64),
    )
    meta = {
        "n": len(teachers),
        "path": str(path),
        "n_tasks": int(n_tasks),
        "mode": "static_sample_m2like",
    }
    (OUT / "bc_demos_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[RL-RH-PP] collected {meta['n']} static snaps → {path}", flush=True)
    return meta


if __name__ == "__main__":
    collect_snapshots()
