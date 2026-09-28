"""Fixed lifelong task schedule + A/B: plain A* vs engine_swapnet+SwapNet (same tasks)."""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ml_research.common.paths import CKPT, POSITION_CSV, RESULTS, ROOT

OUT = RESULTS / "lifelong" / "swapnet_ab"


def schedule_path(max_time: int) -> Path:
    return OUT / f"fixed_schedule_{int(max_time)}.csv"



def _load_engine(path: Path, module_name: str):
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_fixed_schedule(
    *,
    pickups: Sequence[str],
    dropoffs: Sequence[str],
    seed: int = 42,
    max_time: int = 3000,
    inject_every: int = 3,
    urgent_prob: float = 0.12,
    seed_n: int = 12,
    urgent_rt_range: Tuple[int, int] = (80, 220),
) -> List[dict]:
    """Deterministic spawn list — independent of sim backlog (fair A/B)."""
    rng = random.Random(seed)
    rows: List[dict] = []
    seq = defaultdict(int)

    def next_id(p: str) -> str:
        seq[p] += 1
        return f"{p}-F{seq[p]:04d}"

    # Initial seed tasks at t=0 (one-ish per pickup, round-robin dropoffs)
    for i in range(seed_n):
        p = pickups[i % len(pickups)]
        d = dropoffs[i % len(dropoffs)]
        urgent = rng.random() < urgent_prob
        rows.append(
            {
                "spawn_t": 0,
                "task_id": next_id(p),
                "start_point": p,
                "end_point": d,
                "priority": "Urgent" if urgent else "Normal",
                "remaining_time": str(rng.randint(*urgent_rt_range)) if urgent else "",
            }
        )

    t = inject_every
    while t <= max_time:
        p = rng.choice(list(pickups))
        d = rng.choice(list(dropoffs))
        urgent = rng.random() < urgent_prob
        rows.append(
            {
                "spawn_t": t,
                "task_id": next_id(p),
                "start_point": p,
                "end_point": d,
                "priority": "Urgent" if urgent else "Normal",
                "remaining_time": str(rng.randint(*urgent_rt_range)) if urgent else "",
            }
        )
        t += inject_every
    return rows


def write_schedule(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "spawn_t",
                "task_id",
                "start_point",
                "end_point",
                "priority",
                "remaining_time",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)


def load_schedule(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


class FixedScheduleInjector:
    """Inject predetermined tasks at spawn_t (identical for both arms)."""

    def __init__(self, rows: List[dict], mod, start_points, end_points):
        self.mod = mod
        self.start_points = start_points
        self.end_points = end_points
        by_t: Dict[int, List[dict]] = defaultdict(list)
        for r in rows:
            by_t[int(r["spawn_t"])].append(r)
        self.by_t = dict(by_t)
        self.n_injected = 0
        self.n_urgent = 0

    def _build_item(self, row: dict) -> dict:
        pickup = row["start_point"]
        dropoff = row["end_point"]
        urgent = str(row.get("priority", "")).lower() == "urgent"
        rt = row.get("remaining_time") or None
        if rt == "":
            rt = None
        if rt is not None:
            rt = int(rt)
        pickup_pos = self.mod.get_pickup_coord(pickup, self.start_points[pickup])
        unload_point, end_pts = self.mod.get_end_points(dropoff, self.end_points)
        return {
            "task_id": row["task_id"],
            "pickup_point": pickup_pos,
            "unload_point": unload_point,
            "end_points": list(end_pts),
            "destination": dropoff,
            "priority": "Urgent" if urgent else "Normal",
            "remaining_time": rt,
            "numbers_before_urgent": 0 if urgent else -1,
            "numbers_left": 0,
        }

    def _expose_head(self, sim, pickup: str) -> None:
        for info in (sim.surface_tasks or {}).values():
            if info.get("pickup_name") == pickup:
                return
        queue = sim.task_states.get(pickup) or []
        if not queue:
            return
        head_id = queue[0]["task_id"]
        for agv in sim.agvs:
            if agv.task_id == head_id:
                return
        inflight = getattr(sim, "_inflight_tasks", None) or {}
        for info in inflight.values():
            if info.get("task_id") == head_id:
                return
        first = queue[0]
        tid = first["task_id"]
        details = {k: v for k, v in first.items() if k != "task_id"}
        if "end_points" in details:
            details["end_points"] = list(details["end_points"])
        sim.surface_tasks[tid] = details
        sim.surface_tasks[tid]["pickup_name"] = pickup

    def inject_at(self, sim, t: int) -> int:
        rows = self.by_t.get(int(t), [])
        n = 0
        for row in rows:
            item = self._build_item(row)
            if item["priority"] == "Urgent":
                self.n_urgent += 1
            p = row["start_point"]
            q = sim.task_states.setdefault(p, [])
            q.append(item)
            for i, task in enumerate(q):
                task["numbers_left"] = len(q) - i - 1
            self._expose_head(sim, p)
            self.n_injected += 1
            n += 1
        return n

    def install(self, sim) -> None:
        sim.all_over = lambda: False  # type: ignore[method-assign]
        # Seed t=0 before loop
        self.inject_at(sim, 0)
        orig = sim.time_forward

        def hooked():
            orig()
            self.inject_at(sim, sim.time)

        sim.time_forward = hooked  # type: ignore[method-assign]


def _write_traj(path: Path, steps_dict: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "name",
        "X",
        "Y",
        "pitch",
        "loaded",
        "destination",
        "Emergency",
        "task-id",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t in sorted(steps_dict.keys()):
            for s in steps_dict[t]:
                w.writerow(
                    [
                        s["timestamp"],
                        s["name"],
                        s["X"],
                        s["Y"],
                        s["pitch"],
                        str(s["loaded"]).lower(),
                        s["destination"],
                        str(s["Emergency"]).lower(),
                        s.get("task-id", ""),
                    ]
                )


def _count_pickups(steps_dict: dict) -> int:
    # group by agv timeline
    by_agv: Dict[str, List[dict]] = defaultdict(list)
    for t, rows in steps_dict.items():
        for s in rows:
            by_agv[s["name"]].append(s)
    n = 0
    for rows in by_agv.values():
        rows.sort(key=lambda s: int(s["timestamp"]))
        prev = False
        for s in rows:
            loaded = str(s.get("loaded", "")).lower() in ("true", "1", "yes")
            tid = str(s.get("task-id") or "").strip()
            if loaded and not prev and tid:
                n += 1
            prev = loaded
    return n


def _count_unloads(steps_dict: dict) -> int:
    by_agv: Dict[str, List[dict]] = defaultdict(list)
    for t, rows in steps_dict.items():
        for s in rows:
            by_agv[s["name"]].append(s)
    n = 0
    for rows in by_agv.values():
        rows.sort(key=lambda s: int(s["timestamp"]))
        prev_l = False
        prev_tid = ""
        for s in rows:
            loaded = str(s.get("loaded", "")).lower() in ("true", "1", "yes")
            tid = str(s.get("task-id") or "").strip()
            if prev_l and not loaded and prev_tid:
                n += 1
            prev_l = loaded
            prev_tid = tid if loaded else ""
    return n


def _loaded_steps(steps_dict: dict) -> int:
    n = 0
    for rows in steps_dict.values():
        for s in rows:
            if str(s.get("loaded", "")).lower() in ("true", "1", "yes"):
                n += 1
    return n


def _manhattan_travel(steps_dict: dict) -> int:
    by_agv: Dict[str, List[dict]] = defaultdict(list)
    for rows in steps_dict.values():
        for s in rows:
            by_agv[s["name"]].append(s)
    dist = 0
    for rows in by_agv.values():
        rows.sort(key=lambda s: int(s["timestamp"]))
        for a, b in zip(rows, rows[1:]):
            if int(b["timestamp"]) != int(a["timestamp"]) + 1:
                continue
            dist += abs(int(b["X"]) - int(a["X"])) + abs(int(b["Y"]) - int(a["Y"]))
    return dist


def _motion_stats(steps_dict: dict) -> dict:
    """Move cells + turn ticks (90° = 1) between consecutive timestamps."""
    by_agv: Dict[str, List[dict]] = defaultdict(list)
    for rows in steps_dict.values():
        for s in rows:
            by_agv[s["name"]].append(s)

    move_cells = 0
    turn_ticks = 0
    turn_events = 0
    wait_steps = 0
    total_steps = 0
    for rows in by_agv.values():
        rows.sort(key=lambda s: int(s["timestamp"]))
        for a, b in zip(rows, rows[1:]):
            if int(b["timestamp"]) != int(a["timestamp"]) + 1:
                continue
            total_steps += 1
            dx = abs(int(b["X"]) - int(a["X"]))
            dy = abs(int(b["Y"]) - int(a["Y"]))
            move = dx + dy
            move_cells += move
            pa = int(a.get("pitch", 0) or 0) % 360
            pb = int(b.get("pitch", 0) or 0) % 360
            diff = abs(pb - pa) % 360
            if diff > 180:
                diff = 360 - diff
            turns = int(diff // 90)
            if turns > 0:
                turn_ticks += turns
                turn_events += 1
            if move == 0 and turns == 0:
                wait_steps += 1
    return {
        "move_cells": move_cells,
        "turn_ticks": turn_ticks,
        "turn_events": turn_events,
        "wait_steps": wait_steps,
        "motion_steps": total_steps,
        "move_plus_turn": move_cells + turn_ticks,
    }


def _mean_task_cycle(steps_dict: dict) -> Optional[float]:
    """Mean timestamps from pickup(loaded rise) to unload(loaded fall)."""
    by_agv: Dict[str, List[dict]] = defaultdict(list)
    for rows in steps_dict.values():
        for s in rows:
            by_agv[s["name"]].append(s)
    cycles: List[int] = []
    for rows in by_agv.values():
        rows.sort(key=lambda s: int(s["timestamp"]))
        pickup_t = None
        prev_l = False
        for s in rows:
            loaded = str(s.get("loaded", "")).lower() in ("true", "1", "yes")
            t = int(s["timestamp"])
            if loaded and not prev_l:
                pickup_t = t
            if prev_l and not loaded and pickup_t is not None:
                cycles.append(t - pickup_t)
                pickup_t = None
            prev_l = loaded
    if not cycles:
        return None
    return round(sum(cycles) / len(cycles), 3)

def run_arm(
    *,
    name: str,
    engine_path: Path,
    module_name: str,
    schedule: List[dict],
    position_csv: Path,
    max_time: int,
    enable_swapnet: bool,
) -> dict:
    mod = _load_engine(engine_path, module_name)
    start_points, end_points, agv_list = mod.get_object_position(str(position_csv))
    env = mod.ENV(list(start_points.values()), list(end_points.values()))
    agv_states = mod.get_agv_state(agv_list)
    task_states = {k: [] for k in start_points}

    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )

    if enable_swapnet:
        from ml_research.benchmarks.m5_replan import ensure_m5_hooks
        from ml_research.benchmarks.station_eta import ensure_station_eta_hooks
        from ml_research.benchmarks.dock_alley_ai.hooks import attach_swap_net

        ensure_m5_hooks(sim, horizon=40, enable_replan=True, aggressive=False)
        ensure_station_eta_hooks(sim)
        ckpt = CKPT / "swap_net_v3.pt"
        if not ckpt.exists():
            ckpt = CKPT / "swap_net.pt"
        attach_swap_net(
            sim,
            ckpt=ckpt if ckpt.exists() else None,
            threshold_normal=0.48,
            threshold_urgent=0.40,
        )
        sim._swapnet_enabled = True
        sim._swapnet_swap_count = 0
        if not hasattr(sim, "_replan_stats") or not isinstance(sim._replan_stats, dict):
            sim._replan_stats = {}
        sim._replan_stats.setdefault("station_swap", 0)

    injector = FixedScheduleInjector(schedule, mod, start_points, end_points)
    injector.install(sim)

    t0 = time.perf_counter()
    while sim.time < max_time:
        sim.time_forward()
        if sim.time % 500 == 0:
            print(
                f"  [{name}] t={sim.time} backlog={sum(len(v) for v in sim.task_states.values())} "
                f"injected={injector.n_injected} wall={time.perf_counter()-t0:.1f}s",
                flush=True,
            )

    wall = time.perf_counter() - t0
    steps = sim.steps_reorganize()
    traj_path = OUT / f"traj_{name}.csv"
    _write_traj(traj_path, steps)

    pickups = _count_pickups(steps)
    unloads = _count_unloads(steps)
    backlog_end = sum(len(v) for v in (sim.task_states or {}).values())
    rs = getattr(sim, "_replan_stats", {}) or {}
    motion = _motion_stats(steps)
    out = {
        "name": name,
        "swapnet": bool(enable_swapnet),
        "sim_time": int(sim.time),
        "wall_seconds": round(wall, 3),
        "schedule_tasks": len(schedule),
        "injected": injector.n_injected,
        "n_urgent_injected": injector.n_urgent,
        "pickups": pickups,
        "unloads": unloads,
        "loaded_agv_steps": _loaded_steps(steps),
        "manhattan_travel": _manhattan_travel(steps),
        "move_cells": motion["move_cells"],
        "turn_ticks": motion["turn_ticks"],
        "turn_events": motion["turn_events"],
        "wait_steps": motion["wait_steps"],
        "move_plus_turn": motion["move_plus_turn"],
        "mean_pickup_to_unload": _mean_task_cycle(steps),
        "backlog_end": backlog_end,
        "station_swap": int(rs.get("station_swap", 0) or getattr(sim, "_swapnet_swap_count", 0) or 0),
        "swap_policy_yes": int(rs.get("swap_policy_yes", 0) or 0),
        "swap_policy_no": int(rs.get("swap_policy_no", 0) or 0),
        "swap_policy_sources": dict(rs.get("swap_policy_sources") or {}),
        "trajectory": str(traj_path),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A/B lifelong: plain A* vs SwapNet (fixed schedule)")
    ap.add_argument("--max-time", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--position", type=Path, default=POSITION_CSV)
    ap.add_argument("--rebuild-schedule", action="store_true")
    args = ap.parse_args(argv)

    os.environ.pop("AGV_GPU_BFS", None)
    OUT.mkdir(parents=True, exist_ok=True)
    sched_csv = schedule_path(args.max_time)

    # Need pickup/dropoff names from map
    mod0 = _load_engine(ROOT / "simulation" / "engine.py", "mioverse_ab_map")
    starts, ends, _ = mod0.get_object_position(str(args.position))

    if args.rebuild_schedule or not sched_csv.exists():
        rows = build_fixed_schedule(
            pickups=list(starts.keys()),
            dropoffs=list(ends.keys()),
            seed=args.seed,
            max_time=args.max_time,
        )
        write_schedule(sched_csv, rows)
        print(f"[AB] wrote schedule {sched_csv} n={len(rows)}", flush=True)
    else:
        rows = load_schedule(sched_csv)
        # Keep only spawn_t within horizon if file was longer
        rows = [r for r in rows if int(r["spawn_t"]) <= args.max_time]
        print(f"[AB] loaded schedule {sched_csv} n={len(rows)}", flush=True)

    print("[AB] arm=baseline (engine.py, no SwapNet)", flush=True)
    base = run_arm(
        name=f"baseline_noswap_t{args.max_time}",
        engine_path=ROOT / "simulation" / "engine.py",
        module_name="mioverse_ab_baseline",
        schedule=rows,
        position_csv=args.position,
        max_time=args.max_time,
        enable_swapnet=False,
    )
    print(f"[AB] baseline done {base}", flush=True)

    print("[AB] arm=swapnet (engine_swapnet.py + SwapNet)", flush=True)
    sw = run_arm(
        name=f"swapnet_t{args.max_time}",
        engine_path=ROOT / "simulation" / "engine_swapnet.py",
        module_name="mioverse_ab_swapnet",
        schedule=rows,
        position_csv=args.position,
        max_time=args.max_time,
        enable_swapnet=True,
    )
    print(f"[AB] swapnet done {sw}", flush=True)

    def delta(key: str):
        if base.get(key) is None or sw.get(key) is None:
            return None
        return sw[key] - base[key]

    def pct_save(key: str):
        """Positive = swapnet smaller (saved)."""
        b, s = base.get(key), sw.get(key)
        if b is None or s is None or not b:
            return None
        return round(100.0 * (b - s) / b, 3)

    metric_keys = [
        "pickups",
        "unloads",
        "loaded_agv_steps",
        "manhattan_travel",
        "move_cells",
        "turn_ticks",
        "turn_events",
        "move_plus_turn",
        "wait_steps",
        "mean_pickup_to_unload",
        "backlog_end",
        "station_swap",
        "wall_seconds",
    ]
    deltas = {k: delta(k) for k in metric_keys}
    savings = {
        "move_cells_saved": (base["move_cells"] - sw["move_cells"]),
        "turn_ticks_saved": (base["turn_ticks"] - sw["turn_ticks"]),
        "move_plus_turn_saved": (base["move_plus_turn"] - sw["move_plus_turn"]),
        "wait_steps_saved": (base["wait_steps"] - sw["wait_steps"]),
        "manhattan_travel_saved": (base["manhattan_travel"] - sw["manhattan_travel"]),
        "mean_cycle_time_saved": (
            None
            if base.get("mean_pickup_to_unload") is None
            or sw.get("mean_pickup_to_unload") is None
            else round(base["mean_pickup_to_unload"] - sw["mean_pickup_to_unload"], 3)
        ),
        "pct_move_plus_turn": pct_save("move_plus_turn"),
        "pct_manhattan_travel": pct_save("manhattan_travel"),
        "pct_wait_steps": pct_save("wait_steps"),
        "extra_unloads": sw["unloads"] - base["unloads"],
    }

    summary = {
        "max_time": args.max_time,
        "seed": args.seed,
        "schedule": str(sched_csv),
        "baseline": base,
        "swapnet": sw,
        "delta_swapnet_minus_baseline": deltas,
        "savings_baseline_minus_swapnet": savings,
    }
    out_json = OUT / f"ab_summary_t{args.max_time}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # also keep legacy name for latest
    (OUT / "ab_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n========== SwapNet A/B (identical fixed schedule) ==========", flush=True)
    print(f"schedule tasks: {len(rows)}  horizon: {args.max_time}", flush=True)
    print(
        f"{'metric':<24} {'baseline':>12} {'swapnet':>12} {'delta(sw-base)':>14}",
        flush=True,
    )
    for k in metric_keys:
        bv, sv = base.get(k), sw.get(k)
        dv = deltas.get(k)
        print(f"{k:<24} {bv!s:>12} {sv!s:>12} {dv!s:>14}", flush=True)
    print("\n--- Savings (baseline - swapnet; >0 means SwapNet used less) ---", flush=True)
    for k, v in savings.items():
        print(f"  {k}: {v}", flush=True)
    print(
        f"swap_policy_yes/no: {sw['swap_policy_yes']}/{sw['swap_policy_no']} "
        f"sources={sw.get('swap_policy_sources')}",
        flush=True,
    )
    print(f"summary -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
