"""Lifelong MAPD task stream: controlled backlog, random pickup/dropoff, random urgent."""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


class LifelongTaskGenerator:
    """Keep a healthy online task pool without flooding or starving dropoffs.

    Design:
      - backlog in [min_backlog, max_backlog]
      - random pickup / dropoff each spawn
      - force coverage so no dropoff stays unused too long
      - prefer stations that currently have empty queues (when possible)
      - urgent tasks with random remaining_time
    """

    def __init__(
        self,
        *,
        pickups: Sequence[str],
        dropoffs: Sequence[str],
        start_points: Dict[str, Tuple[int, int]],
        end_points: Dict[str, Tuple[int, int]],
        mod,
        seed: int = 0,
        min_backlog: int = 12,
        max_backlog: int = 28,
        inject_every: int = 3,
        urgent_prob: float = 0.12,
        urgent_rt_range: Tuple[int, int] = (80, 220),
        dest_cover_every: int = 24,
        catalog_csv: Optional[Path] = None,
    ):
        self.pickups = list(pickups)
        self.dropoffs = list(dropoffs)
        self.start_points = dict(start_points)
        self.end_points = dict(end_points)
        self.mod = mod
        self.rng = random.Random(seed)
        self.min_backlog = int(min_backlog)
        self.max_backlog = int(max_backlog)
        self.inject_every = max(1, int(inject_every))
        self.urgent_prob = float(urgent_prob)
        self.urgent_rt_range = (int(urgent_rt_range[0]), int(urgent_rt_range[1]))
        self.dest_cover_every = max(1, int(dest_cover_every))
        self.catalog_csv = Path(catalog_csv) if catalog_csv else None

        self._seq = defaultdict(int)  # pickup -> next id suffix
        self.n_generated = 0
        self.n_urgent = 0
        self._since_dest: Dict[str, int] = {d: self.dest_cover_every for d in self.dropoffs}
        self._catalog_rows: List[dict] = []

        if self.catalog_csv is not None:
            self.catalog_csv.parent.mkdir(parents=True, exist_ok=True)
            if not self.catalog_csv.exists():
                with self.catalog_csv.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=[
                            "task_id",
                            "start_point",
                            "end_point",
                            "priority",
                            "remaining_time",
                            "spawn_t",
                        ],
                    )
                    w.writeheader()

    @staticmethod
    def backlog(sim) -> int:
        return sum(len(v) for v in (sim.task_states or {}).values())

    def _next_id(self, pickup: str) -> str:
        self._seq[pickup] += 1
        return f"{pickup}-L{self._seq[pickup]:04d}"

    def _choose_dropoff(self) -> str:
        overdue = [d for d, age in self._since_dest.items() if age >= self.dest_cover_every]
        if overdue:
            return self.rng.choice(overdue)
        return self.rng.choice(self.dropoffs)

    def _choose_pickup(self, sim) -> str:
        empty = [p for p in self.pickups if not (sim.task_states.get(p) or [])]
        if empty and self.rng.random() < 0.7:
            return self.rng.choice(empty)
        # Prefer shorter queues to avoid one station stacking forever
        lengths = [(len(sim.task_states.get(p) or []), p) for p in self.pickups]
        lengths.sort(key=lambda x: (x[0], self.rng.random()))
        # Soft bias to bottom half of queue lengths
        k = max(1, len(lengths) // 2)
        return self.rng.choice([p for _, p in lengths[:k]])

    def _build_item(
        self,
        pickup: str,
        dropoff: str,
        task_id: str,
        *,
        urgent: bool,
        remaining_time: Optional[int],
    ) -> dict:
        pickup_pos = self.mod.get_pickup_coord(pickup, self.start_points[pickup])
        unload_point, end_pts = self.mod.get_end_points(dropoff, self.end_points)
        return {
            "task_id": task_id,
            "pickup_point": pickup_pos,
            "unload_point": unload_point,
            "end_points": list(end_pts),
            "destination": dropoff,
            "priority": "Urgent" if urgent else "Normal",
            "remaining_time": remaining_time,
            "numbers_before_urgent": 0 if urgent else -1,
            "numbers_left": 0,
        }

    def _append_catalog(self, row: dict) -> None:
        self._catalog_rows.append(row)
        if self.catalog_csv is None:
            return
        with self.catalog_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "task_id",
                    "start_point",
                    "end_point",
                    "priority",
                    "remaining_time",
                    "spawn_t",
                ],
            )
            w.writerow(row)

    def _station_has_surface(self, sim, pickup: str) -> bool:
        for info in (sim.surface_tasks or {}).values():
            if info.get("pickup_name") == pickup:
                return True
        return False

    def _expose_if_needed(self, sim, pickup: str) -> None:
        """If station has queue but no surface view, expose queue head (FIFO)."""
        queue = sim.task_states.get(pickup) or []
        if not queue:
            return
        if self._station_has_surface(sim, pickup):
            return
        # Head still assigned/unpicked: do not promote past it
        head_id = queue[0].get("task_id")
        inflight = getattr(sim, "_inflight_tasks", None) or {}
        for info in inflight.values():
            if info.get("task_id") == head_id:
                return
        # Also skip if any AGV currently holds this task_id
        for agv in sim.agvs:
            if agv.task_id == head_id:
                return
        first = queue[0]
        tid = first["task_id"]
        details = {k: v for k, v in first.items() if k != "task_id"}
        if "end_points" in details:
            details["end_points"] = list(details["end_points"])
        sim.surface_tasks[tid] = details
        sim.surface_tasks[tid]["pickup_name"] = pickup

    def spawn_one(self, sim, *, spawn_t: Optional[int] = None) -> Optional[str]:
        pickup = self._choose_pickup(sim)
        dropoff = self._choose_dropoff()
        urgent = self.rng.random() < self.urgent_prob
        rt = None
        if urgent:
            rt = self.rng.randint(*self.urgent_rt_range)
            self.n_urgent += 1
        tid = self._next_id(pickup)
        item = self._build_item(pickup, dropoff, tid, urgent=urgent, remaining_time=rt)

        queue = sim.task_states.setdefault(pickup, [])
        if urgent:
            idx = len(queue)
            for j in range(idx):
                prev = queue[j].get("numbers_before_urgent", -1)
                if prev < 0:
                    queue[j]["numbers_before_urgent"] = idx - j
                else:
                    queue[j]["numbers_before_urgent"] = min(int(prev), idx - j)
        queue.append(item)
        for i, t in enumerate(queue):
            t["numbers_left"] = len(queue) - i - 1

        self._expose_if_needed(sim, pickup)

        for d in self._since_dest:
            self._since_dest[d] += 1
        self._since_dest[dropoff] = 0

        self.n_generated += 1
        self._append_catalog(
            {
                "task_id": tid,
                "start_point": pickup,
                "end_point": dropoff,
                "priority": "Urgent" if urgent else "Normal",
                "remaining_time": "" if rt is None else str(rt),
                "spawn_t": int(spawn_t if spawn_t is not None else getattr(sim, "time", 0)),
            }
        )
        return tid

    def seed(self, sim, n: Optional[int] = None) -> int:
        """Initial fill so the system starts with work."""
        target = int(n if n is not None else max(self.min_backlog, len(self.pickups)))
        added = 0
        # First pass: one task per pickup toward diverse dropoffs
        for i, p in enumerate(self.pickups):
            if added >= target:
                break
            # temporarily bias pickup choice
            dropoff = self.dropoffs[i % len(self.dropoffs)]
            urgent = self.rng.random() < self.urgent_prob
            rt = self.rng.randint(*self.urgent_rt_range) if urgent else None
            tid = self._next_id(p)
            item = self._build_item(p, dropoff, tid, urgent=urgent, remaining_time=rt)
            if urgent:
                self.n_urgent += 1
            sim.task_states.setdefault(p, []).append(item)
            self._expose_if_needed(sim, p)
            self._since_dest[dropoff] = 0
            self.n_generated += 1
            added += 1
            self._append_catalog(
                {
                    "task_id": tid,
                    "start_point": p,
                    "end_point": dropoff,
                    "priority": "Urgent" if urgent else "Normal",
                    "remaining_time": "" if rt is None else str(rt),
                    "spawn_t": 0,
                }
            )
        while added < target and self.backlog(sim) < self.max_backlog:
            if self.spawn_one(sim, spawn_t=0):
                added += 1
            else:
                break
        return added

    def tick(self, sim) -> int:
        """Called after each time_forward. Returns number of newly spawned tasks."""
        # Age urgent remaining_time lightly (optional soft clock)
        for queue in (sim.task_states or {}).values():
            for t in queue:
                if str(t.get("priority", "")).lower() == "urgent" and t.get("remaining_time") is not None:
                    try:
                        t["remaining_time"] = max(0, int(t["remaining_time"]) - 1)
                    except (TypeError, ValueError):
                        pass

        if int(sim.time) % self.inject_every != 0:
            return 0

        added = 0
        bl = self.backlog(sim)
        # Top up toward mid of [min,max] when below min; never exceed max
        mid = (self.min_backlog + self.max_backlog) // 2
        want = 0
        if bl < self.min_backlog:
            want = mid - bl
        elif bl < mid and self.rng.random() < 0.55:
            want = 1
        want = max(0, min(want, self.max_backlog - bl))
        for _ in range(want):
            if self.backlog(sim) >= self.max_backlog:
                break
            if self.spawn_one(sim, spawn_t=sim.time):
                added += 1
        return added


def install_lifelong_generator(sim, gen: LifelongTaskGenerator) -> LifelongTaskGenerator:
    """Hook generator into Simulation.time_forward; disable finite all_over."""
    sim._lifelong_gen = gen
    sim.all_over = lambda: False  # type: ignore[method-assign]
    orig = sim.time_forward

    def hooked_time_forward():
        orig()
        n = gen.tick(sim)
        if n and int(sim.time) % 50 == 0:
            print(
                f"[LIFELONG] t={sim.time} spawned+{n} backlog={gen.backlog(sim)} "
                f"generated={gen.n_generated} urgent={gen.n_urgent}",
                flush=True,
            )

    sim.time_forward = hooked_time_forward  # type: ignore[method-assign]
    return gen
