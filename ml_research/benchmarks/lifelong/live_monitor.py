"""Real-time AGV monitor: tails live trajectory + task catalog and plays frames.

CSV layout (same as simulation export):
  header + repeating blocks of N rows per sim-time, where N = number of AGVs.
  e.g. 12 AGVs ⇒ rows [1..12]=t0, [13..24]=t1, ...
Playback advances exactly one block (= one sim second) per displayed frame.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pygame

from ml_research.common.paths import POSITION_CSV, RESULTS

GRID_SIZE = 36
WINDOW = (20 * GRID_SIZE, 20 * GRID_SIZE + 48)
FPS_DEFAULT = 10

WHITE = (255, 255, 255)
BLUE = (100, 149, 237)
GREEN = (50, 205, 50)
BLACK = (0, 0, 0)
RED = (220, 40, 40)
PINK = (255, 182, 193)
GRAY = (200, 200, 200)
HUD_BG = (30, 30, 36)
HUD_FG = (230, 230, 230)

AGV_COLORS = [
    (255, 182, 193),
    (255, 218, 185),
    (176, 224, 230),
    (221, 160, 221),
    (152, 251, 152),
    (135, 206, 250),
    (240, 230, 140),
    (255, 228, 225),
    (216, 191, 216),
    (238, 232, 170),
    (230, 230, 250),
    (255, 239, 213),
]

CITY_TO_NUMBER = {
    "Hangzhou": 1,
    "Guangzhou": 2,
    "Urumqi": 3,
    "Chongqing": 4,
    "Suzhou": 5,
    "Changsha": 6,
    "Kunming": 7,
    "Tianjin": 8,
    "Shanghai": 9,
    "Wuhan": 10,
    "Xiamen": 11,
    "Dalian": 12,
    "Beijing": 13,
    "Nanjing": 14,
    "Chengdu": 15,
    "Shenzhen": 16,
}


def _cy(y: int) -> int:
    return 19 - y


def load_positions(path: Path):
    starts, ends = {}, {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t, name = row["type"].strip(), row["name"].strip()
            x, y = int(row["x"]), int(row["y"])
            if t == "start_point":
                starts[name] = (x - 1, y - 1)
            elif t == "end_point":
                ends[name] = (x - 1, y - 1)
    return starts, ends


def load_task_info(path: Path) -> Dict[str, dict]:
    info: Dict[str, dict] = {}
    if not path.exists():
        return info
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row.get("task_id") or ""
            if not tid:
                continue
            try:
                spawn_t = int(row.get("spawn_t") or 0)
            except ValueError:
                spawn_t = 0
            info[tid] = {
                "start_point": row.get("start_point", ""),
                "end_point": row.get("end_point", ""),
                "priority": row.get("priority", "Normal"),
                "spawn_t": spawn_t,
            }
    return info


def build_task_sequence(task_info: Dict[str, dict]) -> Dict[str, List[str]]:
    seq: Dict[str, List[str]] = defaultdict(list)
    for tid, info in task_info.items():
        sp = info.get("start_point") or ""
        if sp:
            seq[sp].append(tid)
    return dict(seq)


def detect_n_agv(rows: List[dict]) -> int:
    """First contiguous run with the same timestamp = one sim-second block."""
    if not rows:
        return 0
    t0 = rows[0].get("timestamp")
    n = 0
    for r in rows:
        if r.get("timestamp") != t0:
            break
        n += 1
    return n


def load_frame_list(path: Path) -> Tuple[List[Dict[str, dict]], List[int], int]:
    """Parse trajectory as fixed-size AGV blocks.

    Returns:
      frames: list[dict[name -> row]]  index = playhead frame
      times:  list[int]                timestamp field of each block
      n_agv:  rows per sim-second
    Incomplete trailing blocks (len < n_agv) are ignored until the file grows.
    """
    if not path.exists() or path.stat().st_size < 40:
        return [], [], 0
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], [], 0
    n_agv = detect_n_agv(rows)
    if n_agv <= 0:
        return [], [], 0
    frames: List[Dict[str, dict]] = []
    times: List[int] = []
    for i in range(0, len(rows), n_agv):
        chunk = rows[i : i + n_agv]
        if len(chunk) < n_agv:
            break  # partial second — wait for more rows
        try:
            t = int(chunk[0]["timestamp"])
        except (KeyError, ValueError):
            t = len(frames)
        frames.append({r["name"]: r for r in chunk})
        times.append(t)
    return frames, times, n_agv


def pickup_events_through(
    frames: List[Dict[str, dict]], upto_idx: int
) -> Tuple[Set[str], Dict[str, str]]:
    picked: Set[str] = set()
    carrying: Dict[str, str] = {}
    if upto_idx < 0 or not frames:
        return picked, carrying
    upto_idx = min(upto_idx, len(frames) - 1)
    for i in range(0, upto_idx + 1):
        cur = frames[i]
        prev = frames[i - 1] if i > 0 else {}
        for name, row in cur.items():
            loaded = str(row.get("loaded", "")).lower() in ("true", "1", "yes")
            tid = str(row.get("task-id") or "").strip()
            prev_loaded = False
            if name in prev:
                prev_loaded = str(prev[name].get("loaded", "")).lower() in ("true", "1", "yes")
            if loaded and not prev_loaded and tid:
                picked.add(tid)
                carrying[name] = tid
            elif loaded and tid:
                carrying[name] = tid
            elif not loaded and name in carrying:
                del carrying[name]
    return picked, carrying


class LiveMonitor:
    def __init__(
        self,
        *,
        traj_csv: Path,
        task_csv: Path,
        position_csv: Path,
        fps: float = FPS_DEFAULT,
        follow: bool = True,
        start_at_live: bool = False,
    ):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW)
        pygame.display.set_caption("Lifelong AGV Live Monitor")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 18)
        self.big = pygame.font.Font(None, 24)
        self.traj_csv = Path(traj_csv)
        self.task_csv = Path(task_csv)
        self.fps = float(fps)
        self.follow = follow
        self.start_at_live = start_at_live
        self.starts, self.ends = load_positions(Path(position_csv))
        self.task_info = load_task_info(self.task_csv)
        self.task_sequence = build_task_sequence(self.task_info)
        self.frames: List[Dict[str, dict]] = []
        self.times: List[int] = []
        self.n_agv = 0
        self.idx = 0  # playhead: which block / sim-second
        self._traj_mtime = -1.0
        self._traj_size = -1
        self._task_mtime = -1.0
        self.completed: Set[str] = set()
        self.current_tasks: Dict[str, str] = {}
        self._state_idx = -1
        self.agv_colors: Dict[str, tuple] = {}
        self._color_i = 0
        self.paused = False
        self._bootstrapped = False
        self.status = "waiting for trajectory…"

    def _color(self, name: str):
        if name not in self.agv_colors:
            self.agv_colors[name] = AGV_COLORS[self._color_i % len(AGV_COLORS)]
            self._color_i += 1
        return self.agv_colors[name]

    @property
    def max_idx(self) -> int:
        return len(self.frames) - 1

    def refresh(self) -> None:
        if self.task_csv.exists():
            mt = self.task_csv.stat().st_mtime
            if mt != self._task_mtime:
                self.task_info = load_task_info(self.task_csv)
                self.task_sequence = build_task_sequence(self.task_info)
                self._task_mtime = mt
                self._state_idx = -1

        if not self.traj_csv.exists():
            return
        st = self.traj_csv.stat()
        if st.st_mtime == self._traj_mtime and st.st_size == self._traj_size:
            return

        frames, times, n_agv = load_frame_list(self.traj_csv)
        self._traj_mtime = st.st_mtime
        self._traj_size = st.st_size
        self.frames = frames
        self.times = times
        self.n_agv = n_agv
        self._state_idx = -1

        if not frames:
            self.status = "waiting for complete AGV block…"
            return

        if not self._bootstrapped:
            # Default: play from t=0 so the picture actually moves.
            self.idx = self.max_idx if self.start_at_live else 0
            self._bootstrapped = True
        else:
            # Keep playhead; clamp if file shrank (rewrite).
            self.idx = min(self.idx, self.max_idx)

        self.status = f"n_agv={self.n_agv} frames={len(self.frames)}"

    def _sync_task_state(self, idx: int) -> None:
        if idx < 0 or not self.frames:
            self.completed = set()
            self.current_tasks = {}
            self._state_idx = -1
            return
        if self._state_idx == idx:
            return
        if self._state_idx < 0 or idx < self._state_idx or idx > self._state_idx + 1:
            self.completed, self.current_tasks = pickup_events_through(self.frames, idx)
            self._state_idx = idx
            return
        cur = self.frames[idx]
        prev = self.frames[idx - 1] if idx > 0 else {}
        for name, row in cur.items():
            loaded = str(row.get("loaded", "")).lower() in ("true", "1", "yes")
            tid = str(row.get("task-id") or "").strip()
            prev_loaded = False
            if name in prev:
                prev_loaded = str(prev[name].get("loaded", "")).lower() in ("true", "1", "yes")
            if loaded and not prev_loaded and tid:
                self.completed.add(tid)
                self.current_tasks[name] = tid
            elif loaded and tid:
                self.current_tasks[name] = tid
            elif not loaded and name in self.current_tasks:
                del self.current_tasks[name]
        self._state_idx = idx

    def _surface_display_task(self, start: str, sim_t: int) -> Optional[str]:
        for tid in self.task_sequence.get(start, []):
            info = self.task_info.get(tid) or {}
            if int(info.get("spawn_t", 0)) > sim_t:
                continue
            if tid in self.completed:
                continue
            return tid
        return None

    def _draw_grid(self) -> None:
        self.screen.fill(WHITE)
        for i in range(21):
            pygame.draw.line(self.screen, GRAY, (i * GRID_SIZE, 0), (i * GRID_SIZE, 20 * GRID_SIZE))
            pygame.draw.line(self.screen, GRAY, (0, i * GRID_SIZE), (20 * GRID_SIZE, i * GRID_SIZE))

    def _draw_stations(self) -> None:
        for name, (x, y) in self.starts.items():
            pygame.draw.rect(
                self.screen,
                BLUE,
                (x * GRID_SIZE, _cy(y) * GRID_SIZE, GRID_SIZE, GRID_SIZE),
            )
        for name, (x, y) in self.ends.items():
            pygame.draw.rect(
                self.screen,
                GREEN,
                (x * GRID_SIZE, _cy(y) * GRID_SIZE, GRID_SIZE, GRID_SIZE),
            )
            n = CITY_TO_NUMBER.get(name, "?")
            txt = self.font.render(str(n), True, BLACK)
            self.screen.blit(
                txt,
                txt.get_rect(center=(x * GRID_SIZE + GRID_SIZE // 2, _cy(y) * GRID_SIZE + GRID_SIZE // 2)),
            )

    def _draw_cargo(self, cell, priority: str, dest: str) -> None:
        x, y = cell
        color = RED if str(priority).lower() == "urgent" else PINK
        pygame.draw.circle(
            self.screen,
            color,
            (x * GRID_SIZE + GRID_SIZE // 2, _cy(y) * GRID_SIZE + GRID_SIZE // 2),
            8,
        )
        n = CITY_TO_NUMBER.get(dest, "?")
        txt = self.font.render(str(n), True, BLACK)
        self.screen.blit(
            txt,
            txt.get_rect(center=(x * GRID_SIZE + GRID_SIZE // 2, _cy(y) * GRID_SIZE + GRID_SIZE // 2)),
        )

    def draw_frame(self, idx: int) -> None:
        self._sync_task_state(idx)
        sim_t = self.times[idx] if 0 <= idx < len(self.times) else idx
        self._draw_grid()
        self._draw_stations()
        for start, (x, y) in self.starts.items():
            tid = self._surface_display_task(start, sim_t)
            if tid and tid in self.task_info:
                info = self.task_info[tid]
                self._draw_cargo((x, y), info.get("priority", "Normal"), info.get("end_point", ""))

        cur = self.frames[idx] if 0 <= idx < len(self.frames) else {}
        for name, row in cur.items():
            try:
                x, y = int(row["X"]) - 1, int(row["Y"]) - 1
            except (KeyError, ValueError):
                continue
            if not (0 <= x < 20 and 0 <= y < 20):
                continue
            pygame.draw.rect(
                self.screen,
                self._color(name),
                (x * GRID_SIZE, _cy(y) * GRID_SIZE, GRID_SIZE, GRID_SIZE),
            )
            loaded = str(row.get("loaded", "")).lower() in ("true", "1", "yes")
            tid = str(row.get("task-id") or "").strip()
            if loaded and tid and tid in self.task_info:
                info = self.task_info[tid]
                self._draw_cargo((x, y), info.get("priority", "Normal"), info.get("end_point", ""))
            try:
                pitch = int(row.get("pitch") or 0)
            except ValueError:
                pitch = 0
            import math

            cx = x * GRID_SIZE + GRID_SIZE // 2
            cy = _cy(y) * GRID_SIZE + GRID_SIZE // 2
            ang = math.radians(-pitch)
            pygame.draw.line(
                self.screen,
                BLACK,
                (cx, cy),
                (cx + 12 * math.cos(ang), cy + 12 * math.sin(ang)),
                2,
            )

        pygame.draw.rect(self.screen, HUD_BG, (0, 20 * GRID_SIZE, WINDOW[0], 48))
        lag = max(0, self.max_idx - idx)
        mode = "[PAUSED]" if self.paused else ("[PLAY]" if self.follow else "[HOLD]")
        line = (
            f"frame={idx}/{self.max_idx}  t={sim_t}  "
            f"block={self.n_agv}rows/s  lag={lag}  picked={len(self.completed)}  {mode}"
        )
        self.screen.blit(self.big.render(line, True, HUD_FG), (8, 20 * GRID_SIZE + 12))
        tip = "Space pause | F play | 0 restart | ←/→ ±1s | R reload | Esc quit"
        self.screen.blit(self.font.render(tip, True, (160, 160, 170)), (8, 20 * GRID_SIZE + 32))
        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_f:
                        self.follow = not self.follow
                        if self.follow:
                            self.paused = False
                    elif event.key == pygame.K_0:
                        self.idx = 0
                        self._state_idx = -1
                        self.follow = True
                        self.paused = False
                    elif event.key == pygame.K_r:
                        self._traj_mtime = -1.0
                        self._traj_size = -1
                        self._task_mtime = -1.0
                        self._state_idx = -1
                    elif event.key == pygame.K_LEFT:
                        self.follow = False
                        self.idx = max(0, self.idx - 1)
                        self._state_idx = -1
                    elif event.key == pygame.K_RIGHT:
                        self.follow = False
                        if self.max_idx >= 0:
                            self.idx = min(self.max_idx, self.idx + 1)
                            self._state_idx = -1

            self.refresh()
            # One AGV-block (= one sim second) per displayed frame — never skip.
            if not self.paused and self.follow and self.max_idx >= 0:
                if self.idx < self.max_idx:
                    self.idx += 1
                # else: wait at tip for new complete blocks (live) or stay (finished)

            if self.max_idx >= 0:
                self.draw_frame(self.idx)
            else:
                self.screen.fill(WHITE)
                self.screen.blit(
                    self.big.render("Waiting for trajectory (need full AGV block)…", True, BLACK),
                    (40, 200),
                )
                pygame.display.flip()
            self.clock.tick(self.fps)
        pygame.quit()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Live lifelong AGV monitor")
    out = RESULTS / "lifelong"
    ap.add_argument("--traj", type=Path, default=out / "live_trajectory.csv")
    ap.add_argument("--tasks", type=Path, default=out / "live_tasks.csv")
    ap.add_argument("--position", type=Path, default=POSITION_CSV)
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--no-follow", action="store_true", help="start paused at frame 0")
    ap.add_argument(
        "--start-at-live",
        action="store_true",
        help="start at newest frame (default: start at t=0 and play forward)",
    )
    args = ap.parse_args(argv)
    mon = LiveMonitor(
        traj_csv=args.traj,
        task_csv=args.tasks,
        position_csv=args.position,
        fps=args.fps,
        follow=not args.no_follow,
        start_at_live=args.start_at_live,
    )
    mon.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
