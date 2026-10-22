"""SH03 lifelong pygame viewer — same renderer as MP4 (simulation/display.py).

Plays a contiguous sim-time axis: play_t += 1 every frame (never skips).
Tails an append-only viz CSV so the timeline does not jump from rolling rewrites.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pygame

from simulation import display as disp

from greed_coop_mcts.paths import MAP_EXPORTS, RESULTS, TRAJ

DEFAULT_TRAJ = TRAJ / "SH_custom_03_viz.csv"
DEFAULT_TASKS = RESULTS / "SH03_lifelong_seed400_task.csv"
DEFAULT_POS = MAP_EXPORTS / "SH_custom_03_position.csv"
DEFAULT_OBS = MAP_EXPORTS / "SH_custom_03.json"


def _load_obstacles(path: Path) -> list:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        pts = raw.get("extra_obstacles") or raw.get("obstacles") or []
    elif isinstance(raw, list):
        pts = raw
    else:
        pts = []
    out = []
    for p in pts:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            out.append((int(p[0]), int(p[1])))
        elif isinstance(p, dict):
            out.append((int(p["x"]), int(p["y"])))
    return out


class LifelongDisplay(disp.DisplayManager):
    """MP4-identical drawing + contiguous play_t axis."""

    def __init__(self, *args, play_fps: float = 3.0, **kwargs):
        kwargs["record"] = False
        super().__init__(*args, **kwargs)
        self.play_fps = float(play_fps)
        self._traj_mtime = -1.0
        self._traj_size = -1
        self.play_t: int | None = None  # contiguous sim clock
        self._last_good_t: int | None = None

    def refresh_traj(self) -> None:
        path = Path(self.trajectory_file)
        if not path.is_file():
            return
        st = path.stat()
        # Append-only: size grows; also handle rare compact (size shrink).
        if st.st_mtime == self._traj_mtime and st.st_size == self._traj_size:
            return
        shrunk = self._traj_size > 0 and st.st_size < self._traj_size
        self.load_trajectory_data()
        self._ingest_unknown_tasks()
        self._traj_mtime = st.st_mtime
        self._traj_size = st.st_size
        if not self.traj_by_t:
            return
        t_min = min(self.traj_by_t)
        t_max = max(self.traj_by_t)
        if self.play_t is None:
            self.play_t = int(t_min)
        elif shrunk and self.play_t < t_min:
            # Compact dropped unplayed history — unavoidable; resume at new start.
            print(
                f"[lifelong-display] viz compacted; resume t {self.play_t} -> {t_min}",
                flush=True,
            )
            self.play_t = int(t_min)
        elif self.play_t > t_max + 1:
            # Should not happen with append; clamp gently.
            self.play_t = int(t_max)

    def _ingest_unknown_tasks(self) -> None:
        for tmap in self.traj_by_t.values():
            for row in tmap.values():
                tid = str(row.get("task-id") or "").strip()
                if not tid or tid in self.task_info:
                    continue
                dest = str(row.get("destination") or "").strip()
                if not dest:
                    continue
                start = tid.rsplit("-", 1)[0] if "-" in tid else ""
                self.task_info[tid] = {
                    "start_point": start,
                    "end_point": dest,
                    "priority": "Normal",
                }
                if start and tid not in self.start_point_tasks.get(start, []):
                    self.start_point_tasks.setdefault(start, []).append(tid)
                    self.task_sequence.setdefault(start, []).append(tid)
                    self.tasks_per_start[start] = self.tasks_per_start.get(start, 0) + 1
                    self.remaining_tasks[start] = self.remaining_tasks.get(start, 0) + 1
                    if start not in self.current_task_index:
                        self.current_task_index[start] = 0
                name = row.get("name")
                if name and name not in self.agv_colors:
                    i = len(self.agv_colors)
                    self.agv_colors[name] = disp.AGV_COLORS[i % len(disp.AGV_COLORS)]

    def run_live(self) -> None:
        pygame.display.set_caption(
            "SH03 Lifelong (MP4-style) — contiguous t, 1 sim-sec/frame"
        )
        clock = pygame.time.Clock()
        running = True
        paused = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_0:
                        self.play_t = min(self.traj_by_t) if self.traj_by_t else None
                        self.completed_tasks.clear()
                        self.current_tasks.clear()
                        self.n_picked = 0
                        self.n_delivered = 0
                        self.remaining_tasks = self.tasks_per_start.copy()
                        self.initialize_task_sequence()

            self.refresh_traj()
            if not self.traj_by_t or self.play_t is None:
                self.screen.fill(disp.WHITE)
                msg = self.font.render("Waiting for viz trajectory…", True, disp.BLACK)
                self.screen.blit(msg, (40, 200))
                pygame.display.flip()
                clock.tick(self.play_fps)
                continue

            t_max = max(self.traj_by_t)
            t = int(self.play_t)

            if paused:
                draw_t = t if t in self.traj_by_t else (
                    self._last_good_t if self._last_good_t is not None else t_max
                )
                self.draw_frame(draw_t)
                hud = self.font.render(
                    f"t={draw_t}  [PAUSE]  fps={self.play_fps:.0f}  done≈{self.n_delivered}",
                    True,
                    disp.RED,
                )
                self.screen.blit(hud, (8, 8))
                pygame.display.flip()
                clock.tick(self.play_fps)
                continue

            if t in self.traj_by_t:
                self.draw_frame(t)
                self._last_good_t = t
                lag = max(0, t_max - t)
                hud = self.font.render(
                    f"t={t}/{t_max}  lag={lag}  fps={self.play_fps:.0f}  "
                    f"[PLAY]  done≈{self.n_delivered}",
                    True,
                    disp.RED,
                )
                self.screen.blit(hud, (8, 8))
                pygame.display.flip()
                # Contiguous clock: always +1, never jump to sparse index.
                self.play_t = t + 1
            elif t > t_max:
                # Live tip — hold last available frame until next sim-second arrives.
                hold = self._last_good_t if self._last_good_t is not None else t_max
                self.draw_frame(hold)
                hud = self.font.render(
                    f"t={hold}/{t_max}  LIVE TIP  waiting…  fps={self.play_fps:.0f}",
                    True,
                    disp.RED,
                )
                self.screen.blit(hud, (8, 8))
                pygame.display.flip()
            else:
                # Missing middle tick (should be rare with append-only): hold pose, still +1.
                hold = self._last_good_t if self._last_good_t is not None else t_max
                self.draw_frame(hold)
                hud = self.font.render(
                    f"t={t} (hold {hold})  gap-fill  fps={self.play_fps:.0f}",
                    True,
                    disp.RED,
                )
                self.screen.blit(hud, (8, 8))
                pygame.display.flip()
                self.play_t = t + 1

            clock.tick(self.play_fps)

        pygame.quit()


def main() -> int:
    ap = argparse.ArgumentParser(description="SH03 lifelong MP4-style live display")
    ap.add_argument("--traj", type=Path, default=DEFAULT_TRAJ)
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--position", type=Path, default=DEFAULT_POS)
    ap.add_argument("--obstacles", type=Path, default=DEFAULT_OBS)
    ap.add_argument(
        "--fps",
        type=float,
        default=float(disp.FPS),
        help="wall-clock frames/sec; each frame = exactly 1 sim second (default 3)",
    )
    args = ap.parse_args()

    deadline = time.time() + 60
    while time.time() < deadline:
        if args.traj.is_file() and args.traj.stat().st_size > 200:
            break
        time.sleep(0.5)

    obs = _load_obstacles(args.obstacles)
    print(
        f"[lifelong-display] traj={args.traj} fps={args.fps} "
        f"obstacles={len(obs)} contiguous play_t",
        flush=True,
    )
    viewer = LifelongDisplay(
        speed=1,
        record=False,
        video_filename="unused.mp4",
        task_file=str(args.tasks),
        trajectory_file=str(args.traj),
        position_file=str(args.position),
        obstacle_cells=obs,
        play_fps=float(args.fps),
    )
    viewer.run_live()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
