"""Render a trajectory CSV to MP4 via simulation/display.py."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from ml_research.common.paths import DATA_DIR, POSITION_CSV, ROOT, TASK_CSV

DISPLAY = ROOT / "simulation" / "display.py"
VIDEO_DIR = DATA_DIR / "outputs" / "videos"


def render_mp4(
    traj_csv: Path,
    mp4_path: Path,
    *,
    task_csv: Path = TASK_CSV,
    position_csv: Path = POSITION_CSV,
    obstacles_json: Path | None = None,
    speed: float = 1.0,
    duration: float | None = None,
) -> int:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    mp4_path = Path(mp4_path)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SDL_VIDEODRIVER"] = "dummy"
    env["SDL_AUDIODRIVER"] = "dummy"
    cmd = [
        sys.executable,
        str(DISPLAY),
        str(speed),
        "--record",
        str(mp4_path),
        "--task",
        str(task_csv),
        "--traj",
        str(traj_csv),
        "--position",
        str(position_csv),
    ]
    if obstacles_json and Path(obstacles_json).exists():
        cmd.extend(["--obstacles", str(obstacles_json)])
    if duration is not None:
        cmd.extend(["--duration", str(duration)])
    return subprocess.call(cmd, env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default=str(TASK_CSV))
    ap.add_argument("--position", default=str(POSITION_CSV))
    ap.add_argument("--obstacles", default="")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()
    obs = Path(args.obstacles) if args.obstacles else None
    return render_mp4(
        Path(args.traj),
        Path(args.out),
        task_csv=Path(args.task),
        position_csv=Path(args.position),
        obstacles_json=obs,
        speed=args.speed,
        duration=args.duration,
    )


if __name__ == "__main__":
    raise SystemExit(main())
