"""Batch-render SH01–20 trajectory CSVs to MP4.

Default: MP4 duration_sec = sim_time / 5 (playback fps ≈ 5).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from greed_coop_mcts.paths import RESULTS, TRAJ
from greed_coop_mcts.scenario import load_slot_meta
from ml_research.tools.render_video import render_mp4

VIDEO_DIR = RESULTS / "videos"


def _parse_slots(s: str) -> list[int]:
    slots: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            slots.extend(range(int(a), int(b) + 1))
        else:
            slots.append(int(part))
    return sorted(set(slots))


def _sim_time(slot: int) -> int | None:
    p = RESULTS / f"SH{int(slot):02d}_gcm_gcm.json"
    if not p.is_file():
        return None
    try:
        return int(json.loads(p.read_text(encoding="utf-8")).get("sim_time"))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", default="1-20")
    ap.add_argument(
        "--duration",
        type=float,
        default=-1.0,
        help="fixed MP4 length (sec); default -1 → sim_time * --sec-per-sim",
    )
    ap.add_argument(
        "--sec-per-sim",
        type=float,
        default=0.2,
        help="when --duration < 0: duration = sim_time * sec_per_sim "
        "(default 0.2 → duration = sim/5)",
    )
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out-dir", default=str(VIDEO_DIR))
    ap.add_argument(
        "--delete-existing",
        action="store_true",
        help="delete existing SH*_gcm.mp4 in out-dir before render",
    )
    args = ap.parse_args()

    slots = _parse_slots(args.slots)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.delete_existing:
        for s in slots:
            old = out_dir / f"SH{s:02d}_gcm.mp4"
            if old.is_file():
                old.unlink()
                print(f"[video] deleted {old.name}", flush=True)

    ok = fail = 0
    for s in slots:
        meta = load_slot_meta(s)
        traj = TRAJ / f"SH{s:02d}_gcm_gcm.csv"
        if meta is None:
            print(f"[video] SH{s:02d} SKIP meta missing", flush=True)
            fail += 1
            continue
        if not traj.is_file():
            print(f"[video] SH{s:02d} SKIP traj missing {traj}", flush=True)
            fail += 1
            continue

        if float(args.duration) > 0:
            duration = float(args.duration)
        else:
            sim = _sim_time(s)
            if sim is None or sim <= 0:
                print(f"[video] SH{s:02d} SKIP sim_time missing in JSON", flush=True)
                fail += 1
                continue
            duration = float(sim) * float(args.sec_per_sim)

        out = out_dir / f"SH{s:02d}_gcm.mp4"
        print(
            f"[video] SH{s:02d} -> {out.name} duration={duration:.1f}s "
            f"(sim*{args.sec_per_sim})",
            flush=True,
        )
        rc = render_mp4(
            traj,
            out,
            task_csv=Path(meta["task_csv"]),
            position_csv=Path(meta["position_csv"]),
            obstacles_json=Path(meta["map_json"]),
            speed=float(args.speed),
            duration=float(duration),
        )
        if rc == 0 and out.is_file() and out.stat().st_size > 1000:
            print(
                f"[video] SH{s:02d} OK size={out.stat().st_size // 1024}KB",
                flush=True,
            )
            ok += 1
        else:
            print(f"[video] SH{s:02d} FAIL rc={rc}", flush=True)
            fail += 1
    print(f"[video] done ok={ok} fail={fail} dir={out_dir}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
