"""Ablation: GREED_LIVELOCK_MODE ∈ {off, detect, plan, full} on SH03.

Fixed budget: wall_timeout seconds, same 400-seed + lifelong stream cap.
Metrics from result JSON + log unlock lines.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "greed_coop_mcts" / "results"
SEED = RESULTS / "SH03_lifelong_seed400_task.csv"
OUT_JSON = RESULTS / "ablate_livelock_ai.json"
CKPT = ROOT / "greed_coop_mcts" / "checkpoints" / "facn_k_v1.pt"

MODES = ("off", "detect", "plan", "full")
UNLOCK_RE = re.compile(
    r"\[greed_coop\] (?P<tag>livelock-AI|livelock-detect|livelock-plan|corridor-unlock soft)"
    r" t=(?P<t>\d+) stall=(?P<stall>\d+).*?done=(?P<done>\d+)/(?P<total>\d+)"
)
PROG_RE = re.compile(
    r"\[SH_custom_03/gcm\] t=(?P<t>\d+) done=(?P<done>\d+)/(?P<total>\d+).*?wall=(?P<wall>[\d.]+)s"
)


def run_mode(mode: str, *, wall: float, max_total: int, seed: int) -> dict:
    log = RESULTS / f"ablate_livelock_{mode}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["GREED_SKIP_TRAJ"] = "1"
    env["GREED_VIZ_TRAJ"] = "0"
    env["GREED_LIVELOCK_MODE"] = mode
    env["GREED_FACN_CKPT"] = str(CKPT)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "greed_coop_mcts.solve",
        "--slot",
        "3",
        "--policy",
        "rrsc",
        "--device",
        "cpu",
        "--facn",
        "--lifelong-random",
        "--lifelong-seed",
        str(seed),
        "--lifelong-batch",
        "24",
        "--lifelong-low-water",
        "8",
        "--lifelong-max-total",
        str(max_total),
        "--task-csv",
        str(SEED),
        "--wall-timeout",
        str(wall),
        "--max-sim-time",
        "300000",
    ]
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as f:
        f.write(f"==== ablate mode={mode} wall={wall} max_total={max_total} ====\n")
        f.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_used = time.perf_counter() - t0
    text = log.read_text(encoding="utf-8", errors="replace")
    unlocks = []
    max_unlock_stall = 0
    for m in UNLOCK_RE.finditer(text):
        st = int(m.group("stall"))
        unlocks.append(
            {
                "tag": m.group("tag"),
                "t": int(m.group("t")),
                "stall": st,
                "done": int(m.group("done")),
            }
        )
        max_unlock_stall = max(max_unlock_stall, st)
    # approx max stall from progress plateaus: when done unchanged across lines
    max_stall_proxy = max_unlock_stall
    done_series = []
    for m in PROG_RE.finditer(text):
        done_series.append(
            (int(m.group("t")), int(m.group("done")), float(m.group("wall")))
        )
    # stall proxy: gaps in done between progress prints * (approx)
    # better: parse stall from unlock only + final JSON
    rep_path = RESULTS / "SH_custom_03_gcm.json"
    rep = {}
    if rep_path.is_file():
        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rep = {}
    # time to 300 / 400 done
    t_at = {}
    for thr in (100, 200, 300, 400):
        hit = next((x for x in done_series if x[1] >= thr), None)
        t_at[f"sim_to_{thr}"] = hit[0] if hit else None
        t_at[f"wall_to_{thr}"] = hit[2] if hit else None

    return {
        "mode": mode,
        "exit_code": int(proc.returncode),
        "wall_budget_s": float(wall),
        "wall_used_s": round(wall_used, 2),
        "tasks_completed": int(rep.get("tasks_completed") or 0),
        "tasks_total": int(rep.get("tasks_total") or 0),
        "sim_time": int(rep.get("sim_time") or 0),
        "n_unlocks": len(unlocks),
        "unlock_tags": {u["tag"]: 0 for u in unlocks}
        if False
        else None,
        "unlock_tag_counts": _count_tags(unlocks),
        "max_unlock_stall": int(max_unlock_stall),
        "mean_unlock_stall": (
            round(sum(u["stall"] for u in unlocks) / len(unlocks), 1)
            if unlocks
            else None
        ),
        "first_unlock_stall": unlocks[0]["stall"] if unlocks else None,
        **t_at,
        "log": str(log),
    }


def _count_tags(unlocks: list) -> dict:
    d: dict = {}
    for u in unlocks:
        d[u["tag"]] = d.get(u["tag"], 0) + 1
    return d


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--wall", type=float, default=180.0, help="wall budget per mode")
    ap.add_argument("--max-total", type=int, default=600)
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument(
        "--modes",
        default="off,detect,plan,full",
        help="comma modes",
    )
    args = ap.parse_args()
    if not SEED.is_file():
        print(f"missing seed csv {SEED}", flush=True)
        return 1
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    rows = []
    for mode in modes:
        print(f"\n===== RUN mode={mode} wall={args.wall}s =====", flush=True)
        row = run_mode(
            mode, wall=float(args.wall), max_total=int(args.max_total), seed=int(args.seed)
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
        # brief pause so result json settles
        time.sleep(1.0)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "wall_budget_s": float(args.wall),
        "max_total": int(args.max_total),
        "seed": int(args.seed),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}", flush=True)

    # Markdown-ish table
    print("\n| mode | done | sim | wall | unlocks | first_stall | max_stall | mean_stall | sim@300 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['mode']} | {r['tasks_completed']} | {r['sim_time']} | "
            f"{r['wall_used_s']} | {r['n_unlocks']} | {r['first_unlock_stall']} | "
            f"{r['max_unlock_stall']} | {r['mean_unlock_stall']} | {r.get('sim_to_300')} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
