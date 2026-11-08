"""Probe synth-map completion under current rrsc+FACN (skip traj)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "greed_coop_mcts" / "map_exports_synth"
CKPT = ROOT / "greed_coop_mcts" / "checkpoints" / "facn_k_v1.pt"
OUT = ROOT / "greed_coop_mcts" / "results" / "synth_completion_probe.json"


def _one(idx: int, wall: float, max_sim: int) -> dict:
    code = (
        "import json,os,time;"
        "os.environ['PYTHONHASHSEED']='0';"
        "os.environ['GREED_SKIP_TRAJ']='1';"
        "os.environ['GREED_LIVELOCK_MODE']='full';"
        f"os.environ['GREED_FACN_CKPT']={str(CKPT)!r};"
        "from pathlib import Path;"
        "from greed_coop_mcts.synth_maps.generate import load_synth_meta;"
        "from greed_coop_mcts.solve import solve;"
        f"meta=dict(load_synth_meta({idx},exports=Path({str(EXPORTS)!r})));"
        "meta['n_tasks']=100;"
        "t0=time.perf_counter();"
        f"r=solve({idx},meta=meta,policy='rrsc',use_facn=True,"
        f"wall_timeout={wall},max_sim_time={max_sim},device='cpu');"
        "print(json.dumps({"
        f"'idx':{idx},"
        "'done':r.get('tasks_completed'),'total':r.get('tasks_total'),"
        "'sim':r.get('sim_time'),'valid':r.get('validate_ok'),"
        "'wall':round(time.perf_counter()-t0,2),"
        "'timed_out':bool(r.get('timed_out') or r.get('wall_timeout')),"
        "}))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONHASHSEED"] = "0"
    env["GREED_SKIP_TRAJ"] = "1"
    env["GREED_LIVELOCK_MODE"] = "full"
    env["GREED_FACN_CKPT"] = str(CKPT)
    proc = subprocess.run(
        [sys.executable, "-u", "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        return {
            "idx": idx,
            "done": 0,
            "total": 100,
            "complete": False,
            "error": (proc.stderr or proc.stdout or "")[-500:],
        }
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {"idx": idx, "done": 0, "total": 100, "complete": False, "error": "no_json"}
    row = json.loads(lines[-1])
    done = int(row.get("done") or 0)
    total = int(row.get("total") or 100)
    row["complete"] = done >= total and bool(row.get("valid"))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="hard", choices=("easy", "mid", "hard", "all"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wall", type=float, default=300.0)
    ap.add_argument("--max-sim", type=int, default=40000)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    index = json.loads((EXPORTS / "index.json").read_text(encoding="utf-8"))
    by: dict[str, list[int]] = {"easy": [], "mid": [], "hard": []}
    for r in index:
        by[str(r["regime"])].append(int(r["idx"]))

    ids: list[int] = []
    regimes = ("easy", "mid", "hard") if args.regime == "all" else (args.regime,)
    for reg in regimes:
        chunk = by[reg][args.offset : args.offset + args.n]
        ids.extend(chunk)

    print(
        f"[probe] regime={args.regime} n={len(ids)} workers={args.workers} "
        f"wall={args.wall} max_sim={args.max_sim}",
        flush=True,
    )
    rows: list[dict] = []
    incomplete: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, i, args.wall, args.max_sim): i for i in ids}
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            ok = bool(row.get("complete"))
            tag = "OK" if ok else "FAIL"
            print(
                f"  [{tag}] SYN_{int(row['idx']):05d} "
                f"done={row.get('done')}/{row.get('total')} "
                f"sim={row.get('sim')} wall={row.get('wall')}",
                flush=True,
            )
            if not ok:
                incomplete.append(row)

    rows.sort(key=lambda x: int(x["idx"]))
    summary = {
        "regime": args.regime,
        "n": len(rows),
        "complete": sum(1 for r in rows if r.get("complete")),
        "incomplete": len(incomplete),
        "wall": args.wall,
        "max_sim": args.max_sim,
        "incomplete_ids": [int(r["idx"]) for r in incomplete],
        "rows": rows,
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[probe] done complete={summary['complete']}/{summary['n']} "
        f"incomplete={summary['incomplete']} -> {OUT}",
        flush=True,
    )
    if incomplete:
        print(f"[probe] FAIL ids={summary['incomplete_ids']}", flush=True)
    return 0 if not incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
