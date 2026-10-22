"""Compare FULL vs current suite under a fixed PYTHONHASHSEED."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FULL = {
    1: 965,
    2: 2021,
    3: 4440,
    4: 1053,
    5: 1616,
    6: 2024,
    7: 3897,
    8: 1320,
    9: 1337,
    10: 2215,
    11: 2384,
    12: 1558,
    13: 832,
    14: 1216,
    15: 2523,
    16: 1643,
    17: 2395,
    18: 1711,
    19: 1446,
    20: 947,
}


def run_slot(slot: int, seed: str) -> dict:
    code = f"""
import json, os
os.environ['PYTHONHASHSEED'] = '{seed}'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from greed_coop_mcts.batch import _run_slot
print(json.dumps(_run_slot({slot}, wall_timeout=1800.0, max_sim_time=200000,
    sims=24, k=3, device='cpu', policy='rrsc', use_cgpn=False, result_tag='gcm')))
"""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = seed
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    if r.returncode != 0:
        return {"slot": slot, "error": r.stderr[-500:], "validate_ok": False}
    lines = [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    return json.loads(lines[-1])


def main() -> int:
    seed = sys.argv[1] if len(sys.argv) > 1 else "2"
    slots = list(range(1, 21))
    if len(sys.argv) > 2:
        # e.g. 11,15,3
        slots = []
        for part in sys.argv[2].split(","):
            if "-" in part:
                a, b = part.split("-", 1)
                slots.extend(range(int(a), int(b) + 1))
            else:
                slots.append(int(part))
    rows = []
    worse = []
    sims = []
    for s in slots:
        rep = run_slot(s, seed)
        sim = rep.get("sim_time")
        ok = bool(rep.get("validate_ok")) and int(rep.get("tasks_completed") or 0) == 100
        base = FULL[s]
        dlt = int(sim) - base if sim is not None else None
        dlt_s = f"{dlt:+d}" if dlt is not None else "NA"
        print(
            f"SH{s:02d} seed={seed} sim={sim} FULL={base} dlt={dlt_s} ok={ok}",
            flush=True,
        )
        rows.append(rep)
        if ok and sim is not None:
            sims.append(int(sim))
            if int(sim) > base:
                worse.append((s, int(sim) - base))
    if sims:
        avg = sum(sims) / len(sims)
        favg = sum(FULL[s] for s in slots) / len(slots)
        print(f"avg={avg:.1f} FULL_subset={favg:.1f} delta={avg - favg:+.1f} worse={worse}")
    out = Path(__file__).resolve().parent / f"seed_{seed}_summary.json"
    out.write_text(json.dumps({"seed": seed, "rows": rows}, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
