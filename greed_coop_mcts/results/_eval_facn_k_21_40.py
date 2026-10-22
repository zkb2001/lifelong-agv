"""SH21–40: FACN one-shot vs force_fleet_cap sweep — is FACN's K near-best?

Note: these slots were in the v2 oracle used to train facn_v2.pt, so this is a
reproduction / deployment check more than pure held-out generalization.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTHONHASHSEED", "0")

from greed_coop_mcts.paths import RESULTS
from greed_coop_mcts.results._sweep_fleet_k import _geom, run_one

SLOTS = list(range(21, 41))
OUT = RESULTS / "facn_eval_k_21_40.json"
MD = RESULTS / "FACN_EVAL_K_21_40.md"


def run_facn(slot: int, wall: float) -> dict:
    code = (
        "import json,os;"
        "os.environ['PYTHONHASHSEED']='0';"
        "os.environ['OMP_NUM_THREADS']='1';"
        "from greed_coop_mcts.solve import solve;"
        f"r=solve({slot},policy='rrsc',use_facn=True,"
        f"wall_timeout={wall},device='cpu');"
        "print(json.dumps({"
        "'slot':r.get('slot'),'sim':r.get('sim_time'),"
        "'done':r.get('tasks_completed'),'total':r.get('tasks_total'),"
        "'valid':r.get('validate_ok'),"
        "'k':r.get('base_fleet_busy'),"
        "'delta':r.get('facn_delta'),"
        "'n_updates':r.get('n_facn_updates'),"
        "'use_facn':r.get('use_facn'),"
        "}))"
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        return {
            "slot": slot,
            "error": (proc.stderr or "")[-500:],
            "valid": False,
        }
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return json.loads(lines[-1])


def k_range(slot: int) -> List[int]:
    g = _geom(slot)
    if not g:
        return []
    k0 = int(g["k_tbc"])
    n = int(g["n"])
    # small → up to full fleet (cap 16 for wall); always include TBC..n if n small
    lo = max(2, min(k0, 6))
    hi = min(n, 16)
    return list(range(lo, hi + 1))


def main() -> int:
    wall = 900.0
    jobs = 4
    payload: dict = {"slots": SLOTS, "facn": {}, "sweep": {}, "compare": {}}

    # 1) FACN runs (sequential — cleaner logs / one net load pattern)
    for s in SLOTS:
        print(f"[facn] SH{s:02d}", flush=True)
        r = run_facn(s, wall)
        payload["facn"][str(s)] = r
        print(
            f"  -> sim={r.get('sim')} K={r.get('k')} Δ={r.get('delta')} "
            f"updates={r.get('n_updates')} valid={r.get('valid')}",
            flush=True,
        )
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # 2) K sweeps (parallel over (slot,k))
    pending: List[Tuple[int, int]] = []
    for s in SLOTS:
        pending.extend((s, k) for k in k_range(s))
    print(f"[sweep] pending={len(pending)} jobs={jobs}", flush=True)

    rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(run_one, s, k, wall): (s, k) for s, k in pending}
        done_n = 0
        for fut in as_completed(futs):
            s, k = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001
                r = {"slot": s, "k": k, "valid": False, "error": str(exc)[-300:]}
            rows.append(r)
            done_n += 1
            print(
                f"[sweep] SH{s:02d} K={k} sim={r.get('sim')} "
                f"valid={r.get('valid')} ({done_n}/{len(pending)})",
                flush=True,
            )
            # incremental
            by_slot: Dict[str, List[dict]] = {}
            for x in rows:
                by_slot.setdefault(str(int(x["slot"])), []).append(x)
            payload["sweep"] = by_slot
            OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # 3) Compare
    lines = [
        "# FACN vs K-sweep — SH21–40",
        "",
        "FACN one run (may adjust Δ online) vs fixed-`force_fleet_cap` sweep. "
        "**Caveat:** SH21–40 were in the v2 oracle used to train `facn_v2.pt` "
        "(reproduction check, not pure held-out).",
        "",
        "| Slot | TBC | FACN K | Δ | updates | FACN sim | best K | best sim | "
        "FACN−best | rank |",
        "|------|-----|--------|---|---------|----------|--------|----------|"
        "-----------|------|",
    ]
    n_best = n_near = n_miss = 0
    update_slots = 0
    for s in SLOTS:
        g = _geom(s) or {}
        k_tbc = int(g.get("k_tbc") or 6)
        fr = payload["facn"].get(str(s)) or {}
        sw = [
            r
            for r in (payload["sweep"].get(str(s)) or [])
            if r.get("valid")
            and r.get("done") == r.get("total")
            and r.get("sim") is not None
        ]
        if not sw or fr.get("sim") is None:
            lines.append(
                f"| SH{s:02d} | {k_tbc} | {fr.get('k')} | {fr.get('delta')} | "
                f"{fr.get('n_updates')} | {fr.get('sim')} | — | — | — | — |"
            )
            continue
        win = min(sw, key=lambda x: (int(x["sim"]), -int(x["k"])))
        best_k, best_sim = int(win["k"]), int(win["sim"])
        fk = int(fr.get("k") or k_tbc)
        fs = int(fr["sim"])
        dlt = fs - best_sim
        # rank of FACN K among sweep (by sim of that K if present)
        same_k = [r for r in sw if int(r["k"]) == fk]
        if same_k:
            facn_k_sim = int(same_k[-1]["sim"])
        else:
            facn_k_sim = fs
        ordered = sorted(sw, key=lambda x: (int(x["sim"]), -int(x["k"])))
        rank = 1
        for i, r in enumerate(ordered, 1):
            if int(r["k"]) == fk:
                rank = i
                break
        else:
            # FACN K not in sweep range
            rank = -1
        # BEST: hit oracle K, or FACN sim ≤ oracle-best sim (incl. faster-than-sweep).
        if fk == best_k or dlt <= 0:
            n_best += 1
            tag = "BEST"
        elif dlt <= 50 or (best_sim > 0 and dlt / best_sim <= 0.05):
            n_near += 1
            tag = "NEAR"
        else:
            n_miss += 1
            tag = "MISS"
        if int(fr.get("n_updates") or 0) > 0:
            update_slots += 1
        payload["compare"][str(s)] = {
            "k_tbc": k_tbc,
            "facn_k": fk,
            "facn_delta": fr.get("delta"),
            "n_updates": fr.get("n_updates"),
            "facn_sim": fs,
            "best_k": best_k,
            "best_sim": best_sim,
            "gap": dlt,
            "rank": rank,
            "tag": tag,
            "facn_k_sweep_sim": facn_k_sim,
        }
        lines.append(
            f"| SH{s:02d} | {k_tbc} | {fk} | {fr.get('delta')} | "
            f"{fr.get('n_updates')} | {fs} | {best_k} | {best_sim} | "
            f"{dlt:+d} | {tag}/{rank} |"
        )

    lines += [
        "",
        "## Summary",
        "",
        f"- FACN hits best K (or equal sim): **{n_best}/20**",
        f"- Near (≤50 sim or ≤5%): **{n_near}/20**",
        f"- Miss: **{n_miss}/20**",
        f"- Slots with mid-run Δ change (`n_updates>0`): **{update_slots}/20**",
        "",
        "Online adjust: `FacnController.maybe_update` every "
        "`FACN_PERIOD` / TTL / stall events; `n_updates` counts Δ changes "
        "after cold start.",
        "",
    ]
    MD.write_text("\n".join(lines), encoding="utf-8")
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eval] wrote {OUT}", flush=True)
    print(f"[eval] wrote {MD}", flush=True)
    print(
        f"[eval] best={n_best} near={n_near} miss={n_miss} "
        f"online_updates={update_slots}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
