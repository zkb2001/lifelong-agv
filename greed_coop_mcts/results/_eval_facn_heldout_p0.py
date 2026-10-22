"""P0: run FACN with GREED_FACN_CKPT on SH21-40; compare to cached K-sweep."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

from greed_coop_mcts.paths import RESULTS

PRIOR = RESULTS / "facn_eval_k_21_40.json"
OUT = RESULTS / "facn_eval_p0_heldout_21_40.json"
MD = RESULTS / "FACN_P0_HELDOUT.md"
SLOTS = list(range(21, 41))


def run_facn(slot: int, wall: float, ckpt: str) -> dict:
    code = (
        "import json,os;"
        "os.environ['PYTHONHASHSEED']='0';"
        "os.environ['OMP_NUM_THREADS']='1';"
        f"os.environ['GREED_FACN_CKPT']={ckpt!r};"
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
    env["GREED_FACN_CKPT"] = ckpt
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        return {"slot": slot, "valid": False, "error": (proc.stderr or "")[-400:]}
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return json.loads(lines[-1])


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        type=str,
        default=str(RESULTS.parent / "checkpoints" / "facn_p0_heldout.pt"),
    )
    ap.add_argument("--wall", type=float, default=900.0)
    ap.add_argument("--out-md", type=str, default=str(MD))
    ap.add_argument("--out-json", type=str, default=str(OUT))
    ap.add_argument("--title", type=str, default="FACN P0 held-out (SH21–40)")
    args = ap.parse_args()
    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    sweep = prior.get("sweep") or {}
    facn: dict = {}
    for s in SLOTS:
        print(f"[p0] FACN SH{s:02d} ckpt={args.ckpt}", flush=True)
        r = run_facn(s, float(args.wall), str(args.ckpt))
        facn[str(s)] = r
        print(
            f"  -> sim={r.get('sim')} K={r.get('k')} Δ={r.get('delta')} "
            f"upd={r.get('n_updates')}",
            flush=True,
        )

    lines = [
        f"# {args.title}",
        "",
        f"Checkpoint: `{args.ckpt}`",
        "",
        "| Slot | FACN K | Δ | upd | FACN sim | best K | best sim | gap | tag |",
        "|------|--------|---|-----|----------|--------|----------|-----|-----|",
    ]
    n_best = n_near = n_miss = n_upd = 0
    compare = {}
    for s in SLOTS:
        fr = facn.get(str(s)) or {}
        sw = [
            r
            for r in (sweep.get(str(s)) or [])
            if r.get("valid")
            and r.get("done") == r.get("total")
            and r.get("sim") is not None
        ]
        if not sw or fr.get("sim") is None:
            lines.append(
                f"| SH{s:02d} | {fr.get('k')} | {fr.get('delta')} | "
                f"{fr.get('n_updates')} | {fr.get('sim')} | — | — | — | — |"
            )
            continue
        win = min(sw, key=lambda x: (int(x["sim"]), -int(x["k"])))
        best_k, best_sim = int(win["k"]), int(win["sim"])
        fk = int(fr.get("k") or 0)
        fs = int(fr["sim"])
        dlt = fs - best_sim
        if int(fr.get("n_updates") or 0) > 0:
            n_upd += 1
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
        compare[str(s)] = {
            "facn_k": fk,
            "best_k": best_k,
            "facn_sim": fs,
            "best_sim": best_sim,
            "gap": dlt,
            "tag": tag,
        }
        lines.append(
            f"| SH{s:02d} | {fk} | {fr.get('delta')} | {fr.get('n_updates')} | "
            f"{fs} | {best_k} | {best_sim} | {dlt:+d} | {tag} |"
        )
    sims = [int(c["facn_sim"]) for c in compare.values()]
    mean_sim = sum(sims) / max(1, len(sims))
    lines += [
        "",
        "## Summary",
        "",
        f"- BEST: **{n_best}/20**",
        f"- NEAR: **{n_near}/20**",
        f"- MISS: **{n_miss}/20**",
        f"- online updates>0: **{n_upd}/20**",
        f"- mean FACN sim: **{mean_sim:.1f}**",
        "",
    ]
    payload = {"ckpt": args.ckpt, "facn": facn, "compare": compare, "summary": {
        "best": n_best, "near": n_near, "miss": n_miss, "updates": n_upd,
        "mean_sim": mean_sim,
    }}
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[p0] BEST={n_best} NEAR={n_near} MISS={n_miss} "
        f"mean_sim={mean_sim:.1f} → {out_md}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
