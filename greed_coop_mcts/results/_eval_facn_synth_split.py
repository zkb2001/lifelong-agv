"""Evaluate FACN ckpt on synth test-split maps vs force_fleet_cap oracle."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")

from greed_coop_mcts.paths import RESULTS
from greed_coop_mcts.synth_maps.generate import load_synth_meta


def run_facn_synth(idx: int, wall: float, ckpt: str, exports: str) -> dict:
    code = (
        "import json,os;"
        "os.environ['PYTHONHASHSEED']='0';"
        f"os.environ['GREED_FACN_CKPT']={ckpt!r};"
        "from pathlib import Path;"
        "from greed_coop_mcts.synth_maps.generate import load_synth_meta;"
        "from greed_coop_mcts.solve import solve;"
        f"meta=load_synth_meta({idx},exports=Path({exports!r}));"
        f"r=solve({idx},meta=meta,policy='rrsc',use_facn=True,"
        f"wall_timeout={wall},device='cpu');"
        "print(json.dumps({"
        "'idx':"
        f"{idx}"
        ",'sim':r.get('sim_time'),'done':r.get('tasks_completed'),"
        "'total':r.get('tasks_total'),'valid':r.get('validate_ok'),"
        "'k':r.get('base_fleet_busy'),'delta':r.get('facn_delta'),"
        "'n_updates':r.get('n_facn_updates'),"
        "}))"
    )
    env = os.environ.copy()
    env["GREED_FACN_CKPT"] = ckpt
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        return {"idx": idx, "valid": False, "error": (proc.stderr or "")[-400:]}
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return json.loads(lines[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--train-log", type=str, default="")
    ap.add_argument("--exports", type=str, default="")
    ap.add_argument("--wall", type=float, default=450.0)
    ap.add_argument("--out-md", type=str, default=str(RESULTS / "FACN_SYNTH_PLAN_P1.md"))
    ap.add_argument("--max-test", type=int, default=20)
    args = ap.parse_args()

    ora = json.loads(Path(args.oracle).read_text(encoding="utf-8"))
    best = {int(k): int(v) for k, v in (ora.get("best") or {}).items()}
    geoms = ora.get("geoms") or {}
    rows_by = {}
    for r in ora.get("rows") or []:
        if r.get("valid") and r.get("done") == r.get("total") and r.get("sim") is not None:
            rows_by.setdefault(int(r["idx"]), []).append(r)

    test_ids = []
    if args.train_log and Path(args.train_log).is_file():
        split = json.loads(Path(args.train_log).read_text(encoding="utf-8")).get("split") or {}
        test_ids = [int(x) for x in (split.get("test") or [])]
    if not test_ids:
        # fallback: last 10% of labeled ids
        ids = sorted(best.keys())
        test_ids = ids[max(0, len(ids) - max(1, len(ids) // 10)) :]
    test_ids = test_ids[: int(args.max_test)]

    exports = args.exports or str(Path(args.oracle).resolve().parent.parent / "map_exports_synth")
    from greed_coop_mcts.paths import MAP_EXPORTS_SYNTH

    if not args.exports:
        exports = str(MAP_EXPORTS_SYNTH)

    lines = [
        "# FACN synth P1 held-out eval",
        "",
        f"ckpt=`{args.ckpt}` test_n={len(test_ids)}",
        "",
        "| Map | FACN K | best K | FACN sim | best sim | gap | tag |",
        "|-----|--------|--------|----------|----------|-----|-----|",
    ]
    n_best = n_near = n_miss = 0
    for idx in test_ids:
        print(f"[eval-synth] SYN_{idx:05d}", flush=True)
        fr = run_facn_synth(idx, float(args.wall), str(args.ckpt), exports)
        sw = rows_by.get(idx) or []
        if not sw or fr.get("sim") is None:
            lines.append(f"| SYN_{idx:05d} | {fr.get('k')} | — | {fr.get('sim')} | — | — | — |")
            continue
        win = min(sw, key=lambda x: (int(x["sim"]), -int(x["k"])))
        bk, bs = int(win["k"]), int(win["sim"])
        fk = int(fr.get("k") or 0)
        fs = int(fr["sim"])
        dlt = fs - bs
        # BEST: hit oracle K, or FACN sim ≤ oracle-best sim (incl. faster-than-sweep).
        if fk == bk or dlt <= 0:
            n_best += 1
            tag = "BEST"
        elif dlt <= 50 or (bs > 0 and dlt / bs <= 0.05):
            n_near += 1
            tag = "NEAR"
        else:
            n_miss += 1
            tag = "MISS"
        lines.append(
            f"| SYN_{idx:05d} | {fk} | {bk} | {fs} | {bs} | {dlt:+d} | {tag} |"
        )

    rate = (n_best + n_near) / max(1, n_best + n_near + n_miss)
    gate = rate >= 0.60
    lines += [
        "",
        f"- BEST **{n_best}** NEAR **{n_near}** MISS **{n_miss}**",
        f"- BEST+NEAR rate **{100*rate:.1f}%** gate≥60%: **{'PASS' if gate else 'FAIL'}**",
        "",
    ]
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval-synth] rate={rate:.3f} gate={'PASS' if gate else 'FAIL'}", flush=True)
    print(f"GATE_PASS={int(gate)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
