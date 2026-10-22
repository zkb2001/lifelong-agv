"""Retest ghost-border fix: former INVALID gen maps + SH01/12/17 smoke."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from greed_coop_mcts.results._gen_eval_curriculum import eval_one  # noqa: E402
from greed_coop_mcts.scenario import load_slot_meta  # noqa: E402
from greed_coop_mcts.solve import solve  # noqa: E402
from greed_coop_mcts.validate import format_validation_summary, validate_trajectory  # noqa: E402

ROOT = Path(__file__).resolve().parent


def eval_prod(slot: int) -> dict:
    meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
    assert meta
    meta = dict(meta)
    meta["id"] = f"SH{slot:02d}_borderfix"
    rep = solve(
        slot,
        meta=meta,
        policy="rrsc",
        use_stage_net=True,
        device="cpu",
        wall_timeout=600.0,
        max_sim_time=200000,
    )
    traj = Path(rep["trajectory"])
    v = validate_trajectory(meta, traj)
    return {
        "slot": slot,
        "sim": int(rep.get("sim_time") or -1),
        "done": int(rep.get("tasks_completed") or 0),
        "ok": bool(v.get("ok")) and int(rep.get("tasks_completed") or 0) == 100,
        "mot": int((v.get("issues") or {}).get("n_illegal_motion") or 0),
        "vnote": format_validation_summary(v),
        "wall": rep.get("wall_seconds"),
    }


def main() -> int:
    gen_slots = [23, 25, 26, 28, 29, 31, 33, 39]  # former INVALID samples
    print("=== GEN (was INVALID) ===", flush=True)
    gen_rows = []
    for s in gen_slots:
        r = eval_one(s)
        gen_rows.append(r)
        print(
            f"SH{s:02d} sim={r['sim']} done={r['done']} ok={r['ok']} mot={r['mot']} "
            f"wall={r['wall']}",
            flush=True,
        )

    print("\n=== PROD smoke SH01/12/17 ===", flush=True)
    prod_rows = []
    for s in (1, 12, 17):
        r = eval_prod(s)
        prod_rows.append(r)
        print(
            f"SH{s:02d} sim={r['sim']} done={r['done']} ok={r['ok']} mot={r['mot']} "
            f"wall={r['wall']} {r['vnote']}",
            flush=True,
        )

    out = {
        "gen": gen_rows,
        "prod": prod_rows,
        "gen_ok": sum(1 for r in gen_rows if r["ok"]),
        "prod_ok": sum(1 for r in prod_rows if r["ok"]),
    }
    (ROOT / "BORDER_FIX_RETEST.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"\ngen VALID {out['gen_ok']}/{len(gen_rows)}  prod VALID {out['prod_ok']}/{len(prod_rows)}",
        flush=True,
    )
    return 0 if out["gen_ok"] == len(gen_rows) and out["prod_ok"] == len(prod_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
