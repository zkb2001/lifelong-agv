"""Train Transformer RH-PP then eval vs M0/M2 on original CSV."""
from __future__ import annotations

import argparse
import json
import time

from ml_research.benchmarks.runner import run_m0, run_m2, _run_with_allocator
from ml_research.common.paths import POSITION_CSV, RESULTS, TASK_CSV

from .allocator_transformer import allocator_transformer_pp, load_net, reset_load
from .train_bc_ppo import light_ppo, train_bc, train_sil

OUT = RESULTS / "rl_rh_pp"
OUT.mkdir(parents=True, exist_ok=True)


def run_tf(
    task_csv=TASK_CSV,
    position_csv=POSITION_CSV,
    max_time=2000,
    scenario_tag="original",
    wall_timeout=None,
):
    reset_load()
    load_net(force=True)
    return _run_with_allocator(
        "TF_RH_PP_transformer",
        task_csv,
        position_csv,
        max_time,
        allocator=allocator_transformer_pp,
        notes="Paper-style Transformer AR priority + PP (baseline untouched)",
        scenario_tag=scenario_tag,
        wall_timeout=wall_timeout,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--snaps", type=int, default=400)
    ap.add_argument("--ppo", type=int, default=10)
    ap.add_argument("--max-time", type=int, default=2000)
    ap.add_argument("--wall", type=float, default=600.0)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-m0", action="store_true")
    ap.add_argument("--skip-m2", action="store_true")
    ap.add_argument("--sil", action="store_true", help="run SIL after train (or alone with --skip-train)")
    args = ap.parse_args()

    if not args.skip_train:
        train_bc(epochs=args.epochs, max_snaps=args.snaps, collect=True)
        if args.ppo > 0:
            light_ppo(updates=args.ppo)
    if args.sil:
        train_sil(max_time=args.max_time)

    rows = []
    t_all = time.perf_counter()

    def _go(name, fn):
        print(f"\n===== {name} =====", flush=True)
        t0 = time.perf_counter()
        r = dict(
            fn(
                TASK_CSV,
                POSITION_CSV,
                max_time=args.max_time,
                scenario_tag="original_tf_rh_pp",
                wall_timeout=args.wall,
            )
        )
        r["label"] = name
        r["wall_clock"] = round(time.perf_counter() - t0, 2)
        print(
            f"[{name}] done={r.get('tasks_completed')}/{r.get('tasks_total')} "
            f"ratio={r.get('completion_ratio')} sim={r.get('sim_time')} "
            f"wall={r['wall_clock']}s",
            flush=True,
        )
        rows.append(r)
        (OUT / f"eval_{name}.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if not args.skip_m0:
        _go("M0_baseline", run_m0)
    if not args.skip_m2:
        _go("M2_priority_pp", run_m2)
    _go("TF_RH_PP", run_tf)

    summary = {
        "scenario": "original",
        "methods": [
            {
                "label": r["label"],
                "completion_ratio": r.get("completion_ratio"),
                "sim_time": r.get("sim_time"),
                "tasks_completed": r.get("tasks_completed"),
                "tasks_total": r.get("tasks_total"),
                "urgent_on_time": r.get("urgent_on_time"),
                "urgent_late": r.get("urgent_late"),
                "n_conflicts": r.get("n_conflicts"),
                "wall_clock": r.get("wall_clock"),
                "forced_stop": r.get("forced_stop"),
            }
            for r in rows
        ],
        "total_wall": round(time.perf_counter() - t_all, 2),
        "note": "TF_RH_PP = Transformer AR priority (RL-RH-PP) + PP; BC then RL then optional SIL.",
    }
    path = OUT / "transformer_eval.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
