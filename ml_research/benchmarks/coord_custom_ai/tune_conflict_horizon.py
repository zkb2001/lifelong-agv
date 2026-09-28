"""Offline Optuna search for Conflict-Horizon heuristic knobs (CHTuning).

Objective: minimize sim_time on a fixed scenario subject to validate_ok and
full task completion.

Example:
  python -m ml_research.benchmarks.coord_custom_ai.tune_conflict_horizon \\
    --slot 3 --max-tasks 20 --max-active 4 --trials 40
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.solve_conflict_horizon import (
    CHTuning,
    DEFAULT_CH_TUNING,
    solve_conflict_horizon,
)
from ml_research.common.paths import RESULTS

TUNING_OUT = RESULTS / "coord_custom_ai" / "conflict_horizon" / "tuning"
FAIL_PENALTY = 50_000.0


def _score(rep: dict) -> float:
    ok = (
        bool(rep.get("validate_ok"))
        and float(rep.get("completion_ratio") or 0) >= 0.999
        and not rep.get("stall_exit")
    )
    sim = float(rep.get("sim_time") or 0)
    return sim if ok else FAIL_PENALTY + sim


def run_study(
    *,
    slot: int,
    max_tasks: int,
    max_active: int,
    deadline: int,
    trials: int,
    seed: int,
    storage: str,
    study_name: str,
) -> dict:
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna is required: pip install optuna (see ml_research/requirements.txt)"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    TUNING_OUT.mkdir(parents=True, exist_ok=True)

    def objective(trial: "optuna.Trial") -> float:
        tuning = CHTuning(
            recent_parks_k=trial.suggest_int("recent_parks_k", 3, 10),
            idle_path_cap=trial.suggest_int("idle_path_cap", 4, 16),
            dispatch_lambda=trial.suggest_int("dispatch_lambda", 0, 5),
            staging_max_d=trial.suggest_int("staging_max_d", 8, 22),
            idle_hold_dist=trial.suggest_int("idle_hold_dist", 6, 16),
        )
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            rep = solve_conflict_horizon(
                slot,
                max_tasks=max_tasks,
                max_active=max_active,
                deadline=deadline,
                tuning=tuning,
                quiet=True,
            )
        score = _score(rep)
        trial.set_user_attr("sim_time", rep.get("sim_time"))
        trial.set_user_attr("validate_ok", rep.get("validate_ok"))
        trial.set_user_attr("completion_ratio", rep.get("completion_ratio"))
        trial.set_user_attr("wall_seconds", round(time.perf_counter() - t0, 2))
        trial.set_user_attr("tuning", tuning.to_dict())
        print(
            f"[tune] trial={trial.number} score={score:.0f} "
            f"sim_t={rep.get('sim_time')} ok={rep.get('validate_ok')} "
            f"done={rep.get('tasks_completed')}/{rep.get('tasks_total')} "
            f"tuning={tuning.to_dict()}",
            flush=True,
        )
        return score

    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="minimize",
        )
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = study.best_trial
    best_tuning = CHTuning.from_dict(best.user_attrs.get("tuning", best.params))
    summary = {
        "slot": slot,
        "max_tasks": max_tasks,
        "max_active": max_active,
        "deadline": deadline,
        "trials": trials,
        "best_score": best.value,
        "best_sim_time": best.user_attrs.get("sim_time"),
        "best_validate_ok": best.user_attrs.get("validate_ok"),
        "best_completion_ratio": best.user_attrs.get("completion_ratio"),
        "best_tuning": best_tuning.to_dict(),
        "default_tuning": DEFAULT_CH_TUNING.to_dict(),
        "n_trials_completed": len(study.trials),
    }
    out_path = TUNING_OUT / f"SH_custom_{slot:02d}_tuning_best.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slot", type=int, default=3)
    ap.add_argument("--max-tasks", type=int, default=20)
    ap.add_argument("--max-active", type=int, default=4)
    ap.add_argument("--deadline", type=int, default=5000)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--storage",
        default="",
        help="Optuna storage URL (optional, enables resume)",
    )
    ap.add_argument("--study-name", default="ch_sh03_tuning")
    args = ap.parse_args()

    summary = run_study(
        slot=args.slot,
        max_tasks=args.max_tasks,
        max_active=args.max_active,
        deadline=args.deadline,
        trials=args.trials,
        seed=args.seed,
        storage=args.storage or "",
        study_name=args.study_name,
    )
    ok = (
        summary.get("best_validate_ok")
        and float(summary.get("best_completion_ratio") or 0) >= 0.999
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
