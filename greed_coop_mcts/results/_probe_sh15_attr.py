"""Probe SH15 attractors under varying determinism knobs."""
from __future__ import annotations

import os
import sys

# Probe variants by env before importing torch/solve.
VARIANT = os.environ.get("SH15_PROBE", "hash0")


def _configure() -> None:
    if VARIANT == "hash0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    elif VARIANT == "nohash":
        os.environ.pop("PYTHONHASHSEED", None)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    elif VARIANT == "loose":
        os.environ.pop("PYTHONHASHSEED", None)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.pop(k, None)


_configure()

from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve


def main() -> int:
    meta = load_slot_meta(15, max_sim_time=200000, wall_timeout=1800.0, n_tasks=100)
    meta = dict(meta)
    meta["id"] = f"SH15_probe_{VARIANT}"
    meta["n_tasks"] = 100
    rep = solve(
        15,
        meta=meta,
        wall_timeout=1800.0,
        max_sim_time=200000,
        policy="rrsc",
        use_cgpn=False,
        device="cpu",
    )
    print(
        f"PROBE variant={VARIANT} sim={rep.get('sim_time')} "
        f"done={rep.get('tasks_completed')} ok={rep.get('validate_ok')} "
        f"stage_f={rep.get('n_stage_final')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
