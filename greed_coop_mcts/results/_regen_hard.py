"""Regen+eval SH31/SH32 only."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from greed_coop_mcts.results._gen_eval_curriculum import SPECS, generate_one, eval_one  # noqa: E402


def main() -> None:
    for slot, style, seed, n, pat, tag in SPECS:
        if slot not in (31, 32):
            continue
        generate_one(slot, style, seed, n, pat, tag)
        r = eval_one(slot)
        print(
            f"SH{slot:02d} sim={r['sim']} done={r['done']} ok={r['ok']} "
            f"near={r['near']} n={r['n']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
