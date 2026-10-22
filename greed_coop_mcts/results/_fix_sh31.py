"""Replace SH31 with a completable hard maze (connected), then eval."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from greed_coop_mcts.results._gen_eval_curriculum import (  # noqa: E402
    _geom,
    eval_one,
    generate_one,
)

ROOT = Path(__file__).resolve().parent


def main() -> None:
    # Prefer milder maze + fewer AGVs than near=136×12 which stalled at 0
    candidates = [
        (3118, 10, "maze_mild"),
        (3120, 8, "maze_mild"),
        (3125, 10, "maze"),
        (3130, 8, "hutong"),
    ]
    best = None
    for seed, n, style in candidates:
        generate_one(31, style, seed, n, "clustered_dest", "hard")
        g = _geom(31)
        print(
            f"try seed={seed} style={style} n={n} near={g['near']} obs={g['n_obs']}",
            flush=True,
        )
        if g["near"] > 120:
            continue
        r = eval_one(31)
        r.update(style=style, difficulty="hard", pattern="clustered_dest", seed=seed)
        print(
            f"  -> sim={r['sim']} done={r['done']} ok={r['ok']} mot={r['mot']}",
            flush=True,
        )
        best = r
        if r["done"] == 100:
            break

    if best is None:
        print("no candidate accepted", flush=True)
        return

    # patch JSON/MD row for SH31
    path = ROOT / "GEN_CURRICULUM_EVAL.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    for i, row in enumerate(rows):
        if int(row["slot"]) == 31:
            rows[i] = {**row, **best, "slot": 31}
            break
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("updated SH31 in GEN_CURRICULUM_EVAL.json", flush=True)


if __name__ == "__main__":
    main()
