"""Why only 7/20 VALID on generated maps?"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from greed_coop_mcts.scenario import load_slot_meta
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    validate_hybrid_trajectory as full,
)

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "GEN_CURRICULUM_EVAL.json").read_text(encoding="utf-8"))


def kind_stats(slot: int, traj: Path, meta_id: str) -> tuple[bool, Counter, list]:
    meta = dict(load_slot_meta(slot) or {})
    meta["id"] = meta_id
    meta["n_tasks"] = 100
    rep = full(meta, traj)
    mot = (rep.get("issues") or {}).get("illegal_motion") or []
    kinds = Counter(m.get("kind", "?") for m in mot)
    # also count OOB cells
    oob_cells = Counter(tuple(m["cell"]) for m in mot if m.get("kind") == "out_of_bounds")
    return bool(rep.get("ok")), kinds, oob_cells.most_common(5)


print("=== GEN INVALID breakdown ===")
all_kinds: Counter = Counter()
for r in rows:
    slot = int(r["slot"])
    p = ROOT / "trajectories" / f"SH{slot:02d}_gen_gcm.csv"
    if not p.exists():
        continue
    ok, kinds, oob = kind_stats(slot, p, f"SH{slot:02d}_gen")
    all_kinds.update(kinds)
    status = "VALID" if ok else "INVALID"
    print(
        f"SH{slot:02d} [{r['difficulty']}/{r['style']}] n={r['n']} "
        f"{status} mot={r['mot']} kinds={dict(kinds)} oob_top={oob}"
    )

print("\nALL INVALID kinds:", dict(all_kinds))

print("\n=== PROD SH01-20 (if traj present) ===")
prod_ok = 0
prod_n = 0
for slot in range(1, 21):
    p = ROOT / "trajectories" / f"SH{slot:02d}_gcm.csv"
    if not p.exists():
        print(f"SH{slot:02d} no traj")
        continue
    prod_n += 1
    ok, kinds, oob = kind_stats(slot, p, f"SH{slot:02d}")
    prod_ok += int(ok)
    print(f"SH{slot:02d} ok={ok} kinds={dict(kinds)} oob_top={oob[:2]}")
print(f"prod VALID {prod_ok}/{prod_n}")

# Does VALID traj ever visit y=0?
print("\n=== y<=0 visits in VALID vs INVALID gen ===")
for r in rows:
    slot = int(r["slot"])
    p = ROOT / "trajectories" / f"SH{slot:02d}_gen_gcm.csv"
    if not p.exists():
        continue
    n_y0 = 0
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["Y"]) < 1 or int(row["X"]) < 1:
                n_y0 += 1
    print(f"SH{slot:02d} ok={r['ok']} oob_ticks={n_y0}")
