"""Export A* baseline stats for SH_custom_01..20 (100 tasks)."""
from __future__ import annotations

import json
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.compare_astar_ecbs_all_slots import (
    row_for_slot,
)

OUT = Path(__file__).resolve().parents[2] / "results" / "coord_custom_ai" / "compare"


def main() -> int:
    rows = [row_for_slot(slot, skip_ecbs=True) for slot in range(1, 21)]
    path = OUT / "SH_custom_astar_baseline_only.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {path}\n")
    print(f"{'slot':>4}  {'sim':>6}  {'move':>6}  {'val':>5}  {'tasks':>5}  summary")
    for r in rows:
        a = r.get("astar") or {}
        print(
            f"{r['slot']:>4}  "
            f"{a.get('sim_time', '-'):>6}  "
            f"{a.get('total_move_steps', '-'):>6}  "
            f"{str(a.get('validate_ok', '-')):>5}  "
            f"{a.get('tasks_seen_in_traj', '-'):>5}  "
            f"{a.get('validate_summary', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
