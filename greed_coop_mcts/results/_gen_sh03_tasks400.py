"""Write a 400-task seed CSV for SH03 (uniform station→dest)."""
from __future__ import annotations

import csv
from pathlib import Path

from ml_research.benchmarks.scenarios.generator import gen_tasks
from greed_coop_mcts.paths import MAP_EXPORTS, RESULTS

OUT = RESULTS / "SH03_lifelong_seed400_task.csv"


def main() -> int:
    rows = gen_tasks("uniform", 400, seed=303)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "start_point",
                "end_point",
                "priority",
                "remaining_time",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} tasks → {OUT}")
    print(f"map_exports={MAP_EXPORTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
