"""Strip ECBS SH11 (and optionally SH12 if invalid) then print status."""
from __future__ import annotations

import json
import sys

from ml_research.benchmarks.coord_custom_ai.compare_100 import JSONL, _write_summary


def main() -> None:
    slots = {int(x) for x in (sys.argv[1:] or ["11"])}
    rows = []
    rem = 0
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("method_id") == "ecbs" and int(r.get("slot") or -1) in slots:
            rem += 1
            continue
        rows.append(r)
    JSONL.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    _write_summary(rows)
    print(f"removed={rem} kept={len(rows)} slots={sorted(slots)}")
    for r in rows:
        if r.get("method_id") == "ecbs" and int(r.get("slot") or -1) in (11, 12):
            print(
                f"SH{int(r['slot']):02d} valid={r.get('valid')} "
                f"done={r.get('tasks_done')}/{r.get('tasks_total')} "
                f"sim={r.get('sim_time')}"
            )


if __name__ == "__main__":
    main()
