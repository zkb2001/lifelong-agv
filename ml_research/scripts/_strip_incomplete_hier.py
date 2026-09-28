"""Strip incomplete hier rows, then caller re-runs with wall_timeout=0."""
from __future__ import annotations

import json
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.compare_100 import (
    JSONL,
    OUT,
    SUMMARY,
    SUMMARY_JSON,
    _write_summary,
)

INCOMPLETE = {6, 8, 12}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    removed = []
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("method_id") == "hier" and int(r.get("slot") or -1) in INCOMPLETE:
            removed.append(r)
            continue
        rows.append(r)
    JSONL.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    _write_summary(rows)
    print(
        f"removed {len(removed)} hier rows for slots {sorted(INCOMPLETE)}; "
        f"kept {len(rows)}; wrote {SUMMARY}",
        flush=True,
    )


if __name__ == "__main__":
    main()
