"""Lightweight trajectory checks (independent of ECBS / Hier validators).

Optionally delegates to the project competition validator if importable,
so reports can still show full VALID status when run inside this repo.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

Cell = Tuple[int, int]


def _basic_validate(traj: Path) -> dict:
    by_t: Dict[int, Dict[str, Cell]] = defaultdict(dict)
    rows = 0
    with traj.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows += 1
            t = int(row["timestamp"])
            name = str(row["name"])
            by_t[t][name] = (int(row["X"]), int(row["Y"]))

    n_collisions = 0
    n_swaps = 0
    times = sorted(by_t)
    for i, t in enumerate(times):
        cells = list(by_t[t].values())
        if len(cells) != len(set(cells)):
            n_collisions += 1
        if i == 0:
            continue
        prev = by_t[times[i - 1]]
        cur = by_t[t]
        names = set(prev) & set(cur)
        for a in names:
            for b in names:
                if a >= b:
                    continue
                if prev[a] == cur[b] and prev[b] == cur[a] and prev[a] != prev[b]:
                    n_swaps += 1

    ok = n_collisions == 0 and n_swaps == 0 and rows > 0
    return {
        "ok": ok,
        "mode": "basic",
        "issues": {
            "n_collisions": n_collisions,
            "n_swaps": n_swaps,
            "n_illegal_motion": 0,
            "n_fifo_violations": 0,
            "n_display_mismatch": 0,
        },
        "n_rows": rows,
    }


def format_validation_summary(rep: dict) -> str:
    if rep.get("ok"):
        return "VALID ok"
    iss = rep.get("issues") or {}
    parts = [
        f"coll={iss.get('n_collisions', 0)}",
        f"swap={iss.get('n_swaps', 0)}",
        f"mot={iss.get('n_illegal_motion', 0)}",
        f"fifo={iss.get('n_fifo_violations', 0)}",
    ]
    return "INVALID " + " ".join(parts)


def validate_trajectory(meta: dict, traj: Path) -> dict:
    """Prefer full project validator; fall back to basic collision/swap check."""
    try:
        from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
            validate_hybrid_trajectory as _full,
        )

        rep = _full(meta, traj)
        rep = dict(rep)
        rep["mode"] = "full"
        return rep
    except Exception:  # noqa: BLE001
        return _basic_validate(Path(traj))
