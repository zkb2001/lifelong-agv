"""Build compare_100 full metrics table including Manhattan travel from traj CSVs."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from ml_research.benchmarks.coord_custom_ai.compare_100 import JSONL, OUT

ORDER = ["pibt", "m0", "pp", "ecbs", "hier"]
LABEL = {
    "pibt": "PIBT",
    "m0": "M0",
    "pp": "PP",
    "ecbs": "ECBS",
    "hier": "Hier",
}


def _travel(traj_path: str | None) -> int | None:
    if not traj_path:
        return None
    p = Path(traj_path)
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    need = {"timestamp", "name", "X", "Y"}
    if not need.issubset(df.columns):
        return None
    dist = 0
    for _, g in df.groupby("name", sort=False):
        g = g.sort_values("timestamp")
        ts = g["timestamp"].to_numpy(dtype=int)
        xs = g["X"].to_numpy(dtype=int)
        ys = g["Y"].to_numpy(dtype=int)
        for i in range(1, len(ts)):
            if ts[i] != ts[i - 1] + 1:
                continue
            dist += abs(int(xs[i]) - int(xs[i - 1])) + abs(int(ys[i]) - int(ys[i - 1]))
    return int(dist)


def main() -> None:
    rows = [json.loads(l) for l in JSONL.read_text(encoding="utf-8").splitlines() if l.strip()]
    enriched = []
    for r in rows:
        travel = _travel(r.get("trajectory"))
        enriched.append(
            {
                "slot": int(r["slot"]),
                "method_id": r["method_id"],
                "method": LABEL.get(r["method_id"], r.get("method") or r["method_id"]),
                "done": int(r.get("tasks_completed") or 0),
                "total": int(r.get("tasks_total") or 100),
                "completion": float(r.get("completion_ratio") or 0.0),
                "valid": bool(r.get("validate_ok")),
                "sim_time": int(r.get("sim_time") or 0),
                "wall_s": float(r.get("wall_seconds") or 0.0),
                "travel": travel,
                "validate_summary": r.get("validate_summary") or "",
            }
        )

    # detail CSV + MD
    detail_csv = OUT / "COMPARE_FULL.csv"
    detail_md = OUT / "COMPARE_FULL.md"
    enriched.sort(key=lambda x: (x["slot"], ORDER.index(x["method_id"]) if x["method_id"] in ORDER else 99))
    pd.DataFrame(enriched).to_csv(detail_csv, index=False)

    lines = [
        "# Compare 100 — full metrics (after Hier unlimited-wall rerun)",
        "",
        "Travel = sum of Manhattan cell moves over consecutive timestamps (all AGVs).",
        "",
        "## Aggregate by method",
        "",
        "| Method | maps | full100 | VALID | avg_done | avg_sim | avg_wall_s | sum_travel | avg_travel |",
        "|--------|------|---------|-------|----------|---------|------------|------------|------------|",
    ]
    by_m: dict[str, list] = defaultdict(list)
    for e in enriched:
        by_m[e["method_id"]].append(e)
    for mid in ORDER:
        xs = by_m.get(mid) or []
        if not xs:
            continue
        n = len(xs)
        full = sum(1 for x in xs if x["done"] >= x["total"])
        valid = sum(1 for x in xs if x["valid"])
        avg_done = sum(x["done"] for x in xs) / n
        avg_sim = sum(x["sim_time"] for x in xs) / n
        avg_wall = sum(x["wall_s"] for x in xs) / n
        travels = [x["travel"] for x in xs if x["travel"] is not None]
        sum_tr = sum(travels) if travels else None
        avg_tr = (sum(travels) / len(travels)) if travels else None
        lines.append(
            f"| {LABEL[mid]} | {n} | {full}/{n} | {valid}/{n} | {avg_done:.1f} | "
            f"{avg_sim:.0f} | {avg_wall:.1f} | "
            f"{sum_tr if sum_tr is not None else 'NA'} | "
            f"{avg_tr:.0f}" if avg_tr is not None else f"| {LABEL[mid]} | {n} | {full}/{n} | {valid}/{n} | {avg_done:.1f} | {avg_sim:.0f} | {avg_wall:.1f} | NA | NA |"
        )
        if avg_tr is not None:
            # fix: previous append already done with f-string that may have broken
            pass

    # rebuild aggregate cleanly
    lines = [
        "# Compare 100 — full metrics (after Hier unlimited-wall rerun)",
        "",
        "Travel = sum of Manhattan cell moves over consecutive timestamps (all AGVs).",
        "",
        "## Aggregate by method",
        "",
        "| Method | maps | full100 | VALID | avg_done | avg_sim | avg_wall_s | sum_travel | avg_travel |",
        "|--------|------|---------|-------|----------|---------|------------|------------|------------|",
    ]
    for mid in ORDER:
        xs = by_m.get(mid) or []
        if not xs:
            continue
        n = len(xs)
        full = sum(1 for x in xs if x["done"] >= x["total"])
        valid = sum(1 for x in xs if x["valid"])
        avg_done = sum(x["done"] for x in xs) / n
        avg_sim = sum(x["sim_time"] for x in xs) / n
        avg_wall = sum(x["wall_s"] for x in xs) / n
        travels = [x["travel"] for x in xs if x["travel"] is not None]
        sum_tr = sum(travels) if travels else 0
        avg_tr = (sum(travels) / len(travels)) if travels else 0
        lines.append(
            f"| {LABEL[mid]} | {n} | {full}/{n} | {valid}/{n} | {avg_done:.1f} | "
            f"{avg_sim:.0f} | {avg_wall:.1f} | {sum_tr} | {avg_tr:.0f} |"
        )

    lines += [
        "",
        "## Per map × method",
        "",
        "| Slot | Method | done | valid | sim_time | wall_s | travel |",
        "|------|--------|------|-------|----------|--------|--------|",
    ]
    for e in enriched:
        v = "Y" if e["valid"] else "N"
        tr = e["travel"] if e["travel"] is not None else "NA"
        lines.append(
            f"| SH{e['slot']:02d} | {e['method']} | {e['done']}/{e['total']} | {v} | "
            f"{e['sim_time']} | {e['wall_s']:.1f} | {tr} |"
        )

    detail_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {detail_md}")
    print(f"wrote {detail_csv}")
    print("\n".join(lines[5:12]))


if __name__ == "__main__":
    main()
