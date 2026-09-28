"""Write per-map five-method comparison tables (done/valid/sim/wall/travel)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.compare_100 import OUT

ORDER = ["PIBT", "M0", "PP", "ECBS", "Hier"]
CSV = OUT / "COMPARE_FULL.csv"
MD = OUT / "PER_MAP_COMPARE.md"


def main() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    by_slot: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_slot[int(r["slot"])][r["method"]] = r

    lines: list[str] = [
        "# Compare 100 — 逐图五方法对比",
        "",
        "指标：完成度 / VALID / sim_time / 墙钟(s) / 总行驶距离（全车队曼哈顿移动格数）",
        "",
        "方法：PIBT | M0（时空A*） | PP | ECBS | Hier（M0↔ECBS 自动切换）",
        "",
    ]

    # Overview: one wide row per map showing done+valid only
    lines += [
        "## 总览（完成度 + VALID）",
        "",
        "| 地图 | PIBT | M0 | PP | ECBS | Hier |",
        "|------|------|----|----|------|------|",
    ]
    for slot in range(1, 21):
        cells = [f"SH{slot:02d}"]
        for m in ORDER:
            r = by_slot[slot].get(m)
            if not r:
                cells.append("—")
                continue
            v = "Y" if r["valid"] == "True" else "N"
            cells.append(f"{r['done']}/{r['total']} {v}")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## 分图明细", ""]

    for slot in range(1, 21):
        lines += [
            f"### SH{slot:02d}",
            "",
            "| 方法 | 完成度 | VALID | sim_time | 墙钟(s) | 总行驶距离 |",
            "|------|--------|-------|----------|---------|------------|",
        ]
        for m in ORDER:
            r = by_slot[slot].get(m)
            if not r:
                lines.append(f"| {m} | — | — | — | — | — |")
                continue
            v = "Y" if r["valid"] == "True" else "N"
            travel = r["travel"] if r.get("travel") not in (None, "") else "NA"
            lines.append(
                f"| {m} | {r['done']}/{r['total']} | {v} | {r['sim_time']} | "
                f"{float(r['wall_s']):.1f} | {travel} |"
            )
        lines.append("")

    # Aggregate
    lines += [
        "## 方法汇总",
        "",
        "| 方法 | maps | full100 | VALID | avg_done | avg_sim | avg_wall_s | sum_travel | avg_travel |",
        "|------|-------|---------|-------|----------|---------|------------|------------|------------|",
    ]
    for m in ORDER:
        xs = [by_slot[s][m] for s in range(1, 21) if m in by_slot[s]]
        n = len(xs)
        if not n:
            continue
        full = sum(1 for x in xs if int(x["done"]) >= int(x["total"]))
        valid = sum(1 for x in xs if x["valid"] == "True")
        avg_done = sum(int(x["done"]) for x in xs) / n
        avg_sim = sum(int(x["sim_time"]) for x in xs) / n
        avg_wall = sum(float(x["wall_s"]) for x in xs) / n
        travels = [int(float(x["travel"])) for x in xs if x.get("travel")]
        sum_tr = sum(travels)
        avg_tr = sum_tr / len(travels) if travels else 0
        lines.append(
            f"| {m} | {n} | {full}/{n} | {valid}/{n} | {avg_done:.1f} | "
            f"{avg_sim:.0f} | {avg_wall:.1f} | {sum_tr} | {avg_tr:.0f} |"
        )
    lines.append("")

    MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {MD}")
    print("\n".join(lines[7:30]))


if __name__ == "__main__":
    main()
