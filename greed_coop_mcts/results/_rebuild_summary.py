import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
for i in range(1, 21):
    d = json.loads((root / f"SH{i:02d}_gcm_gcm.json").read_text(encoding="utf-8"))
    rows.append(d)
(root / "summary.json").write_text(
    json.dumps({"policy": "rrsc", "rows": rows}, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
lines = [
    "# greed_coop_mcts MAPD (policy=rrsc)",
    "",
    "| slot | done | VALID | sim | wall_s | planner |",
    "|------|------|-------|-----|--------|---------|",
]
for d in rows:
    ok = "Y" if d.get("validate_ok") else "N"
    planner = d.get("planner") or "mrsc"
    lines.append(
        f"| SH{int(d['slot']):02d} | {d['tasks_completed']}/{d['tasks_total']} | "
        f"{ok} | {d['sim_time']} | {d['wall_seconds']} | {planner} |"
    )
(root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("avg", sum(int(d["sim_time"]) for d in rows) / 20)
