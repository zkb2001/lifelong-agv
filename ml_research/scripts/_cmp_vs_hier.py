import json
from collections import defaultdict
from pathlib import Path

rows = [
    json.loads(l)
    for l in Path("ml_research/results/coord_custom_ai/compare_100/results.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if l.strip()
]
by: dict = defaultdict(dict)
for r in rows:
    by[int(r["slot"])][r.get("method_id")] = r

methods = ["pibt", "m0", "pp", "ecbs", "hier"]
complete = sorted(s for s, d in by.items() if all(m in d for m in methods))
print("complete slots:", complete)
print()
for s in complete:
    h = by[s]["hier"]
    hd = int(h.get("tasks_completed") or 0)
    print(
        f"SH{s:02d}  M0+ECBS(hier)={hd}/100  "
        f"VALID={'Y' if h.get('validate_ok') else 'N'}  "
        f"sim={h.get('sim_time')}  wall={h.get('wall_seconds')}"
    )
    for m in ["pibt", "m0", "pp", "ecbs"]:
        r = by[s][m]
        d = int(r.get("tasks_completed") or 0)
        print(
            f"  vs {m:5s}: done={d:3d} ({d - hd:+4d})  "
            f"VALID={'Y' if r.get('validate_ok') else 'N'}  "
            f"sim={r.get('sim_time')}  wall={r.get('wall_seconds')}"
        )
    print()

print("=== summary vs M0+ECBS (hier) on", len(complete), "maps ===")
for m in ["pibt", "m0", "pp", "ecbs", "hier"]:
    rs = [by[s][m] for s in complete]
    n = len(rs)
    full = sum(1 for r in rs if int(r.get("tasks_completed") or 0) >= 100)
    valid = sum(1 for r in rs if r.get("validate_ok"))
    avg_done = sum(int(r.get("tasks_completed") or 0) for r in rs) / n
    avg_sim = sum(int(r.get("sim_time") or 0) for r in rs) / n
    avg_wall = sum(float(r.get("wall_seconds") or 0) for r in rs) / n
    beat = sum(
        1
        for s in complete
        if int(by[s][m].get("tasks_completed") or 0)
        > int(by[s]["hier"].get("tasks_completed") or 0)
    )
    lose = sum(
        1
        for s in complete
        if int(by[s][m].get("tasks_completed") or 0)
        < int(by[s]["hier"].get("tasks_completed") or 0)
    )
    tie = n - beat - lose
    print(
        f"{m:5s} full={full}/{n} VALID={valid}/{n} "
        f"avg_done={avg_done:.1f} avg_sim={avg_sim:.0f} avg_wall={avg_wall:.1f} "
        f"| vs_hier done: win={beat} tie={tie} lose={lose}"
    )
