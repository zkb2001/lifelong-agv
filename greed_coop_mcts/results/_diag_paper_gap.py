"""Paper-gap diagnosis: ablation AI lift + maps still flat vs FULL."""
import json
from pathlib import Path

FULL = {
    1: 965, 2: 2021, 3: 4440, 4: 1053, 5: 1616, 6: 2024, 7: 3897, 8: 1320,
    9: 1337, 10: 2215, 11: 2384, 12: 1558, 13: 832, 14: 1216, 15: 2523,
    16: 1643, 17: 2395, 18: 1711, 19: 1446, 20: 947,
}
root = Path(__file__).resolve().parent

print("=== flat vs FULL (dlt==0) ===")
flat = []
better = []
for i in range(1, 21):
    d = json.loads((root / f"SH{i:02d}_gcm_gcm.json").read_text(encoding="utf-8"))
    sim = int(d["sim_time"])
    dlt = sim - FULL[i]
    fleet = d.get("max_fleet_busy")
    if dlt == 0:
        flat.append(i)
    elif dlt < 0:
        better.append((i, dlt, fleet))
print("flat", flat)
print("better", better)

print("\n=== ablation full vs no_ai / no_stagenet (where available) ===")
for i in range(1, 21):
    rows = {}
    for v in ("full", "no_ai", "no_stagenet", "no_rulenet", "no_pgcp", "no_pgfa"):
        p = root / "ablation" / v / f"SH{i:02d}.json"
        if p.exists():
            rows[v] = json.loads(p.read_text(encoding="utf-8")).get("sim_time")
    if len(rows) >= 2:
        print(f"SH{i:02d}", rows)

print("\n=== historical CGPN wins ===")
for name in ("SH12_cgpn_gcm.json", "SH20_cgpn_gcm.json", "SH11_cgpn_gcm.json", "SH15_cgpn_gcm.json"):
    p = root / name
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        print(name, d.get("sim_time"), "fleet", d.get("max_fleet_busy"))
