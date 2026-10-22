import json
from pathlib import Path

p = Path(__file__).with_name("facn_sh20_k_full_sweep.json")
d = json.loads(p.read_text(encoding="utf-8"))
rows = [
    r
    for r in (d.get("rows") or [])
    if int(r.get("slot", -1)) == 20 and r.get("sim") is not None
]
by = {}
for r in rows:
    by[int(r["k"])] = r
print("K\tsim\tdone\tvalid")
best = None
for k in range(6, 31):
    r = by.get(k)
    if not r:
        print(f"{k}\tMISSING")
        continue
    sim = r.get("sim")
    ok = bool(r.get("valid")) and r.get("done") == r.get("total")
    print(f"{k}\t{sim}\t{r.get('done')}/{r.get('total')}\t{ok}")
    if ok and (
        best is None
        or int(sim) < best[0]
        or (int(sim) == best[0] and k > best[1])
    ):
        best = (int(sim), k)
print("BEST", best)
base = by.get(7, {}).get("sim")
s6 = by.get(6, {}).get("sim")
if base and best:
    dlt = best[0] - int(base)
    print(
        f"vs TBC7: {base} -> {best[0]} (K={best[1]}) "
        f"d={dlt:+d} ({100.0 * dlt / float(base):+.1f}%)"
    )
if s6 and best:
    dlt = best[0] - int(s6)
    print(
        f"vs K6: {s6} -> {best[0]} "
        f"d={dlt:+d} ({100.0 * dlt / float(s6):+.1f}%)"
    )
