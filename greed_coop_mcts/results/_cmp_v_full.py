import json
from pathlib import Path

full = {
    1: 965, 2: 2021, 3: 4440, 4: 1053, 5: 1616, 6: 2024, 7: 3897, 8: 1320,
    9: 1337, 10: 2215, 11: 2384, 12: 1558, 13: 832, 14: 1216, 15: 2523,
    16: 1643, 17: 2395, 18: 1711, 19: 1446, 20: 947,
}
root = Path(__file__).resolve().parent
print((root / "SUMMARY.md").read_text(encoding="utf-8"))
rows = json.loads((root / "summary.json").read_text(encoding="utf-8")).get("rows") or []
sims, worse, better, fail = [], [], [], []
for r in rows:
    s = int(r["slot"])
    sim = r.get("sim_time")
    done = r.get("tasks_completed")
    ok = r.get("validate_ok")
    base = full[s]
    if done != 100 or not ok:
        fail.append(s)
        print("FAIL", s, done, ok, sim)
        continue
    dlt = int(sim) - base
    sims.append(int(sim))
    print(f"SH{s:02d} sim={sim} FULL={base} dlt={dlt:+d}")
    (worse if dlt > 0 else better).append((s, dlt))
avg = sum(sims) / len(sims) if sims else float("nan")
favg = sum(full.values()) / 20
print(f"avg={avg:.1f} FULL={favg:.1f} delta={avg - favg:+.1f}")
print("better_or_eq", len(better), "worse", len(worse), "fail", fail)
gate_ok = True
for s in (1, 2, 3, 8):
    r = next(x for x in rows if int(x["slot"]) == s)
    sim = int(r["sim_time"])
    ratio = sim / full[s]
    ok = ratio <= 1.15
    gate_ok = gate_ok and ok
    print(f"gate SH{s:02d} {sim} vs {full[s]} ratio={ratio:.3f} ok={ok}")
print("GATE_PASS", gate_ok)
