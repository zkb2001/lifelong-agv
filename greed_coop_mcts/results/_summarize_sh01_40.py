"""Summarize SH01-40 batch after border fix (vs FULL for 01-20)."""
from __future__ import annotations

import json
from pathlib import Path

FULL = {
    1: 965,
    2: 2021,
    3: 4440,
    4: 1053,
    5: 1616,
    6: 2024,
    7: 3897,
    8: 1320,
    9: 1337,
    10: 2215,
    11: 2384,
    12: 1558,
    13: 832,
    14: 1216,
    15: 2523,
    16: 1643,
    17: 2395,
    18: 1711,
    19: 1446,
    20: 947,
}

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "summary.json").read_text(encoding="utf-8")).get("rows") or []
by = {int(r["slot"]): r for r in rows}

lines = [
    "# SH01–40 retest (rrsc + ghost-border fix)",
    "",
]

# --- SH01-20 ---
sims20, worse, better, fail20 = [], [], [], []
lines += [
    "## SH01–20 vs FULL",
    "",
    "| Slot | sim | FULL | Δ | VALID | done |",
    "|------|-----|------|---|-------|------|",
]
for s in range(1, 21):
    r = by.get(s) or {}
    sim = r.get("sim_time")
    done = int(r.get("tasks_completed") or 0)
    ok = bool(r.get("validate_ok")) and done == 100
    base = FULL[s]
    if not ok or sim is None:
        fail20.append(s)
        lines.append(f"| SH{s:02d} | {sim} | {base} | — | {int(ok)} | {done}/100 |")
        continue
    dlt = int(sim) - base
    sims20.append(int(sim))
    (worse if dlt > 0 else better).append((s, dlt))
    lines.append(
        f"| SH{s:02d} | {sim} | {base} | {dlt:+d} | {int(ok)} | {done}/100 |"
    )

avg20 = sum(sims20) / len(sims20) if sims20 else float("nan")
favg = sum(FULL.values()) / 20
gate_ok = True
gate_lines = []
for s in (1, 2, 3, 8):
    r = by.get(s) or {}
    sim = int(r.get("sim_time") or 10**9)
    ratio = sim / FULL[s]
    ok = ratio <= 1.15 and bool(r.get("validate_ok")) and int(r.get("tasks_completed") or 0) == 100
    gate_ok = gate_ok and ok
    gate_lines.append(f"- SH{s:02d}: {sim}/{FULL[s]} ratio={ratio:.3f} {'OK' if ok else 'FAIL'}")

lines += [
    "",
    f"- complete VALID: **{len(sims20)}/20** (fail={fail20})",
    f"- avg sim: **{avg20:.1f}** vs FULL **{favg:.1f}** ({avg20 - favg:+.1f})",
    f"- better_or_eq: **{len(better)}**  worse: **{len(worse)}**",
    f"- GATE (≤15% vs FULL on SH01/02/03/08): **{'PASS' if gate_ok else 'FAIL'}**",
    *gate_lines,
    "",
]

# --- SH21-40 ---
ok40 = 0
sims40 = []
lines += [
    "## SH21–40 generated curriculum",
    "",
    "| Slot | sim | VALID | done | mot |",
    "|------|-----|-------|------|-----|",
]
fail40 = []
for s in range(21, 41):
    r = by.get(s) or {}
    sim = r.get("sim_time")
    done = int(r.get("tasks_completed") or 0)
    ok = bool(r.get("validate_ok")) and done == 100
    iss = r.get("validate_issues") or {}
    mot = iss.get("n_illegal_motion", r.get("n_illegal_motion", "—"))
    if ok:
        ok40 += 1
        sims40.append(int(sim))
    else:
        fail40.append(s)
    lines.append(
        f"| SH{s:02d} | {sim} | {int(ok)} | {done}/100 | {mot} |"
    )

avg40 = sum(sims40) / len(sims40) if sims40 else float("nan")
lines += [
    "",
    f"- complete VALID: **{ok40}/20** (fail={fail40})",
    f"- avg sim (VALID): **{avg40:.1f}**" if sims40 else "- avg sim (VALID): n/a",
    "",
    "## Overall SH01–40",
    f"- VALID: **{len(sims20) + ok40}/40**",
]

out = ROOT / "SH01_40_RETEST.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out.read_text(encoding="utf-8"))
print(f"wrote {out}")
