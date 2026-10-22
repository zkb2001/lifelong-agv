"""Build FACN vs no-FACN SH01–40 comparison from batch summary JSONs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load(name: str) -> dict[int, dict]:
    path = ROOT / name
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    return {int(r["slot"]): r for r in rows if r.get("slot") is not None}


def main() -> int:
    facn = _load("summary_facn_40.json")
    nofacn = _load("summary_nofacn_40.json")
    slots = sorted(set(facn) | set(nofacn) | set(range(1, 41)))

    lines = [
        "# FACN vs no-FACN — SH01–40",
        "",
        "Arm A: `--facn` (tag=`facn`). Arm B: `--no-facn` (tag=`nofacn`, pure TBC).",
        "",
        "| Slot | noFACN sim | FACN sim | Δsim | FACN on | K | ΔK | VALID | done |",
        "|------|------------|----------|------|---------|---|----|-------|------|",
    ]

    pair_sims_a: list[int] = []
    pair_sims_b: list[int] = []
    better = worse = same = 0
    facn_on = 0
    valid_both = 0
    fail: list[int] = []

    for s in slots:
        a = nofacn.get(s) or {}
        b = facn.get(s) or {}
        sa, sb = a.get("sim_time"), b.get("sim_time")
        ok_a = bool(a.get("validate_ok")) and int(a.get("tasks_completed") or 0) == 100
        ok_b = bool(b.get("validate_ok")) and int(b.get("tasks_completed") or 0) == 100
        on = bool(b.get("use_facn"))
        k = b.get("base_fleet_busy")
        dk = b.get("facn_delta")
        done = f"{b.get('tasks_completed', a.get('tasks_completed'))}/100"
        if not (ok_a and ok_b and sa is not None and sb is not None):
            fail.append(s)
            dlt = "—"
            valid = f"{int(ok_a)}/{int(ok_b)}"
        else:
            valid_both += 1
            dlt_i = int(sb) - int(sa)
            dlt = f"{dlt_i:+d}"
            pair_sims_a.append(int(sa))
            pair_sims_b.append(int(sb))
            if dlt_i < 0:
                better += 1
            elif dlt_i > 0:
                worse += 1
            else:
                same += 1
            if on:
                facn_on += 1
            valid = "Y"
        lines.append(
            f"| SH{s:02d} | {sa} | {sb} | {dlt} | {int(on)} | {k} | {dk} | "
            f"{valid} | {done} |"
        )

    avg_a = sum(pair_sims_a) / len(pair_sims_a) if pair_sims_a else float("nan")
    avg_b = sum(pair_sims_b) / len(pair_sims_b) if pair_sims_b else float("nan")
    dlt_avg = avg_b - avg_a if pair_sims_a else float("nan")
    pct = (100.0 * dlt_avg / avg_a) if pair_sims_a and avg_a else float("nan")

    # Highlight maps where FACN actually attached.
    attach_lines = []
    for s in slots:
        b = facn.get(s) or {}
        a = nofacn.get(s) or {}
        if not bool(b.get("use_facn")):
            continue
        sa, sb = a.get("sim_time"), b.get("sim_time")
        if sa is None or sb is None:
            continue
        attach_lines.append(
            f"- SH{s:02d}: {sa} → {sb} ({int(sb) - int(sa):+d}), "
            f"K={b.get('base_fleet_busy')} Δ={b.get('facn_delta')} "
            f"updates={b.get('n_facn_updates')}"
        )

    lines += [
        "",
        "## Summary",
        "",
        f"- paired VALID 100/100: **{valid_both}/40** (fail={fail})",
        f"- avg sim noFACN: **{avg_a:.1f}**",
        f"- avg sim FACN: **{avg_b:.1f}** ({dlt_avg:+.1f}, {pct:+.2f}%)",
        f"- FACN better / same / worse: **{better} / {same} / {worse}**",
        f"- maps where FACN attached: **{facn_on}**",
        "",
        "### FACN-attached maps",
        "",
    ]
    lines.extend(attach_lines or ["- (none)"])
    lines.append("")

    out = ROOT / "FACN_COMPARE_40.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "avg_nofacn": avg_a,
        "avg_facn": avg_b,
        "delta_avg": dlt_avg,
        "pct": pct,
        "better": better,
        "same": same,
        "worse": worse,
        "facn_on": facn_on,
        "valid_both": valid_both,
        "fail": fail,
    }
    (ROOT / "facn_compare_40.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
