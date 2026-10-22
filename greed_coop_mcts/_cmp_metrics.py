"""Compare suite metrics across result JSON / markdown snapshots."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "results"


def from_summary_md(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(
            r"\|\s*SH(\d+)\s*\|\s*(\d+)/(\d+)\s*\|\s*([YN])\s*\|\s*(\d+)\s*\|",
            line,
        )
        if m:
            out[int(m.group(1))] = {
                "done": int(m.group(2)),
                "total": int(m.group(3)),
                "valid": m.group(4) == "Y",
                "sim": int(m.group(5)),
            }
    return out


def from_full_suite(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(
            r"\|\s*SH(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([YN])\s*\|",
            line,
        )
        if m:
            out[int(m.group(1))] = {
                "done": int(m.group(2)),
                "total": 100,
                "valid": m.group(5) == "Y",
                "sim": int(m.group(3)),
                "wall": float(m.group(4)),
            }
    return out


def avg(d: dict, key="sim") -> float:
    vals = [v[key] for v in d.values() if v.get("done", 0) >= 100]
    return sum(vals) / max(1, len(vals))


def main() -> None:
    cur = from_summary_md(ROOT / "SUMMARY.md")
    hist = from_full_suite(ROOT / "FULL_SUITE_WALL_SIM.md")
    print("current n", len(cur), "avg_sim", round(avg(cur), 1), "full100", sum(1 for v in cur.values() if v["done"] >= 100))
    print("FULL_SUITE n", len(hist), "avg_sim", round(avg(hist), 1))
    print(f"{'slot':>4} {'cur':>7} {'hist':>7} {'dSim':>8}")
    for s in range(1, 21):
        a = cur.get(s, {}).get("sim")
        b = hist.get(s, {}).get("sim")
        if a is None or b is None:
            continue
        print(f"{s:4d} {a:7d} {b:7d} {a-b:+8d}")
    # old SH11 json
    p = ROOT / "SH11_gcm_gcm.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        print("SH11_gcm_gcm.json sim", d.get("sim_time"), "repair", d.get("n_window_repairs"), "gates", d.get("n_dest_gates"))


if __name__ == "__main__":
    main()
