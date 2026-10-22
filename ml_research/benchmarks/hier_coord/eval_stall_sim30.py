"""Re-test Hier with StallEscalateNet (+sim30_done). Compare vs compare_100 baseline.

Writes: ml_research/results/hier_coord/stall_sim30_eval/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ml_research.benchmarks.coord_custom_ai.compare_100 import run_method
from ml_research.common.paths import RESULTS

BASELINE_JSON = RESULTS / "coord_custom_ai" / "compare_100" / "summary.json"
BASELINE_JSONL = RESULTS / "coord_custom_ai" / "compare_100" / "results.jsonl"
OUT = RESULTS / "hier_coord" / "stall_sim30_eval"


def _load_baseline_hier() -> Dict[int, dict]:
    by: Dict[int, dict] = {}
    if BASELINE_JSON.is_file():
        blob = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
        rows = blob.get("rows") if isinstance(blob, dict) else blob
        for r in rows or []:
            if str(r.get("method_id") or "") != "hier":
                continue
            by[int(r["slot"])] = r
    if by:
        return by
    if BASELINE_JSONL.is_file():
        for line in BASELINE_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r.get("method_id") or "") != "hier":
                continue
            by[int(r["slot"])] = r
    return by


def _delta(new: dict, old: Optional[dict]) -> dict:
    if not old:
        return {"has_baseline": False}
    return {
        "has_baseline": True,
        "d_sim": int(new.get("sim_time") or 0) - int(old.get("sim_time") or 0),
        "d_wall": round(
            float(new.get("wall_seconds") or 0) - float(old.get("wall_seconds") or 0),
            2,
        ),
        "d_done": int(new.get("tasks_completed") or 0)
        - int(old.get("tasks_completed") or 0),
        "base_sim": int(old.get("sim_time") or 0),
        "base_wall": float(old.get("wall_seconds") or 0),
        "base_valid": bool(old.get("validate_ok")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument(
        "--wall-timeout",
        type=float,
        default=0.0,
        help="wall seconds; <=0 means no wall limit (run until sim finishes)",
    )
    ap.add_argument("--max-sim-time", type=int, default=200000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    jsonl = OUT / "results.jsonl"
    baseline = _load_baseline_hier()
    print(f"[stall_sim30] baseline hier maps={len(baseline)}", flush=True)

    done_slots = set()
    rows: List[dict] = []
    if args.resume and jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            done_slots.add(int(r.get("slot") or -1))
        print(f"[stall_sim30] resume {len(done_slots)}", flush=True)
    else:
        jsonl.write_text("", encoding="utf-8")

    for slot in range(int(args.start), int(args.end) + 1):
        if slot in done_slots:
            print(f"[skip] SH{slot:02d}", flush=True)
            continue
        print(f"\n===== SH{slot:02d} Hier stall_sim30 =====", flush=True)
        t0 = time.perf_counter()
        row = run_method(
            "hier",
            slot,
            wall_timeout=float(args.wall_timeout),
            max_sim_time=int(args.max_sim_time),
            reuse_pibt=False,
        )
        row["eval"] = "stall_sim30_v3_near_done_flat"
        row["wall_seconds"] = round(
            float(row.get("wall_seconds") or (time.perf_counter() - t0)), 2
        )
        row["vs_baseline"] = _delta(row, baseline.get(slot))
        with jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows = [r for r in rows if int(r.get("slot") or -1) != slot] + [row]
        _write(rows)
        vb = row["vs_baseline"]
        print(
            f"[stall_sim30] SH{slot:02d} done={row.get('tasks_completed')}/"
            f"{row.get('tasks_total')} valid={row.get('validate_ok')} "
            f"sim={row.get('sim_time')} wall={row.get('wall_seconds')}s "
            f"d_sim={vb.get('d_sim')} d_wall={vb.get('d_wall')}",
            flush=True,
        )

    _write(rows)
    print(f"[stall_sim30] wrote {OUT}", flush=True)
    return 0


def _write(rows: List[Dict[str, Any]]) -> None:
    (OUT / "summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Hier StallEscalateNet v3 (sim30 + near-done flat) vs compare_100 baseline",
        "",
        "| slot | done | VALID | sim | wall | Δsim | Δwall |",
        "|------|------|-------|-----|------|------|-------|",
    ]
    for r in sorted(rows, key=lambda x: int(x.get("slot") or 0)):
        vb = r.get("vs_baseline") or {}
        lines.append(
            f"| SH{int(r.get('slot') or 0):02d} | "
            f"{r.get('tasks_completed')}/{r.get('tasks_total')} | "
            f"{'Y' if r.get('validate_ok') else 'N'} | {r.get('sim_time')} | "
            f"{r.get('wall_seconds')} | {vb.get('d_sim', '-')} | "
            f"{vb.get('d_wall', '-')} |"
        )
    # Aggregate vs baseline where available
    paired = [r for r in rows if (r.get("vs_baseline") or {}).get("has_baseline")]
    if paired:
        n = len(paired)
        avg_dsim = sum(int((r["vs_baseline"]["d_sim"])) for r in paired) / n
        avg_dwall = sum(float((r["vs_baseline"]["d_wall"])) for r in paired) / n
        avg_sim = sum(int(r.get("sim_time") or 0) for r in paired) / n
        avg_wall = sum(float(r.get("wall_seconds") or 0) for r in paired) / n
        full = sum(1 for r in paired if int(r.get("tasks_completed") or 0) >= 100)
        valid = sum(1 for r in paired if r.get("validate_ok"))
        lines += [
            "",
            "## Aggregate (paired with baseline)",
            "",
            f"- maps: {n}",
            f"- full100: {full}/{n}",
            f"- VALID: {valid}/{n}",
            f"- avg_sim: {avg_sim:.1f} (Δ {avg_dsim:+.1f})",
            f"- avg_wall: {avg_wall:.1f}s (Δ {avg_dwall:+.1f})",
            "",
            "Negative Δsim / Δwall = better than baseline Hier.",
        ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
