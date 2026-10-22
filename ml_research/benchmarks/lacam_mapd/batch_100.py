"""Batch: SH01–SH20 × official 100 tasks with LaCAM*-MAPD."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.lacam_mapd.solve_lacam import OUT, solve_lacam

OUT.mkdir(parents=True, exist_ok=True)
JSONL = OUT / "results.jsonl"
SUMMARY = OUT / "SUMMARY.md"


def _run_slot(slot: int, *, wall_timeout: float, max_sim_time: int) -> Dict[str, Any]:
    meta = load_custom_meta(
        int(slot),
        max_sim_time=int(max_sim_time),
        wall_timeout=float(wall_timeout),
        n_tasks=100,
    )
    if not meta:
        return {"slot": slot, "error": "meta_missing", "validate_ok": False}
    meta = dict(meta)
    meta["id"] = f"SH{int(slot):02d}_lacam100"
    meta["n_tasks"] = 100
    print(f"\n===== SH{int(slot):02d} ×100 LaCAM* =====", flush=True)
    t0 = time.perf_counter()
    try:
        rep = solve_lacam(
            int(slot),
            meta=meta,
            wall_timeout=float(wall_timeout),
            max_sim_time=int(max_sim_time),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "slot": slot,
            "error": f"{type(exc).__name__}: {exc}",
            "validate_ok": False,
            "wall_seconds": round(time.perf_counter() - t0, 2),
        }
    row = {
        "slot": slot,
        "scenario_id": meta["id"],
        "method": "lacam_star_mapd_v1",
        "sim_time": rep.get("sim_time"),
        "completion_ratio": rep.get("completion_ratio"),
        "validate_ok": rep.get("validate_ok"),
        "validate_summary": rep.get("validate_summary"),
        "tasks_completed": rep.get("tasks_completed"),
        "tasks_total": rep.get("tasks_total"),
        "wall_seconds": rep.get("wall_seconds"),
        "n_fifo": rep.get("n_fifo"),
        "n_collisions": rep.get("n_collisions"),
        "trajectory": rep.get("trajectory"),
        "error": None,
    }
    print(
        f"[done] SH{slot:02d} sim={row['sim_time']} valid={row['validate_ok']} "
        f"done={row['tasks_completed']}/{row['tasks_total']} "
        f"wall={row['wall_seconds']}s {row['validate_summary']}",
        flush=True,
    )
    return row


def _write_summary(rows: List[Dict[str, Any]]) -> None:
    ok = [r for r in rows if r.get("validate_ok")]
    done100 = [r for r in ok if int(r.get("tasks_completed") or 0) >= 100]
    lines = [
        "# LaCAM*-MAPD × SH01–SH20 (official 100)",
        "",
        f"- maps: {len(rows)}",
        f"- VALID: {len(ok)}/{len(rows)}",
        f"- done=100: {len(done100)}/{len(rows)}",
        "",
        "| Slot | sim_t | VALID | done | wall_s | notes |",
        "|------|-------|-------|------|--------|-------|",
    ]
    for r in sorted(rows, key=lambda x: int(x.get("slot") or 0)):
        if r.get("error"):
            lines.append(
                f"| SH{int(r['slot']):02d} | - | - | - | - | {r['error']} |"
            )
            continue
        notes = str(r.get("validate_summary") or "")[:48]
        lines.append(
            f"| SH{int(r['slot']):02d} | {r.get('sim_time')} | "
            f"{'Y' if r.get('validate_ok') else 'N'} | "
            f"{r.get('tasks_completed')}/{r.get('tasks_total')} | "
            f"{r.get('wall_seconds')} | {notes} |"
        )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] {SUMMARY}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch LaCAM* MAPD SH01-20")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--wall-timeout", type=float, default=1800.0)
    ap.add_argument("--max-sim-time", type=int, default=200000)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    if JSONL.is_file():
        JSONL.write_text("", encoding="utf-8")
    for slot in range(int(args.start), int(args.end) + 1):
        row = _run_slot(
            slot,
            wall_timeout=float(args.wall_timeout),
            max_sim_time=int(args.max_sim_time),
        )
        rows.append(row)
        with JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        _write_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
