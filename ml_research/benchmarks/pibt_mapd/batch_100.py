"""Batch: SH01–SH20 × official 100 tasks with independent PIBT-MAPD."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.pibt_mapd.solve_pibt import OUT, solve_pibt

# Re-export OUT for clarity; batch writes beside solver outputs.
OUT.mkdir(parents=True, exist_ok=True)
JSONL = OUT / "results.jsonl"
SUMMARY = OUT / "SUMMARY.md"
EXCEL_DIR = OUT / "excel"
_EXCEL_ROWS = 1_000_000


def _write_excel(slot: int, rep: Dict[str, Any]) -> str:
    import pandas as pd

    EXCEL_DIR.mkdir(parents=True, exist_ok=True)
    path = EXCEL_DIR / f"SH{int(slot):02d}.xlsx"
    summary = pd.DataFrame(
        [
            ("地图", f"SH{int(slot):02d}"),
            ("方法", "pibt_mapd_v1"),
            ("完成单数", rep.get("tasks_completed")),
            ("总单数", rep.get("tasks_total")),
            ("仿真时间", rep.get("sim_time")),
            ("墙钟秒", rep.get("wall_seconds")),
            ("校验", "VALID" if rep.get("validate_ok") else "INVALID"),
            ("校验摘要", rep.get("validate_summary")),
            ("碰撞", rep.get("n_collisions")),
            ("对穿", rep.get("n_swaps")),
            ("非法动作", rep.get("n_illegal_motion")),
            ("FIFO", rep.get("n_fifo")),
            ("队首不符", rep.get("n_display_mismatch")),
            ("轨迹文件", rep.get("trajectory")),
        ],
        columns=["项目", "值"],
    )
    traj_path = Path(str(rep.get("trajectory") or ""))
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="结果", index=False)
        if traj_path.is_file():
            traj = pd.read_csv(traj_path)
            if len(traj) == 0:
                traj.to_excel(writer, sheet_name="轨迹", index=False)
            else:
                for i, start in enumerate(range(0, len(traj), _EXCEL_ROWS)):
                    name = "轨迹" if i == 0 else f"轨迹{i + 1}"
                    traj.iloc[start : start + _EXCEL_ROWS].to_excel(
                        writer, sheet_name=name, index=False
                    )
    print(f"[excel] {path}", flush=True)
    return str(path)


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
    meta["id"] = f"SH{int(slot):02d}_pibt100"
    meta["n_tasks"] = 100
    print(f"\n===== SH{int(slot):02d} ×100 PIBT =====", flush=True)
    t0 = time.perf_counter()
    try:
        rep = solve_pibt(
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
    excel = ""
    try:
        excel = _write_excel(int(slot), rep)
    except Exception as exc:  # noqa: BLE001
        excel = f"excel_fail:{type(exc).__name__}: {exc}"
        print(f"[excel] SH{slot:02d} {excel}", flush=True)
    row = {
        "slot": slot,
        "scenario_id": meta["id"],
        "method": "pibt_mapd_v1",
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
        "excel": excel,
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
    lines = [
        "# PIBT-MAPD × SH01–SH20 (official 100)",
        "",
        f"- maps: {len(rows)}",
        f"- VALID: {len(ok)}/{len(rows)}",
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
    (OUT / "summary.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--wall-timeout", type=float, default=1800.0)
    ap.add_argument("--max-sim-time", type=int, default=200000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    done_slots: set = set()
    rows: List[Dict[str, Any]] = []
    if args.resume and JSONL.is_file():
        for line in JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            done_slots.add(int(r.get("slot") or -1))
        print(f"[resume] loaded {len(done_slots)} slots", flush=True)
    else:
        JSONL.write_text("", encoding="utf-8")

    for slot in range(int(args.start), int(args.end) + 1):
        if slot in done_slots:
            print(f"[skip] SH{slot:02d}", flush=True)
            continue
        row = _run_slot(
            slot,
            wall_timeout=float(args.wall_timeout),
            max_sim_time=int(args.max_sim_time),
        )
        with JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows = [r for r in rows if int(r.get("slot") or -1) != slot] + [row]
        _write_summary(rows)

    _write_summary(rows)
    print(f"\n[batch] wrote {SUMMARY}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
