"""Smoke: SH01-SH20 through hierarchical ECBS (limited tasks)."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.common.paths import RESULTS

OUT = RESULTS / "coord_custom_ai" / "hier_sh_smoke"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=str, default="1-20")
    ap.add_argument("--max-tasks", type=int, default=12)
    ap.add_argument(
        "--max-active",
        type=int,
        default=0,
        help="0=等于场景 AGV 数（默认不人为限流）",
    )
    ap.add_argument("--time-limit", type=float, default=25.0)
    ap.add_argument("--no-pipeline", action="store_true")
    ap.add_argument(
        "--tag",
        type=str,
        default="",
        help="输出子目录后缀",
    )
    ap.add_argument(
        "--no-traffic-accel",
        action="store_true",
        default=True,
        help="关闭 TrafficNet-lite（默认关闭）",
    )
    args = ap.parse_args()
    out_dir = OUT if not args.tag else OUT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    slots: list[int] = []
    for part in args.slots.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            slots.extend(range(int(a), int(b) + 1))
        else:
            slots.append(int(part))

    rows = []
    t_all = time.perf_counter()
    print(
        f"[HIER-SMOKE] slots={slots} max_tasks={args.max_tasks} "
        f"k={args.max_active} pipeline={not args.no_pipeline} "
        f"tag={args.tag or '-'}",
        flush=True,
    )

    for slot in slots:
        meta = load_custom_meta(slot, max_sim_time=300000, wall_timeout=1e9)
        if not meta:
            rows.append({"slot": slot, "status": "missing"})
            print(f"  SH{slot:02d} MISSING", flush=True)
            continue
        print(f"\n======== SH{slot:02d} ========", flush=True)
        t0 = time.perf_counter()
        try:
            rep = solve_ecbs(
                slot=slot,
                meta=meta,
                max_tasks=args.max_tasks,
                max_active=args.max_active,
                weight=1.5,
                time_limit=args.time_limit,
                plan="joint",
                pipeline=not args.no_pipeline,
                turn_aware=True,
                initial_park=False,
                use_hierarchical=True,
                use_wavenet=True,
                hierarchical_force=False,
                wave_hard_cap=4,
                use_traffic_accel=False,
                use_lane_rules=False,
            )
            wall = round(time.perf_counter() - t0, 2)
            cr = float(rep.get("completion_ratio") or 0)
            ok = bool(rep.get("validate_ok"))
            failed = rep.get("tasks_failed") or []
            hier = rep.get("hierarchical") or {}
            status = "PASS" if ok and cr >= 0.999 and not failed else "FAIL"
            row = {
                "slot": slot,
                "status": status,
                "completion_ratio": cr,
                "tasks_completed": rep.get("tasks_completed"),
                "tasks_total": rep.get("tasks_total"),
                "sim_time": rep.get("sim_time"),
                "wall_seconds": wall,
                "validate_ok": ok,
                "n_failed": len(failed),
                "tasks_failed": failed[:8],
                "waves_escalated": hier.get("waves_escalated"),
                "waves_passthrough": hier.get("waves_passthrough"),
                "accel_waves": hier.get("accel_waves"),
                "last_reason": hier.get("last_reason"),
                "map_hardness": hier.get("hardness") or hier.get("map_hardness"),
            }
            print(
                f"  [{status}] cr={cr} sim={rep.get('sim_time')} "
                f"wall={wall}s esc={hier.get('waves_escalated')} "
                f"val={ok}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            wall = round(time.perf_counter() - t0, 2)
            row = {
                "slot": slot,
                "status": "CRASH",
                "error": str(exc),
                "wall_seconds": wall,
                "traceback": traceback.format_exc()[-800:],
            }
            print(f"  [CRASH] {exc}", flush=True)
        rows.append(row)
        (out_dir / f"SH{slot:02d}.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary = {
        "max_tasks": args.max_tasks,
        "max_active": args.max_active,
        "pipeline": not args.no_pipeline,
        "use_traffic_accel": not bool(args.no_traffic_accel),
        "wall_total": round(time.perf_counter() - t_all, 2),
        "n_pass": sum(1 for r in rows if r.get("status") == "PASS"),
        "n_fail": sum(1 for r in rows if r.get("status") == "FAIL"),
        "n_crash": sum(1 for r in rows if r.get("status") == "CRASH"),
        "n_missing": sum(1 for r in rows if r.get("status") == "missing"),
        "rows": rows,
    }
    outp = out_dir / f"summary_t{args.max_tasks}_k{args.max_active}.json"
    outp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== SUMMARY ==========", flush=True)
    print(
        f"PASS={summary['n_pass']} FAIL={summary['n_fail']} "
        f"CRASH={summary['n_crash']} MISSING={summary['n_missing']} "
        f"wall={summary['wall_total']}s",
        flush=True,
    )
    for r in rows:
        st = r.get("status")
        extra = ""
        if st == "PASS":
            extra = f"cr={r.get('completion_ratio')} sim={r.get('sim_time')}"
        elif st == "FAIL":
            extra = (
                f"cr={r.get('completion_ratio')} "
                f"failed={r.get('n_failed')} val={r.get('validate_ok')}"
            )
        elif st == "CRASH":
            extra = str(r.get("error") or "")[:80]
        print(f"  SH{int(r['slot']):02d} {st:6s} {extra}", flush=True)
    print(f"wrote {outp}", flush=True)
    return 0 if summary["n_fail"] == 0 and summary["n_crash"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
