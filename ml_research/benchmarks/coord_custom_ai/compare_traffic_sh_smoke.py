"""SH01-20 A/B: hierarchical WaveNet with TrafficNet ON vs OFF."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.common.paths import RESULTS

OUT = RESULTS / "coord_custom_ai" / "traffic_ab_sh"
OUT.mkdir(parents=True, exist_ok=True)


def _run_one(
    slot: int,
    *,
    meta: dict,
    max_tasks: int,
    max_active: int,
    time_limit: float,
    use_traffic: bool,
) -> dict:
    t0 = time.perf_counter()
    try:
        rep = solve_ecbs(
            slot=slot,
            meta=meta,
            max_tasks=max_tasks,
            max_active=max_active,
            weight=1.5,
            time_limit=time_limit,
            plan="joint",
            pipeline=True,
            turn_aware=True,
            initial_park=False,
            use_hierarchical=True,
            use_wavenet=True,
            hierarchical_force=False,
            wave_hard_cap=4,
            use_traffic_accel=bool(use_traffic),
        )
        wall = round(time.perf_counter() - t0, 2)
        cr = float(rep.get("completion_ratio") or 0)
        ok = bool(rep.get("validate_ok"))
        failed = rep.get("tasks_failed") or []
        hier = rep.get("hierarchical") or {}
        status = "PASS" if ok and cr >= 0.999 and not failed else "FAIL"
        return {
            "status": status,
            "completion_ratio": cr,
            "tasks_completed": rep.get("tasks_completed"),
            "tasks_total": rep.get("tasks_total"),
            "sim_time": rep.get("sim_time"),
            "wall_seconds": wall,
            "validate_ok": ok,
            "n_failed": len(failed),
            "waves_escalated": hier.get("waves_escalated"),
            "accel_waves": hier.get("accel_waves"),
            "last_accel": hier.get("last_accel"),
            "map_hardness": hier.get("hardness") or hier.get("map_hardness"),
            "last_k": hier.get("last_k"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "CRASH",
            "error": str(exc),
            "wall_seconds": round(time.perf_counter() - t0, 2),
            "traceback": traceback.format_exc()[-800:],
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=str, default="1-20")
    ap.add_argument("--max-tasks", type=int, default=12)
    ap.add_argument("--max-active", type=int, default=4)
    ap.add_argument("--time-limit", type=float, default=25.0)
    args = ap.parse_args()

    slots: list[int] = []
    for part in args.slots.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            slots.extend(range(int(a), int(b) + 1))
        else:
            slots.append(int(part))

    rows: list[dict] = []
    t_all = time.perf_counter()
    print(
        f"[TRAFFIC-AB] slots={slots} max_tasks={args.max_tasks} "
        f"k={args.max_active}",
        flush=True,
    )

    for slot in slots:
        meta = load_custom_meta(slot, max_sim_time=300000, wall_timeout=1e9)
        if not meta:
            row = {"slot": slot, "status": "missing"}
            rows.append(row)
            print(f"  SH{slot:02d} MISSING", flush=True)
            continue

        print(f"\n======== SH{slot:02d} Traffic ON ========", flush=True)
        on = _run_one(
            slot,
            meta=meta,
            max_tasks=args.max_tasks,
            max_active=args.max_active,
            time_limit=args.time_limit,
            use_traffic=True,
        )
        print(
            f"  ON  [{on.get('status')}] sim={on.get('sim_time')} "
            f"accel={on.get('accel_waves')} wall={on.get('wall_seconds')}s",
            flush=True,
        )

        print(f"======== SH{slot:02d} Traffic OFF ========", flush=True)
        off = _run_one(
            slot,
            meta=meta,
            max_tasks=args.max_tasks,
            max_active=args.max_active,
            time_limit=args.time_limit,
            use_traffic=False,
        )
        print(
            f"  OFF [{off.get('status')}] sim={off.get('sim_time')} "
            f"accel={off.get('accel_waves')} wall={off.get('wall_seconds')}s",
            flush=True,
        )

        sim_on = on.get("sim_time")
        sim_off = off.get("sim_time")
        delta = None
        if isinstance(sim_on, (int, float)) and isinstance(sim_off, (int, float)):
            delta = int(sim_off) - int(sim_on)  # OFF - ON; <0 means OFF faster

        row = {
            "slot": slot,
            "on": on,
            "off": off,
            "delta_sim_off_minus_on": delta,
            "winner": (
                "off"
                if delta is not None and delta < 0
                else "on"
                if delta is not None and delta > 0
                else "tie"
                if delta == 0
                else "n/a"
            ),
        }
        rows.append(row)
        (OUT / f"SH{slot:02d}.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"  DELTA off-on={delta} winner={row['winner']}",
            flush=True,
        )
        # progressive summary so partial runs are usable
        _write_summary(rows, args, t_all)

    summary = _write_summary(rows, args, t_all)
    print("\n========== TRAFFIC A/B SUMMARY ==========", flush=True)
    print(
        f"on_pass={summary['on_pass']} off_pass={summary['off_pass']} "
        f"off_faster={summary['off_faster']} on_faster={summary['on_faster']} "
        f"tie={summary['tie']} wall={summary['wall_total']}s",
        flush=True,
    )
    for r in rows:
        if r.get("status") == "missing":
            print(f"  SH{int(r['slot']):02d} missing", flush=True)
            continue
        on, off = r["on"], r["off"]
        print(
            f"  SH{int(r['slot']):02d} "
            f"ON={on.get('status'):5s} sim={str(on.get('sim_time')):5s} "
            f"OFF={off.get('status'):5s} sim={str(off.get('sim_time')):5s} "
            f"d={r.get('delta_sim_off_minus_on')} {r.get('winner')}",
            flush=True,
        )
    print(f"wrote {OUT / 'summary.json'}", flush=True)
    return 0


def _write_summary(rows: list[dict], args, t_all: float) -> dict:
    cmp_rows = [r for r in rows if "on" in r]
    summary = {
        "max_tasks": args.max_tasks,
        "max_active": args.max_active,
        "wall_total": round(time.perf_counter() - t_all, 2),
        "on_pass": sum(1 for r in cmp_rows if r["on"].get("status") == "PASS"),
        "off_pass": sum(1 for r in cmp_rows if r["off"].get("status") == "PASS"),
        "off_faster": sum(1 for r in cmp_rows if r.get("winner") == "off"),
        "on_faster": sum(1 for r in cmp_rows if r.get("winner") == "on"),
        "tie": sum(1 for r in cmp_rows if r.get("winner") == "tie"),
        "rows": rows,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
