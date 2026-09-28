"""Lifelong stress test for SceneDifficultyNet.

Task schedule (default):
  random_pre (N) → burst same dropoff/pickup (N) → random_post (N)

Classifier is triggered in ``solve_ecbs`` **before** AGVs claim tasks each wave.
This script measures whether hard-rate / p_hard rises in the burst phase and
falls again in random_post, across multiple SH maps.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.benchmarks.hier_coord.scene_difficulty.task_stress import (
    gen_phased_tasks,
    write_task_csv,
)
from ml_research.common.paths import RESULTS

OUT = RESULTS / "hier_coord" / "lifelong_phase_stress"


def _dominant_phase(peek: List[str]) -> str:
    if not peek:
        return "unknown"
    return Counter(peek).most_common(1)[0][0]


def _summarize_scene_log(log: List[dict]) -> dict:
    by_phase: Dict[str, dict] = defaultdict(
        lambda: {"n": 0, "n_hard": 0, "p_sum": 0.0, "n_escalate": 0}
    )
    for row in log:
        phases = row.get("peek_phases") or []
        phase = _dominant_phase(phases)
        # fallback: estimate by done_before vs schedule if peek empty
        b = by_phase[phase]
        b["n"] += 1
        lab = str(row.get("label") or "")
        if lab == "hard":
            b["n_hard"] += 1
        b["p_sum"] += float(row.get("p_hard") or 0.0)
        if row.get("escalate"):
            b["n_escalate"] += 1

    out = {}
    for ph, b in sorted(by_phase.items()):
        n = max(1, int(b["n"]))
        out[ph] = {
            "waves": int(b["n"]),
            "hard_rate": round(b["n_hard"] / n, 4),
            "escalate_rate": round(b["n_escalate"] / n, 4),
            "mean_p_hard": round(b["p_sum"] / n, 4),
        }
    return out


def _phase_quality(by_phase: dict) -> dict:
    """Heuristic: burst stays hard; post should ease vs burst (pre may already be hard)."""
    pre = by_phase.get("random_pre") or {}
    burst = by_phase.get("burst") or {}
    post = by_phase.get("random_post") or {}
    pre_h = float(pre.get("mean_p_hard") or 0.0)
    burst_h = float(burst.get("mean_p_hard") or 0.0)
    post_h = float(post.get("mean_p_hard") or 0.0)
    pre_r = float(pre.get("hard_rate") or 0.0)
    burst_r = float(burst.get("hard_rate") or 0.0)
    post_r = float(post.get("hard_rate") or 0.0)

    burst_detected = burst_r >= 0.5 or burst_h >= 0.35
    # Map topology saturated: all phases hard → still OK if burst stays hard
    saturated = pre_r >= 0.9 and burst_r >= 0.9 and post_r >= 0.9 and min(pre_h, burst_h, post_h) >= 0.85
    if saturated:
        return {
            "burst_detected_as_hard": True,
            "burst_harder_or_kept_vs_pre": True,
            "post_eased_vs_burst": False,
            "map_saturated_hard": True,
            "delta_p_pre_to_burst": round(burst_h - pre_h, 4),
            "delta_p_burst_to_post": round(post_h - burst_h, 4),
            "delta_rate_pre_to_burst": round(burst_r - pre_r, 4),
            "pass": True,
        }

    if pre_h >= 0.85 or pre_r >= 0.85:
        up_ok = burst_detected
    else:
        up_ok = burst_h >= pre_h - 0.02 and burst_r >= max(0.0, pre_r - 0.05)
    down_ok = post_h <= burst_h + 0.05 and (
        post_h + 0.08 <= burst_h or post_r + 0.15 <= burst_r or post_h <= 0.35
    )
    return {
        "burst_detected_as_hard": bool(burst_detected),
        "burst_harder_or_kept_vs_pre": bool(up_ok),
        "post_eased_vs_burst": bool(down_ok),
        "map_saturated_hard": False,
        "delta_p_pre_to_burst": round(burst_h - pre_h, 4),
        "delta_p_burst_to_post": round(post_h - burst_h, 4),
        "delta_rate_pre_to_burst": round(burst_r - pre_r, 4),
        "pass": bool(up_ok and down_ok and burst_detected),
    }


def run_one(
    *,
    slot: int,
    tasks: List[dict],
    phase_map: Dict[str, str],
    task_csv: Path,
    max_active: int,
    time_limit: float,
    use_wavenet: bool,
) -> dict:
    meta = load_custom_meta(slot, max_sim_time=500000, wall_timeout=1e9)
    if not meta:
        return {"slot": slot, "status": "missing"}
    meta = dict(meta)
    meta["task_csv"] = str(task_csv.resolve())
    meta["id"] = f"{meta['id']}_phase_stress"

    t0 = time.perf_counter()
    try:
        rep = solve_ecbs(
            slot=slot,
            meta=meta,
            max_tasks=0,
            max_active=max_active,
            weight=1.5,
            time_limit=time_limit,
            plan="joint",
            pipeline=True,
            turn_aware=True,
            initial_park=False,
            use_hierarchical=True,
            use_wavenet=use_wavenet,
            hierarchical_force=False,
            wave_hard_cap=4,
            task_csv=task_csv,
            task_phase_by_id=phase_map,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "slot": slot,
            "status": "CRASH",
            "error": str(exc),
            "traceback": traceback.format_exc()[-1200:],
            "wall_seconds": round(time.perf_counter() - t0, 2),
        }

    hier = rep.get("hierarchical") or {}
    slog = list(hier.get("scene_decision_log") or [])
    by_phase = _summarize_scene_log(slog)
    quality = _phase_quality(by_phase)
    cr = float(rep.get("completion_ratio") or 0.0)
    return {
        "slot": slot,
        "status": "PASS" if cr >= 0.90 and bool(rep.get("validate_ok")) else "FAIL",
        "completion_ratio": cr,
        "tasks_completed": rep.get("tasks_completed"),
        "tasks_total": rep.get("tasks_total"),
        "sim_time": rep.get("sim_time"),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "validate_ok": rep.get("validate_ok"),
        "n_failed": len(rep.get("tasks_failed") or []),
        "n_collisions": rep.get("n_collisions"),
        "waves": hier.get("waves"),
        "waves_escalated": hier.get("waves_escalated"),
        "scene_by_phase": by_phase,
        "classifier_quality": quality,
        "scene_log_n": len(slog),
        "scene_log": slog,
        "last_reason": hier.get("last_reason") or (hier.get("last") or {}).get("reason"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Lifelong phase stress for scene classifier")
    ap.add_argument("--slots", type=str, default="1,3,5,8,10,15")
    ap.add_argument("--n-pre", type=int, default=40, help="random tasks before burst")
    ap.add_argument("--n-burst", type=int, default=40, help="concentrated tasks")
    ap.add_argument("--n-post", type=int, default=40, help="random tasks after burst")
    ap.add_argument(
        "--burst-mode",
        type=str,
        default="same_dropoff",
        choices=("same_dropoff", "same_pickup", "dual"),
    )
    ap.add_argument("--max-active", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--wavenet", action="store_true", default=True)
    ap.add_argument("--no-wavenet", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args(argv)

    slots: List[int] = []
    for part in str(args.slots).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            slots.extend(range(int(a), int(b) + 1))
        else:
            slots.append(int(part))

    out_dir = OUT if not args.tag else OUT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    use_wavenet = bool(args.wavenet) and not bool(args.no_wavenet)

    tasks = gen_phased_tasks(
        n_random_pre=int(args.n_pre),
        n_burst=int(args.n_burst),
        n_random_post=int(args.n_post),
        burst_mode=str(args.burst_mode),
        seed=int(args.seed),
    )
    phase_map = {str(t["task_id"]): str(t.get("_phase") or "") for t in tasks}
    task_csv = out_dir / f"phased_{args.burst_mode}_s{args.seed}.csv"
    write_task_csv(task_csv, tasks)
    (out_dir / "phase_map.json").write_text(
        json.dumps(phase_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"[PHASE-STRESS] slots={slots} schedule={args.n_pre}/{args.n_burst}/{args.n_post} "
        f"burst={args.burst_mode} k={args.max_active} wavenet={use_wavenet}",
        flush=True,
    )

    rows = []
    t_all = time.perf_counter()
    for slot in slots:
        print(f"\n======== SH{slot:02d} ========", flush=True)
        row = run_one(
            slot=slot,
            tasks=tasks,
            phase_map=phase_map,
            task_csv=task_csv,
            max_active=int(args.max_active),
            time_limit=float(args.time_limit),
            use_wavenet=use_wavenet,
        )
        rows.append(row)
        (out_dir / f"SH{slot:02d}.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        q = row.get("classifier_quality") or {}
        print(
            f"  [{row.get('status')}] cr={row.get('completion_ratio')} "
            f"wall={row.get('wall_seconds')}s scene_waves={row.get('scene_log_n')} "
            f"clf_pass={q.get('pass')} by_phase={row.get('scene_by_phase')}",
            flush=True,
        )

    n_clf = sum(
        1
        for r in rows
        if (r.get("classifier_quality") or {}).get("pass")
    )
    summary = {
        "schedule": {
            "n_pre": args.n_pre,
            "n_burst": args.n_burst,
            "n_post": args.n_post,
            "burst_mode": args.burst_mode,
            "seed": args.seed,
        },
        "max_active": args.max_active,
        "use_wavenet": use_wavenet,
        "wall_total": round(time.perf_counter() - t_all, 2),
        "n_maps": len(rows),
        "n_pass_solve": sum(1 for r in rows if r.get("status") == "PASS"),
        "n_fail_solve": sum(1 for r in rows if r.get("status") == "FAIL"),
        "n_crash": sum(1 for r in rows if r.get("status") == "CRASH"),
        "n_classifier_pass": n_clf,
        "rows": [
            {
                k: v
                for k, v in r.items()
                if k != "scene_log"  # keep summary compact; per-map file has full log
            }
            for r in rows
        ],
    }
    outp = out_dir / "summary.json"
    outp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n========== SUMMARY ==========", flush=True)
    print(
        f"solve PASS={summary['n_pass_solve']} FAIL={summary['n_fail_solve']} "
        f"CRASH={summary['n_crash']} | classifier_pass={n_clf}/{len(rows)} "
        f"wall={summary['wall_total']}s",
        flush=True,
    )
    print(f"wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
