"""Re-enable mid-map yield when M0/A* is slower than wall standards."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOLVE = ROOT / "solve_ecbs.py"
BATCH = ROOT / "batch_400_switch_20.py"


def patch_solve() -> None:
    text = SOLVE.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # --- replace do_rulepark block ---
    start = None
    end = None
    for i, l in enumerate(lines):
        if "Mid/easy maps (hardness < scene_t2)" in l or (
            start is None and "do_rulepark = bool(allow_switch)" in l
        ):
            if start is None:
                # include preceding comment lines
                j = i
                while j > 0 and lines[j - 1].lstrip().startswith("#"):
                    j -= 1
                start = j
        if start is not None and 'if not m0_rep.get("handoff"):' in l:
            end = i
            break
    if start is None or end is None:
        raise SystemExit(f"rulepark bounds start={start} end={end}")

    new_block = '''            # Easy/mid start: M0+SwapNet WITH yield. If wall/tick or
            # wall/done exceeds batch standards -> medium/ECBS wave.
            # (RulePark-only no-yield left SH02 crawling at ~40s/task.)
            want_yield = (
                bool(allow_switch)
                and str(_os.environ.get("AGV_M0_HANDOFF", "1")).strip().lower()
                not in ("0", "false", "no", "off")
                and meta.get("m0_handoff_enabled", True) is not False
            )
            if allow_switch:
                print(
                    "[HIER] allow_mode_switch=1 -> M0+SwapNet+yield "
                    f"(probe={probe.planner}/{gate.last_scene_label} "
                    f"h={float(hardness0):.3f})",
                    flush=True,
                )
            m0_rep = _solve_via_m0_engine(
                meta=meta,
                task_csv=eng_csv,
                total=int(n_w),
                use_swapnet=bool(do_swap),
                swapnet_margin=float(swapnet_margin),
                hier_stats=hier_stats,
                t0=t0,
                plan=plan,
                max_active=max_active,
                weight=weight,
                pipeline=pipeline,
                turn_aware=turn_aware,
                plan_horizon=plan_horizon,
                exec_horizon=exec_horizon,
                use_hierarchical=use_hierarchical,
                use_wavenet=use_wavenet,
                gate=gate if want_yield else None,
                allow_yield=bool(want_yield),
                traffic_recovery=False,
            )
'''
    lines = lines[:start] + [new_block if new_block.endswith("\n") else new_block + "\n"] + lines[end:]
    text = "".join(lines)

    # --- wall_done_cost params ---
    if "wall_done_cost_s" not in text:
        needle = '    wall_window_min_done = int(meta.get("m0_wall_window_min_done") or 2)'
        if needle not in text:
            raise SystemExit("wall_window_min_done not found")
        text = text.replace(
            needle,
            needle
            + "\n"
            + "    # Avg wall-seconds per completed task; SH01~0.8s, mid crawl>>.\n"
            + '    wall_done_cost_s = float(meta.get("m0_wall_done_cost_s") or 12.0)\n'
            + '    wall_done_cost_min_done = int(meta.get("m0_wall_done_cost_min_done") or 8)',
            1,
        )

    if "wall_done_cost=" not in text:
        old_idle = '''        elif (
            int(stall_state["idle_surface_streak"]) >= _need(idle_need, floor=2)
            and sig["has_work"]
            and not cold
        ):
            force_reason = (
                f"idle_surface_streak={stall_state['idle_surface_streak']}"
            )
'''
        insert = '''        elif (
            done_est >= int(wall_done_cost_min_done)
            and sig["has_work"]
            and remaining > int(wall_done_cost_min_done)
            and (float(wall_elapsed) / max(1, int(done_est))) >= float(wall_done_cost_s)
        ):
            cost = float(wall_elapsed) / max(1, int(done_est))
            force_reason = (
                f"wall_done_cost={cost:.1f}s/task>={wall_done_cost_s:.1f}"
            )
        elif (
            int(stall_state["idle_surface_streak"]) >= _need(idle_need, floor=2)
            and sig["has_work"]
            and not cold
        ):
            force_reason = (
                f"idle_surface_streak={stall_state['idle_surface_streak']}"
            )
'''
        if old_idle not in text:
            raise SystemExit("idle_surface block not found")
        text = text.replace(old_idle, insert, 1)

    SOLVE.write_text(text, encoding="utf-8")
    print(f"patched {SOLVE.name} size={SOLVE.stat().st_size}")


def patch_batch() -> None:
    text = BATCH.read_text(encoding="utf-8")
    start = text.find('    meta["m0_assign_fail_handoff"]')
    end = text.find('    meta["m0_handoff_enabled"] = True')
    if start < 0 or end < 0:
        raise SystemExit("batch meta block not found")
    end = text.find("\n", end) + 1
    block = '''    # Yield ON: if M0/A* slower than standards -> ECBS/hybrid wave.
    # Standards vs SH01 (~5 sim/s, ~0.8s wall/task): mid crawl was ~40s/task.
    meta["m0_assign_fail_handoff"] = 12
    meta["m0_min_handoff_t"] = 40
    meta["m0_min_handoff_wall_s"] = 45.0
    # >1.0s wall per sim-tick for 2 gate windows -> too slow.
    meta["m0_wall_tick_s"] = 1.0
    meta["m0_wall_tick_streak"] = 2
    meta["m0_wall_starve_s"] = 90.0
    meta["m0_wall_starve_done"] = 15
    meta["m0_wall_starve_min_t"] = 40
    # 45s wall window must advance >=20 sim and >=2 completions.
    meta["m0_wall_window_s"] = 45.0
    meta["m0_wall_window_min_sim"] = 20
    meta["m0_wall_window_min_done"] = 2
    # Avg wall cost per done task (bypasses cold).
    meta["m0_wall_done_cost_s"] = 8.0
    meta["m0_wall_done_cost_min_done"] = 6
    meta["m0_throughput_drop_streak"] = 2
    meta["m0_done_plateau_streak"] = 3
    meta["handback_calm"] = 5
    meta["handback_dwell"] = 200
    meta["m0_handback"] = False
    meta["max_handbacks"] = 2
    meta["serial_recover_budget"] = 6
    meta["m0_handback_astar_wall_s"] = 2.0
    meta["m0_handoff_enabled"] = True
'''
    text = text[:start] + block + text[end:]
    BATCH.write_text(text, encoding="utf-8")
    print(f"patched {BATCH.name}")


if __name__ == "__main__":
    patch_solve()
    patch_batch()
