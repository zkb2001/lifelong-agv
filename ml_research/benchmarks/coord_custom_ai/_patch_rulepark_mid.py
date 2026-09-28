"""Easy/mid maps: M0+RulePark (no yield). Hard maps keep ECBS wave start."""
from __future__ import annotations

from pathlib import Path

SOLVE = Path(__file__).resolve().parent / "solve_ecbs.py"


def main() -> None:
    text = SOLVE.read_text(encoding="utf-8")
    old = '''            if allow_switch:
                print(
                    "[HIER] allow_mode_switch=1 → true M0+SwapNet with yield "
                    f"(probe={probe.planner}/{gate.last_scene_label})",
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
                gate=(
                    gate
                    if (
                        allow_switch
                        and str(_os.environ.get("AGV_M0_HANDOFF", "1")).strip().lower()
                        not in ("0", "false", "no", "off")
                        and meta.get("m0_handoff_enabled", True) is not False
                    )
                    else None
                ),
                allow_yield=bool(
                    allow_switch
                    and str(_os.environ.get("AGV_M0_HANDOFF", "1")).strip().lower()
                    not in ("0", "false", "no", "off")
                    and meta.get("m0_handoff_enabled", True) is not False
                ),
                traffic_recovery=False,
            )
'''
    new = '''            # Mid/easy maps (hardness < scene_t2): RulePark unsticks hubs.
            # Yield→ECBS/serial blew SH02/SH03 sim_t to 30k+. Hard maps never
            # enter this easy-probe branch (start hard→wave ECBS).
            scene_t2 = float(getattr(gate, "scene_t2", 0.35) or 0.35)
            do_rulepark = bool(allow_switch) and float(hardness0) < scene_t2
            want_yield = (
                bool(allow_switch)
                and (not do_rulepark)
                and str(_os.environ.get("AGV_M0_HANDOFF", "1")).strip().lower()
                not in ("0", "false", "no", "off")
                and meta.get("m0_handoff_enabled", True) is not False
            )
            if allow_switch:
                mode = "M0+RulePark(no-yield)" if do_rulepark else "M0+SwapNet+yield"
                print(
                    f"[HIER] allow_mode_switch=1 → {mode} "
                    f"(probe={probe.planner}/{gate.last_scene_label} "
                    f"h={float(hardness0):.3f}<t2={scene_t2:.2f}={int(do_rulepark)})",
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
                traffic_recovery=bool(do_rulepark),
            )
'''
    # The arrow char may be special - try flexible match
    if old not in text:
        # Try with replacement arrow variants
        old2 = old.replace("→", "\u2192")
        if old2 in text:
            old = old2
        else:
            # Find by unique substring
            idx = text.find("traffic_recovery=False,\n            )\n            if not m0_rep.get(\"handoff\"):")
            if idx < 0:
                raise SystemExit("anchor not found")
            # Use regex-free: locate print block start
            start = text.find(
                '            if allow_switch:\n                print(\n'
                '                    "[HIER] allow_mode_switch=1'
            )
            end = text.find(
                "            if not m0_rep.get(\"handoff\"):",
                start,
            )
            if start < 0 or end < 0:
                raise SystemExit(f"block bounds start={start} end={end}")
            text = text[:start] + new + text[end:]
            SOLVE.write_text(text, encoding="utf-8")
            print("patched via bounds", SOLVE.stat().st_size)
            return
    text = text.replace(old, new, 1)
    SOLVE.write_text(text, encoding="utf-8")
    print("patched via exact", SOLVE.stat().st_size)


if __name__ == "__main__":
    main()
