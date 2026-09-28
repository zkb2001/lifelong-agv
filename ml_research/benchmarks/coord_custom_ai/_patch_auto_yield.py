from pathlib import Path
import ast

p = Path(r"d:/compitition/MioVerse/final_version/ml_research/benchmarks/coord_custom_ai/solve_ecbs.py")
text = p.read_text(encoding="utf-8")
old = '''        force_reason = ""
        cold = t < int(min_handoff_t) or done_est < int(min_handoff_done)
        if int(stall_state["idle_surface_streak"]) >= _need(idle_need, floor=2):
            force_reason = (
                f"idle_surface_streak={stall_state['idle_surface_streak']}"
            )
        elif (
            int(stall_state["done_plateau_streak"]) >= _need(plateau_need, floor=2)
            and sig["has_work"]
        ):
            force_reason = (
                f"done_plateau_streak={stall_state['done_plateau_streak']}"
            )
'''
new = '''        force_reason = ""
        cold = t < int(min_handoff_t) or done_est < int(min_handoff_done)
        # Near completion: last crumbs often look like idle+surface but are not
        # map hardness — do not hand off into wave ECBS for that.
        remaining = max(0, int(n_tasks) - int(done_est))
        near_done = remaining <= max(2, int(round(0.01 * float(n_tasks))))
        if near_done:
            force_reason = ""
        elif (
            int(stall_state["idle_surface_streak"]) >= _need(idle_need, floor=2)
            and sig["has_work"]
            and not cold
        ):
            force_reason = (
                f"idle_surface_streak={stall_state['idle_surface_streak']}"
            )
        elif (
            int(stall_state["done_plateau_streak"]) >= _need(plateau_need, floor=2)
            and sig["has_work"]
            and not cold
        ):
            force_reason = (
                f"done_plateau_streak={stall_state['done_plateau_streak']}"
            )
'''
if text.count(old) != 1:
    raise SystemExit(f"force_reason anchor count={text.count(old)}")
text = text.replace(old, new, 1)

# Simplify allow_yield call site comment is enough; keep AGV_M0_HANDOFF opt-out.

ast.parse(text)
p.write_text(text, encoding="utf-8")
print("solve_ecbs near_done guard ok", p.stat().st_size)
