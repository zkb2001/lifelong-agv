"""Patch M0 hard-livelock yield + cap serial recover carriers."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "solve_ecbs.py"


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    old = '''        force_reason = ""
        # Cold: need BOTH sim-t and wall still "fresh". Wall-clock rules bypass.
        cold = (t < int(min_handoff_t) and wall_elapsed < float(min_handoff_wall_s)) or (
            done_est < int(min_handoff_done)
        )
        remaining = max(0, int(n_tasks) - int(done_est))
        near_done = remaining <= max(2, int(round(0.01 * float(n_tasks))))
        if near_done:
            force_reason = ""
        elif (
            bool(stall_state.get("wall_window_hit"))
            and sig["has_work"]
            and t >= 10
        ):
            force_reason = (
                f"wall_window<{wall_window_s:.0f}s "
                f"dsim={stall_state['wall_window_dsim']}<{wall_window_min_sim} "
                f"or ddone={stall_state['wall_window_ddone']}<{wall_window_min_done}"
            )
        elif (
            int(stall_state["wall_tick_streak"]) >= int(wall_tick_streak_need)
            and sig["has_work"]
            and t >= 10
        ):
            force_reason = (
                f"wall_tick_streak={stall_state['wall_tick_streak']}"
                f"@{wall_tick_s:.2f}s/tick"
            )
        elif (
            wall_elapsed >= float(wall_starve_s)
            and done_est < int(wall_starve_done)
            and t >= int(wall_starve_min_t)
            and sig["has_work"]
            and remaining > int(wall_starve_done)
        ):
            force_reason = (
                f"wall_starve wall={wall_elapsed:.1f}s done={done_est}"
                f"<{wall_starve_done}"
            )
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
    new = '''        force_reason = ""
        # Cold: need BOTH sim-t and wall still "fresh". Wall-clock rules bypass.
        cold = (t < int(min_handoff_t) and wall_elapsed < float(min_handoff_wall_s)) or (
            done_est < int(min_handoff_done)
        )
        remaining = max(0, int(n_tasks) - int(done_est))
        near_done = remaining <= max(2, int(round(0.01 * float(n_tasks))))
        # Hard livelock: long done-plateau / idle+surface with exposed heads
        # must escape cold — otherwise mid maps freeze in M0 forever.
        hard_livelock = (
            sig["has_work"]
            and int(sig.get("n_surface") or 0) >= 3
            and t >= 80
            and (
                int(stall_state["done_plateau_streak"])
                >= max(4, _need(plateau_need, floor=2) + 1)
                or (
                    int(stall_state["idle_surface_streak"])
                    >= max(2, _need(idle_need, floor=2))
                    and bool(sig.get("idle_all"))
                )
            )
        )
        if near_done:
            force_reason = ""
        elif hard_livelock:
            force_reason = (
                f"hard_livelock plateau={stall_state['done_plateau_streak']} "
                f"idle_surf={stall_state['idle_surface_streak']} "
                f"surface={sig.get('n_surface')} cold={int(cold)}"
            )
        elif (
            bool(stall_state.get("wall_window_hit"))
            and sig["has_work"]
            and t >= 10
        ):
            force_reason = (
                f"wall_window<{wall_window_s:.0f}s "
                f"dsim={stall_state['wall_window_dsim']}<{wall_window_min_sim} "
                f"or ddone={stall_state['wall_window_ddone']}<{wall_window_min_done}"
            )
        elif (
            int(stall_state["wall_tick_streak"]) >= int(wall_tick_streak_need)
            and sig["has_work"]
            and t >= 10
        ):
            force_reason = (
                f"wall_tick_streak={stall_state['wall_tick_streak']}"
                f"@{wall_tick_s:.2f}s/tick"
            )
        elif (
            wall_elapsed >= float(wall_starve_s)
            and done_est < int(wall_starve_done)
            and t >= int(wall_starve_min_t)
            and sig["has_work"]
            and remaining > int(wall_starve_done)
        ):
            force_reason = (
                f"wall_starve wall={wall_elapsed:.1f}s done={done_est}"
                f"<{wall_starve_done}"
            )
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
    if old not in text:
        raise SystemExit("anchor _should_stop_gate block not found")
    text = text.replace(old, new, 1)

    old2 = '''            serial_recover_used += 1
            if joint_core_mode == "prioritized":
                joint_core_override = "ecbs"
            _force_wait_tick("no_progress")
            # Force serial unload: try all carriers one-by-one (hub jam needs >2).
            carriers = sorted(pending_delivery.keys())
'''
    new2 = '''            serial_recover_used += 1
            if joint_core_mode == "prioritized":
                joint_core_override = "ecbs"
            _force_wait_tick("no_progress")
            # Cap serial unload: at most 2 carriers/wave. Full-fleet serial
            # stacked makespan into 30k+ ticks on mid maps (SH02/SH03).
            carriers = sorted(pending_delivery.keys())[:2]
'''
    if old2 not in text:
        raise SystemExit("anchor serial_recover carriers not found")
    text = text.replace(old2, new2, 1)

    TARGET.write_text(text, encoding="utf-8")
    print(f"patched {TARGET.name} ok size={TARGET.stat().st_size}")


if __name__ == "__main__":
    main()
