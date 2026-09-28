"""Generalizable ECBS/M0 fixes for SH01-20 (signal-based, no slot special-case).

Restored solve_ecbs.py from backup; apply:
1) Wall-clock / done-cost escalate out of slow M0 -> wave
2) After M0 handoff: sticky hybrid floor (no fall-back to wave-astar serial)
3) Delivery: concurrent unload <=3; ST fail -> serial <=1
4) Pickup serial / serial_recover <=1
5) Cap new pickup claims while fleet saturated with carriers
"""
from __future__ import annotations

from pathlib import Path

SOLVE = Path(__file__).resolve().parent / "solve_ecbs.py"


def main() -> None:
    text = SOLVE.read_text(encoding="utf-8")
    if "GENERALIZABLE_SWITCH_V1" in text:
        print("already patched")
        return

    # --- 1) Expand stall_state + wall params before _should_stop_gate ---
    old_stall = '''    escalate_hit = {"label": "", "t": -1, "reason": ""}
    stall_state = {
        "idle_surface_streak": 0,
        "done_plateau_streak": 0,
        "prev_done": None,
        "prev_t": None,
        "throughput_ema": None,
        "throughput_drop_streak": 0,
        "done_at_window_start": None,
        "t_at_window_start": None,
    }
    idle_need = int(meta.get("m0_idle_surface_streak") or 2)
    plateau_need = int(meta.get("m0_done_plateau_streak") or 3)
    # Default 8: more sensitive than legacy 16 (SwapNet absorbs mild noise first).
    assign_fail_need = int(meta.get("m0_assign_fail_handoff") or 8)
    plan_fail_rate_need = float(meta.get("m0_plan_fail_rate") or 0.4)
    plan_fail_samples_need = int(meta.get("m0_plan_fail_samples") or 10)
    throughput_ratio_need = float(meta.get("m0_throughput_ratio") or 0.4)
    throughput_drop_need = int(meta.get("m0_throughput_drop_streak") or 2)
    # Guard against cold-start / single noisy gate window handoffs.
    min_handoff_t = int(meta.get("m0_min_handoff_t") or 80)
    min_handoff_done = int(meta.get("m0_min_handoff_done") or 0)
'''
    new_stall = '''    escalate_hit = {"label": "", "t": -1, "reason": ""}
    # GENERALIZABLE_SWITCH_V1
    import time as _time
    stall_state = {
        "idle_surface_streak": 0,
        "done_plateau_streak": 0,
        "prev_done": None,
        "prev_t": None,
        "throughput_ema": None,
        "throughput_drop_streak": 0,
        "done_at_window_start": None,
        "t_at_window_start": None,
        "wall_at_gate": None,
        "wall_tick_streak": 0,
        "win_wall0": None,
        "win_t0": None,
        "win_done0": None,
        "wall_window_hit": False,
        "wall_window_dsim": 0,
        "wall_window_ddone": 0,
    }
    idle_need = int(meta.get("m0_idle_surface_streak") or 2)
    plateau_need = int(meta.get("m0_done_plateau_streak") or 3)
    assign_fail_need = int(meta.get("m0_assign_fail_handoff") or 8)
    plan_fail_rate_need = float(meta.get("m0_plan_fail_rate") or 0.4)
    plan_fail_samples_need = int(meta.get("m0_plan_fail_samples") or 10)
    throughput_ratio_need = float(meta.get("m0_throughput_ratio") or 0.4)
    throughput_drop_need = int(meta.get("m0_throughput_drop_streak") or 2)
    min_handoff_t = int(meta.get("m0_min_handoff_t") or 80)
    min_handoff_done = int(meta.get("m0_min_handoff_done") or 0)
    min_handoff_wall_s = float(meta.get("m0_min_handoff_wall_s") or 45.0)
    wall_tick_s = float(meta.get("m0_wall_tick_s") or 1.0)
    wall_tick_streak_need = int(meta.get("m0_wall_tick_streak") or 2)
    wall_starve_s = float(meta.get("m0_wall_starve_s") or 90.0)
    wall_starve_done = int(meta.get("m0_wall_starve_done") or 15)
    wall_starve_min_t = int(meta.get("m0_wall_starve_min_t") or 40)
    wall_window_s = float(meta.get("m0_wall_window_s") or 45.0)
    wall_window_min_sim = int(meta.get("m0_wall_window_min_sim") or 20)
    wall_window_min_done = int(meta.get("m0_wall_window_min_done") or 2)
    wall_done_cost_s = float(meta.get("m0_wall_done_cost_s") or 8.0)
    wall_done_cost_min_done = int(meta.get("m0_wall_done_cost_min_done") or 6)
    m0_loop_t0 = _time.perf_counter()
'''
    if old_stall not in text:
        raise SystemExit("stall_state block not found")
    text = text.replace(old_stall, new_stall, 1)

    # --- replace _should_stop_gate body force logic ---
    old_gate = '''    def _should_stop_gate(sim_live) -> bool:
        if not allow_yield:
            return False
        t = int(getattr(sim_live, "time", 0) or 0)
        if t <= 0 or (gate_every > 0 and t % max(1, gate_every) != 0):
            return False

        sig = _m0_stall_signals(sim_live, n_tasks=int(n_tasks))
        if sig["idle_all"] and sig["has_work"]:
            stall_state["idle_surface_streak"] = (
                int(stall_state["idle_surface_streak"]) + 1
            )
        else:
            stall_state["idle_surface_streak"] = 0
        prev_done = stall_state["prev_done"]
        done_est = int(sig["done_est"])
        if prev_done is not None and done_est <= int(prev_done) and sig["has_work"]:
            stall_state["done_plateau_streak"] = (
                int(stall_state["done_plateau_streak"]) + 1
            )
        else:
            stall_state["done_plateau_streak"] = 0

        # Rolling throughput (completions per gate window) vs EMA baseline.
        prev_t = stall_state["prev_t"]
        if prev_done is not None and prev_t is not None and t > int(prev_t):
            dt = max(1, t - int(prev_t))
            d_done = max(0, done_est - int(prev_done))
            rate = float(d_done) / float(dt)
            ema = stall_state["throughput_ema"]
            if ema is None:
                stall_state["throughput_ema"] = rate
            else:
                stall_state["throughput_ema"] = 0.8 * float(ema) + 0.2 * rate
                base = float(stall_state["throughput_ema"])
                if (
                    sig["has_work"]
                    and base > 1e-6
                    and rate <= throughput_ratio_need * base
                ):
                    stall_state["throughput_drop_streak"] = (
                        int(stall_state["throughput_drop_streak"]) + 1
                    )
                else:
                    stall_state["throughput_drop_streak"] = 0
        stall_state["prev_done"] = done_est
        stall_state["prev_t"] = t

        # Hotspot only discounts thresholds — never escalates alone.
        # Do NOT discount throughput/assign below 2 windows (single-gate noise).
        hotspot_discount = 1.0
        try:
            drop_s = float(getattr(gate, "last", None) and gate.last.dropoff_score or 0.0)
            thr = float(getattr(gate, "dropoff_escalate_score", 0.45) or 0.45)
            if drop_s >= thr:
                hotspot_discount = 0.75
        except Exception:  # noqa: BLE001
            hotspot_discount = 1.0

        def _need(v: float, *, floor: int = 1) -> int:
            return max(int(floor), int(round(float(v) * hotspot_discount)))

        force_reason = ""
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
        elif (
            # Do not hotspot-discount assign_fail — too easy to trip mid-M0.
            int(sig["assign_fail_streak"]) >= int(assign_fail_need)
            and sig["has_work"]
            and not cold
        ):
            force_reason = f"assign_fail_streak={sig['assign_fail_streak']}"
        elif (
            float(sig.get("plan_fail_rate") or 0) >= plan_fail_rate_need
            and int(sig.get("plan_fail_samples") or 0) >= plan_fail_samples_need
            and sig["has_work"]
            and not cold
        ):
            force_reason = (
                f"plan_fail_rate={float(sig['plan_fail_rate']):.2f}"
                f"@{sig['plan_fail_samples']}"
            )
        elif (
            int(stall_state["throughput_drop_streak"])
            >= _need(throughput_drop_need, floor=2)
            and sig["has_work"]
            and not cold
        ):
            force_reason = (
                f"throughput_drop_streak={stall_state['throughput_drop_streak']}"
            )

        if force_reason:
            try:
                label = gate.force_commit_at_least(
                    "medium", reason=f"m0_degrade:{force_reason}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[M0] WARN force_commit failed: {exc}", flush=True)
                label = "medium"
            escalate_hit["label"] = str(label or "medium")
            escalate_hit["t"] = t
            escalate_hit["reason"] = force_reason
            print(
                f"[HIER] M0 degrade escalate -> {escalate_hit['label']} at t={t} "
                f"reason={force_reason} hotspot_discount={hotspot_discount}",
                flush=True,
            )
            return True
'''
    new_gate = '''    def _should_stop_gate(sim_live) -> bool:
        if not allow_yield:
            return False
        t = int(getattr(sim_live, "time", 0) or 0)
        if t <= 0 or (gate_every > 0 and t % max(1, gate_every) != 0):
            return False

        wall_elapsed = float(_time.perf_counter() - m0_loop_t0)
        sig = _m0_stall_signals(sim_live, n_tasks=int(n_tasks))
        if sig["idle_all"] and sig["has_work"]:
            stall_state["idle_surface_streak"] = (
                int(stall_state["idle_surface_streak"]) + 1
            )
        else:
            stall_state["idle_surface_streak"] = 0
        prev_done = stall_state["prev_done"]
        done_est = int(sig["done_est"])
        if prev_done is not None and done_est <= int(prev_done) and sig["has_work"]:
            stall_state["done_plateau_streak"] = (
                int(stall_state["done_plateau_streak"]) + 1
            )
        else:
            stall_state["done_plateau_streak"] = 0

        prev_t = stall_state["prev_t"]
        if prev_done is not None and prev_t is not None and t > int(prev_t):
            dt = max(1, t - int(prev_t))
            d_done = max(0, done_est - int(prev_done))
            rate = float(d_done) / float(dt)
            ema = stall_state["throughput_ema"]
            if ema is None:
                stall_state["throughput_ema"] = rate
            else:
                stall_state["throughput_ema"] = 0.8 * float(ema) + 0.2 * rate
                base = float(stall_state["throughput_ema"])
                if (
                    sig["has_work"]
                    and base > 1e-6
                    and rate <= throughput_ratio_need * base
                ):
                    stall_state["throughput_drop_streak"] = (
                        int(stall_state["throughput_drop_streak"]) + 1
                    )
                else:
                    stall_state["throughput_drop_streak"] = 0
            prev_wall = stall_state["wall_at_gate"]
            if prev_wall is not None:
                wpt = float(wall_elapsed - float(prev_wall)) / float(dt)
                if wpt >= float(wall_tick_s) and sig["has_work"]:
                    stall_state["wall_tick_streak"] = (
                        int(stall_state["wall_tick_streak"]) + 1
                    )
                else:
                    stall_state["wall_tick_streak"] = 0
        stall_state["prev_done"] = done_est
        stall_state["prev_t"] = t
        stall_state["wall_at_gate"] = wall_elapsed

        stall_state["wall_window_hit"] = False
        if stall_state["win_wall0"] is None:
            stall_state["win_wall0"] = wall_elapsed
            stall_state["win_t0"] = t
            stall_state["win_done0"] = done_est
        else:
            d_wall = float(wall_elapsed) - float(stall_state["win_wall0"])
            if d_wall >= float(wall_window_s):
                d_sim = int(t) - int(stall_state["win_t0"] or 0)
                d_done = int(done_est) - int(stall_state["win_done0"] or 0)
                stall_state["wall_window_dsim"] = int(d_sim)
                stall_state["wall_window_ddone"] = int(d_done)
                if d_sim < int(wall_window_min_sim) or d_done < int(
                    wall_window_min_done
                ):
                    stall_state["wall_window_hit"] = True
                stall_state["win_wall0"] = wall_elapsed
                stall_state["win_t0"] = t
                stall_state["win_done0"] = done_est

        hotspot_discount = 1.0
        try:
            drop_s = float(getattr(gate, "last", None) and gate.last.dropoff_score or 0.0)
            thr = float(getattr(gate, "dropoff_escalate_score", 0.45) or 0.45)
            if drop_s >= thr:
                hotspot_discount = 0.75
        except Exception:  # noqa: BLE001
            hotspot_discount = 1.0

        def _need(v: float, *, floor: int = 1) -> int:
            return max(int(floor), int(round(float(v) * hotspot_discount)))

        force_reason = ""
        cold = (t < int(min_handoff_t) and wall_elapsed < float(min_handoff_wall_s)) or (
            done_est < int(min_handoff_done)
        )
        remaining = max(0, int(n_tasks) - int(done_est))
        near_done = remaining <= max(2, int(round(0.01 * float(n_tasks))))
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
                f"surface={sig.get('n_surface')}"
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
            and (
                int(stall_state["done_plateau_streak"]) >= 2
                or int(stall_state["idle_surface_streak"]) >= 1
            )
        ):
            force_reason = (
                f"wall_starve wall={wall_elapsed:.1f}s done={done_est}"
                f"<{wall_starve_done}"
            )
        elif (
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
        elif (
            int(stall_state["done_plateau_streak"]) >= _need(plateau_need, floor=2)
            and sig["has_work"]
            and not cold
        ):
            force_reason = (
                f"done_plateau_streak={stall_state['done_plateau_streak']}"
            )
        elif (
            int(sig["assign_fail_streak"]) >= int(assign_fail_need)
            and sig["has_work"]
            and not cold
        ):
            force_reason = f"assign_fail_streak={sig['assign_fail_streak']}"
        elif (
            float(sig.get("plan_fail_rate") or 0) >= plan_fail_rate_need
            and int(sig.get("plan_fail_samples") or 0) >= plan_fail_samples_need
            and sig["has_work"]
            and not cold
        ):
            force_reason = (
                f"plan_fail_rate={float(sig['plan_fail_rate']):.2f}"
                f"@{sig['plan_fail_samples']}"
            )
        elif (
            int(stall_state["throughput_drop_streak"])
            >= _need(throughput_drop_need, floor=2)
            and sig["has_work"]
            and not cold
        ):
            force_reason = (
                f"throughput_drop_streak={stall_state['throughput_drop_streak']}"
            )

        if force_reason:
            try:
                label = gate.force_commit_at_least(
                    "medium", reason=f"m0_degrade:{force_reason}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[M0] WARN force_commit failed: {exc}", flush=True)
                label = "medium"
            escalate_hit["label"] = str(label or "medium")
            escalate_hit["t"] = t
            escalate_hit["reason"] = force_reason
            print(
                f"[HIER] M0 degrade escalate -> {escalate_hit['label']} at t={t} "
                f"wall={wall_elapsed:.1f}s reason={force_reason} "
                f"hotspot_discount={hotspot_discount}",
                flush=True,
            )
            return True
'''
    if old_gate not in text:
        raise SystemExit("_should_stop_gate block not found")
    text = text.replace(old_gate, new_gate, 1)

    # --- 2) sticky hybrid after handoff print ---
    old_h = '''            print(
                f"[HIER] M0 handoff -> wave loop label={handoff_label} t={now} "
                f"done={done}/{total} q={sum(len(v) for v in queues.values())} "
                f"delivery={len(pending_delivery)} pickup={len(pending_pickup)} "
                f"joint_core={joint_core_mode}",
                flush=True,
            )
            # Skip cold-start [ECBS] banner; continue into lifelong wave loop below.
'''
    new_h = '''            print(
                f"[HIER] M0 handoff -> wave loop label={handoff_label} t={now} "
                f"done={done}/{total} q={sum(len(v) for v in queues.values())} "
                f"delivery={len(pending_delivery)} pickup={len(pending_pickup)} "
                f"joint_core={joint_core_mode}",
                flush=True,
            )
            if gate is not None:
                try:
                    want = (
                        handoff_label
                        if handoff_label in ("medium", "hard")
                        else "medium"
                    )
                    gate.force_commit_at_least(
                        want, reason="m0_handoff_sticky", sim_t=int(now)
                    )
                    gate.calm_waves_to_release = max(
                        int(getattr(gate, "calm_waves_to_release", 5) or 5), 40
                    )
                    gate.scene_calm_streak = 0
                    print(
                        f"[HIER] handoff sticky band={gate.committed_scene} "
                        f"calm_need={gate.calm_waves_to_release}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[HIER] WARN sticky commit failed: {exc}", flush=True)
            # Skip cold-start [ECBS] banner; continue into lifelong wave loop below.
'''
    if old_h not in text:
        raise SystemExit("handoff print not found")
    text = text.replace(old_h, new_h, 1)

    # --- 3) floor hybrid while handoff backlog ---
    old_wp = '''            wave_planner = str(getattr(pre_gate, "planner", "ecbs") or "ecbs")
            if gate.scene_decision_log:
'''
    new_wp = '''            wave_planner = str(getattr(pre_gate, "planner", "ecbs") or "ecbs")
            # Signal-based: after M0 yield, keep hybrid/ECBS until backlog small.
            # Falling to wave-astar serializes delivery (sim_t tens of thousands).
            if m0_handoff_active and (int(total) - int(done)) > max(
                8, int(0.05 * float(total))
            ):
                if wave_planner == "astar":
                    wave_planner = "hybrid"
                try:
                    gate.force_commit_at_least(
                        "medium", reason="handoff_floor_hybrid", sim_t=int(now)
                    )
                except Exception:
                    pass
            if gate.scene_decision_log:
'''
    if old_wp not in text:
        raise SystemExit("wave_planner assign not found")
    text = text.replace(old_wp, new_wp, 1)

    # --- 4) delivery small batch + serial<=1 ---
    old_jam = '''                    jam_hard = (
                        int(recent_joint_fail) >= 6
                        or int(no_progress_waves) >= 2
                    )
                    batch = ranked[:1] if jam_hard else list(ranked)
                    for agv in ranked[len(batch) :]:
'''
    new_jam = '''                    jam_hard = (
                        int(recent_joint_fail) >= 6
                        or int(no_progress_waves) >= 2
                    )
                    # General: small concurrent unload (all maps). Full-fleet ST
                    # fails then serial-stacks makespan.
                    _cap = min(
                        3,
                        int(wave_hard_cap) if int(wave_hard_cap) > 0 else 3,
                        len(ranked),
                    )
                    _bn = 1 if jam_hard else max(1, _cap)
                    batch = ranked[:_bn]
                    for agv in ranked[len(batch) :]:
'''
    if old_jam not in text:
        raise SystemExit("jam_hard batch not found")
    text = text.replace(old_jam, new_jam, 1)

    old_ser = '''                        else:
                            need = batch
                            print(
                                f"[ECBS] delivery batch-ST pose-conflict "
                                f"k={len(batch)} -> serial @t={now}",
                                flush=True,
                            )
                    else:
                        need = batch
                        print(
                            f"[ECBS] delivery batch-ST fail k={len(batch)} "
                            f"jam={int(jam_hard)} -> serial @t={now}",
                            flush=True,
                        )
'''
    new_ser = '''                        else:
                            need = list(batch[:1])
                            for _a in batch[1:]:
                                pending_delivery[_a] = deliverable[_a]
                                loaded[_a] = True
                                dest[_a] = str(deliverable[_a].get("destination") or "")
                                tid[_a] = str(deliverable[_a].get("task_id") or "")
                            print(
                                f"[ECBS] delivery batch-ST pose-conflict "
                                f"k={len(batch)} -> serial<=1 @t={now}",
                                flush=True,
                            )
                    else:
                        need = list(batch[:1])
                        for _a in batch[1:]:
                            pending_delivery[_a] = deliverable[_a]
                            loaded[_a] = True
                            dest[_a] = str(deliverable[_a].get("destination") or "")
                            tid[_a] = str(deliverable[_a].get("task_id") or "")
                        print(
                            f"[ECBS] delivery batch-ST fail k={len(batch)} "
                            f"jam={int(jam_hard)} -> serial<=1 @t={now}",
                            flush=True,
                        )
'''
    if old_ser not in text:
        raise SystemExit("delivery serial fallback not found")
    text = text.replace(old_ser, new_ser, 1)

    # --- 5) serial_recover carriers[:1] ---
    if "carriers = sorted(pending_delivery.keys())[:1]" not in text:
        text = text.replace(
            "            carriers = sorted(pending_delivery.keys())\n",
            "            carriers = sorted(pending_delivery.keys())[:1]\n",
            1,
        )

    # --- 6) pickup keep_n = 1 for shrink+serial ---
    old_keep = (
        "_keep_n = max(2, min(int(wave_hard_cap) if int(wave_hard_cap) > 0 else 4, len(ranked)))"
    )
    if old_keep in text:
        text = text.replace(
            old_keep,
            "_keep_n = 1  # multi-serial stacks makespan on every map",
        )

    SOLVE.write_text(text, encoding="utf-8")
    assert "GENERALIZABLE_SWITCH_V1" in text
    assert "handoff_floor_hybrid" in text
    assert "serial<=1" in text
    print(f"patched {SOLVE.name} size={SOLVE.stat().st_size}")


if __name__ == "__main__":
    main()
