"""Fix ECBS makespan blow-up after M0 handoff.

1) Force hybrid (not wave-astar) while handoff backlog remains.
2) Delivery ST fail -> defer+retry (no per-wave serial_move_to).
3) serial_recover unload at most 1 carrier.
"""
from __future__ import annotations

from pathlib import Path

SOLVE = Path(__file__).resolve().parent / "solve_ecbs.py"


def main() -> None:
    text = SOLVE.read_text(encoding="utf-8")

    # --- sticky hybrid after deciding planner ---
    old_wp = '''            wave_planner = str(getattr(pre_gate, "planner", "ecbs") or "ecbs")
            if gate.scene_decision_log:
'''
    new_wp = '''            wave_planner = str(getattr(pre_gate, "planner", "ecbs") or "ecbs")
            # M0 handoff must NOT fall back to wave-astar: that path serializes
            # delivery into 50k+ sim ticks on mid maps (SH02).
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
    if "handoff_floor_hybrid" not in text:
        if old_wp not in text:
            raise SystemExit("wave_planner assign block not found")
        text = text.replace(old_wp, new_wp, 1)

    # --- handoff sticky at entry ---
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
    if "m0_handoff_sticky" not in text:
        if old_h not in text:
            raise SystemExit("handoff print block not found")
        text = text.replace(old_h, new_h, 1)

    # --- delivery ST fail: defer all ---
    old_pc = '''                        else:
                            # Serial at most 1 AGV; defer the rest (anti-makespan).
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
    new_pc = '''                        else:
                            # Defer all carriers; serial every wave stacked SH02 to 55k.
                            # Progress via disperse + next-wave joint / rare recover.
                            for _a in batch:
                                pending_delivery[_a] = deliverable[_a]
                                loaded[_a] = True
                                dest[_a] = str(deliverable[_a].get("destination") or "")
                                tid[_a] = str(deliverable[_a].get("task_id") or "")
                            need = []
                            print(
                                f"[ECBS] delivery batch-ST pose-conflict "
                                f"k={len(batch)} -> defer+retry @t={now}",
                                flush=True,
                            )
                    else:
                        for _a in batch:
                            pending_delivery[_a] = deliverable[_a]
                            loaded[_a] = True
                            dest[_a] = str(deliverable[_a].get("destination") or "")
                            tid[_a] = str(deliverable[_a].get("task_id") or "")
                        need = []
                        print(
                            f"[ECBS] delivery batch-ST fail k={len(batch)} "
                            f"jam={int(jam_hard)} -> defer+retry @t={now}",
                            flush=True,
                        )
'''
    if "-> defer+retry @t=" not in text:
        if old_pc not in text:
            raise SystemExit("serial<=1 delivery block not found")
        text = text.replace(old_pc, new_pc, 1)

    # serial_recover: at most 1 carrier
    old_sr = "            carriers = sorted(pending_delivery.keys())[:2]"
    new_sr = "            carriers = sorted(pending_delivery.keys())[:1]"
    if old_sr in text:
        text = text.replace(old_sr, new_sr, 1)
    elif new_sr not in text and "carriers = sorted(pending_delivery.keys())" in text:
        text = text.replace(
            "            carriers = sorted(pending_delivery.keys())\n",
            "            carriers = sorted(pending_delivery.keys())[:1]\n",
            1,
        )

    SOLVE.write_text(text, encoding="utf-8")
    assert "handoff_floor_hybrid" in text
    assert "defer+retry" in text
    assert "m0_handoff_sticky" in text
    print(f"patched ok size={SOLVE.stat().st_size}")


if __name__ == "__main__":
    main()
