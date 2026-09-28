"""After M0 handoff: sticky medium + no per-wave serial (defer ST fails)."""
from __future__ import annotations

from pathlib import Path

SOLVE = Path(__file__).resolve().parent / "solve_ecbs.py"


def main() -> None:
    text = SOLVE.read_text(encoding="utf-8")

    # Sticky medium right after handoff print.
    anchor = '''            print(
                f"[HIER] M0 handoff -> wave loop label={handoff_label} t={now} "
                f"done={done}/{total} q={sum(len(v) for v in queues.values())} "
                f"delivery={len(pending_delivery)} pickup={len(pending_pickup)} "
                f"joint_core={joint_core_mode}",
                flush=True,
            )
            # Skip cold-start [ECBS] banner; continue into lifelong wave loop below.
'''
    sticky = '''            print(
                f"[HIER] M0 handoff -> wave loop label={handoff_label} t={now} "
                f"done={done}/{total} q={sum(len(v) for v in queues.values())} "
                f"delivery={len(pending_delivery)} pickup={len(pending_pickup)} "
                f"joint_core={joint_core_mode}",
                flush=True,
            )
            # Keep medium/hybrid sticky — gate.decide would flip back to easy/astar
            # and serial-stack makespan (SH02 hit sim_t=26k at done=190).
            if gate is not None:
                try:
                    want = handoff_label if handoff_label in ("medium", "hard") else "medium"
                    gate.force_commit_at_least(
                        want, reason="m0_handoff_sticky", sim_t=int(now)
                    )
                    gate.calm_waves_to_release = max(
                        int(getattr(gate, "calm_waves_to_release", 5) or 5), 30
                    )
                    print(
                        f"[HIER] handoff sticky band={gate.committed_scene} "
                        f"calm_need={gate.calm_waves_to_release}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[HIER] WARN sticky commit failed: {exc}", flush=True)
            # Skip cold-start [ECBS] banner; continue into lifelong wave loop below.
'''
    if sticky.split("handoff sticky")[0] not in text and "m0_handoff_sticky" not in text:
        if anchor not in text:
            raise SystemExit("handoff print anchor not found")
        text = text.replace(anchor, sticky, 1)

    # Re-assert sticky each wave while handoff active and not near done.
    # Find wave scene decide block - look for before-claim
    needle = "        if gate is not None and pre_gate is not None:"
    # Better: after pre_gate = gate.decide(...), force medium if handoff
    # Search for typical pattern
    if "handoff_sticky_reassert" not in text:
        # Find: easy_calm / label_now after decide in wave loop
        marker = "            if m0_handoff_active and gate is not None:"
        idx = text.find(marker)
        if idx < 0:
            raise SystemExit("m0_handoff_active calm marker not found")
        inject = '''            if m0_handoff_active and gate is not None:
                # Reassert medium while backlog large (prevent easy/astar serial).
                remain = max(0, int(total) - int(done))
                if remain > max(8, int(0.05 * float(total))):
                    try:
                        gate.force_commit_at_least(
                            "medium", reason="handoff_sticky_reassert", sim_t=int(now)
                        )
                    except Exception:
                        pass
            if m0_handoff_active and gate is not None:'''
        # The marker appears once for calm - replace first occurrence carefully
        text = text.replace(marker, inject, 1)

    # ST fail -> defer_all (no serial<=1 every wave)
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
    if old_pc not in text:
        if "-> serial<=1 @t=" in text:
            text = text.replace(
                "-> serial<=1 @t={now}",
                "-> defer+retry @t={now}",
            )
            # Also need to change need = list(batch[:1]) blocks - do full replace via lines
            raise SystemExit("serial<=1 block format changed; check manually")
        raise SystemExit("serial<=1 block not found")
    text = text.replace(old_pc, new_pc, 1)

    SOLVE.write_text(text, encoding="utf-8")
    print(f"ok size={SOLVE.stat().st_size}")


if __name__ == "__main__":
    main()
