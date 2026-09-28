"""Restore golden delivery-fail / disperse / gate behavior; keep corridor clear only."""
from __future__ import annotations

import ast
from pathlib import Path

SOLVE = Path(__file__).with_name("solve_ecbs.py")
GATE = Path(__file__).with_name("hierarchical_gate.py")


def must_replace(text: str, old: str, new: str, label: str) -> str:
    c = text.count(old)
    if c != 1:
        raise SystemExit(f"{label}: count={c}")
    print("ok", label)
    return text.replace(old, new, 1)


def main() -> None:
    text = SOLVE.read_text(encoding="utf-8")

    text = must_replace(
        text,
        '''        if not joint_applied:
            print(
                f"[ECBS] delivery joint fail/skip k={len(deliverable)} "
                f"unique={len(unique_movers)} -> per-agent",
                flush=True,
            )
            # Defer fail bump until ST parallel also fails (avoid shrink sticky).

        need = sorted(a for a in deliverable if loaded.get(a, False))
        if need and not joint_applied:
''',
        '''        if not joint_applied:
            print(
                f"[ECBS] delivery joint fail/skip k={len(deliverable)} "
                f"unique={len(unique_movers)} -> per-agent",
                flush=True,
            )
            recent_joint_fail += 1

        need = sorted(a for a in deliverable if loaded.get(a, False))
        if need and not joint_applied:
''',
        "restore_early_fail_bump",
    )

    text = must_replace(
        text,
        '''                        joint_applied = True
                        recent_joint_fail = max(0, int(recent_joint_fail) - 1)
                        print(
                            f"[ECBS] delivery ST parallel k={len(movers_b)}",
                            flush=True,
                            )
            if not joint_applied:
                recent_joint_fail += 1
            need = sorted(a for a in deliverable if loaded.get(a, False))

        if need:
''',
        '''                        joint_applied = True
                        print(
                            f"[ECBS] delivery ST parallel k={len(movers_b)}",
                            flush=True,
                            )
            need = sorted(a for a in deliverable if loaded.get(a, False))

        if need:
''',
        "restore_st_no_double_bump",
    )

    text = must_replace(
        text,
        '''            if need:
                # Before serial: clear blockers (k<=2 / any fail → aggressive).
                if last_mile or int(recent_joint_fail) >= 1 or len(need) <= 2:
                    keep_s: Set[Cell] = set()
                    for a in need:
                        keep_s.add(pose[a][:2])
                        keep_s.add(goals.get(a, pose[a][:2]))
                        keep_s.update(
                            _valid_unload_pads(deliverable[a], free, static)[:6]
                        )
                    now = _disperse_idle_agents(
                        pose,
                        steps_by,
                        now,
                        movers=set(need),
                        free=free,
                        blocked_plan=blocked_plan,
                        keep_clear=keep_s,
                        loaded={n: bool(loaded.get(n)) for n in names},
                        dest=dict(dest),
                        tid=dict(tid),
                        aggressive=True,
                    )
''',
        '''            if need:
                # Before serial: clear blockers around the single (or few) targets.
                if last_mile or int(recent_joint_fail) >= 3:
                    keep_s: Set[Cell] = set()
                    for a in need:
                        keep_s.add(pose[a][:2])
                        keep_s.update(
                            _valid_unload_pads(deliverable[a], free, static)[:6]
                        )
                    now = _disperse_idle_agents(
                        pose,
                        steps_by,
                        now,
                        movers=set(need),
                        free=free,
                        blocked_plan=blocked_plan,
                        keep_clear=keep_s,
                        loaded={n: bool(loaded.get(n)) for n in names},
                        dest=dict(dest),
                        tid=dict(tid),
                    )
''',
        "restore_serial_disperse",
    )

    assert "_manhattan_corridor" in text  # keep corridor clear
    ast.parse(text)
    SOLVE.write_text(text, encoding="utf-8")
    print("solve ok", SOLVE.stat().st_size)

    g = GATE.read_text(encoding="utf-8")
    g = must_replace(
        g,
        '''            elif joint_fail_n >= 2:
                k = suggested_wave_k(
                    float(runtime_h), self.max_active, hard_cap=self.hard_cap
                )
                # Shrink only after sustained fails (single miss often recovered
                # by ST parallel). Never below 2 on hard.
                floor = 2 if committed == "hard" else 1
''',
        '''            elif joint_fail_n > 0:
                k = suggested_wave_k(
                    float(runtime_h), self.max_active, hard_cap=self.hard_cap
                )
                # Each consecutive joint fail peels one more slot, but never below 2
                # on hard (k=1 killed joint ability and livelocked deliveries).
                floor = 2 if committed == "hard" else 1
''',
        "restore_gate_shrink",
    )
    ast.parse(g)
    GATE.write_text(g, encoding="utf-8")
    print("gate ok", GATE.stat().st_size)


if __name__ == "__main__":
    main()
