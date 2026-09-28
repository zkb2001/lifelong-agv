"""P0/P1/P2: SH10/11 single-AGV / k=1 efficiency without breaking VALID."""
from __future__ import annotations

import ast
from pathlib import Path

TARGET = Path(__file__).with_name("solve_ecbs.py")
GATE = Path(__file__).resolve().parents[0] / "hierarchical_gate.py"


def must_replace(text: str, old: str, new: str, label: str) -> str:
    c = text.count(old)
    if c != 1:
        raise SystemExit(f"{label}: count={c}")
    print("ok", label)
    return text.replace(old, new, 1)


def patch_solve() -> None:
    text = TARGET.read_text(encoding="utf-8")

    text = must_replace(
        text,
        '''def _dest_inflight_cap(
    task: dict,
    free: Set[Cell],
    static: Set[Cell],
    n_agvs: int,
) -> int:
    """Per-destination concurrency = walkable unload pads (obstacle-free).

    Falls back to 1 if geometry yields no pad (avoid hard deadlock).
    ``n_agvs`` kept for API compat; pad count is the real ceiling.
    """
    del n_agvs
    pads = _valid_unload_pads(task, free, static)
    return max(1, len(pads) if pads else 1)


def _count_dest_inflight(
''',
        '''def _dest_inflight_cap(
    task: dict,
    free: Set[Cell],
    static: Set[Cell],
    n_agvs: int,
) -> int:
    """Hard unload concurrency = walkable unload pads (obstacle-free).

    Falls back to 1 if geometry yields no pad (avoid hard deadlock).
    Used for unload uniqueness; claim uses ``_dest_approach_cap``.
    """
    del n_agvs
    pads = _valid_unload_pads(task, free, static)
    return max(1, len(pads) if pads else 1)


def _dest_approach_cap(
    task: dict,
    free: Set[Cell],
    static: Set[Cell],
    n_agvs: int,
) -> int:
    """Soft claim concurrency per destination (hub anti-starvation).

    Maps often have 1 unload pad (e.g. Tianjin), so pad-cap alone forces
    wave=k=1 during hub windows. Allow several AGVs to approach / carry
    toward the same dest; ST / unique-pad logic still serializes unload.
    """
    pads = _valid_unload_pads(task, free, static)
    n_pads = len(pads) if pads else 1
    soft = max(2, min(4, max(2, int(n_agvs) // 3)))
    return max(int(n_pads), int(soft))


def _count_dest_inflight(
''',
        "approach_cap_fn",
    )

    text = must_replace(
        text,
        '''    for d, task in dest_task.items():
        pads = _valid_unload_pads(task, free, static)
        n_pads = len(pads) if pads else 1
        inflight = _count_dest_inflight(d, *bags)
        room = max(0, int(n_pads) - int(inflight))
        detail[d] = {
            "pads": int(n_pads),
            "inflight": int(inflight),
            "room": int(room),
        }
        total_room += int(room)
''',
        '''    for d, task in dest_task.items():
        pads = _valid_unload_pads(task, free, static)
        n_pads = len(pads) if pads else 1
        approach = _dest_approach_cap(task, free, static, max(1, int(n_free_agvs)))
        inflight = _count_dest_inflight(d, *bags)
        room = max(0, int(approach) - int(inflight))
        detail[d] = {
            "pads": int(n_pads),
            "approach_cap": int(approach),
            "inflight": int(inflight),
            "room": int(room),
        }
        total_room += int(room)
''',
        "claim_budget_approach",
    )

    text = must_replace(
        text,
        '''                for st, task in heads:
                    dest_name = str(task.get("destination") or "")
                    if dest_name:
                        cap = _dest_inflight_cap(task, free, static, len(names))
                        inflight = _count_dest_inflight(
                            dest_name, assigned, pending_delivery
                        )
                        if inflight >= cap:
                            continue
''',
        '''                for st, task in heads:
                    dest_name = str(task.get("destination") or "")
                    if dest_name:
                        # Soft approach cap (not hard pad=1) so hub heads
                        # across stations can share a wave.
                        cap = _dest_approach_cap(task, free, static, len(names))
                        inflight = _count_dest_inflight(
                            dest_name, assigned, pending_delivery
                        )
                        if inflight >= cap:
                            continue
''',
        "claim_loop_approach",
    )

    text = must_replace(
        text,
        '''            else:
                # ECBS window: round-robin among claimable surface heads
                task = None
                st = None
                for k in range(len(order)):
                    cand = order[(rr + k) % len(order)]
                    if queues.get(cand) and not (inflight_station.get(cand) or []):
                        st, task = cand, queues[cand][0]
                        rr = (rr + k + 1) % max(1, len(order))
                        break
''',
        '''            else:
                # ECBS window: round-robin among claimable surface heads
                task = None
                st = None
                for k in range(len(order)):
                    cand = order[(rr + k) % len(order)]
                    if not (queues.get(cand) and not (inflight_station.get(cand) or [])):
                        continue
                    t_cand = queues[cand][0]
                    d_cand = str(t_cand.get("destination") or "")
                    if d_cand:
                        cap_c = _dest_approach_cap(t_cand, free, static, len(names))
                        if _count_dest_inflight(d_cand, assigned, pending_delivery) >= cap_c:
                            continue
                    st, task = cand, t_cand
                    rr = (rr + k + 1) % max(1, len(order))
                    break
''',
        "ecbs_rr_approach",
    )

    text = must_replace(
        text,
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
        "defer_joint_fail_bump",
    )

    text = must_replace(
        text,
        '''                        joint_applied = True
                        print(
                            f"[ECBS] delivery ST parallel k={len(movers_b)}",
                            flush=True,
                            )
            need = sorted(a for a in deliverable if loaded.get(a, False))

        if need:
''',
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
        "st_parallel_fail_bump",
    )

    text = must_replace(
        text,
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
        "serial_aggressive_disperse",
    )

    ast.parse(text)
    TARGET.write_text(text, encoding="utf-8")
    print("solve_ecbs AST ok", TARGET.stat().st_size)


def patch_gate() -> None:
    text = GATE.read_text(encoding="utf-8")
    text = must_replace(
        text,
        '''            elif joint_fail_n > 0:
                k = suggested_wave_k(
                    float(runtime_h), self.max_active, hard_cap=self.hard_cap
                )
                # Each consecutive joint fail peels one more slot, but never below 2
                # on hard (k=1 killed joint ability and livelocked deliveries).
                floor = 2 if committed == "hard" else 1
''',
        '''            elif joint_fail_n >= 2:
                k = suggested_wave_k(
                    float(runtime_h), self.max_active, hard_cap=self.hard_cap
                )
                # Shrink only after sustained fails (single miss often recovered
                # by ST parallel). Never below 2 on hard.
                floor = 2 if committed == "hard" else 1
''',
        "gate_shrink_threshold",
    )
    # Peel uses joint_fail_n - 1; with threshold >=2 this still peels from 1 when n=2.
    ast.parse(text)
    GATE.write_text(text, encoding="utf-8")
    print("gate AST ok", GATE.stat().st_size)


def main() -> None:
    patch_solve()
    patch_gate()
    t = TARGET.read_text(encoding="utf-8")
    assert "_dest_approach_cap" in t
    assert "aggressive=True" in t
    print("all patches applied")


if __name__ == "__main__":
    main()
