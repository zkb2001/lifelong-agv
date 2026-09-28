"""Revert harmful approach_cap; keep disperse + shrink tweaks."""
from __future__ import annotations

import ast
from pathlib import Path

TARGET = Path(__file__).with_name("solve_ecbs.py")


def must_replace(text: str, old: str, new: str, label: str) -> str:
    c = text.count(old)
    if c != 1:
        raise SystemExit(f"{label}: count={c}")
    print("ok", label)
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    # Remove _dest_approach_cap, restore pad-only _dest_inflight_cap docstring
    text = must_replace(
        text,
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
        "revert_approach_fn",
    )

    text = must_replace(
        text,
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
        "revert_budget",
    )

    text = must_replace(
        text,
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
        "revert_claim_loop",
    )

    text = must_replace(
        text,
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
        "revert_ecbs_rr",
    )

    ast.parse(text)
    assert "_dest_approach_cap" not in text
    assert "aggressive=True" in text  # P1 kept
    TARGET.write_text(text, encoding="utf-8")
    print("reverted P0; kept P1/P2", TARGET.stat().st_size)


if __name__ == "__main__":
    main()
