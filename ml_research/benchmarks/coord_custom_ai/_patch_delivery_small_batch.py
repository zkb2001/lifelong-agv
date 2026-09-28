"""Delivery: small parallel batches + serial-1 on fail (not defer-all livelock)."""
from __future__ import annotations

from pathlib import Path

SOLVE = Path(__file__).resolve().parent / "solve_ecbs.py"


def main() -> None:
    lines = SOLVE.read_text(encoding="utf-8").splitlines(keepends=True)
    text = "".join(lines)

    # Replace jam_hard batch sizing via line scan
    start = None
    end = None
    for i, l in enumerate(lines):
        if "jam_hard = (" in l and start is None:
            # include from jam_hard through batch = list(ranked)
            start = i
        if start is not None and "batch = list(ranked)" in l:
            end = i + 1
            break
    if start is None or end is None:
        raise SystemExit(f"jam batch bounds {start},{end}")

    new_batch = '''                    jam_hard = (
                        int(recent_joint_fail) >= 6
                        or int(no_progress_waves) >= 2
                    )
                    # Always small concurrent unload. Full-fleet ST fails then
                    # either serial-stacks (55k) or defer-livelocks (carry=12).
                    _cap = min(
                        3,
                        int(wave_hard_cap) if int(wave_hard_cap) > 0 else 3,
                        len(ranked),
                    )
                    _bn = 1 if jam_hard else max(1, _cap)
                    batch = ranked[:_bn]
'''
    lines = lines[:start] + [new_batch] + lines[end:]
    text = "".join(lines)

    # Replace defer+retry back to serial<=1
    old_pc = '''                        else:
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
    new_pc = '''                        else:
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
    if old_pc not in text:
        raise SystemExit("defer+retry block not found")
    text = text.replace(old_pc, new_pc, 1)

    # Limit new pickup claims while many carriers (avoid fill-fleet deadlocks).
    # Find free_agvs filter near claim.
    claim_anchor = "        free_agvs = [\n            n\n"
    if "carrier_cap_skip" not in text and claim_anchor in text:
        # Insert after free_agvs list is built - find the closing of free_agvs
        pass

    SOLVE.write_text(text, encoding="utf-8")
    assert "Always small concurrent unload" in text
    assert "-> serial<=1 @t=" in text
    print(f"ok size={SOLVE.stat().st_size}")


if __name__ == "__main__":
    main()
