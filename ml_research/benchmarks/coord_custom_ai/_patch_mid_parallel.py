"""Mid-map: wall_starve guard + defer ST-fail instead of serial<=2."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOLVE = ROOT / "solve_ecbs.py"
BATCH = ROOT / "batch_400_switch_20.py"


def patch_solve() -> None:
    text = SOLVE.read_text(encoding="utf-8")
    old = '''        elif (
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
'''
    new = '''        elif (
            wall_elapsed >= float(wall_starve_s)
            and done_est < int(wall_starve_done)
            and t >= int(wall_starve_min_t)
            and sig["has_work"]
            and remaining > int(wall_starve_done)
            # Require a real stall signal — bare low done_est at slow A*
            # wall clocks (SH02 t=25) was a false positive into ECBS+serial.
            and (
                int(stall_state["done_plateau_streak"]) >= 2
                or int(stall_state["idle_surface_streak"]) >= 1
            )
        ):
            force_reason = (
                f"wall_starve wall={wall_elapsed:.1f}s done={done_est}"
                f"<{wall_starve_done}"
            )
'''
    if old not in text:
        raise SystemExit("wall_starve block not found")
    text = text.replace(old, new, 1)

    old_pc = '''                        else:
                            need = list(batch[: max(1, min(2, len(batch)))])
                            for _a in batch[len(need):]:
                                pending_delivery[_a] = deliverable[_a]
                                loaded[_a] = True
                                dest[_a] = str(deliverable[_a].get("destination") or "")
                                tid[_a] = str(deliverable[_a].get("task_id") or "")
                            print(
                                f"[ECBS] delivery batch-ST pose-conflict "
                                f"k={len(batch)} -> serial<={len(need)} @t={now}",
                                flush=True,
                            )
                    else:
                        need = list(batch[: max(1, min(2, len(batch)))])
                        for _a in batch[len(need):]:
                            pending_delivery[_a] = deliverable[_a]
                            loaded[_a] = True
                            dest[_a] = str(deliverable[_a].get("destination") or "")
                            tid[_a] = str(deliverable[_a].get("task_id") or "")
                        print(
                            f"[ECBS] delivery batch-ST fail k={len(batch)} "
                            f"jam={int(jam_hard)} -> serial<={len(need)} @t={now}",
                            flush=True,
                        )
'''
    new_pc = '''                        else:
                            # Defer all: serial<=2 here stacked mid-map makespan
                            # to 20k+. Let serial_recover (cap 2) handle stuck.
                            for _a in batch:
                                pending_delivery[_a] = deliverable[_a]
                                loaded[_a] = True
                                dest[_a] = str(deliverable[_a].get("destination") or "")
                                tid[_a] = str(deliverable[_a].get("task_id") or "")
                            need = []
                            print(
                                f"[ECBS] delivery batch-ST pose-conflict "
                                f"k={len(batch)} -> defer_all @t={now}",
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
                            f"jam={int(jam_hard)} -> defer_all @t={now}",
                            flush=True,
                        )
'''
    if old_pc not in text:
        raise SystemExit("pose-conflict/fail serial block not found")
    text = text.replace(old_pc, new_pc, 1)

    SOLVE.write_text(text, encoding="utf-8")
    print(f"patched {SOLVE.name} size={SOLVE.stat().st_size}")


def patch_batch() -> None:
    text = BATCH.read_text(encoding="utf-8")
    if 'joint = "prioritized"' not in text:
        if 'joint = "ecbs"' not in text:
            raise SystemExit("joint assignment not found")
        text = text.replace('joint = "ecbs"', 'joint = "prioritized"', 1)
    repls = [
        ('meta["m0_wall_starve_s"] = 120.0', 'meta["m0_wall_starve_s"] = 300.0'),
        ('meta["m0_wall_starve_min_t"] = 20', 'meta["m0_wall_starve_min_t"] = 150'),
        ('meta["m0_wall_window_s"] = 60.0', 'meta["m0_wall_window_s"] = 90.0'),
    ]
    for a, b in repls:
        if a in text:
            text = text.replace(a, b, 1)
        elif b in text:
            pass
        else:
            raise SystemExit(f"batch missing {a}")
    BATCH.write_text(text, encoding="utf-8")
    print(f"patched {BATCH.name}")


if __name__ == "__main__":
    patch_solve()
    patch_batch()
