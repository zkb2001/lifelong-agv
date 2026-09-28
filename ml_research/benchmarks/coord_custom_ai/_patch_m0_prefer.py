"""Revert defer_all; keep M0 on mid maps; serial<=1 on ST fail."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOLVE = ROOT / "solve_ecbs.py"
BATCH = ROOT / "batch_400_switch_20.py"


def patch_solve() -> None:
    text = SOLVE.read_text(encoding="utf-8")
    old = '''                        else:
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
    new = '''                        else:
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
    if old not in text:
        raise SystemExit("defer_all block not found")
    text = text.replace(old, new, 1)
    SOLVE.write_text(text, encoding="utf-8")
    print("solve ok")


def patch_batch() -> None:
    text = BATCH.read_text(encoding="utf-8")
    # Prefer staying on M0: disable wall_window / wall_starve false positives.
    # Escape only via hard_livelock / plateau / assign_fail after cold.
    reps = {
        'meta["m0_wall_starve_s"] = 300.0': 'meta["m0_wall_starve_s"] = 1e9',
        'meta["m0_wall_window_s"] = 90.0': 'meta["m0_wall_window_s"] = 1e9',
        'meta["m0_wall_tick_s"] = 3.0': 'meta["m0_wall_tick_s"] = 1e9',
        'meta["m0_min_handoff_t"] = 120': 'meta["m0_min_handoff_t"] = 150',
        'meta["m0_min_handoff_t"] = 100': 'meta["m0_min_handoff_t"] = 150',
        'os.environ["AGV_ASTAR_WALL_S"] = "0.45"': 'os.environ["AGV_ASTAR_WALL_S"] = "0.60"',
    }
    for a, b in reps.items():
        if a in text:
            text = text.replace(a, b, 1)
    # Keep joint prioritized for rare handoff.
    if 'joint = "prioritized"' not in text:
        text = text.replace('joint = "ecbs"', 'joint = "prioritized"', 1)
    BATCH.write_text(text, encoding="utf-8")
    print("batch ok")
    for line in BATCH.read_text(encoding="utf-8").splitlines():
        if any(k in line for k in ("ASTAR_WALL", "wall_starve", "wall_window", "wall_tick", "min_handoff", "joint =")):
            print(line.strip())


if __name__ == "__main__":
    patch_solve()
    patch_batch()
