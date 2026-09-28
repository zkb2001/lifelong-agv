"""Turn-before-move on the VALID SH10 400 trajectory, then audit facing."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import (
    _pitch_from_delta,
    _plan_paths_as_poses,
    _turn_pitch_chain,
)
from ml_research.benchmarks.coord_custom_ai.validate_hybrid_trajectory import (
    validate_hybrid_trajectory,
    write_trajectory_rows,
)

ROOT = Path(r"d:\compitition\MioVerse\final_version\ml_research")
SRC = (
    ROOT
    / "results"
    / "coord_custom_ai"
    / "ecbs"
    / "trajectories"
    / "SH10_leave400_collidefix3_ecbs_joint_w1.5_k6_hier.csv"
)
DST = (
    ROOT
    / "results"
    / "coord_custom_ai"
    / "ecbs"
    / "trajectories"
    / "SH10_leave400_facing_ecbs_joint_w1.5_k6_hier.csv"
)


def _unit() -> None:
    pose = {"Bumblebee": (6, 3, 90), "Jazz": (9, 3, 90), "Sideswipe": (12, 3, 90)}
    paths = {
        "Bumblebee": [(6, 3), (7, 3)],
        "Jazz": [(9, 3), (10, 3)],
        "Sideswipe": [(12, 3), (11, 3)],
    }
    tl = _plan_paths_as_poses(pose, paths)
    assert tl["Bumblebee"] == [(6, 3, 0), (7, 3, 0)], tl["Bumblebee"]
    assert tl["Jazz"] == [(9, 3, 0), (10, 3, 0)], tl["Jazz"]
    assert tl["Sideswipe"][0][2] == 180 and tl["Sideswipe"][-1] == (11, 3, 180), tl["Sideswipe"]
    # 180° is two in-place ticks, cell unchanged, then the step.
    chain = _turn_pitch_chain(90, 270)
    assert chain == [180, 270], chain
    pose2 = {"A": (1, 1, 90), "B": (4, 1, 0)}
    paths2 = {"A": [(1, 1), (1, 2)], "B": [(4, 1), (5, 1)]}
    tl2 = _plan_paths_as_poses(pose2, paths2)
    assert tl2["A"] == [(1, 2, 90)], tl2["A"]
    assert tl2["B"] == [(5, 1, 0)], tl2["B"]
    print("UNIT ok", flush=True)


def _load_frames(path: Path):
    frames = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            frames[int(row["timestamp"])][row["name"]] = row
    return frames


def _facing_rows(frames) -> list:
    times = sorted(frames)
    t0 = times[0]
    pitches = {n: int(frames[t0][n]["pitch"]) % 360 for n in frames[t0]}
    first = {}
    for n, row in frames[t0].items():
        r = dict(row)
        r["pitch"] = pitches[n]
        first[n] = r
    out_frames = [first]
    for t_prev, t_next in zip(times, times[1:]):
        cur = frames[t_prev]
        nxt = frames[t_next]
        for _spin in range(4):
            turning = {}
            for n, row in cur.items():
                x, y = int(row["X"]), int(row["Y"])
                nr = nxt[n]
                nx, ny = int(nr["X"]), int(nr["Y"])
                if (nx, ny) == (x, y):
                    continue
                need = _pitch_from_delta(nx - x, ny - y, pitches[n])
                chain = _turn_pitch_chain(pitches[n], need)
                if chain:
                    turning[n] = chain[0]
            if not turning:
                break
            hold = {}
            for n, row in cur.items():
                r = dict(row)
                if n in turning:
                    pitches[n] = turning[n]
                r["pitch"] = pitches[n] % 360
                hold[n] = r
            out_frames.append(hold)
        move = {}
        for n, nr in nxt.items():
            r = dict(nr)
            r["pitch"] = pitches[n] % 360
            move[n] = r
        out_frames.append(move)
    rows = []
    for ts, frame in enumerate(out_frames):
        for n in sorted(frame):
            r = dict(frame[n])
            r["timestamp"] = ts
            r["name"] = n
            rows.append(r)
    return rows


def _scan(rows) -> None:
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    moves = aligned = strafe = mat = jump = 0
    for seq in by.values():
        seq.sort(key=lambda r: int(r["timestamp"]))
        for a, b in zip(seq, seq[1:]):
            if int(b["timestamp"]) != int(a["timestamp"]) + 1:
                continue
            dx = int(b["X"]) - int(a["X"])
            dy = int(b["Y"]) - int(a["Y"])
            dist = abs(dx) + abs(dy)
            pa = int(a["pitch"]) % 360
            pb = int(b["pitch"]) % 360
            if dist == 0:
                continue
            if dist != 1:
                jump += 1
                continue
            moves += 1
            if pa != pb:
                mat += 1
            need = _pitch_from_delta(dx, dy, pa)
            if pb == need and pa == pb:
                aligned += 1
            else:
                strafe += 1
    print(
        "SCAN",
        "ticks",
        max(int(r["timestamp"]) for r in rows) + 1,
        "moves",
        moves,
        "aligned",
        aligned,
        "strafe",
        strafe,
        "move_and_turn",
        mat,
        "jump",
        jump,
        "strafe_ratio",
        round(strafe / moves, 4) if moves else 0,
        flush=True,
    )


def main() -> None:
    _unit()
    frames = _load_frames(SRC)
    rows = _facing_rows(frames)
    write_trajectory_rows(DST, rows)
    print("WROTE", DST, "rows", len(rows), flush=True)
    _scan(rows)
    meta = load_custom_meta(10, n_tasks=400, max_sim_time=500000, wall_timeout=900)
    meta = dict(meta)
    meta["task_csv"] = str(
        (ROOT / "results" / "coord_custom_ai" / "batch_400_switch_20" / "SH10_demo400_tasks.csv").resolve()
    )
    meta["n_tasks"] = 400
    meta["id"] = "SH10_leave400_facing"
    rep = validate_hybrid_trajectory(meta, DST)
    issues = rep.get("issues") or {}
    print(
        "VALID",
        rep.get("ok"),
        "collisions",
        issues.get("n_collisions"),
        "swaps",
        issues.get("n_swaps"),
        "illegal",
        issues.get("n_illegal_motion"),
        "dwell",
        issues.get("n_pickup_unload_dwell"),
        "fifo",
        issues.get("n_fifo_violations"),
        "display",
        issues.get("n_display_mismatch"),
        "carry",
        issues.get("n_task_carry_violations"),
        "pickup_cell",
        issues.get("n_pickup_cell_violations"),
        "hard_wall",
        issues.get("n_hard_wall"),
        "summary",
        rep.get("summary") or rep.get("validate_summary"),
        flush=True,
    )
    if not rep.get("ok"):
        for k, v in issues.items():
            if str(k).startswith("n_"):
                continue
            if v:
                print("ISSUE", k, v if not isinstance(v, list) else v[:3], flush=True)


if __name__ == "__main__":
    main()
