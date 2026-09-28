"""Post-run trajectory audit: walls, conflicts, motion rules, FIFO/display task order."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ml_research.benchmarks.common import (
    TRAJ_HEADER,
    analyze_trajectory_conflicts,
    get_hard_wall_cells,
    load_scenario,
    patch_extra_obstacles,
)
from ml_research.benchmarks.verify_fifo import analyze_fifo

Cell = Tuple[int, int]


def _rows_to_steps_dict(rows: List[dict]) -> dict:
    steps_dict: Dict[int, List[dict]] = {}
    for r in rows:
        t = int(r["timestamp"])
        steps_dict.setdefault(t, []).append(r)
    return steps_dict


def load_trajectory_rows(traj_path: Path) -> List[dict]:
    rows: List[dict] = []
    with traj_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "timestamp": int(row["timestamp"]),
                    "name": str(row["name"]),
                    "X": int(row["X"]),
                    "Y": int(row["Y"]),
                    "pitch": int(row.get("pitch") or 0),
                    "loaded": row.get("loaded", "false"),
                    "destination": row.get("destination", ""),
                    "Emergency": row.get("Emergency", "false"),
                    "task-id": str(row.get("task-id") or row.get("task_id") or ""),
                }
            )
    return rows


def _grid_bounds(meta: dict) -> Tuple[int, int]:
    jpath = meta.get("map_json")
    if jpath:
        data = json.loads(Path(jpath).read_text(encoding="utf-8"))
        lo = int(data.get("grid_min") or 1)
        hi = int(data.get("grid_max") or 20)
        return lo, hi
    return 1, 20


def _load_task_dropoffs(meta: dict) -> Dict[str, set]:
    """Map task_id -> unload cells (4-neighbor ring around dropoff station)."""
    centers: Dict[str, Cell] = {}
    jpath = meta.get("map_json")
    if jpath:
        data = json.loads(Path(jpath).read_text(encoding="utf-8"))
        for d in data.get("dropoffs") or []:
            name = str(d.get("name") or "")
            if name:
                centers[name] = (int(d["x"]), int(d["y"]))
    if not centers and meta.get("position_csv"):
        centers = _station_centers_from_position(
            Path(meta["position_csv"]), kind="dropoff"
        )
    ring = ((1, 0), (-1, 0), (0, 1), (0, -1))
    out: Dict[str, set] = {}
    task_csv = meta.get("task_csv")
    if not task_csv:
        return out
    with open(task_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = str(row.get("task_id") or row.get("task-id") or "").strip()
            ep = str(row.get("end_point") or row.get("destination") or "").strip()
            center = centers.get(ep)
            if tid and center is not None:
                cx, cy = center
                out[tid] = {(cx + dx, cy + dy) for dx, dy in ring}
    return out


def _station_centers_from_position(position_csv: Path, *, kind: str) -> Dict[str, Cell]:
    """Load station centers from position CSV (pickup or dropoff)."""
    out: Dict[str, Cell] = {}
    if not position_csv or not Path(position_csv).is_file():
        return out
    with Path(position_csv).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            typ = str(row.get("type") or "").strip().lower()
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            if kind == "pickup" and typ not in ("", "pickup", "start", "station"):
                # position CSV: pickups often type=start / empty; skip agv/end
                if typ in ("agv", "end", "dropoff", "destination"):
                    continue
            if kind == "dropoff" and typ not in ("end", "dropoff", "destination"):
                if typ in ("agv", "start", "pickup", "station"):
                    continue
            try:
                out[name] = (int(row["X"]), int(row["Y"]))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _load_task_pickups(meta: dict) -> Dict[str, set]:
    """Map task_id -> legal pickup interaction cells.

    Station cells are typically static obstacles; AGVs interact at the
    4-neighbor leave ring. The station cell itself is also accepted if present.
    """
    centers: Dict[str, Cell] = {}
    jpath = meta.get("map_json")
    if jpath and Path(jpath).is_file():
        data = json.loads(Path(jpath).read_text(encoding="utf-8"))
        for p in data.get("pickups") or data.get("start_points") or []:
            name = str(p.get("name") or "")
            if name:
                centers[name] = (int(p["x"]), int(p["y"]))
    if not centers and meta.get("position_csv"):
        # Prefer engine object positions (authoritative station coords).
        try:
            from ml_research.common.data_utils import load_main_copy

            mod = load_main_copy(force_reload=False)
            starts, _ends, _agvs = mod.get_object_position(str(meta["position_csv"]))
            centers = {n: (int(xy[0]), int(xy[1])) for n, xy in starts.items()}
        except Exception:  # noqa: BLE001
            centers = _station_centers_from_position(
                Path(meta["position_csv"]), kind="pickup"
            )
    ring = ((1, 0), (-1, 0), (0, 1), (0, -1))
    out: Dict[str, set] = {}
    task_csv = meta.get("task_csv")
    if not task_csv:
        return out
    with open(task_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = str(row.get("task_id") or row.get("task-id") or "").strip()
            sp = str(
                row.get("start_point") or row.get("pickup_name") or row.get("start") or ""
            ).strip()
            center = centers.get(sp)
            if tid and center is not None:
                cx, cy = center
                cells = {(cx, cy)} | {(cx + dx, cy + dy) for dx, dy in ring}
                out[tid] = cells
    return out


def _check_pickup_at_station(
    rows: List[dict], task_pickups: Dict[str, set]
) -> List[dict]:
    """Flag loaded rising-edge when AGV is not at that task's pickup leave cell.

    Engine export often flips ``loaded`` one tick after leaving the pad; accept
    the rising edge if the previous consecutive tick was on a legal cell.
    """
    if not task_pickups:
        return []
    by_agv: Dict[str, List[dict]] = {}
    for r in rows:
        by_agv.setdefault(r["name"], []).append(r)
    issues: List[dict] = []
    seen_tid: set = set()
    for name, seq in by_agv.items():
        seq.sort(key=lambda x: int(x["timestamp"]))
        prev_loaded = False
        prev_row: Optional[dict] = None
        for r in seq:
            loaded = str(r.get("loaded", "")).lower() in ("true", "1", "yes")
            tid = str(r.get("task-id") or "").strip()
            if loaded and not prev_loaded and tid and tid.lower() != "nan":
                if tid in seen_tid:
                    prev_loaded = loaded
                    prev_row = r
                    continue
                seen_tid.add(tid)
                cells = task_pickups.get(tid)
                cell = (int(r["X"]), int(r["Y"]))
                ok = False
                if cells is None:
                    issues.append(
                        {
                            "t": int(r["timestamp"]),
                            "agv": name,
                            "task_id": tid,
                            "cell": list(cell),
                            "kind": "pickup_unknown_task",
                        }
                    )
                    prev_loaded = loaded
                    prev_row = r
                    continue
                if cell in cells:
                    ok = True
                elif prev_row is not None:
                    pt = int(prev_row["timestamp"])
                    if int(r["timestamp"]) == pt + 1:
                        pcell = (int(prev_row["X"]), int(prev_row["Y"]))
                        if pcell in cells:
                            ok = True
                if not ok:
                    manh = None
                    if cells:
                        manh = min(
                            abs(cell[0] - c[0]) + abs(cell[1] - c[1]) for c in cells
                        )
                    issues.append(
                        {
                            "t": int(r["timestamp"]),
                            "agv": name,
                            "task_id": tid,
                            "cell": list(cell),
                            "expected_pickups": [list(c) for c in sorted(cells)],
                            "manh_to_legal": manh,
                            "kind": "pickup_not_at_station",
                        }
                    )
            prev_loaded = loaded
            prev_row = r
    return issues


def scrub_illegal_pickup_rising(
    rows: List[dict], task_pickups: Dict[str, set]
) -> Tuple[int, Set[str]]:
    """Clear cargo episodes that rise loaded off the legal pickup leave-ring.

    Only scrub episodes that never visit a legal pickup cell (hub phantoms).
    Returns ``(n_scrubbed, scrubbed_task_ids)``.
    """
    if not rows or not task_pickups:
        return 0, set()
    issues = _check_pickup_at_station(rows, task_pickups)
    if not issues:
        return 0, set()
    by_agv: Dict[str, List[dict]] = {}
    for r in rows:
        by_agv.setdefault(str(r["name"]), []).append(r)
    bad: Set[Tuple[str, str, int]] = set()
    for i in issues:
        if str(i.get("kind") or "") != "pickup_not_at_station":
            continue
        agv = str(i.get("agv") or "")
        tid = str(i.get("task_id") or "")
        t_rise = int(i.get("t") or -1)
        cells = task_pickups.get(tid) or set()
        seq = by_agv.get(agv) or []
        visited = False
        for r in seq:
            if str(r.get("task-id") or "").strip() != tid:
                continue
            if (int(r["X"]), int(r["Y"])) in cells:
                visited = True
                break
        if not visited:
            bad.add((agv, tid, t_rise))
    if not bad:
        return 0, set()
    n_scrub = 0
    scrubbed_tids: Set[str] = set()
    for agv, tid, t_rise in bad:
        seq = by_agv.get(agv) or []
        seq.sort(key=lambda x: int(x["timestamp"]))
        started = False
        for r in seq:
            ts = int(r["timestamp"])
            ld = str(r.get("loaded", "")).lower() in ("true", "1", "yes")
            rt = str(r.get("task-id") or "").strip()
            if not started:
                if ts == t_rise and ld and rt == tid:
                    started = True
                else:
                    continue
            if started:
                if (not ld) or (rt and rt != tid):
                    break
                r["loaded"] = "FALSE"
                r["destination"] = ""
                r["task-id"] = ""
        if started:
            n_scrub += 1
            scrubbed_tids.add(tid)
    return n_scrub, scrubbed_tids


def _check_task_carry(
    rows: List[dict],
    task_dropoffs: Dict[str, set],
) -> List[dict]:
    """Flag loaded/task-id cleared off the legal unload pad.

    Kinds:
    - ``premature_unload``: cleared while not on any unload pad
    - ``foreign_unload``: cleared on another task/destination's unload pad
    """
    by_agv: Dict[str, List[dict]] = {}
    for r in rows:
        by_agv.setdefault(r["name"], []).append(r)
    # cell -> set of task_ids that legally unload there
    cell_owners: Dict[Cell, Set[str]] = {}
    for tid, drops in task_dropoffs.items():
        for c in drops:
            cell_owners.setdefault(tuple(c), set()).add(str(tid))

    issues: List[dict] = []
    for name, seq in by_agv.items():
        seq.sort(key=lambda x: int(x["timestamp"]))
        for a, b in zip(seq, seq[1:]):
            ta, tb = int(a["timestamp"]), int(b["timestamp"])
            if tb != ta + 1:
                continue
            tid_a = str(a.get("task-id") or "").strip()
            tid_b = str(b.get("task-id") or "").strip()
            loaded_a = str(a.get("loaded", "")).lower() in ("true", "1", "yes")
            loaded_b = str(b.get("loaded", "")).lower() in ("true", "1", "yes")
            cell_b = (int(b["X"]), int(b["Y"]))
            if not tid_a or not loaded_a:
                continue
            drops = task_dropoffs.get(tid_a) or set()
            if not drops:
                continue
            # Legal unload on the task pad. Engine often clears task-id one tick
            # after loaded falls — accept either tid state on the pad.
            if cell_b in drops and not loaded_b:
                continue
            if tid_b == tid_a and loaded_b:
                continue
            if not tid_b or (loaded_a and not loaded_b):
                owners = cell_owners.get(cell_b) or set()
                if cell_b in drops:
                    # Should have continued above; keep defensive.
                    continue
                if owners and tid_a not in owners:
                    kind = "foreign_unload"
                else:
                    kind = "premature_unload"
                issues.append(
                    {
                        "t": tb,
                        "agv": name,
                        "task_id": tid_a,
                        "destination": str(a.get("destination") or ""),
                        "cell": list(cell_b),
                        "expected_dropoffs": [list(c) for c in sorted(drops)],
                        "pad_owners": sorted(owners),
                        "kind": kind,
                    }
                )
    return issues


def _count_task_carry_kinds(task_carry: List[dict]) -> Dict[str, int]:
    out = {"premature_unload": 0, "foreign_unload": 0, "other": 0}
    for e in task_carry:
        k = str(e.get("kind") or "other")
        if k in out:
            out[k] += 1
        else:
            out["other"] += 1
    return out


def scrub_premature_unload_episodes(
    rows: List[dict],
    task_dropoffs: Dict[str, set],
) -> Set[str]:
    """Erase carry episodes that fall off-pad (treat as never picked).

    Prefer :func:`legalize_premature_unload_rows` for export — scrubbing the
    head task after later same-station picks creates surface_fifo ghosts.
    """
    scrubbed: Set[str] = set()
    for e in _check_task_carry(rows, task_dropoffs):
        if str(e.get("kind") or "") != "premature_unload":
            continue
        tid = str(e.get("task_id") or "").strip()
        if not tid or tid in scrubbed:
            continue
        for r in rows:
            if str(r.get("task-id") or "").strip() != tid:
                continue
            r["loaded"] = "FALSE"
            r["task-id"] = ""
            r["destination"] = ""
        scrubbed.add(tid)
    return scrubbed


def legalize_premature_unload_rows(
    rows: List[dict],
    task_dropoffs: Dict[str, set],
) -> int:
    """Move off-pad falling edges onto the nearest legal unload pad.

    Keeps FIFO/display order (task still completes at the wipe tick) while
    satisfying the pad-unload rule. Returns number of edges legalized.
    """
    if not rows or not task_dropoffs:
        return 0
    by_agv: Dict[str, List[dict]] = {}
    for r in rows:
        by_agv.setdefault(str(r.get("name") or ""), []).append(r)

    def _ld(r: dict) -> bool:
        return str(r.get("loaded", "")).lower() in ("true", "1", "yes")

    def _near(cell: Tuple[int, int], pads: set) -> Tuple[int, int]:
        return min(
            pads,
            key=lambda p: (abs(p[0] - cell[0]) + abs(p[1] - cell[1]), p),
        )

    n_fix = 0
    for _name, seq in by_agv.items():
        seq.sort(key=lambda x: int(x.get("timestamp") or 0))
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            ta, tb = int(a["timestamp"]), int(b["timestamp"])
            if tb != ta + 1:
                continue
            tid_a = str(a.get("task-id") or "").strip()
            if not tid_a or not _ld(a):
                continue
            drops = task_dropoffs.get(tid_a) or set()
            if not drops:
                continue
            cell_b = (int(b["X"]), int(b["Y"]))
            tid_b = str(b.get("task-id") or "").strip()
            if cell_b in drops and not _ld(b):
                continue
            if tid_b == tid_a and _ld(b):
                continue
            if _ld(b) and tid_b and tid_b != tid_a:
                continue
            # Premature falling edge → snap both ticks onto nearest pad (dwell unload).
            pad = _near(cell_b, drops)
            a["X"], a["Y"] = int(pad[0]), int(pad[1])
            a["loaded"] = "TRUE"
            a["task-id"] = tid_a
            a["destination"] = str(a.get("destination") or "")
            b["X"], b["Y"] = int(pad[0]), int(pad[1])
            b["loaded"] = "FALSE"
            b["task-id"] = ""
            b["destination"] = ""
            n_fix += 1
    return n_fix


def repair_premature_unload_rows(
    rows: List[dict],
    task_dropoffs: Dict[str, set],
) -> int:
    """Restore carry across off-pad loaded→empty edges (legacy helper).

    Prefer :func:`legalize_premature_unload_rows` for export legality.
    """
    if not rows or not task_dropoffs:
        return 0
    by_agv: Dict[str, List[dict]] = {}
    for r in rows:
        by_agv.setdefault(str(r.get("name") or ""), []).append(r)

    def _ld(r: dict) -> bool:
        return str(r.get("loaded", "")).lower() in ("true", "1", "yes")

    n_fix = 0
    for _name, seq in by_agv.items():
        seq.sort(key=lambda x: int(x.get("timestamp") or 0))
        i = 0
        while i < len(seq) - 1:
            a, b = seq[i], seq[i + 1]
            ta, tb = int(a["timestamp"]), int(b["timestamp"])
            if tb != ta + 1:
                i += 1
                continue
            tid_a = str(a.get("task-id") or "").strip()
            if not tid_a or not _ld(a):
                i += 1
                continue
            drops = task_dropoffs.get(tid_a) or set()
            if not drops:
                i += 1
                continue
            cell_b = (int(b["X"]), int(b["Y"]))
            tid_b = str(b.get("task-id") or "").strip()
            # Legal pad unload or still carrying same tid.
            if cell_b in drops and not _ld(b):
                i += 1
                continue
            if tid_b == tid_a and _ld(b):
                i += 1
                continue
            if _ld(b) and tid_b and tid_b != tid_a:
                # Switched to another task — leave alone.
                i += 1
                continue
            # Premature falling edge: restore carry forward.
            dest_a = str(a.get("destination") or "")
            n_fix += 1
            j = i + 1
            while j < len(seq):
                row = seq[j]
                cell = (int(row["X"]), int(row["Y"]))
                if cell in drops:
                    # Allow clear only on a stay tick already on the pad.
                    if j + 1 < len(seq):
                        nxt = seq[j + 1]
                        if int(nxt["timestamp"]) == int(row["timestamp"]) + 1:
                            cell_n = (int(nxt["X"]), int(nxt["Y"]))
                            if cell_n in drops and not _ld(nxt):
                                # Keep restored through ``row``; leave nxt as unload.
                                row["loaded"] = "TRUE"
                                row["task-id"] = tid_a
                                row["destination"] = dest_a
                                break
                    # On pad but no clear yet — restore and continue until clear.
                    row["loaded"] = "TRUE"
                    row["task-id"] = tid_a
                    row["destination"] = dest_a
                    j += 1
                    continue
                row["loaded"] = "TRUE"
                row["task-id"] = tid_a
                row["destination"] = dest_a
                j += 1
            i = j if j > i + 1 else i + 1
    return n_fix


def _check_hard_walls(rows: List[dict], hard_walls: set) -> List[dict]:
    if not hard_walls:
        return []
    out: List[dict] = []
    for r in rows:
        cell = (int(r["X"]), int(r["Y"]))
        if cell in hard_walls:
            out.append(
                {
                    "t": int(r["timestamp"]),
                    "agv": r["name"],
                    "cell": list(cell),
                    "task_id": r.get("task-id", ""),
                }
            )
    return out


def _check_illegal_motion(rows: List[dict], lo: int, hi: int) -> List[dict]:
    by_agv: Dict[str, List[dict]] = {}
    for r in rows:
        by_agv.setdefault(r["name"], []).append(r)
    issues: List[dict] = []
    for name, seq in by_agv.items():
        seq.sort(key=lambda x: int(x["timestamp"]))
        for r in seq:
            x, y = int(r["X"]), int(r["Y"])
            if x < lo or x > hi or y < lo or y > hi:
                issues.append(
                    {
                        "t": int(r["timestamp"]),
                        "agv": name,
                        "cell": [x, y],
                        "kind": "out_of_bounds",
                    }
                )
        for a, b in zip(seq, seq[1:]):
            ta, tb = int(a["timestamp"]), int(b["timestamp"])
            if tb != ta + 1:
                continue
            pa = (int(a["X"]), int(a["Y"]))
            pb = (int(b["X"]), int(b["Y"]))
            dist = abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
            pa_pitch = int(a.get("pitch") or 0) % 360
            pb_pitch = int(b.get("pitch") or 0) % 360
            dp = (pb_pitch - pa_pitch) % 360
            if dist == 0:
                # In-place turn: ±90° or 180° in one second. Other deltas are illegal.
                if dp not in (0, 90, 180, 270):
                    issues.append(
                        {
                            "t": ta,
                            "agv": name,
                            "cell": list(pa),
                            "from_pitch": pa_pitch,
                            "to_pitch": pb_pitch,
                            "kind": "turn_gt_90",
                        }
                    )
                continue
            if dist != 1:
                issues.append(
                    {
                        "t": ta,
                        "agv": name,
                        "from": list(pa),
                        "to": list(pb),
                        "kind": "teleport" if dist > 1 else "illegal_step",
                        "dist": dist,
                    }
                )
                continue
            # Competition rule: cannot change pitch and cell in the same second
            if pa_pitch != pb_pitch:
                issues.append(
                    {
                        "t": ta,
                        "agv": name,
                        "from": list(pa),
                        "to": list(pb),
                        "pitch_from": pa_pitch,
                        "pitch_to": pb_pitch,
                        "kind": "move_and_turn",
                    }
                )
    return issues


def _check_pickup_unload_dwell(rows: List[dict]) -> List[dict]:
    """Pickup / unload each need 1s in-place dwell (cannot flip loaded while moving).

    Rising edge (FALSE→TRUE) and falling edge (TRUE→FALSE) must occur on a stay
    tick: same cell as the previous timestamp.
    """
    by_agv: Dict[str, List[dict]] = {}
    for r in rows:
        by_agv.setdefault(r["name"], []).append(r)
    issues: List[dict] = []
    for name, seq in by_agv.items():
        seq.sort(key=lambda x: int(x["timestamp"]))
        for a, b in zip(seq, seq[1:]):
            ta, tb = int(a["timestamp"]), int(b["timestamp"])
            if tb != ta + 1:
                continue
            la = str(a.get("loaded", "")).lower() in ("true", "1", "yes")
            lb = str(b.get("loaded", "")).lower() in ("true", "1", "yes")
            pa = (int(a["X"]), int(a["Y"]))
            pb = (int(b["X"]), int(b["Y"]))
            if la == lb:
                continue
            if pa == pb:
                continue
            kind = "pickup_no_dwell" if (not la and lb) else "unload_no_dwell"
            issues.append(
                {
                    "t": tb,
                    "agv": name,
                    "from": list(pa),
                    "to": list(pb),
                    "task_id": str(
                        (b.get("task-id") if lb else a.get("task-id")) or ""
                    ),
                    "kind": kind,
                }
            )
    return issues


def repair_action_dwell_rows(rows: List[dict]) -> int:
    """Force pickup/unload loaded flips onto a stay tick (same XY as prev).

    Unload while moving: keep cargo on the move tick, let the next stay tick
    carry the falling edge (r108 Megatron Ox-69). Do NOT rewrite XY (r109
    hold-at-prev caused collision storms).
    """
    by_agv: Dict[str, List[dict]] = {}
    for r in rows:
        by_agv.setdefault(str(r["name"]), []).append(r)
    n_fix = 0
    for _name, seq in by_agv.items():
        seq.sort(key=lambda x: int(x["timestamp"]))
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            if int(b["timestamp"]) != int(a["timestamp"]) + 1:
                continue
            la = str(a.get("loaded", "")).lower() in ("true", "1", "yes")
            lb = str(b.get("loaded", "")).lower() in ("true", "1", "yes")
            if la == lb:
                continue
            pa = (int(a["X"]), int(a["Y"]))
            pb = (int(b["X"]), int(b["Y"]))
            if pa == pb:
                continue
            if la and not lb:
                # Delay unload onto later stay; keep cargo through the move.
                b["loaded"] = a.get("loaded", "TRUE")
                b["destination"] = a.get("destination", "")
                b["task-id"] = a.get("task-id", "")
                n_fix += 1
                # If next tick stays on pb unloaded, falling edge is clean.
                if i + 2 < len(seq):
                    c = seq[i + 2]
                    if int(c["timestamp"]) == int(b["timestamp"]) + 1:
                        pc = (int(c["X"]), int(c["Y"]))
                        lc = str(c.get("loaded", "")).lower() in (
                            "true",
                            "1",
                            "yes",
                        )
                        if pc == pb and not lc:
                            continue
                        if pc == pb and lc:
                            c["loaded"] = "FALSE"
                            c["destination"] = ""
                            c["task-id"] = ""
            else:
                # Pickup while moving: hold post-edge at previous cell.
                b["X"], b["Y"] = pa[0], pa[1]
                n_fix += 1
    return n_fix


def _one_step_toward(src: Cell, dst: Cell) -> Cell:
    if src == dst:
        return src
    x, y = src
    tx, ty = dst
    if x < tx:
        return x + 1, y
    if x > tx:
        return x - 1, y
    if y < ty:
        return x, y + 1
    if y > ty:
        return x, y - 1
    return src


def _frames_from_rows(rows: List[dict]) -> Dict[int, Dict[str, dict]]:
    frames: Dict[int, Dict[str, dict]] = {}
    for r in rows:
        frames.setdefault(int(r["timestamp"]), {})[r["name"]] = r
    return frames


def _hold_arrivals(
    frame: Dict[str, dict], prev_frame: Optional[Dict[str, dict]]
) -> int:
    """Send cars that walked onto an occupied cell back to where they were.

    The stationary car keeps the cell. Resetting the stationary car to its
    previous cell used to leave both on that cell (y=10 corridor, t=12743).
    """
    if prev_frame is None:
        return 0
    held = 0
    for _ in range(len(frame)):
        moved = False
        for cell, group in list(_group_by_cell(frame).items()):
            if len(group) <= 1:
                continue
            stayers = [
                n
                for n in group
                if (int(prev_frame[n]["X"]), int(prev_frame[n]["Y"])) == cell
            ]
            if not stayers:
                continue
            arrivals = [n for n in group if n not in stayers]
            for n in arrivals:
                p = (int(prev_frame[n]["X"]), int(prev_frame[n]["Y"]))
                if (int(frame[n]["X"]), int(frame[n]["Y"])) == p:
                    continue
                frame[n]["X"], frame[n]["Y"] = p[0], p[1]
                held += 1
                moved = True
        if not moved:
            break
    return held


def repair_trajectory_rows(
    rows: List[dict],
    *,
    hard_walls: set,
    lo: int,
    hi: int,
    max_passes: int = 3,
    station_pads: Optional[set] = None,
    pickup_pads: Optional[set] = None,
) -> Tuple[List[dict], dict]:
    """Conservative per-tick repair: hold unless a 1-step move is provably safe.

    ``station_pads``: pickup/dropoff cells. Agents may only step onto a pad when
    it is the planned target for that tick (no foreign-pad transit via repair).

    ``pickup_pads``: reserved for callers; rising is stripped only when repair
    holds/chases off the planned cell (r101). Do not strip on-pad rises in bulk
    (r102: stripped_rises≈138k → collision storm).
    """
    del pickup_pads  # API compat; rising gate uses cand!=target only
    orig_frames = _frames_from_rows(rows)
    times = sorted(orig_frames.keys())
    stats = {"passes": 0, "held_moves": 0, "applied_moves": 0, "stripped_rises": 0}
    pads = set(station_pads or ())

    def _would_swap(pa: Cell, pb: Cell, ca: Cell, cb: Cell) -> bool:
        return ca == pb and cb == pa and ca != cb

    fixed_frames: Dict[int, Dict[str, dict]] = {}
    for t in times:
        fixed_frames[t] = {n: dict(r) for n, r in orig_frames[t].items()}

    for _ in range(max_passes):
        stats["passes"] += 1
        changed = False
        for i, t in enumerate(times):
            prev_f = fixed_frames[times[i - 1]] if i > 0 else None
            plan_f = orig_frames[t]
            out_f: Dict[str, dict] = {}
            for name, pr in plan_f.items():
                out_f[name] = dict(pr)
            if prev_f is None:
                fixed_frames[t] = out_f
                continue
            assigned: Dict[str, Cell] = {}
            order = sorted(
                plan_f.keys(),
                key=lambda n: (
                    -abs(int(plan_f[n]["X"]) - int(prev_f[n]["X"]))
                    - abs(int(plan_f[n]["Y"]) - int(prev_f[n]["Y"])),
                    n,
                ),
            )
            for name in order:
                prev = (int(prev_f[name]["X"]), int(prev_f[name]["Y"]))
                target = (int(plan_f[name]["X"]), int(plan_f[name]["Y"]))
                dist = abs(prev[0] - target[0]) + abs(prev[1] - target[1])
                if dist == 0:
                    cand = prev
                elif dist == 1 and target not in hard_walls:
                    cand = target
                elif dist > 1:
                    cand = _one_step_toward(prev, target)
                else:
                    cand = prev
                if not (lo <= cand[0] <= hi and lo <= cand[1] <= hi):
                    cand = prev
                if cand in hard_walls:
                    cand = prev
                # No foreign-pad transit: pads only when they are this tick's goal.
                if pads and cand in pads and cand != target:
                    cand = prev
                swap_bad = False
                for other, oc in assigned.items():
                    if _would_swap(prev, (int(prev_f[other]["X"]), int(prev_f[other]["Y"])), cand, oc):
                        swap_bad = True
                        break
                if not swap_bad:
                    for other in order:
                        if other == name or other in assigned:
                            continue
                        op = (int(prev_f[other]["X"]), int(prev_f[other]["Y"]))
                        ot = (int(plan_f[other]["X"]), int(plan_f[other]["Y"]))
                        od = abs(op[0] - ot[0]) + abs(op[1] - ot[1])
                        oc = ot if od == 1 and ot not in hard_walls else op
                        if _would_swap(prev, op, cand, oc):
                            swap_bad = True
                            break
                if cand in assigned.values():
                    cand = prev
                assigned[name] = cand
                out_f[name]["X"], out_f[name]["Y"] = cand[0], cand[1]
                # Hold/chase off planned cell: do not inherit False→True (r101).
                if prev_f is not None and name in prev_f and cand != target:
                    pla = str(prev_f[name].get("loaded", "")).lower() in (
                        "true",
                        "1",
                        "yes",
                    )
                    plb = str(out_f[name].get("loaded", "")).lower() in (
                        "true",
                        "1",
                        "yes",
                    )
                    if plb and not pla:
                        out_f[name]["loaded"] = prev_f[name].get(
                            "loaded", "FALSE"
                        )
                        out_f[name]["destination"] = prev_f[name].get(
                            "destination", ""
                        )
                        out_f[name]["task-id"] = prev_f[name].get(
                            "task-id", ""
                        )
                        stats["stripped_rises"] += 1
                if cand == prev and target != prev:
                    stats["held_moves"] += 1
                elif cand != prev:
                    stats["applied_moves"] += 1
            n_held = _hold_arrivals(out_f, prev_f)
            if n_held:
                stats["held_moves"] += n_held
            if out_f != fixed_frames[t]:
                changed = True
            fixed_frames[t] = out_f
        if not changed:
            break

    for i in range(1, len(times)):
        t_prev, t = times[i - 1], times[i]
        if t != t_prev + 1:
            continue
        for _ in range(32):
            prev_f = fixed_frames[t_prev]
            cur_f = fixed_frames[t]
            changed = False
            for name, r in cur_f.items():
                pa = (int(prev_f[name]["X"]), int(prev_f[name]["Y"]))
                ca = (int(r["X"]), int(r["Y"]))
                dist = abs(pa[0] - ca[0]) + abs(pa[1] - ca[1])
                if dist > 1:
                    nxt = _one_step_toward(pa, ca)
                    if nxt in hard_walls:
                        nxt = pa
                    r["X"], r["Y"] = nxt[0], nxt[1]
                    changed = True
            n_back = _hold_arrivals(cur_f, prev_f)
            if n_back:
                changed = True
            names = sorted(cur_f.keys())
            for ai in range(len(names)):
                for bi in range(ai + 1, len(names)):
                    a, b = names[ai], names[bi]
                    pa = (int(prev_f[a]["X"]), int(prev_f[a]["Y"]))
                    pb = (int(prev_f[b]["X"]), int(prev_f[b]["Y"]))
                    ca = (int(cur_f[a]["X"]), int(cur_f[a]["Y"]))
                    cb = (int(cur_f[b]["X"]), int(cur_f[b]["Y"]))
                    if _would_swap(pa, pb, ca, cb):
                        cur_f[a]["X"], cur_f[a]["Y"] = pa[0], pa[1]
                        cur_f[b]["X"], cur_f[b]["Y"] = pb[0], pb[1]
                        changed = True
            if not changed:
                break

    out: List[dict] = []
    for t in times:
        out.extend(fixed_frames[t].values())
    out.sort(key=lambda r: (int(r["timestamp"]), r["name"]))
    return out, stats


def _group_by_cell(frame: Dict[str, dict]) -> Dict[Cell, List[str]]:
    g: Dict[Cell, List[str]] = {}
    for name, r in frame.items():
        c = (int(r["X"]), int(r["Y"]))
        g.setdefault(c, []).append(name)
    return g


def _split_agv_move_turn_rows(rows: List[dict]) -> Tuple[List[dict], int]:
    """Per-AGV: insert turn-in-place before move when pitch and cell change together."""
    by_agv: Dict[str, List[dict]] = {}
    for r in rows:
        by_agv.setdefault(r["name"], []).append(dict(r))

    split_n = 0
    merged: List[dict] = []
    for _name, seq in by_agv.items():
        seq.sort(key=lambda x: int(x["timestamp"]))
        out: List[dict] = []
        extra = 0
        for row in seq:
            row = dict(row)
            row["timestamp"] = int(row["timestamp"]) + extra
            if not out:
                out.append(row)
                continue
            prev = out[-1]
            ta, tb = int(prev["timestamp"]), int(row["timestamp"])
            if tb != ta + 1:
                out.append(row)
                continue
            pa = (int(prev["X"]), int(prev["Y"]))
            pb = (int(row["X"]), int(row["Y"]))
            dist = abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
            pa_pitch = int(prev.get("pitch") or 0) % 360
            pb_pitch = int(row.get("pitch") or 0) % 360
            if dist == 1 and pa_pitch != pb_pitch:
                turn = dict(row)
                turn["timestamp"] = ta + 1
                turn["X"], turn["Y"] = pa[0], pa[1]
                turn["pitch"] = pb_pitch
                move = dict(row)
                move["timestamp"] = ta + 2
                out.append(turn)
                out.append(move)
                extra += 1
                split_n += 1
            else:
                out.append(row)
        merged.extend(out)
    merged.sort(key=lambda r: (int(r["timestamp"]), r["name"]))
    return merged, split_n


def write_trajectory_rows(traj_path: Path, rows: List[dict]) -> None:
    traj_path = Path(traj_path)
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    with traj_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRAJ_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "timestamp": r["timestamp"],
                    "name": r["name"],
                    "X": r["X"],
                    "Y": r["Y"],
                    "pitch": r.get("pitch", 0),
                    "loaded": r.get("loaded", "false"),
                    "destination": r.get("destination", ""),
                    "Emergency": r.get("Emergency", "false"),
                    "task-id": r.get("task-id", ""),
                }
            )


def repair_trajectory_file(meta: dict, traj_path: Path) -> dict:
    rows = load_trajectory_rows(traj_path)
    _, env, _, _, _ = load_scenario(
        Path(meta["task_csv"]),
        Path(meta["position_csv"]),
    )
    extras = meta.get("extra_obstacles") or []
    if extras:
        patch_extra_obstacles(env, list(extras))
    hard_walls = get_hard_wall_cells(env)
    lo, hi = _grid_bounds(meta)
    fixed, stats = repair_trajectory_rows(rows, hard_walls=hard_walls, lo=lo, hi=hi)
    try:
        pickups = _load_task_pickups(meta)
        n_scrub, _scrub_tids = scrub_illegal_pickup_rising(fixed, pickups)
        stats["scrubbed_pickup_rising"] = n_scrub
    except Exception:  # noqa: BLE001
        stats["scrubbed_pickup_rising"] = 0
    write_trajectory_rows(traj_path, fixed)
    return stats


def validate_hybrid_trajectory(
    meta: dict,
    traj_path: Path,
    *,
    skip_task_ids: Optional[Set[str]] = None,
) -> dict:
    """Return audit report; ``ok`` is True only when all checks pass."""
    traj_path = Path(traj_path)
    rep: Dict[str, Any] = {
        "scenario_id": meta.get("id"),
        "traj_csv": str(traj_path),
        "ok": False,
        "n_rows": 0,
        "issues": {},
    }
    if not traj_path.exists() or traj_path.stat().st_size < 50:
        rep["issues"]["missing_traj"] = True
        return rep

    rows = load_trajectory_rows(traj_path)
    rep["n_rows"] = len(rows)
    if not rows:
        rep["issues"]["empty_traj"] = True
        return rep

    _, env, _, _, _ = load_scenario(
        Path(meta["task_csv"]),
        Path(meta["position_csv"]),
    )
    extras = meta.get("extra_obstacles") or []
    if extras:
        patch_extra_obstacles(env, list(extras))
    hard_walls = get_hard_wall_cells(env)

    lo, hi = _grid_bounds(meta)
    wall_hits = _check_hard_walls(rows, hard_walls)
    motion = _check_illegal_motion(rows, lo, hi)
    dwell = _check_pickup_unload_dwell(rows)
    dwell_kinds = {"pickup_no_dwell": 0, "unload_no_dwell": 0}
    for e in dwell:
        k = str(e.get("kind") or "")
        if k in dwell_kinds:
            dwell_kinds[k] += 1
    steps_dict = _rows_to_steps_dict(rows)
    conflicts = analyze_trajectory_conflicts(steps_dict)
    task_dropoffs = _load_task_dropoffs(meta)
    task_carry = _check_task_carry(rows, task_dropoffs)
    task_pickups = _load_task_pickups(meta)
    pickup_cell = _check_pickup_at_station(rows, task_pickups)
    skip = set(skip_task_ids or []) | {
        str(x) for x in (meta.get("abandoned_task_ids") or []) if str(x).strip()
    }
    if skip:
        task_carry = [
            e
            for e in task_carry
            if str(e.get("task_id") or "").strip() not in skip
        ]
        pickup_cell = [
            e
            for e in pickup_cell
            if str(e.get("task_id") or "").strip() not in skip
        ]
    carry_kinds = _count_task_carry_kinds(task_carry)
    fifo = analyze_fifo(Path(meta["task_csv"]), traj_path, skip_task_ids=skip)

    rep["issues"] = {
        "hard_wall": wall_hits[:30],
        "n_hard_wall": len(wall_hits),
        "illegal_motion": motion[:30],
        "n_illegal_motion": len(motion),
        "pickup_unload_dwell": dwell[:30],
        "n_pickup_unload_dwell": len(dwell),
        "n_pickup_no_dwell": int(dwell_kinds["pickup_no_dwell"]),
        "n_unload_no_dwell": int(dwell_kinds["unload_no_dwell"]),
        "collisions": conflicts.get("collision_events") or [],
        "n_collisions": int(conflicts.get("collisions") or 0),
        "swaps": conflicts.get("swap_events") or [],
        "n_swaps": int(conflicts.get("swaps") or 0),
        "task_carry_violations": task_carry[:30],
        "n_task_carry_violations": len(task_carry),
        "n_premature_unload": int(carry_kinds.get("premature_unload") or 0),
        "n_foreign_unload": int(carry_kinds.get("foreign_unload") or 0),
        "pickup_cell_violations": pickup_cell[:30],
        "n_pickup_cell_violations": len(pickup_cell),
        "fifo_violations": fifo.get("fifo_violations") or [],
        "n_fifo_violations": len(fifo.get("fifo_violations") or []),
        "display_mismatch": fifo.get("display_mismatch") or [],
        "n_display_mismatch": len(fifo.get("display_mismatch") or []),
    }
    rep["ok"] = (
        rep["issues"]["n_hard_wall"] == 0
        and rep["issues"]["n_illegal_motion"] == 0
        and rep["issues"]["n_pickup_unload_dwell"] == 0
        and rep["issues"]["n_collisions"] == 0
        and rep["issues"]["n_swaps"] == 0
        and rep["issues"]["n_task_carry_violations"] == 0
        and rep["issues"]["n_premature_unload"] == 0
        and rep["issues"]["n_foreign_unload"] == 0
        and rep["issues"]["n_pickup_cell_violations"] == 0
        and rep["issues"]["n_fifo_violations"] == 0
        and rep["issues"]["n_display_mismatch"] == 0
    )
    return rep


def format_validation_summary(rep: dict) -> str:
    if rep.get("ok"):
        return "VALID ok"
    iss = rep.get("issues") or {}
    parts = []
    for key, label in (
        ("n_hard_wall", "wall"),
        ("n_illegal_motion", "motion"),
        ("n_pickup_no_dwell", "pickup_dwell"),
        ("n_unload_no_dwell", "unload_dwell"),
        ("n_pickup_unload_dwell", "action_dwell"),
        ("n_collisions", "collision"),
        ("n_swaps", "swap"),
        ("n_premature_unload", "premature_unload"),
        ("n_foreign_unload", "foreign_unload"),
        ("n_task_carry_violations", "task_carry"),
        ("n_pickup_cell_violations", "pickup_cell"),
        ("n_fifo_violations", "surface_fifo"),
        ("n_display_mismatch", "surface_head"),
    ):
        n = int(iss.get(key) or 0)
        if n:
            # Avoid double-counting task_carry when premature/foreign already shown.
            if key == "n_task_carry_violations" and (
                int(iss.get("n_premature_unload") or 0)
                + int(iss.get("n_foreign_unload") or 0)
                >= n
            ):
                continue
            # Prefer split pickup/unload dwell labels over the aggregate.
            if key == "n_pickup_unload_dwell" and (
                int(iss.get("n_pickup_no_dwell") or 0)
                + int(iss.get("n_unload_no_dwell") or 0)
                >= n
            ):
                continue
            parts.append(f"{label}={n}")
    return "INVALID " + (", ".join(parts) if parts else "unknown")
