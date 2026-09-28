"""Careful one-shot dwell restore — no loops that can corrupt the file."""
from __future__ import annotations

import ast
from pathlib import Path

TARGET = Path(__file__).with_name("solve_ecbs.py")


def must_replace(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"MISSING anchor: {label}")
    if text.count(old) != 1:
        raise SystemExit(f"NON-UNIQUE anchor ({text.count(old)}): {label}")
    print("ok", label)
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    text = must_replace(
        text,
        '''def _hold(
    name: str,
    pose: Pose,
    t: int,
    *,
    loaded: bool = False,
    dest: str = "",
    tid: str = "",
) -> dict:
    return {
        "timestamp": t,
        "name": name,
        "X": pose[0],
        "Y": pose[1],
        "pitch": int(pose[2]) % 360,
        "loaded": "TRUE" if loaded else "FALSE",
        "destination": dest,
        "Emergency": "FALSE",
        "task-id": tid,
    }


def _build_grid(static: Set[Cell], stations: Set[Cell], W: int = 20, H: int = 20) -> GridMap:''',
        '''def _hold(
    name: str,
    pose: Pose,
    t: int,
    *,
    loaded: bool = False,
    dest: str = "",
    tid: str = "",
) -> dict:
    return {
        "timestamp": t,
        "name": name,
        "X": pose[0],
        "Y": pose[1],
        "pitch": int(pose[2]) % 360,
        "loaded": "TRUE" if loaded else "FALSE",
        "destination": dest,
        "Emergency": "FALSE",
        "task-id": tid,
    }


def _fleet_hold_tick(
    steps_by: Dict[str, List[dict]],
    pose: Dict[str, Pose],
    names: List[str],
    now: int,
    *,
    loaded: Dict[str, bool],
    dest: Dict[str, str],
    tid: Dict[str, str],
) -> int:
    """Advance one sim tick: every AGV stays put (competition action dwell)."""
    now = int(now) + 1
    for n in names:
        ld = bool(loaded.get(n, False))
        steps_by.setdefault(n, []).append(
            _hold(
                n,
                pose[n],
                now,
                loaded=ld,
                dest=str(dest.get(n, "") or "") if ld else "",
                tid=str(tid.get(n, "") or "") if ld else "",
            )
        )
    return now


def _build_grid(static: Set[Cell], stations: Set[Cell], W: int = 20, H: int = 20) -> GridMap:''',
        "fleet_hold_tick",
    )

    text = must_replace(
        text,
        '''            tid[agv] = tid_s
            loaded[agv] = True
            dest[agv] = str(task.get("destination") or "")
            steps_by[agv][-1]["loaded"] = "TRUE"
            steps_by[agv][-1]["destination"] = dest[agv]
            steps_by[agv][-1]["task-id"] = tid[agv]
            if tid[agv]:
                picked_ids.add(tid[agv])
                q_inf = inflight_station.get(st_name) or []
                if q_inf and q_inf[0] == tid[agv]:
                    q_inf.pop(0)
                    if not q_inf:
                        inflight_station.pop(st_name, None)
                    else:
                        inflight_station[st_name] = q_inf
            work_count[agv] = int(work_count.get(agv, 0)) + 1
            committed[agv] = task
            committed_stations.add(st_name)
        assigned = committed''',
        '''            tid[agv] = tid_s
            loaded[agv] = True
            dest[agv] = str(task.get("destination") or "")
            if tid[agv]:
                picked_ids.add(tid[agv])
                q_inf = inflight_station.get(st_name) or []
                if q_inf and q_inf[0] == tid[agv]:
                    q_inf.pop(0)
                    if not q_inf:
                        inflight_station.pop(st_name, None)
                    else:
                        inflight_station[st_name] = q_inf
            work_count[agv] = int(work_count.get(agv, 0)) + 1
            committed[agv] = task
            committed_stations.add(st_name)
        if committed:
            # One stay tick for all pad arrivals this wave (pickup dwell).
            now = _fleet_hold_tick(
                steps_by, pose, names, now, loaded=loaded, dest=dest, tid=tid
            )
        assigned = committed''',
        "pickup_dwell",
    )

    text = must_replace(
        text,
        '''        def _unload_arrivals(movers: Set[str]) -> None:
            cleared: List[str] = []
            for agv in sorted(movers):
                if agv not in deliverable:
                    continue
                ok_pads = set(_valid_unload_pads(deliverable[agv], free, static))
                if pose[agv][:2] not in ok_pads:
                    continue
                loaded[agv] = False
                dest[agv] = ""
                tid[agv] = ""
                steps_by[agv][-1]["loaded"] = "FALSE"
                steps_by[agv][-1]["destination"] = ""
                steps_by[agv][-1]["task-id"] = ""
                _count_done(str(deliverable[agv]["task_id"]))
                pending_delivery.pop(agv, None)
                _clear_affinity(agv)
                cleared.append(agv)
            _batch_step_off(cleared)''',
        '''        def _unload_arrivals(movers: Set[str]) -> None:
            nonlocal now
            cleared: List[str] = []
            for agv in sorted(movers):
                if agv not in deliverable:
                    continue
                ok_pads = set(_valid_unload_pads(deliverable[agv], free, static))
                if pose[agv][:2] not in ok_pads:
                    continue
                loaded[agv] = False
                dest[agv] = ""
                tid[agv] = ""
                _count_done(str(deliverable[agv]["task_id"]))
                pending_delivery.pop(agv, None)
                _clear_affinity(agv)
                cleared.append(agv)
            if cleared:
                now = _fleet_hold_tick(
                    steps_by, pose, names, now, loaded=loaded, dest=dest, tid=tid
                )
                _batch_step_off(cleared)''',
        "unload_arrivals_dwell",
    )

    # First ST parallel after joint fail (unique occurrence with _clear_after_unload only)
    text = must_replace(
        text,
        '''                        for agv in list(movers_b):
                            pads = set(
                                _valid_unload_pads(deliverable[agv], free, static)
                            )
                            if pose[agv][:2] in pads:
                                loaded[agv] = False
                                dest[agv] = ""
                                tid[agv] = ""
                                steps_by[agv][-1]["loaded"] = "FALSE"
                                steps_by[agv][-1]["destination"] = ""
                                steps_by[agv][-1]["task-id"] = ""
                                _count_done(str(deliverable[agv]["task_id"]))
                                pending_delivery.pop(agv, None)
                                _clear_after_unload(agv)
                        _flush_step_off()
                        joint_applied = True
                        print(
                            f"[ECBS] delivery ST parallel k={len(movers_b)}",
                            flush=True,
                            )''',
        '''                        for agv in list(movers_b):
                            pads = set(
                                _valid_unload_pads(deliverable[agv], free, static)
                            )
                            if pose[agv][:2] in pads:
                                loaded[agv] = False
                                dest[agv] = ""
                                tid[agv] = ""
                                _count_done(str(deliverable[agv]["task_id"]))
                                pending_delivery.pop(agv, None)
                                _clear_after_unload(agv)
                        now = _fleet_hold_tick(
                            steps_by, pose, names, now, loaded=loaded, dest=dest, tid=tid
                        )
                        _flush_step_off()
                        joint_applied = True
                        print(
                            f"[ECBS] delivery ST parallel k={len(movers_b)}",
                            flush=True,
                            )''',
        "st_parallel_dwell",
    )

    text = must_replace(
        text,
        '''                    if arrived:
                        loaded[agv] = False
                        dest[agv] = ""
                        tid[agv] = ""
                        steps_by[agv][-1]["loaded"] = "FALSE"
                        steps_by[agv][-1]["destination"] = ""
                        steps_by[agv][-1]["task-id"] = ""
                        _count_done(str(deliverable[agv]["task_id"]))
                        pending_delivery.pop(agv, None)
                        _clear_affinity(agv)
                        _clear_after_unload(agv)
                        _flush_step_off()
                    else:''',
        '''                    if arrived:
                        loaded[agv] = False
                        dest[agv] = ""
                        tid[agv] = ""
                        _count_done(str(deliverable[agv]["task_id"]))
                        pending_delivery.pop(agv, None)
                        _clear_affinity(agv)
                        _clear_after_unload(agv)
                        now = _fleet_hold_tick(
                            steps_by, pose, names, now, loaded=loaded, dest=dest, tid=tid
                        )
                        _flush_step_off()
                    else:''',
        "serial_arrived_dwell",
    )

    ast.parse(text)
    TARGET.write_text(text, encoding="utf-8")
    print("AST ok; wrote", TARGET.stat().st_size)


if __name__ == "__main__":
    main()
