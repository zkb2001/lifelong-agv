"""Restore pickup/unload dwell via _fleet_hold_tick (VALID dwell rules)."""
from __future__ import annotations

from pathlib import Path

TARGET = Path(__file__).with_name("solve_ecbs.py")


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    orig = text

    if "_fleet_hold_tick" not in text:
        old = '''def _hold(
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


def _build_grid(static: Set[Cell], stations: Set[Cell], W: int = 20, H: int = 20) -> GridMap:'''
        new = '''def _hold(
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


def _build_grid(static: Set[Cell], stations: Set[Cell], W: int = 20, H: int = 20) -> GridMap:'''
        if old not in text:
            raise SystemExit("hold/_build_grid anchor missing")
        text = text.replace(old, new, 1)
        print("patched: _fleet_hold_tick")
    else:
        print("skip: _fleet_hold_tick present")

    old_pickup = '''            tid[agv] = tid_s
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
        assigned = committed'''
    new_pickup = '''            tid[agv] = tid_s
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
                else:
                    _clear_inflight_tid(tid[agv])
            work_count[agv] = int(work_count.get(agv, 0)) + 1
            committed[agv] = task
            committed_stations.add(st_name)
        if committed:
            # One stay tick for all pad arrivals this wave (pickup dwell).
            now = _fleet_hold_tick(
                steps_by, pose, names, now, loaded=loaded, dest=dest, tid=tid
            )
        assigned = committed'''
    if old_pickup in text:
        text = text.replace(old_pickup, new_pickup, 1)
        print("patched: pickup dwell")
    elif "One stay tick for all pad arrivals this wave" in text:
        print("skip: pickup dwell")
    else:
        # maybe _clear_inflight_tid missing - try without else branch
        raise SystemExit("pickup commit anchor missing")

    old_unload = '''        def _unload_arrivals(movers: Set[str]) -> None:
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
            _batch_step_off(cleared)'''
    new_unload = '''        def _unload_arrivals(movers: Set[str]) -> None:
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
                # One shared stay tick: all clear loaded in-place (VALID dwell).
                now = _fleet_hold_tick(
                    steps_by, pose, names, now, loaded=loaded, dest=dest, tid=tid
                )
                _batch_step_off(cleared)'''
    if old_unload in text:
        text = text.replace(old_unload, new_unload, 1)
        print("patched: unload_arrivals dwell")
    elif "One shared stay tick: all clear loaded" in text:
        print("skip: unload_arrivals dwell")
    else:
        raise SystemExit("unload_arrivals anchor missing")

    # ST parallel unload: flip last-step FALSE → defer to fleet hold via step_off buf pattern
    # Replace common pattern of writing FALSE on last step after pad arrival in delivery ST.
    needle = '''                            if pose[agv][:2] in pads:
                                loaded[agv] = False
                                dest[agv] = ""
                                tid[agv] = ""
                                steps_by[agv][-1]["loaded"] = "FALSE"
                                steps_by[agv][-1]["destination"] = ""
                                steps_by[agv][-1]["task-id"] = ""
                                _count_done(str(deliverable[agv]["task_id"]))
                                pending_delivery.pop(agv, None)
                                _clear_after_unload(agv)
                        _flush_step_off()'''
    repl = '''                            if pose[agv][:2] in pads:
                                loaded[agv] = False
                                dest[agv] = ""
                                tid[agv] = ""
                                _count_done(str(deliverable[agv]["task_id"]))
                                pending_delivery.pop(agv, None)
                                _clear_after_unload(agv)
                        if _step_off_buf:
                            now = _fleet_hold_tick(
                                steps_by,
                                pose,
                                names,
                                now,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                            )
                        _flush_step_off()'''
    n = text.count(needle)
    if n:
        text = text.replace(needle, repl)
        print(f"patched: ST parallel unload dwell x{n}")
    else:
        print("warn: ST parallel unload pattern not found (may already differ)")

    # serial unload path often flips last step too
    needle2 = '''                    if arrived:
                        loaded[agv] = False
                        dest[agv] = ""
                        tid[agv] = ""
                        steps_by[agv][-1]["loaded"] = "FALSE"
                        steps_by[agv][-1]["destination"] = ""
                        steps_by[agv][-1]["task-id"] = ""
                        _count_done(str(deliverable[agv]["task_id"]))
                        pending_delivery.pop(agv, None)
                        _clear_affinity(agv)
                        _clear_after_unload(agv)'''
    # Check actual serial unload block
    if "steps_by[agv][-1][\"loaded\"] = \"FALSE\"" in text:
        print("note: remaining last-step FALSE flips:", text.count('steps_by[agv][-1]["loaded"] = "FALSE"'))

    if not hasattr(text, 'replace'):
        pass
    # Ensure _clear_inflight_tid exists if we referenced it
    if "_clear_inflight_tid" in new_pickup and "def _clear_inflight_tid" not in text:
        # pickup used _clear_inflight_tid - check if function exists in file
        if "def _clear_inflight_tid" not in text and "_clear_inflight_tid(" in text:
            print("WARN: _clear_inflight_tid referenced but not defined — checking")
        if "def _clear_inflight_tid" not in text:
            # fallback: remove else branch call
            text2 = text.replace(
                '''                else:
                    _clear_inflight_tid(tid[agv])
''',
                "",
            )
            if text2 != text:
                text = text2
                print("removed _clear_inflight_tid call (undefined)")

    if text == orig:
        print("no changes")
        return
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET} size={TARGET.stat().st_size}")


if __name__ == "__main__":
    main()
