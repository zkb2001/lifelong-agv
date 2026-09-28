"""Fix solve_ecbs.py structure after pending_delivery partial edit."""
from __future__ import annotations

from pathlib import Path

p = Path(__file__).resolve().parents[1] / "coord_custom_ai" / "solve_ecbs.py"
# script lives in hier_coord/; solve_ecbs is sibling under benchmarks
p = Path(r"d:\compitition\MioVerse\final_version\ml_research\benchmarks\coord_custom_ai\solve_ecbs.py")
text = p.read_text(encoding="utf-8")

marker_head = "                reserved_pk.add(pk)\n"
cut = text.find(marker_head)
if cut < 0:
    raise SystemExit("head marker missing")
cut = cut + len(marker_head)

old_prefix = (
    "\n        if not assigned:\n"
    "            continue\n\n"
    "        loaded = {n: False for n in names}\n"
    '        dest = {n: "" for n in names}\n'
    '        tid = {n: "" for n in names}\n'
    "        for agv, task in assigned.items():\n"
    '            tid[agv] = str(task["task_id"])\n\n'
)
mid = text.find(old_prefix, cut)
if mid < 0:
    raise SystemExit("old_prefix missing")

body = text[cut:mid]
lines = body.splitlines(True)
indented = []
for ln in lines:
    if ln.strip() == "":
        indented.append(ln)
    else:
        indented.append("    " + ln)

tail_fix = """
        if not assigned and not pending_delivery:
            continue

        # Apply joint pickup paths when we did not already serial-move.
        if assigned and not pickup_serial_done:
            assert paths is not None
            if _cell_paths_conflict(paths, names) or _pose_timelines_conflict(
                pose, _plan_paths_as_poses(pose, paths)
            ):
                if wave_planner == "astar":
                    print(
                        "[ECBS] pickup baseline conflict after expand -> replan ST",
                        flush=True,
                    )
                    paths2 = _prioritized_st_paths(
                        starts, goals, set(assigned), blocked_plan
                    )
                    if paths2 is None or _cell_paths_conflict(paths2, names):
                        paths2 = _wcbs_paths(starts, goals, set(assigned))
                    if paths2 is None or _cell_paths_conflict(paths2, names):
                        print(
                            "[ECBS] pickup baseline conflict unresolved -> ECBS/serial",
                            flush=True,
                        )
                        for agv, task in list(assigned.items()):
                            goal = _snap_free(
                                tuple(task["pickup_point"]),
                                free,
                                static,
                                prefer=pose[agv][:2],
                            )
                            now, ok = _serial_move_to(
                                pose,
                                steps_by,
                                now,
                                agv,
                                goal,
                                blocked_plan,
                                loaded=loaded,
                                dest=dest,
                                tid=tid,
                            )
                            if not ok:
                                _requeue(queues, order, task)
                                del assigned[agv]
                        if not assigned and not pending_delivery:
                            continue
                        pickup_serial_done = True
                    else:
                        paths = paths2
                        timelines = _plan_paths_as_poses(pose, paths)
                        now = _apply_pose_timelines(
                            steps_by,
                            pose,
                            timelines,
                            now,
                            loaded=loaded,
                            dest=dest,
                            tid=tid,
                        )
                else:
                    print("[ECBS] pickup joint conflict after expand -> serial", flush=True)
                    for agv, task in list(assigned.items()):
                        goal = _snap_free(
                            tuple(task["pickup_point"]), free, static, prefer=pose[agv][:2]
                        )
                        now, ok = _serial_move_to(
                            pose,
                            steps_by,
                            now,
                            agv,
                            goal,
                            blocked_plan,
                            loaded=loaded,
                            dest=dest,
                            tid=tid,
                        )
                        if not ok:
                            _requeue(queues, order, task)
                            del assigned[agv]
                    if not assigned and not pending_delivery:
                        continue
            else:
                timelines = _plan_paths_as_poses(pose, paths)
                now = _apply_pose_timelines(
                    steps_by, pose, timelines, now, loaded=loaded, dest=dest, tid=tid
                )

        for agv, task in assigned.items():
            tid[agv] = str(task["task_id"])
            loaded[agv] = True
            dest[agv] = str(task.get("destination") or "")
            steps_by[agv][-1]["loaded"] = "TRUE"
            steps_by[agv][-1]["destination"] = dest[agv]
            steps_by[agv][-1]["task-id"] = tid[agv]
            work_count[agv] = int(work_count.get(agv, 0)) + 1

"""

rest = text[mid + len(old_prefix) :]
# Drop old conflict-apply + loaded True loop (replaced by tail_fix)
dup_start = rest.find("        if not pickup_serial_done:")
if dup_start < 0:
    raise SystemExit("pickup_serial_done block missing in rest")
dup_end = rest.find("        # ---- delivery ----")
if dup_end < 0:
    raise SystemExit("delivery marker missing")
rest_from_delivery = rest[dup_end:]

# Also patch indented body: on total pickup fail with pending, don't continue away
body_txt = "".join(indented)
body_txt = body_txt.replace(
    """            assigned = kept
            if not assigned:
                pickup_stall_streak += 1
                print(
                    f"[ECBS] FAIL pickup wave @t={now} "
                    f"(drop wave, stall={pickup_stall_streak})",
                    flush=True,
                )
""",
    """            assigned = kept
            if not assigned:
                if pending_delivery:
                    pickup_serial_done = True
                    # keep trying carriers; do not advance stall for empty new claims
                    pass
                else:
                    pickup_stall_streak += 1
                    print(
                        f"[ECBS] FAIL pickup wave @t={now} "
                        f"(drop wave, stall={pickup_stall_streak})",
                        flush=True,
                    )
""",
)

# Fix hard-drop to flush whole station (preserve FIFO for later picks)
body_txt = body_txt.replace(
    """                if pickup_stall_streak >= 5:
                    dropped = 0
                    for st in list(order):
                        q = queues.get(st) or []
                        if not q:
                            continue
                        task = q.pop(0)
                        tid_drop = str(task.get("task_id") or "")
                        if tid_drop:
                            _mark_failed(tid_drop)
                        dropped += 1
                        if not q and st in queues:
                            del queues[st]
                        if dropped >= max(1, min(3, pickup_stall_streak // 2)):
                            break
                    print(
                        f"[ECBS] pickup stall -> hard-drop {dropped} task(s) "
                        f"failed_n={len(failed)}",
                        flush=True,
                    )
""",
    """                if pickup_stall_streak >= 5:
                    dropped = 0
                    stations_flushed = 0
                    for st in list(order):
                        q = queues.get(st) or []
                        if not q:
                            continue
                        # Flush entire station queue: skipping a head then picking
                        # later tids violates FIFO/display validation.
                        for task in list(q):
                            tid_drop = str(task.get("task_id") or "")
                            if tid_drop:
                                _mark_failed(tid_drop)
                            dropped += 1
                        queues[st] = []
                        del queues[st]
                        stations_flushed += 1
                        if stations_flushed >= max(1, min(2, pickup_stall_streak // 5)):
                            break
                    print(
                        f"[ECBS] pickup stall -> flush-station drop={dropped} "
                        f"failed_n={len(failed)}",
                        flush=True,
                    )
""",
)

# Fix continue after stall when pending exists — the continue at end of fail branch
body_txt = body_txt.replace(
    """                    queues = {k: v for k, v in queues.items() if v}
                    break
                continue
            pickup_serial_done = True
            pickup_stall_streak = 0
""",
    """                    queues = {k: v for k, v in queues.items() if v}
                    pending_delivery.clear()
                    break
                if not pending_delivery:
                    continue
                pickup_serial_done = True
            if assigned:
                pickup_serial_done = True
                pickup_stall_streak = 0
""",
)

new_text = text[:cut] + body_txt + tail_fix + rest_from_delivery

# Patch delivery miss: keep cargo / pending instead of requeue pickup
old_miss = '''                if not arrived:
                    # Requeue a few times; then hard-fail to avoid infinite loops.
                    tid_miss = str(deliverable[agv]["task_id"])
                    delivery_miss_count[tid_miss] = int(
                        delivery_miss_count.get(tid_miss, 0)
                    ) + 1
                    if delivery_miss_count[tid_miss] <= 5:
                        print(
                            f"[ECBS] delivery miss {tid_miss} @t={now} -> requeue "
                            f"(try={delivery_miss_count[tid_miss]})",
                            flush=True,
                        )
                        recent_joint_fail += 1
                        _requeue(queues, order, deliverable[agv])
                        loaded[agv] = False
                        steps_by[agv][-1]["loaded"] = "FALSE"
                        steps_by[agv][-1]["destination"] = ""
                        steps_by[agv][-1]["task-id"] = ""
                        continue
                    _mark_failed(tid_miss)
                    recent_joint_fail += 1
                    print(
                        f"[ECBS] FAIL delivery {tid_miss} @t={now} "
                        f"(after {delivery_miss_count[tid_miss]} misses)",
                        flush=True,
                    )
                    loaded[agv] = False
                    steps_by[agv][-1]["loaded"] = "FALSE"
                    steps_by[agv][-1]["destination"] = ""
                    steps_by[agv][-1]["task-id"] = ""
                    continue
'''
new_miss = '''                if not arrived:
                    # Keep cargo on AGV and retry unload next wave.
                    # Re-pickup after a loaded rising edge breaks FIFO/display.
                    tid_miss = str(deliverable[agv]["task_id"])
                    delivery_miss_count[tid_miss] = int(
                        delivery_miss_count.get(tid_miss, 0)
                    ) + 1
                    if delivery_miss_count[tid_miss] <= 5:
                        print(
                            f"[ECBS] delivery miss {tid_miss} @t={now} -> retry-carry "
                            f"(try={delivery_miss_count[tid_miss]})",
                            flush=True,
                        )
                        recent_joint_fail += 1
                        pending_delivery[agv] = deliverable[agv]
                        loaded[agv] = True
                        dest[agv] = str(deliverable[agv].get("destination") or "")
                        tid[agv] = tid_miss
                        steps_by[agv][-1]["loaded"] = "TRUE"
                        steps_by[agv][-1]["destination"] = dest[agv]
                        steps_by[agv][-1]["task-id"] = tid_miss
                        continue
                    _mark_failed(tid_miss)
                    recent_joint_fail += 1
                    print(
                        f"[ECBS] FAIL delivery {tid_miss} @t={now} "
                        f"(after {delivery_miss_count[tid_miss]} misses)",
                        flush=True,
                    )
                    pending_delivery.pop(agv, None)
                    loaded[agv] = False
                    steps_by[agv][-1]["loaded"] = "FALSE"
                    steps_by[agv][-1]["destination"] = ""
                    steps_by[agv][-1]["task-id"] = ""
                    continue
'''
if old_miss not in new_text:
    raise SystemExit("delivery miss block not found")
new_text = new_text.replace(old_miss, new_miss)

# Delivery section: merge pending into deliverable sources
old_del_head = '''        # ---- delivery ----
        starts = {n: pose[n][:2] for n in names}
        goals = {n: starts[n] for n in names}
        deliverable: Dict[str, dict] = {}
        reserved_goals: Set[Cell] = set()
        unique_movers: Set[str] = set()
        for agv, task in sorted(assigned.items()):
            # Unique pad per joint wave; NEVER use maze-wall end_points as goals.
            walk = _valid_unload_pads(task, free, static, prefer=starts[agv])
            if not walk:
                _mark_failed(str(task["task_id"]))
                print(
                    f"[ECBS] no free unload pad for {task['task_id']} -> fail",
                    flush=True,
                )
                continue
'''
new_del_head = '''        # ---- delivery ----
        starts = {n: pose[n][:2] for n in names}
        goals = {n: starts[n] for n in names}
        deliverable: Dict[str, dict] = {}
        reserved_goals: Set[Cell] = set()
        unique_movers: Set[str] = set()
        delivery_agents = dict(pending_delivery)
        delivery_agents.update(assigned)
        for agv, task in sorted(delivery_agents.items()):
            # Unique pad per joint wave; NEVER use maze-wall end_points as goals.
            walk = _valid_unload_pads(task, free, static, prefer=starts[agv])
            if not walk:
                _mark_failed(str(task["task_id"]))
                print(
                    f"[ECBS] no free unload pad for {task['task_id']} -> fail",
                    flush=True,
                )
                pending_delivery.pop(agv, None)
                loaded[agv] = False
                steps_by[agv][-1]["loaded"] = "FALSE"
                steps_by[agv][-1]["destination"] = ""
                steps_by[agv][-1]["task-id"] = ""
                continue
'''
if old_del_head not in new_text:
    raise SystemExit("delivery head not found")
new_text = new_text.replace(old_del_head, new_del_head)

# On successful unload, clear pending
# Replace _count_done + _clear_after_unload patterns to also pop pending
new_text = new_text.replace(
    "                                _count_done(str(deliverable[agv][\"task_id\"]))\n"
    "                                _clear_after_unload(agv)",
    "                                _count_done(str(deliverable[agv][\"task_id\"]))\n"
    "                                pending_delivery.pop(agv, None)\n"
    "                                _clear_after_unload(agv)",
)
new_text = new_text.replace(
    "                    _count_done(str(deliverable[agv][\"task_id\"]))\n"
    "                    _clear_after_unload(agv)",
    "                    _count_done(str(deliverable[agv][\"task_id\"]))\n"
    "                    pending_delivery.pop(agv, None)\n"
    "                    _clear_after_unload(agv)",
)
new_text = new_text.replace(
    "                            _count_done(str(deliverable[agv][\"task_id\"]))\n"
    "                            _clear_after_unload(agv)",
    "                            _count_done(str(deliverable[agv][\"task_id\"]))\n"
    "                            pending_delivery.pop(agv, None)\n"
    "                            _clear_after_unload(agv)",
)
new_text = new_text.replace(
    "                                    _count_done(str(deliverable[agv][\"task_id\"]))\n"
    "                                    _clear_after_unload(agv)",
    "                                    _count_done(str(deliverable[agv][\"task_id\"]))\n"
    "                                    pending_delivery.pop(agv, None)\n"
    "                                    _clear_after_unload(agv)",
)
new_text = new_text.replace(
    "                _count_done(str(deliverable[agv][\"task_id\"]))\n"
    "                _clear_after_unload(agv)",
    "                _count_done(str(deliverable[agv][\"task_id\"]))\n"
    "                pending_delivery.pop(agv, None)\n"
    "                _clear_after_unload(agv)",
)

# Syntax check the stall branch: when pending and not assigned after serial fail,
# we must not fall into `continue` without fixing control flow.
# Read the modified fail branch carefully by compiling.

p.write_text(new_text, encoding="utf-8")
compile(new_text, str(p), "exec")
print("OK wrote", p, "bytes", len(new_text))
