from pathlib import Path

TARGET = Path(r"d:\compitition\MioVerse\final_version\ml_research\benchmarks\coord_custom_ai\solve_ecbs.py")
text = TARGET.read_text(encoding="utf-8")

HOLD_AFTER = '''                        if _step_off_buf:
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

# Pattern A: agv leftover-parallel / similar with _clear_after_unload
old_a = '''                                if pose[agv][:2] in pads:
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
new_a = '''                                if pose[agv][:2] in pads:
                                    loaded[agv] = False
                                    dest[agv] = ""
                                    tid[agv] = ""
                                    _count_done(str(deliverable[agv]["task_id"]))
                                    pending_delivery.pop(agv, None)
                                    _clear_after_unload(agv)
''' + HOLD_AFTER

# Pattern B: for a in list(...) with affinity
old_b = '''                                if pose[a][:2] in pads:
                                    loaded[a] = False
                                    dest[a] = ""
                                    tid[a] = ""
                                    steps_by[a][-1]["loaded"] = "FALSE"
                                    steps_by[a][-1]["destination"] = ""
                                    steps_by[a][-1]["task-id"] = ""
                                    _count_done(str(deliverable[a]["task_id"]))
                                    pending_delivery.pop(a, None)
                                    _clear_affinity(a)
                                    _clear_after_unload(a)
                            _flush_step_off()'''
new_b = '''                                if pose[a][:2] in pads:
                                    loaded[a] = False
                                    dest[a] = ""
                                    tid[a] = ""
                                    _count_done(str(deliverable[a]["task_id"]))
                                    pending_delivery.pop(a, None)
                                    _clear_affinity(a)
                                    _clear_after_unload(a)
''' + HOLD_AFTER

n = 0
if old_a in text:
    c = text.count(old_a)
    text = text.replace(old_a, new_a)
    n += c
    print("patched A x", c)
if old_b in text:
    c = text.count(old_b)
    text = text.replace(old_b, new_b)
    n += c
    print("patched B x", c)

# serial arrived
old_s = '''                    if arrived:
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
                        _flush_step_off()'''
new_s = '''                    if arrived:
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
                        _flush_step_off()'''
if old_s in text:
    text = text.replace(old_s, new_s)
    n += 1
    print("patched serial arrived")

# forced recover unload: add hold before clear flags on last step
old_r = '''                if arrived:
                    _count_done(str(task.get("task_id") or ""))
                    pending_delivery.pop(agv, None)
                    _clear_affinity(agv)
                    # Clear loaded flags for trajectory consistency
                    if steps_by.get(agv):
                        steps_by[agv][-1]["loaded"] = "FALSE"
                        steps_by[agv][-1]["destination"] = ""
                        steps_by[agv][-1]["task-id"] = ""
                    unloaded_any = True'''
new_r = '''                if arrived:
                    _count_done(str(task.get("task_id") or ""))
                    pending_delivery.pop(agv, None)
                    _clear_affinity(agv)
                    loaded[agv] = False
                    dest[agv] = ""
                    tid[agv] = ""
                    now = _fleet_hold_tick(
                        steps_by, pose, names, now, loaded=loaded, dest=dest, tid=tid
                    )
                    unloaded_any = True'''
if old_r in text:
    text = text.replace(old_r, new_r)
    n += 1
    print("patched recover forced unload")

TARGET.write_text(text, encoding="utf-8")
print("total patches", n, "size", TARGET.stat().st_size)
print("remaining FALSE flips", text.count('[-1]["loaded"] = "FALSE"'))
