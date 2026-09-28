from pathlib import Path
import ast

p = Path(__file__).with_name("solve_ecbs.py")
text = p.read_text(encoding="utf-8")

# leftover-parallel
old = '''                            for agv in list(ok_need):
                                pads = set(
                                    _valid_unload_pads(
                                    deliverable[agv], free, static
                                    )
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
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f"[ECBS] delivery ST leftover-parallel "
                                f"k={len(ok_need)} left={len(need)}",
                                flush=True,
                            )'''
new = '''                            for agv in list(ok_need):
                                pads = set(
                                    _valid_unload_pads(
                                    deliverable[agv], free, static
                                    )
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
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f"[ECBS] delivery ST leftover-parallel "
                                f"k={len(ok_need)} left={len(need)}",
                                flush=True,
                            )'''
assert old in text and text.count(old) == 1
text = text.replace(old, new, 1)
print("leftover ok")

# batch-ST success
old2 = '''                            for a in list(movers_p):
                                pads = set(
                                    _valid_unload_pads(deliverable[a], free, static)
                                )
                                if pose[a][:2] in pads:
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
                            _flush_step_off()
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f"[ECBS] delivery batch-ST k={len(movers_p)} "
                                f"left={len(need)} jam={int(jam_hard)} @t={now}",
                                flush=True,
                            )'''
new2 = '''                            for a in list(movers_p):
                                pads = set(
                                    _valid_unload_pads(deliverable[a], free, static)
                                )
                                if pose[a][:2] in pads:
                                    loaded[a] = False
                                    dest[a] = ""
                                    tid[a] = ""
                                    _count_done(str(deliverable[a]["task_id"]))
                                    pending_delivery.pop(a, None)
                                    _clear_affinity(a)
                                    _clear_after_unload(a)
                            now = _fleet_hold_tick(
                                steps_by, pose, names, now, loaded=loaded, dest=dest, tid=tid
                            )
                            _flush_step_off()
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f"[ECBS] delivery batch-ST k={len(movers_p)} "
                                f"left={len(need)} jam={int(jam_hard)} @t={now}",
                                flush=True,
                            )'''
assert old2 in text and text.count(old2) == 1
text = text.replace(old2, new2, 1)
print("batch-ST ok")

# recover disperse+prio
old3 = '''                            for a in list(need):
                                pads = set(
                                    _valid_unload_pads(deliverable[a], free, static)
                                )
                                if pose[a][:2] in pads:
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
                            _flush_step_off()
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f"[RECOVER] delivery disperse+prio k_left={len(need)} "
                                f"@t={now}",
                                flush=True,
                            )'''
new3 = '''                            for a in list(need):
                                pads = set(
                                    _valid_unload_pads(deliverable[a], free, static)
                                )
                                if pose[a][:2] in pads:
                                    loaded[a] = False
                                    dest[a] = ""
                                    tid[a] = ""
                                    _count_done(str(deliverable[a]["task_id"]))
                                    pending_delivery.pop(a, None)
                                    _clear_affinity(a)
                                    _clear_after_unload(a)
                            now = _fleet_hold_tick(
                                steps_by, pose, names, now, loaded=loaded, dest=dest, tid=tid
                            )
                            _flush_step_off()
                            need = sorted(
                                a for a in deliverable if loaded.get(a, False)
                            )
                            print(
                                f"[RECOVER] delivery disperse+prio k_left={len(need)} "
                                f"@t={now}",
                                flush=True,
                            )'''
assert old3 in text and text.count(old3) == 1
text = text.replace(old3, new3, 1)
print("disperse+prio ok")

ast.parse(text)
p.write_text(text, encoding="utf-8")
print("AST ok", p.stat().st_size)
print("remaining success FALSE", text.count('[-1]["loaded"] = "FALSE"'))
