"""Patch remaining success-path unload dwell: hold tick before FALSE."""
from pathlib import Path
import ast

p = Path(__file__).with_name("solve_ecbs.py")
text = p.read_text(encoding="utf-8")
hold = (
    "now = _fleet_hold_tick(\n"
    "                                steps_by, pose, names, now, "
    "loaded=loaded, dest=dest, tid=tid\n"
    "                            )\n"
)

patches = []

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
                            _flush_step_off()'''
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
                            _flush_step_off()'''
patches.append(("leftover-parallel", old, new))

# disperse+prio
old = '''                            for a in list(need):
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
new = '''                            for a in list(need):
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
patches.append(("disperse+prio", old, new))

# batch-ST
old = '''                            for a in list(movers_p):
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
new = '''                            for a in list(movers_p):
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
patches.append(("batch-ST", old, new))

# forced unload recover
old = '''                if arrived:
                    _count_done(str(task.get("task_id") or ""))
                    pending_delivery.pop(agv, None)
                    _clear_affinity(agv)
                    # Clear loaded flags for trajectory consistency
                    if steps_by.get(agv):
                        steps_by[agv][-1]["loaded"] = "FALSE"
                        steps_by[agv][-1]["destination"] = ""
                        steps_by[agv][-1]["task-id"] = ""
                    unloaded_any = True'''
new = '''                if arrived:
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
patches.append(("forced-unload", old, new))

for name, o, n in patches:
    c = text.count(o)
    if c != 1:
        raise SystemExit(f"FAIL {name}: count={c}")
    text = text.replace(o, n, 1)
    print("ok", name)

ast.parse(text)
p.write_text(text, encoding="utf-8")
print("AST ok", p.stat().st_size)
print("remaining success-ish FALSE", text.count('[-1]["loaded"] = "FALSE"'))
