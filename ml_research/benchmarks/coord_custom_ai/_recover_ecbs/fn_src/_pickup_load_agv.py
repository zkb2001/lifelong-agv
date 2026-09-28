def _pickup_load_agv(agv, task, *, loaded, dest, tid, steps_by, pose, names, now):
    loaded[agv] = True; dest[agv] = str(task.get("destination") or ""); tid[agv] = str(task["task_id"])
    
    now += 1
    for n in names:
        steps_by[n].append(_hold(n, pose[n], now, loaded=loaded[n], dest=dest[n], tid=tid[n]))
    return now
