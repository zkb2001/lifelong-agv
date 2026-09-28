def _maybe_embed_pipeline_idles(pose, names, idle_park_movers, timelines, cargo, free, static, batch, goals, *, extra_forbidden):
    if not idle_park_movers:
        return None
    forbidden = set((goals.get(a, pose[a][:2]) for a in batch)) | set(extra_forbidden or ()); spare = []
    for n in idle_park_movers:
        if n not in names:
            continue
        match extra_forbidden:
            case 2 as moves if moves <= 1:
                return None
            case _:
                return None
