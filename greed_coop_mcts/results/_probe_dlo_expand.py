"""Temporarily patch-test expanded DLO via monkeypatch."""
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import greed_coop_mcts.meeting_rh as mrh
from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

ORIG = mrh.legalize_desired


def legalize_wide(**kw):
    # Expand DLO to all fleets with >=10 agents (was 10-14 only).
    names = kw["names"]
    poses = kw["poses"]
    desired = kw["desired"]
    priority = kw["priority"]
    force_stay = kw["force_stay"]
    blocked = kw["blocked"]
    goals = kw.get("goals") or {}
    carrying = kw.get("carrying") or set()
    positions = {n: poses[n][:2] for n in names}

    def _run(order):
        out = dict(desired)
        for i, a in enumerate(names):
            pa = positions[a]
            ca = out.get(a, pa)
            for b in names[i + 1 :]:
                pb = positions[b]
                cb = out.get(b, pb)
                if ca == pb and cb == pa and pa != pb:
                    if float(priority.get(a, 0.0)) >= float(priority.get(b, 0.0)):
                        out[b] = pb
                    else:
                        out[a] = pa
        claim = {}
        for n in order:
            if n in force_stay:
                out[n] = positions[n]
                continue
            cur = positions[n]
            nxt = out.get(n, cur)
            if nxt in blocked and nxt != cur:
                out[n] = cur
                continue
            if nxt == cur:
                continue
            if nxt in claim:
                out[n] = cur
            else:
                claim[nxt] = n
        return out

    def _carrier_sum(out):
        s = 0.0
        for n in carrying:
            if n not in positions:
                continue
            g = goals.get(n, positions[n])
            c = out.get(n, positions[n])
            s += float(abs(c[0] - g[0]) + abs(c[1] - g[1]))
        return s

    order_name = sorted(names, key=lambda a: (-float(priority.get(a, 0.0)), a))
    use_dlo = len(names) >= 10 and len(carrying) >= 2 and bool(goals)
    if not use_dlo:
        return _run(order_name)

    def _manh_key(a):
        cur = positions[a]
        g = goals.get(a, cur)
        return (-float(priority.get(a, 0.0)), abs(cur[0] - g[0]) + abs(cur[1] - g[1]), a)

    order_prog = sorted(names, key=_manh_key)
    out_a = _run(order_name)
    out_b = _run(order_prog)
    if _carrier_sum(out_b) < _carrier_sum(out_a) - 0.5:
        return out_b
    return out_a


mrh.legalize_desired = legalize_wide

FULL = {2: 2021, 8: 1285, 11: 1707, 12: 1558, 13: 832, 19: 1437, 20: 883, 15: 2358}
SLOTS = [2, 8, 12, 13, 19, 11, 15, 20]


def main():
    for slot in SLOTS:
        meta = dict(load_slot_meta(slot, max_sim_time=200000, wall_timeout=600, n_tasks=100))
        meta["id"] = f"SH{slot:02d}_dlox"
        meta["n_tasks"] = 100
        rep = solve(
            slot, meta=meta, policy="rrsc", use_stage_net=True, device="cpu",
            wall_timeout=600, max_sim_time=200000,
        )
        sim = int(rep.get("sim_time") or -1)
        print(
            f"SH{slot:02d} sim={sim} base={FULL[slot]} dlt={sim - FULL[slot]:+d} "
            f"ok={rep.get('validate_ok')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
