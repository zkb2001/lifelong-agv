"""Bootstrap-train EscalateNetMap + WaveNetMap on real SH maps + synth fleets."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ml_research.benchmarks.coord_custom_ai.hierarchical_gate import (
    dropoff_hotspot_score,
    map_hardness,
)
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.hier_coord.escalate import (
    EscalateContext,
    build_escalate_features,
    rule_escalate_prob,
)
from ml_research.benchmarks.hier_coord.escalate_map import (
    EscalateNetMap,
    save_escalate_net_map,
)
from ml_research.benchmarks.hier_coord.obs import build_hier_map_obs
from ml_research.benchmarks.hier_coord.wave_map import WaveNetMap, save_wave_net_map
from ml_research.benchmarks.wave_net.features import build_wave_features
from ml_research.benchmarks.wave_net.policy import rule_score
from ml_research.benchmarks.wave_net.features import WaveAgentContext

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]


def _load_slot_geometry(slot: int):
    meta = load_custom_meta(slot, max_sim_time=1, wall_timeout=1)
    if not meta:
        return None
    # Lightweight: read obstacles from meta; stations from map json if present
    static: Set[Cell] = set()
    for p in meta.get("extra_obstacles") or []:
        static.add((int(p[0]), int(p[1])))
    stations: Set[Cell] = set()
    mj = meta.get("map_json")
    if mj:
        import json
        from pathlib import Path as P

        j = json.loads(P(mj).read_text(encoding="utf-8"))
        for p in (j.get("pickups") or []) + (j.get("dropoffs") or []):
            stations.add((int(p["x"]), int(p["y"])))
    free = {
        (x, y)
        for x in range(1, 21)
        for y in range(1, 21)
        if (x, y) not in static | stations
    }
    hard = map_hardness(
        extra_obstacles=list(meta.get("extra_obstacles") or []),
        static=static,
        free=free,
    )
    return {
        "slot": slot,
        "static": static,
        "stations": stations,
        "free": sorted(free),
        "hardness": hard,
    }


def _synth_scene(geo: dict, rng: random.Random):
    free: List[Cell] = list(geo["free"])
    stations: List[Cell] = list(geo["stations"]) or free[:8]
    n_agv = rng.randint(4, 8)
    pose: Dict[str, Pose] = {}
    for i in range(n_agv):
        c = rng.choice(free)
        pose[f"A{i}"] = (c[0], c[1], rng.choice([0, 90, 180, 270]))

    n_asg = rng.randint(2, min(6, n_agv))
    names = list(pose.keys())
    rng.shuffle(names)
    assigned = {}
    # hotspot mode sometimes
    hotspot = rng.random() < 0.45
    shared_dest = rng.choice(["Beijing", "Tianjin", "Shanghai", "Dalian"])
    shared_drop = rng.choice(stations) if stations else rng.choice(free)
    for a in names[:n_asg]:
        pk = rng.choice(stations) if stations else rng.choice(free)
        if hotspot:
            dest = shared_dest
            ends = [shared_drop]
        else:
            dest = rng.choice(["Beijing", "Tianjin", "Shanghai", "Dalian", "Xiamen"])
            ends = [rng.choice(stations) if stations else rng.choice(free)]
        assigned[a] = {
            "pickup_point": pk,
            "end_points": ends,
            "destination": dest,
            "priority": "urgent" if rng.random() < 0.15 else "normal",
        }

    # queue leftovers
    queues = {"Tiger": []}
    for _ in range(rng.randint(0, 12)):
        pk = rng.choice(stations) if stations else rng.choice(free)
        queues["Tiger"].append(
            {
                "pickup_point": pk,
                "end_points": [rng.choice(stations) if stations else rng.choice(free)],
                "destination": rng.choice(["Beijing", "Tianjin"]),
            }
        )

    drop_score, drop_meta = dropoff_hotspot_score(assigned, queues)
    pickup_s = float(drop_meta.get("pickup_score") or 0.0)
    ctx = EscalateContext(
        map_hardness=float(geo["hardness"]),
        dropoff_score=float(drop_score),
        pickup_score=pickup_s,
        recent_joint_fail=rng.choice([0, 0, 0, 1, 2]),
        n_assigned=len(assigned),
        queue_depth=sum(len(v) for v in queues.values()),
        top_share=float(drop_meta.get("share") or 0.0),
        assigned_top_n=int(drop_meta.get("assigned_top_n") or 0),
        was_escalated=rng.random() < 0.4,
        calm_streak=rng.randint(0, 4),
    )
    maps = build_hier_map_obs(
        static=geo["static"],
        stations=set(geo["stations"]),
        pose=pose,
        assigned=assigned,
        queues=queues,
    )
    y = rule_escalate_prob(ctx)
    return maps, ctx, y, pose, assigned, geo


def train_escalate(geos: List[dict], *, steps: int, batch: int, lr: float, seed: int) -> Path:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = EscalateNetMap()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for step in range(1, steps + 1):
        xs_m, xs_f, ys = [], [], []
        for _ in range(batch):
            geo = rng.choice(geos)
            maps, ctx, y, *_ = _synth_scene(geo, rng)
            xs_m.append(maps)
            xs_f.append(build_escalate_features(ctx))
            ys.append(y)
        m = torch.from_numpy(np.stack(xs_m).astype(np.float32))
        f = torch.from_numpy(np.stack(xs_f).astype(np.float32))
        yt = torch.tensor(ys, dtype=torch.float32)
        pred = torch.sigmoid(net(m, f))
        loss = F.binary_cross_entropy(pred, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"[escalate_map] step={step} loss={float(loss):.4f}", flush=True)
    return save_escalate_net_map(net, train_steps=steps, seed=seed)


def train_wave(geos: List[dict], *, steps: int, batch: int, lr: float, seed: int) -> Path:
    rng = random.Random(seed + 7)
    torch.manual_seed(seed + 7)
    net = WaveNetMap()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def _bfs(a: Cell, b: Cell, static: Set[Cell]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    for step in range(1, steps + 1):
        loss_acc = []
        for _ in range(batch):
            geo = rng.choice(geos)
            maps, ctx, _y, pose, assigned, g = _synth_scene(geo, rng)
            if len(assigned) < 2:
                continue
            work = {a: rng.randint(0, 8) for a in pose}
            # build contexts manually light
            from ml_research.benchmarks.wave_net.features import build_wave_contexts

            ctxs = build_wave_contexts(
                assigned,
                pose,
                g["static"],
                _bfs,
                work,
                map_hardness=float(g["hardness"]),
                k_budget=rng.randint(2, 6),
            )
            feats = np.stack([build_wave_features(c) for c in ctxs]).astype(np.float32)
            targets = np.array([rule_score(c) for c in ctxs], dtype=np.float32)
            # normalize targets per batch scene
            targets = (targets - targets.mean()) / (targets.std() + 1e-3)
            m = torch.from_numpy(maps.astype(np.float32)).unsqueeze(0)
            f = torch.from_numpy(feats)
            t = torch.from_numpy(targets)
            pred = net(m, f)
            loss = F.mse_loss(pred, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_acc.append(float(loss.detach()))
        if step % 200 == 0 or step == 1:
            print(
                f"[wave_map] step={step} loss={np.mean(loss_acc) if loss_acc else 0:.4f}",
                flush=True,
            )
    return save_wave_net_map(net, train_steps=steps, seed=seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=str, default="1-20")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--escalate-only", action="store_true")
    ap.add_argument("--wave-only", action="store_true")
    args = ap.parse_args()

    slots: List[int] = []
    for part in args.slots.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            slots.extend(range(int(a), int(b) + 1))
        else:
            slots.append(int(part))

    geos = []
    for s in slots:
        g = _load_slot_geometry(s)
        if g and g["free"]:
            geos.append(g)
            print(f"[train_map] slot={s} hard={g['hardness']:.3f} free={len(g['free'])}", flush=True)
    if not geos:
        raise SystemExit("no geometries loaded")

    if not args.wave_only:
        p = train_escalate(geos, steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed)
        print("saved", p, flush=True)
    if not args.escalate_only:
        p = train_wave(geos, steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed)
        print("saved", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
