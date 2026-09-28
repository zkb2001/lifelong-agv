"""Train WaveNetMap v2: real SH maps + true BFS + set-aware oracle + ListNet.

Improvements over train_map_nets.train_wave:
  - Manhattan replaced by real BFS for ETA / ranking features
  - Hard-map oversampling
  - Set-aware oracle teacher (subset cost), not plain rule MSE
  - ListNet + pairwise ranking losses
  - Stronger WaveNetMap head (LayerNorm / dropout)
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ml_research.benchmarks.coord_custom_ai.hierarchical_gate import (
    dropoff_hotspot_score,
    map_hardness,
)
from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.hier_coord.obs import build_hier_map_obs
from ml_research.benchmarks.hier_coord.wave_map import (
    WaveNetMap,
    save_wave_net_map,
)
from ml_research.benchmarks.wave_net.features import (
    WaveAgentContext,
    build_wave_contexts,
    build_wave_features,
)
from ml_research.benchmarks.wave_net.policy import rule_score
from ml_research.common.paths import CKPT

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]

WAVE_MAP_V2 = CKPT / "wave_net_map_v2.pt"


def _bfs_len(start: Cell, goal: Cell, static: Set[Cell]) -> int:
    if start == goal:
        return 0
    blocked = set(static)
    parent = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not (1 <= nxt[0] <= 20 and 1 <= nxt[1] <= 20):
                continue
            if nxt in parent:
                continue
            if nxt in blocked and nxt != goal:
                continue
            parent[nxt] = cur
            q.append(nxt)
    if goal not in parent:
        return 10**6
    n = 0
    cur = goal
    while cur is not None and cur != start:
        n += 1
        cur = parent[cur]
    return n


def _load_slot_geometry(slot: int) -> Optional[dict]:
    meta = load_custom_meta(slot, max_sim_time=1, wall_timeout=1)
    if not meta:
        return None
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


def _bbox_overlap(a0: Cell, a1: Cell, b0: Cell, b1: Cell) -> float:
    ax0, ax1 = min(a0[0], a1[0]), max(a0[0], a1[0])
    ay0, ay1 = min(a0[1], a1[1]), max(a0[1], a1[1])
    bx0, bx1 = min(b0[0], b1[0]), max(b0[0], b1[0])
    by0, by1 = min(b0[1], b1[1]), max(b0[1], b1[1])
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    if ix0 > ix1 or iy0 > iy1:
        return 0.0
    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    area_a = max(1, (ax1 - ax0 + 1) * (ay1 - ay0 + 1))
    area_b = max(1, (bx1 - bx0 + 1) * (by1 - by0 + 1))
    return float(inter) / float(max(area_a, area_b))


def _synth_wave_scene(geo: dict, rng: random.Random) -> Optional[dict]:
    free: List[Cell] = list(geo["free"])
    if len(free) < 8:
        return None
    stations: List[Cell] = list(geo["stations"]) or free[:8]
    n_agv = rng.randint(5, 8)
    pose: Dict[str, Pose] = {}
    for i in range(n_agv):
        c = rng.choice(free)
        pose[f"A{i}"] = (c[0], c[1], rng.choice([0, 90, 180, 270]))

    n_asg = rng.randint(3, min(7, n_agv))
    names = list(pose.keys())
    rng.shuffle(names)
    assigned: Dict[str, dict] = {}
    mode = rng.choice(["hotspot", "mixed", "spread", "urgent_burst"])
    shared_dest = rng.choice(["Beijing", "Tianjin", "Shanghai", "Dalian"])
    shared_drop = rng.choice(stations)
    shared_pick = rng.choice(stations)

    for a in names[:n_asg]:
        if mode == "hotspot":
            pk, dest, ends = shared_pick, shared_dest, [shared_drop]
            pri = "urgent" if rng.random() < 0.25 else "normal"
        elif mode == "urgent_burst":
            pk = rng.choice(stations)
            dest = shared_dest if rng.random() < 0.6 else rng.choice(
                ["Beijing", "Tianjin", "Shanghai", "Dalian", "Xiamen"]
            )
            ends = [shared_drop if dest == shared_dest else rng.choice(stations)]
            pri = "urgent" if rng.random() < 0.55 else "normal"
        elif mode == "mixed":
            pk = shared_pick if rng.random() < 0.35 else rng.choice(stations)
            dest = shared_dest if rng.random() < 0.4 else rng.choice(
                ["Beijing", "Tianjin", "Shanghai", "Dalian", "Xiamen"]
            )
            ends = [shared_drop if dest == shared_dest else rng.choice(stations)]
            pri = "urgent" if rng.random() < 0.2 else "normal"
        else:
            pk = rng.choice(stations)
            dest = rng.choice(["Beijing", "Tianjin", "Shanghai", "Dalian", "Xiamen"])
            ends = [rng.choice(stations)]
            pri = "urgent" if rng.random() < 0.12 else "normal"
        assigned[a] = {
            "pickup_point": pk,
            "end_points": ends,
            "destination": dest,
            "priority": pri,
        }

    queues = {"Tiger": []}
    for _ in range(rng.randint(0, 10)):
        queues["Tiger"].append(
            {
                "pickup_point": rng.choice(stations),
                "end_points": [rng.choice(stations)],
                "destination": rng.choice(["Beijing", "Tianjin", "Shanghai"]),
            }
        )

    maps = build_hier_map_obs(
        static=geo["static"],
        stations=set(geo["stations"]),
        pose=pose,
        assigned=assigned,
        queues=queues,
    )
    k_budget = rng.randint(2, max(2, min(5, len(assigned) - 1)))
    seed_n = rng.randint(1, min(2, len(assigned)))
    seeds = set(rng.sample(list(assigned.keys()), seed_n))
    work = {a: rng.randint(0, 12) for a in pose}
    drop_score, drop_meta = dropoff_hotspot_score(assigned, queues)
    return {
        "maps": maps,
        "pose": pose,
        "assigned": assigned,
        "geo": geo,
        "k_budget": k_budget,
        "seeds": seeds,
        "work": work,
        "drop_score": float(drop_score),
        "top_share": float(drop_meta.get("share") or 0.0),
        "mode": mode,
    }


def _pair_matrix(
    assigned: Dict[str, dict],
    pose: Dict[str, Pose],
) -> Dict[Tuple[str, str], float]:
    items = list(assigned.items())
    mat: Dict[Tuple[str, str], float] = {}
    for i, (a, ta) in enumerate(items):
        pa = pose[a][:2]
        ga = tuple(ta["pickup_point"])
        da = tuple((ta.get("end_points") or [ga])[0])
        for b, tb in items[i + 1 :]:
            pb = pose[b][:2]
            gb = tuple(tb["pickup_point"])
            db = tuple((tb.get("end_points") or [gb])[0])
            ov = 0.55 * _bbox_overlap(pa, ga, pb, gb) + 0.45 * _bbox_overlap(ga, da, gb, db)
            mat[(a, b)] = ov
            mat[(b, a)] = ov
    return mat


def _subset_cost(
    keep: Sequence[str],
    ctx_by: Dict[str, WaveAgentContext],
    pair: Dict[Tuple[str, str], float],
    *,
    top_share: float,
    map_hardness: float,
) -> float:
    if not keep:
        return 1e9
    eta = sum(ctx_by[a].eta for a in keep)
    # unreachable penalty
    eta += 80.0 * sum(1 for a in keep if ctx_by[a].eta >= 10**5)
    ov = 0.0
    for i, a in enumerate(keep):
        for b in keep[i + 1 :]:
            ov += pair.get((a, b), 0.0)
    urg = sum(ctx_by[a].urgency for a in keep)
    seeds = sum(1.0 for a in keep if ctx_by[a].is_seed)
    # same-dest concentration inside keep
    dests = [str(ctx_by[a].task.get("destination") or "") for a in keep]
    share = 0.0
    if dests:
        c = Counter(dests)
        share = max(c.values()) / float(len(dests))
    hotspot_pen = 4.0 * share * (0.4 + map_hardness) * (0.5 + top_share)
    fairness = sum(min(1.0, ctx_by[a].work_count / 20.0) for a in keep)
    return (
        1.0 * eta
        + 10.0 * ov
        + hotspot_pen
        + 1.2 * fairness
        - 6.0 * urg
        - 7.0 * seeds
    )


def _candidate_subsets(
    names: List[str],
    ctx_by: Dict[str, WaveAgentContext],
    k: int,
    rng: random.Random,
) -> List[Tuple[str, ...]]:
    k = max(1, min(k, len(names)))
    scored_rule = sorted(names, key=lambda a: (-rule_score(ctx_by[a]), a))
    scored_eta = sorted(names, key=lambda a: (ctx_by[a].eta, a))
    scored_urg = sorted(
        names,
        key=lambda a: (-ctx_by[a].urgency, -float(ctx_by[a].is_seed), ctx_by[a].eta, a),
    )
    scored_fair = sorted(names, key=lambda a: (ctx_by[a].work_count, ctx_by[a].eta, a))
    scored_lowov = sorted(names, key=lambda a: (ctx_by[a].pair_overlap, ctx_by[a].eta, a))

    cands: List[Tuple[str, ...]] = []
    for ranked in (scored_rule, scored_eta, scored_urg, scored_fair, scored_lowov):
        cands.append(tuple(ranked[:k]))
        # always protect seeds first then fill
        seeds = [a for a in ranked if ctx_by[a].is_seed]
        rest = [a for a in ranked if a not in seeds]
        cands.append(tuple((seeds + rest)[:k]))

    for _ in range(4):
        pool = list(names)
        rng.shuffle(pool)
        seeds = [a for a in names if ctx_by[a].is_seed]
        fill = [a for a in pool if a not in seeds]
        cands.append(tuple((seeds + fill)[:k]))

    # unique
    uniq = []
    seen = set()
    for c in cands:
        if c in seen or len(c) != k:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


def oracle_teacher_scores(
    ctxs: List[WaveAgentContext],
    pose: Dict[str, Pose],
    assigned: Dict[str, dict],
    *,
    k_budget: int,
    top_share: float,
    map_hardness: float,
    rng: random.Random,
) -> np.ndarray:
    """Higher = better to keep. Soft credit from best subsets."""
    names = [c.agv for c in ctxs]
    ctx_by = {c.agv: c for c in ctxs}
    pair = _pair_matrix(assigned, pose)
    subsets = _candidate_subsets(names, ctx_by, k_budget, rng)
    costs = [
        _subset_cost(
            sub,
            ctx_by,
            pair,
            top_share=top_share,
            map_hardness=map_hardness,
        )
        for sub in subsets
    ]
    best = min(costs) if costs else 0.0
    credit = {a: 0.0 for a in names}
    for sub, cost in zip(subsets, costs):
        # soft: near-best subsets still count
        w = float(np.exp(-(cost - best) / max(8.0, abs(best) * 0.05 + 1.0)))
        for a in sub:
            credit[a] += w
    # blend with rule for stability
    out = np.zeros(len(ctxs), dtype=np.float32)
    for i, c in enumerate(ctxs):
        out[i] = 0.65 * credit[c.agv] + 0.35 * rule_score(c)
    return out


def _listnet_loss(pred: torch.Tensor, teacher: torch.Tensor, *, temp: float = 1.0) -> torch.Tensor:
    t = teacher / max(1e-3, temp)
    return -(F.softmax(t, dim=0) * F.log_softmax(pred, dim=0)).sum()


def _pairwise_loss(pred: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    # mean logistic pairwise over all ordered pairs
    n = pred.shape[0]
    if n < 2:
        return pred.new_zeros(())
    loss = pred.new_zeros(())
    cnt = 0
    for i in range(n):
        for j in range(n):
            if teacher[i] <= teacher[j]:
                continue
            # want pred[i] > pred[j]
            loss = loss + F.softplus(-(pred[i] - pred[j]))
            cnt += 1
    if cnt == 0:
        return pred.new_zeros(())
    return loss / float(cnt)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    ra = np.argsort(np.argsort(-a))
    rb = np.argsort(np.argsort(-b))
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def train_wave_v2(
    geos: List[dict],
    *,
    steps: int,
    batch_scenes: int,
    lr: float,
    seed: int,
    out: Path,
    hard_boost: float = 2.5,
) -> Path:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = WaveNetMap(map_dim=48, hidden=96, dropout=0.1, mid_ch=48, head_version="v2")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))

    weights = []
    for g in geos:
        w = 1.0 + hard_boost * float(g["hardness"])
        weights.append(w)
    # normalize
    s = sum(weights)
    weights = [w / s for w in weights]

    best_rho = -1.0
    best_state = None
    running = []

    for step in range(1, steps + 1):
        net.train()
        loss_acc = []
        for _ in range(batch_scenes):
            geo = rng.choices(geos, weights=weights, k=1)[0]
            scene = _synth_wave_scene(geo, rng)
            if scene is None or len(scene["assigned"]) < 2:
                continue
            ctxs = build_wave_contexts(
                scene["assigned"],
                scene["pose"],
                scene["geo"]["static"],
                _bfs_len,
                scene["work"],
                map_hardness=float(scene["geo"]["hardness"]),
                k_budget=int(scene["k_budget"]),
                seed_agvs=scene["seeds"],
            )
            if len(ctxs) < 2:
                continue
            feats_np = np.stack([build_wave_features(c) for c in ctxs]).astype(np.float32)
            teacher = oracle_teacher_scores(
                ctxs,
                scene["pose"],
                scene["assigned"],
                k_budget=int(scene["k_budget"]),
                top_share=float(scene["top_share"]),
                map_hardness=float(scene["geo"]["hardness"]),
                rng=rng,
            )
            m = torch.from_numpy(scene["maps"].astype(np.float32)).unsqueeze(0)
            f_t = torch.from_numpy(feats_np)
            t_t = torch.from_numpy(teacher)
            pred = net(m, f_t)
            loss = 0.7 * _listnet_loss(pred, t_t, temp=1.25) + 0.3 * _pairwise_loss(pred, t_t)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            loss_acc.append(float(loss.detach()))
        sched.step()

        # quick held-out spearman every 100 steps
        if step % 100 == 0 or step == 1:
            net.eval()
            rhos = []
            with torch.no_grad():
                for _ in range(24):
                    geo = rng.choice(geos)
                    scene = _synth_wave_scene(geo, rng)
                    if scene is None or len(scene["assigned"]) < 2:
                        continue
                    ctxs = build_wave_contexts(
                        scene["assigned"],
                        scene["pose"],
                        scene["geo"]["static"],
                        _bfs_len,
                        scene["work"],
                        map_hardness=float(scene["geo"]["hardness"]),
                        k_budget=int(scene["k_budget"]),
                        seed_agvs=scene["seeds"],
                    )
                    if len(ctxs) < 2:
                        continue
                    feats = np.stack([build_wave_features(c) for c in ctxs]).astype(np.float32)
                    teacher = oracle_teacher_scores(
                        ctxs,
                        scene["pose"],
                        scene["assigned"],
                        k_budget=int(scene["k_budget"]),
                        top_share=float(scene["top_share"]),
                        map_hardness=float(scene["geo"]["hardness"]),
                        rng=rng,
                    )
                    m = torch.from_numpy(scene["maps"].astype(np.float32)).unsqueeze(0)
                    f_t = torch.from_numpy(feats)
                    pred = net(m, f_t).cpu().numpy()
                    rhos.append(_spearman(pred, teacher))
            rho = float(np.mean(rhos)) if rhos else 0.0
            mean_loss = float(np.mean(loss_acc)) if loss_acc else 0.0
            running.append(rho)
            print(
                f"[wave_v2] step={step} loss={mean_loss:.4f} spearman={rho:.3f} lr={sched.get_last_lr()[0]:.2e}",
                flush=True,
            )
            if rho >= best_rho:
                best_rho = rho
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    path = save_wave_net_map(
        net,
        out,
        train_steps=steps,
        seed=seed,
        version="v2_wave_map_listnet",
        best_spearman=best_rho,
        map_dim=48,
        hidden=96,
        feat_note="feat[11]=dest_share",
    )
    # also publish as default map ckpt used by policy
    save_wave_net_map(
        net,
        CKPT / "wave_net_map_v1.pt",
        train_steps=steps,
        seed=seed,
        version="v2_wave_map_listnet",
        best_spearman=best_rho,
        map_dim=48,
        hidden=96,
        feat_note="feat[11]=dest_share",
    )
    print(f"[wave_v2] saved {path} (best_spearman={best_rho:.3f})", flush=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Train WaveNetMap v2")
    ap.add_argument("--slots", type=str, default="1-20")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-scenes", type=int, default=8)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(WAVE_MAP_V2))
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
            print(
                f"[wave_v2] slot={s} hard={g['hardness']:.3f} free={len(g['free'])}",
                flush=True,
            )
    if not geos:
        raise SystemExit("no geometries loaded")

    train_wave_v2(
        geos,
        steps=args.steps,
        batch_scenes=args.batch_scenes,
        lr=args.lr,
        seed=args.seed,
        out=Path(args.out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
