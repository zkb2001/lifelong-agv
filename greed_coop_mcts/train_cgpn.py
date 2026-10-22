"""Train IP-CGPN with intra-phase pairwise labels from short-window PP tournaments."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from greed_coop_mcts.cgpn import (
    CGPN_CKPT,
    REL_I_BEFORE,
    REL_NONE,
    ConflictGraphPriorityNet,
    build_agent_matrix,
    labels_from_total_order,
    save_cgpn,
)
from greed_coop_mcts.paths import RESULTS
from greed_coop_mcts.rule_obs import GRID
from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.stage_net import PAIR_FEAT, _manh, build_pairwise_features, region_id

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]
_MAP_CACHE: Dict[int, Set[Cell]] = {}


def _load_blocked(slot: int) -> Optional[Set[Cell]]:
    if slot in _MAP_CACHE:
        return set(_MAP_CACHE[slot])
    meta = load_slot_meta(slot)
    if not meta:
        return None
    blocked: Set[Cell] = set()
    for x in range(GRID):
        blocked.add((x, 0))
        blocked.add((0, x))
    jpath = Path(meta.get("map_json") or "")
    if jpath.is_file():
        data = json.loads(jpath.read_text(encoding="utf-8"))
        for key in ("extra_obstacles",):
            for p in data.get(key) or []:
                if isinstance(p, dict):
                    blocked.add((int(p["x"]), int(p["y"])))
                else:
                    blocked.add((int(p[0]), int(p[1])))
        for p in (data.get("pickups") or []) + (data.get("dropoffs") or []):
            blocked.add((int(p["x"]), int(p["y"])))
    _MAP_CACHE[slot] = blocked
    return set(blocked)


def _free(rng: random.Random, blocked: Set[Cell], used: Optional[Set[Cell]] = None) -> Cell:
    ban = set(blocked) | (used or set())
    for _ in range(400):
        c = (rng.randint(1, GRID - 1), rng.randint(1, GRID - 1))
        if c not in ban:
            return c
    return (1, 1)


def _bfs_path(start: Cell, goal: Cell, blocked: Set[Cell]) -> List[Cell]:
    from collections import deque

    if start == goal:
        return []
    prev: Dict[Cell, Optional[Cell]] = {start: None}
    q = deque([start])
    found = False
    while q:
        cur = q.popleft()
        if cur == goal:
            found = True
            break
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
                continue
            if nxt in blocked or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if not found:
        return []
    path: List[Cell] = []
    cur: Optional[Cell] = goal
    while cur is not None and cur != start:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


def _random_maze(rng: random.Random) -> Set[Cell]:
    blocked: Set[Cell] = set()
    for x in range(GRID):
        blocked.add((x, 0))
        blocked.add((0, x))
    for y in range(1, GRID):
        for x in range(1, GRID):
            if rng.random() < 0.22:
                blocked.add((x, y))
    for _ in range(rng.randint(2, 5)):
        y = rng.randint(1, GRID - 1)
        for x in range(1, GRID):
            blocked.discard((x, y))
    for _ in range(rng.randint(2, 5)):
        x = rng.randint(1, GRID - 1)
        for y in range(1, GRID):
            blocked.discard((x, y))
    return blocked


def _pp_cost(
    order: List[str],
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    paths: Dict[str, List[Cell]],
) -> float:
    cost = 0.0
    reserved: Set[Cell] = set()
    for n in order:
        path = (paths.get(n) or [])[:10]
        for c in path:
            if c in reserved:
                cost += 2.5
            reserved.add(c)
        cur = poses[n][:2]
        cost += 0.04 * float(_manh(cur, goals.get(n, cur)))
    return cost


def _tournament_order(
    rng: random.Random,
    names: List[str],
    poses: Dict[str, Pose],
    goals: Dict[str, Cell],
    paths: Dict[str, List[Cell]],
    loaded: Set[str],
    assigned: Set[str],
    n_trials: int = 10,
) -> List[str]:
    """Best short-horizon PP order; phase bands preserved (loaded → assigned → idle).

    Labels must be learnable from geometry: never random-shuffle after sorting
    (that collapsed CE ≈ ln(3)). Explore with path-prefix jitter + local swaps.
    """

    def phase_key(n: str) -> int:
        return 0 if n in loaded else (1 if n in assigned else 2)

    bands: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    for n in names:
        bands[phase_key(n)].append(n)

    def seed_group(group: List[str]) -> List[str]:
        # Closer-to-goal first, then shorter committed path (less reservation cost).
        return sorted(
            group,
            key=lambda n: (
                _manh(poses[n][:2], goals.get(n, poses[n][:2])),
                len(paths.get(n) or []),
                n,
            ),
        )

    best_order: List[str] = []
    best_cost = 1e18
    for trial in range(max(1, n_trials)):
        order: List[str] = []
        for b in (0, 1, 2):
            group = seed_group(list(bands[b]))
            if trial > 0 and len(group) >= 2:
                # Local swap exploration (keeps geometric signal).
                i = rng.randrange(len(group) - 1)
                group[i], group[i + 1] = group[i + 1], group[i]
                if len(group) >= 3 and rng.random() < 0.4:
                    j = rng.randrange(len(group) - 1)
                    group[j], group[j + 1] = group[j + 1], group[j]
            order.extend(group)
        c = _pp_cost(order, poses, goals, paths)
        if c < best_cost:
            best_cost = c
            best_order = order
    return best_order


def synth_meeting(rng: random.Random, *, max_n: int = 30):
    if rng.random() < 0.80:
        slot = rng.randint(1, 20)
        blocked = _load_blocked(slot) or _random_maze(rng)
    else:
        blocked = _random_maze(rng)
    # Bias toward denser meetings (where IP-CGPN activates) without map-id.
    n = rng.randint(10, max_n) if rng.random() < 0.55 else rng.randint(3, max_n)
    heat = np.zeros((GRID, GRID), dtype=np.float32)
    for _ in range(rng.randint(0, int(6 + 14 * rng.random()))):
        c = _free(rng, blocked)
        heat[c[1], c[0]] = rng.uniform(0.2, 1.0)

    names = [f"a{i}" for i in range(n)]
    poses: Dict[str, Pose] = {}
    goals: Dict[str, Cell] = {}
    paths: Dict[str, List[Cell]] = {}
    loaded: Set[str] = set()
    assigned: Set[str] = set()
    streaks: Dict[str, int] = {}
    used: Set[Cell] = set()
    p_load = rng.uniform(0.20, 0.55)
    p_asg = rng.uniform(0.15, 0.40)
    for nm in names:
        cur = _free(rng, blocked, used)
        used.add(cur)
        final = _free(rng, blocked)
        poses[nm] = (cur[0], cur[1], rng.choice([0, 90, 180, 270]))
        goals[nm] = final
        paths[nm] = _bfs_path(cur, final, blocked)
        r = rng.random()
        if r < p_load:
            loaded.add(nm)
        elif r < p_load + p_asg:
            assigned.add(nm)
        streaks[nm] = rng.randint(0, 28)

    region_occ: Dict[Tuple[int, int], int] = {}
    for nm in names:
        if nm in loaded or nm in assigned:
            rid = region_id(goals[nm])
            region_occ[rid] = int(region_occ.get(rid, 0)) + 1

    order = _tournament_order(
        rng, names, poses, goals, paths, loaded, assigned, n_trials=6
    )
    labels = labels_from_total_order(order, loaded=loaded, assigned=assigned)
    agent = build_agent_matrix(
        names=names,
        poses=poses,
        goals=goals,
        loaded=loaded,
        assigned=assigned,
        blocked_streak=streaks,
        region_occ=region_occ,
        conflict_heat=heat,
    )
    pair = build_pairwise_features(
        names=names,
        poses=poses,
        goals=goals,
        paths=paths,
        loaded=loaded,
        assigned=assigned,
        conflict_heat=heat,
    )
    return agent, pair, labels


def train(
    *,
    steps: int = 5000,
    batch_size: int = 8,
    lr: float = 2e-3,
    device: str = "cpu",
    seed: int = 1,
) -> Path:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = ConflictGraphPriorityNet().to(device)
    opt = AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    running = 0.0
    running_acc = 0.0
    for step in range(1, steps + 1):
        samples = [synth_meeting(rng) for _ in range(batch_size)]
        max_n = max(a.shape[0] for a, _, _ in samples)
        agents, pairs, labels = [], [], []
        for a, p, lab in samples:
            n = a.shape[0]
            ap = np.zeros((max_n, a.shape[1]), dtype=np.float32)
            pp = np.zeros((max_n, max_n, PAIR_FEAT), dtype=np.float32)
            lp = np.full((max_n, max_n), REL_NONE, dtype=np.int64)
            ap[:n] = a
            pp[:n, :n] = p
            lp[:n, :n] = lab
            agents.append(ap)
            pairs.append(pp)
            labels.append(lp)
        a_t = torch.as_tensor(np.stack(agents), device=device)
        p_t = torch.as_tensor(np.stack(pairs), device=device)
        y_t = torch.as_tensor(np.stack(labels), device=device)
        logits = net(a_t, p_t)
        b, n, _, _ = logits.shape
        valid_i = a_t.abs().sum(dim=-1) > 1e-6
        valid = valid_i.unsqueeze(2) & valid_i.unsqueeze(1)
        eye = torch.eye(n, device=device, dtype=torch.bool).unsqueeze(0)
        # Supervise only same-phase labeled pairs (I_BEFORE / J_BEFORE).
        labeled = valid & (~eye) & (y_t != REL_NONE)
        if labeled.sum() < 1:
            continue
        ce = F.cross_entropy(logits[labeled], y_t[labeled], reduction="mean")
        # Mild none-class on cross-phase off-diag (do not drown the main signal).
        unlabeled = valid & (~eye) & (y_t == REL_NONE)
        if unlabeled.sum() > 0:
            none_t = torch.full(
                (int(unlabeled.sum().item()),),
                REL_NONE,
                device=device,
                dtype=torch.long,
            )
            ce = ce + 0.05 * F.cross_entropy(logits[unlabeled], none_t, reduction="mean")
        opt.zero_grad(set_to_none=True)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        with torch.no_grad():
            pred = logits[labeled].argmax(dim=-1)
            acc = float((pred == y_t[labeled]).float().mean().item())
        running = 0.95 * running + 0.05 * float(ce.item()) if step > 1 else float(ce.item())
        running_acc = 0.95 * running_acc + 0.05 * acc if step > 1 else acc
        if step % 250 == 0 or step == 1:
            print(
                f"[train_cgpn] step={step}/{steps} loss={running:.4f} acc={running_acc:.3f}",
                flush=True,
            )

    path = save_cgpn(
        net.cpu(),
        CGPN_CKPT,
        meta={"steps": steps, "seed": seed, "arch": "ip_cgpn_v2"},
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "train_cgpn_v1.json").write_text(
        json.dumps(
            {"ckpt": str(path), "steps": steps, "seed": seed, "final_loss": running},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[train_cgpn] saved {path}", flush=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        seed=int(args.seed),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
