"""Train TERN residual head to match TrafficFlow analytic edge costs."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.optim import AdamW

from greed_coop_mcts.paths import RESULTS
from greed_coop_mcts.rule_obs import GRID, build_rule_state
from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.tern import (
    N_TERN_CH,
    TERN_CKPT,
    TrafficEdgeResidualNet,
    accumulate_path_flows,
    pack_tern_state,
    save_tern,
    traffic_flow_cost_target,
)
from greed_coop_mcts.train_cgpn import _bfs_path, _free, _load_blocked, _random_maze

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]


def synth_tern_sample(rng: random.Random):
    if rng.random() < 0.80:
        slot = rng.randint(1, 20)
        blocked = _load_blocked(slot) or _random_maze(rng)
    else:
        blocked = _random_maze(rng)
    walk = np.ones((GRID, GRID), dtype=np.float32)
    walk[0, :] = 0.0
    walk[:, 0] = 0.0
    for x, y in blocked:
        if 0 <= x < GRID and 0 <= y < GRID:
            walk[y, x] = 0.0

    n = rng.randint(4, 24)
    poses: Dict[str, Pose] = {}
    deliver: Dict[str, Cell] = {}
    pickup: Dict[str, Cell] = {}
    loaded: Set[str] = set()
    paths: Dict[str, List[Cell]] = {}
    used: Set[Cell] = set()
    heat = np.zeros((GRID, GRID), dtype=np.float32)
    for i in range(n):
        nm = f"a{i}"
        cur = _free(rng, blocked, used)
        used.add(cur)
        goal = _free(rng, blocked)
        poses[nm] = (cur[0], cur[1], rng.choice([0, 90, 180, 270]))
        paths[nm] = _bfs_path(cur, goal, blocked)
        if rng.random() < 0.45:
            loaded.add(nm)
            deliver[nm] = goal
        else:
            pickup[nm] = goal
    for _ in range(rng.randint(0, 10)):
        c = _free(rng, blocked)
        heat[c[1], c[0]] = rng.uniform(0.0, 1.0)

    rule_state = build_rule_state(
        blocked=blocked,
        poses=poses,
        deliver_goals=deliver,
        pickup_goals=pickup,
        loaded=loaded,
        conflict_heat=heat,
    )
    fwd, _, vload = accumulate_path_flows(paths, horizon=16)
    # Fake SoftGuidance EMA from path flows.
    occ = vload.copy()
    edge = fwd.copy()
    target = traffic_flow_cost_target(
        walk=walk, flow_fwd=fwd, vertex_load=vload, heat=heat
    )
    # Residual target = full cost - 1.0 free-flow (network predicts Δ).
    delta_tgt = np.clip(target - 1.0, -0.75, 0.75).astype(np.float32)
    state = pack_tern_state(rule_state, occ=occ, edge=edge)
    return state, delta_tgt, walk


def train(
    *,
    steps: int = 3000,
    batch_size: int = 6,
    lr: float = 2e-3,
    device: str = "cpu",
    seed: int = 0,
) -> Path:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = TrafficEdgeResidualNet(channels=N_TERN_CH).to(device)
    opt = AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    net.train()
    running = 0.0
    for step in range(1, steps + 1):
        xs = []
        ys = []
        ws = []
        for _ in range(batch_size):
            st, dt, walk = synth_tern_sample(rng)
            xs.append(st)
            ys.append(np.transpose(dt, (2, 0, 1)))  # (4,H,W)
            ws.append(walk)
        x = torch.as_tensor(np.stack(xs), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.stack(ys), dtype=torch.float32, device=device)
        w = torch.as_tensor(np.stack(ws), dtype=torch.float32, device=device)
        pred = net(x)
        mask = w.unsqueeze(1).expand_as(pred)
        loss = ((pred - y) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        running = 0.95 * running + 0.05 * float(loss.item()) if step > 1 else float(loss.item())
        if step % 200 == 0 or step == 1:
            print(
                f"[train_tern] step={step}/{steps} loss={running:.4f}",
                flush=True,
            )
    path = save_tern(
        net.cpu(),
        TERN_CKPT,
        meta={"steps": steps, "batch_size": batch_size, "seed": seed},
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "train_tern_v1.json").write_text(
        json.dumps(
            {"ckpt": str(path), "steps": steps, "seed": seed, "final_loss": running},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[train_tern] saved {path}", flush=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
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
