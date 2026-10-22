"""Imitation-learn CostTracer from heuristic teacher on synthetic local scenes."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from ml_research.benchmarks.mats_local.network import (
    COSTTRACER_CKPT,
    CostTracerNet,
    save_cost_tracer,
)
from ml_research.benchmarks.mats_local.obs import (
    ACTIONS,
    Cell,
    OBS_RADIUS,
    action_mask,
    build_observation,
)
from ml_research.benchmarks.mats_local.teacher import teacher_action_probs, teacher_value
from ml_research.common.paths import RESULTS

CellT = Tuple[int, int]


def _rand_obstacles(rng: random.Random, n: int, lo: int, hi: int) -> Set[CellT]:
    obs: Set[CellT] = set()
    while len(obs) < n:
        obs.add((rng.randint(lo, hi), rng.randint(lo, hi)))
    return obs


def _free_cell(rng: random.Random, obstacles: Set[CellT], lo: int, hi: int) -> CellT:
    for _ in range(200):
        c = (rng.randint(lo, hi), rng.randint(lo, hi))
        if c not in obstacles:
            return c
    return (lo, lo)


def synth_sample(rng: random.Random) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """One training sample: obs, teacher π, value, mask."""
    lo, hi = 1, 20
    density = rng.uniform(0.08, 0.28)
    n_obs = int(density * (hi - lo + 1) ** 2)
    obstacles = _rand_obstacles(rng, n_obs, lo, hi)
    ego = _free_cell(rng, obstacles, lo, hi)
    goal = _free_cell(rng, obstacles | {ego}, lo, hi)
    n_others = rng.randint(1, 5)
    others: List[CellT] = []
    banned = set(obstacles) | {ego}
    for _ in range(n_others):
        others.append(_free_cell(rng, banned, lo, hi))
        banned.add(others[-1])

    # Corridor stress: plant a 1-wide choke near ego sometimes
    if rng.random() < 0.35:
        x0, y0 = ego
        for dy in range(-2, 3):
            obstacles.add((x0 - 1, y0 + dy))
            obstacles.add((x0 + 1, y0 + dy))
        obstacles.discard(ego)
        obstacles.discard(goal)
        for o in others:
            obstacles.discard(o)

    obs = build_observation(ego, goal, others, obstacles, radius=OBS_RADIUS)
    mask = action_mask(ego, obstacles)
    pi = teacher_action_probs(ego, goal, others, obstacles)
    pi = pi * mask
    z = float(pi.sum())
    if z > 0:
        pi = pi / z
    else:
        pi = mask / max(mask.sum(), 1.0)
    v = teacher_value(ego, goal, others, obstacles)
    return obs, pi.astype(np.float32), float(v), mask.astype(np.float32)


def train(
    *,
    steps: int = 4000,
    batch: int = 64,
    lr: float = 3e-4,
    seed: int = 0,
    device: str = "cpu",
    out: Optional[Path] = None,
) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = CostTracerNet().to(device)
    opt = AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    out = Path(out or COSTTRACER_CKPT)

    losses = []
    for step in range(1, steps + 1):
        obs_b, pi_b, v_b, mask_b = [], [], [], []
        for _ in range(batch):
            o, pi, v, m = synth_sample(rng)
            obs_b.append(o)
            pi_b.append(pi)
            v_b.append(v)
            mask_b.append(m)
        x = torch.as_tensor(np.stack(obs_b), dtype=torch.float32, device=device)
        pi_t = torch.as_tensor(np.stack(pi_b), dtype=torch.float32, device=device)
        v_t = torch.as_tensor(v_b, dtype=torch.float32, device=device)
        m_t = torch.as_tensor(np.stack(mask_b), dtype=torch.float32, device=device)

        net.train()
        logits, value = net(x, m_t)
        logp = F.log_softmax(logits, dim=-1)
        p_loss = -(pi_t * logp).sum(dim=-1).mean()
        v_loss = F.mse_loss(value, v_t)
        loss = p_loss + 0.5 * v_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        losses.append(float(loss.item()))

        if step == 1 or step % 500 == 0 or step == steps:
            # Teacher agreement
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                tgt = pi_t.argmax(dim=-1)
                acc = float((pred == tgt).float().mean().item())
            print(
                f"[mats_local] step={step}/{steps} loss={loss.item():.4f} "
                f"p={p_loss.item():.4f} v={v_loss.item():.4f} acc={acc:.3f}",
                flush=True,
            )

    path = save_cost_tracer(
        net,
        out,
        train_steps=steps,
        batch=batch,
        seed=seed,
        final_loss=float(np.mean(losses[-50:])),
    )
    metrics = {
        "ckpt": str(path),
        "steps": steps,
        "final_loss": float(np.mean(losses[-50:])),
        "n_params": int(sum(p.numel() for p in net.parameters())),
    }
    meta_dir = RESULTS / "mats_local"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "train_il_v1.json"
    meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[mats_local] saved {path} params={metrics['n_params']}", flush=True)
    print(f"[mats_local] metrics -> {meta_path}", flush=True)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="IL train mats_local CostTracer")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        device=device,
        out=Path(args.out) if args.out else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
