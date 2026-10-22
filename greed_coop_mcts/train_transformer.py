"""Heavy imitation learning for MapdTransformer (BFS expert, dense mazes).

Not parameter fiddling: train a CNN+Transformer communication policy until
it agrees with obstacle-aware BFS experts on single- and multi-agent scenes.
"""
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

from greed_coop_mcts.obs import ACTIONS, OBS_RADIUS, action_mask
from greed_coop_mcts.obs_xfmr import (
    bfs_expert_probs,
    bfs_expert_value,
    build_observation_xfmr,
)
from greed_coop_mcts.paths import RESULTS
from greed_coop_mcts.transformer_net import (
    TRANSFORMER_CKPT,
    MapdTransformer,
    save_transformer,
)

Cell = Tuple[int, int]


def _rand_obstacles(rng: random.Random, n: int, lo: int, hi: int) -> Set[Cell]:
    obs: Set[Cell] = set()
    while len(obs) < n:
        obs.add((rng.randint(lo, hi), rng.randint(lo, hi)))
    return obs


def _free(rng: random.Random, banned: Set[Cell], lo: int, hi: int) -> Cell:
    for _ in range(400):
        c = (rng.randint(lo, hi), rng.randint(lo, hi))
        if c not in banned:
            return c
    return (lo, lo)


def _carve_maze(rng: random.Random, lo: int, hi: int) -> Set[Cell]:
    """Dense obstacle field with corridors (SH03-like difficulty)."""
    obs: Set[Cell] = set()
    for y in range(lo, hi + 1):
        for x in range(lo, hi + 1):
            if rng.random() < 0.32:
                obs.add((x, y))
    # punch horizontal / vertical corridors
    for _ in range(rng.randint(3, 6)):
        y = rng.randint(lo, hi)
        for x in range(lo, hi + 1):
            obs.discard((x, y))
    for _ in range(rng.randint(3, 6)):
        x = rng.randint(lo, hi)
        for y in range(lo, hi + 1):
            obs.discard((x, y))
    return obs


def synth_team(rng: random.Random, n_agents: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return obs (N,C,H,W), pi (N,A), v (N,), mask (N,A)."""
    lo, hi = 1, 20
    if rng.random() < 0.55:
        obstacles = _carve_maze(rng, lo, hi)
    else:
        density = rng.uniform(0.10, 0.30)
        n_obs = int(density * (hi - lo + 1) ** 2)
        obstacles = _rand_obstacles(rng, n_obs, lo, hi)

    banned = set(obstacles)
    agents: List[Cell] = []
    goals: List[Optional[Cell]] = []
    for _ in range(n_agents):
        p = _free(rng, banned, lo, hi)
        agents.append(p)
        banned.add(p)
    banned_g = set(obstacles)
    for i in range(n_agents):
        if rng.random() < 0.1:
            goals.append(None)
        else:
            g = _free(rng, banned_g | {agents[i]}, lo, hi)
            goals.append(g)
            banned_g.add(g)

    obs_l, pi_l, v_l, m_l = [], [], [], []
    for i in range(n_agents):
        others = [agents[j] for j in range(n_agents) if j != i]
        o = build_observation_xfmr(
            agents[i], goals[i], others, obstacles, radius=OBS_RADIUS
        )
        m = action_mask(agents[i], obstacles)
        pi = bfs_expert_probs(agents[i], goals[i], others, obstacles)
        pi = pi * m
        z = float(pi.sum())
        pi = pi / z if z > 0 else m / max(m.sum(), 1.0)
        v = bfs_expert_value(agents[i], goals[i], others, obstacles)
        obs_l.append(o)
        pi_l.append(pi.astype(np.float32))
        v_l.append(float(v))
        m_l.append(m.astype(np.float32))
    return (
        np.stack(obs_l),
        np.stack(pi_l),
        np.asarray(v_l, dtype=np.float32),
        np.stack(m_l),
    )


def train(
    *,
    steps: int = 8000,
    batch_teams: int = 8,
    max_agents: int = 6,
    lr: float = 2e-4,
    seed: int = 0,
    device: str = "cpu",
    out: Optional[Path] = None,
) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = MapdTransformer().to(device)
    opt = AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    out = Path(out or TRANSFORMER_CKPT)

    losses: List[float] = []
    accs: List[float] = []
    for step in range(1, steps + 1):
        # Pad variable-N teams into a fixed N_max batch
        n_max = rng.randint(2, max_agents)
        obs_b, pi_b, v_b, m_b, pad_b = [], [], [], [], []
        for _ in range(batch_teams):
            n = rng.randint(1, n_max)
            o, pi, v, m = synth_team(rng, n)
            # pad to n_max
            if n < n_max:
                pad_o = np.zeros((n_max - n, *o.shape[1:]), dtype=np.float32)
                pad_pi = np.zeros((n_max - n, pi.shape[1]), dtype=np.float32)
                pad_v = np.zeros((n_max - n,), dtype=np.float32)
                pad_m = np.zeros((n_max - n, m.shape[1]), dtype=np.float32)
                o = np.concatenate([o, pad_o], axis=0)
                pi = np.concatenate([pi, pad_pi], axis=0)
                v = np.concatenate([v, pad_v], axis=0)
                m = np.concatenate([m, pad_m], axis=0)
                pad = np.array([1.0] * n + [0.0] * (n_max - n), dtype=np.float32)
            else:
                pad = np.ones((n_max,), dtype=np.float32)
            obs_b.append(o)
            pi_b.append(pi)
            v_b.append(v)
            m_b.append(m)
            pad_b.append(pad)

        obs_t = torch.as_tensor(np.stack(obs_b), dtype=torch.float32, device=device)
        pi_t = torch.as_tensor(np.stack(pi_b), dtype=torch.float32, device=device)
        v_t = torch.as_tensor(np.stack(v_b), dtype=torch.float32, device=device)
        m_t = torch.as_tensor(np.stack(m_b), dtype=torch.float32, device=device)
        pad_t = torch.as_tensor(np.stack(pad_b), dtype=torch.float32, device=device)

        net.train()
        logits, values = net.forward_team(obs_t, m_t, pad_t)
        logp = F.log_softmax(logits, dim=-1)
        # only supervise valid agents
        w = pad_t.unsqueeze(-1)
        p_loss = -((pi_t * logp) * w).sum() / w.sum().clamp_min(1.0)
        v_loss = (((values - v_t) ** 2) * pad_t).sum() / pad_t.sum().clamp_min(1.0)
        loss = p_loss + 0.5 * v_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        losses.append(float(loss.item()))

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            tgt = pi_t.argmax(dim=-1)
            correct = ((pred == tgt).float() * pad_t).sum()
            acc = float((correct / pad_t.sum().clamp_min(1.0)).item())
            accs.append(acc)

        if step == 1 or step % 500 == 0 or step == steps:
            print(
                f"[xfmr] step={step}/{steps} loss={loss.item():.4f} "
                f"p={p_loss.item():.4f} v={v_loss.item():.4f} acc={acc:.3f} "
                f"n_max={n_max}",
                flush=True,
            )

    path = save_transformer(
        net,
        out,
        train_steps=steps,
        batch_teams=batch_teams,
        max_agents=max_agents,
        seed=seed,
        final_loss=float(np.mean(losses[-100:])),
        final_acc=float(np.mean(accs[-100:])),
        n_params=int(sum(p.numel() for p in net.parameters())),
    )
    metrics = {
        "ckpt": str(path),
        "steps": steps,
        "final_loss": float(np.mean(losses[-100:])),
        "final_acc": float(np.mean(accs[-100:])),
        "n_params": int(sum(p.numel() for p in net.parameters())),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    meta = RESULTS / "train_transformer_v1.json"
    meta.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[xfmr] saved {path} params={metrics['n_params']} acc={metrics['final_acc']:.3f}", flush=True)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description="IL train MAPD Transformer policy")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-teams", type=int, default=8)
    ap.add_argument("--max-agents", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        steps=args.steps,
        batch_teams=args.batch_teams,
        max_agents=args.max_agents,
        lr=args.lr,
        seed=args.seed,
        device=device,
        out=Path(args.out) if args.out else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
