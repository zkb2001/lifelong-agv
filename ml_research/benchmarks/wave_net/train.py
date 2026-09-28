"""Bootstrap WaveNet by imitating the rule teacher (synthetic BC)."""
from __future__ import annotations

import argparse
import random
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml_research.common.paths import CKPT

from .features import FEAT_DIM, WaveAgentContext, build_wave_features
from .model import WaveNet, save_wave_net
from .policy import rule_score


def _synth_ctx(rng: random.Random) -> WaveAgentContext:
    return WaveAgentContext(
        agv=f"A{rng.randint(0, 20)}",
        task={"priority": "Urgent" if rng.random() < 0.15 else "Normal"},
        eta=rng.uniform(1, 50),
        work_count=rng.randint(0, 25),
        local_obs=rng.random(),
        conflict_proxy=rng.random(),
        urgency=1.0 if rng.random() < 0.15 else 0.0,
        is_seed=rng.random() < 0.2,
        map_hardness=rng.uniform(0.0, 0.35),
        k_budget=rng.randint(2, 6),
        n_cand=rng.randint(2, 8),
        dist_rank=rng.random(),
        pair_overlap=rng.random(),
        was_idle=rng.random() < 0.3,
    )


def train(
    *,
    steps: int = 4000,
    batch: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    out: Optional[Path] = None,
) -> Path:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = WaveNet(feat_dim=FEAT_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    out = Path(out or (CKPT / "wave_net_v1.pt"))

    for step in range(1, steps + 1):
        xs = []
        ys = []
        for _ in range(batch):
            ctx = _synth_ctx(rng)
            xs.append(build_wave_features(ctx))
            # Soft target: squash rule score into a stable regression range
            ys.append(rule_score(ctx) / 8.0)
        x = torch.from_numpy(np.stack(xs).astype(np.float32))
        y = torch.tensor(ys, dtype=torch.float32)
        pred = net(x)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == 1:
            print(f"[wave_net] step={step} loss={float(loss):.4f}", flush=True)

    path = save_wave_net(net, out, train_steps=steps, seed=seed)
    print(f"[wave_net] saved {path}", flush=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Train WaveNet to imitate rule wave scores")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    train(steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
