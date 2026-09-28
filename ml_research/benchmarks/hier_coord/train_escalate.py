"""Train EscalateNet to imitate hotspot / map-hardness gate teacher."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ml_research.common.paths import CKPT

from .escalate import (
    EscalateContext,
    EscalateNet,
    build_escalate_features,
    rule_escalate_prob,
    save_escalate_net,
)


def _synth(rng: random.Random) -> EscalateContext:
    return EscalateContext(
        map_hardness=rng.uniform(0.0, 0.35),
        dropoff_score=rng.uniform(0.0, 1.0),
        pickup_score=rng.uniform(0.0, 1.0),
        recent_joint_fail=rng.choice([0, 0, 0, 1, 2]),
        n_assigned=rng.randint(1, 8),
        queue_depth=rng.randint(0, 60),
        top_share=rng.random(),
        assigned_top_n=rng.randint(0, 5),
        was_escalated=rng.random() < 0.4,
        calm_streak=rng.randint(0, 4),
    )


def train(
    *,
    steps: int = 3000,
    batch: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    out: Optional[Path] = None,
) -> Path:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = EscalateNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    out = Path(out or (CKPT / "escalate_net_v1.pt"))

    for step in range(1, steps + 1):
        xs, ys = [], []
        for _ in range(batch):
            ctx = _synth(rng)
            xs.append(build_escalate_features(ctx))
            ys.append(rule_escalate_prob(ctx))
        x = torch.from_numpy(np.stack(xs).astype(np.float32))
        y = torch.tensor(ys, dtype=torch.float32)
        pred = torch.sigmoid(net(x))
        loss = F.binary_cross_entropy(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == 1:
            print(f"[escalate_net] step={step} loss={float(loss):.4f}", flush=True)

    path = save_escalate_net(net, out, train_steps=steps, seed=seed)
    print(f"[escalate_net] saved {path}", flush=True)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    train(steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
