"""Lightweight PPO skeleton for ParkArterySemaphoreNet recovery actions."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import CKPT_PATH, ensure_dirs
from .net import ParkArterySemaphoreNet
from .policy import save_ckpt
from .train_bc import make_synthetic_batch


def train_ppo_skeleton(
    *,
    updates: int = 20,
    batch_size: int = 8,
    lr: float = 3e-4,
    seed: int = 0,
    init_ckpt: Optional[Path] = None,
    out_ckpt: Optional[Path] = None,
    device: Optional[str] = None,
) -> Path:
    """PPO-style fine-tune on synthetic teacher-consistency reward.

    Full env rollouts can replace the reward later; this ships the training loop
    and value head usage without requiring a long sim.
    """
    ensure_dirs()
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = ParkArterySemaphoreNet().to(device_t)
    ckpt = Path(init_ckpt or CKPT_PATH)
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device_t, weights_only=False)
        net.load_state_dict(payload["model"], strict=False)
        print(f"[PPO] warm-start {ckpt}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for u in range(int(updates)):
        x, y = make_synthetic_batch(batch_size, seed=seed + u)
        x = x.to(device_t)
        sem_feat = y["sem_feat"].to(device_t)
        y_sem = y["semaphore"].to(device_t)
        out = net(x, sem_feat=sem_feat)
        # treat teacher semaphore as advantage proxy
        logits = out["semaphore_logits"]
        probs = torch.sigmoid(logits)
        # reward: match teacher + small value bootstrap
        reward = 1.0 - (probs - y_sem).abs()
        value = out["value"]
        advantage = (reward - value).detach()
        policy_loss = -(
            torch.log(probs.clamp(1e-6, 1 - 1e-6)) * y_sem
            + torch.log((1 - probs).clamp(1e-6, 1 - 1e-6)) * (1 - y_sem)
        ) * advantage.abs().clamp(max=2.0)
        value_loss = F.mse_loss(value, reward.detach())
        # keep field heads near teacher
        loss_fields = (
            F.binary_cross_entropy_with_logits(
                out["parking_logits"][:, 0], y["parking"].to(device_t)
            )
            + F.binary_cross_entropy_with_logits(
                out["artery_logits"][:, 0], y["artery"].to(device_t)
            )
            + F.binary_cross_entropy_with_logits(
                out["connectivity_logits"][:, 0], y["connectivity"].to(device_t)
            )
        )
        loss = policy_loss.mean() + 0.5 * value_loss + 0.5 * loss_fields
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if u % 5 == 0 or u + 1 == updates:
            print(
                f"[PPO] update={u+1}/{updates} loss={float(loss):.4f} "
                f"v={float(value_loss):.3f} r_mean={float(reward.mean()):.3f}",
                flush=True,
            )

    return save_ckpt(net, Path(out_ckpt or CKPT_PATH), extra={"ppo_updates": int(updates)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    train_ppo_skeleton(
        updates=args.updates,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        init_ckpt=Path(args.init) if args.init else None,
        out_ckpt=Path(args.out) if args.out else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
