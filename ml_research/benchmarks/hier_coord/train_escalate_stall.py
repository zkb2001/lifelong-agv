"""Train StallEscalateNet for M0→ECBS handoff (imitate stall teacher + hard cases)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ml_research.common.paths import CKPT, RESULTS
from ml_research.benchmarks.hier_coord.escalate_stall import (
    StallEscalateContext,
    StallEscalateNet,
    build_stall_features,
    save_stall_escalate_net,
    teacher_stall_escalate_prob,
)


def _synth(rng: random.Random, *, hard_bias: bool = False) -> StallEscalateContext:
    if hard_bias:
        # SH06/08/12-like: long wall, stalled completions, mid plateau.
        return StallEscalateContext(
            map_hardness=rng.uniform(0.12, 0.45),
            narrow_cut=rng.uniform(0.2, 1.0),
            done_ratio=rng.uniform(0.05, 0.40),
            plateau_streak=rng.randint(10, 28),
            idle_surface_streak=rng.randint(0, 5),
            assign_fail_streak=rng.randint(0, 30),
            plan_fail_rate=rng.uniform(0.1, 0.7),
            throughput_drop_streak=rng.randint(0, 6),
            wall_tick_streak=rng.randint(0, 5),
            wall_window_hit=rng.random() < 0.55,
            sim_per_task=rng.uniform(30.0, 90.0),
            wall_elapsed_s=rng.uniform(80.0, 400.0),
            has_work=True,
            loaded_ratio=rng.uniform(0.1, 0.7),
        )
    if rng.random() < 0.35:
        # Healthy M0.
        return StallEscalateContext(
            map_hardness=rng.uniform(0.0, 0.18),
            narrow_cut=rng.uniform(0.0, 0.4),
            done_ratio=rng.uniform(0.1, 0.85),
            plateau_streak=rng.randint(0, 6),
            idle_surface_streak=rng.randint(0, 1),
            assign_fail_streak=rng.randint(0, 4),
            plan_fail_rate=rng.uniform(0.0, 0.25),
            throughput_drop_streak=rng.randint(0, 1),
            wall_tick_streak=rng.randint(0, 1),
            wall_window_hit=False,
            sim_per_task=rng.uniform(8.0, 32.0),
            wall_elapsed_s=rng.uniform(5.0, 100.0),
            has_work=True,
            loaded_ratio=rng.uniform(0.2, 0.8),
        )
    return StallEscalateContext(
        map_hardness=rng.uniform(0.0, 0.4),
        narrow_cut=rng.uniform(0.0, 1.0),
        done_ratio=rng.uniform(0.0, 0.95),
        plateau_streak=rng.randint(0, 24),
        idle_surface_streak=rng.randint(0, 6),
        assign_fail_streak=rng.randint(0, 40),
        plan_fail_rate=rng.uniform(0.0, 0.8),
        throughput_drop_streak=rng.randint(0, 8),
        wall_tick_streak=rng.randint(0, 6),
        wall_window_hit=rng.random() < 0.25,
        sim_per_task=rng.uniform(5.0, 100.0),
        wall_elapsed_s=rng.uniform(0.0, 500.0),
        has_work=rng.random() < 0.95,
        loaded_ratio=rng.uniform(0.0, 1.0),
    )


def _labels_from_compare_jsonl(path: Path) -> List[Tuple[StallEscalateContext, float]]:
    """Optional map-level priors from compare_100 hier vs ecbs walls."""
    if not path.exists():
        return []
    by: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by.setdefault(int(r["slot"]), {})[r.get("method_id")] = r
    out: List[Tuple[StallEscalateContext, float]] = []
    rng = random.Random(7)
    for slot, methods in by.items():
        h, e = methods.get("hier"), methods.get("ecbs")
        if not h or not e:
            continue
        hw = float(h.get("wall_seconds") or 0)
        ew = float(e.get("wall_seconds") or 0)
        hs = float(h.get("sim_time") or 0)
        es = float(e.get("sim_time") or 0)
        # High wall vs ECBS → positive stall-escalate mid-run samples.
        if ew > 1 and hw > 1.5 * ew:
            for _ in range(40):
                ctx = _synth(rng, hard_bias=True)
                y = max(0.75, teacher_stall_escalate_prob(ctx))
                out.append((ctx, y))
        # Hier already much better sim+wall → stay M0 samples.
        elif hs > 0 and hs < 0.5 * es and hw < ew:
            for _ in range(30):
                ctx = _synth(rng, hard_bias=False)
                ctx.plateau_streak = rng.randint(0, 7)
                ctx.wall_elapsed_s = rng.uniform(10.0, 80.0)
                ctx.sim_per_task = rng.uniform(8.0, 30.0)
                y = min(0.2, teacher_stall_escalate_prob(ctx))
                out.append((ctx, y))
        del slot
    return out


def train(
    *,
    steps: int = 4000,
    batch: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    out: Optional[Path] = None,
    compare_jsonl: Optional[Path] = None,
) -> Path:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    net = StallEscalateNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    out = Path(out or (CKPT / "escalate_net_stall_v1.pt"))

    compare_path = Path(
        compare_jsonl
        or (
            RESULTS
            / "coord_custom_ai"
            / "compare_100"
            / "results.jsonl"
        )
    )
    priors = _labels_from_compare_jsonl(compare_path)
    print(
        f"[stall_escalate] priors={len(priors)} from {compare_path}",
        flush=True,
    )

    for step in range(1, steps + 1):
        xs, ys = [], []
        for i in range(batch):
            if priors and rng.random() < 0.35:
                ctx, y = priors[rng.randrange(len(priors))]
            else:
                hard = rng.random() < 0.4
                ctx = _synth(rng, hard_bias=hard)
                y = teacher_stall_escalate_prob(ctx)
            xs.append(build_stall_features(ctx))
            ys.append(float(y))
        x = torch.from_numpy(np.stack(xs).astype(np.float32))
        y = torch.tensor(ys, dtype=torch.float32)
        pred = torch.sigmoid(net(x))
        loss = F.binary_cross_entropy(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == 1:
            with torch.no_grad():
                acc = float(((pred.detach() >= 0.5) == (y >= 0.5)).float().mean())
                loss_v = float(loss.detach())
            print(
                f"[stall_escalate] step={step} loss={loss_v:.4f} "
                f"batch_acc@0.5={acc:.3f}",
                flush=True,
            )

    # Quick eval on held-out synth
    net.eval()
    xs, ys = [], []
    for _ in range(512):
        ctx = _synth(rng, hard_bias=rng.random() < 0.5)
        xs.append(build_stall_features(ctx))
        ys.append(teacher_stall_escalate_prob(ctx))
    x = torch.from_numpy(np.stack(xs).astype(np.float32))
    y = torch.tensor(ys, dtype=torch.float32)
    with torch.no_grad():
        p = torch.sigmoid(net(x))
        loss = float(F.binary_cross_entropy(p, y))
        acc = float(((p >= 0.5) == (y >= 0.5)).float().mean())
        # High-recall on hard cases
        hard_mask = y >= 0.7
        if hard_mask.any():
            recall = float(((p >= 0.5) & hard_mask).float().sum() / hard_mask.float().sum())
        else:
            recall = 0.0
    path = save_stall_escalate_net(
        net,
        out,
        train_steps=steps,
        seed=seed,
        eval_loss=loss,
        eval_acc=acc,
        hard_recall=recall,
    )
    print(
        f"[stall_escalate] saved {path} eval_loss={loss:.4f} "
        f"acc={acc:.3f} hard_recall={recall:.3f}",
        flush=True,
    )
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare-jsonl", type=Path, default=None)
    args = ap.parse_args()
    train(
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        out=args.out,
        compare_jsonl=args.compare_jsonl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
