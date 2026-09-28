"""Lifelong MAPD: M0-teacher BC + online RL (reward = unloads)."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml_research.common.paths import RESULTS

from .allocator_transformer import make_transformer_allocator, reset_load
from .collect_lifelong_bc import NPZ, collect_lifelong_bc
from .lifelong_env import DEFAULT_SCHEDULE, load_schedule, make_lifelong_sim, run_lifelong_episode
from .train_bc_ppo import _bc_loss
from .transformer_priority import PriorityTransformer

OUT = RESULTS / "rl_rh_pp"
CKPT = OUT / "priority_transformer_lifelong.pt"
CKPT_BEST = OUT / "priority_transformer_lifelong_best.pt"
CKPT_WARM = OUT / "priority_transformer.pt"


def train_bc_lifelong(
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    batch: int = 32,
    collect: bool = True,
    max_time: int = 3000,
) -> Path:
    if collect or not NPZ.exists():
        collect_lifelong_bc(max_time=max_time)
    z = np.load(NPZ)
    agv_f = torch.as_tensor(z["agv_f"])
    agv_m = torch.as_tensor(z["agv_m"])
    task_f = torch.as_tensor(z["task_f"])
    task_m = torch.as_tensor(z["task_m"])
    teacher = torch.as_tensor(z["teacher"])
    n = int(agv_f.size(0))
    print(f"[LIFELONG-RL] BC n={n}", flush=True)

    net = PriorityTransformer()
    if CKPT_WARM.exists():
        blob = torch.load(CKPT_WARM, map_location="cpu", weights_only=False)
        net.load_state_dict(blob["model"], strict=False)
        print(f"[LIFELONG-RL] warm-start from {CKPT_WARM}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n)
        total = 0.0
        steps = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            L = int((teacher[idx] >= 0).sum(dim=1).max().item())
            L = max(1, min(L, teacher.size(1), 12))
            out = net.forward_bc(
                agv_f[idx], agv_m[idx], task_f[idx], task_m[idx], teacher[idx, :L]
            )
            loss = _bc_loss(out["logits"], teacher[idx, :L])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            steps += 1
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"  [L-BC] ep={ep}/{epochs} loss={total/max(1,steps):.4f}", flush=True)

    meta = {
        "train": "lifelong_bc_m0",
        "n": n,
        "epochs": epochs,
        "seconds": round(time.time() - t0, 1),
    }
    torch.save({"model": net.state_dict(), "meta": meta}, CKPT)
    return CKPT


def _bc_aux_lifelong(net: PriorityTransformer, batch: int = 24) -> torch.Tensor:
    p0 = next(net.parameters())
    if not NPZ.exists():
        return p0.new_tensor(0.0)
    z = np.load(NPZ)
    n = int(z["agv_f"].shape[0])
    idx = np.random.choice(n, size=min(batch, n), replace=False)
    agv_f = torch.as_tensor(z["agv_f"][idx])
    agv_m = torch.as_tensor(z["agv_m"][idx])
    task_f = torch.as_tensor(z["task_f"][idx])
    task_m = torch.as_tensor(z["task_m"][idx])
    teacher = torch.as_tensor(z["teacher"][idx])
    L = int((teacher >= 0).sum(dim=1).max().item())
    L = max(1, min(L, teacher.size(1), 8))
    out = net.forward_bc(agv_f, agv_m, task_f, task_m, teacher[:, :L])
    return _bc_loss(out["logits"], teacher[:, :L])


def _eval_unloads(
    net: PriorityTransformer,
    schedule: list,
    *,
    max_time: int,
    deterministic: bool,
) -> int:
    reset_load()
    net.eval()
    alloc = make_transformer_allocator(net, deterministic=deterministic)
    _mod, sim, _ = make_lifelong_sim(schedule, allocator=alloc)
    _, unloads, _ = run_lifelong_episode(sim, max_time=max_time)
    return int(unloads)


def _rollout_lifelong(
    net: PriorityTransformer,
    schedule: list,
    *,
    max_time: int,
    deterministic: bool,
    collect: bool = False,
):
    reset_load()
    traj: list = []
    if collect:
        net.train()
        alloc = make_transformer_allocator(
            net, deterministic=False, collect_logprob=True, traj_buf=traj
        )
    else:
        net.eval()
        alloc = make_transformer_allocator(net, deterministic=deterministic)
    _mod, sim, _ = make_lifelong_sim(schedule, allocator=alloc)

    prev_u = 0
    n_dec = 0
    step_events: list = []

    with contextlib.redirect_stdout(io.StringIO()):
        while sim.time < max_time:
            sim.time_forward()

    steps = sim.steps_reorganize()
    from .lifelong_env import unloads_from_steps

    unloads = unloads_from_steps(steps)
    if collect and traj:
        bonus = 0.08 * float(unloads)
        share = bonus / max(1, len(traj))
        step_events = [(i, share) for i in range(len(traj))]
    return int(unloads), traj, step_events


def rl_lifelong(
    *,
    updates: int = 40,
    lr: float = 3e-5,
    train_horizon: int = 2500,
    eval_horizon: int = 8000,
    bc_coef: float = 0.35,
    schedule_csv: Path = DEFAULT_SCHEDULE,
    resume: bool = False,
    start_u: int = 1,
) -> Path:
    if resume and CKPT_BEST.exists():
        blob = torch.load(CKPT_BEST, map_location="cpu", weights_only=False)
        print(f"[LIFELONG-RL] resume from {CKPT_BEST} meta={blob.get('meta')}", flush=True)
    elif not CKPT.exists():
        train_bc_lifelong(collect=True)
        blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    else:
        blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = PriorityTransformer()
    net.load_state_dict(blob["model"])

    train_sched = load_schedule(schedule_csv, max_spawn_t=train_horizon)
    eval_sched = load_schedule(schedule_csv, max_spawn_t=eval_horizon)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    meta0 = blob.get("meta") or {}
    if resume and meta0.get("best_unloads_8000") is not None:
        best = int(meta0["best_unloads_8000"])
        start_u = max(start_u, int(meta0.get("update", start_u - 1)) + 1)
    else:
        best = _eval_unloads(net, eval_sched, max_time=eval_horizon, deterministic=True)
        torch.save(
            {
                "model": net.state_dict(),
                "meta": {**meta0, "best_unloads_8000": best, "update": 0},
            },
            CKPT_BEST,
        )
    end_u = start_u + updates - 1
    print(
        f"[LIFELONG-RL] resume={resume} u={start_u}..{end_u} "
        f"train_t={train_horizon} eval_t={eval_horizon} best@8000={best}",
        flush=True,
    )

    log_path = OUT / "train_lifelong_rl.jsonl"
    if not resume and log_path.exists():
        log_path.unlink()

    for u in range(start_u, end_u + 1):
        unloads, traj, step_events = _rollout_lifelong(
            net, train_sched, max_time=train_horizon, deterministic=False, collect=True
        )
        if len(traj) < 2:
            print(f"  [L-RL] u={u} skip dec={len(traj)} unloads={unloads}", flush=True)
            continue

        rewards = [0.0] * len(traj)
        for i, r in step_events:
            if i < len(rewards):
                rewards[i] += r

        rew = torch.tensor(rewards, dtype=torch.float32)
        lps = torch.stack([x["logprob"] for x in traj])
        vals = torch.stack([x["value"] for x in traj])
        ret = torch.zeros_like(rew)
        run = 0.0
        gamma = 0.99
        for i in reversed(range(len(rew))):
            run = float(rew[i]) + gamma * run
            ret[i] = run
        ret = (ret - ret.mean()) / (ret.std() + 1e-6)
        vals_n = (vals - vals.detach().mean()) / (vals.detach().std() + 1e-6)
        adv = ret - vals_n.detach()
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        pg = -(lps * adv.detach()).mean()
        vf = F.mse_loss(vals_n, ret.detach())
        bc_l = _bc_aux_lifelong(net)
        loss = pg + 0.5 * vf + bc_coef * bc_l

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if u == start_u or u % 5 == 0 or u == end_u:
            det = _eval_unloads(net, eval_sched, max_time=eval_horizon, deterministic=True)
            if det > best:
                best = det
                torch.save(
                    {
                        "model": net.state_dict(),
                        "meta": {
                            **meta0,
                            "best_unloads_8000": best,
                            "update": u,
                            "train": "lifelong_bc_rl",
                        },
                    },
                    CKPT_BEST,
                )
            row = {
                "u": u,
                "sto_unloads": unloads,
                "det_unloads_8000": det,
                "best": best,
                "dec": len(traj),
                "loss": float(loss.detach()),
            }
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(
                f"  [L-RL] u={u}/{end_u} sto={unloads} det@8000={det} best={best} "
                f"dec={len(traj)} loss={float(loss.detach()):.4f}",
                flush=True,
            )

    best_blob = torch.load(CKPT_BEST, map_location="cpu", weights_only=False)
    net.load_state_dict(best_blob["model"])
    meta = {
        **meta0,
        "train": "lifelong_bc_rl",
        "best_unloads_8000": best,
        "rl_updates": end_u,
        "eval_horizon": eval_horizon,
    }
    torch.save({"model": net.state_dict(), "meta": meta}, CKPT)
    print(f"[LIFELONG-RL] done best_unloads@8000={best} -> {CKPT}", flush=True)
    return CKPT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-epochs", type=int, default=50)
    ap.add_argument("--collect-time", type=int, default=3000)
    ap.add_argument("--rl-updates", type=int, default=40)
    ap.add_argument("--train-horizon", type=int, default=2500)
    ap.add_argument("--eval-horizon", type=int, default=8000)
    ap.add_argument("--skip-bc", action="store_true")
    ap.add_argument("--skip-rl", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue from lifelong_best ckpt")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    if args.eval_only:
        path = CKPT_BEST if CKPT_BEST.exists() else CKPT
        if not path.exists():
            print(f"missing {path}", flush=True)
            return 1
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net = PriorityTransformer()
        net.load_state_dict(blob["model"])
        sched = load_schedule(DEFAULT_SCHEDULE, max_spawn_t=args.eval_horizon)
        u = _eval_unloads(net, sched, max_time=args.eval_horizon, deterministic=True)
        print(f"eval unloads@{args.eval_horizon}={u} ckpt={path}", flush=True)
        return 0

    if not args.skip_bc and not args.resume:
        train_bc_lifelong(epochs=args.bc_epochs, collect=True, max_time=args.collect_time)
    if not args.skip_rl:
        rl_lifelong(
            updates=args.rl_updates,
            train_horizon=args.train_horizon,
            eval_horizon=args.eval_horizon,
            resume=args.resume,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
