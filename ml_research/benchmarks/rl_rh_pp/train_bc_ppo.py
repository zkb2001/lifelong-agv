"""BC (+ online RL) for PriorityTransformer.

BC clones M2 teacher (cannot beat it alone). Online RL optimizes throughput
with BC regularization so the policy does not collapse.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ml_research.common.paths import RESULTS

from .collect_bc import collect_snapshots
from .transformer_priority import PriorityTransformer

OUT = RESULTS / "rl_rh_pp"
CKPT = OUT / "priority_transformer.pt"
CKPT_BEST = OUT / "priority_transformer_best.pt"
OUT.mkdir(parents=True, exist_ok=True)


def _bc_loss(logits: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    loss = logits.new_tensor(0.0)
    n = 0
    for t in range(logits.size(1)):
        y = teacher[:, t]
        valid = y >= 0
        if valid.any():
            loss = loss + F.cross_entropy(logits[valid, t], y[valid])
            n += 1
    return loss / max(1, n)


def train_bc(
    *,
    epochs: int = 60,
    lr: float = 1e-3,
    batch: int = 32,
    max_snaps: int = 400,
    collect: bool = True,
) -> Path:
    if collect or not (OUT / "bc_demos.npz").exists():
        collect_snapshots(max_snaps=max_snaps)
    z = np.load(OUT / "bc_demos.npz")
    agv_f = torch.as_tensor(z["agv_f"])
    agv_m = torch.as_tensor(z["agv_m"])
    task_f = torch.as_tensor(z["task_f"])
    task_m = torch.as_tensor(z["task_m"])
    teacher = torch.as_tensor(z["teacher"])
    n = int(agv_f.size(0))
    print(f"[RL-RH-PP] BC dataset n={n}", flush=True)
    if n < 4:
        raise RuntimeError("too few demos")

    net = PriorityTransformer()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    log_path = OUT / "train_bc.jsonl"
    if log_path.exists():
        log_path.unlink()
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
        row = {"epoch": ep, "loss": total / max(1, steps)}
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"  [BC] ep={ep}/{epochs} loss={row['loss']:.4f}", flush=True)

    net.eval()
    with torch.no_grad():
        out = net.forward_bc(agv_f, agv_m, task_f, task_m, teacher[:, :1])
        pred = out["logits"][:, 0].argmax(-1)
        y0 = teacher[:, 0]
        valid = y0 >= 0
        acc = float((pred[valid] == y0[valid]).float().mean()) if valid.any() else 0.0
    meta = {
        "train": "bc_m2_teacher",
        "n": n,
        "epochs": epochs,
        "top1_acc_step0": acc,
        "seconds": round(time.time() - t0, 1),
        "arch": "PriorityTransformer_d64_h4_e2_d2",
    }
    torch.save({"model": net.state_dict(), "meta": meta}, CKPT)
    (OUT / "train_bc_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[RL-RH-PP] BC done acc@0={acc:.3f} -> {CKPT}", flush=True)
    return CKPT


def _bc_aux_loss(net: PriorityTransformer, batch: int = 32) -> torch.Tensor:
    path = OUT / "bc_demos.npz"
    p0 = next(net.parameters())
    if not path.exists():
        return p0.new_tensor(0.0)
    z = np.load(path)
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


def _setup_sim_like_runner(mod, sim) -> None:
    from ml_research.benchmarks.common import patch_moving_obstacle_horizon

    del mod  # unused; kept for call-site signature
    patch_moving_obstacle_horizon(sim)


def _rollout(net, *, max_time: int, deterministic: bool, collect: bool = False):
    import contextlib
    import io

    from ml_research.benchmarks.allocators import patch_allocator
    from ml_research.benchmarks.common import load_scenario
    from ml_research.common.paths import POSITION_CSV, TASK_CSV

    from .allocator_transformer import make_transformer_allocator, reset_load

    mod, env, agv_states, task_states, n_tasks = load_scenario(TASK_CSV, POSITION_CSV)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = mod.Simulation(
            agv_states, task_states, env, getattr(mod, "DEFAULT_GRID_SIZE", (20, 20))
        )
    _setup_sim_like_runner(mod, sim)
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
    patch_allocator(sim, alloc)

    prev_left = sum(len(v) for v in sim.task_states.values())
    n_dec_before = 0
    step_events = []

    for _t in range(max_time):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sim.time_forward()
        except Exception:
            break
        left = sum(len(v) for v in sim.task_states.values())
        done_delta = max(0, prev_left - left)
        prev_left = left
        r = float(done_delta) * 1.0 - 0.001
        if collect:
            n_now = len(traj)
            if n_now > n_dec_before:
                step_events.append((n_dec_before, n_now, r))
                n_dec_before = n_now
            elif done_delta > 0 and n_dec_before > 0:
                step_events.append((n_dec_before - 1, n_dec_before, r))
        if left <= 0:
            break

    done = int(n_tasks - sum(len(v) for v in sim.task_states.values()))
    return done, n_tasks, traj, step_events


def light_ppo(
    *,
    updates: int = 60,
    lr: float = 5e-5,
    max_time: int = 1800,
    gamma: float = 0.99,
    bc_coef: float = 0.4,
    ent_coef: float = 0.005,
    eval_every: int = 5,
) -> Path:
    """Online RL; checkpoint by deterministic completion (matches official eval)."""
    if not CKPT.exists():
        train_bc(epochs=40, collect=True)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = PriorityTransformer()
    net.load_state_dict(blob["model"])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    print(
        f"[RL-RH-PP] online RL updates={updates} max_time={max_time} "
        f"bc_coef={bc_coef} lr={lr} (best by deterministic eval)",
        flush=True,
    )

    d0, n_tasks, _, _ = _rollout(net, max_time=max_time, deterministic=True)
    best_det = int(d0)
    torch.save(
        {
            "model": net.state_dict(),
            "meta": {**(blob.get("meta") or {}), "best_det": best_det, "update": 0},
        },
        CKPT_BEST,
    )
    print(f"  [RL] init deterministic done={d0}/{n_tasks}", flush=True)

    for u in range(1, updates + 1):
        done, n_tasks, traj, step_events = _rollout(
            net, max_time=max_time, deterministic=False, collect=True
        )
        if len(traj) < 2:
            print(f"  [RL] u={u} skip (decisions={len(traj)}) done={done}", flush=True)
            continue

        rewards = [0.0] * len(traj)
        for i0, i1, r in step_events:
            share = r / max(1, i1 - i0)
            for i in range(i0, i1):
                rewards[i] += share
        bonus = 0.02 * float(done)
        rewards = [r + bonus / max(1, len(rewards)) for r in rewards]

        rew = torch.tensor(rewards, dtype=torch.float32)
        vals = torch.stack([x["value"] for x in traj])
        lps = torch.stack([x["logprob"] for x in traj])
        ret = torch.zeros_like(rew)
        run = 0.0
        for i in reversed(range(len(rew))):
            run = float(rew[i]) + gamma * run
            ret[i] = run
        # normalize returns to stabilize value loss
        ret = (ret - ret.mean()) / (ret.std() + 1e-6)
        vals_n = (vals - vals.detach().mean()) / (vals.detach().std() + 1e-6)

        adv = ret - vals_n.detach()
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        pg = -(lps * adv.detach()).mean()
        vf = F.mse_loss(vals_n, ret.detach())
        bc_l = _bc_aux_loss(net)
        ent_bonus = -ent_coef * lps.mean()
        loss = pg + 0.5 * vf + bc_coef * bc_l + ent_bonus

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if u == 1 or u % eval_every == 0 or u == updates:
            det_done, _, _, _ = _rollout(net, max_time=max_time, deterministic=True)
            if int(det_done) > best_det:
                best_det = int(det_done)
                torch.save(
                    {
                        "model": net.state_dict(),
                        "meta": {
                            **(blob.get("meta") or {}),
                            "best_det": best_det,
                            "update": u,
                            "train": "bc_then_online_rl",
                        },
                    },
                    CKPT_BEST,
                )
            print(
                f"  [RL] u={u}/{updates} sto={done} det={det_done}/{n_tasks} "
                f"dec={len(traj)} loss={float(loss.detach()):.4f} "
                f"pg={float(pg.detach()):.4f} bc={float(bc_l.detach()):.4f} "
                f"best_det={best_det}",
                flush=True,
            )

    best_blob = torch.load(CKPT_BEST, map_location="cpu", weights_only=False)
    net.load_state_dict(best_blob["model"])
    meta = {
        **(blob.get("meta") or {}),
        "ppo_updates": updates,
        "train": "bc_then_online_rl",
        "best_det_in_rl": best_det,
        "rl_max_time": max_time,
        "bc_coef": bc_coef,
        "source": "best_det_ckpt",
    }
    torch.save({"model": net.state_dict(), "meta": meta}, CKPT)
    print(f"[RL-RH-PP] RL done best_det={best_det} -> {CKPT}", flush=True)
    return CKPT


def _sil_snaps_from_traj(traj: list) -> list:
    out = []
    for x in traj:
        if "agv_f" not in x or "teacher" not in x:
            continue
        y = np.asarray(x["teacher"])
        if int((y >= 0).sum()) < 1:
            continue
        out.append(
            {
                "agv_f": x["agv_f"],
                "agv_m": x["agv_m"],
                "task_f": x["task_f"],
                "task_m": x["task_m"],
                "teacher": y,
            }
        )
    return out


def _sil_epoch(net: PriorityTransformer, opt, snaps: list, *, batch: int = 32) -> float:
    if len(snaps) < 2:
        return 0.0
    net.train()
    perm = np.random.permutation(len(snaps))
    total = 0.0
    steps = 0
    for i in range(0, len(snaps), batch):
        chunk = [snaps[int(j)] for j in perm[i : i + batch]]
        agv_f = torch.as_tensor(np.stack([s["agv_f"] for s in chunk]))
        agv_m = torch.as_tensor(np.stack([s["agv_m"] for s in chunk]))
        task_f = torch.as_tensor(np.stack([s["task_f"] for s in chunk]))
        task_m = torch.as_tensor(np.stack([s["task_m"] for s in chunk]))
        teacher = torch.as_tensor(np.stack([s["teacher"] for s in chunk]))
        L = int((teacher >= 0).sum(dim=1).max().item())
        L = max(1, min(L, teacher.size(1), 12))
        out = net.forward_bc(agv_f, agv_m, task_f, task_m, teacher[:, :L])
        loss = _bc_loss(out["logits"], teacher[:, :L])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        total += float(loss.item())
        steps += 1
    return total / max(1, steps)


def train_sil(
    *,
    collect_eps: int = 40,
    rounds: int = 8,
    sil_epochs: int = 6,
    min_done: int = 88,
    max_time: int = 2000,
    lr: float = 1e-4,
    keep_snaps: int = 800,
) -> Path:
    """Self-imitation: clone high-completion stochastic rollouts into greedy policy."""
    if not CKPT.exists():
        train_bc(epochs=40, collect=True)
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = PriorityTransformer()
    net.load_state_dict(blob["model"])
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    d0, n_tasks, _, _ = _rollout(net, max_time=max_time, deterministic=True)
    best_det = int(d0)
    torch.save(
        {
            "model": net.state_dict(),
            "meta": {**(blob.get("meta") or {}), "best_det": best_det, "train": "sil"},
        },
        CKPT_BEST,
    )
    print(
        f"[RL-RH-PP] SIL rounds={rounds} collect={collect_eps}/round "
        f"min_done={min_done} init_det={d0}/{n_tasks}",
        flush=True,
    )

    buffer: list = []
    n_kept_eps = 0
    best_sto = -1
    log_path = OUT / "train_sil.jsonl"
    if log_path.exists():
        log_path.unlink()

    for r in range(1, rounds + 1):
        kept_this = 0
        sto_scores = []
        round_eps: list = []  # (done, snaps)
        for _e in range(collect_eps):
            done, n_tasks, traj, _ = _rollout(
                net, max_time=max_time, deterministic=False, collect=True
            )
            snaps = _sil_snaps_from_traj(traj)
            sto_scores.append(int(done))
            best_sto = max(best_sto, int(done))
            round_eps.append((int(done), snaps))
        thresh = max(int(min_done), best_det)
        for done, snaps in round_eps:
            if done >= thresh and snaps:
                buffer.extend(snaps)
                n_kept_eps += 1
                kept_this += 1
        if kept_this == 0:
            round_eps.sort(key=lambda x: x[0], reverse=True)
            for done, snaps in round_eps[:2]:
                if snaps and done >= 80:
                    buffer.extend(snaps)
                    n_kept_eps += 1
                    kept_this += 1
        if len(buffer) > keep_snaps:
            buffer = buffer[-keep_snaps:]

        sil_loss = 0.0
        if len(buffer) >= 8:
            for _ in range(sil_epochs):
                sil_loss = _sil_epoch(net, opt, buffer)
            # mild M2 BC anchor so we do not forget FIFO/urgent
            bc_l = _bc_aux_loss(net)
            if float(bc_l.detach()) > 0:
                opt.zero_grad()
                (0.15 * bc_l).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        else:
            print(
                f"  [SIL] r={r} no buffer (need done>={max(min_done, best_det)}; "
                f"sto max={max(sto_scores) if sto_scores else 0})",
                flush=True,
            )

        det_done, _, _, _ = _rollout(net, max_time=max_time, deterministic=True)
        improved = int(det_done) > best_det
        if improved:
            best_det = int(det_done)
            torch.save(
                {
                    "model": net.state_dict(),
                    "meta": {
                        **(blob.get("meta") or {}),
                        "best_det": best_det,
                        "round": r,
                        "train": "sil",
                    },
                },
                CKPT_BEST,
            )
        row = {
            "round": r,
            "kept_eps": kept_this,
            "buf": len(buffer),
            "sto_mean": float(np.mean(sto_scores)) if sto_scores else 0.0,
            "sto_max": int(max(sto_scores) if sto_scores else 0),
            "sil_loss": sil_loss,
            "det": int(det_done),
            "best_det": best_det,
        }
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(
            f"  [SIL] r={r}/{rounds} kept={kept_this} buf={len(buffer)} "
            f"sto_mean={row['sto_mean']:.1f} sto_max={row['sto_max']} "
            f"loss={sil_loss:.4f} det={det_done}/{n_tasks} best_det={best_det}",
            flush=True,
        )

    best_blob = torch.load(CKPT_BEST, map_location="cpu", weights_only=False)
    net.load_state_dict(best_blob["model"])
    meta = {
        **(blob.get("meta") or {}),
        "train": "bc_rl_then_sil",
        "best_det_in_sil": best_det,
        "best_sto_in_sil": best_sto,
        "sil_kept_eps": n_kept_eps,
        "sil_rounds": rounds,
        "source": "best_det_ckpt",
    }
    torch.save({"model": net.state_dict(), "meta": meta}, CKPT)
    (OUT / "train_sil_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[RL-RH-PP] SIL done best_det={best_det} best_sto={best_sto} -> {CKPT}", flush=True)
    return CKPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--snaps", type=int, default=400)
    ap.add_argument("--no-collect", action="store_true")
    ap.add_argument("--ppo", type=int, default=60)
    ap.add_argument("--rl-time", type=int, default=1800)
    ap.add_argument("--bc-coef", type=float, default=0.4)
    ap.add_argument("--sil", action="store_true")
    ap.add_argument("--sil-only", action="store_true")
    ap.add_argument("--sil-eps", type=int, default=40)
    ap.add_argument("--sil-rounds", type=int, default=8)
    ap.add_argument("--sil-min-done", type=int, default=88)
    args = ap.parse_args()
    if not args.sil_only:
        train_bc(epochs=args.epochs, max_snaps=args.snaps, collect=not args.no_collect)
        if args.ppo > 0:
            light_ppo(updates=args.ppo, max_time=args.rl_time, bc_coef=args.bc_coef)
    if args.sil or args.sil_only:
        train_sil(
            collect_eps=args.sil_eps,
            rounds=args.sil_rounds,
            min_done=args.sil_min_done,
            max_time=args.rl_time if args.rl_time else 2000,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
