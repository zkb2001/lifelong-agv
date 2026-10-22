"""Train GlobalRuleNet for high completion: goal-flow teachers on real SH maps."""
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

from greed_coop_mcts.paths import MAP_EXPORTS, RESULTS
from greed_coop_mcts.rule_net import RULE_CKPT, GlobalRuleNet, load_rule_net, save_rule_net
from greed_coop_mcts.rule_obs import (
    GRID,
    Pose,
    build_rule_state,
    conflict_aware_teacher,
    goal_flow_teacher,
    load_aware_goal_flow_teacher,
    teacher_oneway_field,
)
from greed_coop_mcts.scenario import load_slot_meta

Cell = Tuple[int, int]

_MAP_CACHE: Dict[int, Tuple[Set[Cell], np.ndarray]] = {}


def _maze(rng: random.Random) -> Set[Cell]:
    blocked: Set[Cell] = set()
    for x in range(GRID):
        blocked.add((x, 0))
        blocked.add((0, x))
    for y in range(1, GRID):
        for x in range(1, GRID):
            if rng.random() < 0.28:
                blocked.add((x, y))
    for _ in range(rng.randint(3, 7)):
        y = rng.randint(1, GRID - 1)
        for x in range(1, GRID):
            blocked.discard((x, y))
    for _ in range(rng.randint(3, 7)):
        x = rng.randint(1, GRID - 1)
        for y in range(1, GRID):
            blocked.discard((x, y))
    return blocked


def _load_real_map(slot: int) -> Optional[Tuple[Set[Cell], np.ndarray]]:
    if slot in _MAP_CACHE:
        return _MAP_CACHE[slot]
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
        for p in data.get("extra_obstacles") or []:
            if isinstance(p, dict):
                blocked.add((int(p["x"]), int(p["y"])))
            else:
                blocked.add((int(p[0]), int(p[1])))
        for p in (data.get("pickups") or []) + (data.get("dropoffs") or []):
            blocked.add((int(p["x"]), int(p["y"])))
    for p in meta.get("extra_obstacles") or []:
        if isinstance(p, dict):
            blocked.add((int(p["x"]), int(p["y"])))
        else:
            blocked.add((int(p[0]), int(p[1])))
    walk = np.ones((GRID, GRID), dtype=np.float32)
    walk[0, :] = 0
    walk[:, 0] = 0
    for x, y in blocked:
        if 0 <= x < GRID and 0 <= y < GRID:
            walk[y, x] = 0.0
    _MAP_CACHE[slot] = (blocked, walk)
    return blocked, walk


def _free(rng: random.Random, blocked: Set[Cell]) -> Cell:
    for _ in range(500):
        c = (rng.randint(1, GRID - 1), rng.randint(1, GRID - 1))
        if c not in blocked:
            return c
    return (1, 1)


def _edge_cost_target(
    labels: np.ndarray,
    walk: np.ndarray,
    heat: np.ndarray,
) -> np.ndarray:
    """Soft edge-cost field (H,W,4): cost of leaving cell in each dir.

    Prefer teacher direction (cost≈1); reverse / into heat costs more.
    Used as auxiliary regression target for guidance-graph training.
    """
    from greed_coop_mcts.rule_obs import DIR_DELTA

    costs = np.full((GRID, GRID, 4), 2.0, dtype=np.float32)
    for y in range(GRID):
        for x in range(GRID):
            if walk[y, x] < 0.5:
                continue
            lab = int(labels[y, x])
            for di, (dx, dy) in enumerate(DIR_DELTA):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < GRID and 0 <= ny < GRID) or walk[ny, nx] < 0.5:
                    costs[y, x, di] = 3.0
                    continue
                base = 1.0 if di == lab else 1.75
                base += 0.5 * float(heat[ny, nx])
                # Stronger penalty for exact reverse of teacher.
                if lab >= 0 and di == (lab + 2) % 4:
                    base += 0.35
                costs[y, x, di] = base
    return costs


def synth_sample(
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """state, dir labels, walkable, edge_cost target (H,W,4)."""
    # Prefer real SH maps (high completion needs real geometry); cover all slots.
    if rng.random() < 0.90:
        slot = rng.randint(1, 20)
        loaded_map = _load_real_map(slot)
        if loaded_map is None:
            blocked = _maze(rng)
            walk = np.ones((GRID, GRID), dtype=np.float32)
            walk[0, :] = 0
            walk[:, 0] = 0
            for x, y in blocked:
                if 0 <= x < GRID and 0 <= y < GRID:
                    walk[y, x] = 0.0
        else:
            blocked, walk = loaded_map
            blocked = set(blocked)
            walk = walk.copy()
    else:
        blocked = _maze(rng)
        walk = np.ones((GRID, GRID), dtype=np.float32)
        walk[0, :] = 0
        walk[:, 0] = 0
        for x, y in blocked:
            if 0 <= x < GRID and 0 <= y < GRID:
                walk[y, x] = 0.0

    n_ag = rng.randint(3, 8)
    poses: Dict[str, Pose] = {}
    deliver: Dict[str, Cell] = {}
    pickup: Dict[str, Cell] = {}
    loaded: Set[str] = set()
    used: Set[Cell] = set(blocked)
    for i in range(n_ag):
        name = f"a{i}"
        p = _free(rng, used)
        used.add(p)
        pitch = rng.choice([0, 90, 180, 270])
        poses[name] = (p[0], p[1], pitch)
        g = _free(rng, blocked)
        if rng.random() < 0.55:
            loaded.add(name)
            deliver[name] = g
        else:
            pickup[name] = g

    heat = np.zeros((GRID, GRID), dtype=np.float32)
    # Denser maps / more agents → more conflict heat samples (guidance stay).
    wall_ratio = float(np.sum(walk < 0.5)) / float(GRID * GRID)
    heat_p = 0.55 + 0.35 * min(1.0, wall_ratio / 0.35)
    if rng.random() < heat_p:
        for _ in range(rng.randint(2, 8)):
            c = _free(rng, blocked)
            heat[c[1], c[0]] = rng.uniform(0.4, 1.0)

    state = build_rule_state(
        blocked=blocked,
        poses=poses,
        deliver_goals=deliver,
        pickup_goals=pickup,
        loaded=loaded,
        conflict_heat=heat,
    )
    goals = {**deliver, **pickup}
    # Primary: load-aware goal-flow (separate deliver/pickup). Mixing both
    # into one vote teaches head-on thrash on SH17-like corridors.
    r = rng.random()
    if r < 0.55:
        labels = load_aware_goal_flow_teacher(
            walk, blocked, deliver, pickup, poses, loaded, blend_circulation=0.15
        )
    elif wall_ratio > 0.28 and r < 0.75:
        labels = conflict_aware_teacher(walk, poses, goals, blocked)
    elif r < 0.90:
        labels = goal_flow_teacher(walk, blocked, goals, blend_circulation=0.2)
    else:
        labels = teacher_oneway_field(walk)
    edge = _edge_cost_target(labels.astype(np.int64), walk, heat)
    return state, labels.astype(np.int64), walk, edge


def train(
    *,
    steps: int = 12000,
    batch: int = 16,
    lr: float = 2e-4,
    seed: int = 0,
    device: str = "cpu",
    out: Path | None = None,
    resume: bool = True,
) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out = Path(out or RULE_CKPT)
    if resume and out.exists():
        net = load_rule_net(out, device=device)
        print(f"[rule-ai] resume from {out}", flush=True)
    else:
        net = GlobalRuleNet().to(device)
    opt = AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    # Warm map cache
    for s in range(1, 21):
        _load_real_map(s)

    losses: List[float] = []
    accs: List[float] = []
    for step in range(1, steps + 1):
        states, labels, walks, edges = [], [], [], []
        for _ in range(batch):
            s, lab, w, e = synth_sample(rng)
            states.append(s)
            labels.append(lab)
            walks.append(w)
            edges.append(e)
        x = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.stack(labels), dtype=torch.long, device=device)
        w = torch.as_tensor(np.stack(walks), dtype=torch.float32, device=device)
        edge_t = torch.as_tensor(np.stack(edges), dtype=torch.float32, device=device)

        net.train()
        logits = net(x)
        b, d, h, ww = logits.shape
        flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, d)
        flat_y = y.reshape(-1)
        flat_w = w.reshape(-1)
        valid = flat_w > 0.5
        ce = F.cross_entropy(flat_logits[valid], flat_y[valid].clamp(min=0))
        # Aux guidance: expected edge cost under predicted dir distribution.
        probs = F.softmax(logits, dim=1).permute(0, 2, 3, 1)  # B,H,W,4
        exp_cost = (probs * edge_t).sum(dim=-1)
        aux = (exp_cost * w).sum() / w.sum().clamp_min(1.0)
        loss = ce + 0.15 * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        opt.step()
        losses.append(float(loss.item()))

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = ((pred == y) & (w > 0.5)).float().sum()
            total = (w > 0.5).float().sum().clamp_min(1.0)
            acc = float((correct / total).item())
            accs.append(acc)

        if step == 1 or step % 500 == 0 or step == steps:
            print(
                f"[rule-ai] step={step}/{steps} loss={loss.item():.4f} "
                f"ce={ce.item():.4f} aux={aux.item():.4f} acc={acc:.3f}",
                flush=True,
            )
        if step % 2000 == 0:
            save_rule_net(
                net,
                out,
                train_steps=step,
                interim=True,
                final_acc=float(np.mean(accs[-50:])),
            )

    path = save_rule_net(
        net,
        out,
        train_steps=steps,
        batch=batch,
        seed=seed,
        final_loss=float(np.mean(losses[-100:])),
        final_acc=float(np.mean(accs[-100:])),
        n_params=int(sum(p.numel() for p in net.parameters())),
        teacher="goal_flow_guidance_v3",
    )
    metrics = {
        "ckpt": str(path),
        "steps": steps,
        "final_loss": float(np.mean(losses[-100:])),
        "final_acc": float(np.mean(accs[-100:])),
        "n_params": int(sum(p.numel() for p in net.parameters())),
        "teacher": "goal_flow_guidance_v3",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "train_rule_system_v3.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        f"[rule-ai] saved {path} params={metrics['n_params']} acc={metrics['final_acc']:.3f}",
        flush=True,
    )
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--no-resume", action="store_true")
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
        resume=not bool(args.no_resume),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
