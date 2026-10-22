"""IL train StageSelectorNet v2: stage CE + listwise priority + fleet MSE."""
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

from greed_coop_mcts.paths import RESULTS
from greed_coop_mcts.rule_obs import GRID
from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.stage_net import (
    AGENT_FEAT_V2,
    ARCH_VERSION_V2,
    CAND_FEAT,
    GLOBAL_FEAT,
    N_CAND,
    PAIR_FEAT,
    STAGE_CKPT_V2,
    StageSelectorNetV2,
    _agent_feat,
    _cand_feat,
    _manh,
    build_candidates,
    build_global_feat,
    build_pairwise_features,
    region_id,
    save_stage_net,
    teacher_fleet_cap,
    teacher_kind,
    teacher_priority,
)

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]
_MAP_CACHE: Dict[int, Set[Cell]] = {}


def _load_blocked(slot: int) -> Optional[Set[Cell]]:
    if slot in _MAP_CACHE:
        return set(_MAP_CACHE[slot])
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
        for key in ("extra_obstacles",):
            for p in data.get(key) or []:
                if isinstance(p, dict):
                    blocked.add((int(p["x"]), int(p["y"])))
                else:
                    blocked.add((int(p[0]), int(p[1])))
        for p in (data.get("pickups") or []) + (data.get("dropoffs") or []):
            blocked.add((int(p["x"]), int(p["y"])))
    _MAP_CACHE[slot] = blocked
    return set(blocked)


def _free(rng: random.Random, blocked: Set[Cell], used: Optional[Set[Cell]] = None) -> Cell:
    ban = set(blocked) | (used or set())
    for _ in range(400):
        c = (rng.randint(1, GRID - 1), rng.randint(1, GRID - 1))
        if c not in ban:
            return c
    return (1, 1)


def _bfs_path(start: Cell, goal: Cell, blocked: Set[Cell]) -> List[Cell]:
    from collections import deque

    if start == goal:
        return []
    prev: Dict[Cell, Optional[Cell]] = {start: None}
    q = deque([start])
    found = False
    while q:
        cur = q.popleft()
        if cur == goal:
            found = True
            break
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID):
                continue
            if nxt in blocked or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if not found:
        return []
    path: List[Cell] = []
    cur: Optional[Cell] = goal
    while cur is not None and cur != start:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


def listwise_loss(pred: torch.Tensor, target: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """ListNet KL: soft ranks from teacher priorities vs predicted scores."""
    t = F.softmax(target / max(1e-3, temp), dim=-1)
    log_p = F.log_softmax(pred, dim=-1)
    return -(t * log_p).sum(dim=-1).mean()


def _random_maze(rng: random.Random) -> Set[Cell]:
    blocked: Set[Cell] = set()
    for x in range(GRID):
        blocked.add((x, 0))
        blocked.add((0, x))
    for y in range(1, GRID):
        for x in range(1, GRID):
            if rng.random() < 0.22:
                blocked.add((x, y))
    # Carve a few corridors so paths exist.
    for _ in range(rng.randint(2, 5)):
        y = rng.randint(1, GRID - 1)
        for x in range(1, GRID):
            blocked.discard((x, y))
    for _ in range(rng.randint(2, 5)):
        x = rng.randint(1, GRID - 1)
        for y in range(1, GRID):
            blocked.discard((x, y))
    return blocked


def synth_meeting(rng: random.Random, *, max_n: int = 30):
    """Domain-randomized meeting — uniform over maps/sizes, no forced slots.

    Generalization comes from broad coverage (real maps ∪ mazes, n∈[3,30],
    random phase mix / heat), not from oversampling hard cases.
    """
    if rng.random() < 0.75:
        slot = rng.randint(1, 20)
        blocked = _load_blocked(slot) or _random_maze(rng)
    else:
        blocked = _random_maze(rng)
    # Uniform fleet size — do not skew toward sparse/dense curricula.
    n = rng.randint(3, max_n)
    stay = np.zeros((GRID, GRID), dtype=np.float32)
    heat = np.zeros((GRID, GRID), dtype=np.float32)
    # Variable congestion intensity (domain random).
    n_hot = rng.randint(0, int(4 + 12 * rng.random()))
    for _ in range(n_hot):
        c = _free(rng, blocked)
        stay[c[1], c[0]] = rng.uniform(0.2, 1.5)
        heat[c[1], c[0]] = rng.uniform(0.0, 1.0)

    names = [f"a{i}" for i in range(n)]
    poses: Dict[str, Pose] = {}
    goals: Dict[str, Cell] = {}
    paths: Dict[str, List[Cell]] = {}
    loaded: Set[str] = set()
    assigned: Set[str] = set()
    streaks: Dict[str, int] = {}
    used: Set[Cell] = set()
    # Random phase mix (not fixed 40/30/30).
    p_load = rng.uniform(0.15, 0.55)
    p_asg = rng.uniform(0.10, 0.45)
    for nm in names:
        cur = _free(rng, blocked, used)
        used.add(cur)
        final = _free(rng, blocked)
        pitch = rng.choice([0, 90, 180, 270])
        poses[nm] = (cur[0], cur[1], pitch)
        goals[nm] = final
        paths[nm] = _bfs_path(cur, final, blocked)
        streaks[nm] = rng.randint(0, 30)
        r = rng.random()
        if r < p_load:
            loaded.add(nm)
        elif r < p_load + p_asg:
            assigned.add(nm)

    occupied = {poses[nm][:2] for nm in names}
    region_occ_map: Dict[Tuple[int, int], int] = {}
    for nm in names:
        rid = region_id(goals[nm])
        region_occ_map[rid] = region_occ_map.get(rid, 0) + 1

    agent_rows = []
    cand_rows = []
    masks = []
    labels = []
    pri_t = []
    for nm in names:
        cur = poses[nm][:2]
        final = goals[nm]
        path = paths[nm]
        cands = build_candidates(
            cur=cur,
            final=final,
            path=path,
            blocked=blocked,
            stay_penalty=stay,
            occupied=occupied - {cur},
        )
        rid = region_id(final)
        rocc = float(region_occ_map.get(rid, 0))
        heat_v = float(heat[cur[1], cur[0]])
        gheat = float(heat[final[1], final[0]])
        af = _agent_feat(
            cur=cur,
            final=final,
            loaded=nm in loaded,
            assigned=nm in assigned,
            streak=streaks[nm],
            region_occ=rocc,
            heat=heat_v,
            path=path,
            blocked=blocked,
            occupied=occupied - {cur},
            goal_heat=gheat,
            v2=True,
        )
        cfeats = []
        mask = []
        for kind, cell in cands:
            cfeats.append(
                _cand_feat(
                    cur=cur,
                    final=final,
                    kind=kind,
                    cell=cell,
                    stay_penalty=stay,
                    occupied=occupied - {cur},
                    edge_cost=rng.uniform(1.0, 2.0),
                    region_occ=rocc,
                )
            )
            mask.append(1.0)
        while len(cfeats) < N_CAND:
            cfeats.append(np.zeros(CAND_FEAT, dtype=np.float32))
            mask.append(0.0)
        kinds = [k for k, _ in cands]
        cells = [c for _, c in cands]
        lab = teacher_kind(
            cur=cur,
            final=final,
            kinds=kinds,
            cells=cells,
            region_occ=rocc,
            streak=streaks[nm],
            heat=heat_v,
            loaded=nm in loaded,
        )
        pri = teacher_priority(
            loaded=nm in loaded,
            assigned=nm in assigned,
            streak=streaks[nm],
            dist_final=_manh(cur, final),
            heat=heat_v,
        )
        agent_rows.append(af)
        cand_rows.append(np.stack(cfeats[:N_CAND]))
        masks.append(np.asarray(mask[:N_CAND], dtype=np.float32))
        labels.append(int(lab))
        pri_t.append(float(np.log1p(pri)))

    pair = build_pairwise_features(
        names=names,
        poses=poses,
        goals=goals,
        paths=paths,
        loaded=loaded,
        assigned=assigned,
        conflict_heat=heat,
    )
    # Fleet teacher from synthetic congestion.
    heats = [float(heat[poses[nm][1], poses[nm][0]]) for nm in names]
    mean_heat = float(np.mean(heats)) if heats else 0.0
    max_heat = float(np.max(heats)) if heats else 0.0
    mean_rocc = float(np.mean(list(region_occ_map.values()))) if region_occ_map else 0.0
    stall = rng.randint(0, 120)
    remaining = rng.randint(0, 100)
    no_prog = rng.random() < 0.25
    gfeat = build_global_feat(
        n_agents=n,
        n_carrying=len(loaded),
        n_assigned=len(assigned),
        mean_heat=mean_heat,
        max_heat=max_heat,
        mean_region_occ=mean_rocc,
        stall=stall,
        remaining=remaining,
        no_prog=no_prog,
    )
    cap = teacher_fleet_cap(
        n_agents=n,
        n_carrying=len(loaded),
        n_assigned=len(assigned),
        mean_heat=mean_heat,
        max_heat=max_heat,
        mean_region_occ=mean_rocc,
        stall=stall,
        remaining=remaining,
        no_prog=no_prog,
    )
    fleet_frac = float(cap) / float(max(1, n))
    return {
        "agent": np.stack(agent_rows),
        "cands": np.stack(cand_rows),
        "mask": np.stack(masks),
        "pair": pair,
        "label": np.asarray(labels, dtype=np.int64),
        "pri": np.asarray(pri_t, dtype=np.float32),
        "global": gfeat,
        "fleet": fleet_frac,
        "n": n,
    }


def _pad_batch(samples: List[dict], device: str):
    b = len(samples)
    n_max = max(int(s["n"]) for s in samples)
    agent = np.zeros((b, n_max, AGENT_FEAT_V2), dtype=np.float32)
    cands = np.zeros((b, n_max, N_CAND, CAND_FEAT), dtype=np.float32)
    mask = np.zeros((b, n_max, N_CAND), dtype=np.float32)
    pair = np.zeros((b, n_max, n_max, PAIR_FEAT), dtype=np.float32)
    label = np.zeros((b, n_max), dtype=np.int64)
    pri = np.zeros((b, n_max), dtype=np.float32)
    amask = np.zeros((b, n_max), dtype=np.float32)
    gfeat = np.stack([s["global"] for s in samples])
    fleet = np.asarray([s["fleet"] for s in samples], dtype=np.float32)
    for i, s in enumerate(samples):
        n = int(s["n"])
        agent[i, :n] = s["agent"]
        cands[i, :n] = s["cands"]
        mask[i, :n] = s["mask"]
        pair[i, :n, :n] = s["pair"]
        label[i, :n] = s["label"]
        pri[i, :n] = s["pri"]
        amask[i, :n] = 1.0
    return {
        "agent": torch.as_tensor(agent, device=device),
        "cands": torch.as_tensor(cands, device=device),
        "mask": torch.as_tensor(mask, device=device),
        "pair": torch.as_tensor(pair, device=device),
        "label": torch.as_tensor(label, device=device),
        "pri": torch.as_tensor(pri, device=device),
        "amask": torch.as_tensor(amask, device=device),
        "global": torch.as_tensor(gfeat, device=device),
        "fleet": torch.as_tensor(fleet, device=device),
    }


def train(
    *,
    steps: int = 6000,
    batch: int = 8,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
    out: Path | None = None,
    fleet_only: bool = False,
) -> dict:
    rng = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for s in range(1, 21):
        _load_blocked(s)

    net = StageSelectorNetV2().to(device)
    ckpt = out or STAGE_CKPT_V2
    if Path(ckpt).is_file():
        try:
            blob = torch.load(ckpt, map_location=device, weights_only=False)
            state = blob.get("state_dict", blob)
            cur = net.state_dict()
            filtered = {
                k: v
                for k, v in state.items()
                if k in cur and tuple(cur[k].shape) == tuple(v.shape)
            }
            net.load_state_dict(filtered, strict=False)
            print(
                f"[stage-net] warm-start {len(filtered)}/{len(cur)} tensors from {ckpt}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[stage-net] resume skip: {exc}", flush=True)

    if fleet_only:
        for name, p in net.named_parameters():
            p.requires_grad = name.startswith("fleet_head")
    opt = AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )

    losses: List[float] = []
    accs: List[float] = []
    for step in range(1, steps + 1):
        samples = [synth_meeting(rng) for _ in range(batch)]
        b = _pad_batch(samples, device)
        net.train()
        logits, pri = net(
            b["agent"], b["cands"], b["mask"], pair=b["pair"], agent_mask=b["amask"]
        )
        # Stage CE on valid agents
        flat_logits = logits.reshape(-1, N_CAND)
        flat_y = b["label"].reshape(-1)
        flat_am = b["amask"].reshape(-1) > 0.5
        ce = F.cross_entropy(flat_logits[flat_am], flat_y[flat_am].clamp(min=0))
        # Listwise priority (mask padded agents with -inf before softmax)
        pri_masked = pri.masked_fill(b["amask"] < 0.5, -1e9)
        tgt_masked = b["pri"].masked_fill(b["amask"] < 0.5, -1e9)
        lw = listwise_loss(pri_masked, tgt_masked, temp=2.0)
        # Fleet: pool agent embeddings
        h = net.encode_agents(b["agent"], pair=b["pair"], agent_mask=b["amask"])
        denom = b["amask"].sum(dim=-1, keepdim=True).clamp_min(1.0)
        pool = (h * b["amask"].unsqueeze(-1)).sum(dim=1) / denom
        fleet_frac = net.forward_fleet_frac(b["global"], agent_pool=pool)
        fleet_mse = F.mse_loss(fleet_frac, b["fleet"])
        if fleet_only:
            loss = fleet_mse
        else:
            loss = ce + 0.5 * lw + 0.25 * fleet_mse
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad], 2.0
        )
        opt.step()
        losses.append(float(loss.item()))
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct = ((pred == b["label"]) & (b["amask"] > 0.5)).float().sum()
            total = (b["amask"] > 0.5).float().sum().clamp_min(1.0)
            accs.append(float((correct / total).item()))
        if step == 1 or step % 500 == 0 or step == steps:
            print(
                f"[stage-net] step={step}/{steps} loss={loss.item():.4f} "
                f"ce={ce.item():.4f} list={lw.item():.4f} fleet={fleet_mse.item():.4f} "
                f"acc={accs[-1]:.3f}",
                flush=True,
            )

    path = save_stage_net(
        net,
        out or STAGE_CKPT_V2,
        steps=steps,
        final_acc=float(np.mean(accs[-100:])),
        teacher="meeting_il_listwise_v2",
        arch=ARCH_VERSION_V2,
    )
    metrics = {
        "ckpt": str(path),
        "steps": steps,
        "final_loss": float(np.mean(losses[-100:])),
        "final_acc": float(np.mean(accs[-100:])),
        "agent_feat": AGENT_FEAT_V2,
        "cand_feat": CAND_FEAT,
        "pair_feat": PAIR_FEAT,
        "global_feat": GLOBAL_FEAT,
        "arch": ARCH_VERSION_V2,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "train_stage_net_v2.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"[stage-net] saved {path} acc={metrics['final_acc']:.3f}", flush=True)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument(
        "--fleet-only",
        action="store_true",
        help="only train fleet heads (legacy safety mode)",
    )
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
        fleet_only=bool(args.fleet_only),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
