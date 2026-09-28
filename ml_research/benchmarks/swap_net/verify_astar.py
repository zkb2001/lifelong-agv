"""A*-grounded SwapNet labels + verification (swap vs no-swap rollouts)."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ml_research.benchmarks.common import is_urgent_task
from ml_research.benchmarks.swap_net.features import FEAT_DIM, SwapContext, build_swap_features
from ml_research.benchmarks.swap_net.micro_env import FAR_ARRIVAL_GUARD, _SAMPLERS, sample_spec, spec_to_context
from ml_research.benchmarks.swap_net.model import SwapNet, load_swap_net, save_swap_net
from ml_research.benchmarks.swap_net.policy import urgent_aware_decide
from ml_research.common.paths import CKPT, RESULTS

OUT = RESULTS / "lifelong" / "swapnet_ab"
# Keep agents on-map so feature ETA ≈ A* length
MAX_ETA = 16.0
PICKUP = (18, 10)
DROP = (18, 14)


@dataclass
class ArmScore:
    t_pickup: float
    total: float
    sla_miss: float
    ok: bool


def _clamp_xy(xy: Tuple[int, int]) -> Tuple[int, int]:
    return (max(1, min(20, int(xy[0]))), max(1, min(20, int(xy[1]))))


def pos_for_eta(eta: float, pickup: Tuple[int, int] = PICKUP) -> Tuple[int, int]:
    e = int(round(max(0.0, min(MAX_ETA, float(eta)))))
    return _clamp_xy((pickup[0] - e, pickup[1]))


def plan_pickup_time(mod, start_xy, pickup_xy, end_xy, static_obs, t0: int = 0) -> float:
    start = (int(start_xy[0]), int(start_xy[1]), int(t0), 0)
    task = {
        "agv": "probe",
        "task_id": "T1",
        "agv_start_point": start,
        "pickup_point": tuple(pickup_xy),
        "end_points": [tuple(end_xy)],
        "destination": "D",
        "priority": "Normal",
        "pickup_name": "S",
    }
    path, _steps = mod.A_Star(
        task,
        start,
        tuple(pickup_xy),
        [tuple(end_xy)],
        static_obs,
        {},
        max_path_len=80,
        max_visited=5000,
        max_frontier=12000,
        wall_budget_s=0.35,
    )
    if not path:
        return float("inf")
    for p in path:
        if (int(p[0]), int(p[1])) == (int(pickup_xy[0]), int(pickup_xy[1])):
            return float(int(p[2]) - t0)
    return float(abs(start_xy[0] - pickup_xy[0]) + abs(start_xy[1] - pickup_xy[1]))


def score_pair(
    mod,
    ctx: SwapContext,
    *,
    static_obs,
    thrash: float = 2.0,
) -> Tuple[ArmScore, ArmScore, float, float]:
    """Return (no_swap, swap, t_far_astar, t_near_astar)."""
    far_xy = pos_for_eta(min(ctx.far_eta, MAX_ETA))
    near_xy = pos_for_eta(min(ctx.near_eta, MAX_ETA))
    # Ensure near not behind far
    if abs(near_xy[0] - PICKUP[0]) >= abs(far_xy[0] - PICKUP[0]):
        near_xy = pos_for_eta(max(0.0, min(ctx.near_eta, ctx.far_eta - 1.0)))

    t_far = plan_pickup_time(mod, far_xy, PICKUP, DROP, static_obs)
    t_near = plan_pickup_time(mod, near_xy, PICKUP, DROP, static_obs)

    urgent = is_urgent_task(ctx.task_info)
    rt = ctx.task_info.get("remaining_time")
    try:
        rt_f = float(rt) if rt not in (None, "") else None
    except (TypeError, ValueError):
        rt_f = None
    # Cap SLA to map-feasible horizon so labels stay consistent with A*
    if rt_f is not None:
        rt_f = min(rt_f, MAX_ETA + 8.0)

    def sla_pen(t: float) -> float:
        if not urgent or rt_f is None or t >= float("inf"):
            return 0.0
        if t <= rt_f:
            return 0.0
        return 50.0 + 5.0 * (t - rt_f)

    # no-swap
    side_ns = 0.35 * max(0.0, (t_far - t_near) if t_far < float("inf") else 0.0)
    side_ns += 0.1 * float(ctx.queue_depth) * max(0.0, t_far - t_near if t_far < float("inf") else 0.0)
    total_ns = (t_far if t_far < float("inf") else 1e6) + sla_pen(t_far) + side_ns
    arm_ns = ArmScore(t_far, total_ns, sla_pen(t_far), t_far < float("inf"))

    # swap
    side_sw = thrash
    if min(ctx.far_eta, t_far) <= FAR_ARRIVAL_GUARD:
        side_sw += 12.0
    if ctx.near_task_info and is_urgent_task(ctx.near_task_info) and not urgent:
        side_sw += 150.0
    gap = (t_far - t_near) if (t_far < float("inf") and t_near < float("inf")) else 0.0
    if ctx.queue_depth >= 4 and gap < float(ctx.margin) + 2.0:
        side_sw += 8.0 + 0.5 * float(ctx.queue_depth)
    # pointless: both miss SLA
    if urgent and rt_f is not None and t_far > rt_f and t_near > rt_f:
        side_sw += 20.0
    # far already OK on A* time: thrash discouraged unless gap huge
    if urgent and rt_f is not None and t_far <= rt_f and gap < float(ctx.margin) + 3.0:
        side_sw += 10.0

    total_sw = (t_near if t_near < float("inf") else 1e6) + sla_pen(t_near) + side_sw
    arm_sw = ArmScore(t_near, total_sw, sla_pen(t_near), t_near < float("inf"))
    return arm_ns, arm_sw, t_far, t_near


def empirical_label(arm_ns: ArmScore, arm_sw: ArmScore, eps: float = 0.5) -> Optional[int]:
    if not arm_ns.ok and arm_sw.ok:
        return 1
    if arm_ns.ok and not arm_sw.ok:
        return 0
    if not arm_ns.ok and not arm_sw.ok:
        return None
    if arm_sw.total + eps < arm_ns.total:
        return 1
    if arm_ns.total + eps < arm_sw.total:
        return 0
    return None


def grounded_context(ctx: SwapContext, t_far: float, t_near: float) -> SwapContext:
    """Replace ETA features with measured A* times (map-feasible)."""
    rt = ctx.task_info.get("remaining_time")
    task = dict(ctx.task_info)
    if rt not in (None, ""):
        try:
            task["remaining_time"] = min(float(rt), MAX_ETA + 8.0)
        except (TypeError, ValueError):
            pass
    return SwapContext(
        far_eta=float(t_far) if t_far < float("inf") else float(min(ctx.far_eta, MAX_ETA)),
        near_eta=float(t_near) if t_near < float("inf") else float(min(ctx.near_eta, MAX_ETA)),
        margin=ctx.margin,
        far_dist=float(t_far) if t_far < float("inf") else ctx.far_dist,
        near_dist=float(t_near) if t_near < float("inf") else ctx.near_dist,
        task_info=task,
        queue_depth=ctx.queue_depth,
        near_has_task=ctx.near_has_task,
        near_task_info=ctx.near_task_info,
    )


def build_astar_dataset(mod, static_obs, n: int = 12000, seed: int = 42):
    rng = random.Random(seed)
    # Oversample soft-boundary scenarios the old rules used to hard-block
    soft_focus = (
        ["no_swap_gap_small"] * 3
        + ["no_swap_deep_queue_marginal"] * 3
        + ["no_swap_urgent_already_ok"] * 2
        + ["random"] * 2
        + ["yes_swap_dock_wander"] * 2
        + ["yes_swap_urgent_rescue"] * 2
        + list(_SAMPLERS.keys())
    )
    xs, ys, ws, tags = [], [], [], []
    skipped = 0
    for _i in range(n):
        name = rng.choice(soft_focus)
        if rng.random() < 0.2:
            spec = sample_spec(rng)
        else:
            spec = _SAMPLERS[name](rng)
        if spec.far_eta is not None:
            spec.far_eta = min(float(spec.far_eta), MAX_ETA)
        if spec.near_eta is not None:
            spec.near_eta = min(
                float(spec.near_eta), max(0.0, min(float(spec.near_eta), MAX_ETA - 1))
            )
        if spec.urgent and spec.remaining_time:
            spec.remaining_time = min(float(spec.remaining_time), MAX_ETA + 8.0)

        ctx, _, scenario = spec_to_context(spec)
        arm_ns, arm_sw, t_far, t_near = score_pair(mod, ctx, static_obs=static_obs)
        y = empirical_label(arm_ns, arm_sw)
        if y is None:
            skipped += 1
            continue
        gctx = grounded_context(ctx, t_far, t_near)
        xs.append(build_swap_features(gctx))
        ys.append(float(y))
        # Emphasize soft-boundary learning; balance classes lightly
        w = 1.0
        if y == 1:
            w = 1.35  # lift swap recall (was under-agreed)
        else:
            w = 1.15
        if scenario in (
            "no_swap_gap_small",
            "no_swap_deep_queue_marginal",
            "no_swap_urgent_already_ok",
            "random",
        ):
            w *= 1.6
        if is_urgent_task(gctx.task_info):
            w *= 1.15
        # Extra weight when A* disagrees with naive margin rule (soft case)
        gap = gctx.far_eta - gctx.near_eta
        if y == 1 and gap < gctx.margin:
            w *= 2.0  # should-swap despite small gap
        if y == 1 and gctx.queue_depth >= 4:
            w *= 1.5
        ws.append(w)
        tags.append(scenario)
    return (
        np.stack(xs),
        np.array(ys, dtype=np.float32),
        np.array(ws, dtype=np.float32),
        tags,
        skipped,
    )


def train_on_astar(*, n: int = 8000, epochs: int = 12, seed: int = 42) -> dict:
    import simulation.engine as mod

    static_obs = list(mod.ENV([], []).get_static_obstacles())
    print(f"[astar-train] building {n} A*-labeled samples...", flush=True)
    x, y, w, tags, skipped = build_astar_dataset(mod, static_obs, n=n, seed=seed)
    print(
        f"[astar-train] kept={len(y)} skipped_ties={skipped} "
        f"swap_rate={float(y.mean()):.3f}",
        flush=True,
    )
    from collections import Counter

    c = Counter(tags)
    for k, v in sorted(c.items()):
        rate = float(y[[i for i, t in enumerate(tags) if t == k]].mean()) if v else 0
        print(f"  {k:<32} n={v:>4} swap_rate={rate:.2f}", flush=True)

    idx = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(len(y) * 0.9)
    tr, va = idx[:split], idx[split:]

    net = SwapNet(feat_dim=FEAT_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=2.5e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    torch.manual_seed(seed)

    for ep in range(1, epochs + 1):
        net.train()
        losses = []
        for i in range(0, len(tr), 256):
            b = tr[i : i + 256]
            xb = torch.from_numpy(x[b])
            yb = torch.from_numpy(y[b])
            wb = torch.from_numpy(w[b])
            opt.zero_grad()
            loss = (loss_fn(net(xb), yb) * wb).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(torch.from_numpy(x[va])))
            pred = (pv >= 0.5).float().numpy()
            acc = float((pred == y[va]).mean())
            ns = y[va] < 0.5
            ys_ = y[va] >= 0.5
            ns_acc = float((pred[ns] == y[va][ns]).mean()) if ns.any() else 0.0
            sw_acc = float((pred[ys_] == y[va][ys_]).mean()) if ys_.any() else 0.0
        print(
            f"[astar-train] ep={ep} loss={np.mean(losses):.4f} "
            f"val_acc={acc:.3f} no_swap={ns_acc:.3f} swap={sw_acc:.3f}",
            flush=True,
        )

    path = save_swap_net(
        net,
        CKPT / "swap_net_v3.pt",
        version="v3_net_soft_boundary",
        swap_rate=float(y.mean()),
    )
    save_swap_net(net, CKPT / "swap_net.pt", version="v3_net_soft_boundary")
    return {"ckpt": str(path), "net": net, "n": len(y), "swap_rate": float(y.mean())}


def verify(*, n: int = 400, seed: int = 11, use_net=None) -> dict:
    import simulation.engine as mod

    net = use_net or load_swap_net(CKPT / "swap_net_v3.pt")
    static_obs = list(mod.ENV([], []).get_static_obstacles())
    rng = random.Random(seed)
    names = list(_SAMPLERS.keys())
    per = max(1, n // len(names))
    plan = []
    for name in names:
        plan.extend([name] * per)
    rng.shuffle(plan)
    plan = plan[:n]

    conf = defaultdict(lambda: {"n": 0, "agree": 0, "disagree": 0, "tie": 0})
    rows = []
    agree = decided = ties = 0

    for i, name in enumerate(plan):
        spec = _SAMPLERS[name](rng)
        if spec.far_eta is not None:
            spec.far_eta = min(float(spec.far_eta), MAX_ETA)
        if spec.near_eta is not None:
            spec.near_eta = min(float(spec.near_eta), MAX_ETA)
        if spec.urgent and spec.remaining_time:
            spec.remaining_time = min(float(spec.remaining_time), MAX_ETA + 8.0)
        ctx, _, scenario = spec_to_context(spec)
        arm_ns, arm_sw, t_far, t_near = score_pair(mod, ctx, static_obs=static_obs)
        truth = empirical_label(arm_ns, arm_sw)
        gctx = grounded_context(ctx, t_far, t_near)
        # Decision uses A*-grounded features (fair: same info as label)
        dec = urgent_aware_decide(gctx, net)

        conf[scenario]["n"] += 1
        row = {
            "scenario": scenario,
            "t_far": t_far if t_far < float("inf") else None,
            "t_near": t_near if t_near < float("inf") else None,
            "score_ns": round(arm_ns.total, 3),
            "score_sw": round(arm_sw.total, 3),
            "emp_swap": truth,
            "pred_swap": bool(dec.swap),
            "source": dec.source,
            "reason": dec.reason,
        }
        if truth is None:
            conf[scenario]["tie"] += 1
            ties += 1
            row["agree"] = None
        else:
            decided += 1
            ok = bool(dec.swap) == bool(truth)
            conf[scenario]["agree" if ok else "disagree"] += 1
            if ok:
                agree += 1
            row["agree"] = ok
        rows.append(row)

    by = {}
    for k, v in conf.items():
        d = v["agree"] + v["disagree"]
        by[k] = {**v, "acc": (v["agree"] / d) if d else None}

    should_rej = [r for r in rows if r["emp_swap"] in (False, 0)]
    should_sw = [r for r in rows if r["emp_swap"] in (True, 1)]
    rej_ok = sum(1 for r in should_rej if not r["pred_swap"])
    sw_ok = sum(1 for r in should_sw if r["pred_swap"])
    net_frac = sum(1 for r in rows if r["source"] == "net") / max(1, len(rows))

    summary = {
        "n": n,
        "decided": decided,
        "ties": ties,
        "true_acc": (agree / decided) if decided else None,
        "agree": agree,
        "disagree": decided - agree,
        "reject_success_rate": (rej_ok / len(should_rej)) if should_rej else None,
        "agree_success_rate": (sw_ok / len(should_sw)) if should_sw else None,
        "n_should_reject": len(should_rej),
        "n_should_swap": len(should_sw),
        "net_decision_frac": net_frac,
        "by_scenario": by,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verify_swap_astar.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="retrain on A* labels")
    ap.add_argument("--samples", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--n", type=int, default=400, help="verify count")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)

    net = None
    if args.train and not args.verify_only:
        out = train_on_astar(n=args.samples, epochs=args.epochs, seed=args.seed)
        net = out["net"]
        print(f"[astar-train] saved {out['ckpt']}", flush=True)

    summary = verify(n=args.n, seed=args.seed, use_net=net)
    print("\n========== A* grounded verification ==========", flush=True)
    acc = summary["true_acc"] or 0.0
    print(
        f"decided={summary['decided']} ties={summary['ties']} "
        f"true_acc={acc:.4f} agree={summary['agree']} disagree={summary['disagree']}",
        flush=True,
    )
    print(
        f"reject_success={summary['reject_success_rate']:.4f} "
        f"(n={summary['n_should_reject']})  "
        f"agree_success={summary['agree_success_rate']:.4f} "
        f"(n={summary['n_should_swap']})  "
        f"net_frac={summary['net_decision_frac']:.3f}",
        flush=True,
    )
    print(f"{'scenario':<32} {'n':>4} {'acc':>7} {'ag':>5} {'dis':>5} {'tie':>4}", flush=True)
    for k, v in sorted(summary["by_scenario"].items()):
        a = f"{v['acc']:.3f}" if v["acc"] is not None else "  n/a"
        print(
            f"{k:<32} {v['n']:>4} {a:>7} {v['agree']:>5} {v['disagree']:>5} {v['tie']:>4}",
            flush=True,
        )
    print(f"report -> {OUT / 'verify_swap_astar.json'}", flush=True)
    rej = summary["reject_success_rate"] or 0.0
    agr = summary["agree_success_rate"] or 0.0
    if acc < 0.85 or rej < 0.90 or agr < 0.85:
        print(
            f"[FAIL] true_acc={acc:.3f} reject={rej:.3f} agree={agr:.3f} "
            f"(need >=0.85 / 0.90 / 0.85)",
            flush=True,
        )
        return 1
    print(f"[PASS] true_acc={acc:.3f} reject={rej:.3f} agree={agr:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
