"""Train SwapNet with no-swap hard negatives + per-scenario eval."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from ml_research.benchmarks.swap_net.features import FEAT_DIM, SwapContext, build_swap_features
from ml_research.benchmarks.swap_net.micro_env import (
    FAR_ARRIVAL_GUARD,
    build_dataset,
    dataset_stats,
    oracle_should_swap,
    sample_spec,
    spec_to_context,
)
from ml_research.benchmarks.swap_net.model import SwapNet, save_swap_net
from ml_research.benchmarks.swap_net.policy import urgent_aware_decide
from ml_research.common.paths import CKPT, RESULTS


def train(
    *,
    epochs: int = 14,
    batch: int = 256,
    lr: float = 2.5e-3,
    n_samples: int = 20000,
    seed: int = 42,
    out: Path | None = None,
) -> dict:
    torch.manual_seed(seed)
    x, y, w, tags = build_dataset(n=n_samples, seed=seed)
    stats = dataset_stats(y, tags)
    print("[swapnet] dataset mix:", flush=True)
    for k in sorted(stats):
        d = stats[k]
        print(
            f"  {k:<32} n={d['n']:>5} swap_rate={d['swap_rate']:.2f}",
            flush=True,
        )
    print(
        f"[swapnet] overall swap_rate={float(y.mean()):.3f} "
        f"no_swap={int((y < 0.5).sum())} swap={int((y > 0.5).sum())}",
        flush=True,
    )

    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    split = int(n * 0.9)
    tr, va = idx[:split], idx[split:]

    net = SwapNet(feat_dim=FEAT_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def run_epoch(sub, train_mode: bool):
        if train_mode:
            net.train()
        else:
            net.eval()
        losses: List[float] = []
        correct = 0
        no_swap_correct = no_swap_n = 0
        swap_correct = swap_n = 0
        for i in range(0, len(sub), batch):
            b = sub[i : i + batch]
            xb = torch.from_numpy(x[b])
            yb = torch.from_numpy(y[b])
            wb = torch.from_numpy(w[b])
            if train_mode:
                opt.zero_grad()
                logits = net(xb)
                loss = (loss_fn(logits, yb) * wb).mean()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            else:
                with torch.no_grad():
                    prob = torch.sigmoid(net(xb))
                    pred = (prob >= 0.5).float()
                    correct += int((pred == yb).sum().item())
                    ns = yb < 0.5
                    ys = yb >= 0.5
                    if ns.any():
                        no_swap_n += int(ns.sum())
                        no_swap_correct += int((pred[ns] == yb[ns]).sum().item())
                    if ys.any():
                        swap_n += int(ys.sum())
                        swap_correct += int((pred[ys] == yb[ys]).sum().item())
        if train_mode:
            return {"loss": float(np.mean(losses)) if losses else 0.0}
        return {
            "acc": correct / max(1, len(sub)),
            "no_swap_acc": no_swap_correct / max(1, no_swap_n),
            "swap_acc": swap_correct / max(1, swap_n),
            "no_swap_n": no_swap_n,
            "swap_n": swap_n,
        }

    history = []
    for ep in range(1, epochs + 1):
        tr_m = run_epoch(tr, True)
        va_m = run_epoch(va, False)
        history.append({"epoch": ep, **tr_m, **va_m})
        print(
            f"[swapnet] ep={ep} loss={tr_m['loss']:.4f} "
            f"val_acc={va_m['acc']:.3f} "
            f"no_swap_acc={va_m['no_swap_acc']:.3f} "
            f"swap_acc={va_m['swap_acc']:.3f}",
            flush=True,
        )

    out_path = save_swap_net(
        net,
        out or CKPT / "swap_net_v3.pt",
        history=history[-3:],
        dataset_stats=stats,
        version="v3_no_swap_hardneg",
    )
    save_swap_net(
        net,
        CKPT / "swap_net.pt",
        history=history[-3:],
        dataset_stats=stats,
        version="v3_no_swap_hardneg",
    )
    return {"ckpt": str(out_path), "last": history[-1], "dataset_stats": stats, "net": net}


def eval_no_swap_scenarios(net: SwapNet, *, n_per: int = 200, seed: int = 99) -> Dict[str, dict]:
    """Probe dedicated no-swap / yes-swap generators with hybrid policy."""
    from ml_research.benchmarks.swap_net import micro_env as me

    probes = [
        "no_swap_far_at_door",
        "no_swap_gap_small",
        "no_swap_near_holds_urgent",
        "no_swap_deep_queue_marginal",
        "no_swap_urgent_already_ok",
        "no_swap_both_miss_sla",
        "no_swap_urgent_tiny_gap",
        "yes_swap_dock_wander",
        "yes_swap_urgent_rescue",
    ]
    rng = __import__("random").Random(seed)
    report: Dict[str, dict] = {}
    for name in probes:
        sampler = me._SAMPLERS[name]
        want_swap = name.startswith("yes_swap_")
        ok = 0
        net_only_ok = 0
        rule_reject = 0
        for _ in range(n_per):
            ctx, y, _ = spec_to_context(sampler(rng))
            if name.startswith("no_swap_"):
                y = 0
            elif name.startswith("yes_swap_"):
                y = 1
            dec = urgent_aware_decide(ctx, net)
            correct = (dec.swap is True) == (y == 1)
            ok += int(correct)
            if dec.source == "rule" and not dec.swap:
                rule_reject += 1
            p = net.prob(build_swap_features(ctx))
            thr = 0.35 if str(ctx.task_info.get("priority", "")).lower() == "urgent" else 0.55
            net_pred = p >= thr
            net_only_ok += int(net_pred == (y == 1))
        report[name] = {
            "n": n_per,
            "hybrid_acc": ok / n_per,
            "net_only_acc": net_only_ok / n_per,
            "rule_reject_frac": rule_reject / n_per,
            "expect_swap": want_swap,
        }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train SwapNet with no-swap hard negatives")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--samples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    summary = train(
        epochs=args.epochs,
        n_samples=args.samples,
        seed=args.seed,
        out=args.out,
    )
    net = summary.pop("net")
    probe = eval_no_swap_scenarios(net)
    print("\n[swapnet] no-swap / yes-swap probe (hybrid policy):", flush=True)
    for k, v in probe.items():
        print(
            f"  {k:<32} hybrid={v['hybrid_acc']:.3f} "
            f"net={v['net_only_acc']:.3f} rule_rej={v['rule_reject_frac']:.2f}",
            flush=True,
        )

    out_dir = RESULTS / "lifelong" / "swapnet_ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ckpt": summary["ckpt"],
        "last": summary["last"],
        "dataset_stats": summary["dataset_stats"],
        "probe": probe,
        "far_arrival_guard": FAR_ARRIVAL_GUARD,
    }
    path = out_dir / "train_no_swap_report.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[swapnet] saved {summary['ckpt']}", flush=True)
    print(f"[swapnet] report -> {path}", flush=True)

    # Pass bar: all no_swap_* hybrid_acc >= 0.95; yes_swap_* >= 0.85
    fails = []
    for k, v in probe.items():
        thr = 0.95 if k.startswith("no_swap_") else 0.85
        if v["hybrid_acc"] < thr:
            fails.append((k, v["hybrid_acc"], thr))
    if fails:
        print(f"[swapnet] WARN below bar: {fails}", flush=True)
        return 1
    print("[swapnet] PASS no-swap discrimination bar", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
