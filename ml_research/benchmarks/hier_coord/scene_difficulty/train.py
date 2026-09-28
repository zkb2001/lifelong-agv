"""Train SceneDifficultyNet from M0-labeled samples (completion-qualified labels)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ml_research.benchmarks.hier_coord.scene_difficulty.model import (
    SCENE_DIFF_CKPT,
    SceneDifficultyNet,
    save_scene_difficulty_net,
)
from ml_research.common.paths import RESULTS

DEFAULT_DATA = RESULTS / "hier_coord" / "scene_difficulty"


class SceneDiffDataset(Dataset):
    def __init__(self, sample_dirs: List[Path]):
        self.sample_dirs = sample_dirs

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, i: int):
        d = self.sample_dirs[i]
        blob = np.load(d / "obs.npz")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        maps = blob["maps"].astype(np.float32)
        feats = blob["feats"].astype(np.float32)
        y = int(meta["label_int"])
        cr = float(meta.get("completion_ratio") or 0.0)
        return (
            torch.from_numpy(maps),
            torch.from_numpy(feats),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(cr, dtype=torch.float32),
        )


def _collect(data_root: Path, *, subdir: str = "samples") -> List[Path]:
    samples = data_root / subdir
    if not samples.exists():
        return []
    return [
        d
        for d in sorted(samples.iterdir())
        if (d / "obs.npz").exists() and (d / "meta.json").exists()
    ]


def _val_metrics(net: SceneDifficultyNet, loader: DataLoader, threshold: float) -> dict:
    net.eval()
    yt: List[int] = []
    ph: List[float] = []
    with torch.no_grad():
        for batch in loader:
            maps, feats, y = batch[0], batch[1], batch[2]
            logits = net(maps, feats)
            prob = F.softmax(logits, dim=-1)[:, 1]
            yt.extend(int(v) for v in y.tolist())
            ph.extend(float(v) for v in prob.tolist())
    pred = [1 if p >= threshold else 0 for p in ph]
    n = len(yt)
    acc = sum(int(a == b) for a, b in zip(yt, pred)) / max(1, n)
    tp = sum(1 for t, p in zip(yt, pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(yt, pred) if t == 1 and p == 0)
    fp = sum(1 for t, p in zip(yt, pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(yt, pred) if t == 0 and p == 0)
    rec_h = tp / max(1, tp + fn)
    return {
        "acc": acc,
        "hard_recall": rec_h,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "n": n,
    }


def calibrate_threshold(
    net: SceneDifficultyNet,
    loader: DataLoader,
    *,
    min_hard_recall: float = 0.95,
    target_acc: float = 0.90,
) -> tuple[float, dict]:
    """Pick threshold maximizing balanced accuracy (no hard-recall floor).

    If ``min_hard_recall`` > 0, optionally restrict candidates (legacy fail-closed).
    Default balanced policy uses min_hard_recall=0.
    """
    net.eval()
    yt: List[int] = []
    ph: List[float] = []
    with torch.no_grad():
        for batch in loader:
            maps, feats, y = batch[0], batch[1], batch[2]
            logits = net(maps, feats)
            prob = F.softmax(logits, dim=-1)[:, 1]
            yt.extend(int(v) for v in y.tolist())
            ph.extend(float(v) for v in prob.tolist())

    def _at(t: float) -> dict:
        pred = [1 if p >= t else 0 for p in ph]
        tp = sum(1 for a, b in zip(yt, pred) if a == 1 and b == 1)
        fn = sum(1 for a, b in zip(yt, pred) if a == 1 and b == 0)
        fp = sum(1 for a, b in zip(yt, pred) if a == 0 and b == 1)
        tn = sum(1 for a, b in zip(yt, pred) if a == 0 and b == 0)
        rec = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        acc = (tp + tn) / max(1, len(yt))
        tpr = rec
        tnr = tn / max(1, tn + fp)
        bal = 0.5 * (tpr + tnr)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {
            "threshold": t,
            "acc": acc,
            "bal_acc": bal,
            "f1_hard": f1,
            "hard_recall": rec,
            "fp": fp,
            "fn": fn,
        }

    cands = [_at(i / 100.0) for i in range(5, 96)]
    if float(min_hard_recall) > 0:
        feasible = [c for c in cands if c["hard_recall"] + 1e-12 >= float(min_hard_recall)]
        pool = feasible if feasible else cands
    else:
        pool = cands
    best = max(
        pool,
        key=lambda c: (
            c["bal_acc"],
            c["acc"],
            c["f1_hard"],
            -abs(c["acc"] - float(target_acc)),
            c["threshold"],
        ),
    )
    return float(best["threshold"]), best


def calibrate_easy_max(
    net: SceneDifficultyNet,
    loader: DataLoader,
    *,
    threshold: float,
) -> float:
    """Largest easy_max <= threshold with zero hard→easy among confident-easy claims."""
    net.eval()
    yt: List[int] = []
    ph: List[float] = []
    with torch.no_grad():
        for batch in loader:
            maps, feats, y = batch[0], batch[1], batch[2]
            logits = net(maps, feats)
            prob = F.softmax(logits, dim=-1)[:, 1]
            yt.extend(int(v) for v in y.tolist())
            ph.extend(float(v) for v in prob.tolist())
    best = 0.0
    for j in range(0, int(threshold * 100) + 1):
        e = j / 100.0
        fn = sum(1 for t, p in zip(yt, ph) if t == 1 and p <= e)
        if fn == 0:
            best = e
    return float(best)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(SCENE_DIFF_CKPT))
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--hard-weight", type=float, default=1.0, help="extra multiplier on hard class")
    ap.add_argument(
        "--min-hard-recall",
        type=float,
        default=0.0,
        help="optional hard-recall floor when calibrating (0=balanced, no fail-closed)",
    )
    ap.add_argument("--target-acc", type=float, default=0.90)
    ap.add_argument("--cr-loss-w", type=float, default=0.35, help="aux completion-ratio MSE weight")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dirs = _collect(Path(args.data))
    if len(dirs) < 4:
        print(f"[train] need >=4 samples, got {len(dirs)} under {args.data}", flush=True)
        return 1

    random.shuffle(dirs)
    n_val = max(1, int(len(dirs) * float(args.val_ratio)))
    val_dirs = dirs[:n_val]
    train_dirs = dirs[n_val:]
    print(
        f"[train] n_train={len(train_dirs)} n_val={len(val_dirs)} "
        f"(held-out test is separate; not used here)",
        flush=True,
    )
    train_ds = SceneDiffDataset(train_dirs)
    val_ds = SceneDiffDataset(val_dirs)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    ys = []
    for d in train_dirs:
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        ys.append(int(m["label_int"]))
    n0 = max(1, ys.count(0))
    n1 = max(1, ys.count(1))
    w = torch.tensor(
        [len(ys) / (2 * n0), len(ys) / (2 * n1) * float(args.hard_weight)],
        dtype=torch.float32,
    )
    print(f"[train] class_counts easy={n0} hard={n1} weights={w.tolist()}", flush=True)

    net = SceneDifficultyNet()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    best_score = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        net.train()
        total = 0.0
        for maps, feats, y, cr in train_loader:
            opt.zero_grad()
            logits, cr_hat = net.forward_with_cr(maps, feats)
            loss_cls = F.cross_entropy(logits, y, weight=w)
            loss_cr = F.mse_loss(cr_hat, cr)
            loss = loss_cls + float(args.cr_loss_w) * loss_cr
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
        # select by overall accuracy (balanced; no hard-recall over-weight)
        m = _val_metrics(net, val_loader, threshold=0.5)
        score = float(m["acc"])
        print(
            f"[train] ep={ep} loss={total/max(1,len(train_ds)):.4f} "
            f"val_acc@0.5={m['acc']:.3f} hard_rec={m['hard_recall']:.3f} "
            f"fn={m['fn']} fp={m['fp']}",
            flush=True,
        )
        if score >= best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    thr, cal = calibrate_threshold(
        net,
        val_loader,
        min_hard_recall=float(args.min_hard_recall),
        target_acc=float(args.target_acc),
    )
    easy_max = float(thr)  # unused under balanced decide(); keep equal to thr
    path = save_scene_difficulty_net(
        net,
        Path(args.out),
        val_score=best_score,
        n_train=len(train_dirs),
        n_val=len(val_dirs),
        decision_threshold=thr,
        easy_max=easy_max,
        calibration=cal,
        min_hard_recall=float(args.min_hard_recall),
        target_acc=float(args.target_acc),
        label_policy="first_wave_st",
        decision_policy="balanced_single_threshold",
        feat_policy="near_horizon_v3",
    )
    print(
        f"[train] saved {path} calib_threshold={thr:.2f} easy_max={easy_max:.2f} "
        f"val={cal}",
        flush=True,
    )
    # also write sidecar for gate
    side = Path(args.out).with_suffix(".decision.json")
    side.write_text(
        json.dumps(
            {
                "threshold": thr,
                "easy_max": easy_max,
                "min_hard_recall": float(args.min_hard_recall),
                "target_acc": float(args.target_acc),
                "calibration": cal,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[train] wrote {side}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
