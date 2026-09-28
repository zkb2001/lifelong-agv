"""Evaluate SceneDifficultyNet against M0 A* labels on a held-out test set."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ml_research.benchmarks.hier_coord.scene_difficulty.model import (
    SCENE_DIFF_CKPT,
    CLASS_NAMES,
    SceneDifficultyNet,
    load_scene_difficulty_net,
)
from ml_research.common.paths import RESULTS

DEFAULT_TEST = RESULTS / "hier_coord" / "scene_difficulty_test"


def _collect(root: Path) -> List[Path]:
    samples = root / "samples"
    if not samples.exists():
        return []
    return [
        d
        for d in sorted(samples.iterdir())
        if (d / "obs.npz").exists() and (d / "meta.json").exists()
    ]


def evaluate(
    net: SceneDifficultyNet,
    sample_dirs: List[Path],
    *,
    threshold: float = 0.5,
) -> dict:
    net.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    conf: List[float] = []
    by_agv = defaultdict(lambda: {"n": 0, "correct": 0})
    by_pat = defaultdict(lambda: {"n": 0, "correct": 0})
    by_src = defaultdict(lambda: {"n": 0, "correct": 0})
    rows = []

    with torch.no_grad():
        for d in sample_dirs:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            blob = np.load(d / "obs.npz")
            maps = torch.from_numpy(blob["maps"].astype(np.float32))
            feats = torch.from_numpy(blob["feats"].astype(np.float32))
            yt = int(meta["label_int"])
            logits = net(maps, feats)
            prob = F.softmax(logits, dim=-1)[0]
            p_hard = float(prob[1].item())
            yp = 1 if p_hard >= float(threshold) else 0
            y_true.append(yt)
            y_pred.append(yp)
            conf.append(p_hard)
            ok = int(yp == yt)
            agv = int(meta.get("n_agvs") or 0)
            pat = str(meta.get("task_pattern") or "?")
            src = str(meta.get("source") or "?")
            by_agv[agv]["n"] += 1
            by_agv[agv]["correct"] += ok
            by_pat[pat]["n"] += 1
            by_pat[pat]["correct"] += ok
            by_src[src]["n"] += 1
            by_src[src]["correct"] += ok
            rows.append(
                {
                    "id": meta.get("id"),
                    "astar_label": CLASS_NAMES[yt],
                    "net_pred": CLASS_NAMES[yp],
                    "p_hard": round(p_hard, 4),
                    "correct": bool(ok),
                    "n_agvs": agv,
                    "task_pattern": pat,
                    "n_tasks": meta.get("n_tasks"),
                    "completion_ratio": meta.get("completion_ratio"),
                }
            )

    n = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    # confusion: rows=true easy/hard, cols=pred easy/hard
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    prec_h = tp / max(1, tp + fp)
    rec_h = tp / max(1, tp + fn)
    prec_e = tn / max(1, tn + fn)
    rec_e = tn / max(1, tn + fp)

    def _acc_map(d: dict) -> Dict[str, float]:
        return {
            str(k): round(v["correct"] / max(1, v["n"]), 4)
            for k, v in sorted(d.items(), key=lambda x: str(x[0]))
        }

    return {
        "n": n,
        "accuracy": round(correct / max(1, n), 4),
        "n_correct": correct,
        "confusion": {
            "tn_easy": tn,
            "fp_easy_as_hard": fp,
            "fn_hard_as_easy": fn,
            "tp_hard": tp,
        },
        "hard_precision": round(prec_h, 4),
        "hard_recall": round(rec_h, 4),
        "easy_precision": round(prec_e, 4),
        "easy_recall": round(rec_e, 4),
        "astar_label_hist": dict(Counter(CLASS_NAMES[t] for t in y_true)),
        "net_pred_hist": dict(Counter(CLASS_NAMES[p] for p in y_pred)),
        "acc_by_agv": _acc_map(by_agv),
        "acc_by_pattern": _acc_map(by_pat),
        "acc_by_source": _acc_map(by_src),
        "threshold": threshold,
        "rows": rows,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Eval scene-diff net vs M0 A* labels")
    ap.add_argument("--test-data", type=str, default=str(DEFAULT_TEST))
    ap.add_argument("--ckpt", type=str, default=str(SCENE_DIFF_CKPT))
    ap.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="hard threshold; <0 ⇒ use ckpt decision_threshold",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="",
        help="json report path (default: test-data/eval_report.json)",
    )
    args = ap.parse_args(argv)

    test_root = Path(args.test_data)
    dirs = _collect(test_root)
    if not dirs:
        print(f"[eval] no test samples under {test_root}/samples", flush=True)
        return 1
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"[eval] missing ckpt {ckpt}", flush=True)
        return 1

    net = load_scene_difficulty_net(ckpt)
    thr = float(args.threshold)
    if thr < 0:
        thr = float(getattr(net, "decision_threshold", 0.35))
    # Also report banded (deploy) decision + shrink-safe FN
    from ml_research.benchmarks.hier_coord.scene_difficulty.model import (
        decide_scene_difficulty,
    )

    report = evaluate(net, dirs, threshold=thr)
    banded = {
        "n": 0,
        "correct": 0,
        "fn": 0,
        "fp": 0,
        "tp": 0,
        "tn": 0,
        "shrink_ok_n": 0,
        "shrink_ok_fn": 0,
    }
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        blob = np.load(d / "obs.npz")
        is_hard, _src, _p, pol = decide_scene_difficulty(
            blob["maps"].astype(np.float32),
            blob["feats"].astype(np.float32),
            net,
        )
        yt = int(meta["label_int"])
        yp = 1 if is_hard else 0
        banded["n"] += 1
        banded["correct"] += int(yp == yt)
        if yt == 1 and yp == 1:
            banded["tp"] += 1
        elif yt == 1 and yp == 0:
            banded["fn"] += 1
        elif yt == 0 and yp == 1:
            banded["fp"] += 1
        else:
            banded["tn"] += 1
        if pol == "shrink_ok":
            banded["shrink_ok_n"] += 1
            if yt == 1:
                banded["shrink_ok_fn"] += 1
    banded["accuracy"] = round(banded["correct"] / max(1, banded["n"]), 4)
    banded["hard_recall"] = round(
        banded["tp"] / max(1, banded["tp"] + banded["fn"]), 4
    )
    report["deploy_banded"] = banded
    report["decision_threshold"] = thr
    report["easy_max"] = float(getattr(net, "easy_max", 0.12))

    out = Path(args.out) if args.out else (test_root / "eval_report.json")
    slim = dict(report)
    slim["rows"] = report["rows"]
    out.write_text(json.dumps(slim, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== SceneDifficultyNet vs M0 (held-out, qualify labels) ===", flush=True)
    print(
        f"n={report['n']}  accuracy={report['accuracy']:.4f}  threshold={thr:.3f}",
        flush=True,
    )
    print(f"confusion={report['confusion']}", flush=True)
    print(
        f"hard P/R={report['hard_precision']:.3f}/{report['hard_recall']:.3f}  "
        f"easy P/R={report['easy_precision']:.3f}/{report['easy_recall']:.3f}",
        flush=True,
    )
    print(
        f"deploy_decide acc={banded['accuracy']:.4f} hard_rec={banded['hard_recall']:.4f} "
        f"fn={banded['fn']} fp={banded['fp']} easy_max={report['easy_max']:.3f} "
        f"shrink_ok_n={banded['shrink_ok_n']} shrink_ok_fn={banded['shrink_ok_fn']}",
        flush=True,
    )
    print(f"label_hist={report['astar_label_hist']}", flush=True)
    print(f"net_hist={report['net_pred_hist']}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
