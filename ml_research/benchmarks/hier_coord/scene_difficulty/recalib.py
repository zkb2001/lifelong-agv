"""Recalibrate decision bands on an existing checkpoint (no retrain)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ml_research.benchmarks.hier_coord.scene_difficulty.model import (
    SCENE_DIFF_CKPT,
    load_scene_difficulty_net,
    save_scene_difficulty_net,
)
from ml_research.benchmarks.hier_coord.scene_difficulty.train import (
    SceneDiffDataset,
    _collect,
    calibrate_easy_max,
    calibrate_threshold,
)
from ml_research.common.paths import RESULTS

DEFAULT_DATA = RESULTS / "hier_coord" / "scene_difficulty"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=str(DEFAULT_DATA))
    ap.add_argument("--ckpt", type=str, default=str(SCENE_DIFF_CKPT))
    ap.add_argument("--min-hard-recall", type=float, default=0.0)
    ap.add_argument("--target-acc", type=float, default=0.90)
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    dirs = _collect(Path(args.data))
    random.shuffle(dirs)
    n_val = max(1, int(len(dirs) * float(args.val_ratio)))
    val_dirs = dirs[:n_val]
    loader = DataLoader(SceneDiffDataset(val_dirs), batch_size=64, shuffle=False)

    ckpt = Path(args.ckpt)
    net = load_scene_difficulty_net(ckpt)
    thr, cal = calibrate_threshold(
        net,
        loader,
        min_hard_recall=float(args.min_hard_recall),
        target_acc=float(args.target_acc),
    )
    easy_max = float(thr)
    net.decision_threshold = thr
    net.easy_max = easy_max

    # reload full blob meta, overwrite thresholds
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        blob = {"model": blob}
    blob["decision_threshold"] = thr
    blob["easy_max"] = easy_max
    blob["calibration"] = cal
    blob["min_hard_recall"] = float(args.min_hard_recall)
    blob["target_acc"] = float(args.target_acc)
    blob["decision_policy"] = "balanced_single_threshold"
    torch.save(blob, ckpt)

    side = ckpt.with_suffix(".decision.json")
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
    print(
        f"[recalib] thr={thr:.3f} easy_max={easy_max:.3f} val_cal={cal} wrote {side}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
