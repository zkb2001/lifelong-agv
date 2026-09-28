"""Smoke-eval WaveNetMap: rule vs net ranking agreement with set-aware oracle."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ml_research.benchmarks.hier_coord.train_wave_v2 import (
    _bfs_len,
    _load_slot_geometry,
    _spearman,
    _synth_wave_scene,
    oracle_teacher_scores,
)
from ml_research.benchmarks.hier_coord.wave_map import load_wave_net_map
from ml_research.benchmarks.wave_net.features import build_wave_contexts, build_wave_features
from ml_research.benchmarks.wave_net.policy import rule_score, select_wave_agents
from ml_research.common.paths import CKPT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ckpt", type=str, default=str(CKPT / "wave_net_map_v2.pt"))
    args = ap.parse_args()

    geos = []
    for s in range(1, 21):
        g = _load_slot_geometry(s)
        if g and g["free"]:
            geos.append(g)
    rng = random.Random(args.seed)
    net = load_wave_net_map(Path(args.ckpt))
    net.eval()

    rho_rule, rho_net = [], []
    topk_rule, topk_net = [], []

    for _ in range(args.n):
        geo = rng.choice(geos)
        scene = _synth_wave_scene(geo, rng)
        if scene is None or len(scene["assigned"]) < 2:
            continue
        ctxs = build_wave_contexts(
            scene["assigned"],
            scene["pose"],
            scene["geo"]["static"],
            _bfs_len,
            scene["work"],
            map_hardness=float(scene["geo"]["hardness"]),
            k_budget=int(scene["k_budget"]),
            seed_agvs=scene["seeds"],
        )
        if len(ctxs) < 2:
            continue
        teacher = oracle_teacher_scores(
            ctxs,
            scene["pose"],
            scene["assigned"],
            k_budget=int(scene["k_budget"]),
            top_share=float(scene["top_share"]),
            map_hardness=float(scene["geo"]["hardness"]),
            rng=rng,
        )
        rule = np.array([rule_score(c) for c in ctxs], dtype=np.float32)
        feats = np.stack([build_wave_features(c) for c in ctxs]).astype(np.float32)
        with torch.no_grad():
            m = torch.from_numpy(scene["maps"].astype(np.float32)).unsqueeze(0)
            f = torch.from_numpy(feats)
            pred = net(m, f).cpu().numpy()

        rho_rule.append(_spearman(rule, teacher))
        rho_net.append(_spearman(pred, teacher))

        k = int(scene["k_budget"])
        oracle_keep = set(
            [c.agv for c, _ in sorted(zip(ctxs, teacher), key=lambda t: -t[1])[:k]]
        )
        rule_keep = set(
            select_wave_agents(
                scene["assigned"],
                scene["pose"],
                scene["geo"]["static"],
                _bfs_len,
                scene["work"],
                escalate=True,
                k_budget=k,
                net=None,
                seed_agvs=scene["seeds"],
                map_hardness=float(scene["geo"]["hardness"]),
                map_obs=scene["maps"],
            ).keepers
        )
        net_keep = set(
            select_wave_agents(
                scene["assigned"],
                scene["pose"],
                scene["geo"]["static"],
                _bfs_len,
                scene["work"],
                escalate=True,
                k_budget=k,
                net=net,
                seed_agvs=scene["seeds"],
                map_hardness=float(scene["geo"]["hardness"]),
                map_obs=scene["maps"],
            ).keepers
        )
        topk_rule.append(len(oracle_keep & rule_keep) / float(k))
        topk_net.append(len(oracle_keep & net_keep) / float(k))

    print(
        f"scenes={len(rho_net)} ckpt={args.ckpt}\n"
        f"spearman  rule={np.mean(rho_rule):.3f}  net={np.mean(rho_net):.3f}\n"
        f"topk@k    rule={np.mean(topk_rule):.3f}  net={np.mean(topk_net):.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
