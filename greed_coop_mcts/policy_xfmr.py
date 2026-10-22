"""Runtime wrapper: joint Transformer policy over AGVs."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from greed_coop_mcts.obs import ACTIONS, action_mask, apply_action
from greed_coop_mcts.obs_xfmr import build_observation_xfmr
from greed_coop_mcts.transformer_net import (
    TRANSFORMER_CKPT,
    MapdTransformer,
    load_transformer,
)

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]

_NET: Dict[str, Optional[MapdTransformer]] = {}


def get_transformer(device: str = "cpu") -> Optional[MapdTransformer]:
    key = str(device)
    if key not in _NET:
        if not TRANSFORMER_CKPT.exists():
            _NET[key] = None
        else:
            try:
                _NET[key] = load_transformer(device=device)
                print(f"[xfmr] loaded {TRANSFORMER_CKPT}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[xfmr] load failed: {exc}", flush=True)
                _NET[key] = None
    return _NET[key]


def plan_desired(
    *,
    names: Sequence[str],
    pose: Dict[str, Pose],
    goals: Dict[str, Cell],
    blocked: Set[Cell],
    force_stay: Set[str],
    net: MapdTransformer,
    device: str = "cpu",
) -> Dict[str, Cell]:
    """One-step desired cells via joint Transformer communication."""
    active = [n for n in names if n not in force_stay]
    desired: Dict[str, Cell] = {n: pose[n][:2] for n in force_stay}
    if not active:
        return {n: pose[n][:2] for n in names}

    obs_l = []
    mask_l = []
    for n in active:
        pos = pose[n][:2]
        others = [pose[m][:2] for m in names if m != n]
        obs_l.append(build_observation_xfmr(pos, goals.get(n), others, blocked))
        mask_l.append(action_mask(pos, blocked))
    obs_n = np.stack(obs_l)
    mask_n = np.stack(mask_l)
    acts, _ = net.team_actions(obs_n, mask_n, device=device)

    claimed: Set[Cell] = {pose[n][:2] for n in force_stay}
    # Priority: keep order as given (caller should put unloaders first)
    for n, ai in zip(active, acts):
        a = ACTIONS[int(ai)]
        cur = pose[n][:2]
        nxt = apply_action(cur, a, blocked)
        if nxt in claimed and nxt != cur:
            nxt = cur
        desired[n] = nxt
        claimed.add(nxt)
    for n in names:
        desired.setdefault(n, pose[n][:2])
    return desired
