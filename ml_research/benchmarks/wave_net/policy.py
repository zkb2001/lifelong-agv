"""Wave selection policy: rule teacher + optional WaveNet ranking."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from ml_research.common.paths import CKPT

from .features import (
    WaveAgentContext,
    build_wave_contexts,
    build_wave_features,
)
from .model import WaveNet, load_wave_net

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]


@dataclass
class WaveSelectResult:
    keepers: Dict[str, dict]
    dropped: Dict[str, dict]
    scores: Dict[str, float] = field(default_factory=dict)
    mode: str = "passthrough"
    k_budget: int = 0


def rule_score(ctx: WaveAgentContext) -> float:
    """Higher = more valuable in this wave.

    Prefer: seed, urgent, short ETA, low pairwise corridor overlap,
    lower historical work (fairness). On hard maps, penalize packing the
    same destination into one joint wave (hotspot congestion).
    """
    s = 0.0
    s += 3.0 if ctx.is_seed else 0.0
    s += 2.0 * float(ctx.urgency)
    s += 2.5 * (1.0 - min(1.0, ctx.eta / 40.0))
    s += 1.5 * (1.0 - float(ctx.pair_overlap))
    s += 0.8 * (1.0 - min(1.0, ctx.work_count / 20.0))
    s -= 0.6 * float(ctx.local_obs)
    s -= 1.2 * float(ctx.map_hardness) * float(ctx.conflict_proxy)
    dest_share = float(getattr(ctx, "dest_share", 0.0))
    s -= 1.0 * float(ctx.map_hardness) * max(0.0, dest_share - 0.34)
    if ctx.eta >= 10**5:
        s -= 8.0
    return float(s)


def score_agent(
    ctx: WaveAgentContext,
    net: Optional[object],
    *,
    map_obs: Optional[np.ndarray] = None,
    net_weight: float = 0.9,
) -> Tuple[float, str]:
    if net is None:
        return rule_score(ctx), "rule"
    feat = build_wave_features(ctx)
    w = float(net_weight)
    w = min(1.0, max(0.0, w))
    # WaveNetMap: needs map channels
    if hasattr(net, "encode_map") and map_obs is not None:
        return w * float(net.score(map_obs, feat)) + (1.0 - w) * rule_score(ctx), "map_net"
    if hasattr(net, "score") and not hasattr(net, "encode_map"):
        return w * float(net.score(feat)) + (1.0 - w) * rule_score(ctx), "net"
    return rule_score(ctx), "rule"


def select_wave_agents(
    assigned: Dict[str, dict],
    pose: Dict[str, Pose],
    static: Set[Cell],
    bfs_len: Callable[[Cell, Cell, Set[Cell]], int],
    work_count: Dict[str, int],
    *,
    escalate: bool,
    k_budget: int,
    net: Optional[object] = None,
    seed_agvs: Optional[Set[str]] = None,
    map_hardness: float = 0.0,
    protect_seeds: bool = True,
    map_obs: Optional[np.ndarray] = None,
) -> WaveSelectResult:
    """If not escalate: return assigned unchanged (easy-map fast path).

    If escalate: keep up to k_budget agents by WaveNet/rule score; drop rest.
    """
    if not assigned:
        return WaveSelectResult(keepers={}, dropped={}, mode="empty", k_budget=0)
    if not escalate:
        return WaveSelectResult(
            keepers=dict(assigned),
            dropped={},
            mode="passthrough",
            k_budget=len(assigned),
        )

    k = max(1, min(int(k_budget), len(assigned)))
    ctxs = build_wave_contexts(
        assigned,
        pose,
        static,
        bfs_len,
        work_count,
        map_hardness=map_hardness,
        k_budget=k,
        seed_agvs=seed_agvs,
    )
    scored: List[Tuple[float, str, WaveAgentContext, str]] = []
    mode = "rule"
    for ctx in ctxs:
        sc, m = score_agent(ctx, net, map_obs=map_obs)
        mode = m
        scored.append((sc, ctx.agv, ctx, m))
    scored.sort(key=lambda t: (-t[0], t[1]))

    keep_ids: List[str] = []
    seeds = seed_agvs or set()
    if protect_seeds:
        for sc, agv, ctx, _m in scored:
            if agv in seeds and agv not in keep_ids:
                keep_ids.append(agv)
            if len(keep_ids) >= k:
                break
    for sc, agv, ctx, _m in scored:
        if agv not in keep_ids:
            keep_ids.append(agv)
        if len(keep_ids) >= k:
            break

    keepers = {a: assigned[a] for a in keep_ids if a in assigned}
    dropped = {a: t for a, t in assigned.items() if a not in keepers}
    scores = {agv: float(sc) for sc, agv, _c, _m in scored}
    return WaveSelectResult(
        keepers=keepers,
        dropped=dropped,
        scores=scores,
        mode=mode,
        k_budget=k,
    )


def load_wave_policy(
    *,
    enabled: bool,
    ckpt: Optional[Path] = None,
) -> Tuple[Optional[object], dict]:
    meta: dict = {"enabled": bool(enabled), "mode": "off", "ckpt": None}
    if not enabled:
        return None, meta
    map_v2 = CKPT / "wave_net_map_v2.pt"
    map_path = CKPT / "wave_net_map_v1.pt"
    path = Path(ckpt) if ckpt else (map_v2 if map_v2.exists() else map_path)
    # Prefer map-conditioned ckpt when present
    if ckpt is None and not path.exists() and map_path.exists():
        path = map_path
    if path.exists():
        try:
            if "map" in path.name:
                from ml_research.benchmarks.hier_coord.wave_map import load_wave_net_map

                net = load_wave_net_map(path)
                meta["mode"] = "map_net"
            else:
                net = load_wave_net(path)
                meta["mode"] = "net"
            meta["ckpt"] = str(path)
            return net, meta
        except Exception as exc:  # noqa: BLE001
            meta["mode"] = "rule_fallback"
            meta["load_error"] = str(exc)
            return None, meta
    # fallback scalar wave net
    legacy = CKPT / "wave_net_v1.pt"
    if legacy.exists():
        try:
            net = load_wave_net(legacy)
            meta["mode"] = "net"
            meta["ckpt"] = str(legacy)
            return net, meta
        except Exception as exc:  # noqa: BLE001
            meta["mode"] = "rule_fallback"
            meta["load_error"] = str(exc)
            return None, meta
    meta["mode"] = "rule_only"
    return None, meta
