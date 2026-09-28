"""StallEscalateNet — predict M0→ECBS handoff from runtime stall features.

Used inside ``_should_stop_gate`` to cut long M0 wall-clock thrash
(SH06/08/12/13-style) while leaving healthy M0 (SH01/04) alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from ml_research.common.paths import CKPT

STALL_ESCALATE_CKPT = CKPT / "escalate_net_stall_v1.pt"
STALL_FEAT_DIM = 14
STALL_FEAT_NAMES = (
    "map_hardness",
    "narrow_cut",
    "done_ratio",
    "plateau_n",
    "idle_streak_n",
    "assign_fail_n",
    "plan_fail_rate",
    "throughput_drop_n",
    "wall_tick_n",
    "wall_window_hit",
    "sim_per_task_n",
    "wall_elapsed_n",
    "has_work",
    "loaded_ratio",
)


@dataclass
class StallEscalateContext:
    map_hardness: float = 0.0
    narrow_cut: float = 0.0
    done_ratio: float = 0.0
    plateau_streak: int = 0
    idle_surface_streak: int = 0
    assign_fail_streak: int = 0
    plan_fail_rate: float = 0.0
    throughput_drop_streak: int = 0
    wall_tick_streak: int = 0
    wall_window_hit: bool = False
    sim_per_task: float = 0.0
    wall_elapsed_s: float = 0.0
    has_work: bool = True
    loaded_ratio: float = 0.0


def build_stall_features(ctx: StallEscalateContext) -> np.ndarray:
    feat = np.array(
        [
            float(np.clip(ctx.map_hardness, 0.0, 1.0)),
            float(np.clip(ctx.narrow_cut, 0.0, 1.0)),
            float(np.clip(ctx.done_ratio, 0.0, 1.0)),
            min(1.0, float(ctx.plateau_streak) / 24.0),
            min(1.0, float(ctx.idle_surface_streak) / 8.0),
            min(1.0, float(ctx.assign_fail_streak) / 40.0),
            float(np.clip(ctx.plan_fail_rate, 0.0, 1.0)),
            min(1.0, float(ctx.throughput_drop_streak) / 8.0),
            min(1.0, float(ctx.wall_tick_streak) / 6.0),
            1.0 if ctx.wall_window_hit else 0.0,
            min(1.0, float(ctx.sim_per_task) / 80.0),
            min(1.0, float(ctx.wall_elapsed_s) / 300.0),
            1.0 if ctx.has_work else 0.0,
            float(np.clip(ctx.loaded_ratio, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    assert feat.shape[0] == STALL_FEAT_DIM
    return feat


def stall_context_from_runtime(
    *,
    map_hardness: float,
    narrow_cut: float,
    n_tasks: int,
    done_completed: int,
    stall_state: Dict[str, Any],
    sig: Dict[str, Any],
    wall_elapsed_s: float,
    sim_t: int,
) -> StallEscalateContext:
    n_tasks = max(1, int(n_tasks))
    done = max(0, int(done_completed))
    n_agv = max(
        1,
        int(sig.get("n_idle", 0) or 0)
        + int(sig.get("n_busy", 0) or 0)
        + int(sig.get("n_loaded", 0) or 0),
    )
    # n_idle+n_busy may already cover fleet; prefer loaded/busy split.
    n_loaded = int(sig.get("n_loaded") or 0)
    sim_per_task = float(sim_t) / max(1, done)
    return StallEscalateContext(
        map_hardness=float(map_hardness),
        narrow_cut=float(narrow_cut),
        done_ratio=float(done) / float(n_tasks),
        plateau_streak=int(stall_state.get("done_plateau_streak") or 0),
        idle_surface_streak=int(stall_state.get("idle_surface_streak") or 0),
        assign_fail_streak=int(sig.get("assign_fail_streak") or 0),
        plan_fail_rate=float(sig.get("plan_fail_rate") or 0.0),
        throughput_drop_streak=int(stall_state.get("throughput_drop_streak") or 0),
        wall_tick_streak=int(stall_state.get("wall_tick_streak") or 0),
        wall_window_hit=bool(stall_state.get("wall_window_hit")),
        sim_per_task=float(sim_per_task),
        wall_elapsed_s=float(wall_elapsed_s),
        has_work=bool(sig.get("has_work")),
        loaded_ratio=float(n_loaded) / float(n_agv),
    )


def teacher_stall_escalate_prob(ctx: StallEscalateContext) -> float:
    """Soft label in [0,1]: escalate when M0 is thrashing, stay when healthy."""
    if not ctx.has_work:
        return 0.05
    if ctx.done_ratio >= 0.97:
        return 0.02

    # Healthy M0: progressing with good sim/task.
    healthy = (
        ctx.done_ratio >= 0.05
        and ctx.sim_per_task <= 35.0
        and ctx.plateau_streak < 8
        and ctx.wall_elapsed_s < 120.0
    )
    if healthy:
        return 0.08

    score = 0.0
    # Long plateau after recover window → escalate.
    if ctx.plateau_streak >= 20:
        score = max(score, 0.95)
    elif ctx.plateau_streak >= 14:
        score = max(score, 0.82)
    elif ctx.plateau_streak >= 10:
        score = max(score, 0.65)
    elif ctx.plateau_streak >= 6:
        score = max(score, 0.40)

    if ctx.wall_window_hit and ctx.plateau_streak >= 3:
        score = max(score, 0.75)
    if ctx.idle_surface_streak >= 2 and ctx.plateau_streak >= 4:
        score = max(score, 0.70)
    if ctx.assign_fail_streak >= 12 and ctx.plateau_streak >= 4:
        score = max(score, 0.68)
    if ctx.plan_fail_rate >= 0.35 and ctx.plateau_streak >= 4:
        score = max(score, 0.66)
    if ctx.throughput_drop_streak >= 3 and ctx.plateau_streak >= 4:
        score = max(score, 0.60)

    # Wall thrash with little progress (SH06/08/12 pattern).
    if ctx.wall_elapsed_s >= 90.0 and ctx.done_ratio < 0.25 and ctx.plateau_streak >= 6:
        score = max(score, 0.88)
    if ctx.wall_elapsed_s >= 150.0 and ctx.done_ratio < 0.45 and ctx.plateau_streak >= 8:
        score = max(score, 0.92)

    # Poor sim/task while stalled.
    if ctx.sim_per_task >= 45.0 and ctx.plateau_streak >= 8 and ctx.done_ratio < 0.8:
        score = max(score, 0.80)

    # Dense / cut maps: escalate a bit earlier when already stalling.
    structure = 0.55 * float(ctx.map_hardness) + 0.45 * float(ctx.narrow_cut)
    if structure >= 0.25 and ctx.plateau_streak >= 8 and ctx.done_ratio < 0.7:
        score = max(score, min(0.9, 0.55 + structure))

    # Early run: keep low unless severe stall.
    if ctx.wall_elapsed_s < 40.0 and ctx.plateau_streak < 12:
        score *= 0.55

    return float(np.clip(score, 0.0, 1.0))


class StallEscalateNet(nn.Module):
    def __init__(self, feat_dim: int = STALL_FEAT_DIM, hidden: int = 48):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def prob(self, x: Union[torch.Tensor, np.ndarray]) -> float:
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(self.forward(x)).item())


def load_stall_escalate_net(
    ckpt: Optional[Path] = None,
    device: str = "cpu",
) -> StallEscalateNet:
    path = Path(ckpt) if ckpt else STALL_ESCALATE_CKPT
    net = StallEscalateNet()
    if path.exists():
        blob = torch.load(path, map_location=device, weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def save_stall_escalate_net(
    net: StallEscalateNet, path: Optional[Path] = None, **meta
) -> Path:
    path = Path(path or STALL_ESCALATE_CKPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "feat_dim": net.feat_dim,
            "feat_names": STALL_FEAT_NAMES,
            "version": "stall_v1",
            **meta,
        },
        path,
    )
    return path


def decide_stall_escalate(
    ctx: StallEscalateContext,
    net: Optional[StallEscalateNet],
    *,
    threshold: float = 0.58,
    min_plateau: int = 12,
) -> tuple[bool, str, float]:
    """Return (escalate, source, p).

    Safety: never escalate without work; respect min_plateau unless wall_window.
    """
    if not ctx.has_work or ctx.done_ratio >= 0.97:
        return False, "stall_skip_done", 0.0

    if net is None:
        p = teacher_stall_escalate_prob(ctx)
        src = "teacher"
    else:
        p = net.prob(build_stall_features(ctx))
        src = "stall_net"

    early_ok = bool(ctx.wall_window_hit) and ctx.plateau_streak >= 6
    if ctx.plateau_streak < int(min_plateau) and not early_ok:
        return False, f"{src}_cold_plateau", p

    # Protect clearly healthy M0.
    if (
        ctx.done_ratio >= 0.05
        and ctx.sim_per_task <= 32.0
        and ctx.plateau_streak < 10
        and not ctx.wall_window_hit
    ):
        return False, f"{src}_healthy", p

    thr = float(threshold)
    if ctx.wall_window_hit or ctx.wall_elapsed_s >= 120.0:
        thr = min(thr, 0.50)
    if p >= thr:
        return True, src, p
    return False, f"{src}_easy", p
