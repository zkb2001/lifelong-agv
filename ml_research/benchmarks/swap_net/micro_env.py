"""DockWander micro-env + oracle labels for SwapNet (no-swap hard negatives)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ml_research.benchmarks.common import is_urgent_task
from ml_research.benchmarks.swap_net.features import SwapContext, build_swap_features

# Far AGV this close to pickup → sunk cost; do not cancel.
FAR_ARRIVAL_GUARD = 2.0


@dataclass
class DockWanderSpec:
    far_xy: Tuple[int, int] = (1, 5)
    near_xy: Tuple[int, int] = (9, 5)
    pickup_xy: Tuple[int, int] = (10, 5)
    margin: float = 4.0
    urgent: bool = False
    remaining_time: float = 120.0
    near_holds_urgent: bool = False
    queue_depth: int = 0
    scenario: str = "random"
    # Optional direct ETA override (for hard cases without xy constraints)
    far_eta: Optional[float] = None
    near_eta: Optional[float] = None


def oracle_should_swap(ctx: SwapContext) -> bool:
    """Teacher: only swap when gain clearly outweighs risk.

    Explicit NO-SWAP cases (must reject):
      - far almost at dock (far_eta <= FAR_ARRIVAL_GUARD)
      - gap below margin (normals; urgents use softer margin unless SLA rescue)
      - near already holds a more urgent task
      - deep queue + only marginal gap
      - urgent but far already meets SLA (no rescue needed)
      - urgent but near also misses SLA (swap does not help)
    """
    gap = ctx.far_eta - ctx.near_eta
    margin = max(1.0, float(ctx.margin))
    urgent = is_urgent_task(ctx.task_info)
    rt = ctx.task_info.get("remaining_time")

    # 1) 临门保护：远车快到站，不换
    if ctx.far_eta <= FAR_ARRIVAL_GUARD:
        return False

    # 2) 近车正持紧急件，远单是普通 → 不打断近车
    if ctx.near_task_info and is_urgent_task(ctx.near_task_info) and not urgent:
        return False

    # 3) 紧急 SLA 逻辑
    if urgent and rt is not None:
        try:
            rt_f = float(rt)
        except (TypeError, ValueError):
            rt_f = None
        if rt_f and rt_f > 0:
            # 远车已经能赶上 → 不必换
            if ctx.far_eta <= rt_f * 0.95:
                return False
            # 近车也赶不上 → 换手无 SLA 收益
            if ctx.near_eta > rt_f:
                return False
            # 近车能救、远车会迟到
            if ctx.near_eta <= rt_f and ctx.far_eta > rt_f * 0.92 and gap >= margin * 0.35:
                return True
            if ctx.far_eta > rt_f and gap >= margin * 0.4 and ctx.near_eta < rt_f * 0.85:
                return True
            if gap < margin * 0.35:
                return False

    # 4) 普通：gap 不够
    if not urgent and gap < margin:
        return False

    # 5) 深队列 + 边际 gap → 不换（换队首收益有限、震荡大）
    if ctx.queue_depth >= 4 and gap < margin + 2.0:
        return False

    # 6) 深队列时要求更大 gap
    if ctx.queue_depth >= 6 and gap < margin + 3.5:
        return False

    return gap >= margin + 0.5


def _man(a, b) -> float:
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def sample_no_swap_far_at_door(rng: random.Random) -> DockWanderSpec:
    """Far eta <= 2: even if near is closer, do NOT swap."""
    pickup = (10, rng.randint(3, 7))
    far_eta = float(rng.choice([0.0, 1.0, 2.0]))
    near_eta = float(rng.uniform(0.0, max(0.0, far_eta - 0.5)))  # near even closer
    return DockWanderSpec(
        far_xy=(int(pickup[0] - far_eta), pickup[1]),
        near_xy=(int(pickup[0] - near_eta), pickup[1]),
        pickup_xy=pickup,
        margin=float(rng.choice([3.0, 4.0, 5.0])),
        urgent=rng.random() < 0.25,
        remaining_time=float(rng.randint(80, 200)),
        queue_depth=rng.randint(0, 5),
        scenario="no_swap_far_at_door",
        far_eta=far_eta,
        near_eta=near_eta,
    )


def sample_no_swap_gap_small(rng: random.Random) -> DockWanderSpec:
    """gap < margin: classic reject."""
    margin = float(rng.choice([4.0, 5.0, 6.0]))
    near_eta = float(rng.uniform(2.0, 10.0))
    gap = float(rng.uniform(0.0, margin - 0.5))
    far_eta = near_eta + gap
    # keep far away from door guard
    if far_eta <= FAR_ARRIVAL_GUARD:
        far_eta = FAR_ARRIVAL_GUARD + 1.0 + gap
        near_eta = far_eta - gap
    return DockWanderSpec(
        far_xy=(0, 5),
        near_xy=(8, 5),
        pickup_xy=(10, 5),
        margin=margin,
        urgent=False,
        queue_depth=rng.randint(0, 3),
        scenario="no_swap_gap_small",
        far_eta=far_eta,
        near_eta=near_eta,
    )


def sample_no_swap_near_holds_urgent(rng: random.Random) -> DockWanderSpec:
    """Near holds urgent; far holds Normal → never preempt near."""
    margin = 4.0
    near_eta = float(rng.uniform(1.0, 6.0))
    far_eta = near_eta + float(rng.uniform(margin + 1.0, margin + 12.0))
    return DockWanderSpec(
        far_xy=(0, 5),
        near_xy=(9, 5),
        pickup_xy=(10, 5),
        margin=margin,
        urgent=False,
        near_holds_urgent=True,
        queue_depth=rng.randint(0, 4),
        scenario="no_swap_near_holds_urgent",
        far_eta=far_eta,
        near_eta=near_eta,
    )


def sample_no_swap_deep_queue_marginal(rng: random.Random) -> DockWanderSpec:
    """Deep FIFO + only marginal ETA gain."""
    margin = float(rng.choice([4.0, 5.0]))
    near_eta = float(rng.uniform(2.0, 8.0))
    gap = float(rng.uniform(margin, margin + 1.8))  # passes raw margin, fails deep-queue rule
    far_eta = near_eta + gap
    if far_eta <= FAR_ARRIVAL_GUARD:
        far_eta = FAR_ARRIVAL_GUARD + gap + 1.0
        near_eta = far_eta - gap
    return DockWanderSpec(
        far_xy=(0, 5),
        near_xy=(8, 5),
        pickup_xy=(10, 5),
        margin=margin,
        urgent=False,
        queue_depth=rng.randint(4, 8),
        scenario="no_swap_deep_queue_marginal",
        far_eta=far_eta,
        near_eta=near_eta,
    )


def sample_no_swap_urgent_already_ok(rng: random.Random) -> DockWanderSpec:
    """Urgent but far already meets SLA — no rescue needed."""
    near_eta = float(rng.uniform(2.0, 8.0))
    far_eta = near_eta + float(rng.uniform(5.0, 14.0))
    rt = far_eta / 0.85  # far finishes with slack
    return DockWanderSpec(
        far_xy=(0, 5),
        near_xy=(8, 5),
        pickup_xy=(10, 5),
        margin=4.0,
        urgent=True,
        remaining_time=float(rt),
        queue_depth=rng.randint(0, 3),
        scenario="no_swap_urgent_already_ok",
        far_eta=far_eta,
        near_eta=near_eta,
    )


def sample_no_swap_both_miss_sla(rng: random.Random) -> DockWanderSpec:
    """Urgent: near also misses SLA → swap does not help deadline."""
    rt = float(rng.randint(30, 70))
    near_eta = rt + float(rng.uniform(2.0, 15.0))
    far_eta = near_eta + float(rng.uniform(4.0, 12.0))
    return DockWanderSpec(
        far_xy=(0, 5),
        near_xy=(8, 5),
        pickup_xy=(10, 5),
        margin=4.0,
        urgent=True,
        remaining_time=rt,
        queue_depth=rng.randint(0, 4),
        scenario="no_swap_both_miss_sla",
        far_eta=far_eta,
        near_eta=near_eta,
    )


def sample_no_swap_urgent_tiny_gap(rng: random.Random) -> DockWanderSpec:
    """Urgent late-ish but gap too tiny to justify thrash."""
    rt = float(rng.randint(80, 150))
    near_eta = float(rng.uniform(10.0, 40.0))
    gap = float(rng.uniform(0.2, 1.2))  # << margin*0.35
    far_eta = near_eta + gap
    # make far slightly over SLA so SLA rules don't force swap on huge gap
    if far_eta <= rt:
        far_eta = rt + 1.0
        near_eta = far_eta - gap
    return DockWanderSpec(
        far_xy=(0, 5),
        near_xy=(8, 5),
        pickup_xy=(10, 5),
        margin=4.0,
        urgent=True,
        remaining_time=rt,
        queue_depth=rng.randint(0, 3),
        scenario="no_swap_urgent_tiny_gap",
        far_eta=far_eta,
        near_eta=near_eta,
    )


def sample_yes_swap_clear_dock_wander(rng: random.Random) -> DockWanderSpec:
    """Clear positive: far still far, near idle, large gap, shallow queue."""
    margin = float(rng.choice([3.0, 4.0, 5.0]))
    near_eta = float(rng.uniform(1.0, 5.0))
    gap = float(rng.uniform(margin + 2.0, margin + 14.0))
    far_eta = max(near_eta + gap, FAR_ARRIVAL_GUARD + gap + 1.0)
    return DockWanderSpec(
        far_xy=(0, 5),
        near_xy=(9, 5),
        pickup_xy=(10, 5),
        margin=margin,
        urgent=False,
        remaining_time=0.0,
        queue_depth=rng.randint(0, 2),
        scenario="yes_swap_dock_wander",
        far_eta=far_eta,
        near_eta=near_eta,
    )


def sample_yes_swap_urgent_rescue(rng: random.Random) -> DockWanderSpec:
    """Positive: far misses SLA, near meets it."""
    rt = float(rng.randint(60, 140))
    near_eta = float(rng.uniform(5.0, rt * 0.7))
    far_eta = float(rng.uniform(rt * 1.05, rt * 1.6))
    if far_eta - near_eta < 2.0:
        far_eta = near_eta + 4.0
    return DockWanderSpec(
        far_xy=(0, 5),
        near_xy=(9, 5),
        pickup_xy=(10, 5),
        margin=4.0,
        urgent=True,
        remaining_time=rt,
        queue_depth=rng.randint(0, 2),
        scenario="yes_swap_urgent_rescue",
        far_eta=far_eta,
        near_eta=near_eta,
    )


def sample_spec_random(rng: random.Random) -> DockWanderSpec:
    urgent = rng.random() < 0.35
    far = (rng.randint(0, 4), rng.randint(2, 8))
    near = (rng.randint(7, 10), rng.randint(2, 8))
    pickup = (10, rng.randint(2, 8))
    rt = float(rng.randint(60, 240)) if urgent else 0.0
    return DockWanderSpec(
        far_xy=far,
        near_xy=near,
        pickup_xy=pickup,
        margin=float(rng.choice([3.0, 4.0, 5.0, 6.0])),
        urgent=urgent,
        remaining_time=rt,
        near_holds_urgent=rng.random() < 0.06,
        queue_depth=rng.randint(0, 6),
        scenario="random",
    )


# Mixture: emphasize no-swap hard negatives (~55%), positives (~30%), random (~15%)
_SCENARIO_MIX: List[Tuple[float, str]] = [
    (0.12, "no_swap_far_at_door"),
    (0.12, "no_swap_gap_small"),
    (0.10, "no_swap_near_holds_urgent"),
    (0.10, "no_swap_deep_queue_marginal"),
    (0.08, "no_swap_urgent_already_ok"),
    (0.08, "no_swap_both_miss_sla"),
    (0.05, "no_swap_urgent_tiny_gap"),
    (0.18, "yes_swap_dock_wander"),
    (0.12, "yes_swap_urgent_rescue"),
    (0.05, "random"),
]

_SAMPLERS = {
    "no_swap_far_at_door": sample_no_swap_far_at_door,
    "no_swap_gap_small": sample_no_swap_gap_small,
    "no_swap_near_holds_urgent": sample_no_swap_near_holds_urgent,
    "no_swap_deep_queue_marginal": sample_no_swap_deep_queue_marginal,
    "no_swap_urgent_already_ok": sample_no_swap_urgent_already_ok,
    "no_swap_both_miss_sla": sample_no_swap_both_miss_sla,
    "no_swap_urgent_tiny_gap": sample_no_swap_urgent_tiny_gap,
    "yes_swap_dock_wander": sample_yes_swap_clear_dock_wander,
    "yes_swap_urgent_rescue": sample_yes_swap_urgent_rescue,
    "random": sample_spec_random,
}


def sample_spec(rng: random.Random) -> DockWanderSpec:
    r = rng.random()
    acc = 0.0
    name = "random"
    for p, n in _SCENARIO_MIX:
        acc += p
        if r <= acc:
            name = n
            break
    return _SAMPLERS[name](rng)


def spec_to_context(spec: DockWanderSpec) -> Tuple[SwapContext, int, str]:
    far_eta = (
        float(spec.far_eta)
        if spec.far_eta is not None
        else _man(spec.far_xy, spec.pickup_xy)
    )
    near_eta = (
        float(spec.near_eta)
        if spec.near_eta is not None
        else _man(spec.near_xy, spec.pickup_xy)
    )
    task = {
        "task_id": "T1",
        "priority": "Urgent" if spec.urgent else "Normal",
        "remaining_time": spec.remaining_time if spec.urgent else None,
        "pickup_name": "S1",
    }
    near_info = (
        {"priority": "Urgent", "remaining_time": max(40.0, spec.remaining_time * 0.8)}
        if spec.near_holds_urgent
        else None
    )
    ctx = SwapContext(
        far_eta=far_eta,
        near_eta=near_eta,
        margin=spec.margin,
        far_dist=far_eta,
        near_dist=near_eta,
        task_info=task,
        queue_depth=spec.queue_depth,
        near_has_task=spec.near_holds_urgent,
        near_task_info=near_info,
    )
    label = 1 if oracle_should_swap(ctx) else 0
    return ctx, label, spec.scenario


def _sample_weight(ctx: SwapContext, y: int, scenario: str) -> float:
    """Up-weight hard no-swap negatives and SLA rescue positives."""
    w = 1.0
    if scenario.startswith("no_swap_"):
        w = 2.5
        if scenario in ("no_swap_far_at_door", "no_swap_near_holds_urgent"):
            w = 3.5
        if scenario in ("no_swap_deep_queue_marginal", "no_swap_both_miss_sla"):
            w = 3.0
    elif scenario.startswith("yes_swap_"):
        w = 2.0 if "urgent" in scenario else 1.5
    if is_urgent_task(ctx.task_info):
        w *= 1.4 if y else 1.2
    # Extra weight when label is no-swap (class balance toward caution)
    if y == 0:
        w *= 1.15
    return float(w)


def build_dataset(
    n: int = 16000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    rng = random.Random(seed)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    ws: List[float] = []
    tags: List[str] = []
    for _ in range(n):
        ctx, y, scenario = spec_to_context(sample_spec(rng))
        if scenario.startswith("no_swap_"):
            y = 0
        elif scenario.startswith("yes_swap_"):
            # Regenerate until oracle agrees (geometry should almost always pass)
            tries = 0
            while y == 0 and tries < 8:
                ctx, y, scenario = spec_to_context(_SAMPLERS[scenario](rng))
                tries += 1
            y = 1
        xs.append(build_swap_features(ctx))
        ys.append(y)
        ws.append(_sample_weight(ctx, y, scenario))
        tags.append(scenario)
    return (
        np.stack(xs),
        np.array(ys, dtype=np.float32),
        np.array(ws, dtype=np.float32),
        tags,
    )


def dataset_stats(y: np.ndarray, tags: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for i, t in enumerate(tags):
        d = out.setdefault(t, {"n": 0, "swap": 0})
        d["n"] += 1
        d["swap"] += int(y[i] > 0.5)
    for t, d in out.items():
        d["swap_rate"] = d["swap"] / max(1, d["n"])
    return out
