"""Transformer priority allocator (baseline sources untouched)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ml_research.benchmarks.allocators import _with_soft_propose
from ml_research.benchmarks.common import (
    enhanced_pair_cost,
    is_urgent_task,
    make_assigned,
    maybe_evacuate_on_block,
)
from ml_research.common.paths import RESULTS

from .features import MAX_TASK, build_agv_tokens, build_task_tokens
from .transformer_priority import PriorityTransformer

CKPT = RESULTS / "rl_rh_pp" / "priority_transformer.pt"
CKPT_LIFELONG = RESULTS / "rl_rh_pp" / "priority_transformer_lifelong.pt"
_LOAD: Dict[str, int] = {}
_NET: Optional[PriorityTransformer] = None


def reset_load(names=None) -> None:
    global _LOAD
    _LOAD = {n: 0 for n in (names or [])}


def load_net(ckpt: Path | None = None, force: bool = False) -> PriorityTransformer:
    global _NET
    if _NET is not None and not force:
        return _NET
    path = Path(ckpt or os.environ.get("AGV_RHPP_CKPT") or CKPT)
    if not path.exists() and CKPT_LIFELONG.exists():
        path = CKPT_LIFELONG
    net = PriorityTransformer()
    if path.exists():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        net.load_state_dict(blob["model"])
        print(f"[RL-RH-PP] loaded {path} meta={blob.get('meta')}", flush=True)
    else:
        print(f"[RL-RH-PP] WARN missing ckpt {path}", flush=True)
    net.eval()
    _NET = net
    return net


def make_transformer_allocator(
    net: Optional[PriorityTransformer] = None,
    *,
    deterministic: bool = True,
    collect_logprob: bool = False,
    traj_buf: Optional[List[dict]] = None,
):
    """If ``traj_buf`` is provided, each decision appends {logprob, value} for RL."""
    net = net or load_net()

    def allocator(sim, unassigned_agvs):
        if not unassigned_agvs or not sim.surface_tasks:
            maybe_evacuate_on_block(sim)
            return []

        af, am, _ = build_agv_tokens(sim, unassigned_agvs)
        tf, tm, tids = build_task_tokens(sim, unassigned_agvs)
        if not tids:
            maybe_evacuate_on_block(sim)
            return _with_soft_propose(sim, unassigned_agvs, [])

        agv_f = torch.as_tensor(af).unsqueeze(0)
        agv_m = torch.as_tensor(am).unsqueeze(0)
        task_f = torch.as_tensor(tf).unsqueeze(0)
        task_m = torch.as_tensor(tm).unsqueeze(0)

        if collect_logprob:
            orders, lp = net.decode_order(
                agv_f, agv_m, task_f, task_m, deterministic=False
            )
            val = net.value_of(agv_f, agv_m, task_f, task_m)[0]
            if traj_buf is not None:
                order = list(orders[0])
                teacher = np.full((MAX_TASK,), -1, dtype=np.int64)
                for k, j in enumerate(order[:MAX_TASK]):
                    teacher[k] = int(j)
                traj_buf.append(
                    {
                        "logprob": lp[0],
                        "value": val,
                        "t": int(getattr(sim, "time", 0)),
                        "agv_f": np.asarray(af, dtype=np.float32).copy(),
                        "agv_m": np.asarray(am, dtype=np.bool_).copy(),
                        "task_f": np.asarray(tf, dtype=np.float32).copy(),
                        "task_m": np.asarray(tm, dtype=np.bool_).copy(),
                        "teacher": teacher,
                    }
                )
            sim._rhpp_last_logprob = lp[0]
            sim._rhpp_last_value = val
        else:
            with torch.no_grad():
                orders, _ = net.decode_order(
                    agv_f, agv_m, task_f, task_m, deterministic=deterministic
                )

        ordered = [tids[i] for i in orders[0] if i < len(tids)]
        for tid in tids:
            if tid not in ordered:
                ordered.append(tid)

        fallback = None
        for tid in ordered:
            info = sim.surface_tasks.get(tid)
            if not info:
                continue
            cands = []
            for agv in unassigned_agvs:
                c = enhanced_pair_cost(sim, agv, tid, info, urgency_mode="m0_plus")
                if c < float("inf"):
                    cands.append(((c, int(_LOAD.get(agv.name, 0)), agv.name), c, agv))
            if not cands:
                continue
            cands.sort(key=lambda x: x[0])
            _, c, agv = cands[0]
            far = 18.0
            try:
                from ml_research.benchmarks.common import get_pair_cost_mode

                if get_pair_cost_mode() == "astar":
                    far = float(os.environ.get("AGV_PP_FAR_COST", "80"))
            except Exception:
                pass
            if (
                not is_urgent_task(info)
                and info.get("numbers_before_urgent", -1) < 0
                and c > far
            ):
                if fallback is None or c < fallback[0]:
                    fallback = (c, agv, tid)
                continue
            assigned = make_assigned(sim, agv, tid)
            _LOAD[agv.name] = int(_LOAD.get(agv.name, 0)) + 1
            sim._rhpp_assigned_flag = True
            if not collect_logprob:
                print(f"分配[TF-RH-PP]: AGV {agv.name} → 任务 {tid}, 代价: {c}")
            return assigned

        if fallback is not None:
            c, agv, tid = fallback
            assigned = make_assigned(sim, agv, tid)
            _LOAD[agv.name] = int(_LOAD.get(agv.name, 0)) + 1
            sim._rhpp_assigned_flag = True
            if not collect_logprob:
                print(f"分配[TF-RH-PP-far]: AGV {agv.name} → 任务 {tid}, 代价: {c}")
            return assigned

        maybe_evacuate_on_block(sim)
        return _with_soft_propose(sim, unassigned_agvs, [])

    return allocator


def allocator_transformer_pp(sim, unassigned_agvs):
    return make_transformer_allocator(deterministic=True)(sim, unassigned_agvs)
