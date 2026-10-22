"""Batch SH01–20 standalone mats_local MAPD (no Hier).

Metrics: completion, VALID, sim, wall.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta
from ml_research.benchmarks.mats_local.solve_standalone import OUT, solve_standalone

OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT / "SUMMARY.md"


def _run_slot(
    slot: int,
    *,
    wall_timeout: float,
    max_sim_time: int,
    sims: int,
    k: int,
    device: str,
    greed_w: float = 0.55,
    coop_w: float = 0.35,
    yield_w: float = 0.10,
) -> Dict[str, Any]:
    meta = load_custom_meta(
        int(slot),
        max_sim_time=int(max_sim_time),
        wall_timeout=float(wall_timeout),
        n_tasks=100,
    )
    if not meta:
        return {"slot": slot, "error": "meta_missing", "validate_ok": False}
    meta = dict(meta)
    meta["id"] = f"SH{int(slot):02d}_mats_standalone"
    meta["n_tasks"] = 100
    print(f"\n===== SH{int(slot):02d} mats_greed_coop =====", flush=True)
    t0 = time.perf_counter()
    try:
        rep = solve_standalone(
            int(slot),
            meta=meta,
            wall_timeout=float(wall_timeout),
            max_sim_time=int(max_sim_time),
            num_simulations=int(sims),
            k_proximal=int(k),
            greed_w=float(greed_w),
            coop_w=float(coop_w),
            yield_w=float(yield_w),
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "slot": slot,
            "error": f"{exc.__class__.__name__}:{exc}",
            "validate_ok": False,
            "wall_seconds": round(time.perf_counter() - t0, 2),
        }
    return rep


def _parse_slots(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=str, default="1")
    ap.add_argument("--wall-timeout", type=float, default=1800.0)
    ap.add_argument("--max-sim-time", type=int, default=200000)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--greed", type=float, default=0.55)
    ap.add_argument("--coop", type=float, default=0.35)
    ap.add_argument("--yield-w", type=float, default=0.10)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()
    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    rows: List[Dict[str, Any]] = []
    for slot in _parse_slots(args.slots):
        rep = _run_slot(
            slot,
            wall_timeout=float(args.wall_timeout),
            max_sim_time=int(args.max_sim_time),
            sims=int(args.sims),
            k=int(args.k),
            device=device,
            greed_w=float(args.greed),
            coop_w=float(args.coop),
            yield_w=float(args.yield_w),
        )
        rows.append(rep)
        (OUT / "summary.json").write_text(
            json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[batch] SH{slot:02d} done={rep.get('tasks_completed')}/"
            f"{rep.get('tasks_total')} valid={rep.get('validate_ok')} "
            f"sim={rep.get('sim_time')} wall={rep.get('wall_seconds')}s",
            flush=True,
        )

    lines = [
        "# mats_local greed–coop neural MCTS MAPD",
        "",
        "| slot | done | VALID | sim | wall_s | mcts |",
        "|------|------|-------|-----|--------|------|",
    ]
    for r in rows:
        lines.append(
            f"| SH{int(r.get('slot') or 0):02d} | "
            f"{r.get('tasks_completed')}/{r.get('tasks_total')} | "
            f"{'Y' if r.get('validate_ok') else 'N'} | {r.get('sim_time')} | "
            f"{r.get('wall_seconds')} | {r.get('n_mcts_calls')} |"
        )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[batch] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
