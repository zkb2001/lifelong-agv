"""Ablation study for MRSC (RuleNet / StageNet / PGCP / PGFA).

Example:
  python -m greed_coop_mcts.ablation --slots 1-20 --max-sim-time 10000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from greed_coop_mcts.paths import RESULTS
from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import solve

ABLATION_DIR = RESULTS / "ablation"

# Leave-one-out + dual-AI-off. "full" is the reference.
# Baseline = production (CGPN/TERN opt-in). Experimental variants enable them.
_BASE = {
    "use_rule_net": True,
    "use_stage_net": True,
    "use_pgcp": True,
    "use_pgfa": True,
    "use_cgpn": False,  # ASDCR opt-in (ablation / research)
    "use_tern": False,
}
VARIANTS: Dict[str, Dict[str, bool]] = {
    "full": dict(_BASE),
    "no_cgpn": {**_BASE, "use_cgpn": False},
    "no_rulenet": {**_BASE, "use_rule_net": False},
    "no_stagenet": {**_BASE, "use_stage_net": False},
    "no_pgcp": {**_BASE, "use_pgcp": False},
    "no_pgfa": {**_BASE, "use_pgfa": False},
    "no_ai": {
        **_BASE,
        "use_rule_net": False,
        "use_stage_net": False,
        "use_cgpn": False,
    },
}


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


def _run_one(
    slot: int,
    variant: str,
    flags: Dict[str, bool],
    *,
    wall_timeout: float,
    max_sim_time: int,
    device: str,
) -> Dict[str, Any]:
    meta = load_slot_meta(
        int(slot),
        max_sim_time=int(max_sim_time),
        wall_timeout=float(wall_timeout),
        n_tasks=100,
    )
    if not meta:
        return {"slot": slot, "variant": variant, "error": "meta_missing"}
    meta = dict(meta)
    meta["id"] = f"SH{int(slot):02d}_abl_{variant}"
    meta["n_tasks"] = 100
    print(f"\n===== ablation {variant} SH{int(slot):02d} =====", flush=True)
    t0 = time.perf_counter()
    try:
        rep = solve(
            int(slot),
            meta=meta,
            wall_timeout=float(wall_timeout),
            max_sim_time=int(max_sim_time),
            policy="rrsc",
            device=device,
            result_tag=variant,
            **flags,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "slot": slot,
            "variant": variant,
            "error": f"{exc.__class__.__name__}:{exc}",
            "validate_ok": False,
            "wall_seconds": round(time.perf_counter() - t0, 2),
            **flags,
        }
    rep = dict(rep)
    rep["variant"] = variant
    for k, v in flags.items():
        rep[k] = v
    return rep


def _write_summary(rows: List[Dict[str, Any]], slots: List[int]) -> Path:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    by_var: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_var.setdefault(str(r.get("variant") or "?"), []).append(r)

    lines: List[str] = [
        "# MRSC Ablation (SH slots)",
        "",
        "Protocol: leave-one-component-out under `policy=rrsc`.",
        "",
        "| Variant | Stage | CGPN | TERN | PGCP | PGFA | full100 | avg done | avg sim(full) | avg wall | VALID |",
        "|---------|-------|------|------|------|------|---------|----------|---------------|----------|-------|",
    ]
    for vname in VARIANTS:
        rs = by_var.get(vname, [])
        if not rs:
            continue
        n = len(rs)
        full100 = sum(
            1
            for r in rs
            if int(r.get("tasks_completed") or 0) >= int(r.get("tasks_total") or 100)
        )
        avg_done = sum(float(r.get("tasks_completed") or 0) for r in rs) / max(1, n)
        full_sims = [
            float(r["sim_time"])
            for r in rs
            if int(r.get("tasks_completed") or 0)
            >= int(r.get("tasks_total") or 100)
        ]
        avg_sim = sum(full_sims) / len(full_sims) if full_sims else float("nan")
        avg_wall = sum(float(r.get("wall_seconds") or 0) for r in rs) / max(1, n)
        valid_n = sum(1 for r in rs if r.get("validate_ok"))
        f = VARIANTS[vname]
        lines.append(
            f"| `{vname}` "
            f"| {'Y' if f.get('use_stage_net') else '—'}"
            f"| {'Y' if f.get('use_cgpn') else '—'}"
            f"| {'Y' if f.get('use_tern') else '—'}"
            f"| {'Y' if f.get('use_pgcp') else '—'}"
            f"| {'Y' if f.get('use_pgfa') else '—'}"
            f"| {full100}/{n}"
            f"| {avg_done:.1f}"
            f"| {avg_sim:.0f}"
            f"| {avg_wall:.1f}s"
            f"| {valid_n}/{n} |"
        )

    lines.extend(["", "## Per-slot: done / sim / wall(s)", ""])
    header = "| Slot | " + " | ".join(VARIANTS.keys()) + " |"
    sep = "|------|" + "|".join(["------"] * len(VARIANTS)) + "|"
    lines.append(header)
    lines.append(sep)
    for slot in slots:
        cells = []
        for vname in VARIANTS:
            hit = next(
                (
                    r
                    for r in by_var.get(vname, [])
                    if int(r.get("slot") or -1) == slot
                ),
                None,
            )
            if hit is None:
                cells.append("—")
            elif hit.get("error"):
                cells.append("ERR")
            else:
                d = int(hit.get("tasks_completed") or 0)
                s = int(hit.get("sim_time") or 0)
                w = float(hit.get("wall_seconds") or 0)
                mark = "" if d >= 100 else "*"
                cells.append(f"{d}/{s}/{w:.2f}{mark}")
        lines.append(f"| SH{slot:02d} | " + " | ".join(cells) + " |")

    # Long + wide CSV for paper / spreadsheet use.
    csv_lines = [
        "variant,slot,done,total,sim_time,wall_seconds,validate_ok,completion_ratio"
    ]
    for vname in VARIANTS:
        for slot in slots:
            hit = next(
                (
                    r
                    for r in by_var.get(vname, [])
                    if int(r.get("slot") or -1) == slot
                ),
                None,
            )
            if hit is None:
                csv_lines.append(f"{vname},{slot},,,,,,,")
            else:
                csv_lines.append(
                    f"{vname},{slot},{hit.get('tasks_completed','')},"
                    f"{hit.get('tasks_total','')},{hit.get('sim_time','')},"
                    f"{hit.get('wall_seconds','')},{hit.get('validate_ok','')},"
                    f"{hit.get('completion_ratio','')}"
                )
    (ABLATION_DIR / "ablation_wall_sim.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )
    for metric, key in (
        ("sim", "sim_time"),
        ("wall", "wall_seconds"),
        ("done", "tasks_completed"),
    ):
        wide = ["Slot," + ",".join(VARIANTS.keys())]
        for slot in slots:
            cells = []
            for vname in VARIANTS:
                hit = next(
                    (
                        r
                        for r in by_var.get(vname, [])
                        if int(r.get("slot") or -1) == slot
                    ),
                    None,
                )
                cells.append("" if hit is None else str(hit.get(key, "")))
            wide.append(f"SH{slot:02d}," + ",".join(cells))
        (ABLATION_DIR / f"ablation_{metric}.csv").write_text(
            "\n".join(wide) + "\n", encoding="utf-8"
        )

    lines.extend(
        [
            "",
            "Notes:",
            "- `no_rulenet`: goal-flow teacher only (no GlobalRuleNet forward).",
            "- `no_stagenet`: StageSelector teacher heuristics (no StageSelectorNet).",
            "- `no_pgcp`: contested-destination gate + pad-egress off.",
            "- `no_pgfa`: focal altruism off.",
            "- `no_ai`: both nets off; PGCP/PGFA still on.",
            "- `*` = incomplete (<100 tasks).",
            "- CSV: `ablation_wall_sim.csv` (long), `ablation_sim.csv` / "
            "`ablation_wall.csv` / `ablation_done.csv` (wide).",
            "",
        ]
    )
    out = ABLATION_DIR / "ABLATION.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    (ABLATION_DIR / "ablation_raw.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="MRSC component ablation")
    ap.add_argument("--slots", type=str, default="1,3,8,12,17")
    ap.add_argument(
        "--variants",
        type=str,
        default="full,no_rulenet,no_stagenet,no_pgcp,no_pgfa,no_ai",
        help="comma-separated keys from VARIANTS",
    )
    ap.add_argument("--wall-timeout", type=float, default=600.0)
    ap.add_argument("--max-sim-time", type=int, default=10000)
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"

    slots = _parse_slots(args.slots)
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variant_names:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {list(VARIANTS)}")

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for variant in variant_names:
        flags = VARIANTS[variant]
        vdir = ABLATION_DIR / variant
        vdir.mkdir(parents=True, exist_ok=True)
        for slot in slots:
            rep = _run_one(
                slot,
                variant,
                flags,
                wall_timeout=float(args.wall_timeout),
                max_sim_time=int(args.max_sim_time),
                device=device,
            )
            rows.append(rep)
            (vdir / f"SH{int(slot):02d}.json").write_text(
                json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            done = rep.get("tasks_completed", "?")
            sim = rep.get("sim_time", "?")
            wall = rep.get("wall_seconds", "?")
            print(
                f"[ablation] {variant} SH{slot:02d} done={done} sim={sim} wall={wall}",
                flush=True,
            )
            _write_summary(rows, slots)

    md = _write_summary(rows, slots)
    print(f"\n[ablation] wrote {md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
