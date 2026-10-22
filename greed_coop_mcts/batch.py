"""Batch SH01–20 greed–coop neural MCTS MAPD (independent package)."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

# Deterministic CPU BLAS/OMP + hash order (set iteration / dict builds).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

from greed_coop_mcts.paths import RESULTS
from greed_coop_mcts.scenario import load_slot_meta
from greed_coop_mcts.solve import OUT, solve

OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT / "SUMMARY.md"

try:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:  # noqa: BLE001
    pass


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
    policy: str = "rrsc",
    use_cgpn: bool = False,
    use_facn: bool = True,
    result_tag: str = "gcm",
) -> Dict[str, Any]:
    meta = load_slot_meta(
        int(slot),
        max_sim_time=int(max_sim_time),
        wall_timeout=float(wall_timeout),
        n_tasks=100,
    )
    if not meta:
        return {"slot": slot, "error": "meta_missing", "validate_ok": False}
    meta = dict(meta)
    tag = result_tag or "gcm"
    meta["id"] = f"SH{int(slot):02d}_{tag}"
    meta["n_tasks"] = 100
    print(
        f"\n===== SH{int(slot):02d} greed_coop policy={policy} "
        f"tag={tag} facn={bool(use_facn)} =====",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        rep = solve(
            int(slot),
            meta=meta,
            wall_timeout=float(wall_timeout),
            max_sim_time=int(max_sim_time),
            num_simulations=int(sims),
            k_proximal=int(k),
            greed_w=float(greed_w),
            coop_w=float(coop_w),
            yield_w=float(yield_w),
            policy=str(policy),
            use_cgpn=bool(use_cgpn),
            use_facn=bool(use_facn),
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[batch] SH{int(slot):02d} ERROR {exc.__class__.__name__}:{exc}", flush=True)
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
    ap.add_argument(
        "--policy",
        type=str,
        default="rrsc",
        choices=["auto", "rrsc", "glocal", "rules", "xfmr", "mcts"],
    )
    ap.add_argument(
        "--cgpn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ASDCR residual (opt-in; --cgpn to enable)",
    )
    ap.add_argument(
        "--facn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FACN fleet concurrency net (default on)",
    )
    ap.add_argument(
        "--result-tag",
        type=str,
        default="",
        help="override result id tag (default: cgpn|facn|nofacn|gcm)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu recommended for deterministic SH01-20",
    )
    ap.add_argument(
        "--summary-name",
        type=str,
        default="summary.json",
        help="batch summary json filename under results/",
    )
    args = ap.parse_args()
    device = "cpu" if args.device == "auto" else str(args.device)
    use_cgpn = bool(args.cgpn)
    use_facn = bool(args.facn)
    if str(args.result_tag or "").strip():
        result_tag = str(args.result_tag).strip()
    elif use_cgpn:
        result_tag = "cgpn"
    elif use_facn:
        result_tag = "facn"
    else:
        result_tag = "nofacn"
    summary_path = OUT / str(args.summary_name or "summary.json")

    # Isolate each slot in a fresh process (avoids BLAS/torch cross-map drift).
    import subprocess
    import sys

    rows: List[Dict[str, Any]] = []
    for slot in _parse_slots(args.slots):
        print(
            f"[batch] starting SH{int(slot):02d} facn={bool(use_facn)} "
            f"tag={result_tag} …",
            flush=True,
        )
        cmd = [
            sys.executable,
            "-c",
            (
                "import json,os,sys;"
                "os.environ['OMP_NUM_THREADS']='1';"
                "os.environ['MKL_NUM_THREADS']='1';"
                "os.environ['OPENBLAS_NUM_THREADS']='1';"
                "os.environ['PYTHONHASHSEED']='0';"
                "os.environ['PYTHONUNBUFFERED']='1';"
                "from greed_coop_mcts.batch import _run_slot;"
                f"print(json.dumps(_run_slot({int(slot)},"
                f"wall_timeout={float(args.wall_timeout)},"
                f"max_sim_time={int(args.max_sim_time)},"
                f"sims={int(args.sims)},k={int(args.k)},"
                f"device={device!r},greed_w={float(args.greed)},"
                f"coop_w={float(args.coop)},yield_w={float(args.yield_w)},"
                f"policy={str(args.policy)!r},use_cgpn={bool(use_cgpn)},"
                f"use_facn={bool(use_facn)},"
                f"result_tag={result_tag!r})), flush=True)"
            ),
        ]
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = "0"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        # Stream child logs live (avoid looking stuck between slots).
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        out_lines: List[str] = []
        assert proc.stdout is not None
        for ln in proc.stdout:
            out_lines.append(ln.rstrip("\n"))
            s = ln.rstrip("\n")
            if (
                s.startswith("=====")
                or s.startswith("[greed_coop]")
                or s.startswith("[batch]")
                or "done=" in s
                or s.startswith("  [")
            ):
                print(s, flush=True)
        rc = proc.wait()
        if rc != 0:
            tail = "\n".join(out_lines[-40:])
            print(tail or f"subprocess failed rc={rc}", flush=True)
            rep = {
                "slot": slot,
                "error": f"subprocess_exit_{rc}",
                "validate_ok": False,
            }
        else:
            # Last JSON line is the report (stdout may include solve logs).
            json_lines = [ln for ln in out_lines if ln.strip().startswith("{")]
            if not json_lines:
                print("subprocess produced no JSON report", flush=True)
                rep = {
                    "slot": slot,
                    "error": "no_json_report",
                    "validate_ok": False,
                }
            else:
                rep = json.loads(json_lines[-1])
        rows.append(rep)
        summary_path.write_text(
            json.dumps(
                {
                    "policy": args.policy,
                    "use_facn": use_facn,
                    "result_tag": result_tag,
                    "rows": rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"[batch] SH{slot:02d} done={rep.get('tasks_completed')}/"
            f"{rep.get('tasks_total')} valid={rep.get('validate_ok')} "
            f"sim={rep.get('sim_time')} wall={rep.get('wall_seconds')}s "
            f"facn={rep.get('use_facn')}",
            flush=True,
        )

    md_name = (
        "SUMMARY_facn.md"
        if use_facn and result_tag == "facn"
        else "SUMMARY_nofacn.md"
        if (not use_facn) and result_tag == "nofacn"
        else "SUMMARY.md"
    )
    md_path = OUT / md_name
    lines = [
        f"# greed_coop_mcts MAPD (policy={args.policy} facn={use_facn} tag={result_tag})",
        "",
        "| slot | done | VALID | sim | wall_s | facn | K | Δ | planner |",
        "|------|------|-------|-----|--------|------|---|---|---------|",
    ]
    for r in rows:
        lines.append(
            f"| SH{int(r.get('slot') or 0):02d} | "
            f"{r.get('tasks_completed')}/{r.get('tasks_total')} | "
            f"{'Y' if r.get('validate_ok') else 'N'} | {r.get('sim_time')} | "
            f"{r.get('wall_seconds')} | {int(bool(r.get('use_facn')))} | "
            f"{r.get('base_fleet_busy')} | {r.get('facn_delta')} | "
            f"{r.get('planner') or r.get('policy')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[batch] wrote {summary_path} and {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
