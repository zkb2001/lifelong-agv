"""Oracle sweep force_fleet_cap; write best-K JSON (v2: idle_pool≥0.2 maps)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

os.environ.setdefault("PYTHONHASHSEED", "0")

from greed_coop_mcts.paths import RESULTS
from greed_coop_mcts.pgcp import near_pad_deg2_count, throat_band_fleet_cap
from greed_coop_mcts.rule_obs import GRID, seal_ghost_border
from greed_coop_mcts.scenario import load_slot_meta

Cell = Tuple[int, int]


def _geom(slot: int) -> Optional[dict]:
    meta = load_slot_meta(slot)
    if not meta:
        return None
    jpath = Path(meta.get("map_json") or "")
    blocked: Set[Cell] = set()
    drops: Set[Cell] = set()
    if jpath.is_file():
        data = json.loads(jpath.read_text(encoding="utf-8"))
        for p in data.get("extra_obstacles") or []:
            if isinstance(p, dict):
                blocked.add((int(p["x"]), int(p["y"])))
            else:
                blocked.add((int(p[0]), int(p[1])))
        for p in data.get("dropoffs") or []:
            c = (int(p["x"]), int(p["y"]))
            drops.add(c)
            blocked.add(c)
        for p in data.get("pickups") or []:
            c = (int(p["x"]), int(p["y"]))
            blocked.add(c)
    blocked = seal_ghost_border(blocked)
    free = sum(
        1 for x in range(1, GRID) for y in range(1, GRID) if (x, y) not in blocked
    )
    n = int(meta.get("n_agvs") or 8)
    near = near_pad_deg2_count(blocked, drops)
    k_tbc = throat_band_fleet_cap(n, near, base=6)
    idle = float(n - k_tbc) / float(max(1, n))
    return {
        "n": n,
        "free": free,
        "near": near,
        "k_tbc": int(k_tbc),
        "idle_pool": idle,
    }


def slots_idle_pool(lo: float = 0.2) -> List[int]:
    out: List[int] = []
    for s in range(1, 41):
        g = _geom(s)
        if g and float(g["idle_pool"]) >= float(lo):
            out.append(s)
    return out


def k_range_for(slot: int, *, max_raise: int = 4) -> List[int]:
    g = _geom(slot)
    if g is None:
        return []
    k0 = int(g["k_tbc"])
    n = int(g["n"])
    hi = min(k0 + int(max_raise), n, 16)
    return list(range(k0, hi + 1))


def run_one(slot: int, k: int, wall: float) -> dict:
    code = (
        "import json,os;"
        "os.environ['PYTHONHASHSEED']='0';"
        "os.environ['OMP_NUM_THREADS']='1';"
        "from greed_coop_mcts.solve import solve;"
        f"r=solve({slot},policy='rrsc',use_facn=False,force_fleet_cap={k},"
        f"wall_timeout={wall},device='cpu');"
        "print(json.dumps({"
        "'slot':r.get('slot'),'sim':r.get('sim_time'),"
        "'done':r.get('tasks_completed'),'total':r.get('tasks_total'),"
        "'valid':r.get('validate_ok'),'k':"
        f"{k}"
        "}))"
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        return {
            "slot": slot,
            "k": k,
            "error": (proc.stderr or "")[-500:],
            "valid": False,
        }
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    return json.loads(lines[-1])


def _load_prior(path: Path) -> Tuple[List[dict], Dict[str, int]]:
    if not path.is_file():
        return [], {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [], {}
    rows = list(raw.get("rows") or [])
    best = {str(k): int(v) for k, v in (raw.get("best") or {}).items()}
    return rows, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--slots",
        type=str,
        default="",
        help="comma slots; empty + --auto-idle → all idle_pool≥lo",
    )
    ap.add_argument("--ks", type=str, default="", help="fixed K list; empty → per-slot range")
    ap.add_argument("--auto-idle", type=float, default=0.2, help="idle_pool floor; 0 disables")
    ap.add_argument("--max-raise", type=int, default=4)
    ap.add_argument("--wall-timeout", type=float, default=600.0)
    ap.add_argument(
        "--out",
        type=str,
        default=str(RESULTS / "facn_oracle_sweep_v2.json"),
    )
    ap.add_argument(
        "--merge",
        type=str,
        default="",
        help="optional prior sweep JSON to merge/resume",
    )
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers")
    args = ap.parse_args()

    if args.slots.strip():
        slots = [int(x) for x in args.slots.split(",") if x.strip()]
    elif float(args.auto_idle) > 0:
        slots = slots_idle_pool(float(args.auto_idle))
    else:
        slots = [20]

    fixed_ks = (
        [int(x) for x in args.ks.split(",") if x.strip()] if args.ks.strip() else None
    )

    out_path = Path(args.out)
    merge_path = Path(args.merge) if args.merge else out_path
    rows, _best0 = _load_prior(merge_path)
    # also merge classic v1 if present and not already merged
    v1 = RESULTS / "facn_oracle_sweep.json"
    if v1.is_file() and merge_path.resolve() != v1.resolve():
        r1, _ = _load_prior(v1)
        rows.extend(r1)

    done_keys = {
        (int(r["slot"]), int(r["k"]))
        for r in rows
        if r.get("slot") is not None
        and r.get("k") is not None
        and "error" not in r
        and r.get("sim") is not None
    }

    geoms: Dict[int, dict] = {}
    for s in slots:
        g = _geom(s)
        if g:
            geoms[s] = g
            print(
                f"[sweep] plan SH{s:02d} n={g['n']} k_tbc={g['k_tbc']} "
                f"idle={g['idle_pool']:.2f} near={g['near']}",
                flush=True,
            )

    pending: List[Tuple[int, int]] = []
    for s in slots:
        ks = (
            fixed_ks
            if fixed_ks is not None
            else k_range_for(s, max_raise=int(args.max_raise))
        )
        for k in ks:
            if (s, k) in done_keys:
                print(f"[sweep] SH{s:02d} K={k} (cached)", flush=True)
            else:
                pending.append((s, k))

    def _recompute() -> Tuple[Dict[str, int], Dict[str, int]]:
        best_local: Dict[str, int] = {}
        delta_local: Dict[str, int] = {}
        by_slot: Dict[int, List[dict]] = {}
        allowed: Dict[int, Set[int]] = {}
        for s in slots:
            ks = (
                fixed_ks
                if fixed_ks is not None
                else k_range_for(s, max_raise=int(args.max_raise))
            )
            allowed[s] = set(ks)
        for r in rows:
            if not (
                r.get("valid")
                and r.get("done") == r.get("total")
                and r.get("sim") is not None
            ):
                continue
            s = int(r["slot"])
            k = int(r["k"])
            if s in allowed and k not in allowed[s]:
                continue
            by_slot.setdefault(s, []).append(r)
        for s, cand in by_slot.items():
            win = min(cand, key=lambda x: (int(x["sim"]), -int(x["k"])))
            g = geoms.get(s) or _geom(s) or {}
            k_tbc = int(g.get("k_tbc") or 6)
            best_local[str(s)] = int(win["k"])
            delta_local[str(s)] = int(win["k"]) - k_tbc
            if g:
                geoms[s] = g
        return best_local, delta_local

    def _save(best: Dict[str, int], delta_star: Dict[str, int]) -> None:
        payload = {
            "best": best,
            "delta_star": delta_star,
            "geoms": {str(k): v for k, v in geoms.items()},
            "rows": rows,
            "slots_planned": slots,
            "auto_idle": float(args.auto_idle),
            "pending_left": max(0, len(pending) - n_done),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n_done = 0
    wall = float(args.wall_timeout)
    jobs = max(1, int(args.jobs))
    print(f"[sweep] pending={len(pending)} jobs={jobs}", flush=True)

    if jobs <= 1:
        for s, k in pending:
            print(f"[sweep] SH{s:02d} K={k}", flush=True)
            r = run_one(s, k, wall)
            rows.append(r)
            n_done += 1
            print(
                f"  -> sim={r.get('sim')} done={r.get('done')}/{r.get('total')} "
                f"valid={r.get('valid')}",
                flush=True,
            )
            best, delta_star = _recompute()
            _save(best, delta_star)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(run_one, s, k, wall): (s, k) for s, k in pending}
            for fut in as_completed(futs):
                s, k = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r = {"slot": s, "k": k, "valid": False, "error": str(exc)[-500:]}
                rows.append(r)
                n_done += 1
                print(
                    f"[sweep] SH{s:02d} K={k} -> sim={r.get('sim')} "
                    f"done={r.get('done')}/{r.get('total')} valid={r.get('valid')} "
                    f"({n_done}/{len(pending)})",
                    flush=True,
                )
                best, delta_star = _recompute()
                _save(best, delta_star)

    best, delta_star = _recompute()
    for s in slots:
        if str(s) in best:
            print(
                f"[sweep] SH{s:02d} BEST K={best[str(s)]} Δ*={delta_star[str(s)]}",
                flush=True,
            )
    n_done = len(pending)
    _save(best, delta_star)
    print(f"[sweep] wrote {out_path} best={best}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
