"""Fine K-sweep oracle labeling for synth maps (resume + parallel + map batches)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple

os.environ.setdefault("PYTHONHASHSEED", "0")

from greed_coop_mcts.paths import MAP_EXPORTS_SYNTH, RESULTS
from greed_coop_mcts.pgcp import near_pad_deg2_count, throat_band_fleet_cap
from greed_coop_mcts.rule_obs import GRID, seal_ghost_border
from greed_coop_mcts.synth_maps.generate import load_synth_meta

Cell = Tuple[int, int]


def _geom_from_meta(meta: dict) -> dict:
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
        "near": int(near),
        "k_tbc": int(k_tbc),
        "idle_pool": idle,
    }


def run_one_synth(map_idx: int, k: int, wall: float, exports: str) -> dict:
    # Soft stop: solve wall_timeout. Hard kill: subprocess timeout = wall+60.
    hard_timeout = float(wall) + 60.0
    code = (
        "import json,os;"
        "os.environ['PYTHONHASHSEED']='0';"
        "os.environ['OMP_NUM_THREADS']='1';"
        "from greed_coop_mcts.synth_maps.generate import load_synth_meta;"
        "from greed_coop_mcts.solve import solve;"
        f"meta=load_synth_meta({map_idx},exports=__import__('pathlib').Path({exports!r}));"
        f"meta=dict(meta);meta['id']=str(meta.get('id') or f'SYN_{map_idx:05d}')+f'_K{k}';"
        f"r=solve({map_idx},meta=meta,policy='rrsc',use_facn=False,"
        f"force_fleet_cap={k},wall_timeout={wall},device='cpu');"
        "print(json.dumps({"
        "'idx':"
        f"{map_idx}"
        ",'slot':r.get('slot'),'sim':r.get('sim_time'),"
        "'done':r.get('tasks_completed'),'total':r.get('tasks_total'),"
        "'valid':r.get('validate_ok'),'k':"
        f"{k}"
        "}))"
    )
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=hard_timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "idx": map_idx,
            "k": k,
            "valid": False,
            "sim": None,
            "done": 0,
            "total": 0,
            "timed_out": True,
            "error": f"hard_timeout>{hard_timeout:.0f}s",
        }
    if proc.returncode != 0:
        return {
            "idx": map_idx,
            "k": k,
            "valid": False,
            "timed_out": False,
            "error": (proc.stderr or "")[-400:],
        }
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.strip()]
    out = json.loads(lines[-1])
    out.setdefault("timed_out", False)
    return out


def _row_attempted(r: dict) -> bool:
    """True if this (idx,K) should not be retried (success, fail, or timeout)."""
    if r.get("idx") is None or r.get("k") is None:
        return False
    if r.get("sim") is not None:
        return True
    if r.get("timed_out") or r.get("error"):
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=10001)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max-raise", type=int, default=4)
    ap.add_argument("--extend-top-idle-frac", type=float, default=0.2)
    ap.add_argument("--wall-timeout", type=float, default=180.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--exports", type=str, default=str(MAP_EXPORTS_SYNTH))
    ap.add_argument(
        "--out",
        type=str,
        default=str(RESULTS / "facn_oracle_synth_p1.json"),
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="persist oracle JSON every N finished (i,K) jobs within a batch",
    )
    ap.add_argument(
        "--batch-maps",
        type=int,
        default=100,
        help="process this many maps per batch (resume-friendly)",
    )
    ap.add_argument(
        "--inflight",
        type=int,
        default=0,
        help="max in-flight (i,K) futures; 0 → jobs*2",
    )
    args = ap.parse_args()
    exports = Path(args.exports)
    out_path = Path(args.out)
    idxs = list(range(int(args.start), int(args.start) + int(args.n)))
    batch_maps = max(1, int(args.batch_maps))

    prior_rows: List[dict] = []
    if out_path.is_file():
        prior_rows = list(
            json.loads(out_path.read_text(encoding="utf-8")).get("rows") or []
        )
    done = {
        (int(r["idx"]), int(r["k"])) for r in prior_rows if _row_attempted(r)
    }

    print(
        f"[label] loading geoms for {len(idxs)} maps "
        f"(cached_attempts={len(done)}) ...",
        flush=True,
    )
    geoms: Dict[int, dict] = {}
    idle_list = []
    for i in idxs:
        meta = load_synth_meta(i, exports=exports)
        g = _geom_from_meta(meta)
        geoms[i] = g
        idle_list.append((i, float(g["idle_pool"])))
    idle_list.sort(key=lambda x: -x[1])
    extend = {
        i
        for i, _ in idle_list[
            : max(1, int(len(idle_list) * float(args.extend_top_idle_frac)))
        ]
    }

    def _ks_for(i: int) -> List[int]:
        g = geoms[i]
        k0 = int(g["k_tbc"])
        n = int(g["n"])
        hi = min(k0 + int(args.max_raise), n, 16)
        if i in extend:
            hi = min(n, 16)
        return list(range(k0, hi + 1))

    rows = list(prior_rows)
    exports_s = str(exports.resolve())
    wall = float(args.wall_timeout)
    jobs = max(1, int(args.jobs))
    save_every = max(1, int(args.save_every))
    inflight_cap = int(args.inflight) if int(args.inflight) > 0 else max(jobs * 2, jobs)

    def _recompute() -> Tuple[Dict[str, int], Dict[str, int]]:
        best: Dict[str, int] = {}
        delta: Dict[str, int] = {}
        by: Dict[int, List[dict]] = {}
        for r in rows:
            if not (
                r.get("done") == r.get("total")
                and r.get("sim") is not None
                and int(r.get("total") or 0) > 0
            ):
                continue
            by.setdefault(int(r["idx"]), []).append(r)
        for i, cand in by.items():
            win = min(cand, key=lambda x: (int(x["sim"]), -int(x["k"])))
            g = geoms.get(i) or {}
            k_tbc = int(g.get("k_tbc") or 6)
            best[str(i)] = int(win["k"])
            delta[str(i)] = int(win["k"]) - k_tbc
        return best, delta

    def _persist(*, batch_i: int, batch_n: int, pending_left: int) -> None:
        best, delta = _recompute()
        out_path.write_text(
            json.dumps(
                {
                    "best": best,
                    "delta_star": delta,
                    "geoms": {str(k): v for k, v in geoms.items()},
                    "rows": rows,
                    "maps": idxs,
                    "pending_left": pending_left,
                    "batch": {"i": batch_i, "n": batch_n, "size": batch_maps},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _run_jobs(
        pending: List[Tuple[int, int]],
        *,
        batch_tag: str,
        batch_i: int,
        batch_n: int,
    ) -> int:
        """Run pending (idx,K) with sliding-window pool. Returns # finished now."""
        if not pending:
            return 0
        n_done = 0
        total = len(pending)
        print(
            f"[label] {batch_tag} jobs={total} inflight≤{inflight_cap} "
            f"wall={wall}s hard={wall+60:.0f}s",
            flush=True,
        )

        def _handle(r: dict) -> None:
            nonlocal n_done
            rows.append(r)
            done.add((int(r["idx"]), int(r["k"])))
            n_done += 1
            flag = ""
            if r.get("timed_out"):
                flag = " TIMEOUT"
            print(
                f"[label] SYN_{int(r['idx']):05d} K={r.get('k')} "
                f"sim={r.get('sim')} valid={r.get('valid')}{flag} "
                f"({n_done}/{total}) {batch_tag}",
                flush=True,
            )
            if n_done % save_every == 0 or n_done == total:
                _persist(
                    batch_i=batch_i,
                    batch_n=batch_n,
                    pending_left=total - n_done,
                )

        if jobs <= 1:
            for i, k in pending:
                _handle(run_one_synth(i, k, wall, exports_s))
            return n_done

        with ProcessPoolExecutor(max_workers=jobs) as ex:
            pending_iter = iter(pending)
            futs = {}

            def _submit_one() -> bool:
                try:
                    i, k = next(pending_iter)
                except StopIteration:
                    return False
                fut = ex.submit(run_one_synth, i, k, wall, exports_s)
                futs[fut] = (i, k)
                return True

            for _ in range(min(inflight_cap, len(pending))):
                if not _submit_one():
                    break

            while futs:
                for fut in as_completed(list(futs.keys())):
                    i, k = futs.pop(fut)
                    try:
                        r = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        r = {
                            "idx": i,
                            "k": k,
                            "valid": False,
                            "timed_out": False,
                            "error": str(exc)[-300:],
                        }
                    _handle(r)
                    while len(futs) < inflight_cap and _submit_one():
                        pass
                    break
        return n_done

    # Map-level batches
    n_batches = (len(idxs) + batch_maps - 1) // batch_maps
    total_pending_est = 0
    for bi in range(n_batches):
        chunk = idxs[bi * batch_maps : (bi + 1) * batch_maps]
        pending: List[Tuple[int, int]] = []
        for i in chunk:
            for k in _ks_for(i):
                if (i, k) not in done:
                    pending.append((i, k))
        total_pending_est += len(pending)
        print(
            f"[label] === batch {bi+1}/{n_batches} maps "
            f"{chunk[0]}-{chunk[-1]} pending_K={len(pending)} ===",
            flush=True,
        )
        _run_jobs(
            pending,
            batch_tag=f"batch {bi+1}/{n_batches}",
            batch_i=bi + 1,
            batch_n=n_batches,
        )
        best, delta = _recompute()
        _persist(batch_i=bi + 1, batch_n=n_batches, pending_left=0)
        print(
            f"[label] batch {bi+1}/{n_batches} done; "
            f"labeled_maps={len(best)} → {out_path}",
            flush=True,
        )

    best, delta = _recompute()
    _persist(batch_i=n_batches, batch_n=n_batches, pending_left=0)
    print(
        f"[label] ALL DONE labeled={len(best)} rows={len(rows)} → {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
