"""Generate connected easy/hard maps (SH21+) and eval with current rrsc stack."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ml_research.benchmarks.hier_coord.scene_difficulty.map_gen import (  # noqa: E402
    gen_map,
    is_connected,
)
from ml_research.benchmarks.scenarios.generator import (  # noqa: E402
    END_POINTS,
    START_POINTS,
    gen_tasks,
    write_position_csv,
    _write_tasks,
)
from greed_coop_mcts.paths import MAP_EXPORTS  # noqa: E402
from greed_coop_mcts.pgcp import (  # noqa: E402
    adaptive_use_stage_net,
    near_pad_deg2_count,
    throat_band_fleet_cap,
    e4_sticky_enabled,
)
from greed_coop_mcts.scenario import load_scenario, load_slot_meta, patch_extra_obstacles  # noqa: E402
from greed_coop_mcts.solve import solve  # noqa: E402
from greed_coop_mcts.validate import validate_trajectory  # noqa: E402

EXPORTS = MAP_EXPORTS

# (slot, style, seed, n_agvs, task_pattern, difficulty_tag)
SPECS = [
    # Easy / open
    (21, "open", 2101, 8, "uniform", "easy"),
    (22, "sprinkle_mild", 2202, 12, "uniform", "easy"),
    (23, "corridor_mild", 2303, 9, "mixed", "easy"),
    (24, "open", 2404, 15, "uniform", "easy"),
    (33, "sprinkle_mild", 3301, 10, "uniform", "easy"),
    (34, "open", 3402, 6, "mixed", "easy"),
    (35, "hutong_mild", 3503, 10, "uniform", "easy"),
    # Medium
    (25, "corridor_hard", 2505, 12, "cross_flow", "medium"),
    (26, "hutong", 2606, 12, "clustered_dest", "medium"),
    (27, "sprinkle", 2707, 16, "mixed", "medium"),
    (28, "corridor_hard", 2808, 18, "cross_flow", "medium"),
    (36, "sprinkle", 3604, 14, "cross_flow", "medium"),
    (37, "hutong_mild", 3705, 16, "clustered_dest", "medium"),
    (38, "maze_mild", 3806, 10, "mixed", "medium"),
    # Hard / dense throat (avoid extreme maze×16 livelock)
    (29, "maze", 2909, 8, "clustered_dest", "hard"),
    (30, "hutong", 3010, 20, "clustered_dest", "hard"),
    (31, "maze_mild", 3118, 10, "clustered_dest", "hard"),
    (32, "hutong", 3212, 9, "clustered_dest", "hard"),
    (39, "sprinkle_dense", 3907, 18, "cross_flow", "hard"),
    (40, "maze", 4010, 10, "clustered_dest", "hard"),
]


def _pickups_dropoffs():
    pickups = [{"name": n, "x": x, "y": y} for n, x, y in START_POINTS]
    dropoffs = [{"name": n, "x": x, "y": y} for n, x, y in END_POINTS]
    return pickups, dropoffs


def _geom(slot: int) -> dict:
    meta = load_slot_meta(slot)
    assert meta
    _m, env, _a, _t, _n = load_scenario(
        Path(meta["task_csv"]), Path(meta["position_csv"]), force_reload=True
    )
    if meta.get("extra_obstacles"):
        patch_extra_obstacles(env, list(meta["extra_obstacles"]))
    static = {(int(p[0]), int(p[1])) for p in (env.get_static_obstacles() or [])}
    j = json.loads(Path(meta["map_json"]).read_text(encoding="utf-8"))
    pickups = {(int(p["x"]), int(p["y"])) for p in j.get("pickups") or []}
    dropoffs = {(int(p["x"]), int(p["y"])) for p in j.get("dropoffs") or []}
    stations = pickups | dropoffs
    blocked = (static - stations) | stations
    near = near_pad_deg2_count(blocked, dropoffs)
    n = int(meta.get("n_agvs") or 0)
    return {
        "n": n,
        "near": near,
        "asa": int(adaptive_use_stage_net(n, near)),
        "cap": throat_band_fleet_cap(n, near, base=6),
        "cebi": int(n >= 10 and near >= 100),
        "e4": int(e4_sticky_enabled(n, near)),
        "n_obs": len(j.get("extra_obstacles") or []),
    }


def generate_one(slot: int, style: str, seed: int, n_agvs: int, pattern: str, tag: str) -> Path:
    sid = f"SH_custom_{slot:02d}"
    g = gen_map(style, seed)
    assert g["connected"], f"{sid} not connected"
    obs_set = {tuple(c) for c in g["obstacles"]}
    assert is_connected(obs_set), f"{sid} connectivity recheck failed"

    pickups, dropoffs = _pickups_dropoffs()
    jpath = EXPORTS / f"{sid}.json"
    pos = EXPORTS / f"{sid}_position.csv"
    task = EXPORTS / f"{sid}_task.csv"

    write_position_csv(pos, n_agvs)
    rows = gen_tasks(pattern, 100, seed + 17)
    assert len(rows) == 100, f"{sid} expected 100 tasks, got {len(rows)}"
    _write_tasks(task, rows)

    payload = {
        "id": sid,
        "map_id": sid,
        "format_version": 2,
        "grid_min": 1,
        "grid_max": 20,
        "n_agvs": int(n_agvs),
        "n_pickups": len(pickups),
        "n_dropoffs": len(dropoffs),
        "n_tasks": 100,
        "pickups": pickups,
        "dropoffs": dropoffs,
        "extra_obstacles": g["obstacles"],
        "notes": f"gen style={style} seed={seed} difficulty={tag} pattern={pattern}",
        "map_shape": style,
        "difficulty": tag,
        "gen_seed": int(seed),
        "connected": True,
        "compatible_with": "curriculum_shape ladder (position CSV + task CSV + extra_obstacles)",
    }
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return jpath


def eval_one(slot: int) -> dict:
    meta = load_slot_meta(slot, max_sim_time=200000, wall_timeout=600.0, n_tasks=100)
    assert meta
    meta = dict(meta)
    meta["id"] = f"SH{slot:02d}_gen"
    meta["n_tasks"] = 100
    geom = _geom(slot)
    rep = solve(
        slot,
        meta=meta,
        policy="rrsc",
        use_stage_net=True,
        device="cpu",
        wall_timeout=600.0,
        max_sim_time=200000,
    )
    sim = int(rep.get("sim_time") or -1)
    done = int(rep.get("tasks_completed") or 0)
    v_ok = bool(rep.get("validate_ok"))
    iss: dict = {}
    traj = rep.get("trajectory") or rep.get("trajectory_csv") or rep.get("traj_csv")
    if traj and Path(traj).exists():
        vrep = validate_trajectory(meta, Path(traj))
        iss = dict(vrep.get("issues") or {})
        v_ok = bool(vrep.get("ok"))
    elif rep.get("validate_summary"):
        # fall back to solve-time validation flags
        v_ok = bool(rep.get("validate_ok"))
    ok = v_ok and done == 100
    return {
        "slot": slot,
        "sim": sim,
        "done": done,
        "ok": ok,
        "validate_ok": v_ok,
        "mot": int(iss.get("n_illegal_motion") or 0),
        "coll": int(iss.get("n_collisions") or 0),
        "swap": int(iss.get("n_swaps") or 0),
        "vnote": rep.get("validate_summary"),
        "wall": rep.get("wall_seconds"),
        **geom,
        "e4_sticky": rep.get("n_e4_sticky"),
        "egress": rep.get("n_pad_egress"),
        "lane": rep.get("n_lane_locks"),
    }


def main() -> int:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    print("=== GENERATE ===", flush=True)
    for slot, style, seed, n_agvs, pattern, tag in SPECS:
        p = generate_one(slot, style, seed, n_agvs, pattern, tag)
        g = _geom(slot)
        print(
            f"SH{slot:02d} [{tag}] style={style} n={n_agvs} near={g['near']} "
            f"obs={g['n_obs']} asa={g['asa']} cap={g['cap']} cebi={g['cebi']} e4={g['e4']} "
            f"-> {p.name}",
            flush=True,
        )

    print("\n=== EVAL (current rrsc) ===", flush=True)
    rows = []
    for slot, style, seed, n_agvs, pattern, tag in SPECS:
        r = eval_one(slot)
        r.update(style=style, difficulty=tag, pattern=pattern)
        rows.append(r)
        print(
            f"SH{slot:02d} [{tag}] sim={r['sim']} done={r['done']}/100 ok={r['ok']} "
            f"mot={r['mot']} near={r['near']} n={r['n']} wall={r['wall']}",
            flush=True,
        )

    out = Path(__file__).resolve().parent / "GEN_CURRICULUM_EVAL.md"
    easy = [r for r in rows if r["difficulty"] == "easy"]
    med = [r for r in rows if r["difficulty"] == "medium"]
    hard = [r for r in rows if r["difficulty"] == "hard"]
    ok_n = sum(1 for r in rows if r["ok"])
    done100 = sum(1 for r in rows if r["done"] == 100)
    sims = [r["sim"] for r in rows if r["ok"]]
    sims_done = [r["sim"] for r in rows if r["done"] == 100]
    lines = [
        "# Generated maps SH21–40 vs current rrsc stack",
        "",
        f"- maps: {len(rows)} (easy={len(easy)}, medium={len(med)}, hard={len(hard)})",
        f"- all free-space 4-connected: **yes** (asserted at gen)",
        f"- complete 100/100: **{done100}/{len(rows)}**",
        f"- complete 100/100 VALID: **{ok_n}/{len(rows)}**",
        f"- avg sim (VALID only): **{sum(sims)/len(sims):.1f}**" if sims else "- avg sim (VALID): n/a",
        (
            f"- avg sim (done=100): **{sum(sims_done)/len(sims_done):.1f}**"
            if sims_done
            else "- avg sim (done=100): n/a"
        ),
        "",
        "| Slot | diff | style | n | near | obs | sim | done | ok | mot | asa | cebi | e4 |",
        "|------|------|-------|---|------|-----|-----|------|----|-----|----|------|----|",
    ]
    for r in rows:
        lines.append(
            f"| SH{r['slot']:02d} | {r['difficulty']} | {r['style']} | {r['n']} | "
            f"{r['near']} | {r['n_obs']} | {r['sim']} | {r['done']}/100 | {int(r['ok'])} | "
            f"{r['mot']} | {r['asa']} | {r['cebi']} | {r['e4']} |"
        )
    fail = [r for r in rows if not r["ok"]]
    if fail:
        lines += ["", "## Failures", ""]
        for r in fail:
            reason = (
                f"incomplete done={r['done']}"
                if r["done"] < 100
                else f"INVALID mot={r['mot']} coll={r['coll']} swap={r['swap']}"
            )
            lines.append(
                f"- SH{r['slot']:02d} [{r['difficulty']}/{r['style']}] n={r['n']} "
                f"near={r['near']}: {reason}"
            )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (Path(__file__).resolve().parent / "GEN_CURRICULUM_EVAL.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out}", flush=True)
    return 0 if ok_n == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
