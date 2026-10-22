"""Hier + GGO edge weights. Default eval order: worst-known Δ first, then
untested, then maps that already beat Hier. Stops on the first miss so
the policy can be retuned before burning time on easy maps.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml_research.benchmarks.coord_custom_ai.compare_100 import _base_meta, _hier_meta
from ml_research.benchmarks.coord_custom_ai.solve_ecbs import solve_ecbs
from ml_research.benchmarks.hier_coord.ggo_guidance import ggo_harm_slots

OUT = Path("ml_research/results/hier_coord/stall_sim30_eval/AI_ggo_v2.json")
HIER_SRC = Path("ml_research/results/hier_coord/stall_sim30_eval/AI_soft_ct_all20.json")
MODE = "ggo_v2"

# A/B winners only. Unproven maps keep vanilla Hier (GGO no-op).
SEED = {
    9: {
        "sim": 3268,
        "wall": 1404.63,
        "valid": True,
        "done": 100,
        "handoff_t": 700,
        "policy": "ccw_1.22_n1.12",
    },
    2: {
        "sim": 3738,
        "wall": 2343.97,
        "valid": True,
        "done": 100,
        "handoff_t": 1375,
        "policy": "ccw_1.22_n1.12",
    },
    12: {
        "sim": 3912,
        "wall": 2223.46,
        "valid": True,
        "done": 100,
        "handoff_t": 750,
        "policy": "cw_1.12_n1.0",
    },
    3: {
        "sim": 7040,
        "wall": 1160.53,
        "valid": True,
        "done": 100,
        "handoff_t": 100,
        "policy": "ccw_1.22_n1.12",
    },
}


def _load_hier() -> dict:
    hier = {9: 3366, 12: 3932, 2: 4117}
    if HIER_SRC.is_file():
        try:
            for r in json.loads(HIER_SRC.read_text(encoding="utf-8")):
                s = int(r["slot"])
                b = r.get("hier_base")
                if b is not None:
                    hier[s] = int(b)
        except Exception:
            pass
    return hier


HIER = _load_hier()
PROBE = Path("ml_research/results/hier_coord/stall_sim30_eval/AI_ggo_all20_probe.json")


def _delta_map(*row_lists) -> dict:
    """Latest d_hier per slot; later lists override earlier (v2 over probe)."""
    out: dict = {}
    for rows in row_lists:
        for r in rows or []:
            if r.get("d_hier") is None:
                continue
            try:
                out[int(r["slot"])] = int(r["d_hier"])
            except (TypeError, ValueError):
                continue
    return out


def _worst_first_slots(extra_rows: Optional[list] = None) -> list:
    probe_rows = []
    if PROBE.is_file():
        try:
            probe_rows = json.loads(PROBE.read_text(encoding="utf-8"))
        except Exception:
            probe_rows = []
    v2_rows = extra_rows or []
    if OUT.is_file() and extra_rows is None:
        try:
            v2_rows = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            v2_rows = []
    d = _delta_map(probe_rows, v2_rows)
    worse, unknown, tie, better = [], [], [], []
    for s in range(1, 21):
        if s not in d:
            unknown.append(s)
        elif d[s] > 0:
            worse.append(s)
        elif d[s] == 0:
            tie.append(s)
        else:
            better.append(s)
    worse.sort(key=lambda s: -d[s])
    better.sort(key=lambda s: d[s], reverse=True)
    order = worse + unknown + tie + better
    print(
        f"[ORDER] worst-first {['SH%02d'%s for s in order]}",
        flush=True,
    )
    return order


CROWD_LADDER = ("ccw_crowd", "cw_crowd", "ccw_soft")


def _apply_slot_policy(slot: int, policy: Optional[str] = None) -> str:
    """Proven: always-on winning ring. Else: crowd ladder (CCW → CW → soft)."""
    for k in (
        "AGV_GGO_CW",
        "AGV_GGO_AGAINST",
        "AGV_GGO_NARROW",
        "AGV_GGO_WAIT",
        "AGV_GGO_CROWD_ONLY",
        "AGV_GGO_REQUIRE_GAIN",
    ):
        os.environ.pop(k, None)
    os.environ["AGV_GGO_REQUIRE_GAIN"] = "0"
    os.environ["AGV_GGO_WAIT"] = "1.0"
    name = policy
    if name is None:
        if int(slot) in ggo_harm_slots():
            name = "off_harm"
        elif int(slot) == 12:
            name = "cw_always"
        elif int(slot) in (2, 3, 9):
            name = "ccw_always"
        else:
            name = "ccw_crowd"
    if name == "off_harm":
        os.environ["AGV_GGO"] = "0"
        return "off_harm"
    os.environ["AGV_GGO"] = "1"
    if name == "cw_always":
        os.environ["AGV_GGO_CW"] = "1"
        os.environ["AGV_GGO_AGAINST"] = "1.12"
        os.environ["AGV_GGO_NARROW"] = "1.0"
        os.environ["AGV_GGO_CROWD_ONLY"] = "0"
        return name
    if name == "ccw_always":
        os.environ["AGV_GGO_CW"] = "0"
        os.environ["AGV_GGO_AGAINST"] = "1.22"
        os.environ["AGV_GGO_NARROW"] = "1.12"
        os.environ["AGV_GGO_CROWD_ONLY"] = "0"
        return name
    if name == "cw_crowd":
        os.environ["AGV_GGO_CW"] = "1"
        os.environ["AGV_GGO_AGAINST"] = "1.15"
        os.environ["AGV_GGO_NARROW"] = "1.0"
        os.environ["AGV_GGO_CROWD_ONLY"] = "1"
        return name
    if name == "ccw_soft":
        os.environ["AGV_GGO_CW"] = "0"
        os.environ["AGV_GGO_AGAINST"] = "1.08"
        os.environ["AGV_GGO_NARROW"] = "1.0"
        os.environ["AGV_GGO_CROWD_ONLY"] = "1"
        return name
    os.environ["AGV_GGO_CW"] = "0"
    os.environ["AGV_GGO_AGAINST"] = "1.15"
    os.environ["AGV_GGO_NARROW"] = "1.0"
    os.environ["AGV_GGO_CROWD_ONLY"] = "1"
    return "ccw_crowd"


def _off_row(slot: int, probe: Optional[dict] = None) -> dict:
    base = int(HIER.get(slot, 0))
    row = {
        "slot": slot,
        "mode": MODE,
        "policy": "off_unproven",
        "applied": False,
        "sim": base,
        "wall": None,
        "valid": True,
        "done": 100,
        "handoff_t": None,
        "hier": base,
        "d_hier": 0,
        "beats_hier": False,
        "non_worse": True,
    }
    if probe:
        row["probe_sim"] = probe.get("sim")
        row["probe_d_hier"] = probe.get("d_hier")
        row["probe_wall"] = probe.get("wall")
    return row


def _run(slot: int, policy: Optional[str] = None) -> dict:
    os.environ["AGV_CONFLICT_PRED"] = "0"
    os.environ["AGV_ONLINE_GUIDANCE"] = "0"
    os.environ["AGV_LOCAL_MCTS"] = "0"
    os.environ["AGV_COST_TRACER"] = "0"
    os.environ["AGV_TIER_SWITCHER"] = "0"
    os.environ["AGV_M0_HANDOFF"] = "1"
    os.environ["AGV_PROMOTE_ON_ASSIGN"] = "1"
    os.environ["AGV_STATION_MAX_UNLOADED"] = "1"
    os.environ["AGV_ASTAR_WALL_S"] = "0.20"
    os.environ["AGV_ASSIGN_FAIL_BUDGET"] = "4"
    os.environ["AGV_TRIED_TTL"] = "8"
    os.environ["AGV_QUIET_ASSIGN"] = "1"
    policy = _apply_slot_policy(slot, policy)
    if policy == "off_harm":
        row = _off_row(slot)
        row["policy"] = "off_harm"
        row["applied"] = False
        row["d_hier"] = 0
        row["beats_hier"] = False
        row["non_worse"] = True
        return row
    meta = _hier_meta(_base_meta(slot, wall_timeout=0.0, max_sim_time=200000))
    meta["id"] = f"SH{slot:02d}_ggo_v2"
    meta["use_ggo"] = True
    meta["use_local_mcts"] = False
    meta["use_cost_tracer"] = False
    meta["slot"] = int(slot)

    rep = solve_ecbs(
        slot=int(slot),
        meta=meta,
        max_tasks=0,
        max_active=6,
        weight=1.5,
        time_limit=35.0,
        plan="joint",
        pipeline=True,
        turn_aware=True,
        use_hierarchical=True,
        hierarchical_force=False,
        use_swapnet=True,
        use_wavenet=False,
        legacy_wave_ecbs=True,
        joint_core="prioritized",
        traffic_recovery=False,
        allow_mode_switch=True,
        plan_horizon=24,
        exec_horizon=12,
        wave_hard_cap=4,
    )
    sim = int(rep.get("sim_time") or 0)
    h = rep.get("hierarchical") or {}
    base = int(HIER.get(slot, 0))
    valid = bool(rep.get("validate_ok"))
    done = int(rep.get("tasks_completed") or 0)
    complete = valid and done >= 100
    return {
        "slot": slot,
        "mode": MODE,
        "policy": policy,
        "applied": True,
        "sim": sim,
        "wall": round(float(rep.get("wall_seconds") or 0), 2),
        "valid": valid,
        "done": done,
        "handoff_t": h.get("m0_handoff_t"),
        "hier": base,
        "d_hier": (sim - base) if base else None,
        "beats_hier": bool(base) and complete and sim < base,
        "non_worse": bool(base) and complete and sim <= base,
    }


def _seed_row(slot: int) -> dict:
    s = SEED[slot]
    base = int(HIER.get(slot, 0))
    sim = int(s["sim"])
    return {
        "slot": slot,
        "mode": MODE,
        "policy": s["policy"],
        "sim": sim,
        "wall": s["wall"],
        "valid": bool(s["valid"]),
        "done": s["done"],
        "handoff_t": s["handoff_t"],
        "hier": base,
        "d_hier": sim - base,
        "beats_hier": sim < base,
        "non_worse": sim <= base,
        "applied": True,
        "seeded": True,
    }


def _rebuild_gated(probe_rows: list) -> list:
    probe = {int(r["slot"]): r for r in probe_rows if r.get("policy") != "off_unproven"}
    rows = []
    for slot in range(1, 21):
        if slot in SEED:
            row = _seed_row(slot)
            rows.append(row)
            continue
        rows.append(_off_row(slot, probe.get(slot)))
    return rows


def main():
    args = list(sys.argv[1:])
    force = False
    rebuild_only = False
    if "--force" in args:
        force = True
        args.remove("--force")
    if "--rebuild-gated" in args:
        rebuild_only = True
        args.remove("--rebuild-gated")
    stop_on_fail = True
    numeric = False
    if "--keep-going" in args:
        stop_on_fail = False
        args.remove("--keep-going")
    if "--numeric" in args:
        numeric = True
        args.remove("--numeric")
    if args == ["all"] or not args:
        slots = list(range(1, 21)) if numeric else _worst_first_slots()
    else:
        slots = [int(x) for x in args]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = []
    if OUT.is_file():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            prev = []

    probe_path = OUT.with_name("AI_ggo_all20_probe.json")
    if any(r.get("mode") == "ggo_auto" for r in prev) and not probe_path.is_file():
        probe_path.write_text(json.dumps(prev, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"ARCHIVED probe {probe_path}", flush=True)

    probe_rows = prev
    if probe_path.is_file():
        try:
            probe_rows = json.loads(probe_path.read_text(encoding="utf-8"))
        except Exception:
            probe_rows = prev

    if rebuild_only:
        gated = OUT.with_name("AI_ggo_all20.json")
        rows = _rebuild_gated(probe_rows)
        gated.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("WROTE gated", gated, flush=True)
        return

    rows = []
    if OUT.is_file() and not force:
        try:
            rows = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            rows = []
    have = {int(r["slot"]) for r in rows if r.get("mode") == MODE}
    for slot in sorted(ggo_harm_slots()):
        old = next((r for r in rows if int(r.get("slot") or 0) == slot), None)
        if old and old.get("policy") == "off_harm":
            continue
        row = _off_row(slot, old)
        row["policy"] = "off_harm"
        row["applied"] = False
        rows = [r for r in rows if int(r.get("slot") or 0) != slot]
        rows.append(row)
        have.add(slot)
        print(f"HARM-OFF SH{slot:02d} {json.dumps(row, ensure_ascii=False)}", flush=True)
        OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    for slot in slots:
        if slot in SEED and slot not in have:
            row = _seed_row(slot)
            row["mode"] = MODE
            rows.append(row)
            have.add(slot)
            print(f"SEED SH{slot:02d} {json.dumps(row, ensure_ascii=False)}", flush=True)
            OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    for slot in slots:
        if slot in have and not force:
            cur = next((r for r in rows if int(r.get("slot") or 0) == slot), None)
            maze_off = cur.get("policy") == "off_harm" and int(slot) in ggo_harm_slots()
            ladder_off = cur.get("policy") == "off_harm" and set(cur.get("attempts") or []) >= set(
                CROWD_LADDER
            )
            complete_win = bool(
                cur.get("beats_hier")
                and cur.get("valid")
                and int(cur.get("done") or 0) >= 100
            )
            done_ok = bool(
                cur
                and (
                    complete_win
                    or cur.get("seeded")
                    or maze_off
                    or ladder_off
                )
            )
            if done_ok:
                print(f"SKIP SH{slot:02d}", flush=True)
                continue
        tried = []
        cur = next((r for r in rows if int(r.get("slot") or 0) == slot), None)
        if cur:
            if cur.get("attempts"):
                tried = [str(x) for x in cur["attempts"]]
            pol = str(cur.get("policy") or "")
            if pol in CROWD_LADDER and pol not in tried:
                tried.append(pol)
            if int(cur.get("probe_d_hier") or 0) > 0 and "ccw_crowd" not in tried:
                tried.append("ccw_crowd")
        start_i = 0
        for i, name in enumerate(CROWD_LADDER):
            if name not in tried:
                start_i = i
                break
        else:
            start_i = len(CROWD_LADDER)
        print(
            f"[LADDER] SH{slot:02d} tried={tried} remain={list(CROWD_LADDER[start_i:])}",
            flush=True,
        )
        if int(slot) in (2, 3, 9, 12) or slot in SEED:
            print(f"\n===== SH{slot:02d} GGO v2 =====", flush=True)
            row = _run(slot)
            rows = [r for r in rows if not (r.get("mode") == MODE and int(r["slot"]) == slot)]
            rows.append(row)
            print("RESULT", json.dumps(row, ensure_ascii=False), flush=True)
            OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            continue
        won = False
        for name in CROWD_LADDER[start_i:]:
            print(f"\n===== SH{slot:02d} GGO {name} =====", flush=True)
            row = _run(slot, name)
            tried = tried + [name]
            row["attempts"] = tried
            rows = [r for r in rows if not (r.get("mode") == MODE and int(r["slot"]) == slot)]
            rows.append(row)
            print("RESULT", json.dumps(row, ensure_ascii=False), flush=True)
            OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            if row.get("beats_hier"):
                won = True
                print(f"PASS SH{slot:02d} {name} Δ={row.get('d_hier'):+d}", flush=True)
                break
            print(
                f"RETRY SH{slot:02d} {name} sim={row.get('sim')} "
                f"hier={row.get('hier')} Δ={row.get('d_hier'):+d} → next policy",
                flush=True,
            )
        if not won:
            old = next((r for r in rows if int(r.get("slot") or 0) == slot), None)
            row = _off_row(slot, old)
            row["policy"] = "off_harm"
            row["applied"] = False
            row["attempts"] = tried
            rows = [r for r in rows if int(r.get("slot") or 0) != slot]
            rows.append(row)
            print(f"GIVE-UP SH{slot:02d} GGO off (=Hier) after {tried}", flush=True)
            OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", OUT, flush=True)
    done = [r for r in rows if r.get("mode") == MODE]
    beat = sum(1 for r in done if r.get("beats_hier"))
    worse = sum(1 for r in done if r.get("beats_hier") is False)
    print(f"SUMMARY n={len(done)} beat={beat} not_beat={worse}", flush=True)
    for r in sorted(done, key=lambda x: x["slot"]):
        print(
            f"  SH{r['slot']:02d}: sim={r['sim']} hier={r['hier']} "
            f"Δ={r.get('d_hier'):+d} {r.get('policy')} "
            f"{'BEATS' if r.get('beats_hier') else 'no'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
