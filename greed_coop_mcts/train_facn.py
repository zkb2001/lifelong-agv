"""IL train FACN: classify absolute K (preferred) or legacy Δ from oracle/teacher."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

os.environ.setdefault("PYTHONHASHSEED", "0")

from greed_coop_mcts.facn_net import (
    FEAT_DIM,
    FACN_CKPT,
    N_DELTA,
    N_K,
    FacnNet,
    build_dyn_feat,
    build_static_feat,
    class_to_delta,
    class_to_k,
    delta_to_class,
    k_to_class,
    save_facn,
    teacher_delta,
)
from greed_coop_mcts.paths import MAP_EXPORTS_SYNTH, RESULTS
from greed_coop_mcts.pgcp import near_pad_deg2_count, throat_band_fleet_cap
from greed_coop_mcts.rule_obs import GRID, seal_ghost_border
from greed_coop_mcts.scenario import load_slot_meta

Cell = Tuple[int, int]
_MAP: Dict[int, dict] = {}
_SYNTH_ROOT = MAP_EXPORTS_SYNTH


def _load_geom(slot: int) -> Optional[dict]:
    if slot in _MAP:
        return dict(_MAP[slot])
    meta = None
    if int(slot) >= 10000:
        from greed_coop_mcts.synth_maps.generate import load_synth_meta

        try:
            meta = load_synth_meta(int(slot), exports=_SYNTH_ROOT)
        except Exception:
            meta = None
    else:
        meta = load_slot_meta(slot)
    if not meta:
        return None
    jpath = Path(meta.get("map_json") or "")
    blocked: Set[Cell] = set()
    drops: Set[Cell] = set()
    picks: Set[Cell] = set()
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
            picks.add(c)
            blocked.add(c)
    blocked = seal_ghost_border(blocked)
    free = sum(
        1 for x in range(1, GRID) for y in range(1, GRID) if (x, y) not in blocked
    )
    n = int(meta.get("n_agvs") or 8)
    near = near_pad_deg2_count(blocked, drops)
    k_tbc = throat_band_fleet_cap(n, near, base=6)
    g = {
        "n": n,
        "free": free,
        "near": near,
        "n_pick": len(picks),
        "n_drop": len(drops),
        "k_tbc": k_tbc,
    }
    _MAP[slot] = g
    return dict(g)


def _split_maps(
    ids: List[int], seed: int, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    train = ids[:n_tr]
    val = ids[n_tr : n_tr + n_va]
    test = ids[n_tr + n_va :]
    if not test and val:
        test = [val.pop()]
    if not val and train:
        val = [train.pop()]
    return {"train": train, "val": val, "test": test}


def _sample(
    rng: random.Random,
    oracle: Dict[int, int],
    train_ids: List[int],
    *,
    oracle_frac: float,
    allow_real_sh: bool,
    target_mode: str = "k",
):
    use_oracle = bool(train_ids) and rng.random() < float(oracle_frac)
    if use_oracle:
        slot = int(rng.choice(train_ids))
        geom = _load_geom(slot)
        if geom is None:
            use_oracle = False
    if not use_oracle:
        if allow_real_sh and rng.random() < 0.35:
            slot = rng.randint(1, 40)
            geom = _load_geom(slot)
            if geom is None:
                slot = -1
                geom = None
        else:
            open_like = rng.random() < 0.45
            n = rng.randint(20, 30) if open_like else rng.randint(8, 20)
            free = rng.randint(320, 390) if open_like else rng.randint(180, 300)
            near = rng.randint(8, 35) if open_like else rng.randint(55, 120)
            geom = {
                "n": n,
                "free": free,
                "near": near,
                "n_pick": rng.randint(2, 8),
                "n_drop": rng.randint(2, 8),
                "k_tbc": throat_band_fleet_cap(n, near, base=6),
            }
            slot = -1

    if geom is None:
        open_like = rng.random() < 0.45
        n = rng.randint(20, 30) if open_like else rng.randint(8, 20)
        free = rng.randint(320, 390) if open_like else rng.randint(180, 300)
        near = rng.randint(8, 35) if open_like else rng.randint(55, 120)
        geom = {
            "n": n,
            "free": free,
            "near": near,
            "n_pick": rng.randint(2, 8),
            "n_drop": rng.randint(2, 8),
            "k_tbc": throat_band_fleet_cap(n, near, base=6),
        }
        slot = -1

    stall = rng.randint(0, 80)
    throat = rng.randint(0, 8)
    same_cl = rng.randint(1, 6)
    no_prog = rng.random() < 0.15
    busy = rng.randint(0, min(geom["n"], geom["k_tbc"] + 4))
    done_rate = rng.uniform(0.0, 0.2)

    if slot in oracle:
        target_k = int(oracle[slot])
        delta = int(np.clip(target_k - geom["k_tbc"], -2, 4))
    else:
        delta = teacher_delta(
            n_agents=geom["n"],
            free=geom["free"],
            near_pad_deg2=geom["near"],
            k_tbc=geom["k_tbc"],
            busy=busy,
            stall=stall,
            throat_occ=throat,
            no_prog=no_prog,
            same_dest_cluster=same_cl,
            done_rate=done_rate,
        )
        target_k = int(geom["k_tbc"]) + int(delta)

    target_k = int(np.clip(target_k, 2, min(16, int(geom["n"]))))
    k_cur = target_k if str(target_mode) == "k" else int(geom["k_tbc"]) + int(delta)

    static = build_static_feat(
        n_agents=geom["n"],
        free=geom["free"],
        near_pad_deg2=geom["near"],
        n_pickups=geom["n_pick"],
        n_dropoffs=geom["n_drop"],
        k_tbc=geom["k_tbc"],
        region_free=[rng.uniform(5, 40) for _ in range(8)],
    )
    dyn = build_dyn_feat(
        busy=busy,
        idle=max(0, geom["n"] - busy),
        remaining=rng.randint(10, 100),
        done_rate=done_rate,
        stall=stall,
        avg_h=rng.uniform(2, 20),
        n_repair_win=rng.randint(0, 100),
        conflict_heat=rng.random() * 0.3,
        region_occ=rng.uniform(0.5, 3.0),
        n_dest_gates=rng.randint(0, 15),
        same_dest_cluster=same_cl,
        throat_occ=throat,
        n_agents=geom["n"],
        k_cur=k_cur,
    )
    feat = np.concatenate([static, dyn]).astype(np.float32)
    if str(target_mode) == "k":
        return feat, k_to_class(target_k), slot
    return feat, delta_to_class(delta), slot


def _map_hit_rate(
    net: FacnNet,
    device: str,
    oracle: Dict[int, int],
    ids: List[int],
    *,
    target_mode: str = "k",
) -> float:
    if not ids:
        return float("nan")
    ok = 0
    for s in ids:
        g = _load_geom(s)
        if not g or s not in oracle:
            continue
        target_k = int(np.clip(int(oracle[s]), 2, min(16, int(g["n"]))))
        target_d = int(np.clip(target_k - int(g["k_tbc"]), -2, 4))
        static = build_static_feat(
            n_agents=g["n"],
            free=g["free"],
            near_pad_deg2=g["near"],
            n_pickups=g["n_pick"],
            n_dropoffs=g["n_drop"],
            k_tbc=g["k_tbc"],
        )
        dyn = build_dyn_feat(
            busy=min(g["n"], g["k_tbc"]),
            idle=max(0, g["n"] - g["k_tbc"]),
            remaining=80,
            done_rate=0.08,
            stall=10,
            avg_h=8.0,
            n_repair_win=0,
            conflict_heat=0.05,
            region_occ=1.0,
            n_dest_gates=2,
            same_dest_cluster=2,
            throat_occ=1,
            n_agents=g["n"],
            k_cur=target_k if str(target_mode) == "k" else g["k_tbc"] + target_d,
        )
        feat = np.concatenate([static, dyn]).astype(np.float32)
        with torch.no_grad():
            pred = int(net(torch.from_numpy(feat[None]).to(device)).argmax(-1).item())
        if str(target_mode) == "k":
            if class_to_k(pred) == target_k:
                ok += 1
        elif class_to_delta(pred) == target_d:
            ok += 1
    return ok / max(1, len(ids))


def main() -> int:
    global _SYNTH_ROOT
    ap = argparse.ArgumentParser(description="Train FACN-Net")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--out", type=str, default=str(FACN_CKPT))
    ap.add_argument(
        "--oracle",
        type=str,
        default=str(RESULTS / "facn_oracle_sweep_v2.json"),
    )
    ap.add_argument("--synth-root", type=str, default=str(MAP_EXPORTS_SYNTH))
    ap.add_argument(
        "--oracle-frac",
        type=float,
        default=0.55,
        help="fraction of samples drawn from train-split oracle maps",
    )
    ap.add_argument(
        "--no-real-sh",
        action="store_true",
        help="do not mix real SH01-40 geometries (synth-only training)",
    )
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--arch", type=str, default="facn_v2")
    ap.add_argument(
        "--target",
        type=str,
        default="k",
        choices=["k", "delta"],
        help="predict absolute K (default) or legacy Δ",
    )
    args = ap.parse_args()
    _SYNTH_ROOT = Path(args.synth_root)
    target_mode = str(args.target)

    oracle: Dict[int, int] = {}
    if args.oracle and Path(args.oracle).is_file():
        raw = json.loads(Path(args.oracle).read_text(encoding="utf-8"))
        if "best" in raw:
            raw = raw["best"]
        oracle = {int(k): int(v) for k, v in raw.items()}

    splits = _split_maps(sorted(oracle.keys()), int(args.split_seed))
    train_ids = splits["train"]
    val_ids = splits["val"]
    test_ids = splits["test"]
    n_out = N_K if target_mode == "k" else N_DELTA
    print(
        f"[facn] target={target_mode} n_out={n_out} oracle={len(oracle)} "
        f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}",
        flush=True,
    )

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = str(args.device)
    net = FacnNet(n_out=n_out).to(device)
    net.target = target_mode
    opt = AdamW(net.parameters(), lr=float(args.lr), weight_decay=1e-4)

    best_acc = -1.0
    best_val_hit = -1.0
    best_state = None
    run_l = 0.0
    run_a = 0.0
    n_log = 0
    hist: List[dict] = []
    allow_sh = not bool(args.no_real_sh)

    for step in range(1, int(args.steps) + 1):
        xb = np.zeros((args.batch, FEAT_DIM), dtype=np.float32)
        yb = np.zeros((args.batch,), dtype=np.int64)
        for b in range(int(args.batch)):
            f, lab, _ = _sample(
                rng,
                oracle,
                train_ids,
                oracle_frac=float(args.oracle_frac),
                allow_real_sh=allow_sh,
                target_mode=target_mode,
            )
            xb[b] = f
            yb[b] = lab
        t_x = torch.from_numpy(xb).to(device)
        t_y = torch.from_numpy(yb).to(device)
        logits = net(t_x)
        loss = F.cross_entropy(logits, t_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            acc = float((logits.argmax(-1) == t_y).float().mean().item())
        run_l += float(loss.item())
        run_a += acc
        n_log += 1
        if step % 200 == 0 or step == 1:
            al = run_l / max(1, n_log)
            aa = run_a / max(1, n_log)
            val_hit = _map_hit_rate(
                net, device, oracle, val_ids, target_mode=target_mode
            )
            print(
                f"[facn] step={step} loss={al:.4f} acc={aa:.3f} "
                f"val_map_hit={val_hit:.3f}",
                flush=True,
            )
            hist.append(
                {"step": step, "loss": al, "acc": aa, "val_map_hit": val_hit}
            )
            if val_hit > best_val_hit or (val_hit != val_hit and aa > best_acc):
                best_val_hit = val_hit if val_hit == val_hit else best_val_hit
                best_acc = aa
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                delta_star = {}
                for s, k in oracle.items():
                    g = _load_geom(int(s))
                    if g:
                        delta_star[int(s)] = int(k) - int(g["k_tbc"])
                save_facn(
                    net,
                    Path(args.out),
                    steps=step,
                    acc=aa,
                    val_map_hit=val_hit,
                    arch=str(args.arch),
                    target=target_mode,
                    n_out=n_out,
                    oracle_slots=sorted(oracle),
                    oracle_delta=delta_star,
                    split=splits,
                )
            run_l = run_a = 0.0
            n_log = 0

    if best_state is not None:
        net.load_state_dict(best_state)
    test_hit = _map_hit_rate(
        net, device, oracle, test_ids, target_mode=target_mode
    )
    delta_star = {}
    for s, k in oracle.items():
        g = _load_geom(int(s))
        if g:
            delta_star[int(s)] = int(k) - int(g["k_tbc"])
    out = save_facn(
        net,
        Path(args.out),
        steps=int(args.steps),
        acc=best_acc,
        val_map_hit=best_val_hit,
        test_map_hit=test_hit,
        arch=str(args.arch),
        target=target_mode,
        n_out=n_out,
        oracle_slots=sorted(oracle),
        oracle_delta=delta_star,
        split=splits,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    log_name = f"train_{Path(args.out).stem}.json"
    (RESULTS / log_name).write_text(
        json.dumps(
            {
                "out": str(out),
                "target": target_mode,
                "n_out": n_out,
                "best_acc": best_acc,
                "best_val_hit": best_val_hit,
                "test_map_hit": test_hit,
                "hist": hist,
                "oracle_n": len(oracle),
                "split": splits,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[facn] saved {out} best_acc={best_acc:.3f} "
        f"val_hit={best_val_hit:.3f} test_hit={test_hit:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
