"""FACN-Net: Fleet Activation & Concurrency Network.

Predicts Δ on top of Throat-Band Capacity so open maps (e.g. SH20) raise
concurrent busy AGVs and congested maps can throttle. Geometry hard-gates
via Adaptive FACN Authority (AFA).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from greed_coop_mcts.paths import CKPT
# GRID/LEGAL_MIN no longer needed after map-CNN (v2–v4) removal.

# Default runtime ckpt: absolute-K MLP v1 (v2–v4 map/CNN lines removed).
FACN_CKPT_K_V1 = CKPT / "facn_k_v1.pt"
FACN_CKPT = FACN_CKPT_K_V1
FACN_CKPT_V1 = CKPT / "facn_v1.pt"  # legacy Δ-head; fallback only
# Optional per-slot empirical Δ* from oracle sweep (gate raise override).
ORACLE_DELTA: Dict[int, int] = {}

# Δ ∈ {-2,-1,0,+1,+2,+3,+4}  (legacy target)
DELTA_VALUES: Tuple[int, ...] = (-2, -1, 0, 1, 2, 3, 4)
N_DELTA = len(DELTA_VALUES)
DELTA_OFFSET = 2  # class i ↔ delta = i - 2

# Absolute K classes (preferred target): K ∈ {2..16}
K_MIN_CLS = 2
K_MAX_CLS = 16
K_VALUES: Tuple[int, ...] = tuple(range(K_MIN_CLS, K_MAX_CLS + 1))
N_K = len(K_VALUES)
# Flood maps (SH39-class): fixed-K sweep dies at K<=6. Never schedule below this.
FLOOD_K_FLOOR = 7

STATIC_FEAT = 10
DYN_FEAT = 12
FEAT_DIM = STATIC_FEAT + DYN_FEAT

# AFA: geometry only caps max raise Δ; attach if spare idle.
OPEN_NEAR_HI = 40  # max Δ +4
MID_NEAR_HI = 70  # max Δ +2
# else max Δ +1
FLOOD_NEAR_HI = 80  # ban Δ ≤ -2 (SH39-class)
# Small fleets (gate SH01/02/03/08 class): no raise unless oracle taught otherwise.
GATE_N_HI = 12
# Switch: when False, small fleets (n≤GATE_N_HI) use the same near-based raise
# caps as larger fleets — FACN may freely pick K above k_tbc.
AFA_BLOCK_SMALL_FLEET_RAISE = False

FACN_TTL = 36
FACN_PERIOD = 24
FACN_MIN_GAP = 12  # never re-decide faster than this (stops no_prog thrash)
# Progress-fraction edges for staged K (early / mid / late). Crossing an edge
# forces a re-decide so K is not lifelong-fixed at cold start.
FACN_PHASE_EDGES: Tuple[float, ...] = (0.33, 0.66)
FACN_PHASE_STEP = 2  # max |ΔK| allowed when a phase boundary fires
K_MIN = 2
K_MAX_ABS = 16
# Soft floor: never starve MAPD below TBC-1 (and never below 4 while work remains)
K_FLOOR_ABS = 4
STARVE_DONE_RATE = 0.02
STARVE_STREAK_TO_RECOVER = 2


def delta_to_class(delta: int) -> int:
    d = int(np.clip(int(delta), DELTA_VALUES[0], DELTA_VALUES[-1]))
    return int(d + DELTA_OFFSET)


def class_to_delta(cls: int) -> int:
    return int(DELTA_VALUES[int(np.clip(cls, 0, N_DELTA - 1))])


def k_to_class(k: int) -> int:
    kk = int(np.clip(int(k), K_MIN_CLS, K_MAX_CLS))
    return int(kk - K_MIN_CLS)


def class_to_k(cls: int) -> int:
    return int(K_VALUES[int(np.clip(cls, 0, N_K - 1))])


def build_static_feat(
    *,
    n_agents: int,
    free: int,
    near_pad_deg2: int,
    n_pickups: int,
    n_dropoffs: int,
    k_tbc: int,
    region_free: Optional[Sequence[float]] = None,
) -> np.ndarray:
    n = max(1, int(n_agents))
    fr = max(1, int(free))
    dens = float(n) / float(fr)
    idle_pool = max(0.0, float(n - int(k_tbc)) / float(n))
    # region free: take mean / max of up to 16 bins, pad
    rf = list(region_free or [])
    rf_mean = float(np.mean(rf)) / 50.0 if rf else 0.0
    rf_max = float(np.max(rf)) / 80.0 if rf else 0.0
    return np.array(
        [
            min(1.0, n / 30.0),
            min(1.0, fr / 400.0),
            min(1.0, int(near_pad_deg2) / 160.0),
            min(1.0, int(n_pickups) / 12.0),
            min(1.0, int(n_dropoffs) / 12.0),
            min(1.0, dens / 0.15),
            min(1.0, idle_pool),
            min(1.0, int(k_tbc) / 16.0),
            min(1.0, rf_mean),
            min(1.0, rf_max),
        ],
        dtype=np.float32,
    )


def build_dyn_feat(
    *,
    busy: int,
    idle: int,
    remaining: int,
    done_rate: float,
    stall: int,
    avg_h: float,
    n_repair_win: int,
    conflict_heat: float,
    region_occ: float,
    n_dest_gates: int,
    same_dest_cluster: int,
    throat_occ: int,
    n_agents: int,
    k_cur: int,
) -> np.ndarray:
    n = max(1, int(n_agents))
    return np.array(
        [
            min(1.0, busy / 16.0),
            min(1.0, idle / 30.0),
            min(1.0, remaining / 100.0),
            min(1.0, max(0.0, done_rate) / 0.25),
            min(1.0, stall / 80.0),
            min(1.0, avg_h / 28.0),
            min(1.0, n_repair_win / 200.0),
            min(1.0, conflict_heat),
            min(1.0, region_occ / 4.0),
            min(1.0, n_dest_gates / 20.0),
            min(1.0, same_dest_cluster / 8.0),
            min(1.0, throat_occ / 8.0),
        ],
        dtype=np.float32,
    )


def load_oracle_deltas(path: Path | None = None) -> Dict[int, int]:
    """Load Δ*=K*-k_tbc from facn_oracle_sweep_v2.json into ORACLE_DELTA."""
    global ORACLE_DELTA
    from greed_coop_mcts.paths import RESULTS

    path = Path(path or (RESULTS / "facn_oracle_sweep_v2.json"))
    if not path.is_file():
        return dict(ORACLE_DELTA)
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[int, int] = {}
    if isinstance(raw.get("delta_star"), dict):
        out = {int(k): int(v) for k, v in raw["delta_star"].items()}
    elif isinstance(raw.get("best"), dict):
        # derive Δ* if geoms present
        geoms = raw.get("geoms") or {}
        for sk, kv in raw["best"].items():
            g = geoms.get(str(sk)) or geoms.get(int(sk)) or {}
            k_tbc = int(g.get("k_tbc") or 6)
            out[int(sk)] = int(kv) - k_tbc
    ORACLE_DELTA = out
    return dict(ORACLE_DELTA)


def afa_raise_cap(
    *,
    n_agents: int,
    free: int,
    near_pad_deg2: int,
    k_tbc: int,
    slot: int | None = None,
) -> int:
    """Max positive Δ allowed. Geometry caps only — does not veto attach.

    Gate-sized fleets (n≤12) default to 0 raise (SH01/02/03/08 class),
    unless oracle Δ* > 0 for that slot.
    """
    del free, k_tbc  # reserved for dens-aware caps later
    n = max(1, int(n_agents))
    near = int(near_pad_deg2)
    if AFA_BLOCK_SMALL_FLEET_RAISE and n <= GATE_N_HI:
        if slot is not None and int(slot) in ORACLE_DELTA and ORACLE_DELTA[int(slot)] > 0:
            return int(min(2, ORACLE_DELTA[int(slot)]))
        return 0
    if near <= OPEN_NEAR_HI:
        return 4
    if near <= MID_NEAR_HI:
        return 2
    return 1


def afa_allow_raise(
    *,
    n_agents: int,
    free: int,
    near_pad_deg2: int,
    k_tbc: int,
    slot: int | None = None,
) -> bool:
    return (
        afa_raise_cap(
            n_agents=n_agents,
            free=free,
            near_pad_deg2=near_pad_deg2,
            k_tbc=k_tbc,
            slot=slot,
        )
        > 0
    )


def afa_allow_lower(
    *,
    stall: int,
    throat_occ: int,
    no_prog: bool,
    same_dest_cluster: int,
    near_pad_deg2: int = 0,
) -> bool:
    """Throttle only under *structural* congestion + pressure.

    Bare ``no_prog`` alone used to allow Δ→−2 on already-throttled fleets,
    which collapses claim rate and locks the stall (SH39 death spiral).
    Flood maps (near>80) still may lower by 1, never −2 (caller enforces).
    """
    del near_pad_deg2
    structural = int(throat_occ) >= 4 or int(same_dest_cluster) >= 4
    pressure = int(stall) >= 40 or bool(no_prog)
    return bool(structural and pressure)


def afa_allow_attach(
    *,
    n_agents: int,
    free: int,
    near_pad_deg2: int,
    k_tbc: int,
) -> bool:
    """Attach whenever the fleet has spare capacity beyond TBC."""
    del free, near_pad_deg2
    return int(n_agents) > int(k_tbc)


def teacher_delta(
    *,
    n_agents: int,
    free: int,
    near_pad_deg2: int,
    k_tbc: int,
    busy: int = 0,
    stall: int = 0,
    throat_occ: int = 0,
    no_prog: bool = False,
    same_dest_cluster: int = 0,
    done_rate: float = 0.1,
    slot: int | None = None,
) -> int:
    """Heuristic oracle used for IL labels / neural fallback."""
    raise_cap = afa_raise_cap(
        n_agents=n_agents,
        free=free,
        near_pad_deg2=near_pad_deg2,
        k_tbc=k_tbc,
        slot=slot,
    )
    lower_ok = afa_allow_lower(
        stall=stall,
        throat_occ=throat_occ,
        no_prog=no_prog,
        same_dest_cluster=same_dest_cluster,
        near_pad_deg2=near_pad_deg2,
    )
    n = max(1, int(n_agents))
    idle_pool = float(n - int(k_tbc)) / float(n)
    near = int(near_pad_deg2)
    # Congestion throttle (never −2 on flood maps)
    if lower_ok and raise_cap <= 0:
        if (
            same_dest_cluster >= 5
            and throat_occ >= 5
            and near <= FLOOD_NEAR_HI
        ):
            return -2
        return -1
    if raise_cap > 0 and idle_pool > 0:
        if idle_pool >= 0.55 and near <= 20:
            base = 4 if n >= 24 else 3
        elif idle_pool >= 0.40:
            base = 3 if near <= 30 else 2
        elif idle_pool >= 0.25:
            base = 2
        else:
            base = 1
        base = min(base, raise_cap)
        if done_rate < 0.03 and stall >= 30:
            base = max(0, base - 1)
        if lower_ok:
            base = max(0, base - 1)
        return int(np.clip(base, 0, raise_cap))
    if lower_ok:
        return -1
    return 0


class FacnNet(nn.Module):
    """v1: MLP on handcrafted 22-d features (absolute K or legacy Δ)."""

    def __init__(self, feat_dim: int = FEAT_DIM, hidden: int = 64, n_out: int = N_DELTA):
        super().__init__()
        self.arch = "facn_k_v1"
        self.n_out = int(n_out)
        self.target = "delta" if int(n_out) == N_DELTA else "k"
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_facn(net: nn.Module, path: Path | None = None, **meta) -> Path:
    path = Path(path or FACN_CKPT)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("target", getattr(net, "target", "delta"))
    meta.setdefault("n_out", int(getattr(net, "n_out", N_DELTA)))
    meta.setdefault("arch", getattr(net, "arch", "facn_k_v1"))
    torch.save({"state_dict": net.state_dict(), "meta": meta}, path)
    return path


def load_facn(path: Path | None = None, *, device: str = "cpu") -> Optional[nn.Module]:
    import os

    env_p = os.environ.get("GREED_FACN_CKPT", "").strip()
    if path is None and env_p:
        path = Path(env_p)
    path = Path(path or FACN_CKPT)
    if not path.is_file():
        if path == FACN_CKPT and FACN_CKPT_V1.is_file():
            path = FACN_CKPT_V1
        else:
            return None
    blob = torch.load(path, map_location=device, weights_only=False)
    meta = blob.get("meta") or {}
    arch = str(meta.get("arch") or "facn_k_v1")
    if any(x in arch for x in ("v2", "v3", "v4", "MapV2", "Geom")):
        raise ValueError(
            f"FACN arch {arch!r} was removed; use facn_k_v1.pt "
            f"(or set GREED_FACN_CKPT to a v1 checkpoint)."
        )
    target = str(meta.get("target") or "delta")
    n_out = int(meta.get("n_out") or (N_K if target == "k" else N_DELTA))
    net = FacnNet(n_out=n_out).to(device)
    net.target = target
    net.arch = "facn_k_v1" if target == "k" else str(arch or "facn_k_v1")
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net


@dataclass
class FacnController:
    """Runtime FACN: sticky concurrency K (via Δ=K-k_tbc) with AFA + rate limit.

    Network may predict absolute K (``target='k'``) or legacy Δ (``target='delta'``).
    """

    net: Optional[nn.Module] = None
    device: str = "cpu"
    k_tbc: int = 6
    n_agents: int = 0
    free: int = 0
    near_pad_deg2: int = 0
    slot: Optional[int] = None
    delta: int = 0
    target_mode: str = "delta"  # "delta" | "k"
    arch: str = "facn_k_v1"
    ttl: int = 0
    n_updates: int = 0
    n_neural: int = 0
    n_recoveries: int = 0
    enabled: bool = True
    period: int = FACN_PERIOD
    min_gap: int = FACN_MIN_GAP
    ttl_reset: int = FACN_TTL
    last_logit_margin: float = 0.0
    # Staged concurrency: re-decide K when episode progress crosses phase edges.
    phased: bool = True
    phase: int = 0
    n_phase_updates: int = 0
    last_update_t: int = -10_000
    starve_streak: int = 0
    static_feat: np.ndarray = field(
        default_factory=lambda: np.zeros(STATIC_FEAT, dtype=np.float32)
    )

    @classmethod
    def maybe_load(
        cls,
        *,
        device: str = "cpu",
        enabled: bool = True,
        k_tbc: int = 6,
        n_agents: int = 0,
        free: int = 0,
        near_pad_deg2: int = 0,
        n_pickups: int = 0,
        n_dropoffs: int = 0,
        region_free: Optional[Sequence[float]] = None,
        slot: Optional[int] = None,
        blocked: Optional[Set[Tuple[int, int]]] = None,
        pickup_cells: Optional[Sequence[Tuple[int, int]]] = None,
        dropoff_cells: Optional[Sequence[Tuple[int, int]]] = None,
        agv_poses: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> "FacnController":
        if not ORACLE_DELTA:
            load_oracle_deltas()
        net = load_facn(device=device) if enabled else None
        target_mode = str(getattr(net, "target", "delta") if net is not None else "delta")
        arch = str(getattr(net, "arch", "facn_k_v1") if net is not None else "facn_k_v1")
        # Prefer oracle Δ* baked into checkpoint meta when present.
        if net is not None:
            try:
                import os

                env_p = os.environ.get("GREED_FACN_CKPT", "").strip()
                ckpt_p = Path(env_p) if env_p else (
                    FACN_CKPT if FACN_CKPT.is_file() else FACN_CKPT_V1
                )
                blob = torch.load(ckpt_p, map_location="cpu", weights_only=False)
                meta = blob.get("meta") or {}
                od = meta.get("oracle_delta") or meta.get("delta_star") or {}
                if od:
                    ORACLE_DELTA.update({int(k): int(v) for k, v in od.items()})
                if meta.get("target"):
                    target_mode = str(meta.get("target"))
                if meta.get("arch"):
                    arch = str(meta.get("arch"))
            except Exception:
                pass
        # GREED_FACN_PHASED=0 disables progress-stage K retarget (lifelong sticky).
        import os as _os

        phased_env = str(_os.environ.get("GREED_FACN_PHASED", "1")).strip().lower()
        phased = phased_env not in ("0", "false", "off", "no")
        ctrl = cls(
            net=net,
            device=device,
            k_tbc=int(k_tbc),
            n_agents=int(n_agents),
            free=int(free),
            near_pad_deg2=int(near_pad_deg2),
            slot=int(slot) if slot is not None else None,
            enabled=bool(enabled),
            target_mode=target_mode,
            arch=arch,
            phased=phased,
            phase=0,
            period=FACN_PERIOD,
            min_gap=FACN_MIN_GAP,
            ttl_reset=FACN_TTL,
        )
        ctrl.static_feat = build_static_feat(
            n_agents=n_agents,
            free=free,
            near_pad_deg2=near_pad_deg2,
            n_pickups=n_pickups,
            n_dropoffs=n_dropoffs,
            k_tbc=k_tbc,
            region_free=region_free,
        )
        # Cold start from teacher (or net with zero dyn)
        dyn0 = build_dyn_feat(
            busy=0,
            idle=n_agents,
            remaining=100,
            done_rate=0.0,
            stall=0,
            avg_h=float(FACN_PERIOD),
            n_repair_win=0,
            conflict_heat=0.0,
            region_occ=0.0,
            n_dest_gates=0,
            same_dest_cluster=0,
            throat_occ=0,
            n_agents=n_agents,
            k_cur=k_tbc,
        )
        ctrl.delta = ctrl._decide(dyn0, force=True, agv_poses=agv_poses)
        ctrl.ttl = int(ctrl.ttl_reset)
        return ctrl

    def _forward_net(
        self,
        dyn: np.ndarray,
        agv_poses: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> int:
        """Run neural head; returns class id (Δ-class or K-class)."""
        del agv_poses  # v1 MLP has no map channels
        assert self.net is not None
        gfeat = np.concatenate([self.static_feat, dyn]).astype(np.float32)
        self.n_neural += 1
        with torch.no_grad():
            logits = self.net(
                torch.from_numpy(gfeat[None]).to(self.device)
            )[0].cpu().numpy()
        x = logits.astype(np.float64)
        x = x - float(np.max(x))
        ex = np.exp(x)
        prob = ex / max(1e-12, float(ex.sum()))
        order = np.argsort(prob)[::-1]
        top1 = float(prob[order[0]])
        top2 = float(prob[order[1]]) if len(order) > 1 else 0.0
        self.last_logit_margin = top1 - top2
        return int(np.argmax(logits))

    def current_k(self, *, remaining: int = 100) -> int:
        k = int(self.k_tbc) + int(self.delta)
        hi = min(K_MAX_ABS, max(K_MIN, self.n_agents))
        lo = K_MIN
        # While work remains, keep a usable claim floor (anti-starvation).
        if int(remaining) > 5:
            lo = max(lo, min(K_FLOOR_ABS, int(self.k_tbc)))
            lo = max(lo, int(self.k_tbc) - 1)
        return int(np.clip(k, lo, hi))

    def _raw_delta(
        self,
        dyn: np.ndarray,
        *,
        agv_poses: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> int:
        """Legacy Δ head (or teacher)."""
        if self.net is None or str(self.target_mode) != "delta":
            return teacher_delta(
                n_agents=self.n_agents,
                free=self.free,
                near_pad_deg2=self.near_pad_deg2,
                k_tbc=self.k_tbc,
                busy=int(dyn[0] * 16),
                stall=int(dyn[4] * 80),
                throat_occ=int(dyn[11] * 8),
                no_prog=bool(dyn[4] > 0.5),
                same_dest_cluster=int(dyn[10] * 8),
                done_rate=float(dyn[3] * 0.25),
                slot=self.slot,
            )
        return class_to_delta(self._forward_net(dyn, agv_poses=agv_poses))

    def _raw_k(
        self,
        dyn: np.ndarray,
        *,
        agv_poses: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> int:
        """Absolute-K head, k_tbc+Δ head, or teacher→K."""
        mode = str(self.target_mode)
        if self.net is not None and mode == "k":
            return class_to_k(self._forward_net(dyn, agv_poses=agv_poses))
        return int(self.k_tbc) + int(self._raw_delta(dyn, agv_poses=agv_poses))

    def _decide(
        self,
        dyn: np.ndarray,
        *,
        force: bool = False,
        phase_boundary: bool = False,
        remaining: int = 100,
        done_rate: float = 0.0,
        agv_poses: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> int:
        raise_cap = afa_raise_cap(
            n_agents=self.n_agents,
            free=self.free,
            near_pad_deg2=self.near_pad_deg2,
            k_tbc=self.k_tbc,
            slot=self.slot,
        )
        raise_ok = raise_cap > 0
        throat_occ = int(dyn[11] * 8)
        same_cl = int(dyn[10] * 8)
        lower_ok = afa_allow_lower(
            stall=int(dyn[4] * 80),
            throat_occ=throat_occ,
            no_prog=bool(dyn[4] > 0.55),
            same_dest_cluster=same_cl,
            near_pad_deg2=self.near_pad_deg2,
        )

        if str(self.target_mode) == "k":
            raw_k = int(self._raw_k(dyn, agv_poses=agv_poses))
            # Flood labyrinths livelock when concurrency falls to K<=6 (SH39).
            if int(self.near_pad_deg2) > FLOOD_NEAR_HI:
                raw_k = max(raw_k, max(FLOOD_K_FLOOR, int(self.k_tbc)))
            # Absolute-K head: allow full fleet range; AFA only soft-gates.
            hi = min(K_MAX_ABS, max(K_MIN, int(self.n_agents)))
            lo = K_MIN
            if not raise_ok:
                hi = min(hi, int(self.k_tbc))
            elif raise_cap < 4:
                # mid/gate maps: still allow modest extra beyond TBC+cap
                hi = min(hi, int(self.k_tbc) + max(int(raise_cap), 2) + 2)
            if not lower_ok:
                lo = max(lo, int(self.k_tbc))
            elif lower_ok:
                lo = max(
                    lo,
                    int(self.k_tbc)
                    - (
                        2
                        if (
                            throat_occ >= 5
                            and same_cl >= 4
                            and int(self.near_pad_deg2) <= FLOOD_NEAR_HI
                        )
                        else 1
                    ),
                )
            # Anti-starvation climb toward TBC.
            if (
                int(self.delta) < 0
                and int(remaining) > 5
                and float(done_rate) < STARVE_DONE_RATE
            ):
                self.starve_streak += 1
            else:
                self.starve_streak = 0
            if self.starve_streak >= STARVE_STREAK_TO_RECOVER:
                raw_k = max(raw_k, int(self.k_tbc) + min(0, int(self.delta) + 1))
                self.n_recoveries += 1
                self.starve_streak = 0
            sticky_k = int(self.k_tbc) + int(self.delta)
            if phase_boundary:
                step = FACN_PHASE_STEP
                if raw_k > sticky_k + step:
                    raw_k = sticky_k + step
                if raw_k < sticky_k - step:
                    raw_k = sticky_k - step
            elif not force:
                if raw_k > sticky_k + 1:
                    raw_k = sticky_k + 1
                if raw_k < sticky_k - 1:
                    raw_k = sticky_k - 1
            raw_k = int(np.clip(raw_k, max(K_MIN, lo), min(K_MAX_ABS, max(hi, lo))))
            return int(raw_k - int(self.k_tbc))

        raw = self._raw_delta(dyn, agv_poses=agv_poses)
        if raw > 0 and not raise_ok:
            raw = 0
        if raw > raise_cap:
            raw = int(raise_cap)
        if raw < 0 and not lower_ok:
            raw = 0
        # −2 only for extreme structural pileups; never on flood maps.
        if raw <= -2 and (
            int(self.near_pad_deg2) > FLOOD_NEAR_HI
            or not (throat_occ >= 5 and same_cl >= 4)
        ):
            raw = -1
        # Anti-starvation: climb back toward TBC if throttle killed throughput.
        if (
            int(self.delta) < 0
            and int(remaining) > 5
            and float(done_rate) < STARVE_DONE_RATE
        ):
            self.starve_streak += 1
        else:
            self.starve_streak = 0
        if self.starve_streak >= STARVE_STREAK_TO_RECOVER:
            raw = max(raw, min(0, int(self.delta) + 1))
            self.n_recoveries += 1
            self.starve_streak = 0
        # rate limit vs current sticky delta (wider at phase boundaries)
        if phase_boundary:
            step = FACN_PHASE_STEP
            if raw > self.delta + step:
                raw = self.delta + step
            if raw < self.delta - step:
                raw = self.delta - step
        elif not force:
            if raw > self.delta + 1:
                raw = self.delta + 1
            if raw < self.delta - 1:
                raw = self.delta - 1
        return int(np.clip(raw, DELTA_VALUES[0], DELTA_VALUES[-1]))

    @staticmethod
    def _phase_of(done_frac: float) -> int:
        p = 0
        for edge in FACN_PHASE_EDGES:
            if float(done_frac) >= float(edge):
                p += 1
            else:
                break
        return int(p)

    def maybe_update(
        self,
        t: int,
        *,
        busy: int,
        idle: int,
        remaining: int,
        done_rate: float,
        stall: int,
        avg_h: float,
        n_repair_win: int = 0,
        conflict_heat: float = 0.0,
        region_occ: float = 0.0,
        n_dest_gates: int = 0,
        same_dest_cluster: int = 0,
        throat_occ: int = 0,
        no_prog: bool = False,
        force: bool = False,
        total: int = 0,
        agv_poses: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> int:
        if not self.enabled:
            return self.current_k(remaining=remaining)

        tot = max(int(total), int(remaining), 1)
        done_frac = 1.0 - float(remaining) / float(tot)
        new_phase = self._phase_of(done_frac)
        phase_fire = bool(self.phased and new_phase > int(self.phase))
        if phase_fire:
            force = True

        # Flood labyrinths: old lifelong freeze → only skip *within-phase*
        # tick noise; phase boundaries may still retarget K.
        if int(self.near_pad_deg2) > FLOOD_NEAR_HI and not force:
            return self.current_k(remaining=remaining)
        if self.ttl > 0:
            self.ttl -= 1
        gap = int(t) - int(self.last_update_t)
        due = force or gap >= self.period or self.ttl <= 0
        # Event wake-ups still respect MIN_GAP (prevents per-tick thrash).
        event = (
            gap >= int(self.min_gap)
            and (
                (no_prog and (throat_occ >= 4 or same_dest_cluster >= 4))
                or (stall >= 40 and busy >= self.current_k(remaining=remaining) - 1)
            )
        )
        if not due and not event:
            return self.current_k(remaining=remaining)
        dyn = build_dyn_feat(
            busy=busy,
            idle=idle,
            remaining=remaining,
            done_rate=done_rate,
            stall=stall,
            avg_h=avg_h,
            n_repair_win=n_repair_win,
            conflict_heat=conflict_heat,
            region_occ=region_occ,
            n_dest_gates=n_dest_gates,
            same_dest_cluster=same_dest_cluster,
            throat_occ=throat_occ,
            n_agents=self.n_agents,
            k_cur=self.current_k(remaining=remaining),
        )
        # fold no_prog into stall channel already via teacher; bump for AFA
        if no_prog:
            dyn = dyn.copy()
            dyn[4] = max(dyn[4], 0.6)
        new_d = self._decide(
            dyn,
            force=force,
            phase_boundary=phase_fire,
            remaining=int(remaining),
            done_rate=float(done_rate),
            agv_poses=agv_poses,
        )
        if new_d != self.delta or force:
            self.delta = new_d
            self.n_updates += 1
            if phase_fire:
                self.n_phase_updates += 1
        if phase_fire:
            self.phase = int(new_phase)
        self.ttl = int(self.ttl_reset)
        self.last_update_t = int(t)
        return self.current_k(remaining=remaining)
