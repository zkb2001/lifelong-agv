"""Policy wrappers: ParkArterySemaphoreNet (v5) + legacy TrafficRuleNet (v4)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from .config import (
    CKPT_PATH,
    CKPT_PATH_V4,
    CKPT_TAG,
    CKPT_TAG_V4,
    GRID,
    N_MAP_CHANNELS,
    N_MAP_CHANNELS_V4,
    SEMAPHORE_THRESH,
    ensure_dirs,
)
from .cost_field import apply_construction_disturbance, rules_to_numpy
from .features import (
    build_map_channels,
    build_map_channels_ecbs,
    build_park_artery_channels,
    snapshot_agv_positions,
)
from .net import ParkArterySemaphoreNet, TrafficRuleNet
from .teacher import (
    build_sem_features,
    teacher_construction_field,
    teacher_from_channels,
    teacher_park_artery_from_channels,
)

Cell = Tuple[int, int]
Pose = Tuple[int, int, int]
AssignedLike = Union[dict, Sequence[dict], Mapping[str, dict]]


class ParkArteryPolicy:
    """v5 policy: parking / artery / connectivity / semaphore."""

    def __init__(
        self,
        net: Optional[ParkArterySemaphoreNet] = None,
        ckpt: Optional[Path] = None,
        device: Optional[str] = None,
        *,
        use_teacher_if_missing: bool = True,
    ):
        ensure_dirs()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = (net or ParkArterySemaphoreNet()).to(self.device)
        self.use_teacher = False
        self._cached: Optional[Dict[str, Any]] = None
        path = Path(ckpt) if ckpt else CKPT_PATH
        if path.exists():
            try:
                payload = torch.load(path, map_location=self.device, weights_only=False)
                tag = payload.get("tag")
                n_ch = payload.get("n_map_channels")
                if tag is not None and tag != CKPT_TAG:
                    raise RuntimeError(f"tag mismatch {tag} != {CKPT_TAG}")
                if n_ch is not None and int(n_ch) != N_MAP_CHANNELS:
                    raise RuntimeError(f"channel mismatch {n_ch} != {N_MAP_CHANNELS}")
                self.net.load_state_dict(payload["model"], strict=True)
                self.net.eval()
                print(f"[PARK-ARTERY] loaded {path} tag={tag}", flush=True)
            except Exception as exc:  # noqa: BLE001
                if use_teacher_if_missing:
                    self.use_teacher = True
                    print(f"[PARK-ARTERY] ckpt incompatible ({exc}); teacher fallback", flush=True)
                else:
                    raise
        elif use_teacher_if_missing:
            self.use_teacher = True
            print("[PARK-ARTERY] no ckpt; using teacher fields", flush=True)
        else:
            self.net.eval()

    def infer(
        self,
        *,
        channels: Optional[np.ndarray] = None,
        sim=None,
        sem_pos: Optional[Cell] = None,
        sem_goal: Optional[Cell] = None,
        pad_free: float = 1.0,
        path_clear: float = 1.0,
    ) -> Dict[str, Any]:
        if channels is None:
            channels = build_park_artery_channels(sim)
        if channels.shape[0] != N_MAP_CHANNELS:
            fixed = np.zeros(
                (N_MAP_CHANNELS, channels.shape[1], channels.shape[2]), dtype=np.float32
            )
            n = min(N_MAP_CHANNELS, channels.shape[0])
            fixed[:n] = channels[:n]
            channels = fixed

        goals = []
        if sem_goal is not None:
            goals = [sem_goal]
        elif float(channels[5].max()) > 0:
            ys, xs = np.where(channels[5] >= float(channels[5].max()) * 0.9)
            goals = [(int(x), int(y)) for y, x in zip(ys, xs)]

        if self.use_teacher:
            lab = teacher_park_artery_from_channels(
                channels,
                goals=goals,
                sem_pos=sem_pos,
                sem_goal=sem_goal,
                pad_free=pad_free,
                path_clear=path_clear,
            )
            out = {
                "parking": lab["parking"],
                "artery": lab["artery"],
                "connectivity": lab["connectivity"],
                "semaphore": float(lab["semaphore"]),
                "sem_feat": lab["sem_feat"],
                "walkable": channels[1],
                "channels": channels,
                "teacher": True,
            }
            self._cached = out
            return out

        # network fields
        x = torch.as_tensor(channels, device=self.device).unsqueeze(0)
        # default sem feat from teacher geometry for pooling context
        lab0 = teacher_park_artery_from_channels(
            channels, goals=goals, sem_pos=sem_pos, sem_goal=sem_goal,
            pad_free=pad_free, path_clear=path_clear,
        )
        sem_feat = torch.as_tensor(lab0["sem_feat"], device=self.device).unsqueeze(0)
        with torch.no_grad():
            act = self.net.act(x, sem_feat=sem_feat, deterministic=True)
        out = {
            "parking": act["parking"][0].detach().cpu().numpy().astype(np.float32),
            "artery": act["artery"][0].detach().cpu().numpy().astype(np.float32),
            "connectivity": act["connectivity"][0].detach().cpu().numpy().astype(np.float32),
            "semaphore": float(act["semaphore"][0].detach().cpu().item()),
            "sem_feat": lab0["sem_feat"],
            "walkable": channels[1],
            "channels": channels,
            "teacher": False,
        }
        self._cached = out
        return out

    def semaphore_allow(
        self,
        fields: Dict[str, Any],
        *,
        pos: Cell,
        goal: Cell,
        pad_free: float,
        path_clear: float,
        thresh: float = SEMAPHORE_THRESH,
    ) -> bool:
        # Hard safety: never activate if path would interfere with active agents
        if float(path_clear) < 0.5:
            return False
        if self.use_teacher or "artery" in fields:
            ch = fields.get("channels")
            if ch is not None and getattr(ch, "shape", None) is not None and ch.shape[0] > 9:
                cong = np.asarray(ch[9], dtype=np.float32)
            else:
                cong = np.zeros_like(np.asarray(fields["artery"]))
            feat = build_sem_features(
                artery=np.asarray(fields["artery"]),
                connectivity=np.asarray(fields["connectivity"]),
                congestion=cong,
                pad_free=pad_free,
                path_clear=path_clear,
                pos=pos,
                goal=goal,
            )
            if self.use_teacher:
                from .teacher import teacher_semaphore_label

                lab = teacher_semaphore_label(
                    connectivity=np.asarray(fields["connectivity"]),
                    pos=pos,
                    goal=goal,
                    pad_free=pad_free,
                    path_clear=path_clear,
                    congestion=float(feat[2]),
                )
                return float(lab) >= thresh
            x = torch.as_tensor(fields["channels"], device=self.device).unsqueeze(0)
            sem_feat = torch.as_tensor(feat, device=self.device).unsqueeze(0)
            with torch.no_grad():
                out = self.net.act(x, sem_feat=sem_feat)
                p = float(out["semaphore"][0].detach().cpu().item())
            return p >= thresh
        return float(path_clear) >= 0.5 and float(pad_free) >= 0.5


class LaneTrafficPolicy:
    """Legacy v4 policy (edge-cost shaping)."""

    def __init__(
        self,
        net: Optional[TrafficRuleNet] = None,
        ckpt: Optional[Path] = None,
        device: Optional[str] = None,
        *,
        use_teacher_if_missing: bool = True,
        urgency: float = 0.0,
        with_parking_head: bool = False,
    ):
        ensure_dirs()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = (net or TrafficRuleNet(with_parking_head=with_parking_head)).to(self.device)
        self.use_teacher = False
        self.urgency = float(urgency)
        self._prev_pos: Dict[str, Any] = {}
        self._cached_rules: Optional[Dict[str, np.ndarray]] = None
        self._construction: Optional[np.ndarray] = None
        path = Path(ckpt) if ckpt else CKPT_PATH_V4
        if path.exists():
            try:
                payload = torch.load(path, map_location=self.device, weights_only=False)
                tag = payload.get("tag")
                n_ch = payload.get("n_map_channels")
                if tag is not None and tag != CKPT_TAG_V4:
                    raise RuntimeError(f"tag mismatch {tag} != {CKPT_TAG_V4}")
                if n_ch is not None and int(n_ch) != N_MAP_CHANNELS_V4:
                    raise RuntimeError(f"channel mismatch {n_ch} != {N_MAP_CHANNELS_V4}")
                missing, unexpected = self.net.load_state_dict(payload["model"], strict=False)
                bad = [k for k in missing if "parking" not in k]
                if bad or unexpected:
                    raise RuntimeError(f"state_dict mismatch missing={missing} unexpected={unexpected}")
                self.net.eval()
                print(f"[LANE-AI] loaded {path} tag={tag}", flush=True)
            except Exception as exc:  # noqa: BLE001
                if use_teacher_if_missing:
                    self.use_teacher = True
                    print(f"[LANE-AI] ckpt incompatible ({exc}); using teacher rules", flush=True)
                else:
                    raise
        elif use_teacher_if_missing:
            self.use_teacher = True
            print("[LANE-AI] no ckpt; using unidirectional teacher rules", flush=True)
        else:
            self.net.eval()

    def set_construction(
        self,
        closed: Optional[np.ndarray] = None,
        *,
        density: float = 0.0,
        seed: Optional[int] = None,
        walkable: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if closed is not None:
            self._construction = np.asarray(closed, dtype=np.float32)
        else:
            walk = walkable
            if walk is None:
                walk = np.zeros((21, 21), dtype=np.float32)
                walk[1:21, 1:21] = 1.0
            self._construction = teacher_construction_field(
                walk, density=density, seed=seed
            )
        return self._construction

    def clear_construction(self) -> None:
        self._construction = None

    def observe(
        self,
        sim,
        *,
        assigned_tasks: Optional[AssignedLike] = None,
    ) -> np.ndarray:
        construction = self._construction
        if construction is None and sim is not None:
            construction = getattr(sim, "_lane_construction", None)
        ch = build_map_channels(
            sim,
            prev_positions=self._prev_pos or None,
            assigned_tasks=assigned_tasks,
            construction=construction,
        )
        self._prev_pos = snapshot_agv_positions(sim)
        return ch

    @torch.no_grad()
    def rules_for_sim(
        self,
        sim=None,
        channels: Optional[np.ndarray] = None,
        *,
        assigned_tasks: Optional[AssignedLike] = None,
        urgency: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        urg = self.urgency if urgency is None else float(urgency)
        if channels is None:
            if sim is None:
                raise ValueError("sim or channels required")
            channels = self.observe(sim, assigned_tasks=assigned_tasks)
        if channels.shape[0] != N_MAP_CHANNELS_V4:
            fixed = np.zeros(
                (N_MAP_CHANNELS_V4, channels.shape[1], channels.shape[2]), dtype=np.float32
            )
            n = min(N_MAP_CHANNELS_V4, channels.shape[0])
            fixed[:n] = channels[:n]
            channels = fixed
        if self.use_teacher:
            lab = teacher_from_channels(
                channels,
                construction=self._construction,
                urgency=urg,
            )
            if self._construction is not None:
                lab = apply_construction_disturbance(lab, self._construction, merge=True)
            lab["urgency"] = urg
            self._cached_rules = lab
            return lab
        x = torch.as_tensor(channels, device=self.device).unsqueeze(0)
        out = self.net.act(x, deterministic=True)
        rules = rules_to_numpy(out)
        if self._construction is not None:
            rules = apply_construction_disturbance(rules, self._construction, merge=True)
        rules["urgency"] = urg
        self._cached_rules = rules
        return rules

    def rules_for_ecbs(
        self,
        *,
        static: Set[Cell],
        stations: Set[Cell],
        pose: Mapping[str, Pose],
        assigned: Mapping[str, dict],
        open_pickups: Optional[Sequence[Cell]] = None,
        urgency: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        ch = build_map_channels_ecbs(
            static=static,
            stations=stations,
            pose=pose,
            assigned=assigned,
            open_pickups=open_pickups,
            prev_positions=self._prev_pos or None,
        )
        self._prev_pos = {n: (int(p[0]), int(p[1])) for n, p in pose.items()}
        return self.rules_for_sim(channels=ch, urgency=urgency)

    def current_rules(self) -> Dict[str, np.ndarray]:
        if self._cached_rules is None:
            walk = np.zeros((21, 21), dtype=np.float32)
            walk[1:21, 1:21] = 1.0
            ch = np.zeros((N_MAP_CHANNELS_V4, 21, 21), dtype=np.float32)
            ch[1] = walk
            self._cached_rules = teacher_from_channels(ch, urgency=self.urgency)
        return self._cached_rules


def save_ckpt(
    net,
    path: Optional[Path] = None,
    extra: Optional[dict] = None,
    *,
    tag: str = CKPT_TAG,
    n_map_channels: int = N_MAP_CHANNELS,
) -> Path:
    ensure_dirs()
    path = Path(path or CKPT_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": net.state_dict(), "tag": tag, "n_map_channels": int(n_map_channels)}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[PARK-ARTERY] saved {path}", flush=True)
    return path
