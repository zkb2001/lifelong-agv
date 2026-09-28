"""Rule-based park/artery recovery policy (no TrafficNet).

Static artery/parking from freq cartesian labels; semaphore from teacher rules.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .config import GRID, N_MAP_CHANNELS, SEMAPHORE_THRESH, ensure_dirs
from .features import build_park_artery_channels
from .map_labels import load_map_labels, map_id_from_sim, resolve_map_id
from .teacher import (
    build_sem_features,
    teacher_connectivity_field,
    teacher_semaphore_label,
)

Cell = Tuple[int, int]


class RuleParkArteryPolicy:
    """Frequency artery/parking + rule semaphore. Drop-in for recovery."""

    def __init__(
        self,
        *,
        artery: Optional[np.ndarray] = None,
        parking: Optional[np.ndarray] = None,
        walkable: Optional[np.ndarray] = None,
        map_id: Optional[str] = None,
        meta: Optional[dict] = None,
    ):
        ensure_dirs()
        self.use_teacher = True  # always rule-based
        self.map_id = resolve_map_id(
            str(map_id or (meta or {}).get("id") or ""), meta
        )
        self._cached: Optional[Dict[str, Any]] = None
        if artery is not None and parking is not None:
            self.artery = np.asarray(artery, dtype=np.float32)
            self.parking = np.asarray(parking, dtype=np.float32)
            self.walkable = (
                np.asarray(walkable, dtype=np.float32)
                if walkable is not None
                else np.clip(self.artery + self.parking, 0, 1)
            )
        else:
            labels = load_map_labels(
                self.map_id or "",
                meta=meta,
                compute_if_missing=meta is not None,
            )
            self.artery = labels["artery"]
            self.parking = labels["parking"]
            self.walkable = labels.get("walkable")
            if self.walkable is None or float(np.sum(self.walkable)) <= 0:
                self.walkable = np.clip(self.artery + self.parking, 0, 1).astype(
                    np.float32
                )
            if not self.map_id and meta:
                self.map_id = str(meta.get("id") or "")
        print(
            f"[RULE-PARK] map={self.map_id or '?'} "
            f"artery={(self.artery >= 0.5).sum()} "
            f"parking={(self.parking >= 0.5).sum()}",
            flush=True,
        )

    @classmethod
    def from_meta(cls, meta: dict) -> "RuleParkArteryPolicy":
        return cls(map_id=str(meta.get("id") or ""), meta=meta)

    @classmethod
    def from_sim(cls, sim, meta: Optional[dict] = None) -> "RuleParkArteryPolicy":
        mid = map_id_from_sim(sim) or str((meta or {}).get("id") or "")
        m = meta or getattr(sim, "_scene_meta", None) or getattr(sim, "meta", None)
        return cls(map_id=mid, meta=m if isinstance(m, dict) else None)

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
            if sim is not None:
                channels = build_park_artery_channels(sim)
            else:
                channels = np.zeros((N_MAP_CHANNELS, GRID, GRID), dtype=np.float32)
                channels[1] = self.walkable

        walk = np.asarray(channels[1], dtype=np.float32)
        # Prefer static freq masks; fall back to walkable ∩ mask
        artery = (self.artery >= 0.5).astype(np.float32) * (walk > 0.5).astype(np.float32)
        parking = (self.parking >= 0.5).astype(np.float32) * (walk > 0.5).astype(
            np.float32
        )
        # Dynamic: never park on active committed paths
        active = (
            np.asarray(channels[3], dtype=np.float32)
            if channels.shape[0] > 3
            else None
        )
        if active is not None:
            parking = parking * (active < 0.5).astype(np.float32)

        goals: Sequence[Cell] = []
        if sem_goal is not None:
            goals = [sem_goal]
        elif float(channels[5].max()) > 0:
            ys, xs = np.where(channels[5] >= float(channels[5].max()) * 0.9)
            goals = [(int(x), int(y)) for y, x in zip(ys, xs)]

        connectivity = teacher_connectivity_field(
            walk, artery, goals, active_path=active, parking=parking
        )
        cong = (
            np.asarray(channels[9], dtype=np.float32)
            if channels.shape[0] > 9
            else np.zeros_like(walk)
        )
        if sem_pos is None:
            if channels.shape[0] > 4 and float(channels[4].sum()) > 0:
                ys, xs = np.where(channels[4] > 0.5)
                sem_pos = (int(xs[0]), int(ys[0]))
            else:
                sem_pos = (2, 2)
        if sem_goal is None:
            sem_goal = goals[0] if goals else sem_pos

        cong_here = (
            float(cong[sem_pos[1], sem_pos[0]])
            if 0 <= sem_pos[0] < cong.shape[1] and 0 <= sem_pos[1] < cong.shape[0]
            else 0.0
        )
        sem = teacher_semaphore_label(
            connectivity=connectivity,
            pos=sem_pos,
            goal=sem_goal,
            pad_free=pad_free,
            path_clear=path_clear,
            congestion=cong_here,
        )
        sem_feat = build_sem_features(
            artery=artery,
            connectivity=connectivity,
            congestion=cong,
            pad_free=pad_free,
            path_clear=path_clear,
            pos=sem_pos,
            goal=sem_goal,
        )
        out = {
            "parking": parking.astype(np.float32),
            "artery": artery.astype(np.float32),
            "connectivity": connectivity.astype(np.float32),
            "semaphore": float(sem),
            "sem_feat": sem_feat,
            "walkable": walk,
            "channels": channels,
            "teacher": True,
            "rule": True,
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
        if float(path_clear) < 0.5:
            return False
        ch = fields.get("channels")
        if ch is not None and getattr(ch, "shape", None) is not None and ch.shape[0] > 9:
            cong = np.asarray(ch[9], dtype=np.float32)
        else:
            cong = np.zeros_like(np.asarray(fields["artery"]))
        x, y = int(pos[0]), int(pos[1])
        cong_here = (
            float(cong[y, x])
            if 0 <= x < cong.shape[1] and 0 <= y < cong.shape[0]
            else 0.0
        )
        lab = teacher_semaphore_label(
            connectivity=np.asarray(fields["connectivity"]),
            pos=pos,
            goal=goal,
            pad_free=pad_free,
            path_clear=path_clear,
            congestion=cong_here,
        )
        return float(lab) >= thresh
