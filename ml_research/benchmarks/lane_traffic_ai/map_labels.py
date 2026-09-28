"""Load / compute static artery + parking masks (no neural net)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .config import GRID, RESULTS_DIR, ensure_dirs

FREQ_DIR = RESULTS_DIR / "freq_probe"
ARTERY_MAPS_DIR = RESULTS_DIR / "artery_maps"

_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def ensure_artery_maps_dir() -> Path:
    ensure_dirs()
    ARTERY_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTERY_MAPS_DIR


def latest_freq_run() -> Optional[Path]:
    if not FREQ_DIR.exists():
        return None
    runs = sorted(FREQ_DIR.glob("run_*"), key=lambda p: p.name)
    return runs[-1] if runs else None


def install_artery_maps_from_freq(run_dir: Optional[Path] = None) -> Path:
    """Copy frequency.npz artery/parking into stable ``artery_maps/<map_id>.npz``."""
    run_dir = Path(run_dir) if run_dir else latest_freq_run()
    if run_dir is None or not run_dir.exists():
        raise FileNotFoundError("no freq_probe run to install from")
    out = ensure_artery_maps_dir()
    n = 0
    for freq_path in sorted(run_dir.glob("*/frequency.npz")):
        map_id = freq_path.parent.name
        data = np.load(freq_path)
        np.savez_compressed(
            out / f"{map_id}.npz",
            artery=np.asarray(data["artery"], dtype=np.float32),
            parking=np.asarray(data["parking"], dtype=np.float32),
            walkable=np.asarray(data["walkable"], dtype=np.float32)
            if "walkable" in data.files
            else np.zeros((GRID, GRID), dtype=np.float32),
            freq=np.asarray(data["freq"], dtype=np.float32)
            if "freq" in data.files
            else np.zeros((GRID, GRID), dtype=np.float32),
            source=str(freq_path),
        )
        n += 1
    print(f"[ARTERY-MAP] installed {n} maps from {run_dir} → {out}", flush=True)
    return out


def _normalize_map_id(map_id: str) -> str:
    s = str(map_id or "").strip()
    if not s:
        return ""
    if s.isdigit():
        return f"SH_custom_{int(s):02d}"
    return s


def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if not path.exists():
        return None
    data = np.load(path)
    artery = np.asarray(data["artery"], dtype=np.float32)
    parking = np.asarray(data["parking"], dtype=np.float32)
    walk = (
        np.asarray(data["walkable"], dtype=np.float32)
        if "walkable" in data.files
        else np.zeros_like(artery)
    )
    return {"artery": artery, "parking": parking, "walkable": walk, "path": path}


def resolve_map_id(map_id: str = "", meta: Optional[dict] = None) -> str:
    mid = _normalize_map_id(map_id)
    if mid.startswith("SH_custom_"):
        return mid
    if meta:
        for key in ("base_id", "map_id", "scene_id"):
            cand = _normalize_map_id(str(meta.get(key) or ""))
            if cand.startswith("SH_custom_"):
                return cand
        jpath = Path(str(meta.get("map_json") or ""))
        stem = jpath.stem if jpath.suffix else ""
        cand = _normalize_map_id(stem)
        if cand.startswith("SH_custom_"):
            return cand
        # smoke ids like SH01_traffic_smoke / SH_custom_01_foo
        raw = str(meta.get("id") or "")
        import re

        m = re.search(r"SH_custom_(\d+)", raw)
        if m:
            return f"SH_custom_{int(m.group(1)):02d}"
        m = re.search(r"SH(\d+)", raw)
        if m:
            return f"SH_custom_{int(m.group(1)):02d}"
    return mid


def load_map_labels(
    map_id: str,
    *,
    meta: Optional[dict] = None,
    compute_if_missing: bool = True,
) -> Dict[str, np.ndarray]:
    """Resolve artery/parking for ``map_id``.

    Order: cache → ``artery_maps/`` → latest ``freq_probe`` → cartesian compute.
    """
    mid = resolve_map_id(map_id, meta)
    if not mid and meta:
        mid = resolve_map_id("", meta)
    if mid and mid in _CACHE:
        return _CACHE[mid]

    labels: Optional[Dict[str, np.ndarray]] = None
    if mid:
        labels = _load_npz(ARTERY_MAPS_DIR / f"{mid}.npz")
        if labels is None:
            run = latest_freq_run()
            if run is not None:
                labels = _load_npz(run / mid / "frequency.npz")

    if labels is None and compute_if_missing and meta is not None:
        labels = compute_labels_from_meta(meta)
        mid = mid or resolve_map_id(str(meta.get("id") or "unknown"), meta)

    if labels is None:
        raise FileNotFoundError(
            f"artery/parking labels missing for map_id={map_id!r}; "
            "run freq_probe or install_artery_maps_from_freq()"
        )

    if mid:
        _CACHE[mid] = labels
    return labels


def compute_labels_from_meta(meta: dict) -> Dict[str, np.ndarray]:
    from .freq_probe import (
        cartesian_delivery_frequency,
        labels_from_frequency,
        stations_dests_from_meta,
        walkable_and_obstacles,
    )

    pickups, dropoffs = stations_dests_from_meta(meta)
    walk, _ = walkable_and_obstacles(meta)
    freq = cartesian_delivery_frequency(meta, pickups, dropoffs)
    lab = labels_from_frequency(freq, walk, pickups=pickups, dropoffs=dropoffs)
    out = {
        "artery": lab["artery"],
        "parking": lab["parking"],
        "walkable": walk.astype(np.float32),
    }
    if "station" in lab:
        out["station"] = lab["station"]
    return out


def map_id_from_sim(sim) -> str:
    for key in ("_map_id", "map_id"):
        v = getattr(sim, key, None)
        if v:
            return _normalize_map_id(str(v))
    meta = getattr(sim, "_scene_meta", None) or getattr(sim, "meta", None)
    if isinstance(meta, dict) and meta.get("id"):
        return _normalize_map_id(str(meta["id"]))
    return ""


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Install freq artery/parking → artery_maps/")
    ap.add_argument(
        "--run",
        type=str,
        default="",
        help="freq_probe run dir (default: latest)",
    )
    args = ap.parse_args()
    run = Path(args.run) if args.run else None
    install_artery_maps_from_freq(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
