"""Render artery / parking overlay PNG from frequency.npz (2-class only)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def render_artery_parking_png(
    npz_path: Path,
    out_png: Path,
    *,
    title: str = "",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    data = np.load(npz_path, allow_pickle=True)
    walk = np.asarray(data["walkable"], dtype=np.float32)
    artery = np.asarray(data["artery"], dtype=np.float32)
    parking = np.asarray(data["parking"], dtype=np.float32)
    freq = np.asarray(data["freq"], dtype=np.float32)
    H, W = walk.shape

    station = (
        np.asarray(data["station"], dtype=np.float32)
        if "station" in data.files
        else np.zeros_like(walk)
    )

    # 0 void, 1 obstacle, 2 artery, 3 parking, 4 pickup station
    canvas = np.zeros((H, W), dtype=np.int32)
    canvas[walk < 0.5] = 1
    canvas[(walk >= 0.5) & (artery >= 0.5)] = 2
    canvas[(walk >= 0.5) & (parking >= 0.5)] = 3
    canvas[(walk >= 0.5) & (station >= 0.5)] = 4

    colors = ["#111111", "#333333", "#ffb300", "#29b6f6", "#66bb6a"]
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), dpi=140)
    ax0, ax1 = axes

    ax0.imshow(canvas, origin="lower", cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
    ax0.set_title(title or "artery / parking")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.set_xticks(range(0, W, 2))
    ax0.set_yticks(range(0, H, 2))
    ax0.legend(
        handles=[
            Patch(facecolor=colors[1], label="obstacle"),
            Patch(facecolor=colors[2], label="artery"),
            Patch(facecolor=colors[3], label="parking"),
            Patch(facecolor=colors[4], label="pickup"),
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    # Heatmap: obstacle cells drawn black (same as left legend), not magma-nan
    obs_color = colors[1]
    fn = np.log1p(np.maximum(freq.astype(np.float64), 0.0))
    fn_masked = np.ma.array(fn, mask=(walk < 0.5))
    heat_cmap = plt.cm.magma.copy()
    heat_cmap.set_bad(obs_color)
    im = ax1.imshow(fn_masked, origin="lower", cmap=heat_cmap, interpolation="nearest")
    ax1.set_title("visit frequency (log1p)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xticks(range(0, W, 2))
    ax1.set_yticks(range(0, H, 2))
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[FREQ-PNG] wrote {out_png}", flush=True)
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--title", type=str, default="")
    ap.add_argument(
        "--relabel",
        action="store_true",
        help="Recompute artery/parking with 2-class rule before render",
    )
    args = ap.parse_args()
    npz = Path(args.npz)
    if args.relabel:
        from .freq_probe import labels_from_frequency, stations_dests_from_meta
        from ml_research.benchmarks.coord_custom_ai.scenes import load_custom_meta

        data = dict(np.load(npz, allow_pickle=True))
        # map id from parent folder e.g. SH_custom_03
        pickups, dropoffs = [], []
        name = npz.parent.name
        if name.startswith("SH_custom_"):
            try:
                slot = int(name.split("_")[-1])
                meta = load_custom_meta(slot)
                if meta:
                    pickups, dropoffs = stations_dests_from_meta(meta)
            except Exception:  # noqa: BLE001
                pass
        lab = labels_from_frequency(
            data["freq"], data["walkable"], pickups=pickups, dropoffs=dropoffs
        )
        data["artery"] = lab["artery"]
        data["parking"] = lab["parking"]
        data["freq_norm"] = lab["freq_norm"]
        np.savez_compressed(npz, **data)
        print(f"[FREQ-PNG] relabeled {npz}", flush=True)
    out = Path(args.out) if args.out else npz.with_name("artery_parking.png")
    render_artery_parking_png(npz, out, title=args.title or npz.parent.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
