"""Render synth hard-map layout PNGs (same palette as SH overview)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ROOT / "greed_coop_mcts" / "map_exports_synth"
OUT_DIR = ROOT / "greed_coop_mcts" / "results" / "synth_hard_png"

# 0 free, 1 obstacle, 2 pickup, 3 dropoff, 4 agv
PALETTE = np.array(
    [
        [242, 237, 228],  # free
        [26, 26, 26],  # obstacle
        [46, 139, 87],  # pickup
        [230, 126, 34],  # dropoff
        [59, 110, 165],  # agv
    ],
    dtype=np.uint8,
)


def _load_agvs(pos_csv: Path) -> list[tuple[int, int]]:
    if not pos_csv.is_file():
        return []
    out: list[tuple[int, int]] = []
    with pos_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("type", "")).lower() != "agv":
                continue
            out.append((int(row["x"]), int(row["y"])))
    return out


def _grid(idx: int) -> tuple[np.ndarray, dict]:
    sid = f"SYN_{int(idx):05d}"
    data = json.loads((EXPORTS / f"{sid}.json").read_text(encoding="utf-8"))
    gmin = int(data.get("grid_min") or 1)
    gmax = int(data.get("grid_max") or 20)
    n = gmax - gmin + 1
    g = np.zeros((n, n), dtype=np.int8)
    for o in data.get("extra_obstacles") or []:
        x, y = int(o[0]), int(o[1])
        if gmin <= x <= gmax and gmin <= y <= gmax:
            g[y - gmin, x - gmin] = 1
    for p in data.get("pickups") or []:
        x, y = int(p["x"]), int(p["y"])
        if gmin <= x <= gmax and gmin <= y <= gmax:
            g[y - gmin, x - gmin] = 2
    for p in data.get("dropoffs") or []:
        x, y = int(p["x"]), int(p["y"])
        if gmin <= x <= gmax and gmin <= y <= gmax:
            g[y - gmin, x - gmin] = 3
    for x, y in _load_agvs(EXPORTS / f"{sid}_position.csv"):
        if gmin <= x <= gmax and gmin <= y <= gmax:
            g[y - gmin, x - gmin] = 4
    info = {
        "id": sid,
        "regime": str(data.get("regime") or ""),
        "n_obs": int(len(data.get("extra_obstacles") or [])),
        "n_agvs": int(data.get("n_agvs") or 0),
        "n_pick": int(data.get("n_pickups") or len(data.get("pickups") or [])),
        "n_drop": int(data.get("n_dropoffs") or len(data.get("dropoffs") or [])),
    }
    return g, info


def _to_rgb(g: np.ndarray, *, cell: int = 16, border: int = 1) -> np.ndarray:
    # flipud so y increases upward in the image
    flipped = np.flipud(g)
    h, w = flipped.shape
    rgb = PALETTE[flipped]
    # upscale with nearest neighbor + thin grid lines
    big = np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)
    if border > 0:
        for i in range(1, h):
            big[i * cell : i * cell + border, :] = 200
        for j in range(1, w):
            big[:, j * cell : j * cell + border] = 200
    return big


def _save_png(path: Path, rgb: np.ndarray) -> None:
    try:
        from PIL import Image
    except ImportError:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(path, rgb)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, optimize=True)


def _render_overview(ids: list[int], out: Path, *, cols: int = 5, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    for fp in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
            prop = font_manager.FontProperties(fname=str(fp))
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            break

    cmap = ListedColormap(
        ["#F2EDE4", "#1A1A1A", "#2E8B57", "#E67E22", "#3B6EA5"]
    )
    n = len(ids)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.9, rows * 3.05))
    axes_arr = np.atleast_2d(axes)
    fig.suptitle(
        title or f"Synth HARD maps  ·  {n} maps  ·  20×20",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    for i, ax in enumerate(axes_arr.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        g, info = _grid(ids[i])
        ax.imshow(
            np.flipud(g),
            cmap=cmap,
            vmin=0,
            vmax=4,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{info['id']}  ·  obs {info['n_obs']}  ·  agv {info['n_agvs']}",
            fontsize=8,
            pad=2,
        )
    legend = [
        Patch(facecolor="#1A1A1A", label="障碍"),
        Patch(facecolor="#2E8B57", label="取货台"),
        Patch(facecolor="#E67E22", label="卸货台"),
        Patch(facecolor="#3B6EA5", label="车起点"),
        Patch(facecolor="#F2EDE4", edgecolor="#CCC", label="空地"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.002),
    )
    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.93, bottom=0.05, wspace=0.15, hspace=0.35
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"[overview] {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all hard maps")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--cell", type=int, default=16)
    ap.add_argument("--overview-n", type=int, default=40, help="first N for overview sheet")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--skip-individual", action="store_true")
    args = ap.parse_args()

    index = json.loads((EXPORTS / "index.json").read_text(encoding="utf-8"))
    hard = [int(r["idx"]) for r in index if str(r.get("regime")) == "hard"]
    hard = hard[args.offset :]
    if args.limit > 0:
        hard = hard[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[render] hard={len(hard)} -> {out_dir}", flush=True)

    if not args.skip_individual:
        for i, idx in enumerate(hard, 1):
            g, info = _grid(idx)
            rgb = _to_rgb(g, cell=args.cell)
            path = out_dir / f"{info['id']}.png"
            _save_png(path, rgb)
            if i == 1 or i % 200 == 0 or i == len(hard):
                print(f"  [{i}/{len(hard)}] {path.name}", flush=True)

    ov_n = min(int(args.overview_n), len(hard))
    if ov_n > 0:
        _render_overview(
            hard[:ov_n],
            out_dir / f"overview_hard_first{ov_n}.png",
            title=f"Synth HARD  ·  first {ov_n}  ·  20×20",
        )
    print(f"[render] done n={len(hard)} dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
