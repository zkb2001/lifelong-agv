"""Render SH21–40 map overview (same visual language as SH01-20_overview)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Prefer Windows CJK fonts so titles/legend render correctly
_font_paths = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\msyhbd.ttc"),
]
for _fp in _font_paths:
    if _fp.exists():
        font_manager.fontManager.addfont(str(_fp))
        _prop = font_manager.FontProperties(fname=str(_fp))
        plt.rcParams["font.family"] = _prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False
        break

REPO = Path(__file__).resolve().parents[2]
EXPORTS = REPO / "ml_research" / "map_editor" / "exports"
OUT = REPO / "docs" / "project_report_figs" / "SH21-40_overview.png"
EVAL = Path(__file__).resolve().parent / "GEN_CURRICULUM_EVAL.json"

# 0 free, 1 obstacle, 2 pickup, 3 dropoff, 4 agv
CMAP = ListedColormap(
    ["#F2EDE4", "#1A1A1A", "#2E8B57", "#E67E22", "#3B6EA5"]
)

SLOTS = list(range(21, 41))


def _meta_by_slot() -> dict[int, dict]:
    if not EVAL.exists():
        return {}
    rows = json.loads(EVAL.read_text(encoding="utf-8"))
    return {int(r["slot"]): r for r in rows}


def _load_agvs(pos_csv: Path) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    with pos_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("type", "")).lower() != "agv":
                continue
            out.append((int(row["x"]), int(row["y"])))
    return out


def _grid(slot: int) -> tuple[np.ndarray, dict]:
    sid = f"SH_custom_{slot:02d}"
    jpath = EXPORTS / f"{sid}.json"
    data = json.loads(jpath.read_text(encoding="utf-8"))
    g = np.zeros((20, 20), dtype=np.int8)  # row=y-1 from bottom when flipped
    for o in data.get("extra_obstacles") or []:
        if isinstance(o, dict):
            x, y = int(o["x"]), int(o["y"])
        else:
            x, y = int(o[0]), int(o[1])
        if 1 <= x <= 20 and 1 <= y <= 20:
            g[y - 1, x - 1] = 1
    for p in data.get("pickups") or []:
        x, y = int(p["x"]), int(p["y"])
        if 1 <= x <= 20 and 1 <= y <= 20:
            g[y - 1, x - 1] = 2
    for p in data.get("dropoffs") or []:
        x, y = int(p["x"]), int(p["y"])
        if 1 <= x <= 20 and 1 <= y <= 20:
            g[y - 1, x - 1] = 3
    pos = EXPORTS / f"{sid}_position.csv"
    for x, y in _load_agvs(pos):
        if 1 <= x <= 20 and 1 <= y <= 20:
            g[y - 1, x - 1] = 4
    info = {
        "n_obs": len(data.get("extra_obstacles") or []),
        "n_agvs": int(data.get("n_agvs") or 0),
        "style": str(data.get("map_shape") or data.get("difficulty") or ""),
        "difficulty": str(data.get("difficulty") or ""),
    }
    return g, info


def main() -> None:
    meta = _meta_by_slot()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 5, figsize=(14.5, 12.2), constrained_layout=False)
    fig.suptitle(
        "SH21–40 Generated Curriculum Maps  ·  20×20  ·  y 向上  ·  全局连通",
        fontsize=14,
        fontweight="bold",
        y=0.965,
    )

    for ax, slot in zip(axes.ravel(), SLOTS):
        g, info = _grid(slot)
        m = meta.get(slot) or {}
        style = m.get("style") or info["style"]
        diff = m.get("difficulty") or info["difficulty"]
        n_obs = int(m.get("n_obs") or info["n_obs"])
        n_agvs = int(m.get("n") or info["n_agvs"])
        # flipud so y=1 is bottom (y 向上)
        ax.imshow(
            np.flipud(g),
            cmap=CMAP,
            vmin=0,
            vmax=4,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#888888")
        tag = f"{diff}/{style}" if diff else style
        ax.set_title(
            f"SH{slot:02d}  ·  {tag}\n障碍 {n_obs}  ·  车 {n_agvs}",
            fontsize=8.5,
            pad=3,
        )

    legend = [
        Patch(facecolor="#1A1A1A", edgecolor="none", label="障碍"),
        Patch(facecolor="#2E8B57", edgecolor="none", label="取货台"),
        Patch(facecolor="#E67E22", edgecolor="none", label="卸货台"),
        Patch(facecolor="#3B6EA5", edgecolor="none", label="车起点"),
        Patch(facecolor="#F2EDE4", edgecolor="#CCCCCC", label="空地"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.06, wspace=0.18, hspace=0.35)
    fig.savefig(OUT, dpi=160, facecolor="white")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
