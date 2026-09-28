"""Publication figures for compare_100 five-way bake-off.

Outputs (PNG + PDF) under:
  ml_research/results/coord_custom_ai/compare_100/figures/

  fig1_aggregate_bars.{png,pdf}   — full100 / VALID / avg sim / avg travel
  fig2_completion_heatmap.{png,pdf}
  fig3_hier_handoff.{png,pdf}     — M0-only vs runtime handoff→ECBS
  fig4_pareto_sim_travel.{png,pdf}— ECBS vs Hier on full-100 maps
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "coord_custom_ai" / "compare_100" / "COMPARE_FULL.csv"
JSONL = ROOT / "results" / "coord_custom_ai" / "compare_100" / "results.jsonl"
OUT = ROOT / "results" / "coord_custom_ai" / "compare_100" / "figures"

METHODS = ["PIBT", "M0", "PP", "ECBS", "Hier"]
# Colorblind-friendly fixed palette (Okabe–Ito)
COLORS = {
    "PIBT": "#E69F00",
    "M0": "#56B4E9",
    "PP": "#009E73",
    "ECBS": "#CC79A7",
    "Hier": "#0072B2",
}


def _load() -> dict[int, dict[str, dict]]:
    by: dict[int, dict[str, dict]] = defaultdict(dict)
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by[int(r["slot"])][r["method"]] = {
                "done": int(r["done"]),
                "total": int(r["total"]),
                "valid": r["valid"] == "True",
                "sim": int(r["sim_time"]),
                "wall": float(r["wall_s"]),
                "travel": int(float(r["travel"])) if r.get("travel") else 0,
            }
    return by


def _hier_kinds() -> dict[int, str]:
    kinds: dict[int, str] = {}
    for line in JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("method_id") != "hier":
            continue
        name = Path(str(r.get("trajectory") or "")).name
        slot = int(r["slot"])
        if "m0_engine" in name:
            kinds[slot] = "m0_only"
        elif "ecbs_" in name:
            kinds[slot] = "handoff"
        else:
            kinds[slot] = "other"
    return kinds


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p)
        print(f"wrote {p}")


def fig1_aggregate(by: dict[int, dict[str, dict]]) -> None:
    n = 20
    full, valid, avg_sim, avg_travel = [], [], [], []
    for m in METHODS:
        xs = [by[s][m] for s in range(1, 21)]
        full.append(100.0 * sum(1 for x in xs if x["done"] >= x["total"]) / n)
        valid.append(100.0 * sum(1 for x in xs if x["valid"]) / n)
        avg_sim.append(sum(x["sim"] for x in xs) / n)
        avg_travel.append(sum(x["travel"] for x in xs) / n)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    x = np.arange(len(METHODS))
    cols = [COLORS[m] for m in METHODS]

    panels = [
        (axes[0, 0], full, "Full completion (% maps)", "% maps with 100/100"),
        (axes[0, 1], valid, "VALID rate (% maps)", "% maps passing legality"),
        (axes[1, 0], avg_sim, "Avg. sim time (steps)", "mean makespan over 20 maps"),
        (axes[1, 1], avg_travel, "Avg. travel (cell-steps)", "mean fleet Manhattan travel"),
    ]
    for ax, vals, title, ylab in panels:
        bars = ax.bar(x, vals, color=cols, width=0.72, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
        for b, v in zip(bars, vals):
            label = f"{v:.0f}" if v >= 10 else f"{v:.1f}"
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                label,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(
        "SH01–20 × 100 tasks: method aggregates",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig1_aggregate_bars")
    plt.close(fig)


def fig2_heatmap(by: dict[int, dict[str, dict]]) -> None:
    mat = np.zeros((20, 5))
    valid_mask = np.zeros((20, 5), dtype=bool)
    for i, slot in enumerate(range(1, 21)):
        for j, m in enumerate(METHODS):
            r = by[slot][m]
            mat[i, j] = 100.0 * r["done"] / max(1, r["total"])
            valid_mask[i, j] = r["valid"]

    # White → steel blue (completion)
    cmap = LinearSegmentedColormap.from_list(
        "comp", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
    )
    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(5))
    ax.set_xticklabels(METHODS)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"SH{s:02d}" for s in range(1, 21)])
    ax.set_xlabel("Method")
    ax.set_ylabel("Map")
    ax.set_title("Per-map task completion (%)")

    for i in range(20):
        for j in range(5):
            v = mat[i, j]
            txt = f"{v:.0f}"
            if not valid_mask[i, j]:
                txt += "×"
            color = "white" if v >= 55 else "#222222"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Completion (%)")
    ax.text(
        0.0,
        -0.06,
        "× = failed VALID (legality)",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout()
    _save(fig, "fig2_completion_heatmap")
    plt.close(fig)


def fig3_hier_handoff(by: dict[int, dict[str, dict]], kinds: dict[int, str]) -> None:
    slots = list(range(1, 21))
    kind_list = [kinds.get(s, "other") for s in slots]
    sims = [by[s]["Hier"]["sim"] for s in slots]
    colors = [
        COLORS["Hier"] if k == "m0_only" else COLORS["ECBS"] for k in kind_list
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1.0, 2.2]}
    )

    # (a) counts
    ax = axes[0]
    n_m0 = sum(1 for k in kind_list if k == "m0_only")
    n_ho = sum(1 for k in kind_list if k == "handoff")
    bars = ax.bar(
        ["M0 only", "Handoff\n→ ECBS"],
        [n_m0, n_ho],
        color=[COLORS["Hier"], COLORS["ECBS"]],
        width=0.55,
        edgecolor="white",
    )
    ax.set_ylabel("# maps")
    ax.set_title("(a) Hier planner path")
    ax.set_ylim(0, 20)
    for b, v in zip(bars, [n_m0, n_ho]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.3,
            str(v),
            ha="center",
            fontsize=9,
        )

    # (b) per-map sim with path color
    ax = axes[1]
    x = np.arange(20)
    ax.bar(x, sims, color=colors, width=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:02d}" for s in slots], fontsize=6)
    ax.set_xlabel("Map (SH)")
    ax.set_ylabel("Hier sim time (steps)")
    ax.set_title("(b) Hier sim time by path")
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["Hier"], label="Finished on M0"),
            Patch(facecolor=COLORS["ECBS"], label="Runtime handoff → ECBS"),
        ],
        loc="upper right",
        frameon=False,
    )

    fig.suptitle(
        "Hierarchical controller: automatic M0 ↔ ECBS switching",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig3_hier_handoff")
    plt.close(fig)


def fig4_pareto(by: dict[int, dict[str, dict]]) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    for m, marker in (("ECBS", "o"), ("Hier", "s")):
        xs, ys, labels = [], [], []
        for s in range(1, 21):
            r = by[s][m]
            if r["done"] < r["total"]:
                continue
            xs.append(r["sim"])
            ys.append(r["travel"])
            labels.append(s)
        ax.scatter(
            xs,
            ys,
            c=COLORS[m],
            marker=marker,
            s=42,
            label=m,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        for x, y, s in zip(xs, ys, labels):
            ax.annotate(
                f"{s:02d}",
                (x, y),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=5.5,
                color="#444444",
            )

    ax.set_xlabel("Sim time (steps)")
    ax.set_ylabel("Travel (cell-steps)")
    ax.set_title("Full-completion maps: sim vs travel")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    _save(fig, "fig4_pareto_sim_travel")
    plt.close(fig)


def write_readme(kinds: dict[int, str]) -> None:
    m0 = [f"SH{s:02d}" for s, k in sorted(kinds.items()) if k == "m0_only"]
    ho = [f"SH{s:02d}" for s, k in sorted(kinds.items()) if k == "handoff"]
    text = f"""# Paper figures (compare_100)

Generated from `COMPARE_FULL.csv` + Hier trajectory paths.

| File | Content |
|------|---------|
| `fig1_aggregate_bars` | Full100 / VALID / avg sim / avg travel (4 panels) |
| `fig2_completion_heatmap` | Per-map completion %; `×` = INVALID |
| `fig3_hier_handoff` | Automatic M0-only vs runtime handoff→ECBS |
| `fig4_pareto_sim_travel` | ECBS vs Hier scatter (full-completion maps) |

## Hier path labels (from traj filename)
- **M0 only** ({len(m0)}): {", ".join(m0)}
- **Handoff → ECBS** ({len(ho)}): {", ".join(ho)}

Regen:
```bash
python -m ml_research.scripts.plot_compare_100_paper_figs
```
"""
    p = OUT / "README.md"
    p.write_text(text, encoding="utf-8")
    print(f"wrote {p}")


def main() -> None:
    _style()
    by = _load()
    kinds = _hier_kinds()
    fig1_aggregate(by)
    fig2_heatmap(by)
    fig3_hier_handoff(by, kinds)
    fig4_pareto(by)
    write_readme(kinds)


if __name__ == "__main__":
    main()
