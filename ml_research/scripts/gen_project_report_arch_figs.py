#!/usr/bin/env python3
"""Publication-quality architecture diagrams for the UM MAPD project report.

IEEE/ACM-style block diagrams with text that fits inside boxes (wrapped titles,
auto-shrunk fonts, generous padding).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "project_report_figs"

C = {
    "ink": "#1a1a1a",
    "muted": "#4a4a4a",
    "line": "#2c2c2c",
    "frame": "#6b7280",
    "bg": "#ffffff",
    "input": "#f3f4f6",
    "m0": "#dbeafe",
    "gate": "#fef3c7",
    "ecbs": "#fce7f3",
    "valid": "#e0e7ff",
    "metric": "#ede9fe",
    "ai": "#d1fae5",
    "reserve": "#fef9c3",
    "exec": "#ffedd5",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "figure.facecolor": C["bg"],
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": C["bg"],
            "savefig.pad_inches": 0.1,
        }
    )


def _fit_fontsize(text: str, max_width_in: float, max_size: float, min_size: float = 6.0) -> float:
    """Rough char-width fit so title/body stays inside the box."""
    lines = text.split("\n")
    longest = max((len(ln) for ln in lines), default=1)
    # ~0.55 em average glyph width for bold DejaVu at size s ≈ s*0.01 inches * 0.55*100...
    # In data coords we approximate: chars * 0.065 * size_pt / 10 ≈ width needed in inches-of-axes.
    # Caller passes max_width in axes data units; assume ~0.11 data-units per char at 9pt.
    size = max_size
    while size > min_size:
        need = longest * 0.095 * (size / 9.0)
        if need <= max_width_in:
            return size
        size -= 0.4
    return min_size


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str] | None = None,
    *,
    facecolor: str = C["input"],
    title_size: float = 9.0,
    body_size: float = 7.4,
    lw: float = 1.15,
) -> tuple[float, float, float, float]:
    pad = 0.10  # inner horizontal padding in data units
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw,
        edgecolor=C["line"],
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(patch)
    cx = x + w / 2
    inner_w = max(w - 2 * pad, 0.5)

    title_fs = _fit_fontsize(title, inner_w, title_size, 6.5)
    n_title = title.count("\n") + 1

    if lines:
        body = "\n".join(lines)
        body_fs = _fit_fontsize(body, inner_w, body_size, 6.0)
        # Title band near top
        title_y = y + h - 0.18 * h - 0.06 * (n_title - 1)
        ax.text(
            cx,
            title_y,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
            color=C["ink"],
            zorder=4,
            linespacing=1.15,
            clip_on=True,
        )
        ax.text(
            cx,
            y + 0.36 * h,
            body,
            ha="center",
            va="center",
            fontsize=body_fs,
            color=C["muted"],
            linespacing=1.28,
            zorder=4,
            clip_on=True,
        )
    else:
        ax.text(
            cx,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_fs,
            fontweight="bold",
            color=C["ink"],
            zorder=4,
            linespacing=1.2,
            clip_on=True,
        )
    return cx, y + h / 2, x, y


def _arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    label: str = "",
    color: str = C["line"],
    lw: float = 1.2,
    ls: str = "-",
    label_offset: tuple[float, float] = (0.0, 0.12),
    label_size: float = 7.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color,
            linestyle=ls,
            shrinkA=1,
            shrinkB=1,
            zorder=2,
        )
    )
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=label_size,
            color=C["muted"],
            fontstyle="italic",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#d1d5db", lw=0.4, alpha=0.95),
        )


def _layer_frame(ax, x, y, w, h, label: str) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            edgecolor=C["frame"],
            zorder=1,
        )
    )
    ax.text(
        x + 0.12,
        y + h - 0.14,
        label,
        ha="left",
        va="top",
        fontsize=7.4,
        color=C["frame"],
        fontweight="bold",
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95),
    )


def fig_arch_stack() -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # Leave a clear mid-band for cross-layer arrow labels
    _layer_frame(ax, 0.2, 3.55, 10.2, 2.3, "Planning / control path")
    _layer_frame(ax, 0.2, 0.25, 10.2, 2.45, "Assurance / evaluation")

    bw, bh, y0 = 2.2, 1.6, 3.9
    gap = 0.3
    xs = [0.4 + i * (bw + gap) for i in range(4)]

    _box(ax, xs[0], y0, bw, bh, "Task input", ["official tasks", "FIFO queues", "stations / hubs"], facecolor=C["input"])
    _box(ax, xs[1], y0, bw, bh, "M0 engine", ["spatiotemporal A*", "reservation table", "+ SwapNet"], facecolor=C["m0"])
    _box(
        ax,
        xs[2],
        y0,
        bw,
        bh,
        "Lifelong gate",
        ["stall / thrash", "structure signals", "StallEscalateNet"],
        facecolor=C["gate"],
        body_size=6.5,
    )
    _box(
        ax,
        xs[3],
        y0,
        bw,
        bh,
        "Windowed\nECBS",
        ["claim budget K", "conflict tree", "cargo / FIFO"],
        facecolor=C["ecbs"],
        title_size=8.6,
    )

    _box(
        ax,
        1.7,
        0.45,
        3.1,
        1.75,
        "Hybrid validator",
        ["collision / swap", "FIFO / dwell", "premature unload", "ghost cargo"],
        facecolor=C["valid"],
    )
    _box(
        ax,
        5.7,
        0.45,
        3.1,
        1.75,
        "Metrics",
        ["VALID / done", "sim / wall", "travel"],
        facecolor=C["metric"],
    )

    mid_y = y0 + bh / 2
    for i in range(3):
        _arrow(ax, xs[i] + bw, mid_y, xs[i + 1], mid_y, label=["", "default", "escalate"][i], label_offset=(0, 0.28))

    # Labels sit in the gap between the two layer frames
    _arrow(ax, xs[1] + bw / 2, y0, 3.25, 2.7, label="trajectories", label_offset=(0.65, 0.0))
    _arrow(ax, xs[2] + bw / 2, y0, 7.25, 2.7, label="handoff logs", label_offset=(-0.45, 0.0))

    # Soft return arc ABOVE the boxes (never inside box bottoms)
    ax.annotate(
        "",
        xy=(xs[1] + bw / 2, y0 + bh),
        xytext=(xs[3] + bw / 2, y0 + bh),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C["frame"],
            lw=1.0,
            linestyle=(0, (3, 2)),
            connectionstyle="arc3,rad=-0.42",
        ),
        zorder=2,
    )
    ax.text(
        (xs[1] + xs[3] + bw) / 2,
        y0 + bh + 0.55,
        "resume M0 when clear",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=C["frame"],
        style="italic",
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#d1d5db", lw=0.35, alpha=0.98),
        zorder=6,
    )

    ax.set_title("Hierarchical Lifelong MAPD Stack", fontsize=11, fontweight="bold", color=C["ink"], pad=8)
    path = OUT / "fig_arch_stack.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def fig_m0_loop() -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    _box(ax, 0.35, 3.25, 2.35, 1.5, "Assign", ["free AGV →", "station FIFO head"], facecolor=C["ai"])
    _box(
        ax,
        3.35,
        3.25,
        2.7,
        1.5,
        "Spatiotemporal A*",
        ["vertex + edge forbid", "orientation / turn", "moving obstacles"],
        facecolor=C["m0"],
        title_size=8.4,
    )
    _box(
        ax,
        6.7,
        3.25,
        2.85,
        1.5,
        "Execute horizon",
        ["hold / dwell 1 s", "pickup / unload", "action shields"],
        facecolor=C["exec"],
    )
    _box(
        ax,
        2.2,
        0.5,
        5.6,
        1.65,
        "Shared world model",
        ["reservation table", "timed obstacles from peer AGVs", "optional SwapNet station swap"],
        facecolor=C["reserve"],
    )

    _arrow(ax, 2.7, 4.0, 3.35, 4.0)
    _arrow(ax, 6.05, 4.0, 6.7, 4.0, label="commit", label_offset=(0, 0.22))
    _arrow(ax, 4.7, 3.25, 4.7, 2.15, label="write", label_offset=(0.35, 0))
    _arrow(ax, 4.2, 2.15, 4.4, 3.25, label="query", label_offset=(-0.45, 0.05), ls="--", lw=1.0)

    ax.annotate(
        "",
        xy=(1.5, 3.25),
        xytext=(8.1, 3.25),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C["line"],
            lw=1.15,
            connectionstyle="arc3,rad=0.5",
            linestyle="--",
        ),
        zorder=2,
    )
    ax.text(5.0, 5.15, "lifelong tick", ha="center", fontsize=7.2, color=C["muted"], style="italic")

    ax.set_title("M0 Spatiotemporal A* Planning Loop", fontsize=11, fontweight="bold", pad=8)
    path = OUT / "fig_m0_loop.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def fig_ecbs_wave() -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 6.1)
    ax.axis("off")

    _layer_frame(ax, 0.25, 3.5, 10.1, 2.3, "Per-wave ECBS")
    _layer_frame(ax, 0.25, 0.25, 10.1, 2.4, "Lifelong windowing & recovery")

    bw, bh, y0 = 2.2, 1.65, 3.8
    gap = 0.3
    xs = [0.4 + i * (bw + gap) for i in range(4)]

    _box(ax, xs[0], y0, bw, bh, "Claim wave k", ["dest-pad budget", "active agent set", "stationary seeds"], facecolor=C["ai"])
    _box(ax, xs[1], y0, bw, bh, "High-level CT", ["focal search (w≥1)", "conflict selection", "constraint split"], facecolor=C["m0"])
    _box(ax, xs[2], y0, bw, bh, "Low-level A*", ["under constraints", "turn-aware"], facecolor=C["gate"])
    _box(ax, xs[3], y0, bw, bh, "Apply", ["FIFO / cargo", "legality guards"], facecolor=C["ecbs"])

    _box(
        ax,
        0.6,
        0.4,
        4.2,
        1.75,
        "Window schedule",
        ["pickup → delivery wave", "claim budget K", "non-claimed as obstacles"],
        facecolor=C["valid"],
    )
    _box(
        ax,
        5.5,
        0.4,
        4.4,
        1.75,
        "Failure recovery",
        ["shrink active set", "ST-floor / corridor spin", "re-claim next window"],
        facecolor=C["metric"],
    )

    mid = y0 + bh / 2
    for i in range(3):
        _arrow(ax, xs[i] + bw, mid, xs[i + 1], mid)

    _arrow(ax, xs[1] + bw / 2, y0, 2.7, 2.65, label="success / fail", label_offset=(0.55, 0))
    _arrow(ax, xs[3] + bw / 2, y0, 7.7, 2.65, label="guards", label_offset=(-0.15, 0))
    _arrow(ax, 2.7, 2.65, xs[0] + bw / 2, y0, label="next k", label_offset=(-0.55, 0))

    ax.set_title("Windowed ECBS Lifelong Wave Pipeline", fontsize=11, fontweight="bold", pad=8)
    path = OUT / "fig_ecbs_wave.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def fig_ai_modules() -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    _layer_frame(ax, 0.25, 3.5, 10.1, 2.4, "Online AI aides (narrow scope)")
    _layer_frame(ax, 0.25, 0.25, 10.1, 2.45, "Rule gate + legality (always on)")

    bw, bh, y0 = 3.0, 1.8, 3.8
    gap = 0.3
    xs = [0.5 + i * (bw + gap) for i in range(3)]

    _box(
        ax,
        xs[0],
        y0,
        bw,
        bh,
        "SwapNet (MLP)",
        ["16-D features", "station task swap", "FIFO / SLA shielded", "under M0 only"],
        facecolor=C["ai"],
    )
    _box(
        ax,
        xs[1],
        y0,
        bw,
        bh,
        "StallEscalate\nNet",
        ["14-D stall features", "wall-thrash handoff", "high precision", "M0 → ECBS"],
        facecolor=C["gate"],
        title_size=8.4,
    )
    _box(
        ax,
        xs[2],
        y0,
        bw,
        bh,
        "Offline (optional)",
        ["WaveNet / difficulty", "not on Hier hot path", "diagnostics only"],
        facecolor=C["input"],
    )

    _box(
        ax,
        0.9,
        0.45,
        8.8,
        1.85,
        "LifelongGateTracker\n+ hybrid validator",
        [
            "rule stall · handoff minima · snapshot alignment",
            "VALID: collision, swap, FIFO, dwell, cargo",
        ],
        facecolor=C["valid"],
        title_size=8.6,
        body_size=7.2,
    )

    # Labels in the mid-band between layers
    _arrow(ax, xs[0] + bw / 2, y0, 3.4, 2.7, label="swap proposals", label_offset=(0.7, 0))
    _arrow(ax, xs[1] + bw / 2, y0, 5.3, 2.7, label="escalate?", label_offset=(0.2, 0))
    _arrow(ax, xs[2] + bw / 2, y0, 7.6, 2.7, label="offline", ls=":", lw=1.0, label_offset=(0.4, 0))

    ax.set_title("AI Modules in the Hierarchical Stack", fontsize=11, fontweight="bold", pad=8)
    path = OUT / "fig_ai_modules.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (fig_arch_stack(), fig_m0_loop(), fig_ecbs_wave(), fig_ai_modules()):
        print(f"wrote {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
