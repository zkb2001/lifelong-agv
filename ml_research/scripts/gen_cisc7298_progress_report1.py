"""Fill CISC7298 Progress Report 1 using the official template layout.

The school template uses one 1x1 content table under each section heading.
We keep that form, write section text into those tables, and insert a
results summary table inside Experimental Results. No References section.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, Twips

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = Path(
    r"C:\Users\zkb17\xwechat_files\wxid_g3r567e59c9a22_605c\msg\file\2026-09"
    r"\CISC7298 Progress Report 1(1).docx"
)
OUT_DOCS = ROOT / "docs" / "CISC7298_Progress_Report_1.docx"
OUT_ROOT = ROOT / "CISC7298_Progress_Report_1.docx"

SECTION_CONTENTS = {
    "Title and Abstract:": [
        (
            "Title: Hierarchical Lifelong Multi-Agent Pickup-and-Delivery for "
            "Warehouse AGVs via Spatiotemporal A* with ECBS Escalation"
        ),
        (
            "Abstract. This project studies lifelong multi-agent pickup-and-delivery "
            "(MAPD) for automated guided vehicles (AGVs) under competition-style "
            "warehouse rules: a 20x20 grid, FIFO station queues, in-place turns of "
            "at most 90° per second, and a mandatory one-second dwell for pickup and "
            "unload. We develop a hierarchical planner that keeps a fast engine-level "
            "spatiotemporal A* layer (M0) as the default and escalates to windowed "
            "Enhanced Conflict-Based Search (ECBS) only when runtime stall signals "
            "indicate that local search is insufficient. Competition legality is "
            "enforced end-to-end (no same-tick move-and-turn; door-side pickup; cargo "
            "continuity; validated trajectories). On a curriculum of twenty custom "
            "maps with official 100-task instances, a completed five-way bake-off "
            "shows that hierarchical M0<->ECBS and pure windowed ECBS both achieve "
            "20/20 full completions with VALID trajectories, while reactive PIBT, "
            "pure M0, and priority planning lose throughput or validity on congested "
            "maps. Hierarchy further reduces average simulation makespan relative to "
            "always-on ECBS (about 4616 vs 7364 ticks), confirming that gated "
            "escalation is an effective accuracy-efficiency trade-off."
        ),
    ],
    "Introduction (with OR w/o Related Work/Literature Review):": [
        (
            "Warehouse AGV fleets must continuously assign pickups, route around "
            "static obstacles and other robots, and respect station FIFO order. "
            "Classical one-shot multi-agent path finding (MAPF) solvers do not "
            "directly cover lifelong task release; pure reactive methods (e.g., PIBT) "
            "scale well but often deadlock in narrow corridors; full joint search "
            "(CBS/ECBS) improves conflict resolution but can inflate makespan and "
            "wall-clock time if used for every planning wave."
        ),
        (
            "This work targets the MioVerse / Siemens Xcelerator AGV track setting "
            "together with the course project requirements: maximize completed "
            "deliveries under strict motion and FIFO validators. The research "
            "question is whether a hierarchical policy-run cheap spatiotemporal A* "
            "by default and invoke ECBS only on throughput stalls-can match "
            "always-ECBS completion while retaining M0 efficiency on easy maps."
        ),
        (
            "Related work. MAPF has a mature literature on Conflict-Based Search "
            "(CBS) and weighted ECBS for bounded-suboptimal joint plans. Priority "
            "planning (PP) and windowed lifelong planners (WHCA*/RHCR-style) address "
            "continuous operation with limited horizons. PIBT provides strongly "
            "scalable one-step conflict resolution without reservations. Our design "
            "sits between these extremes: M0 provides continuous lifelong execution "
            "with reservation tables; an automatic gate monitors stall metrics and "
            "triggers ECBS waves; handoff snapshots preserve cargo and FIFO heads so "
            "escalation does not break competition validators. The gate is not a "
            "per-map hardcode: cold-start hardness scores may bias early behavior, "
            "but escalation decisions are driven by online progress signals."
        ),
    ],
    "Methodology:": [
        (
            "Problem model. Each map provides six pickup stations and sixteen "
            "dropoff hubs on a 20x20 grid. Tasks are released from official CSVs "
            "(100 tasks in the primary comparison protocol; larger 400-task stress "
            "tests are also used). AGVs may carry at most one package; pickup occurs "
            "on the approach door beside a station after a one-second no-turn dwell; "
            "unload occurs on a free four-neighbor of the destination hub with the "
            "same dwell rule."
        ),
        (
            "Layer M0 (default). The simulation engine assigns free AGVs to station "
            "FIFO heads (at most one approacher per station) and plans with "
            "spatiotemporal A* under a rolling moving-obstacle horizon, forbidding "
            "vertex collisions and edge swaps. This layer is fast on open maps but "
            "may stall when corridors jam."
        ),
        (
            "Hierarchical gate and ECBS escalation. A LifelongGateTracker starts "
            "with M0 and monitors runtime signals (done plateau, assign failures, "
            "wall-efficiency). When M0 stalls after a recovery window, the system "
            "freezes a handoff snapshot and continues with windowed joint ECBS "
            "planning. Escalation never reorders surface FIFO queues. Map-hardness / "
            "narrow-cut features are used only as cold-start context; they do not "
            "permanently force ECBS on selected map IDs."
        ),
        (
            "Windowed ECBS legality. Pure windowed ECBS is also evaluated as a strong "
            "joint baseline. Engineering fixes required for VALID trajectories "
            "include: (i) refuse off-pad falling edges that wipe cargo annotations; "
            "(ii) heal “ghost cargo” when the trajectory still shows loaded=true but "
            "the planner bag lost the task; (iii) refuse pickup while already "
            "carrying another task and force unload first. These guards eliminate "
            "premature_unload and surface-FIFO violations observed on dense maps "
            "(e.g., SH11/SH12)."
        ),
        (
            "Baselines. We implemented a five-way compare harness under identical "
            "maps, task CSVs, motion rules, and a shared hybrid validator: "
            "(1) reactive PIBT; (2) M0-only spatiotemporal A*; (3) priority-planning "
            "allocator + A*; (4) windowed ECBS without M0 hierarchy; "
            "(5) hierarchical M0<->ECBS (proposed)."
        ),
    ],
    "Experimental Results:": [
        (
            "Setup. Maps SH_custom_01-20 from the map-editor curriculum; official "
            "100-task CSVs; full AGV fleets. Batch protocol: wall timeout 1800 s for "
            "constrained runs, with unlimited-wall reruns for hierarchical / ECBS "
            "instances that need more planning time; max simulation horizon up to "
            "500000 ticks when measuring completion. Metrics: tasks completed / 100, "
            "VALID flag from the hybrid validator, simulation makespan, wall-clock "
            "seconds, and total Manhattan travel."
        ),
        (
            "Aggregate five-way results (SH01-20 x 100 tasks) are summarized in the "
            "table below. Both ECBS and Hier finish all 100 tasks on every map with "
            "VALID trajectories. Hierarchy cuts average makespan by about 37% versus "
            "always-on ECBS (4616 vs 7364), at a moderate increase in average "
            "wall-clock (704 s vs 543 s). PIBT is extremely fast but averages only "
            "63.9 completions and fully finishes 8/20 maps. Pure M0 stays VALID but "
            "finishes only 4/20 maps. Priority planning is weakest on both completion "
            "and validity (7/20 VALID)."
        ),
        # marker for results table insertion
        "__RESULTS_TABLE__",
        (
            "Representative map behavior. On easy SH01, M0 and Hier finish with low "
            "makespan (sim ≈ 546-612) while PIBT/ECBS need ≈ 2200 ticks; PP finishes "
            "100 tasks but fails FIFO validation. On hutong-like SH03, PIBT/M0/PP "
            "collapse to ≤5 completions, whereas ECBS and Hier both reach 100/100 "
            "VALID. On SH04, Hier closely tracks M0 efficiency (sim 670 vs 684) "
            "instead of paying ECBS’s longer plan (4219). On dense SH11/SH12, after "
            "legality fixes, ECBS recovers to 100/100 VALID (sim 18021 / 16981), "
            "while Hier achieves the same completion with shorter makespan "
            "(4775 / 5220)."
        ),
        (
            "Interpretation. Reactive and local search lose throughput under "
            "congestion; always-ECBS is strong but expensive in simulation time; "
            "hierarchical M0-first with stall-triggered ECBS matches ECBS’s "
            "completion/validity envelope while reclaiming M0-like makespan on many "
            "maps. This is the central Progress Report 1 empirical claim."
        ),
    ],
    "Conclusion:": [
        (
            "Progress to date delivers a competition-legal hierarchical lifelong "
            "MAPD stack, a legality-hardened windowed ECBS baseline, and a completed "
            "SH01-20 x 100 five-method comparison. Evidence shows that gated M0<->ECBS "
            "preserves 20/20 VALID full completions while reducing average makespan "
            "relative to always-on ECBS. Next steps for Progress Report 2 include: "
            "(i) aligned 400-task hierarchical stress results across the same maps; "
            "(ii) quantitative analysis of handoff timing and stall features versus "
            "map structure; (iii) failure-case studies on residual high-wall maps; "
            "and (iv) optional learning-based gates only if they improve escalation "
            "decisions without harming VALID rates."
        ),
    ],
}

RESULTS_HEADERS = [
    "Method",
    "maps",
    "full 100",
    "VALID",
    "avg done",
    "avg sim",
    "avg wall (s)",
]
RESULTS_ROWS = [
    ["PIBT", "20", "8/20", "20/20", "63.9", "62811", "40.5"],
    ["M0", "20", "4/20", "20/20", "44.6", "942", "1478.5"],
    ["PP", "20", "1/20", "7/20", "32.4", "848", "1716.1"],
    ["ECBS", "20", "20/20", "20/20", "100.0", "7364", "542.7"],
    ["Hier (ours)", "20", "20/20", "20/20", "100.0", "4616", "703.7"],
]


def set_run_font(run, size=11, bold=False, name="Times New Roman"):
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = name
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.get_or_add_rFonts()
    r_fonts.set(qn("w:ascii"), name)
    r_fonts.set(qn("w:hAnsi"), name)
    r_fonts.set(qn("w:eastAsia"), name)


def _clear_cell(cell) -> None:
    # Reset to a single empty paragraph (Word cells need ≥1 p).
    cell.text = ""
    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def _add_para_in_cell(cell, text: str, *, first=False, bold=False, size=11):
    if first:
        p = cell.paragraphs[0]
    else:
        p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.first_line_indent = Inches(0.25)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return p


def _add_results_table_in_cell(cell) -> None:
    # Nested table inside the section content table.
    table = cell.add_table(rows=1 + len(RESULTS_ROWS), cols=len(RESULTS_HEADERS))
    table.style = "Table Grid"
    for j, h in enumerate(RESULTS_HEADERS):
        c = table.rows[0].cells[j]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        set_run_font(run, size=9, bold=True)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(RESULTS_ROWS):
        for j, val in enumerate(row):
            c = table.rows[i + 1].cells[j]
            c.text = ""
            run = c.paragraphs[0].add_run(str(val))
            set_run_font(run, size=9)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER


def _heading_to_table_index(doc: Document) -> dict[str, int]:
    """Map section heading text -> following table index (template order)."""
    body = doc.element.body
    headings: list[str] = []
    table_i = -1
    mapping: dict[str, int] = {}
    last_heading = None
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            texts = [t.text for t in child.iter(qn("w:t")) if t.text]
            text = "".join(texts).strip()
            if text.endswith(":") and text in SECTION_CONTENTS:
                last_heading = text
        elif tag == "tbl":
            table_i += 1
            if last_heading is not None and last_heading not in mapping:
                mapping[last_heading] = table_i
                last_heading = None
    return mapping


def fill_template(template: Path) -> Document:
    doc = Document(str(template))
    mapping = _heading_to_table_index(doc)
    if len(mapping) != 5:
        raise RuntimeError(f"Expected 5 section tables, got {mapping}")

    for heading, texts in SECTION_CONTENTS.items():
        ti = mapping[heading]
        cell = doc.tables[ti].cell(0, 0)
        _clear_cell(cell)
        first = True
        for item in texts:
            if item == "__RESULTS_TABLE__":
                _add_results_table_in_cell(cell)
                # spacer paragraph after nested table
                p = cell.add_paragraph()
                p.paragraph_format.space_after = Pt(6)
                first = False
                continue
            _add_para_in_cell(cell, item, first=first)
            first = False
    return doc


def _save(doc: Document, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        doc.save(str(path))
        return path
    except PermissionError:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        doc.save(str(alt))
        print(f"Permission denied for {path}; wrote {alt}")
        return alt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=str, default=str(DEFAULT_TEMPLATE))
    ap.add_argument(
        "--also",
        action="append",
        default=[],
        help="Extra output path(s); may be repeated",
    )
    args = ap.parse_args()

    template = Path(args.template)
    if not template.exists():
        raise SystemExit(f"Template not found: {template}")

    doc = fill_template(template)
    wrote = _save(doc, OUT_DOCS)
    print(f"Wrote {wrote} ({wrote.stat().st_size} bytes)")
    try:
        shutil.copy2(wrote, OUT_ROOT)
        print(f"Copied {OUT_ROOT}")
    except PermissionError:
        alt = OUT_ROOT.with_name(OUT_ROOT.stem + "_new" + OUT_ROOT.suffix)
        shutil.copy2(wrote, alt)
        print(f"Permission denied for {OUT_ROOT}; copied {alt}")

    also = list(args.also) or [
        str(
            Path(r"C:\Users\zkb17\xwechat_files\wxid_g3r567e59c9a22_605c\msg\file\2026-09")
            / "CISC7298 Progress Report 1.docx"
        ),
        str(
            Path(r"C:\Users\zkb17\xwechat_files\wxid_g3r567e59c9a22_605c\msg\file\2026-09")
            / "CISC7298 Progress Report 1(1).docx"
        ),
    ]
    for dest_s in also:
        dest = Path(dest_s)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wrote, dest)
            print(f"Copied {dest}")
        except PermissionError:
            alt = dest.with_name(dest.stem + "_filled" + dest.suffix)
            shutil.copy2(wrote, alt)
            print(f"Permission denied for {dest}; copied {alt}")


if __name__ == "__main__":
    main()
