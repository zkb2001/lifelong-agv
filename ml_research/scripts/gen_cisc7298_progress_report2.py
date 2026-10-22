"""Fill CISC7298 Progress Report 2 using the official template layout.

Addresses Progress Report 1 feedback: add a table of implementation
parameters / hyperparameters. Reports Meeting-RH results vs Report-1 baselines.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
# Prefer the project copy that already has Name / Student ID filled in.
DEFAULT_TEMPLATE = ROOT / "CISC7298_Progress_Report_2.docx"
FALLBACK_TEMPLATE = Path(r"d:\download center\CISC7298 Progress Report 2.docx")
OUT_DOCS = ROOT / "docs" / "CISC7298_Progress_Report_2.docx"
OUT_ROOT = ROOT / "CISC7298_Progress_Report_2.docx"
PRESERVE_NAME = "Zeng Kaibo"
PRESERVE_ID = "MC566823"

SECTION_CONTENTS = {
    "Title and Abstract:": [
        (
            "Title: Meeting-RH Hybrid Planning for Lifelong Multi-Agent Pickup "
            "and Delivery with Explainable Fleet Concurrency Control"
        ),
        (
            "Abstract. Building on Progress Report 1, this report advances lifelong "
            "warehouse MAPD from hierarchical M0–ECBS escalation to a dual-timescale "
            "Meeting-RH hybrid. Slow meetings (period H=12, adaptive window "
            "W∈{16,28}) negotiate stage pose, priority, and short-horizon space-time "
            "commits; fast ticks follow StageContracts with gap yielding and focal "
            "altruism. Two small imitation networks are gated by geometry: StageNet "
            "selects FINAL / APPROACH / CLEAR and a priority under Adaptive Stage "
            "Authority (ASA); FACN predicts a discrete residual Δ that adjusts "
            "throat-band capacity K=clip(K_TBC+Δ) without changing path costs. "
            "Geometry-gated protocols (contested-destination, pad egress, FG-CEBI, "
            "E4 sticky egress, progress pulse, PGFA, DLO, region admit) target "
            "reproduced corridor failures. On SH01–20 with 100 tasks per map, "
            "Meeting-RH matches ECBS/Hier on VALID and full completion (20/20) while "
            "cutting average simulation makespan to about 1708 ticks (~42% of Hier, "
            "~23% of always-on ECBS) and wall-clock to ~28 s. Following the "
            "feedback on Report 1, we include a compact Table I of key "
            "implementation settings (meeting, networks, protocols, training, "
            "and evaluation)."
        ),
    ],
    "Introduction (with OR w/o Related Work/Literature Review):": [
        (
            "Progress Report 1 motivated lifelong MAPD under FIFO stations, "
            "in-place turns, and shared validation, and showed that hierarchical "
            "M0–ECBS matches always-on ECBS on completion while lowering average "
            "makespan. The remaining gap was wall-clock cost and brittle behavior "
            "when cheap search stalls in capacity-1 corridors: always-on ECBS is "
            "reliable but slow; reactive PIBT is fast but incomplete on hard maps."
        ),
        (
            "This report asks whether an explainable hybrid can keep 20/20 full "
            "completion while further reducing simulation and planning time. We "
            "treat operational concurrency K and stage pose as slow variables "
            "amenable to small supervised models and geometry gates, and keep "
            "fast conflict repair algorithmic. The resulting Meeting-RH stack is "
            "intentionally incomplete relative to CBS, but aims for a better "
            "Pareto point inside the VALID ∧ full-completion regime."
        ),
        (
            "Related work (brief). CBS/ECBS and windowed lifelong wrappers (RHCR) "
            "provide strong joint plans; prioritized / WHCA*-style planners commit "
            "inside a horizon; PIBT and LaCAM-style methods scale reactively or "
            "with lightweight search. Learning-augmented MAPF often trains large "
            "policies. We keep learning small and gated (StageNet for pose, FACN "
            "for how-many-busy only) and pair it with auditable corridor protocols."
        ),
        (
            "Response to Report 1 feedback. Reviewers asked for a table of "
            "implementation details. Experimental Results therefore opens with a "
            "compact Table I of key settings before the main comparison table, "
            "instead of listing every protocol threshold."
        ),
    ],
    "Methodology:": [
        (
            "Problem model (unchanged core). Maps are 20×20 warehouse grids with "
            "FIFO pickup stations and drop hubs. Agents carry at most one package; "
            "vertex and edge-swap conflicts are forbidden; turns are in-place. "
            "Evaluation uses official ~100-task streams on SH01–20 with a shared "
            "validator (VALID / done / sim / wall)."
        ),
        (
            "Meeting-RH dual timescales. A meeting triggers at t=0, every H=12 "
            "ticks, on stall≥24 with gap≥8, or under dense/sparse no-progress "
            "rules. Inside a meeting: build path caches; choose stages via "
            "StageNet or teacher (ASA); apply region soft-admission; rewrite "
            "contested destinations and pad egress (FG-CEBI/E4 when gated); run "
            "priority-ordered windowed space-time BFS with W∈{16,28}; repair the "
            "window; dual-lex legalization when 10≤n≤14; issue StageContracts "
            "(TTL=16). Between meetings, agents follow commit heads with gap "
            "yielding; PGFA may clear a BFS prefix for a single focal beneficiary."
        ),
        (
            "StageNet + ASA. Candidates: FINAL, APPROACH@~1/3, APPROACH@~2/3, "
            "CLEAR. Agent features are 8-D (normalized pose/goal, distance, load "
            "status, blocked streak, region occupancy/heat). MLP 8→64→64 with "
            "scoring and priority heads. ASA selects network vs. handcrafted "
            "teacher from (n, near_pad_deg2) so noisy regimes (often n≤9 or n=18) "
            "fall back to the teacher."
        ),
        (
            "FACN + TBC + AFA. Multi-Band TBC sets static baseline K_TBC from "
            "(n, near). FACN maps static g∈ℝ¹⁰ and dynamic x∈ℝ¹² (22-D) to "
            "Δ∈{−2,…,+4}; K=clip(K_TBC+Δ) gates claiming only (not which agent). "
            "AFA clips harmful raises/lowers using throat occupancy, same-destination "
            "clusters, and stall/no-progress; sticky update limits thrash."
        ),
        (
            "Protocol suite (geometry-gated). Contested-destination gates prevent "
            "same-pad head-ons; pad egress and FG-CEBI enforce egress-before-ingress "
            "on large near-pad maps; E4 sticky egress clears post-unload deg≤2 "
            "occupancy on selected large fleets; progress pulses briefly lower K "
            "and widen W under no-progress; PGFA breaks mutual-yield livelock; "
            "DLO reduces hash-order jitter; region admit soft-caps FINAL density. "
            "All gates use geometry proxies, not map IDs."
        ),
        (
            "Baselines. Same SH01–20 / 100-task protocol as Report 1: PIBT, M0 "
            "(full-horizon space-time A*), PP (priority + A*), windowed ECBS, and "
            "Hier M0–ECBS (Report 1 method). Ours is Meeting-RH + protocols + "
            "StageNet(ASA) + FACN(AFA). Seed PYTHONHASHSEED=0; CPU; per-map "
            "subprocess isolation."
        ),
    ],
    "Experimental Results:": [
        (
            "Implementation details. Table I summarizes the main settings used in "
            "this report (meeting horizon, networks, protocols, training, and "
            "evaluation). Fine-grained protocol thresholds are described in the "
            "Methodology text; here we keep only the essentials for readability, "
            "as suggested by the Progress Report 1 feedback."
        ),
        "__PARAM_TABLE__",
        (
            "Main comparison (Table II). On SH01–20 with official 100-task "
            "streams, Meeting-RH achieves 20/20 full completion and 20/20 VALID, "
            "matching ECBS and Hier, while reducing average simulation makespan "
            "to 1708 ticks (~42% of Hier’s 4018, ~23% of ECBS’s 7364) and average "
            "wall-clock to ~28 s (vs 543 s ECBS and 1032 s Hier). PIBT/M0/PP "
            "remain incomplete on many maps; their low sim values are not "
            "successes. Relative to Report 1’s hierarchical M0–ECBS, Meeting-RH "
            "preserves completion/validity and substantially improves both sim "
            "and wall-clock by replacing frequent joint ECBS with sparse meetings "
            "plus contracts and protocols."
        ),
        "__RESULTS_TABLE__",
        (
            "Ablations (summary). Disabling StageNet collapses hard maps (e.g., "
            "SH10 done 60, SH17 done 32). Disabling corridor protocols hurts "
            "SH10/11/12/17 completion. Disabling PGFA can destroy SH17 (done 20). "
            "Disabling FACN mainly regresses open large fleets (e.g., SH20 sim "
            "660→813). Protocol-only improvements versus a legacy full stack yield "
            "about −9% avg_sim with 0 worse maps; FACN v2 versus noFACN on a "
            "broader SH01–40 suite yields about −5.8% avg_sim with gate maps "
            "passing. These results support the intended split: StageNet/protocols "
            "for hard-map completion; FACN for open-fleet throughput; meetings for "
            "wall-clock."
        ),
        (
            "Discussion. Report 1 showed hierarchy can match ECBS completion with "
            "better makespan than always-on ECBS. Report 2 shows that negotiating "
            "slow pose/concurrency variables and repairing corridors with gated "
            "rules can further cut sim and wall while keeping 20/20 full VALID "
            "completions. Table I makes the core settings explicit without "
            "listing every gate threshold."
        ),
    ],
    "Conclusion:": [
        (
            "Progress Report 2 replaces frequent ECBS escalation with Meeting-RH: "
            "dual-timescale contracts, StageNet+ASA, FACN+AFA, and geometry-gated "
            "protocols. On SH01–20 (100 tasks), we keep 20/20 full completion and "
            "VALID trajectories while reducing average sim to ~1708 and wall-clock "
            "to ~28 s, outperforming Report 1 Hier on both efficiency metrics. We "
            "also add a compact parameter table as requested. "
            "Next steps: (i) strengthen held-out FACN generalization across map "
            "families; (ii) tighter anytime windowed search / LNS inside meetings; "
            "(iii) more systematic ablations on 400-task stress streams; "
            "(iv) polish writing for the final project report."
        ),
    ],
}

# Compact parameter table (Report-1 feedback: show settings, but keep readable).
PARAM_HEADERS = ["Module", "Key settings", "Purpose"]
PARAM_ROWS = [
    [
        "Meeting",
        "H=12; W in {16,28}; contract TTL=16; stall-force=24",
        "When to replan and how long commits last",
    ],
    [
        "Planning",
        "Priority space-time BFS inside W (order: -priority, name)",
        "Short-horizon commits; not full-horizon A*/CBS",
    ],
    [
        "StageNet + ASA",
        "MLP 8→64→64; 4 stages (FINAL/APPROACH/CLEAR); ASA gate by (n, near)",
        "Choose stage pose and priority",
    ],
    [
        "FACN + TBC/AFA",
        "22-D input; Δ in {-2..+4}; K=clip(K_TBC+Δ); K_TBC≈6",
        "How many agents may stay busy",
    ],
    [
        "Protocols",
        "Contested-dest, pad egress / FG-CEBI / E4, progress pulse, PGFA, DLO, region admit",
        "Geometry-gated corridor failure tools",
    ],
    [
        "Training",
        "Imitation (CE; StageNet + listwise); AdamW; ckpts stage_selector_v1 / facn_v2",
        "Offline learning of StageNet and FACN",
    ],
    [
        "Evaluation",
        "SH01–20, official 100 tasks; PYTHONHASHSEED=0; CPU",
        "Primary comparison protocol",
    ],
]

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
    ["M0", "20", "4/20", "20/20", "44.6", "942†", "1478.5"],
    ["PP", "20", "1/20", "7/20", "32.4", "848†", "1716.1"],
    ["ECBS", "20", "20/20", "20/20", "100.0", "7364", "542.7"],
    ["Hier (Report 1)", "20", "20/20", "20/20", "100.0", "4018", "1032.1"],
    ["Meeting-RH (ours)", "20", "20/20", "20/20", "100.0", "1708", "~28"],
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


def _add_caption(cell, text: str) -> None:
    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.first_line_indent = Inches(0)
    run = p.add_run(text)
    set_run_font(run, size=10, bold=True)


def _fill_grid(table, headers, rows) -> None:
    for j, h in enumerate(headers):
        c = table.rows[0].cells[j]
        c.text = ""
        run = c.paragraphs[0].add_run(h)
        set_run_font(run, size=8, bold=True)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, row in enumerate(rows):
        highlight = "ours" in str(row[0]).lower() or "meeting" in str(row[0]).lower()
        for j, val in enumerate(row):
            c = table.rows[i + 1].cells[j]
            c.text = ""
            run = c.paragraphs[0].add_run(str(val))
            set_run_font(run, size=8, bold=highlight)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER


def _add_param_table_in_cell(cell) -> None:
    _add_caption(
        cell,
        "Table I. Key implementation settings (Meeting-RH).",
    )
    table = cell.add_table(rows=1 + len(PARAM_ROWS), cols=len(PARAM_HEADERS))
    table.style = "Table Grid"
    _fill_grid(table, PARAM_HEADERS, PARAM_ROWS)
    p = cell.add_paragraph()
    p.paragraph_format.space_after = Pt(6)


def _ensure_name_id(doc: Document, name: str, student_id: str) -> None:
    """Keep / restore Name and Student ID; never wipe them when refilling sections."""
    for para in doc.paragraphs:
        texts = [t.text for t in para._element.iter(qn("w:t")) if t.text]
        full = "".join(texts).strip()
        if full.startswith("Name:") and name not in full:
            # Clear extra runs after label, then append name.
            label_done = False
            for run in para.runs:
                if not label_done and "Name" in (run.text or ""):
                    run.text = "Name:"
                    label_done = True
                elif label_done:
                    run.text = ""
            if para.runs:
                para.runs[-1].text = (para.runs[-1].text or "") + f" {name}"
            else:
                run = para.add_run(f"Name: {name}")
                set_run_font(run)
        elif full.startswith("Student ID:") and student_id not in full:
            label_done = False
            for run in para.runs:
                if not label_done and "ID" in (run.text or ""):
                    # Keep split 'S'+'tudent ID:' as in template
                    label_done = True
                elif label_done:
                    run.text = ""
            if para.runs:
                para.runs[-1].text = (para.runs[-1].text or "") + f" {student_id}"
            else:
                run = para.add_run(f"Student ID: {student_id}")
                set_run_font(run)


def _add_results_table_in_cell(cell) -> None:
    _add_caption(
        cell,
        "Table II. Aggregate results on SH01–20 (official 100). "
        "† sim on incomplete maps is not comparable as “faster.”",
    )
    table = cell.add_table(rows=1 + len(RESULTS_ROWS), cols=len(RESULTS_HEADERS))
    table.style = "Table Grid"
    _fill_grid(table, RESULTS_HEADERS, RESULTS_ROWS)
    p = cell.add_paragraph()
    p.paragraph_format.space_after = Pt(6)


def _heading_to_table_index(doc: Document) -> dict[str, int]:
    table_i = -1
    mapping: dict[str, int] = {}
    last_heading = None
    for child in doc.element.body.iterchildren():
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
    # Capture existing identity before any edits.
    existing_name, existing_id = PRESERVE_NAME, PRESERVE_ID
    for para in doc.paragraphs:
        texts = [t.text for t in para._element.iter(qn("w:t")) if t.text]
        full = "".join(texts).strip()
        if full.startswith("Name:") and len(full) > 5:
            existing_name = full.split(":", 1)[1].strip() or existing_name
        if full.startswith("Student ID:") and len(full) > 11:
            existing_id = full.split(":", 1)[1].strip() or existing_id

    mapping = _heading_to_table_index(doc)
    if len(mapping) != 5:
        raise RuntimeError(f"Expected 5 section tables, got {mapping}")

    for heading, texts in SECTION_CONTENTS.items():
        ti = mapping[heading]
        cell = doc.tables[ti].cell(0, 0)
        _clear_cell(cell)
        first = True
        for item in texts:
            if item == "__PARAM_TABLE__":
                _add_param_table_in_cell(cell)
                first = False
                continue
            if item == "__RESULTS_TABLE__":
                _add_results_table_in_cell(cell)
                first = False
                continue
            _add_para_in_cell(cell, item, first=first)
            first = False

    _ensure_name_id(doc, existing_name, existing_id)
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
    ap.add_argument("--template", type=str, default="")
    ap.add_argument("--also", action="append", default=[])
    args = ap.parse_args()

    if args.template:
        template = Path(args.template)
    elif DEFAULT_TEMPLATE.exists():
        template = DEFAULT_TEMPLATE
    else:
        template = FALLBACK_TEMPLATE
    if not template.exists():
        raise SystemExit(f"Template not found: {template}")

    doc = fill_template(template)
    # Write docs/ first, then root (so Name/ID preserved copy is the source of truth).
    wrote = _save(doc, OUT_DOCS)
    print(f"Wrote {wrote} ({wrote.stat().st_size} bytes)")
    try:
        shutil.copy2(wrote, OUT_ROOT)
        print(f"Copied {OUT_ROOT}")
    except PermissionError:
        alt = OUT_ROOT.with_name(OUT_ROOT.stem + "_new" + OUT_ROOT.suffix)
        shutil.copy2(wrote, alt)
        print(f"Permission denied for {OUT_ROOT}; copied {alt}")

    for dest_s in args.also:
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
