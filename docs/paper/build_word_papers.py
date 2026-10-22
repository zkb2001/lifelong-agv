#!/usr/bin/env python3
"""Build EN/ZH academic manuscripts as .docx (international manuscript format)."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "word"


def set_run_font(run, name="Times New Roman", size=12, bold=False, italic=False, east_asia=None):
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    run.font.name = name
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    ea = east_asia or name
    rFonts.set(qn("w:eastAsia"), ea)
    rFonts.set(qn("w:cs"), name)


def set_paragraph_format(p, *, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=6,
                         first_indent=None, line_spacing=1.15):
    pf = p.paragraph_format
    pf.alignment = align
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line_spacing
    if first_indent is not None:
        pf.first_line_indent = Inches(first_indent)


def add_text(p, text, **kwargs):
    run = p.add_run(text)
    set_run_font(run, **kwargs)
    return run


def heading(doc, text, level=1, east_asia="SimSun"):
    # Manual heading for consistent numbering/font across locales
    p = doc.add_paragraph()
    sizes = {1: 14, 2: 12, 3: 12}
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=12, space_after=6,
                         first_indent=0, line_spacing=1.15)
    add_text(p, text, size=sizes.get(level, 12), bold=True, east_asia=east_asia)
    return p


def body(doc, text, east_asia="SimSun", first_indent=0.3):
    p = doc.add_paragraph()
    set_paragraph_format(p, first_indent=first_indent)
    add_text(p, text, east_asia=east_asia)
    return p


def caption(doc, text, east_asia="SimSun"):
    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=3, space_after=10,
                         first_indent=0)
    add_text(p, text, size=10, italic=True, east_asia=east_asia)
    return p


def add_figure(doc, path: Path, width=5.8):
    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=8, space_after=2,
                         first_indent=0)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))


def add_table(doc, headers, rows, east_asia="SimSun"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=2,
                             first_indent=0)
        add_text(p, h, size=10, bold=True, east_asia=east_asia)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=2, space_after=2,
                                 first_indent=0)
            add_text(p, str(val), size=10, bold=("Ours" in str(row[0]) or "本文" in str(row[0])),
                     east_asia=east_asia)
    return table


def add_refs(doc, refs, east_asia="SimSun"):
    heading(doc, "References" if east_asia == "Times New Roman" else "参考文献", 1, east_asia)
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=4,
                             first_indent=0, line_spacing=1.0)
        pf = p.paragraph_format
        pf.left_indent = Inches(0.35)
        pf.first_line_indent = Inches(-0.35)
        add_text(p, f"[{i}] {r}", size=10, east_asia=east_asia)


def setup_doc(chinese=False):
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "SimSun" if chinese else "Times New Roman")
    return doc


REFS_EN = [
    "Sharon, G., Stern, R., Felner, A., and Sturtevant, N. R. 2015. Conflict-based search for optimal multi-agent pathfinding. Artificial Intelligence 219: 40–66.",
    "Barer, M., Sharon, G., Stern, R., and Felner, A. 2014. Suboptimal variants of the conflict-based search algorithm for the multi-agent pathfinding problem. In Proceedings of SoCS.",
    "Okumura, K., Machida, M., Défago, X., and Tamura, Y. 2022. Priority inheritance with backtracking for iterative multi-agent path finding. Artificial Intelligence 310: 103752.",
    "Ma, H., Li, J., Kumar, T. K. S., and Koenig, S. 2017. Lifelong multi-agent path finding for online pickup and delivery tasks. In Proceedings of AAMAS.",
    "Li, J., Tinka, A., Kiesel, S., Durham, J. W., Kumar, T. K. S., and Koenig, S. 2021. Lifelong multi-agent path finding in large-scale warehouses. In Proceedings of AAAI, 35(13): 11272–11281.",
    "Madar, N., Solovey, K., and Salzman, O. 2022. Leveraging experience in lifelong multi-agent path finding. In Proceedings of SoCS.",
    "Silver, D. 2005. Cooperative pathfinding. In Proceedings of AIIDE.",
    "Stern, R., et al. 2019. Multi-agent pathfinding: Definitions, variants, and benchmarks. In Proceedings of SoCS.",
    "Pearl, J., and Kim, J. H. 1982. Studies in semi-admissible heuristics. IEEE Transactions on Pattern Analysis and Machine Intelligence 4(4): 392–399.",
    "Chen, Z., et al. 2024. Scaling lifelong multi-agent path finding to more realistic settings: Research challenges and opportunities. In Proceedings of SoCS.",
    "Skrynnik, A., Andreychuk, A., Yakovlev, K., and Panov, A. 2024. Learn to follow: Decentralized lifelong multi-agent pathfinding via planning and learning. In Proceedings of AAAI.",
    "Li, J., Chen, Z., Harabor, D., Stuckey, P. J., and Koenig, S. 2021. Anytime multi-agent path finding via large neighborhood search. In Proceedings of IJCAI.",
    "Liu, M., Ma, H., Li, J., and Koenig, S. 2019. Task and path planning for multi-agent pickup and delivery. In Proceedings of AAMAS.",
    "Sartoretti, G., et al. 2019. PRIMAL: Pathfinding via reinforcement and imitation multi-agent learning. IEEE Robotics and Automation Letters.",
    "Čáp, M., Novák, P., Kleiner, A., and Selecký, M. 2015. Prioritized planning algorithms for trajectory coordination of multiple mobile robots. IEEE Transactions on Automation Science and Engineering.",
]


def build_en():
    doc = setup_doc(False)
    ea = "Times New Roman"

    # Title
    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=12,
                         first_indent=0, line_spacing=1.15)
    add_text(
        p,
        "Explainable Fleet Concurrency Control for Lifelong Multi-Agent "
        "Pickup and Delivery via Meeting-based Hybrid Planning",
        size=16, bold=True, east_asia=ea,
    )

    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=4,
                         first_indent=0)
    add_text(p, "Anonymous Authors", size=12, east_asia=ea)

    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=18,
                         first_indent=0)
    add_text(p, "Affiliation withheld for anonymous review", size=11, italic=True, east_asia=ea)

    # Abstract
    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=6, space_after=6,
                         first_indent=0)
    add_text(p, "Abstract. ", size=12, bold=True, italic=True, east_asia=ea)
    add_text(
        p,
        "Lifelong multi-agent pickup and delivery (MAPD) requires continuously assigning and "
        "executing pickup–delivery tasks under collision constraints. Search-based windowed "
        "solvers such as RHCR with ECBS can achieve high completion rates but incur large "
        "wall-clock cost, whereas reactive methods such as PIBT are fast but fragile in "
        "capacity-1 corridors. We propose a Meeting-RH hybrid stack that separates slow "
        "variables—stage pose, priority, and short-horizon commits—from fast conflict repair, "
        "and introduces two small imitation-learned modules with geometry gates: StageNet "
        "selects among FINAL / APPROACH / CLEAR stages and emits a priority scalar, while FACN "
        "predicts a discrete concurrency residual Δ that adjusts the throat-band capacity "
        "K = clip(K_TBC + Δ) without modifying path costs. Geometry-gated protocols "
        "(contested-destination gates, pad egress, fleet-gated CEBI, sticky post-unload egress, "
        "progress pulses, and focal altruism) map to reproduced failure modes without map-id "
        "hardcoding. On a 20-map stress suite with approximately 100 lifelong tasks per map, "
        "our method matches ECBS/Hier on VALID and full completion while reducing average "
        "simulation time to roughly 42% of hierarchical ECBS and wall-clock by 1–2 orders of "
        "magnitude. Ablations show StageNet and corridor protocols are critical for completion "
        "on hard maps, whereas FACN primarily improves throughput on open large fleets.",
        size=12, east_asia=ea,
    )

    p = doc.add_paragraph()
    set_paragraph_format(p, first_indent=0, space_after=12)
    add_text(p, "Keywords: ", size=12, bold=True, italic=True, east_asia=ea)
    add_text(
        p,
        "Lifelong MAPD; Multi-Agent Path Finding; Rolling Horizon; Imitation Learning; "
        "Fleet Sizing; Explainable Coordination",
        size=12, east_asia=ea,
    )

    # 1 Introduction
    heading(doc, "1. Introduction", 1, ea)
    body(
        doc,
        "Multi-Agent Path Finding (MAPF) asks for collision-free paths for multiple agents on "
        "a shared graph [1, 8]. Automated warehouses and sortation centers typically require "
        "the lifelong variant: agents repeatedly receive new goals as tasks stream in [4, 5]. "
        "In lifelong MAPD, planners must jointly handle task progress, corridor capacity, and "
        "limited online compute [10].",
        ea,
    )
    body(
        doc,
        "Two families dominate practice. Search-based methods (CBS/ECBS and their lifelong "
        "wrappers) offer strong validity but scale poorly in wall-clock when replanning many "
        "agents [2, 5]. Reactive methods (e.g., PIBT) decide one step at a time and scale, yet "
        "degrade in narrow throats and dead-ends [3]. Rolling-horizon collision resolution "
        "(RHCR) interpolates by resolving conflicts only inside a window [5], but still pays "
        "for joint search in that window and does not explicitly learn how many agents should "
        "remain busy under heterogeneous geometry.",
        ea,
    )
    body(
        doc,
        "This paper targets a missing axis of lifelong MAPD control: operational concurrency—"
        "the number of agents allowed to remain busy—together with stage pose (whether an "
        "agent should drive to its final goal, an intermediate approach cell, or clear a "
        "corridor). These decisions are slow, explainable, and amenable to small supervised "
        "models, while fast collision repair can remain algorithmic.",
        ea,
    )

    p = doc.add_paragraph()
    set_paragraph_format(p, first_indent=0.3)
    add_text(p, "Contributions. ", bold=True, east_asia=ea)
    add_text(
        p,
        "(1) Meeting-RH dual timescales. Periodically (every H = 12 ticks, or on stall/"
        "no-progress), a meeting negotiates stages, priorities, and windowed space-time "
        "commits; between meetings, agents follow contracts with gap yielding and optional "
        "focal altruism. (2) FACN (Fleet Activation & Concurrency Network). A compact MLP "
        "predicts Δ ∈ {−2, …, +4} from static geometry and dynamic traffic features; "
        "K = clip(K_TBC + Δ) gates claiming. FACN does not select which agent is busy. "
        "(3) StageNet + ASA. An MLP scores FINAL / APPROACH@1/3 / APPROACH@2/3 / CLEAR and a "
        "priority; Adaptive Stage Authority (ASA) chooses network vs. teacher from (n, near). "
        "(4) Geometry-gated protocol suite addressing reproduced failures without map "
        "identifiers. (5) Empirical evidence on SH01–20: VALID and full completion on par with "
        "ECBS/Hier, with substantially lower simulation makespan and wall-clock.",
        east_asia=ea,
    )
    body(
        doc,
        "We emphasize that Meeting-RH is incomplete relative to CBS: it uses bounded-horizon "
        "prioritized space-time BFS and heuristic commitments. Gains are practical Pareto "
        "improvements inside the VALID ∧ full-completion regime, not optimality claims.",
        ea,
    )

    # 2 Related Work
    heading(doc, "2. Related Work", 1, ea)
    body(
        doc,
        "Optimal and bounded-suboptimal MAPF. CBS splits conflicts at a high level and "
        "replans single-agent paths under constraints [1]. ECBS replaces best-first search "
        "with focal search to guarantee a user-specified suboptimality bound [2, 9]. These "
        "solvers underpin many warehouse systems but become expensive under lifelong "
        "replanning.",
        ea,
    )
    body(
        doc,
        "Prioritized and windowed planning. Classical prioritized planning assigns agent "
        "order and plans sequentially against reservations [7, 15]. WHCA* commits only inside "
        "a window [7]. RHCR decomposes lifelong MAPF into windowed instances and replans "
        "every h steps [5]. Experience-warmed variants accelerate successive queries [6]. Our "
        "meetings also use a window, but negotiate stage and concurrency explicitly and "
        "execute with cheap follow between meetings.",
        ea,
    )
    body(
        doc,
        "Reactive lifelong solvers and learning. PIBT uses priority inheritance and "
        "backtracking for iterative MAPF/MAPD [3]. Learning-augmented lifelong methods "
        "combine planning and RL policies [11, 14]. We keep learning small and gated for "
        "auditability. MAPD formulations couple assignment and routing [4, 13]. Competition "
        "analyses highlight concurrency, congestion, and myopia as open challenges [10]; "
        "FACN addresses operational K rather than purchasing agents.",
        ea,
    )

    # 3 Problem
    heading(doc, "3. Problem Formulation", 1, ea)
    body(
        doc,
        "Let G = (V, E) be a 4-connected grid. A fleet A = {a_i}_{i=1}^{n} executes lifelong "
        "pickup–delivery tasks. Each agent occupies one cell per discrete time, may wait, and "
        "may rotate in place by ±90° or 180° (each action costs one tick in our motion model). "
        "Vertex conflicts (two agents in the same cell at the same time) and edge-swap "
        "conflicts are forbidden. Pickup stations enforce FIFO and at most one pickup per "
        "station per tick. A trajectory is VALID iff all kinematic, capacity, and conflict "
        "constraints hold.",
        ea,
    )
    body(
        doc,
        "Quality metrics: done (correct deliveries), full100 (all tasks done), sim (finish "
        "time of the last delivery), and wall (planner runtime). We study stress maps SH01–20 "
        "with heterogeneous n and throat geometry, each with an official stream of "
        "approximately 100 tasks.",
        ea,
    )

    # 4 Method
    heading(doc, "4. Method: Meeting-RH Hybrid Stack", 1, ea)
    heading(doc, "4.1 Dual Timescales", 2, ea)
    body(
        doc,
        "Slow (meeting). A meeting triggers at t = 0, when t − t_last ≥ H = 12, when "
        "stall ≥ 24 with gap ≥ 8, or under dense/sparse no-progress rules. Inside a meeting "
        "the stack: (i) builds path caches; (ii) chooses stages via StageNet or teacher (ASA); "
        "(iii) applies region soft-admission; (iv) rewrites contested destinations and pad "
        "egress (including FG-CEBI/E4 when enabled); (v) runs priority-ordered windowed "
        "space-time BFS with horizon W ∈ {16, 28}; (vi) repairs the window; (vii) "
        "dual-lexicographic legalization when 10 ≤ n ≤ 14; (viii) issues StageContracts with "
        "TTL = 16.",
        ea,
    )
    body(
        doc,
        "Fast (inter-meeting). Agents follow commit heads or waypoints; low-priority agents "
        "gap-yield; PGFA may clear the BFS prefix of a single focal beneficiary when "
        "stall/streak thresholds fire. Contested-dest and pad-egress also run every tick. "
        "Unlike RHCR [5], we do not invoke a full joint windowed MAPF solver every replan "
        "period; we negotiate a compact set of slow variables and rely on protocols plus "
        "local repair.",
        ea,
    )

    p = doc.add_paragraph()
    set_paragraph_format(p, first_indent=0, space_before=6, space_after=4)
    add_text(p, "Algorithm 1. ", bold=True, size=10, east_asia=ea)
    add_text(p, "Meeting-RH tick loop (conceptual).", italic=True, size=10, east_asia=ea)

    algo_lines = [
        "1: Initialize contracts; t_last ← −H",
        "2: for t = 0, 1, 2, … do",
        "3:     if meeting trigger (t − t_last ≥ H or stall / no-progress) then",
        "4:         K ← clip(K_TBC + FACN(g, x)) with AFA",
        "5:         Stages, priorities ← ASA(StageNet, teacher)",
        "6:         Apply dest-gates, pad-egress, region-admit",
        "7:         Priority-ordered space-time BFS with horizon W ∈ {16, 28}",
        "8:         Repair window; issue StageContracts (TTL = 16)",
        "9:         t_last ← t",
        "10:    end if",
        "11:    Follow commits; gap-yield; optional PGFA; tick protocols",
        "12: end for",
    ]
    for line in algo_lines:
        p = doc.add_paragraph()
        set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=0, space_after=0,
                             first_indent=0, line_spacing=1.0)
        pf = p.paragraph_format
        pf.left_indent = Inches(0.25)
        add_text(p, line, size=10, name="Courier New", east_asia="Courier New")

    heading(doc, "4.2 Geometry Gates and FACN", 2, ea)
    body(
        doc,
        "Multi-Band TBC sets a static baseline K_TBC from (n, near) bands (default 6; "
        "mid/dense bands lower to 5; mega fleets raise). FACN concatenates static "
        "g ∈ ℝ¹⁰ and dynamic x ∈ ℝ¹² into ℝ²², scores Δ ∈ {−2, …, +4}, and sets "
        "K = clip(K_TBC + Δ). AFA clips raises/lowers using throat occupancy, same-destination "
        "clusters, and stall/no-progress, preventing thrash. Claiming respects K; BFS edge "
        "costs are untouched.",
        ea,
    )

    heading(doc, "4.3 StageNet", 2, ea)
    body(
        doc,
        "For each agent, candidates include FINAL (goal cell), APPROACH at roughly 1/3 and "
        "2/3 along a geometric path, and CLEAR (a nearby low-stay cell). Agent features "
        "x_a ∈ ℝ⁸ encode normalized position, goal, Manhattan distance, load/assign status, "
        "blocked streak, and region occupancy/heat. An MLP (8→64→64) with scoring/priority "
        "heads outputs stage logits and priority p. Training uses imitation learning (stage "
        "cross-entropy + listwise priority + fleet regularization). ASA switches to a "
        "handcrafted teacher on regimes where the net is noisy (e.g., many n ≤ 9 cases).",
        ea,
    )

    heading(doc, "4.4 Protocol Suite", 2, ea)
    body(
        doc,
        "Each tool maps to a failure mode: contested-destination gates prevent same-pad "
        "head-ons; pad egress and FG-CEBI enforce egress-before-ingress; E4 sticky egress "
        "clears post-unload deg ≤ 2 occupancy on large mid-throat fleets; progress pulses "
        "briefly lower K and widen W under no-progress; PGFA prevents mutual-yield livelock "
        "by helping one focal agent; DLO reduces hash-order jitter in legalization; region "
        "admit soft-caps FINAL density per tile. The umbrella name PGCP denotes the "
        "corridor-oriented subset.",
        ea,
    )

    add_figure(doc, FIG / "fig1_overview.jpg")
    caption(doc, "Figure 1. Meeting-RH stack: geometry gates, meetings, tick execution, and FACN feedback on K.", ea)
    add_figure(doc, FIG / "fig2_meeting.jpg")
    caption(doc, "Figure 2. Meeting lifecycle: pre-follow → trigger → inputs → internal steps → contracts → return.", ea)

    # 5 Experiments
    heading(doc, "5. Experiments", 1, ea)
    heading(doc, "5.1 Setup", 2, ea)
    body(
        doc,
        "We evaluate on SH01–20 with official ~100-task streams. Baselines: PIBT (reactive), "
        "M0 (space-time A* with greedy assignment), PP (priority task allocation + space-time "
        "A*), windowed ECBS, and Hier (M0 on easy maps, ECBS handoff on hard maps). Ours: "
        "Meeting-RH + protocols + StageNet (ASA) + FACN (AFA). Reproduction uses "
        "PYTHONHASHSEED=0, CPU, and per-map subprocess isolation. Primary comparisons require "
        "VALID trajectories.",
        ea,
    )

    heading(doc, "5.2 Main Results", 2, ea)
    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=4,
                         first_indent=0)
    add_text(
        p,
        "Table 1. Aggregate results on SH01–20 (official 100). Lower sim/wall is better among "
        "VALID ∧ full100 methods. † sim on incomplete maps is not comparable as “faster.”",
        size=10, italic=True, east_asia=ea,
    )
    add_table(
        doc,
        ["Method", "full100", "VALID", "avg_done", "avg_sim", "avg_wall"],
        [
            ["PIBT", "8/20", "20/20", "63.9", "62811", "40.5s"],
            ["M0", "4/20", "20/20", "44.6", "942†", "1478s"],
            ["PP", "1/20", "7/20", "32.4", "848†", "1716s"],
            ["ECBS", "20/20", "20/20", "100", "7364", "543s"],
            ["Hier", "20/20", "20/20", "100", "4018", "1032s"],
            ["Ours", "20/20", "20/20", "100", "1708", "~28s"],
        ],
        ea,
    )
    body(
        doc,
        "Discussion. PIBT/M0/PP fail to complete many maps; their low sim values are not "
        "successes. Among methods that achieve 20/20 full100, Ours reduces avg_sim to ~42% of "
        "Hier and ~23% of ECBS, with wall-clock ~28s vs hundreds–thousands of seconds. The "
        "speedup stems from (i) infrequent meetings, (ii) bounded-horizon space-time BFS "
        "rather than repeated full-goal space-time A*, (iii) concurrency limiting, and "
        "(iv) fewer deadlock-induced replans. Protocol-only improvements vs. a legacy FULL "
        "stack yield about −9.0% avg_sim with 0 worse maps. FACN v2 vs. noFACN on SH01–40 "
        "yields about −5.82% avg_sim (20/13/5 better/same/worse) with GATE maps PASS.",
        ea,
    )

    heading(doc, "5.3 Ablations", 2, ea)
    body(
        doc,
        "Leaving out components (relative to full): disabling StageNet collapses hard maps "
        "(e.g., SH10 done 60, SH17 done 32). Disabling PGCP-related corridor tools drops "
        "completion on SH10/11/12/17. Disabling PGFA can destroy SH17 (done 20) and inflate "
        "sim on SH03. Disabling FACN hurts open large fleets (e.g., SH20 sim 660→813). Global "
        "direction-field nets are not used in the default stack; when tested historically "
        "they did not improve done/sim—hence we do not claim them as a contribution.",
        ea,
    )
    add_figure(doc, FIG / "fig3_stagenet.jpg", width=5.2)
    caption(doc, "Figure 3. StageNet I/O (8-D agent state; ASA is an external gate).", ea)
    add_figure(doc, FIG / "fig4_facn.jpg", width=5.2)
    caption(doc, "Figure 4. FACN I/O: static g and dynamic x → Δ → K.", ea)

    heading(doc, "5.4 Why Windowed BFS Beats Serial M0 A*", 2, ea)
    body(
        doc,
        "Although both touch search, M0 repeatedly runs full-horizon space-time A* against a "
        "dense reservation table at assignment time. Meeting-RH mostly executes contract "
        "following; meetings call priority-ordered space-time BFS only inside W. Thus call "
        "frequency, search depth, and conflict thrash all drop—explaining wall-clock gaps "
        "even when “search” remains in the stack.",
        ea,
    )

    # 6 Limitations
    heading(doc, "6. Limitations and Threats to Validity", 1, ea)
    body(
        doc,
        "Meeting-RH is incomplete and may fail on pathological maps. FACN held-out "
        "generalization across map families remains limited. Protocols require maintenance "
        "and can interact. Stress maps are not a full industrial digital twin. Results are "
        "reported for fixed seeds and CPU; GPU training is offline only.",
        ea,
    )

    # 7 Conclusion
    heading(doc, "7. Conclusion", 1, ea)
    body(
        doc,
        "We presented Meeting-RH, an explainable hybrid for lifelong MAPD that couples "
        "geometry-gated protocols with two small networks: StageNet for stage/priority and "
        "FACN for operational concurrency K. Empirically, the stack matches strong search "
        "baselines on validity and completion while substantially improving simulation time "
        "and wall-clock. Future work includes stronger held-out concurrency learning, tighter "
        "integration with anytime windowed search [12], and richer kinodynamic models [10].",
        ea,
    )

    # Appendix
    heading(doc, "Appendix A. FACN Feature Tables", 1, ea)
    body(
        doc,
        "Static g ∈ ℝ¹⁰: n, free cells, near_pad_deg2, #pickups, #dropoffs, density, "
        "idle-pool ratio, K_TBC, region-free mean/max (normalized as in code). Dynamic "
        "x ∈ ℝ¹²: busy, idle, remaining, done_rate, stall, avg commit H, repairs, conflict "
        "heat, region occupancy, #dest-gates, same-dest cluster, throat occupancy.",
        ea,
    )
    heading(doc, "Appendix B. Reproducibility", 1, ea)
    body(
        doc,
        "Entry point: python -m greed_coop_mcts with policy=rrsc, optional FACN checkpoint "
        "facn_v2.pt, StageNet stage_selector_v1.pt. Meeting constants: MEET_H=12, "
        "MEET_W ∈ {16, 28}.",
        ea,
    )

    add_refs(doc, REFS_EN, ea)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "MeetingRH_Paper_EN.docx"
    doc.save(path)
    print(f"Wrote {path}")
    return path


def build_zh():
    doc = setup_doc(True)
    ea = "SimSun"

    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=12,
                         first_indent=0)
    add_text(
        p,
        "面向终身多智体取送货的可解释车队并发控制：基于会商的混合规划",
        size=16, bold=True, east_asia=ea,
    )

    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=4,
                         first_indent=0)
    add_text(p, "匿名作者", size=12, east_asia=ea)

    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=0, space_after=18,
                         first_indent=0)
    add_text(p, "匿名评审，单位信息暂隐", size=11, italic=True, east_asia=ea)

    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=6, space_after=6,
                         first_indent=0)
    add_text(p, "摘要：", size=12, bold=True, east_asia=ea)
    add_text(
        p,
        "终身多智体取送货（Lifelong MAPD）要求在碰撞约束下持续分配并执行取货—送货任务。"
        "基于搜索的窗口求解器（如 RHCR 结合 ECBS）完成率高但墙钟时间昂贵；反应式方法（如 PIBT）"
        "速度快，却在容量为 1 的走廊中脆弱。本文提出 Meeting-RH 混合栈：将慢变量——阶段位姿、"
        "优先级与短视窗提交——与快速冲突修复分离，并引入两个带几何门控的小型模仿学习模块。"
        "StageNet 在 FINAL / APPROACH / CLEAR 阶段间选择并输出优先级标量；FACN 预测离散并发残差 Δ，"
        "将喉带容量调整为 K = clip(K_TBC + Δ)，且不改动路径代价。几何门控协议（竞争目的地门、"
        "垫区出口、车队门控 CEBI、粘性卸货后出口、进度脉冲、焦点利他等）对应可复现失败模式，"
        "无需地图 ID 硬编码。在含每图约 100 条终身任务的 20 图压力套件上，本方法在 VALID 与全量"
        "完成上与 ECBS/Hier 持平，平均仿真时间约为分层 ECBS 的 42%，墙钟时间降低 1–2 个数量级。"
        "消融表明 StageNet 与走廊协议对难图完成至关重要，而 FACN 主要提升开阔大车队吞吐。",
        size=12, east_asia=ea,
    )

    p = doc.add_paragraph()
    set_paragraph_format(p, first_indent=0, space_after=12)
    add_text(p, "关键词：", size=12, bold=True, east_asia=ea)
    add_text(
        p,
        "终身 MAPD；多智体路径规划；滚动视窗；模仿学习；车队规模；可解释协同",
        size=12, east_asia=ea,
    )

    heading(doc, "1 引言", 1, ea)
    body(
        doc,
        "多智体路径规划（MAPF）要求共享图上多智体无碰撞路径[1, 8]。仓储与分拣场景通常需要"
        "终身变体：任务流到来时智体反复接收新目标[4, 5]。在终身 MAPD 中，规划器需同时处理任务"
        "进度、走廊容量与有限在线算力[10]。",
        ea, first_indent=0.3,
    )
    body(
        doc,
        "实践中两大家族主导。基于搜索的方法（CBS/ECBS 及其终身封装）有效性强，但在多智体重规划"
        "下墙钟开销大[2, 5]。反应式方法（如 PIBT）逐步决策、可扩展，却在窄喉与死胡同中退化[3]。"
        "滚动视窗冲突消解（RHCR）仅在窗口内消解冲突[5]，但仍为联合搜索付费，且未显式学习异构"
        "几何下应允许多少智体保持忙碌。",
        ea,
    )
    body(
        doc,
        "本文针对终身 MAPD 控制中缺失的一轴：运营并发度——允许多少智体保持忙碌——以及阶段位姿"
        "（直达最终目标、中间接近点，或清廊）。这些决策慢、可解释、适合小型监督模型；快速碰撞"
        "修复可保持算法化。",
        ea,
    )

    p = doc.add_paragraph()
    set_paragraph_format(p, first_indent=0.3)
    add_text(p, "贡献：", bold=True, east_asia=ea)
    add_text(
        p,
        "（1）Meeting-RH 双时间尺度。周期性（每 H = 12 拍，或停滞/无进展时）召开会商，协商阶段、"
        "优先级与窗口时空提交；会商之间按契约执行，配合空隙让行与可选焦点利他。"
        "（2）FACN（车队激活与并发网络）。紧凑 MLP 由静/动态特征预测 Δ ∈ {−2, …, +4}；"
        "K = clip(K_TBC + Δ) 门控认领。FACN 不选择具体哪辆车忙碌。"
        "（3）StageNet + ASA。MLP 为 FINAL / APPROACH@1/3 / APPROACH@2/3 / CLEAR 打分并输出优先级；"
        "自适应阶段权威（ASA）依 (n, near) 在网络与教师间切换。"
        "（4）几何门控协议套件，针对可复现失败，无地图标识符。"
        "（5）实证：SH01–20 上 VALID 与全量完成对标 ECBS/Hier，仿真与墙钟显著更优。",
        east_asia=ea,
    )
    body(
        doc,
        "需强调 Meeting-RH 相对 CBS 不完备：采用有界视窗优先时空 BFS 与启发式提交。收益是 "
        "VALID ∧ 全量完成体制内的实用帕累托改进，而非最优性声明。",
        ea,
    )

    heading(doc, "2 相关工作", 1, ea)
    body(
        doc,
        "最优与有界次优 MAPF。CBS 在高层拆分冲突、在约束下重规划单智体路径[1]。ECBS 用焦点搜索"
        "保证用户指定次优界[2, 9]。优先与窗口规划方面，WHCA*[7] 与优先规划[15]按序相对预约表规划；"
        "RHCR[5] 将终身 MAPF 分解为窗口实例；经验预热变体可加速相继查询[6]。反应式终身求解器方面，"
        "PIBT[3] 与学习增强方法[11, 14]被广泛研究。MAPD 形式化耦合分配与路由[4, 13]。竞赛分析"
        "突出并发、拥堵与短视等挑战[10]。",
        ea,
    )

    heading(doc, "3 问题形式化", 1, ea)
    body(
        doc,
        "设 G = (V, E) 为四连通网格。车队 A = {a_i}_{i=1}^{n} 执行终身取送货。每拍一格，可等待，"
        "可原地 ±90° 或 180° 旋转（本运动模型中各动作耗一拍）。禁止顶点冲突与边交换冲突。"
        "取货站 FIFO，每站每拍至多一次取货。轨迹 VALID 当且仅当运动学、容量与冲突约束均满足。"
        "质量指标：done（正确送达数）、full100（全部完成）、sim（最后送达仿真时间）、wall"
        "（规划器墙钟）。评测压力图 SH01–20，异构 n 与喉道几何，每图官方约 100 条任务流。",
        ea,
    )

    heading(doc, "4 方法：Meeting-RH 混合栈", 1, ea)
    heading(doc, "4.1 双时间尺度", 2, ea)
    body(
        doc,
        "慢（会商）。触发条件：t = 0；t − t_last ≥ H = 12；stall ≥ 24 且 gap ≥ 8；或密/疏无进展"
        "规则。会商内：（i）构建路径缓存；（ii）ASA 选择 StageNet 或教师阶段；（iii）区域软准入；"
        "（iv）改写竞争目的地与垫区出口（含 FG-CEBI/E4）；（v）优先级序窗口时空 BFS，"
        "W ∈ {16, 28}；（vi）窗口修复；（vii）10 ≤ n ≤ 14 时双字典序合法化；（viii）签发 "
        "TTL = 16 的 StageContract。",
        ea,
    )
    body(
        doc,
        "快（会商间）。跟进提交头或航点；低优先级空隙让行；PGFA 在停滞/连败阈值触发时为单一焦点"
        "受益者清 BFS 前缀。竞争目的地与垫区出口亦每拍运行。相对 RHCR[5]：我们并不每重规划周期"
        "对全部冲突调用完整联合窗口 MAPF；而是协商紧凑慢变量，依赖协议与局部修复。",
        ea,
    )

    p = doc.add_paragraph()
    set_paragraph_format(p, first_indent=0, space_before=6, space_after=4)
    add_text(p, "算法 1. ", bold=True, size=10, east_asia=ea)
    add_text(p, "Meeting-RH 拍循环（概念）。", italic=True, size=10, east_asia=ea)
    for line in [
        "1: 初始化契约；t_last ← −H",
        "2: for t = 0, 1, 2, … do",
        "3:     if 会商触发（t − t_last ≥ H 或停滞/无进展） then",
        "4:         K ← clip(K_TBC + FACN(g, x))（含 AFA）",
        "5:         阶段、优先级 ← ASA(StageNet, 教师)",
        "6:         应用目的地门、垫区出口、区域准入",
        "7:         优先序时空 BFS，视窗 W ∈ {16, 28}",
        "8:         修复窗口；签发 StageContract（TTL = 16）",
        "9:         t_last ← t",
        "10:    end if",
        "11:    跟进提交；空隙让行；可选 PGFA；拍协议",
        "12: end for",
    ]:
        p = doc.add_paragraph()
        set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.LEFT, space_before=0, space_after=0,
                             first_indent=0, line_spacing=1.0)
        p.paragraph_format.left_indent = Inches(0.25)
        add_text(p, line, size=10, name="Courier New", east_asia=ea)

    heading(doc, "4.2 容量与学习的几何门", 2, ea)
    body(
        doc,
        "Multi-Band TBC 由 (n, near) 带给出静态基线 K_TBC。FACN 将静态 g ∈ ℝ¹⁰ 与动态 "
        "x ∈ ℝ¹² 拼接为 ℝ²²，打分 Δ ∈ {−2, …, +4}，令 K = clip(K_TBC + Δ)。AFA 依喉占用、"
        "同目的地簇与停滞/无进展裁剪升降，防抖动。认领服从 K；BFS 边代价不变。",
        ea,
    )

    heading(doc, "4.3 StageNet", 2, ea)
    body(
        doc,
        "候选含 FINAL（目标格）、沿几何路径约 1/3 与 2/3 的 APPROACH，以及 CLEAR（附近低停留格）。"
        "智体特征 x_a ∈ ℝ⁸。MLP（8→64→64）含打分/优先级头。训练为模仿学习。ASA 在网络噪声大的"
        "体制（如多数 n ≤ 9）切回手工教师。",
        ea,
    )

    heading(doc, "4.4 协议套件", 2, ea)
    body(
        doc,
        "各工具对应失败模式：竞争目的地门防同垫对冲；垫区出口与 FG-CEBI 强制先出后进；E4 粘性"
        "出口清卸货后 deg ≤ 2 占位；进度脉冲在无进展时短暂降 K、加宽 W；PGFA 帮单一焦点防互让"
        "活锁；DLO 降低合法化哈希序抖动；区域准入软限制每瓦片 FINAL 密度。走廊向子集统称 PGCP。",
        ea,
    )

    add_figure(doc, FIG / "fig1_overview_zh.jpg")
    caption(doc, "图 1. Meeting-RH 栈：几何门、会商、拍执行与 FACN 对 K 的反馈。", ea)
    add_figure(doc, FIG / "fig2_meeting_zh.jpg")
    caption(doc, "图 2. 会商生命周期：预跟进 → 触发 → 输入 → 内部步骤 → 契约 → 返回。", ea)

    heading(doc, "5 实验", 1, ea)
    heading(doc, "5.1 设置", 2, ea)
    body(
        doc,
        "SH01–20，官方约 100 任务流。基线：PIBT、M0、PP、窗口 ECBS、Hier。本文：Meeting-RH + "
        "协议 + StageNet（ASA）+ FACN（AFA）。复现：PYTHONHASHSEED=0、CPU、每图子进程隔离。"
        "主对比要求 VALID 轨迹。",
        ea,
    )

    heading(doc, "5.2 主结果", 2, ea)
    p = doc.add_paragraph()
    set_paragraph_format(p, align=WD_ALIGN_PARAGRAPH.CENTER, space_before=6, space_after=4,
                         first_indent=0)
    add_text(
        p,
        "表 1. SH01–20（官方 100）汇总。VALID ∧ full100 方法中 sim/wall 越低越好。"
        "† 未全完成图上的 sim 不可作“更快”解读。",
        size=10, italic=True, east_asia=ea,
    )
    add_table(
        doc,
        ["方法", "full100", "VALID", "avg_done", "avg_sim", "avg_wall"],
        [
            ["PIBT", "8/20", "20/20", "63.9", "62811", "40.5s"],
            ["M0", "4/20", "20/20", "44.6", "942†", "1478s"],
            ["PP", "1/20", "7/20", "32.4", "848†", "1716s"],
            ["ECBS", "20/20", "20/20", "100", "7364", "543s"],
            ["Hier", "20/20", "20/20", "100", "4018", "1032s"],
            ["本文", "20/20", "20/20", "100", "1708", "~28s"],
        ],
        ea,
    )
    body(
        doc,
        "讨论。PIBT/M0/PP 多图未完成；低 sim 非成功。在 20/20 全完成方法中，本文 avg_sim 约为 "
        "Hier 的 42%、ECBS 的 23%，墙钟约 28s 对数百至数千秒。加速来自：（i）会商稀疏；"
        "（ii）有界视窗时空 BFS 相对反复全目标时空 A*；（iii）并发限制；（iv）死锁诱导的重规划更少。"
        "相对旧 FULL 栈，仅协议改进约 −9.0% avg_sim 且 0 图变差。FACN v2 相对 noFACN（SH01–40）"
        "约 −5.82% avg_sim（20/13/5 更好/持平/更差），GATE 图通过。",
        ea,
    )

    heading(doc, "5.3 消融", 2, ea)
    body(
        doc,
        "相对完整栈：关 StageNet 难图崩溃（如 SH10 done 60，SH17 done 32）。关 PGCP 走廊工具则 "
        "SH10/11/12/17 完成率下降。关 PGFA 可毁 SH17（done 20）并抬高 SH03 sim。关 FACN 伤害"
        "开阔大车队（如 SH20 sim 660→813）。默认栈不用全局方向场网络。",
        ea,
    )
    add_figure(doc, FIG / "fig3_stagenet_zh.jpg", width=5.2)
    caption(doc, "图 3. StageNet 输入输出（8 维智体状态；ASA 为外部门）。", ea)
    add_figure(doc, FIG / "fig4_facn_zh.jpg", width=5.2)
    caption(doc, "图 4. FACN 输入输出：静态 g 与动态 x → Δ → K。", ea)

    heading(doc, "5.4 为何窗口 BFS 优于串行 M0 A*", 2, ea)
    body(
        doc,
        "二者虽都触及搜索，但 M0 在分配时反复运行相对稠密预约表的全视界时空 A*。Meeting-RH "
        "多数时间契约跟进；会商仅在 W 内调用优先序时空 BFS。调用频率、搜索深度与冲突抖动均下降。",
        ea,
    )

    heading(doc, "6 局限与效度威胁", 1, ea)
    body(
        doc,
        "Meeting-RH 不完备，病理地图可能失败。FACN 跨地图族的留出泛化仍有限。协议需维护并可能"
        "相互作用。压力图非完整工业数字孪生。结果固定种子与 CPU；GPU 仅离线训练。",
        ea,
    )

    heading(doc, "7 结论", 1, ea)
    body(
        doc,
        "本文提出 Meeting-RH：面向终身 MAPD 的可解释混合方法，将几何门控协议与两个小网络——"
        "StageNet（阶段/优先级）与 FACN（运营并发 K）——耦合。实证上，该方法在有效性与完成率上"
        "对标强搜索基线，同时显著改善仿真时间与墙钟。未来工作包括更强的留出并发学习、与 anytime "
        "窗口搜索的更紧集成[12]，以及更丰富的运动学模型[10]。",
        ea,
    )

    heading(doc, "附录 A FACN 特征表", 1, ea)
    body(
        doc,
        "静态 g ∈ ℝ¹⁰：n、自由格、near_pad_deg2、取货点数、卸货点数、密度、空闲池比、K_TBC、"
        "区域自由均值/最大。动态 x ∈ ℝ¹²：忙碌、空闲、剩余、完成率、停滞、平均提交 H、修复次数、"
        "冲突热度、区域占用、目的地门数、同目的地簇、喉占用。",
        ea,
    )
    heading(doc, "附录 B 可复现性", 1, ea)
    body(
        doc,
        "入口：python -m greed_coop_mcts，policy=rrsc，可选 FACN 权重 facn_v2.pt、StageNet "
        "stage_selector_v1.pt。会商常数：MEET_H=12，MEET_W ∈ {16, 28}。",
        ea,
    )

    add_refs(doc, REFS_EN, ea)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "MeetingRH_Paper_ZH.docx"
    doc.save(path)
    print(f"Wrote {path}")
    return path


if __name__ == "__main__":
    build_en()
    build_zh()
