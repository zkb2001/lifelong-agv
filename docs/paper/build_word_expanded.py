#!/usr/bin/env python3
"""Build ~16-page EN/ZH academic Word manuscripts with detailed Method + ~60 refs."""
from __future__ import annotations

from pathlib import Path

from refs60 import REFS
from word_fmt import H, P, Pb, algo, cap, fig, pf, refs_block, setup_doc, add, table
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "word"


def title_block(doc, title, authors, affil, ea):
    p = doc.add_paragraph()
    pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=12, first=0)
    add(p, title, size=16, bold=True, east_asia=ea)
    p = doc.add_paragraph()
    pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=4, first=0)
    add(p, authors, size=12, east_asia=ea)
    p = doc.add_paragraph()
    pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=0, after=16, first=0)
    add(p, affil, size=11, italic=True, east_asia=ea)


def build_en():
    doc = setup_doc(False)
    ea = "Times New Roman"
    title_block(
        doc,
        "Explainable Fleet Concurrency Control for Lifelong Multi-Agent "
        "Pickup and Delivery via Meeting-based Hybrid Planning",
        "Anonymous Authors",
        "Affiliation withheld for anonymous review",
        ea,
    )

    # Abstract
    p = doc.add_paragraph()
    pf(p, first=0, before=4, after=8)
    add(p, "Abstract—", bold=True, italic=True, east_asia=ea)
    add(
        p,
        "Lifelong multi-agent pickup and delivery (MAPD) requires continuously assigning and "
        "executing pickup–delivery tasks under vertex and swap conflicts, station FIFO, and "
        "limited online compute. Search-based windowed solvers such as RHCR with ECBS achieve "
        "high completion but pay large wall-clock cost; reactive solvers such as PIBT are fast "
        "but brittle in capacity-1 corridors. We present Meeting-RH, a dual-timescale hybrid: "
        "slow meetings (period H=12, adaptive window W∈{16,28}) negotiate stage pose, priority, "
        "and short-horizon space-time commits; fast ticks follow StageContracts with gap yielding "
        "and focal altruism. Two small imitation networks, gated by static geometry, provide "
        "explainable decisions: StageNet scores FINAL / APPROACH@1/3 / APPROACH@2/3 / CLEAR and "
        "a priority scalar under Adaptive Stage Authority (ASA); FACN predicts a discrete residual "
        "Δ∈{−2,…,+4} that adjusts throat-band capacity K=clip(K_TBC+Δ) without changing path costs. "
        "A geometry-gated protocol suite (contested-destination gates, pad egress, fleet-gated CEBI, "
        "E4 sticky egress, progress pulses, PGFA, DLO, region admit) maps one-to-one onto reproduced "
        "failure modes. On SH01–20 (~100 lifelong tasks/map), Meeting-RH matches ECBS/Hier on VALID "
        "and full completion while cutting average simulation time to ~42% of hierarchical ECBS and "
        "wall-clock by 1–2 orders of magnitude. Ablations isolate StageNet/protocols for hard-map "
        "completion and FACN for open large-fleet throughput.",
        east_asia=ea,
    )
    p = doc.add_paragraph()
    pf(p, first=0, after=12)
    add(p, "Index Terms—", bold=True, italic=True, east_asia=ea)
    add(
        p,
        "Lifelong MAPD, multi-agent path finding, rolling horizon, imitation learning, "
        "fleet concurrency, explainable coordination, warehouse robotics.",
        east_asia=ea,
    )

    # ===== 1 =====
    H(doc, "I. Introduction", 1, ea)
    P(
        doc,
        "Multi-Agent Path Finding (MAPF) asks for collision-free paths for multiple agents on a "
        "shared graph [1], [39], [40]. Conflict-Based Search (CBS) and its improvements form the "
        "backbone of optimal solvers [1], [3]–[6], [8]–[10], while ECBS/EECBS provide bounded "
        "suboptimality via focal search [2], [7], [11]. Single-agent building blocks include A* "
        "[12] and safe-interval planners [16], [17]. Industrial warehouses, however, typically "
        "require the lifelong variant: agents repeatedly receive new goals as tasks stream in "
        "[31]–[35], [44], [45]. In lifelong MAPD, planners must jointly handle assignment progress, "
        "corridor capacity, kinematic constraints, and limited online compute [34], [51].",
        ea,
    )
    P(
        doc,
        "Two families dominate practice. Search-based lifelong wrappers—especially Rolling-Horizon "
        "Collision Resolution (RHCR) with CBS/ECBS/PBS inside a window—offer strong validity but "
        "scale poorly in wall-clock when many agents replan together [35], [23], [36]. Reactive "
        "methods such as PIBT decide one step at a time with priority inheritance and backtracking "
        "[26], [27]; they scale to large fleets but degrade in narrow throats and dead-ends. Recent "
        "scalable search (LaCAM*) [28]–[30], large-neighborhood repair [37], [38], guidance graphs "
        "[53], and competition systems [52] improve the Pareto frontier, yet an operational question "
        "remains underexplored: given a fixed fleet, how many agents should remain busy, and in which "
        "stage pose, under heterogeneous geometry?",
        ea,
    )
    P(
        doc,
        "This paper targets that missing axis. We treat operational concurrency K and stage pose as "
        "slow, explainable decisions amenable to small supervised models and geometry gates, while "
        "keeping fast collision repair algorithmic. The resulting Meeting-RH stack is intentionally "
        "incomplete relative to CBS: meetings use priority-ordered windowed space-time BFS rather "
        "than joint optimal search. Gains are practical Pareto improvements inside the VALID ∧ "
        "full-completion regime.",
        ea,
    )
    Pb(
        doc,
        "Contributions. ",
        "(1) Dual-timescale Meeting-RH with explicit triggers, adaptive W, and StageContracts. "
        "(2) FACN: a 22-D MLP predicting Δ for K=clip(K_TBC+Δ), with Adaptive FACN Authority (AFA). "
        "(3) StageNet + ASA for FINAL/APPROACH/CLEAR and priority. "
        "(4) A protocol suite with geometry gates mapped to failure modes. "
        "(5) SH01–20 evidence: ECBS/Hier-level completion at much lower sim and wall-clock, with "
        "ablations isolating each component.",
        ea,
    )

    # ===== 2 =====
    H(doc, "II. Related Work", 1, ea)
    H(doc, "A. Optimal and Bounded-Suboptimal MAPF", 2, ea)
    P(
        doc,
        "CBS splits conflicts at a high level and replans under constraints [1]. ICBS, heuristic "
        "CBS, disjoint splitting, pairwise symmetry reasoning, and lazy clause generation "
        "substantially prune the high-level tree [3]–[6], [8]. Meta-agent CBS and iterative-deepening "
        "variants further improve search order [9], [10]. ECBS/EECBS replace best-first search with "
        "focal search under a user suboptimality bound [2], [7], [11]. Subdimensional expansion (M*) "
        "[14], independence detection [13], and enhanced partial expansion [15] provide alternative "
        "optimal/near-optimal families. Continuous-time and robust execution variants address "
        "timing uncertainty [41], [42]. Compilation and mutex encodings connect MAPF to SAT/CSP [43]. "
        "Complexity results clarify intractability on graphs [40].",
        ea,
    )
    H(doc, "B. Prioritized, Windowed, and Reactive Planning", 2, ea)
    P(
        doc,
        "Classical prioritized planning assigns an order and plans sequentially against reservations "
        "[18]–[22]. WHCA* commits only inside a window [18]. PBS searches priority orderings with "
        "consistency [23]. Rule-based complete algorithms such as Push-and-Swap / Push-and-Rotate "
        "guarantee solvability on certain topologies [24], [25]. PIBT and its journal extension "
        "provide iterative lifelong coordination with strong scalability [26], [27]. LaCAM / LaCAM* "
        "target real-time large-scale near-optimal MAPF [28]–[30]. RHCR decomposes lifelong MAPF "
        "into windowed instances and replans every h steps [35]; experience warming accelerates "
        "successive queries [36]. Anytime LNS repair improves quality under time budgets [37], [38].",
        ea,
    )
    H(doc, "C. Lifelong MAPD, Learning, and Congestion Control", 2, ea)
    P(
        doc,
        "Lifelong pickup-and-delivery couples assignment and routing [31]–[34], [56]–[58]. Warehouse "
        "deployments motivate persistent execution and large fleets [44]–[47]. Learning-augmented "
        "methods include PRIMAL/PRIMAL2 and decentralized follow policies [48]–[50]. Guidance-graph "
        "optimization and traffic-aware lifelong planners adapt costs to congestion [53], [54]. "
        "Corridor reasoning highlights geometry-specific bottlenecks [55]. Competition analyses "
        "emphasize concurrency, myopia, and compute limits [51], [52]. Relative to this literature, "
        "Meeting-RH keeps learning small and gated: StageNet decides pose/priority; FACN decides "
        "only how-many-busy; protocols remain auditable rules.",
        ea,
    )

    # ===== 3 =====
    H(doc, "III. Problem Formulation", 1, ea)
    P(
        doc,
        "Let G=(V,E) be a 4-connected grid with blocked cells. Fleet A={a_i}_{i=1}^n occupies one "
        "cell per discrete time. Actions are wait, move to a 4-neighbor, or rotate in place by "
        "±90°/180° (each costing one tick in our motion model). Vertex conflicts (two agents in "
        "the same cell at the same time) and edge-swap conflicts are forbidden [39]. Pickup "
        "stations enforce FIFO and at most one pickup per station per tick. Each lifelong task "
        "is a pickup–delivery pair; agents may be idle, assigned-to-pick, or loaded-to-drop.",
        ea,
    )
    P(
        doc,
        "A trajectory is VALID iff kinematic, capacity, and conflict constraints hold. Metrics: "
        "done (correct deliveries), full100 (all tasks done), sim (finish time of the last "
        "delivery), and wall (planner runtime). Stress maps SH01–20 have heterogeneous n and "
        "throat geometry, each with an official stream of ~100 tasks. We report geometry proxies "
        "such as near_pad_deg2: the number of free cells with graph degree ≤2 inside radius 6 of "
        "drop pads—used only as a gate feature, never as a map identifier.",
        ea,
    )

    # ===== 4 METHOD - the big section =====
    H(doc, "IV. Method: Meeting-RH Hybrid Stack", 1, ea)
    H(doc, "A. Architecture and Design Principles", 2, ea)
    P(
        doc,
        "Meeting-RH separates variables by timescale (Fig. 1). Slow variables—stage pose, priority, "
        "window commits, and fleet concurrency K—are updated in meetings or on gated events. Fast "
        "variables—next-cell follow, gap yield, and local protocol rewrites—run every tick. Learning "
        "is restricted to two compact MLPs with external geometry authorities (ASA for StageNet, "
        "AFA for FACN). Path edge costs are never modified by FACN; concurrency only gates claiming. "
        "Protocols are geometry-gated tools mapped to failure modes, not map-id hardcoding.",
        ea,
    )
    fig(doc, FIG / "fig1_overview.jpg")
    cap(
        doc,
        "Fig. 1. Meeting-RH stack: geometry gates (ASA/TBC/AFA), meetings, tick execution, and FACN feedback on K.",
        ea,
    )
    P(
        doc,
        "Orthogonality is intentional: StageNet answers where an agent should head (pose); FACN "
        "answers how many agents may remain busy (throttle); PGCP-style protocols answer local "
        "corridor traffic rules; windowed prioritized BFS answers short-horizon commits. None of "
        "these layers claims CBS completeness [1].",
        ea,
    )

    H(doc, "B. Meeting Triggers and Lifecycle", 2, ea)
    P(
        doc,
        "Constants: meeting period H=MEET_H=12; base window W=16; adaptive window W_ADAPT=28; "
        "stall-force threshold MEET_STALL_FORCE=24; StageContract TTL=H+4=16. A meeting is triggered "
        "when (i) t=0 or no prior meeting; (ii) t−t_last≥H; or (iii) stall≥24 and the gap since the "
        "last meeting is at least max(8,H/2)=8. Progress-based extensions force meetings when "
        "no_task_progress holds: if n≥12 (dense) and gap≥4; or if not dense but stall≥40 and gap≥8. "
        "When no_progress co-occurs with dense fleets or stall≥40, adaptive_meet_w may raise W from "
        "16 to 28; another widening rule uses stall≥60 with average committed horizon <1.5.",
        ea,
    )
    fig(doc, FIG / "fig2_meeting.jpg")
    cap(
        doc,
        "Fig. 2. Meeting lifecycle: pre-follow → trigger → inputs → internal pipeline → contracts → return to ticks.",
        ea,
    )
    P(
        doc,
        "Inside run_meeting the pipeline is fixed and auditable: (1) build BFS path_cache; "
        "(2) select stages/priorities via StageNet or teacher under ASA; (3) apply_region_admit "
        "(protecting TO_DROP roles); (4) rewrite_contested_destinations; (5) apply_pad_egress "
        "including FG-CEBI / E4 when gates open; (6) plan_committed with priority-ordered "
        "space-time BFS over horizon W; (7) window_repair; (8) legalize_desired with Dual-Lex "
        "Order (DLO) when 10≤n≤14; (9) issue StageContracts with waypoints=commit prefix (≤6 "
        "cells), goal, role, priority, and TTL=16.",
        ea,
    )
    algo(
        doc,
        "Algorithm 1 Meeting-RH tick loop (conceptual).",
        [
            "1: Initialize contracts; t_last ← −H; K ← K_TBC",
            "2: for t = 0,1,2,… do",
            "3:   Update stall / no_prog / geometry proxies (near, throat_occ, …)",
            "4:   Possibly refresh FACN: K ← AFA(clip(K_TBC + Δ̂))",
            "5:   if should_meet(t) then",
            "6:       stages, pri ← ASA(StageNet, teacher)",
            "7:       goals ← region_admit ∘ contested ∘ pad_egress(goals)",
            "8:       commits ← spacetime_BFS_PP(goals, pri, W)",
            "9:       contracts ← StageContract(commits; TTL=16); t_last ← t",
            "10:  Follow contract heads; gap_yield; PGFA if gated; tick protocols",
            "11: end for",
        ],
        ea,
    )

    H(doc, "C. Windowed Space-Time BFS and StageContracts", 2, ea)
    P(
        doc,
        "Unlike serial full-horizon space-time A* used by our M0 baseline, meeting planning calls "
        "priority-ordered spacetime_bfs inside W only [12], [18]. State is (x,y,t); expansions "
        "include wait and 4-neighbors; reservations forbid occupied verts[t] and swap edges[t]. "
        "A BFS distance field to the stage goal provides guidance; if the goal is unreachable in "
        "W, the planner keeps the best progressive state, else a one-step greedy fallback. Agents "
        "are planned in order (−priority, name), each adding reservations for later agents—i.e., "
        "windowed prioritized planning, not joint CBS [18], [23], [35].",
        ea,
    )
    P(
        doc,
        "A StageContract is a typed message (agent, role∈{TO_PICK,TO_DROP,YIELD,HOLD,CLEAR,IDLE}, "
        "waypoints, goal, priority, ttl, exclusive_pad?, issued_t). Between meetings, agents follow "
        "commit heads or waypoints. Contracts expire by TTL or are replaced at the next meeting. "
        "This is the operational analog of WHCA*-style short commits [18], but coupled to stage "
        "pose and protocol rewrites rather than raw goals alone.",
        ea,
    )

    H(doc, "D. StageNet and Adaptive Stage Authority (ASA)", 2, ea)
    P(
        doc,
        "Stage candidates per agent: FINAL (true goal cell); APPROACH at ~1/3 and ~2/3 along a "
        "geometric path toward the goal; CLEAR, chosen on Manhattan rings of radius 1→2→3 by "
        "minimizing stay + 0.05·manh(cand,final). StageNet is a compact MLP: agent encoder "
        "8→64→64 (GELU); candidate scorer on [c⊕h]; priority head 64→32→1. Runtime checkpoint: "
        "stage_selector_v1.pt.",
        ea,
    )
    Pb(
        doc,
        "Agent features x_a∈ℝ⁸: ",
        "(1–2) normalized current xy; (3–4) normalized final xy; (5) manh/(2·GRID); "
        "(6) load status (loaded=1, assigned=0.5, idle=0); (7) min(blocked_streak/40,1); "
        "(8) min(region_occ/6,1)+0.25·heat. Candidate features (10-D) encode cell xy, distances, "
        "kind one-hots, stay, occupancy, and a soft edge/region cost. Runtime priority floors: "
        "loaded → 10⁶+p; assigned → 10⁵+p; else 10+p, preserving task urgency over network noise.",
        ea,
    )
    Pb(
        doc,
        "Teacher (imitation labels / fallback): ",
        "if d≤5 or streak≥18 → FINAL; non-congested or loaded → FINAL (loaded never CLEAR); "
        "congested with streak≥12 → CLEAR; else prefer APPROACH@2/3 then @1/3. Teacher priority "
        "adds heat/distance/streak bonuses on top of role floors. Training uses stage CE, listwise "
        "priority, and a fleet regularizer (synthetic meetings).",
        ea,
    )
    Pb(
        doc,
        "ASA gate (open once from (n, near_pad_deg2)): ",
        "use StageNet if (n≤9 ∧ near≥90) or (n=9 ∧ near∈[50,59]) or (10≤n≤14) or (n=15 ∧ near<60) "
        "or (n>18 except n=18). Otherwise use teacher—especially most n≤9 maps and n=18. ASA "
        "prevents the net from being applied in regimes where imitation noise harms completion.",
        ea,
    )
    fig(doc, FIG / "fig3_stagenet.jpg", width=5.2)
    cap(doc, "Fig. 3. StageNet I/O. ASA is an external geometry gate, not a network input.", ea)

    H(doc, "E. FACN, Multi-Band TBC, and AFA", 2, ea)
    P(
        doc,
        "Multi-Band Throat-Band Capacity (TBC) sets a static baseline K_TBC from (n, near). Default "
        "base=6; if n≥28 then K_TBC≥7; if 10≤n≤14 and near∈[55,85] then K_TBC=5; if 16≤n≤18 and "
        "near∈[80,90] then K_TBC=5. FACN does not replace TBC; it predicts a residual.",
        ea,
    )
    Pb(
        doc,
        "FACN features (22-D). ",
        "Static g∈ℝ¹⁰ (normalized): n/30, free/400, near/160, #pickup/12, #dropoff/12, dens/0.15, "
        "idle_pool=(n−K_TBC)/n, K_TBC/16, region_free mean/50, max/80. Dynamic x∈ℝ¹²: busy/16, "
        "idle/30, remaining/100, done_rate/0.25, stall/80, avg_commit_H/28, repairs/200, "
        "conflict_heat, region_occ/4, n_dest_gates/20, same_dest_cluster/8, throat_occ/8. "
        "MLP 22→64→64→7 classifies Δ∈{−2,−1,0,1,2,3,4}. Then K=clip(K_TBC+Δ) with "
        "K∈[2, min(16,n)], soft floors when remaining>5, sticky TTL=36 / period=24 / min_gap=12, "
        "and per-step Δ change at most ±1. FACN never selects which agent is busy—only how many "
        "may claim work.",
        ea,
    )
    Pb(
        doc,
        "AFA (Adaptive FACN Authority): ",
        "attach only if n>K_TBC (idle pool exists). Raise caps: near≤40 → up to +4; ≤70 → +2; else "
        "+1; small fleets n≤12 default raise 0 unless oracle Δ*>0. Lower only if "
        "(throat_occ≥4 ∨ same_dest≥4) ∧ (stall≥40 ∨ no_prog). Δ=−2 only on non-flood maps "
        "(near≤80) with throat≥5 ∧ same_dest≥4; flood-like maps forbid −2. Event refresh when "
        "no_prog with throat/same-dest pressure and gap≥12, or stall≥40 at busy saturation. "
        "During a progress pulse, effective K←min(K,3).",
        ea,
    )
    fig(doc, FIG / "fig4_facn.jpg", width=5.2)
    cap(doc, "Fig. 4. FACN I/O: static g and dynamic x → Δ → K, clipped by AFA and pulse.", ea)

    H(doc, "F. Geometry-Gated Protocol Suite", 2, ea)
    P(
        doc,
        "Protocols are first-class “tools,” each tied to a reproduced failure. They run every tick "
        "and/or inside meetings. We detail thresholds so the stack is reproducible and auditable.",
        ea,
    )

    H(doc, "1) Contested-Destination Gates", 3, ea)
    P(
        doc,
        "Failure mode: two or more loaded agents head-on into the same drop pad / hub. Trigger: "
        "≥2 loaded agents share a station hub and ≥2 are within CONTEND_NEAR=6 Manhattan of any "
        "pad. A holder (nearest to pad) receives priority +3·10⁵ and a lock TTL=48 (GATE_NO_PROG); "
        "holder streak≥48 allows takeover. Non-holders are rewritten to the nearest degree≥3 cell, "
        "forbidding pads, deg≤2 corridors, and the last CONTEND_K=4 cells of others’ paths, with "
        "PAD_EXCLUSION=3. Frequency: every tick and every meeting.",
        ea,
    )

    H(doc, "2) Pad Egress and Fleet-Gated CEBI (FG-CEBI)", 3, ea)
    P(
        doc,
        "Failure mode: egress–ingress deadlock at unload pads. Empty agents on drop cells (or "
        "pressured siblings/corridors) are rewritten to deg≥3 cells with EGRESS_PRI=5·10⁷. Soft "
        "leave applies just after unload; stale progress can also eject neighbors occupying pads. "
        "FG-CEBI / lane-lock opens only when n≥10 and near≥100: if an egressor occupies a corridor "
        "and a loaded agent’s shortest path to the pad crosses that corridor (Manhattan triangle "
        "equality), the loaded target is rewritten off-corridor. Mild throats and n<10 do not "
        "enable this gate—avoiding over-triggering.",
        ea,
    )

    H(doc, "3) E4 Sticky Post-Unload Egress", 3, ea)
    P(
        doc,
        "Failure mode: after unload, an agent remains on a deg≤2 cell and blocks the next ingress. "
        "Gate: n≥19 and near∈[70,85]. The stack remembers last_unload_pad; if the agent still "
        "resides on deg≤2, forced egress applies without waiting for ingress pressure (T_EGRESS=16 "
        "as design constant; residence on the throat is the practical sticky signal).",
        ea,
    )

    H(doc, "4) Progress Pulse", 3, ea)
    P(
        doc,
        "Failure mode: long no-progress under contested loads. Define no_prog when contested≥1, "
        "loaded≥2, and (done_stale≥120 or manh_stale≥70), with tighter thresholds 80/50 when n≥16. "
        "For n≥12, a busy pulse of TTL=28 forces effective K←min(base,3), meetings become more "
        "frequent, and W may widen to 28. Pulse is a temporary throttle, not a permanent K policy.",
        ea,
    )

    H(doc, "5) PGFA: Progress-Gated Focal Altruism", 3, ea)
    P(
        doc,
        "Failure mode: mutual-yield livelock where everyone politely waits. Trigger when "
        "remaining≤5 or stall≥25 or any loaded streak≥12. Exactly one focal beneficiary is chosen; "
        "others clear a BFS prefix of length PGFA_PATH_PREFIX=24 for that agent. Switch focus after "
        "≥50 ticks without progress; near-goal (manh≤2) sticky up to 120. Pad eviction uses manh≤3; "
        "park_freeze=24. Yield gate: idle agents yield for the focus; non-focus loaded agents do "
        "not yield to each other—breaking symmetric standoffs.",
        ea,
    )

    H(doc, "6) Gap Yield, DLO, Region Admit, Idle Hygiene", 3, ea)
    P(
        doc,
        "Gap yield (inter-meeting): if a low-priority agent occupies the next cell of a "
        "higher-priority agent, it side-steps without inducing a swap. DLO legalization "
        "(meetings, 10≤n≤14, loaded≥2) compares orders (−pri,name) vs (−pri,manh,name) and keeps "
        "the one with smaller loaded ΣManhattan, while stripping swap edges and illegal cells—"
        "reducing hash-order jitter. Region admit (5×5 tiles) soft-caps FINAL density to "
        "clamp(free/8, 2, 5); excess non-protected FINALs become CLEAR/YIELD toward path-near "
        "cells; TO_DROP is hard-protected in meetings. Idle corridor hygiene: idle agents sitting "
        "on loaded deg≤2 corridors for IDLE_CORRIDOR_N=24 ticks are pushed to deg≥3.",
        ea,
    )

    # summary table of protocols
    p = doc.add_paragraph()
    pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4, first=0)
    add(
        p,
        "TABLE II. Protocol → failure mode → primary gate (summary).",
        size=10, italic=True, east_asia=ea,
    )
    table(
        doc,
        ["Protocol", "Failure mode", "Primary gate / threshold"],
        [
            ["Contested-dest", "Same-pad head-on", "loaded≥2, near≤6, TTL=48"],
            ["Pad-egress", "Egress/ingress block", "empty on pad / pressure"],
            ["FG-CEBI", "Lane lock on path", "n≥10 ∧ near≥100"],
            ["E4 sticky", "Post-unload deg≤2 stay", "n≥19 ∧ near∈[70,85]"],
            ["Progress-pulse", "Long no-progress", "n≥12, K←min(K,3), TTL=28"],
            ["PGFA", "Mutual-yield livelock", "stall≥25 / streak≥12; 1 focus"],
            ["DLO", "Hash-order jitter", "10≤n≤14, loaded≥2"],
            ["Region admit", "FINAL overcrowding", "5×5 tiles, cap 2..5"],
        ],
        ea,
    )

    H(doc, "G. Completeness and Non-Claims", 2, ea)
    P(
        doc,
        "Meeting-RH is incomplete: bounded W, prioritized BFS, and heuristic contracts can fail on "
        "pathological instances where CBS would eventually succeed [1], [40]. We do not claim "
        "optimality, bounded suboptimality, or that global direction-field networks help—historical "
        "tests did not improve done/sim in our stack, so they are excluded from the default "
        "contribution claim. Classical AI planning background is assumed [60].",
        ea,
    )

    # ===== 5 =====
    H(doc, "V. Experiments", 1, ea)
    H(doc, "A. Setup and Baselines", 2, ea)
    P(
        doc,
        "We evaluate SH01–20 with official ~100-task streams. Baselines: PIBT [27]; M0 (full-horizon "
        "space-time A* + greedy assignment); PP (priority task allocation + space-time A*); windowed "
        "ECBS [2], [35]; Hier (M0 on easy maps, ECBS handoff on hard maps). Ours: Meeting-RH + "
        "protocol suite + StageNet(ASA) + FACN(AFA). Reproduction: PYTHONHASHSEED=0, CPU, per-map "
        "subprocess isolation. Primary comparisons require VALID trajectories.",
        ea,
    )

    p = doc.add_paragraph()
    pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4, first=0)
    add(
        p,
        "TABLE I. Aggregate results on SH01–20 (official 100). Lower sim/wall is better among "
        "VALID∧full100. † sim on incomplete maps is not comparable as “faster.”",
        size=10, italic=True, east_asia=ea,
    )
    table(
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
    P(
        doc,
        "Among methods with 20/20 full100, Ours reduces avg_sim to ~42% of Hier and ~23% of ECBS, "
        "with wall-clock ~28s versus hundreds–thousands of seconds. PIBT/M0/PP leave many maps "
        "incomplete; their low sim values are not successes. Protocol-only improvements versus a "
        "legacy FULL stack yield about −9.0% avg_sim with 0 worse maps. FACN v2 versus noFACN on "
        "SH01–40 yields about −5.82% avg_sim (20/13/5 better/same/worse) with GATE maps PASS.",
        ea,
    )

    H(doc, "B. Ablations and Analysis", 2, ea)
    P(
        doc,
        "Disabling StageNet collapses hard maps (SH10 done 60; SH17 done 32). Disabling PGCP "
        "corridor tools drops completion on SH10/11/12/17. Disabling PGFA can destroy SH17 "
        "(done 20) and inflate SH03 sim. Disabling FACN hurts open large fleets (SH20 sim 660→813). "
        "These ablations support the claimed orthogonality: pose learning and corridor rules "
        "dominate completion; concurrency residual dominates open-map throughput; meetings convert "
        "slow decisions into cheap follow, explaining wall-clock.",
        ea,
    )
    p = doc.add_paragraph()
    pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4, first=0)
    add(
        p,
        "TABLE III. Ablation highlights (representative maps; lower done is worse).",
        size=10, italic=True, east_asia=ea,
    )
    table(
        doc,
        ["Setting", "SH10 done", "SH17 done", "SH20 sim", "Notes"],
        [
            ["Full (Ours)", "100", "100", "660", "default stack"],
            ["no StageNet", "60", "32", "—", "pose collapse on hard"],
            ["no PGCP", "↓", "↓", "—", "SH10/11/12/17 hurt"],
            ["no PGFA", "—", "20", "—", "mutual-yield failure"],
            ["no FACN", "100", "100", "813", "open-fleet regress"],
        ],
        ea,
    )
    P(
        doc,
        "Why windowed BFS beats serial M0 A*: M0 repeatedly runs full-horizon space-time A* against "
        "a dense reservation table at assignment time. Meeting-RH mostly follows contracts; meetings "
        "call priority-ordered BFS only inside W, cutting call frequency, depth, and thrash—even "
        "though “search” remains in the stack [12], [18], [35]. Relative to RHCR-style joint windowed "
        "ECBS [35], meetings negotiate a compact slow state and rely on protocols for the failure "
        "modes that would otherwise explode high-level conflicts in corridors [55].",
        ea,
    )

    H(doc, "C. Training Details and Geometry Gates", 2, ea)
    P(
        doc,
        "StageNet is trained by imitation on synthetic meetings with teacher labels (stage "
        "cross-entropy, listwise priority, fleet regularizer). FACN is trained as 7-way "
        "classification over Δ with oracle K* inducing Δ*=K*−K_TBC (oracle fraction ≈0.55 in the "
        "training mixture). Importantly, both networks are deployed behind geometry authorities: "
        "ASA may refuse StageNet on small/noisy regimes; AFA may refuse harmful raises/lowers. "
        "This “small net + hard gate” pattern is deliberate for auditability in warehouse settings "
        "[44], [47], [51], and contrasts with end-to-end RL policies that entangle pose, routing, "
        "and concurrency [48]–[50].",
        ea,
    )
    P(
        doc,
        "Geometry proxies (near_pad_deg2, throat occupancy, same-destination clusters) are computed "
        "from the map and online state without map identifiers. Consequently, the same gate "
        "predicates transfer across SH maps that share throat statistics even when layouts differ—"
        "an operational requirement highlighted by scaling discussions [51] and competition "
        "benchmarks [52]. Corridor-specific reasoning in the literature [55] motivates our "
        "explicit PGCP tools, while guidance-graph work [53], [54] is complementary: one could "
        "learn edge costs on top of Meeting-RH without removing K/pose separation.",
        ea,
    )

    H(doc, "D. Discussion: What Each Layer Buys", 2, ea)
    P(
        doc,
        "Meetings buy temporal abstraction: expensive negotiation is amortized over H≈12 ticks of "
        "cheap follow, echoing WHCA*/RHCR horizons [18], [35] but with typed StageContracts. "
        "StageNet buys pose diversity—APPROACH/CLEAR prevent premature FINAL pile-ups that "
        "prioritized planning alone often creates [19]–[21], [23]. FACN buys an adjustable "
        "operational fleet size without retraining path costs, addressing concurrency as a first-class "
        "control [51]. Protocols buy explainable recovery for deadlocks that reactive PIBT cannot "
        "reliably escape in capacity-1 structures [26], [27]. Together they form a hybrid closer in "
        "spirit to “planner + rules + tiny learners” than to monolithic learned policies.",
        ea,
    )

    # ===== 6-7 =====
    H(doc, "VI. Limitations and Threats to Validity", 1, ea)
    P(
        doc,
        "Meeting-RH is incomplete and may fail on pathological maps. FACN held-out generalization "
        "across map families remains limited. Protocols require maintenance and can interact. Stress "
        "maps are not a full industrial digital twin [44], [51]. Results use fixed seeds and CPU; "
        "GPU is offline training only. We do not evaluate continuous-time robust execution [41], [42] "
        "or LaCAM*-style joint search [28]–[30] inside meetings—those are promising future hybrids.",
        ea,
    )

    H(doc, "VII. Conclusion", 1, ea)
    P(
        doc,
        "We presented Meeting-RH, an explainable hybrid for lifelong MAPD that couples geometry-gated "
        "protocols with StageNet (pose/priority) and FACN (operational concurrency K). Empirically, "
        "the stack matches strong search baselines on validity and completion while substantially "
        "improving simulation time and wall-clock. Future work includes stronger held-out concurrency "
        "learning, tighter integration with anytime windowed search / LNS [37], [38], guidance "
        "adaptation [53], and richer kinodynamic models [34], [51].",
        ea,
    )

    H(doc, "Appendix A. Constant Cheat-Sheet", 1, ea)
    P(
        doc,
        "MEET_H=12; MEET_W∈{16,28}; contract TTL=16; stall-force=24; CONTEND_NEAR=6; gate TTL=48; "
        "EGRESS_PRI=5e7; FG-CEBI: n≥10∧near≥100; E4: n≥19∧near∈[70,85]; pulse TTL=28 with K≤3; "
        "PGFA prefix=24; region tiles 5×5; FACN sticky TTL=36, period=24, min_gap=12; "
        "Δ∈{−2..+4}; K∈[2,min(16,n)].",
        ea,
        first=0,
    )
    H(doc, "Appendix B. Reproducibility", 1, ea)
    P(
        doc,
        "Entry: python -m greed_coop_mcts with policy=rrsc; checkpoints facn_v2.pt and "
        "stage_selector_v1.pt; PYTHONHASHSEED=0; per-map subprocesses.",
        ea,
        first=0,
    )

    refs_block(doc, REFS, "References", ea)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "MeetingRH_Paper_EN.docx"
    doc.save(path)
    print("Wrote", path)
    return path


def build_zh():
    doc = setup_doc(True)
    ea = "SimSun"
    title_block(
        doc,
        "面向终身多智体取送货的可解释车队并发控制：基于会商的混合规划",
        "匿名作者",
        "匿名评审，单位信息暂隐",
        ea,
    )

    p = doc.add_paragraph()
    pf(p, first=0, before=4, after=8)
    add(p, "摘要：", bold=True, east_asia=ea)
    add(
        p,
        "终身多智体取送货（Lifelong MAPD）要求在顶点/交换冲突、工位 FIFO 与有限在线算力下持续"
        "分配并执行取送货任务。RHCR+ECBS 等窗口搜索完成率高但墙钟昂贵；PIBT 等反应式方法快但在"
        "容量为 1 的走廊脆弱。本文提出 Meeting-RH 双时间尺度混合方法：慢会商（周期 H=12，自适应"
        "视窗 W∈{16,28}）协商阶段位姿、优先级与短视窗时空提交；快拍执行 StageContract，配合空隙"
        "让行与焦点利他。两个带几何门控的小型模仿网络提供可解释决策：StageNet 在 FINAL / "
        "APPROACH@1/3 / APPROACH@2/3 / CLEAR 间打分并输出优先级（ASA 门控）；FACN 预测离散残差 "
        "Δ∈{−2,…,+4}，将喉带容量调整为 K=clip(K_TBC+Δ) 且不改路径代价。几何门控协议套件"
        "（竞争目的地门、垫区出口、FG-CEBI、E4 粘性出口、进度脉冲、PGFA、DLO、区域准入等）与可复现"
        "失败模式一一对应。在 SH01–20（每图约 100 任务）上，本方法在 VALID 与全量完成上对标 "
        "ECBS/Hier，平均仿真时间约为分层 ECBS 的 42%，墙钟降低 1–2 个数量级。消融表明 StageNet/"
        "协议支撑难图完成，FACN 提升开阔大车队吞吐。",
        east_asia=ea,
    )
    p = doc.add_paragraph()
    pf(p, first=0, after=12)
    add(p, "关键词：", bold=True, east_asia=ea)
    add(p, "终身 MAPD；多智体路径规划；滚动视窗；模仿学习；车队并发；可解释协同；仓储机器人", east_asia=ea)

    H(doc, "1 引言", 1, ea)
    P(
        doc,
        "多智体路径规划（MAPF）要求共享图上无碰撞路径[1],[39],[40]。CBS 及其改进构成最优求解器"
        "主干[1],[3]–[6],[8]–[10]；ECBS/EECBS 通过焦点搜索提供有界次优[2],[7],[11]。单智体积木"
        "包括 A*[12] 与安全间隔规划[16],[17]。工业仓储通常需要终身变体：任务流到来时智体反复接收"
        "新目标[31]–[35],[44],[45]。终身 MAPD 需同时处理分配进度、走廊容量、运动学与在线算力"
        "[34],[51]。",
        ea,
    )
    P(
        doc,
        "实践中两大家族主导。基于搜索的终身封装——尤其 RHCR 在窗口内调用 CBS/ECBS/PBS——有效性强"
        "但多智体重规划墙钟高[35],[23],[36]。PIBT 等反应式方法逐步决策、可扩展，却在窄喉退化"
        "[26],[27]。LaCAM*[28]–[30]、LNS 修复[37],[38]、引导图[53]与竞赛系统[52]改善了帕累托前沿，"
        "但运营问题仍欠讨论：固定车队下，异构几何中应允许多少智体保持忙碌、并处于何种阶段位姿？",
        ea,
    )
    P(
        doc,
        "本文将运营并发度 K 与阶段位姿视为慢、可解释、适合小型监督模型与几何门的决策；快速碰撞"
        "修复保持算法化。Meeting-RH 相对 CBS 有意不完备：会商使用优先序窗口时空 BFS 而非联合最优"
        "搜索。收益是 VALID∧全量完成体制内的实用帕累托改进。",
        ea,
    )
    Pb(
        doc,
        "贡献：",
        "（1）双时间尺度 Meeting-RH：显式触发、自适应 W 与 StageContract。"
        "（2）FACN：22 维 MLP 预测 Δ，配合 AFA。"
        "（3）StageNet + ASA：阶段/优先级。"
        "（4）几何门控协议套件，对应失败模式。"
        "（5）SH01–20 实证与消融。",
        ea,
    )

    H(doc, "2 相关工作", 1, ea)
    H(doc, "2.1 最优与有界次优 MAPF", 2, ea)
    P(
        doc,
        "CBS 在高层拆分冲突[1]；ICBS、启发式 CBS、不相交分裂、成对对称破缺与惰性子句生成显著剪枝"
        "[3]–[6],[8]。Meta-agent CBS 与迭代加深变体改进搜索次序[9],[10]。ECBS/EECBS 用焦点搜索保证"
        "次优界[2],[7],[11]。M*[14]、独立检测[13]、增强部分扩展[15]提供替代族。连续时间与鲁棒执行"
        "应对时间不确定[41],[42]。编译/互斥编码连接 SAT/CSP[43]。复杂性结果阐明图上难解性[40]。",
        ea,
    )
    H(doc, "2.2 优先、窗口与反应式规划", 2, ea)
    P(
        doc,
        "经典优先规划按序相对预约表规划[18]–[22]；WHCA* 仅窗口内提交[18]；PBS 搜索一致优先级"
        "[23]。Push-and-Swap/Rotate 在特定拓扑完备[24],[25]。PIBT 提供强可扩展迭代终身协调"
        "[26],[27]。LaCAM/LaCAM* 面向实时大规模近优[28]–[30]。RHCR 将终身 MAPF 分解为窗口实例"
        "[35]；经验预热加速相继查询[36]；anytime LNS 修复提升时限下质量[37],[38]。",
        ea,
    )
    H(doc, "2.3 终身 MAPD、学习与拥堵控制", 2, ea)
    P(
        doc,
        "终身取送货耦合分配与路由[31]–[34],[56]–[58]。仓储部署强调持久执行与大车队[44]–[47]。"
        "学习增强方法包括 PRIMAL/PRIMAL2 与去中心跟随策略[48]–[50]。引导图与交通感知终身规划适应"
        "拥堵[53],[54]；走廊推理强调几何瓶颈[55]。竞赛分析突出并发、短视与算力[51],[52]。"
        "相对上述工作，Meeting-RH 保持学习小且门控：StageNet 管位姿/优先级，FACN 只管“多少忙碌”，"
        "协议保持可审计规则。",
        ea,
    )

    H(doc, "3 问题形式化", 1, ea)
    P(
        doc,
        "设 G=(V,E) 为含障碍的四连通网格。车队 A={a_i}_{i=1}^n 每拍占一格。动作含等待、四邻移动、"
        "原地 ±90°/180° 旋转（本模型各耗一拍）。禁止顶点冲突与边交换冲突[39]。取货站 FIFO，每站每拍"
        "至多一次取货。终身任务为取—送对；智体可空闲、前往取货或载货送货。轨迹 VALID 当且仅当"
        "运动学、容量与冲突约束均满足。指标：done、full100、sim、wall。压力图 SH01–20 异构 n 与"
        "喉道几何，每图约 100 任务。几何代理 near_pad_deg2（卸点半径 6 内 deg≤2 自由格数）仅作门控"
        "特征，不作地图 ID。",
        ea,
    )

    H(doc, "4 方法：Meeting-RH 混合栈", 1, ea)
    H(doc, "4.1 架构与设计原则", 2, ea)
    P(
        doc,
        "Meeting-RH 按时间尺度分离变量（图 1）。慢变量——阶段位姿、优先级、窗口提交与车队并发 K——"
        "在会商或门控事件中更新；快变量——跟进下一格、空隙让行与局部协议改写——每拍运行。学习限于"
        "两个紧凑 MLP，并配外部几何权威（ASA/AFA）。FACN 不改路径边代价，只门控认领。协议是对应"
        "失败模式的几何门控工具，而非地图硬编码。",
        ea,
    )
    fig(doc, FIG / "fig1_overview_zh.jpg")
    cap(doc, "图 1. Meeting-RH 栈：几何门（ASA/TBC/AFA）、会商、拍执行与 FACN 对 K 的反馈。", ea)
    P(
        doc,
        "正交分工：StageNet 回答“往哪走（位姿）”；FACN 回答“允许多少忙碌（油门）”；PGCP 类协议"
        "回答局部走廊交规；窗口优先 BFS 回答短视窗提交。任一层均不主张 CBS 完备性[1]。",
        ea,
    )

    H(doc, "4.2 会商触发与生命周期", 2, ea)
    P(
        doc,
        "常数：会商周期 H=12；基础视窗 W=16；自适应 W_ADAPT=28；停滞强制阈值 MEET_STALL_FORCE=24；"
        "StageContract TTL=H+4=16。触发条件：（i）t=0 或尚无会商；（ii）t−t_last≥H；"
        "（iii）stall≥24 且距上次会商间隔≥max(8,H/2)=8。进度扩展：当 no_task_progress 成立时，"
        "若 n≥12（稠密）且 gap≥4，或非稠密但 stall≥40 且 gap≥8，则强制开会。无进展且稠密或 "
        "stall≥40 时，adaptive_meet_w 可将 W 从 16 升至 28；另一加宽规则为 stall≥60 且平均已提交"
        "视界 <1.5。",
        ea,
    )
    fig(doc, FIG / "fig2_meeting_zh.jpg")
    cap(doc, "图 2. 会商生命周期：预跟进 → 触发 → 输入 → 内部流水线 → 契约 → 返回拍循环。", ea)
    P(
        doc,
        "run_meeting 内部流水线固定可审计：（1）构建 BFS path_cache；（2）ASA 下 StageNet/教师选"
        "阶段与优先级；（3）apply_region_admit（保护 TO_DROP）；（4）rewrite_contested_destinations；"
        "（5）apply_pad_egress（含 FG-CEBI/E4）；（6）plan_committed：优先序时空 BFS，视界 W；"
        "（7）window_repair；（8）10≤n≤14 时 DLO legalize_desired；（9）签发 StageContract"
        "（waypoints=commit 前缀≤6，TTL=16）。",
        ea,
    )
    algo(
        doc,
        "算法 1 Meeting-RH 拍循环（概念）。",
        [
            "1: 初始化契约；t_last ← −H；K ← K_TBC",
            "2: for t = 0,1,2,… do",
            "3:   更新 stall / no_prog / 几何代理（near, throat_occ,…）",
            "4:   可能刷新 FACN：K ← AFA(clip(K_TBC + Δ̂))",
            "5:   if should_meet(t) then",
            "6:       stages, pri ← ASA(StageNet, 教师)",
            "7:       goals ← region_admit ∘ contested ∘ pad_egress(goals)",
            "8:       commits ← spacetime_BFS_PP(goals, pri, W)",
            "9:       contracts ← StageContract(commits; TTL=16)；t_last ← t",
            "10:  跟进契约头；gap_yield；门控 PGFA；拍协议",
            "11: end for",
        ],
        ea,
    )

    H(doc, "4.3 窗口时空 BFS 与 StageContract", 2, ea)
    P(
        doc,
        "相对基线 M0 的串行全视界时空 A*，会商规划仅在 W 内调用优先序 spacetime_bfs[12],[18]。"
        "状态为 (x,y,t)；扩展含等待与四邻；预约禁止 verts[t] 占用与边交换。目标 BFS 场提供引导；"
        "若 W 内不可达则保留最佳进展态，否则一步贪心回退。智体按 (−priority, name) 序规划并累积"
        "预约——即窗口优先规划，而非联合 CBS[18],[23],[35]。StageContract 字段含 agent、"
        "role∈{TO_PICK,TO_DROP,YIELD,HOLD,CLEAR,IDLE}、waypoints、goal、priority、ttl、"
        "exclusive_pad?、issued_t。会商间跟进提交头；契约按 TTL 过期或被下次会商替换。",
        ea,
    )

    H(doc, "4.4 StageNet 与自适应阶段权威（ASA）", 2, ea)
    P(
        doc,
        "每智体候选：FINAL（真目标格）；沿几何路径约 1/3 与 2/3 的 APPROACH；CLEAR（半径 1→2→3 "
        "曼哈顿环上最小化 stay+0.05·manh(cand,final)）。网络：智体编码 8→64→64（GELU）；候选打分；"
        "优先级头 64→32→1。运行时权重 stage_selector_v1.pt。",
        ea,
    )
    Pb(
        doc,
        "智体特征 x_a∈ℝ⁸：",
        "（1–2）归一化当前 xy；（3–4）归一化最终 xy；（5）manh/(2·GRID)；"
        "（6）负载状态（载货=1，已分配=0.5，空闲=0）；（7）min(阻塞连败/40,1)；"
        "（8）min(区域占用/6,1)+0.25·热度。候选 10 维编码格子、距离、类型、停留、占用与软代价。"
        "运行时优先级地板：载货→10⁶+p；已分配→10⁵+p；否则 10+p。",
        ea,
    )
    Pb(
        doc,
        "教师规则：",
        "d≤5 或 streak≥18 → FINAL；非拥堵或载货 → FINAL（载货永不 CLEAR）；拥堵且 streak≥12 → "
        "CLEAR；否则优先 APPROACH@2/3 再 @1/3。训练：阶段交叉用熵 + listwise 优先级 + 车队正则。",
        ea,
    )
    Pb(
        doc,
        "ASA 门（由 (n, near_pad_deg2) 开局一次）：",
        "在（n≤9∧near≥90）或（n=9∧near∈[50,59]）或（10≤n≤14）或（n=15∧near<60）或"
        "（大车队除 n=18）时用 StageNet；否则用教师——尤其多数 n≤9 与 n=18，避免模仿噪声伤完成率。",
        ea,
    )
    fig(doc, FIG / "fig3_stagenet_zh.jpg", width=5.2)
    cap(doc, "图 3. StageNet 输入输出。ASA 为外部门，不作为网络输入。", ea)

    H(doc, "4.5 FACN、多带 TBC 与 AFA", 2, ea)
    P(
        doc,
        "Multi-Band TBC 由 (n, near) 给静态基线 K_TBC：默认 6；n≥28 时 ≥7；10≤n≤14 且 "
        "near∈[55,85] → 5；16≤n≤18 且 near∈[80,90] → 5。FACN 不取代 TBC，只预测残差。",
        ea,
    )
    Pb(
        doc,
        "FACN 特征（22 维）。",
        "静态 g∈ℝ¹⁰：n/30、free/400、near/160、取/卸点数、密度、idle_pool、K_TBC/16、区域自由"
        "均值/最大。动态 x∈ℝ¹²：忙碌、空闲、剩余、完成率、停滞、平均提交 H、修复、冲突热、区域占用、"
        "目的地门数、同目的地簇、喉占用。MLP 22→64→64→7 分类 Δ∈{−2..+4}；"
        "K=clip(K_TBC+Δ)，K∈[2,min(16,n)]；粘性 TTL=36、周期 24、最小间隔 12；每步 Δ 变化至多 ±1。"
        "FACN 不选择具体哪辆车忙碌。",
        ea,
    )
    Pb(
        doc,
        "AFA：",
        "仅当 n>K_TBC 挂载。抬升：near≤40 最多 +4；≤70 +2；否则 +1；n≤12 默认不抬除非 oracle。"
        "降低仅当 (throat≥4∨same_dest≥4)∧(stall≥40∨no_prog)。Δ=−2 仅非洪水（near≤80）且 "
        "throat≥5∧same_dest≥4；洪水图禁 −2。进度脉冲期间有效 K←min(K,3)。",
        ea,
    )
    fig(doc, FIG / "fig4_facn_zh.jpg", width=5.2)
    cap(doc, "图 4. FACN：静态 g 与动态 x → Δ → K，经 AFA 与脉冲裁剪。", ea)

    H(doc, "4.6 几何门控协议套件", 2, ea)
    P(doc, "协议是一等“工具”，各对应可复现失败；每拍与/或会商内运行。下列阈值保证可复现与可审计。", ea)

    H(doc, "4.6.1 竞争目的地门", 3, ea)
    P(
        doc,
        "失败：多车对冲同一卸垫。触发：≥2 载货共享站 hub 且 ≥2 车到任一垫 Manh≤6。"
        "持有者（最近）优先级 +3·10⁵，锁 TTL=48；持有者 streak≥48 可换人。非持有者改写到最近 "
        "deg≥3，禁止垫、deg≤2 廊与他车路径末 4 格，PAD_EXCLUSION=3。频率：每拍+会商。",
        ea,
    )
    H(doc, "4.6.2 垫区出口与 FG-CEBI", 3, ea)
    P(
        doc,
        "失败：卸点进出口死锁。空载在卸格（或受压 sibling/廊）改写到 deg≥3，EGRESS_PRI=5·10⁷。"
        "刚卸 soft leave；进展陈旧可驱逐邻接占垫。FG-CEBI 仅 n≥10∧near≥100：廊上有 egressor 且"
        "载货到垫最短路穿廊（Manhattan 三角等式）时，将该载货目标改到廊外。中度喉与 n<10 不开门。",
        ea,
    )
    H(doc, "4.6.3 E4 粘性卸后出口", 3, ea)
    P(
        doc,
        "失败：卸后仍停 deg≤2 阻下一次进入。门：n≥19∧near∈[70,85]。记忆 last_unload_pad；"
        "仍驻 deg≤2 则强制 egress，不等待进入压力。",
        ea,
    )
    H(doc, "4.6.4 进度脉冲", 3, ea)
    P(
        doc,
        "失败：竞争负载下长期无进展。no_prog：contested≥1、loaded≥2，且 (done_stale≥120 或 "
        "manh_stale≥70)；n≥16 时阈值收紧为 80/50。对 n≥12，脉冲 TTL=28 令有效 K←min(base,3)，"
        "会商更勤，W 可至 28。脉冲是临时油门，非永久 K 策略。",
        ea,
    )
    H(doc, "4.6.5 PGFA：进展门控焦点利他", 3, ea)
    P(
        doc,
        "失败：互让活锁。触发：剩余≤5 或 stall≥25 或任一载货 streak≥12。仅单一焦点；他车清其 "
        "BFS 前缀长度 24。无进展 ≥50 换焦点；近目标（manh≤2）粘滞至 120。垫驱逐 manh≤3；"
        "park_freeze=24。空闲为焦点让路；非焦点载货互不让——打破对称僵持。",
        ea,
    )
    H(doc, "4.6.6 空隙让行、DLO、区域准入、空闲卫生", 3, ea)
    P(
        doc,
        "空隙让行：低优先级挡在高优先级下一格上则侧移（禁交换）。DLO（会商，10≤n≤14，载货≥2）"
        "比较 (−pri,name) 与 (−pri,manh,name)，取载货 ΣManhattan 更小者，并剥交换边/非法格。"
        "区域准入（5×5 瓦片）将 FINAL 密度软限制为 clamp(free/8,2,5)；超额非保护 FINAL 改 "
        "CLEAR/YIELD；会商硬保护 TO_DROP。空闲廊卫生：空载在载货 deg≤2 廊上连续 24 拍则推到 deg≥3。",
        ea,
    )

    p = doc.add_paragraph()
    pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4, first=0)
    add(p, "表 II. 协议 → 失败模式 → 主门控（摘要）。", size=10, italic=True, east_asia=ea)
    table(
        doc,
        ["协议", "失败模式", "主门控/阈值"],
        [
            ["竞争目的地", "同垫对冲", "loaded≥2, near≤6, TTL=48"],
            ["垫区出口", "进出互卡", "空载在垫/受压"],
            ["FG-CEBI", "廊上锁道", "n≥10 ∧ near≥100"],
            ["E4 粘性", "卸后 deg≤2 滞留", "n≥19 ∧ near∈[70,85]"],
            ["进度脉冲", "长期无进展", "n≥12, K≤3, TTL=28"],
            ["PGFA", "互让活锁", "stall≥25 / streak≥12；单焦点"],
            ["DLO", "哈希序抖动", "10≤n≤14, loaded≥2"],
            ["区域准入", "FINAL 过密", "5×5，容量 2..5"],
        ],
        ea,
    )

    H(doc, "4.7 完备性与非声明", 2, ea)
    P(
        doc,
        "Meeting-RH 不完备：有界 W、优先 BFS 与启发式契约在病理实例上可能失败，而 CBS 终将成功"
        "[1],[40]。我们不主张最优/有界次优，也不主张全局方向场网络有益——历史试验未提升 done/sim，"
        "故不纳入默认贡献。经典 AI 规划背景见[60]。",
        ea,
    )

    H(doc, "5 实验", 1, ea)
    H(doc, "5.1 设置与基线", 2, ea)
    P(
        doc,
        "SH01–20，官方约 100 任务流。基线：PIBT[27]；M0（全视界时空 A*+贪心分配）；PP；窗口 "
        "ECBS[2],[35]；Hier。本文：Meeting-RH+协议+StageNet(ASA)+FACN(AFA)。复现："
        "PYTHONHASHSEED=0、CPU、每图子进程。主对比要求 VALID。",
        ea,
    )
    p = doc.add_paragraph()
    pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4, first=0)
    add(
        p,
        "表 I. SH01–20（官方 100）汇总。VALID∧full100 中 sim/wall 越低越好。"
        "† 未全完成图上的 sim 不可作“更快”。",
        size=10, italic=True, east_asia=ea,
    )
    table(
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
    P(
        doc,
        "在 20/20 全完成方法中，本文 avg_sim 约为 Hier 的 42%、ECBS 的 23%，墙钟约 28s。"
        "仅协议相对旧 FULL 约 −9.0% avg_sim 且 0 图变差；FACN v2 相对 noFACN（SH01–40）约 "
        "−5.82% avg_sim。关 StageNet/PGCP/PGFA/FACN 的消融分别伤害难图完成或开阔图吞吐，支持"
        "正交分工主张。窗口 BFS 相对串行 M0 A* 降低调用频率、深度与抖动[12],[18],[35]。",
        ea,
    )

    H(doc, "5.2 消融与分层收益", 2, ea)
    p = doc.add_paragraph()
    pf(p, align=WD_ALIGN_PARAGRAPH.CENTER, before=8, after=4, first=0)
    add(p, "表 III. 消融要点（代表图；done 越低越差）。", size=10, italic=True, east_asia=ea)
    table(
        doc,
        ["设置", "SH10 done", "SH17 done", "SH20 sim", "说明"],
        [
            ["完整（本文）", "100", "100", "660", "默认栈"],
            ["无 StageNet", "60", "32", "—", "难图位姿崩溃"],
            ["无 PGCP", "↓", "↓", "—", "SH10/11/12/17 受损"],
            ["无 PGFA", "—", "20", "—", "互让失败"],
            ["无 FACN", "100", "100", "813", "开阔车队回退"],
        ],
        ea,
    )
    P(
        doc,
        "相对 RHCR 式联合窗口 ECBS[35]，会商只协商紧凑慢状态，并把走廊失败交给可审计协议[55]。"
        "StageNet/FACN 均部署在几何权威之后（ASA/AFA），形成“小网络 + 硬门”模式，便于仓储场景审计"
        "[44],[47],[51]，并与纠缠位姿/路由/并发的端到端 RL[48]–[50]形成对比。几何代理"
        "（near_pad_deg2、喉占用、同目的地簇）不含地图 ID，可在共享喉统计的 SH 图间迁移[51],[52]。",
        ea,
    )
    H(doc, "5.3 讨论：各层买到了什么", 2, ea)
    P(
        doc,
        "会商买到时间抽象：昂贵协商摊销到约 H=12 拍的廉价跟进，呼应 WHCA*/RHCR 视窗[18],[35] 但"
        "使用类型化 StageContract。StageNet 买到位姿多样性，抑制优先规划常见的过早 FINAL 扎堆"
        "[19]–[21],[23]。FACN 买到可调运营车队规模且不重训路径代价[51]。协议买到对 PIBT 在容量 1 "
        "结构中难以可靠逃脱的死锁的可解释恢复[26],[27]。整体更接近“规划器 + 规则 + 微小学习器”"
        "而非单体学习策略。",
        ea,
    )

    H(doc, "6 局限与效度威胁", 1, ea)
    P(
        doc,
        "Meeting-RH 不完备；FACN 跨地图族留出泛化有限；协议需维护并可相互作用；压力图≠工业数字孪生"
        "[44],[51]。固定种子与 CPU；未在会商内评估连续时间鲁棒执行[41],[42] 或 LaCAM* 联合搜索"
        "[28]–[30]——均为未来混合方向。",
        ea,
    )

    H(doc, "7 结论", 1, ea)
    P(
        doc,
        "本文提出 Meeting-RH：将几何门控协议与 StageNet、FACN 耦合的可解释终身 MAPD 混合方法。"
        "实证上对标强搜索基线的有效性与完成率，并显著改善仿真与墙钟。未来工作包括更强留出并发学习、"
        "与 anytime 窗口搜索/LNS 的更紧集成[37],[38]、引导适应[53] 与更丰富运动学[34],[51]。",
        ea,
    )

    H(doc, "附录 A 常数速查", 1, ea)
    P(
        doc,
        "MEET_H=12；MEET_W∈{16,28}；契约 TTL=16；stall-force=24；CONTEND_NEAR=6；门 TTL=48；"
        "EGRESS_PRI=5e7；FG-CEBI：n≥10∧near≥100；E4：n≥19∧near∈[70,85]；脉冲 TTL=28 且 K≤3；"
        "PGFA 前缀=24；区域瓦片 5×5；FACN 粘性 TTL=36/周期24/间隔12；Δ∈{−2..+4}；"
        "K∈[2,min(16,n)]。",
        ea,
        first=0,
    )
    H(doc, "附录 B 可复现性", 1, ea)
    P(
        doc,
        "入口：python -m greed_coop_mcts，policy=rrsc；权重 facn_v2.pt 与 stage_selector_v1.pt；"
        "PYTHONHASHSEED=0；每图子进程。",
        ea,
        first=0,
    )

    refs_block(doc, REFS, "参考文献", ea)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "MeetingRH_Paper_ZH.docx"
    doc.save(path)
    print("Wrote", path)
    return path


if __name__ == "__main__":
    build_en()
    build_zh()
