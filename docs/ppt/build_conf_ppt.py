# -*- coding: utf-8 -*-
"""~30-page bilingual conference PPT. Narrative: problem → related → method → details → results.
No competition/MioVerse branding. No GlobalRuleNet."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
IMG = ROOT / "generateimage" / "mioverse-conf-ppt"

SW, SH = Inches(13.333), Inches(7.5)
ML, MR = Inches(0.4), Inches(0.4)
CW = SW - ML - MR
CT = Inches(0.9)

NAVY = RGBColor(0x1A, 0x36, 0x5D)
BLUE = RGBColor(0x2B, 0x6C, 0xB0)
SLATE = RGBColor(0x1E, 0x29, 0x3B)
MUTED = RGBColor(0x4A, 0x55, 0x68)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF7, 0xFA, 0xFC)
LINE = RGBColor(0xCB, 0xD5, 0xE0)
ACCENT = RGBColor(0xEB, 0xF4, 0xFF)
CORAL = RGBColor(0xC5, 0x30, 0x30)


def F(run, size=11, bold=False, color=SLATE):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Microsoft YaHei"
    rPr = run._r.get_or_add_rPr()
    from lxml import etree
    ea = rPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
    if ea is None:
        ea = etree.SubElement(rPr, "{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
    ea.set("typeface", "Microsoft YaHei")


def bg(s):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    sh.fill.solid(); sh.fill.fore_color.rgb = WHITE; sh.line.fill.background()
    el = sh._element; t = s.shapes._spTree; t.remove(el); t.insert(2, el)


def bar(s):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(0.06))
    sh.fill.solid(); sh.fill.fore_color.rgb = BLUE; sh.line.fill.background()


def foot(s, i, n, brand):
    b = s.shapes.add_textbox(ML, SH - Inches(0.28), Inches(9.5), Inches(0.22))
    r = b.text_frame.paragraphs[0].add_run(); r.text = brand; F(r, 8, False, MUTED)
    b2 = s.shapes.add_textbox(SW - MR - Inches(1.0), SH - Inches(0.28), Inches(1.0), Inches(0.22))
    p = b2.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r2 = p.add_run(); r2.text = f"{i}/{n}"; F(r2, 8, False, MUTED)


def title(s, main, sub=None):
    box = s.shapes.add_textbox(ML, Inches(0.14), CW, Inches(0.38))
    r = box.text_frame.paragraphs[0].add_run(); r.text = main; F(r, 20, True, NAVY)
    if sub:
        box2 = s.shapes.add_textbox(ML, Inches(0.5), CW, Inches(0.3))
        r2 = box2.text_frame.paragraphs[0].add_run(); r2.text = sub; F(r2, 10, False, MUTED)
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, ML, Inches(0.82), Inches(1.1), Inches(0.03))
    ln.fill.solid(); ln.fill.fore_color.rgb = BLUE; ln.line.fill.background()


def tx(s, l, t, w, h, text, size=11, bold=False, color=SLATE, align=PP_ALIGN.LEFT):
    box = s.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = text; F(r, size, bold, color)


def bullets(s, l, t, w, h, items, size=11, spacing=3):
    box = s.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(spacing)
        if item.startswith("**") and "**" in item[2:]:
            end = item.index("**", 2)
            r = p.add_run(); r.text = "• " + item[2:end]; F(r, size, True, SLATE)
            rest = item[end + 2:]
            if rest:
                r2 = p.add_run(); r2.text = rest; F(r2, size, False, SLATE)
        else:
            r = p.add_run(); r.text = "• " + item; F(r, size, False, SLATE)


def card(s, l, t, w, h, fill=LIGHT):
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = LINE; sh.line.width = Pt(0.75)
    return sh


def table(s, l, t, w, h, headers, rows, fs=10):
    nr, nc = 1 + len(rows), len(headers)
    shape = s.shapes.add_table(nr, nc, l, t, w, h)
    tbl = shape.table
    cw = int(w / nc)
    for c in range(nc):
        tbl.columns[c].width = cw

    def cell(ci, ri, text, header=False):
        c = tbl.cell(ri, ci)
        c.text = ""
        tf = c.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = str(text); F(r, fs, header, WHITE if header else SLATE)
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        c.fill.solid()
        c.fill.fore_color.rgb = NAVY if header else (LIGHT if ri % 2 == 0 else WHITE)

    for j, h_ in enumerate(headers):
        cell(j, 0, h_, True)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell(j, i + 1, v, False)


def fit(s, path: Path, l, t, mw, mh):
    if not path.exists():
        card(s, l, t, mw, mh); tx(s, l, t + mh / 2, mw, Inches(0.3), path.name, 10, False, MUTED, PP_ALIGN.CENTER)
        return
    with Image.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    if ar >= mw / mh:
        w, h = mw, mw / ar
    else:
        h, w = mh, mh * ar
    s.shapes.add_picture(str(path), l + (mw - w) / 2, t + (mh - h) / 2, width=w, height=h)


def news(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6]); bg(s); bar(s); return s


# ---------------- slides ----------------

def cover(s, lang, img):
    card(s, Inches(0.7), Inches(1.5), Inches(11.9), Inches(4.5), LIGHT)
    if lang == "zh":
        tx(s, Inches(1), Inches(1.9), Inches(11.3), Inches(0.4),
           "Lifelong Multi-Agent Pickup-and-Delivery", 13, False, BLUE, PP_ALIGN.CENTER)
        tx(s, Inches(1), Inches(2.5), Inches(11.3), Inches(1.0),
           "会议式混合栈中的可解释车队并发控制", 26, True, NAVY, PP_ALIGN.CENTER)
        tx(s, Inches(1), Inches(3.7), Inches(11.3), Inches(0.5),
           "Meeting-RH × 几何门控协议 × StageNet 姿态选择 × FACN 并发油门", 13, False, SLATE, PP_ALIGN.CENTER)
        tx(s, Inches(1), Inches(4.6), Inches(11.3), Inches(0.5),
           "方法报告 · 约 12 分钟", 12, False, MUTED, PP_ALIGN.CENTER)
    else:
        tx(s, Inches(1), Inches(1.9), Inches(11.3), Inches(0.4),
           "Lifelong Multi-Agent Pickup-and-Delivery", 13, False, BLUE, PP_ALIGN.CENTER)
        tx(s, Inches(1), Inches(2.5), Inches(11.3), Inches(1.0),
           "Explainable Fleet Concurrency Control\nin a Meeting-based Hybrid Stack", 24, True, NAVY, PP_ALIGN.CENTER)
        tx(s, Inches(1), Inches(3.9), Inches(11.3), Inches(0.5),
           "Meeting-RH × Geometry-gated Protocols × StageNet × FACN", 13, False, SLATE, PP_ALIGN.CENTER)
        tx(s, Inches(1), Inches(4.7), Inches(11.3), Inches(0.4),
           "Technical talk · ~12 minutes", 12, False, MUTED, PP_ALIGN.CENTER)


def outline(s, lang, img):
    if lang == "zh":
        title(s, "汇报结构", "问题 → 相关工作 → 方法 → 细节 → 结果")
        items = [
            ("§1", "研究问题与价值", "规则、VALID 标准、核心难点"),
            ("§2", "历史方法", "CBS/ECBS、PP、PIBT、MAPD、RHCR"),
            ("§3", "我们的方法", "贡献与系统总览"),
            ("§4.1", "Meeting 机制", "触发—输入—过程—输出—回归"),
            ("§4.2", "协议工具栈", "每个工具对应的故障模式"),
            ("§4.3", "AI 模块", "StageNet / FACN：结构、I/O、消融"),
            ("§5", "结果对比", "完成率 · sim · wall，为何更优"),
        ]
    else:
        title(s, "Outline", "Problem → Related Work → Method → Details → Results")
        items = [
            ("§1", "Problem & Value", "Rules, VALID criteria, hardness"),
            ("§2", "Prior Work", "CBS/ECBS, PP, PIBT, MAPD, RHCR"),
            ("§3", "Our Method", "Contributions & overview"),
            ("§4.1", "Meeting", "Trigger → I/O → process → return"),
            ("§4.2", "Protocol Stack", "Failure mode → tool"),
            ("§4.3", "AI Modules", "StageNet / FACN: arch, I/O, ablation"),
            ("§5", "Results", "done · sim · wall; why best"),
        ]
    for i, (a, b, c) in enumerate(items):
        t = CT + i * Inches(0.78)
        card(s, ML, t, CW, Inches(0.7), ACCENT if i % 2 == 0 else LIGHT)
        tx(s, ML + Inches(0.2), t + Inches(0.18), Inches(1.0), Inches(0.4), a, 14, True, BLUE)
        tx(s, ML + Inches(1.3), t + Inches(0.12), Inches(3.5), Inches(0.45), b, 13, True, NAVY)
        tx(s, ML + Inches(5.0), t + Inches(0.18), Inches(7.3), Inches(0.4), c, 11, False, SLATE)


def problem(s, lang, img):
    if lang == "zh":
        title(s, "§1 研究问题与价值", "Lifelong MAPD：持续任务流下的多机取卸货与无碰路径")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**问题定义**：在共享网格图 G=(V,E) 上，一组 AGV 持续接收取货→卸货任务；需在线分配与规划，使吞吐尽可能高，且始终无冲突。",
            "**与 one-shot MAPF 的本质区别**：目标点随任务流不断更新，无法一次性求出全局最优路径后静止执行；必须周期再协调。",
            "**应用价值**：自动化仓储/分拣中，廊道容量有限、车队规模变化、开阔区与窄廊并存——静态参数与纯反应式策略均易失效。",
            "**科学问题**：在有限在线算力下，应协商哪些慢变量（姿态、优先级、并发上限），用何种周期触发重规划，才能在异构几何上同时保证完成率与墙钟效率？",
            "**本文立场**：正确性由可解释协议保证；自适应由小型可审计网络承担；几何门控决定何时信任网络。",
        ], 12, 6)
    else:
        title(s, "§1 Problem and Value", "Lifelong MAPD: continuous pickup–delivery under collision constraints")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**Definition**: on a shared grid G=(V,E), AGVs continuously receive pickup→delivery tasks; allocate and plan online for high throughput without conflicts.",
            "**vs one-shot MAPF**: goals keep changing; a single offline plan is insufficient — periodic re-coordination is mandatory.",
            "**Value**: warehouses mix open floors and capacity-1 corridors; static parameters and purely reactive policies fail across regimes.",
            "**Research question**: under limited online compute, which slow variables (pose, priority, concurrency) and which periods keep completion high and wall-clock practical?",
            "**Stance**: protocols for correctness; small auditable nets for adaptation; geometry gates for trust.",
        ], 12, 6)


def rules(s, lang, img):
    if lang == "zh":
        title(s, "运行规则与 VALID 校验标准", "硬约束必须全部满足，否则轨迹判无效")
        headers = ["类别", "规则", "校验含义"]
        rows = [
            ["运动学", "正交移动；转角∈{±90°,180°}；移/转/装/卸/等待各计 1 步", "动作合法"],
            ["运载", "单车同时仅一货；取货站 FIFO；每站每秒最多一车取货", "任务语义正确"],
            ["卸货", "必须在目的口邻格卸货点完成", "目的地正确"],
            ["Vertex 冲突", "同一时刻两车占同一格 → 非法", "VALID=否"],
            ["Swap 冲突", "两车对向交换相邻格 → 非法", "VALID=否"],
            ["界外/障碍", "进入不可行格 → 非法", "VALID=否"],
            ["完成指标", "done = 正确卸货任务数；full100 = done=任务总数", "质量主指标"],
            ["效率指标", "sim = 末任务完成时刻；wall = 求解墙钟", "次级指标"],
        ]
        note = "本文报告的所有对比均要求 VALID；未 VALID 的轨迹即使 sim 更小也不计入「更优」。"
    else:
        title(s, "Operating Rules and VALID Criteria", "All hard constraints must hold; otherwise the trajectory is invalid")
        headers = ["Class", "Rule", "Check"]
        rows = [
            ["Kinematics", "4-connected; turns∈{±90°,180°}; move/turn/load/unload/wait = 1", "Legal actions"],
            ["Capacity", "One package/agent; FIFO at pickup; ≤1 pickup/station/sec", "Task semantics"],
            ["Delivery", "Unload only at designated adjacent cells", "Correct goal"],
            ["Vertex conflict", "Two agents same cell same time → illegal", "VALID=false"],
            ["Swap conflict", "Opposite edge swap → illegal", "VALID=false"],
            ["Bounds", "Obstacle/out-of-map → illegal", "VALID=false"],
            ["Quality", "done = correct deliveries; full100 = all tasks done", "Primary"],
            ["Efficiency", "sim = finish time; wall = runtime", "Secondary"],
        ]
        note = "All comparisons require VALID; a smaller sim on an invalid trajectory does not count as better."
    table(s, ML, CT, CW, Inches(4.9), headers, rows, fs=10)
    card(s, ML, Inches(6.05), CW, Inches(0.7), ACCENT)
    tx(s, ML + Inches(0.2), Inches(6.2), CW - Inches(0.4), Inches(0.45), note, 11, True, NAVY)


def challenges(s, lang, img):
    if lang == "zh":
        title(s, "核心难点：为何需要混合栈", "吞吐、正确性、异构几何、在线算力四者冲突")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**容量-1 廊道**：同卸货点/deg≤2 喉道上多车对顶是结构性死锁源，不是简单加边权能消掉的噪声。",
            "**异构场景**：开阔大车队需要抬高并发；高密迷宫需要限流与廊门——单一静态 K 必然顾此失彼。",
            "**终身流**：任务持续到达 → 必须双时间尺度：慢变量（姿态/优先级/K）周期协商，快冲突 tick 级修补。",
            "**完备搜索昂贵**：窗口 ECBS 可高完成率，但墙钟达数百秒，难在线；反应式 PIBT 墙钟快但完成率不稳。",
            "**可解释性要求**：端到端黑盒难以审计「为何此刻让谁进廊、为何允许 10 车忙碌」——需要模块化决策变量。",
        ], 12, 7)
    else:
        title(s, "Hardness: Why a Hybrid Stack", "Throughput, validity, heterogeneous geometry, online compute")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**Capacity-1 corridors**: same-destination / deg≤2 throats create structural head-ons.",
            "**Heterogeneous maps**: open large fleets need higher concurrency; dense mazes need throttling — one static K fails.",
            "**Lifelong streams**: dual timescales — slow vars negotiated periodically; fast conflicts repaired every tick.",
            "**Complete search is costly**: windowed ECBS completes but costs hundreds of seconds; PIBT is fast but fragile.",
            "**Explainability**: black-box policies cannot audit who enters a corridor and how many agents may stay busy.",
        ], 12, 7)


def related1(s, lang, img):
    if lang == "zh":
        title(s, "§2 相关工作（一）：经典 MAPF 求解器", "引用与一句话机制")
        headers = ["方法", "代表文献", "机制要点", "局限（对本问题）"]
        rows = [
            ["CBS", "Sharon et al., AIJ 2015", "高层冲突拆分 + 低层单机最优", "one-shot；lifelong 需反复调用"],
            ["ECBS", "Barer et al., SoCS 2014", "Focal search 有界次优 CBS", "完成率高但窗内搜索墙钟重"],
            ["Priority Planning", "经典 PP / CA*", "固定优先级串行时空 A*", "优先级差则死锁；VALID 不稳"],
            ["PIBT", "Okumura et al., AIJ 2022", "优先级继承+回溯的逐步反应式", "可扩展，但窄廊/死胡同脆弱"],
        ]
        note = "本仓库对照实现：M0=时空 A* 贪心指派；PP=优先级 A*；ECBS=窗口化 ECBS。"
    else:
        title(s, "§2 Related Work (I): Classical MAPF", "Citations and one-line mechanisms")
        headers = ["Method", "Reference", "Idea", "Limit (for us)"]
        rows = [
            ["CBS", "Sharon et al., AIJ 2015", "High-level conflict split + low-level A*", "One-shot; lifelong needs repeats"],
            ["ECBS", "Barer et al., SoCS 2014", "Focal-search bounded-suboptimal CBS", "High completion, heavy wall-clock"],
            ["Priority Planning", "Classical PP / CA*", "Serial space-time A* by priority", "Bad priorities → deadlock"],
            ["PIBT", "Okumura et al., AIJ 2022", "Priority inheritance + backtracking", "Scalable but fragile in corridors"],
        ]
        note = "Our baselines: M0=space-time A*+greedy assign; PP=priority A*; ECBS=windowed ECBS."
    table(s, ML, CT, CW, Inches(4.5), headers, rows, fs=10)
    card(s, ML, Inches(5.8), CW, Inches(0.85), ACCENT)
    tx(s, ML + Inches(0.2), Inches(6.0), CW - Inches(0.4), Inches(0.55), note, 11, False, NAVY)


def related2(s, lang, img):
    if lang == "zh":
        title(s, "§2 相关工作（二）：Lifelong / MAPD 与缺口", "从任务流框架到滚动窗——表 + 缺口如何导向本文设计")
        headers = ["方法", "代表文献", "机制要点", "对本问题的缺口"]
        rows = [
            ["MAPD", "Ma et al., AAAI 2017", "取卸任务流；分配与路径耦合", "未学异构几何下的运营并发 K"],
            ["RHCR", "Li et al., AAAI 2021", "每 h 步重规划；仅窗 w 内消冲突", "窗内搜索仍贵；无显式姿态/K 学习"],
            ["exRHCR", "Madar et al., SoCS 2022", "历史优先级 warm-start PBS", "加速搜索 ≠ 控制「多少车该忙」"],
            ["Hier(M0↔ECBS)", "本工作对照栈", "易图 M0，难图切换 ECBS", "full100 高，但 wall 700–1000s"],
        ]
        table(s, ML, CT, CW, Inches(3.5), headers, rows, fs=10)
        card(s, ML, Inches(4.65), CW, Inches(2.1), ACCENT)
        bullets(s, ML + Inches(0.15), Inches(4.75), CW - Inches(0.3), Inches(1.9), [
            "**缺口 → 设计映射**：(i) 反应式快但不稳 → 用会议契约+协议，而不是纯 PIBT；(ii) 窗口搜索稳但慢 → 会议只协商慢变量，会间局部修补；(iii) 缺「姿态 vs 油门」分离 → StageNet 管姿态，FACN 只调 K（不选哪辆车）；(iv) 缺可解释故障工具 → ASA/TBC/廊门/PGFA 等按几何门控启用。",
        ], 11, 3)
    else:
        title(s, "§2 Related Work (II): Lifelong / MAPD and Gaps", "From task streams to rolling horizons — table + how gaps drive our design")
        headers = ["Method", "Reference", "Idea", "Gap for us"]
        rows = [
            ["MAPD", "Ma et al., AAAI 2017", "Coupled assign + routing", "No learned operational concurrency K"],
            ["RHCR", "Li et al., AAAI 2021", "Replan every h; resolve in window w", "Still costly; no pose/K learning"],
            ["exRHCR", "Madar et al., SoCS 2022", "Warm-start priorities in PBS", "Faster search ≠ how-many-busy"],
            ["Hier(M0↔ECBS)", "Our baseline stack", "M0 easy / ECBS hard", "High full100, wall 700–1000s"],
        ]
        table(s, ML, CT, CW, Inches(3.5), headers, rows, fs=10)
        card(s, ML, Inches(4.65), CW, Inches(2.1), ACCENT)
        bullets(s, ML + Inches(0.15), Inches(4.75), CW - Inches(0.3), Inches(1.9), [
            "**Gap → design**: (i) reactive fragile → contracts+protocols; (ii) windowed search slow → meet on slow vars only; (iii) pose vs throttle not split → StageNet=pose, FACN=K only (not which agent); (iv) no explainable failure tools → geometry-gated ASA/TBC/corridor/PGFA.",
        ], 11, 3)


def method(s, lang, img):
    if lang == "zh":
        title(s, "§3 方法与创新点", "Meeting-RH 混合栈：协议保 VALID，小网络学自适应；模块边界可消融")
        # 2x2 contribution cards
        cards = [
            ("C1 双时间尺度",
             "慢：MEET_H=12 协商姿态/优先级/窗承诺（W∈{16,28}），签发 StageContract(ttl=16)。\n"
             "快：会间 Follow 契约；Gap 让路；PGFA 单焦点利他。\n"
             "相对 RHCR：不是每步完备消冲突，而是把「该协商什么」显式拆开。"),
            ("C2 FACN 并发油门",
             "学 Δ∈{−2..+4}，K=clip(K_TBC+Δ)，只改 claim 容量阀门，不改 BFS 边权。\n"
             "管「多少车可 busy」，不管「哪几辆」——选车由指派/优先级完成。\n"
             "AFA 几何门控防 thrash；开阔图抬 K，密图可限流。"),
            ("C3 StageNet 姿态",
             "对 FINAL / APPROACH / CLEAR 打分 + 标量优先级；IL 训练。\n"
             "ASA 用 (n, near_pad_deg2) 在 net↔teacher 间切换，降低小车队噪声。\n"
             "消融：难图关 StageNet 完成率崩（如 SH10/SH17）。"),
            ("C4 几何门控协议",
             "Contested-dest / Pad-egress / FG-CEBI / E4 / Progress-pulse / PGFA…\n"
             "每个工具对应可复现故障（同 dest 对顶、卸后占喉、互让死锁等）。\n"
             "无 map-id：条件只用几何与状态量。"),
        ]
        for i, (h, d) in enumerate(cards):
            r, c = divmod(i, 2)
            l = ML + c * Inches(6.35)
            t = CT + r * Inches(2.55)
            card(s, l, t, Inches(6.15), Inches(2.4), LIGHT if i % 2 == 0 else ACCENT)
            tx(s, l + Inches(0.15), t + Inches(0.12), Inches(5.8), Inches(0.35), h, 13, True, BLUE)
            tx(s, l + Inches(0.15), t + Inches(0.5), Inches(5.8), Inches(1.75), d, 10, False, SLATE)
        tx(s, ML, Inches(6.15), CW, Inches(0.45),
           "明确不做：端到端逐步黑盒策略；按地图编号硬编码；把全局方向场当作主杠杆。",
           10, True, MUTED)
    else:
        title(s, "§3 Method and Contributions",
              "Meeting-RH hybrid: protocols for VALID; small nets for adaptation; ablatable boundaries")
        cards = [
            ("C1 Dual timescales",
             "Slow: MEET_H=12 negotiates pose/priority/commits (W∈{16,28}); StageContract ttl=16.\n"
             "Fast: Follow contracts; Gap yield; PGFA single-focal altruism.\n"
             "vs RHCR: we explicitly factor what must be negotiated, not full conflict resolution every step."),
            ("C2 FACN throttle",
             "Learn Δ∈{−2..+4}, K=clip(K_TBC+Δ); claim valve only — BFS costs untouched.\n"
             "Controls how-many-busy, not which agents (assignment/priority chooses who).\n"
             "AFA gates thrash; raise K on open maps, throttle when dense."),
            ("C3 StageNet pose",
             "Score FINAL/APPROACH/CLEAR + scalar priority; trained by IL.\n"
             "ASA switches net↔teacher via (n, near_pad_deg2).\n"
             "Ablation: hard maps collapse without StageNet (e.g., SH10/SH17)."),
            ("C4 Geometry-gated protocols",
             "Contested-dest / Pad-egress / FG-CEBI / E4 / Progress-pulse / PGFA…\n"
             "Each tool maps to a reproduced failure (pad head-on, post-unload throat, mutual yield…).\n"
             "No map-id: gates use geometry/state only."),
        ]
        for i, (h, d) in enumerate(cards):
            r, c = divmod(i, 2)
            l = ML + c * Inches(6.35)
            t = CT + r * Inches(2.55)
            card(s, l, t, Inches(6.15), Inches(2.4), LIGHT if i % 2 == 0 else ACCENT)
            tx(s, l + Inches(0.15), t + Inches(0.12), Inches(5.8), Inches(0.35), h, 13, True, BLUE)
            tx(s, l + Inches(0.15), t + Inches(0.5), Inches(5.8), Inches(1.75), d, 10, False, SLATE)
        tx(s, ML, Inches(6.15), CW, Inches(0.45),
           "Non-goals: end-to-end black-box step policies; map-id hardcoding; direction fields as the main lever.",
           10, True, MUTED)


def overview_fig(s, lang, img):
    title(s, "方法总览" if lang == "zh" else "Method Overview",
          "三层：几何门控 → Meeting-RH → Tick 执行；FACN 反馈调节 K" if lang == "zh"
          else "Layers: geometry gates → Meeting-RH → tick execution; FACN feedback on K")
    fit(s, img / f"{lang}-fig-overview.jpg", ML, CT, CW, Inches(5.7))


def design(s, lang, img):
    if lang == "zh":
        title(s, "设计原则：解耦 · 门控 · 双时间尺度", "每个模块对应可命名故障，避免「一个大网包打天下」")
        cols = [
            ("解耦", "StageNet=姿态；FACN=并发油门；PGCP=廊道红绿灯。姿态错误导致进廊时机错；并发错误导致闲置或洪水——两者正交，分开学、分开关。"),
            ("门控", "ASA 决定是否信任 StageNet；AFA 决定是否允许 Δ ≠ 0；FG-CEBI/E4 仅在几何条件满足时启用。评委可问「此刻为何听网络」。"),
            ("双尺度", "Meeting TTL≈16 发慢契约；tick 级 Gap/Egress 快修补；FACN TTL=36/PERIOD=24 防 K 抖动。进度停滞可强制开会并加宽窗 W:16→28。"),
        ]
    else:
        title(s, "Design: Decouple · Gate · Dual Timescale", "Each module maps to a named failure — no giant catch-all net")
        cols = [
            ("Decouple", "StageNet=pose; FACN=throttle; PGCP=corridor lights. Wrong pose ≠ wrong concurrency — learn and ablate separately."),
            ("Gate", "ASA trusts StageNet; AFA allows Δ ≠ 0; FG-CEBI/E4 only under geometry. Trust is auditable."),
            ("Dual scale", "Meeting TTL≈16; tick Gap/Egress; FACN TTL=36/PERIOD=24. Progress stall forces meet and widens W:16→28."),
        ]
    for i, (h, d) in enumerate(cols):
        l = ML + i * Inches(4.2)
        card(s, l, CT, Inches(4.05), Inches(5.5), LIGHT)
        tx(s, l + Inches(0.2), CT + Inches(0.25), Inches(3.6), Inches(0.45), h, 15, True, BLUE)
        tx(s, l + Inches(0.2), CT + Inches(0.9), Inches(3.6), Inches(4.2), d, 12, False, SLATE)


def meet_fig(s, lang, img):
    title(s, "§4.1 Meeting 生命周期" if lang == "zh" else "§4.1 Meeting Lifecycle",
          "会前 → 触发 → 输入 → 内部过程 → 输出 → 回归会间" if lang == "zh"
          else "Pre → trigger → inputs → process → outputs → return")
    fit(s, img / f"{lang}-fig-meeting.jpg", ML, CT, CW, Inches(5.7))


def meet_io(s, lang, img):
    if lang == "zh":
        title(s, "Meeting：触发条件与输入/输出", "should_meet + solve 强制条件；输出可执行契约而非自然语言")
        headers = ["阶段", "内容", "关键阈值/字段"]
        rows = [
            ["会前", "Follow 既有 StageContract；tick 级 egress/contested", "ttl=16"],
            ["触发", "周期 / stall 强制 / dense·sparse no_prog", "MEET_H=12; stall≥24; n≥12∧gap≥4"],
            ["输入", "位姿、载货、目标、stall、冲突热、near_pad_deg2", "全局可观察状态"],
            ["过程", "StageNet/teacher→区域准入→窗PP→repair→legalize", "W∈{16,28}"],
            ["输出", "stages, priorities, commits, contracts, desired", "waypoints[:6], exclusive_pad"],
            ["回归", "Follow + Gap；事件触发 PGFA", "stall≥25 / streak≥12"],
        ]
    else:
        title(s, "Meeting: Triggers and I/O", "should_meet + force rules; executable contracts, not natural language")
        headers = ["Stage", "Content", "Key fields / thresholds"]
        rows = [
            ["Pre", "Follow StageContract; tick egress/contested", "ttl=16"],
            ["Trigger", "period / stall / dense·sparse no_prog", "MEET_H=12; stall≥24; n≥12∧gap≥4"],
            ["Inputs", "pose, load, goals, stall, heat, near_pad_deg2", "observable state"],
            ["Process", "StageNet/teacher→admit→window PP→repair→legalize", "W∈{16,28}"],
            ["Outputs", "stages, priorities, commits, contracts, desired", "waypoints[:6], exclusive_pad"],
            ["Return", "Follow + Gap; event PGFA", "stall≥25 / streak≥12"],
        ]
    table(s, ML, CT, CW, Inches(5.5), headers, rows, fs=10)


def meet_steps(s, lang, img):
    if lang == "zh":
        title(s, "Meeting 内部过程与会间修补", "run_meeting 十步 + 会间三机制（合并叙述）")
        bullets(s, ML, CT, Inches(6.3), Inches(5.9), [
            "**①–②** BFS path_cache；StageNet/teacher（ASA 已定 use_net）批量选阶段。",
            "**③–④** region_admit 软准入；装配 stage_goals 与优先级（holder/egress boost）。",
            "**⑤–⑥** contested-dest 改写同 dest；pad-egress（含 FG-CEBI / E4 sticky）。",
            "**⑦–⑧** plan_committed(horizon=W)；window_repair 修窗内冲突。",
            "**⑨–⑩** DLO legalize（仅 10≤n≤14）；签发 StageContract(ttl=16)。",
            "**会间 Follow**：优先走 commit 头，否则朝下一 waypoint。",
            "**Gap**：低优先挡高优先 next cell → 侧移让路。",
            "**PGFA**：单焦点受益人清 BFS 前缀；防止无条件互让死锁。",
            "**注意**：FACN 的 K 在 tick 环事件更新，不在会议函数内——但影响后续 claim 集合。",
        ], 11, 3)
        card(s, ML + Inches(6.5), CT, Inches(5.7), Inches(5.6), ACCENT)
        tx(s, ML + Inches(6.7), CT + Inches(0.3), Inches(5.3), Inches(0.4), "协商到底在争什么？", 14, True, NAVY)
        bullets(s, ML + Inches(6.7), CT + Inches(1.0), Inches(5.3), Inches(4.2), [
            "每车以何姿态接近目标（FINAL/APP/CLEAR）",
            "相对优先级（谁先过廊）",
            "未来 W 步内时空承诺",
            "同 dest 谁持有 exclusive_pad",
            "空载/卸后如何清喉",
        ], 12, 8)
    else:
        title(s, "Inside the Meeting and Inter-meeting Repair", "Ten steps in run_meeting + three inter-meeting mechanisms")
        bullets(s, ML, CT, Inches(6.3), Inches(5.9), [
            "**①–②** BFS cache; StageNet/teacher batch stages (ASA sets use_net).",
            "**③–④** region_admit; assemble goals & priorities (holder/egress boosts).",
            "**⑤–⑥** contested-dest rewrite; pad-egress (FG-CEBI / E4).",
            "**⑦–⑧** plan_committed(W); window_repair.",
            "**⑨–⑩** DLO legalize (10≤n≤14 only); issue StageContract(ttl=16).",
            "**Follow**: commit head else next waypoint.",
            "**Gap**: low-priority yields next cell.",
            "**PGFA**: clear BFS prefix for one focal agent.",
            "**Note**: FACN updates K on the tick loop, outside run_meeting.",
        ], 11, 3)
        card(s, ML + Inches(6.5), CT, Inches(5.7), Inches(5.6), ACCENT)
        tx(s, ML + Inches(6.7), CT + Inches(0.3), Inches(5.3), Inches(0.4), "What is negotiated?", 14, True, NAVY)
        bullets(s, ML + Inches(6.7), CT + Inches(1.0), Inches(5.3), Inches(4.2), [
            "Pose toward goal (FINAL/APP/CLEAR)",
            "Relative priorities",
            "Space-time commits over W",
            "Who holds exclusive_pad",
            "How idles clear the throat",
        ], 12, 8)


def _proto_cards(s, items):
    """items: list of (title, bullet_lines). Layout 3 vertical cards."""
    n = len(items)
    gap = Inches(0.12)
    w = (CW - gap * (n - 1)) / n
    for i, (h, lines) in enumerate(items):
        l = ML + i * (w + gap)
        card(s, l, CT, w, Inches(5.65), LIGHT if i % 2 == 0 else ACCENT)
        tx(s, l + Inches(0.12), CT + Inches(0.12), w - Inches(0.2), Inches(0.55), h, 11, True, BLUE)
        bullets(s, l + Inches(0.1), CT + Inches(0.7), w - Inches(0.18), Inches(4.8), lines, 9, 2)


def proto_fig(s, lang, img):
    title(s, "§4.2 协议工具总览" if lang == "zh" else "§4.2 Protocol Toolbox Overview",
          "下一页起逐项写清：触发 / 改什么 / 防什么 / 不做什么" if lang == "zh"
          else "Following slides detail trigger / effect / failure / non-goals")
    fit(s, img / f"{lang}-fig-protocols.jpg", ML, CT, CW, Inches(5.7))


def proto_stack(s, lang, img):
    if lang == "zh":
        title(s, "协议栈怎么叠在一起", "near = 卸货点半径 6 内、度数≤2 的可行走格数（喉道拥挤代理量）")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**PGCP（Progress-Gated Corridor Protocol）**：廊道协议伞名。下面挂 Contested-dest、Pad-egress（含 FG-CEBI / E4）、Progress-pulse、空载卫生等——不是单一开关。",
            "**几何先决**：开局用 (n, near) 定 ASA（用不用 StageNet）与 Multi-Band TBC（基础并发 K0）；运行中 AFA 裁剪 FACN 的 Δ，得到有效 K。",
            "**每 tick 必跑**：Pad-egress（+条件 FG-CEBI/E4）与 Contested-dest——开会不开会都改写目标/清喉。",
            "**仅会议内**：StageNet/teacher → region_admit → 窗 PP → DLO legalize → 签发契约。",
            "**卡住时**：Progress-pulse（短时降 K、加窗、更勤开会）+ PGFA（只服务一个焦点受益人清廊）。",
            "**消融事实**：关掉 PGCP 套件后 SH10/11/12/17 完成率显著下降——协议与网络同等关键，不是装饰。",
            "**共同约束**：无 map-id 特判；条件只用几何与可观察状态。",
        ], 11, 5)
    else:
        title(s, "How Protocols Stack", "near = #walkable deg≤2 cells within radius 6 of unload pads")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**PGCP**: umbrella for Contested-dest, Pad-egress (+FG-CEBI/E4), Progress-pulse, idle hygiene — not one switch.",
            "**Geometry first**: (n,near) sets ASA and Multi-Band TBC (K0); AFA clips FACN Δ → effective K.",
            "**Every tick**: Pad-egress (+conditional FG-CEBI/E4) and Contested-dest rewrite goals / clear throats.",
            "**Meeting only**: StageNet/teacher → region_admit → window PP → DLO → contracts.",
            "**When stuck**: Progress-pulse (brief lower K, widen W, meet more) + PGFA (one focal beneficiary).",
            "**Ablation**: disabling PGCP suite drops completion on SH10/11/12/17.",
            "**Shared rule**: no map-id; gates use geometry/state only.",
        ], 11, 5)


def tools_gate(s, lang, img):
    if lang == "zh":
        title(s, "协议详解：ASA · Multi-Band TBC · AFA", "决定「信不信网络」与「基础允许多少车忙」")
        _proto_cards(s, [
            ("ASA — Adaptive Stage Authority\n自适应阶段权威", [
                "**干什么**：开局用 (n, near) 决定会议里姿态由 StageNet 还是 goal-flow teacher 产生。",
                "**触发（定一次）**：n∈[10,14]→用网；多数 n≤9→teacher（除非 near≥90 等特判）；n==18→teacher；其余默认用网。",
                "**改什么**：StageSelector(use_net=…)。只影响 FINAL/APPROACH/CLEAR 来源。",
                "**防什么**：小车队盲用网络噪声；部分中队盲用 teacher 打不开死锁。",
                "**不做什么**：不改 K、路径、任务分配；可被 --force/--no-stage-net 覆盖。",
            ]),
            ("Multi-Band TBC — Throat-Band Capacity\n多带喉道容量", [
                "**干什么**：按车队规模与喉道几何给静态基础并发 K0（max_fleet_busy）。",
                "**触发**：默认 K=6；mid(n∈[10,14]∧near∈[55,85])→5；dense(n∈[16,18]∧near∈[80,90])→5；mega(n≥28)→≥7。",
                "**改什么**：claim 时 busy 上限初值；也是 FACN 的 k_tbc 基准。",
                "**防什么**：中等喉道上固定 K=6 过挤；超大车队过保守。",
                "**不做什么**：不改单车目标/优先级/路径；运行中变 K 靠 FACN/pulse，不靠 TBC 自己每 tick 改。",
            ]),
            ("AFA — Adaptive FACN Authority\n自适应 FACN 权威", [
                "**干什么**：几何裁剪 FACN 输出的 Δ，防止乱升/乱降并发。",
                "**触发**：仅当 n>k_tbc（有闲车）才挂 FACN；升幅随 near 封顶（≤40→+4，≤70→+2，否则+1；n≤12 默认不升除非 oracle）；降须 throat/same_dest 拥堵且 stall/no_prog。",
                "**改什么**：Δ∈{−2..+4} 被裁剪后 K=clip(k_tbc+Δ)。",
                "**防什么**：小车队乱抬；无结构拥堵猛降导致饿死螺旋（如 SH39）。",
                "**不做什么**：不选哪辆车、不选路径/阶段；只约束油门。",
            ]),
        ])
    else:
        title(s, "Protocols: ASA · Multi-Band TBC · AFA", "Trust the net? Base how-many-busy?")
        _proto_cards(s, [
            ("ASA — Adaptive Stage Authority", [
                "**What**: choose StageNet vs goal-flow teacher from (n, near) once at start.",
                "**When**: n∈[10,14]→net; most n≤9→teacher (exceptions if near≥90…); n==18→teacher; else net.",
                "**Changes**: StageSelector(use_net). Only pose source.",
                "**Prevents**: net noise on small fleets; teacher stuck on some mid fleets.",
                "**Not**: K, paths, assignment; overridable by CLI flags.",
            ]),
            ("Multi-Band TBC — Throat-Band Capacity", [
                "**What**: static base concurrency K0 from fleet size + throat geometry.",
                "**When**: default 6; mid→5; dense→5; mega n≥28→≥7.",
                "**Changes**: claim busy cap init; FACN's k_tbc baseline.",
                "**Prevents**: K=6 too hot in mid throats; too conservative on huge fleets.",
                "**Not**: per-agent goals/priorities/paths; runtime K changes via FACN/pulse.",
            ]),
            ("AFA — Adaptive FACN Authority", [
                "**What**: geometry clips FACN's Δ to stop thrash.",
                "**When**: attach only if n>k_tbc; raise capped by near; lower only if structural congestion ∧ (stall/no_prog).",
                "**Changes**: clipped Δ → K=clip(k_tbc+Δ).",
                "**Prevents**: raise on tiny fleets; death-spiral lowers (e.g. SH39).",
                "**Not**: which agent / path / stage — throttle only.",
            ]),
        ])


def tools_corridor(s, lang, img):
    if lang == "zh":
        title(s, "协议详解：Contested-dest · Pad-egress · FG-CEBI", "同站对顶与「先出后进」")
        _proto_cards(s, [
            ("Contested-dest\n竞争目的地闸门", [
                "**干什么**：多辆载货挤同一卸货站时，只保留一名 holder 进站，其余改道等待。",
                "**触发**：同站≥2 载货，且≥2 车到任一 pad 的 Manh≤6。",
                "**改什么**：选 holder（近 pad；锁 TTL=48；streak≥48 可换人）；非 holder 目标改到最近 deg≥3，禁入站 pad、deg≤2 廊、各方前缀末 4 格；holder 优先级 +3e5。每 tick+会议都跑。",
                "**防什么**：多车冲同一窄卸点导致廊内对顶/互卡。",
                "**不做什么**：不改任务身份（仍 TO_DROP）；远车(>6)不拦；不管「已在廊内空载挡路」（那是 Pad-egress）。",
            ]),
            ("Pad-egress\n垫点先出后进（基础层）", [
                "**干什么**：空载/刚卸完堵在卸货格或 deg≤2 喉上时，强制先撤出到 deg≥3，再让载货进入。",
                "**触发**：空载在卸货格/同站 sibling pad/压力廊内；刚卸 soft leave；进展陈旧时邻接占垫。每 tick+会议。",
                "**改什么**：空载目标→最近 deg≥3，优先级可抬到 EGRESS_PRI=5e7；必要时把路上正朝该廊挤入的载货目标改到廊外等待（不降其承运角色优先级）。",
                "**防什么**：卸完空车堵喉，迎面载货进不去（egress↔ingress 对顶）。",
                "**不做什么**：不做跨目的地廊并集；不默认全图 sticky（E4 另开）。",
            ]),
            ("FG-CEBI\n车队门控廊道先出后进（E2）", [
                "**干什么**：极端喉道上，对「最短路穿过正在外撤廊」的外面载货做廊锁——只改其阶段目标到廊外。",
                "**触发**：n≥10 ∧ near≥100，且廊上有 egressor，载货不在廊上但其到 pad 最短路穿过该廊。",
                "**改什么**：改写该载货 stage/motion 目标到廊外 deg≥3（或外推一步）；不改优先级角色。",
                "**防什么**：外面车仍沿最短路挤进正在清廊的极端喉道（SH08/19 类）。",
                "**不做什么**：n<10 或 near<100 不启；不覆盖中度喉/脆弱小车队；不是完整 E1–E5 状态机。",
            ]),
        ])
    else:
        title(s, "Protocols: Contested-dest · Pad-egress · FG-CEBI", "Same-pad head-on and egress-before-ingress")
        _proto_cards(s, [
            ("Contested-dest", [
                "**What**: when multiple loaded agents crowd one unload hub, only one holder may enter; others wait off-throat.",
                "**When**: ≥2 loaded at same hub and ≥2 within Manh≤6 of a pad.",
                "**Changes**: pick holder (TTL=48); non-holders retarget deg≥3; ban pads/deg≤2/prefix last 4; holder pri +3e5. Every tick + meeting.",
                "**Prevents**: head-on jams into one narrow unload.",
                "**Not**: task identity; far agents (>6); idle blockers already in throat (Pad-egress).",
            ]),
            ("Pad-egress", [
                "**What**: force idle/post-unload agents off pads/deg≤2 throats to deg≥3 before ingress.",
                "**When**: idle on pad/sibling/pressured throat; soft leave after unload. Every tick + meeting.",
                "**Changes**: idle goal→deg≥3 with high EGRESS_PRI; may retarget approaching carriers off-corridor without demoting carrier role pri.",
                "**Prevents**: egress↔ingress deadlock at pads.",
                "**Not**: multi-destination corridor unions; global sticky (that is E4).",
            ]),
            ("FG-CEBI (E2 lane-lock)", [
                "**What**: on extreme throats, lane-lock carriers whose shortest path crosses an active egress corridor — rewrite their stage goal outward.",
                "**When**: n≥10 ∧ near≥100 and path crosses an egressor corridor.",
                "**Changes**: stage/motion goal only; priority role unchanged.",
                "**Prevents**: outsiders squeezing into a clearing extreme throat.",
                "**Not**: n<10 or near<100; mid throats / fragile tiny fleets; full E1–E5 FSM.",
            ]),
        ])


def tools_e4_pulse(s, lang, img):
    if lang == "zh":
        title(s, "协议详解：E4 sticky · Progress-pulse · PGFA", "卸后粘性外撤、进展脉冲、焦点利他")
        _proto_cards(s, [
            ("E4 sticky — CEBI E4 SHARP-lite\n卸后粘性外撤", [
                "**干什么**：大车队中度喉上，卸完若仍停在 deg≤2，则持续强制外撤，不必再等「有人要进来」才推。",
                "**触发**：n≥19 ∧ near∈[70,85]；记住 last_unload_pad，车还在 deg≤2 则不清除记忆→持续 egress。",
                "**改什么**：刚卸空载进入强制 egress（非 soft），最高优先撤到 deg≥3。",
                "**防什么**：卸完停在终点/喉内导致反复对顶（SH12 类）。",
                "**不做什么**：无门控全图 sticky（会崩 SH02/08/11/15/17）；不做固定 Haven 预留。",
            ]),
            ("Progress-pulse\n进展势 + 忙碌脉冲", [
                "**干什么**：任务进展冻结时，短时降并发并更勤重规划，松绑廊道僵局。",
                "**触发 no_prog**：存在 contested，且 done 停滞≥120（n≥16 时 80）或载货 ΣManh 停滞≥70（n≥16 时 50），且载货≥2。",
                "**改什么**：n≥12 时 busy_pulse_left=28，期间 fleet_cap=min(base,3)；可缩短开会间隔、把会议窗 W 从 16 加到 28。",
                "**防什么**：完成数与 Manh 冻结的密队 livelock。",
                "**不做什么**：n<12 不降 K（防饿死接近车）；不是永久降 cap；不单独改目标几何。",
            ]),
            ("PGFA — Progress-Gated Focal Altruism\n进展门控焦点利他", [
                "**干什么**：卡住时只服务一个焦点受益人——清开其 BFS 前缀上的挡路车，避免全员互让。",
                "**触发**：剩余任务≤5，或 stall≥25，或任一载货 blocked_streak≥12。",
                "**改什么**：焦点优先 egress 空载 > contested holder > 最近目标载货；焦点 pri+1e7；挡路空载侧移/外推并可 park_freeze=24；无进展达阈值才换焦点。",
                "**防什么**：无条件全员让路导致互让/抖振死锁（SH03/11/17）。",
                "**不做什么**：不为所有 busy 清廊并集；非焦点载货之间不互让。",
            ]),
        ])
    else:
        title(s, "Protocols: E4 · Progress-pulse · PGFA", "Sticky egress, progress pulse, focal altruism")
        _proto_cards(s, [
            ("E4 sticky (SHARP-lite)", [
                "**What**: after unload on mid throats with large fleets, keep forcing egress while still on deg≤2 — no need to wait for ingress pressure.",
                "**When**: n≥19 ∧ near∈[70,85]; sticky memory of last_unload_pad.",
                "**Changes**: hard egress to deg≥3 at top priority.",
                "**Prevents**: parking on endpoint/throat after unload (SH12-class).",
                "**Not**: ungated global sticky (breaks SH02/08/11/15/17); fixed havens.",
            ]),
            ("Progress-pulse", [
                "**What**: when progress freezes, briefly lower concurrency and replan more often.",
                "**When no_prog**: contested ∧ (done-stale or manh-stale) with ≥2 carriers.",
                "**Changes**: if n≥12, busy_pulse 28 ticks with cap→min(base,3); may shorten meet gap and widen W 16→28.",
                "**Prevents**: dense livelock with frozen done/Manh.",
                "**Not**: lower K when n<12; permanent cap cut; geometry rewrite alone.",
            ]),
            ("PGFA — Focal Altruism", [
                "**What**: help exactly one beneficiary — clear blockers on its BFS prefix.",
                "**When**: rem≤5 or stall≥25 or any carrier streak≥12.",
                "**Changes**: focal priority order egress>holder>nearest carrier; +1e7 pri; side-step blockers; switch focal only after no-progress thresholds.",
                "**Prevents**: mutual-yield livelock from unconditional altruism.",
                "**Not**: clear corridors for all busy agents; yield among non-focal carriers.",
            ]),
        ])


def tools_dlo_region(s, lang, img):
    if lang == "zh":
        title(s, "协议详解：DLO · region_admit · 小结", "合法化次序与区域软准入")
        _proto_cards(s, [
            ("DLO — Dual Lexicographic Order\n双字典序合法化", [
                "**干什么**：多人抢同一格时，试两种破同分次序，选让载货总曼哈顿更小的那种。",
                "**触发**：legalize_desired 内，仅当 10≤n≤14 且载货≥2。",
                "**改什么**：比较 (−pri, name) 与 (−pri, Manh到目标, name) 两种顶点占用序；并剥边交换、禁非法格。",
                "**防什么**：中队上 PYTHONHASHSEED/名字序吸引子造成的次优抢格抖动。",
                "**不做什么**：不改 stage 目标与任务；n 不在 [10,14] 时退回单序。",
            ]),
            ("region_admit\n区域软容量准入", [
                "**干什么**：会议里防止过多车以 FINAL 姿态挤进同一 5×5 瓦片。",
                "**触发**：每次 meeting，StageNet 出选择之后；cap=clamp(区域内空闲格/8, 2, 5)。",
                "**改什么**：按 −pri 排序；非保护角色的工作 FINAL 若区域已满→降为 CLEAR/YIELD，目标改到路径近端；会议中硬保护主要是 TO_DROP。",
                "**防什么**：多车 FINAL 挤同一瓦片造成动脉堵塞。",
                "**不做什么**：不限制同 unload 前缀载货数；IDLE/CLEAR 等不占 work-cap；不是全局舰队硬 cap（那是 TBC/FACN）。",
            ]),
            ("记住边界", [
                "**FACN**：只调「多少车可 busy」，不选哪辆。",
                "**StageNet**：只选姿态+优先级，不直接输出逐步动作。",
                "**协议**：改目标/优先级/K/合法化次序——用可观察几何触发。",
                "**窗 PP**：在会议内生成有限 horizon 承诺，不是完备 CBS。",
                "**消融**：PGCP 套件与 StageNet 对完成率最关键；FACN 对开阔图 sim 最关键。",
            ]),
        ])
    else:
        title(s, "Protocols: DLO · region_admit · Boundaries", "Legalize order and soft region caps")
        _proto_cards(s, [
            ("DLO — Dual Lexicographic Order", [
                "**What**: when agents contend for a cell, try two tie-break orders; keep the one with smaller carrier ΣManh.",
                "**When**: inside legalize_desired, only 10≤n≤14 and ≥2 carriers.",
                "**Changes**: compare (−pri,name) vs (−pri,Manh,name); strip swaps; ban illegal cells.",
                "**Prevents**: hash/name-order attractor jitter on mid fleets.",
                "**Not**: stage goals/tasks; outside n∈[10,14] falls back to one order.",
            ]),
            ("region_admit", [
                "**What**: in meetings, stop too many FINAL poses packing one 5×5 tile.",
                "**When**: after StageNet choices; cap=clamp(free/8,2,5).",
                "**Changes**: lower excess FINAL to CLEAR/YIELD toward near path cells; protect TO_DROP.",
                "**Prevents**: arterial jams from many FINALs in one tile.",
                "**Not**: same-unload prefix carrier count; global hard fleet cap (TBC/FACN).",
            ]),
            ("Boundaries to remember", [
                "**FACN**: how-many-busy only — not which agent.",
                "**StageNet**: pose+priority only — not step actions.",
                "**Protocols**: rewrite goals/pri/K/legalize via observable geometry.",
                "**Window PP**: finite-horizon commits — not complete CBS.",
                "**Ablation**: PGCP suite + StageNet for completion; FACN for open-map sim.",
            ]),
        ])


def sn_fig(s, lang, img):
    title(s, "§4.3 StageNet：结构与 I/O" if lang == "zh" else "§4.3 StageNet: Architecture and I/O",
          "左栏写明 agent 8 维；下一页表格再列一遍 + 消融" if lang == "zh"
          else "Left panel lists all 8 agent dims; next slide table + ablation")
    fit(s, img / f"{lang}-fig-stagenet.jpg", ML, CT, CW, Inches(5.7))


def sn_ablate(s, lang, img):
    if lang == "zh":
        title(s, "StageNet：8 维状态、ASA 与消融", "特征写全；关掉网络后难图完成率崩")
        headers = ["维", "符号/含义", "归一化"]
        rows = [
            ["1–2", "当前格 cur_x, cur_y", "/ GRID"],
            ["3–4", "最终目标 final_x, final_y", "/ GRID"],
            ["5", "到目标曼哈顿距离 manh(cur,final)", "/ (2·GRID)"],
            ["6", "载货/指派：loaded=1, assigned=0.5, 空闲=0", "离散编码"],
            ["7", "卡住 streak（blocked_streak）", "min(streak/40, 1)"],
            ["8", "区域占用 region_occ + 0.25·冲突热 heat", "截断到 [0,1+]"],
        ]
        table(s, ML, CT, CW, Inches(3.55), headers, rows, fs=10)
        bullets(s, ML, Inches(4.7), CW, Inches(1.0), [
            "**候选 ∈ R¹⁰**：候选格 xy、到车/到终点 Manh、kind、是否 FINAL/CLEAR、stay、是否占用、边代价+区域占用。K=4：FINAL / APP@1/3 / APP@2/3 / CLEAR。",
            "**消融**：no_stagenet → SH10 done=60*，SH17=32*；ASA 按 (n,near) 在 net↔teacher 间切换。",
        ], 10, 3)
    else:
        title(s, "StageNet: 8-D State, ASA, Ablation", "List every agent feature; hard maps collapse without the net")
        headers = ["Dim", "Meaning", "Norm"]
        rows = [
            ["1–2", "current cell cur_x, cur_y", "/ GRID"],
            ["3–4", "final goal final_x, final_y", "/ GRID"],
            ["5", "Manhattan manh(cur,final)", "/ (2·GRID)"],
            ["6", "load status: loaded=1, assigned=0.5, idle=0", "discrete"],
            ["7", "blocked streak", "min(streak/40, 1)"],
            ["8", "region_occ + 0.25·conflict heat", "clipped"],
        ]
        table(s, ML, CT, CW, Inches(3.55), headers, rows, fs=10)
        bullets(s, ML, Inches(4.7), CW, Inches(1.0), [
            "**Cand ∈ R¹⁰**: cell xy, manh to agent/goal, kind, FINAL/CLEAR flags, stay, occupied, edge+region. K=4 stages.",
            "**Ablation**: no_stagenet → SH10=60*, SH17=32*; ASA switches net↔teacher via (n,near).",
        ], 10, 3)


def facn_fig(s, lang, img):
    title(s, "§4.3 FACN：结构与 I/O" if lang == "zh" else "§4.3 FACN: Architecture and I/O",
          "静态+动态 22 维 → Δ 分类 → 修正车队并发 K" if lang == "zh"
          else "Static+dynamic 22-D → Δ class → fleet concurrency K")
    fit(s, img / f"{lang}-fig-facn.jpg", ML, CT, CW, Inches(5.7))


def facn_ablate(s, lang, img):
    if lang == "zh":
        title(s, "FACN：控制对象、AFA 与效果", "主创新：学「允许多少车忙碌」，而非「走哪条边」")
        bullets(s, ML, CT, CW, Inches(2.5), [
            "**控制对象**：仅 effective_fleet_cap（claim/max_assigned）；K = clip(K_TBC + Δ)，Δ ∈ {−2..+4}。",
            "**粘滞**：TTL=36，PERIOD=24，MIN_GAP=12；AFA 限制 raise/lower/attach。",
            "**训练**：离线扫 K* 得 Δ* = K* − K_TBC，CrossEntropy 行为克隆。",
        ], 11, 4)
        headers = ["对比", "avg sim", "better/same/worse", "GATE"]
        rows = [
            ["noFACN → FACN v2 (SH01–40)", "1579.7 → 1487.8 (−5.82%)", "20 / 13 / 5", "PASS"],
            ["SH20", "813 → 660 (−153)", "K:7→10", "主案例"],
            ["SH24", "1036 → 602 (−434)", "K→10 Δ=+4", "大增益"],
        ]
        table(s, ML, Inches(3.6), CW, Inches(2.9), headers, rows, fs=11)
    else:
        title(s, "FACN: Control Target, AFA, Gains", "Novelty: learn how-many-busy, not which-edge")
        bullets(s, ML, CT, CW, Inches(2.5), [
            "**Target**: effective_fleet_cap only; K = clip(K_TBC + Δ), Δ ∈ {−2..+4}.",
            "**Stickiness**: TTL=36, PERIOD=24, MIN_GAP=12; AFA gates raise/lower/attach.",
            "**Train**: offline K* sweep → Δ* = K* − K_TBC → CE cloning.",
        ], 11, 4)
        headers = ["Compare", "avg sim", "better/same/worse", "GATE"]
        rows = [
            ["noFACN → FACN v2 (SH01–40)", "1579.7 → 1487.8 (−5.82%)", "20 / 13 / 5", "PASS"],
            ["SH20", "813 → 660 (−153)", "K:7→10", "Main case"],
            ["SH24", "1036 → 602 (−434)", "K→10 Δ=+4", "Large gain"],
        ]
        table(s, ML, Inches(3.6), CW, Inches(2.9), headers, rows, fs=11)


def exp_setup(s, lang, img):
    if lang == "zh":
        title(s, "§5 实验设置", "SH01–20 官方 100 任务对照；复现设定固定")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**地图**：SH01–20 压力图（车队规模、开阔/喉道比异构）；扩展评测含 SH21–40。",
            "**任务**：每图约 100 个 lifelong 取卸任务。",
            "**指标**：VALID；full100；avg_done；avg_sim；avg_wall。",
            "**复现**：PYTHONHASHSEED=0；CPU；每图子进程隔离。",
            "**基线**：PIBT、M0（时空 A*）、PP、窗口 ECBS、Hier(M0↔ECBS)。",
            "**本文**：Meeting-RH + 协议栈 + StageNet（ASA）+ FACN（AFA）。",
        ], 12, 6)
    else:
        title(s, "§5 Experimental Setup", "SH01–20 with official 100-task streams; fixed reproduction knobs")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**Maps**: SH01–20 stress suite; extended SH21–40.",
            "**Tasks**: ~100 lifelong pickup–delivery jobs per map.",
            "**Metrics**: VALID; full100; avg_done; avg_sim; avg_wall.",
            "**Repro**: PYTHONHASHSEED=0; CPU; per-map subprocess isolation.",
            "**Baselines**: PIBT, M0, PP, windowed ECBS, Hier(M0↔ECBS).",
            "**Ours**: Meeting-RH + protocols + StageNet(ASA) + FACN(AFA).",
        ], 12, 6)


def results(s, lang, img):
    if lang == "zh":
        title(s, "主结果：完成率 · sim · wall（SH01–20）", "同一任务集；全部报告 VALID")
        headers = ["方法", "full100", "VALID", "avg_done", "avg_sim", "avg_wall"]
        rows = [
            ["PIBT", "8/20", "20/20", "63.9", "62811", "40.5s"],
            ["M0 时空A*", "4/20", "20/20", "44.6", "942†", "1478s"],
            ["PP", "1/20", "7/20", "32.4", "848†", "1716s"],
            ["ECBS", "20/20", "20/20", "100", "7364", "543s"],
            ["Hier", "20/20", "20/20", "100", "4018", "1032s"],
            ["Ours", "20/20", "20/20", "100", "1708", "~28s"],
        ]
        note = "† 未完成图的 sim 不能解释为更快。Ours：在 VALID∧full100 前提下，sim 约为 Hier 的 42%，wall 低 1–2 个数量级。"
    else:
        title(s, "Main Results: done · sim · wall (SH01–20)", "Same task set; VALID reported for all")
        headers = ["Method", "full100", "VALID", "avg_done", "avg_sim", "avg_wall"]
        rows = [
            ["PIBT", "8/20", "20/20", "63.9", "62811", "40.5s"],
            ["M0", "4/20", "20/20", "44.6", "942†", "1478s"],
            ["PP", "1/20", "7/20", "32.4", "848†", "1716s"],
            ["ECBS", "20/20", "20/20", "100", "7364", "543s"],
            ["Hier", "20/20", "20/20", "100", "4018", "1032s"],
            ["Ours", "20/20", "20/20", "100", "1708", "~28s"],
        ]
        note = "† sim on incomplete maps is not “faster”. Ours: given VALID∧full100, sim≈42% of Hier with 1–2 orders lower wall."
    table(s, ML, CT, CW, Inches(4.6), headers, rows, fs=11)
    card(s, ML, Inches(5.9), CW, Inches(0.85), ACCENT)
    tx(s, ML + Inches(0.2), Inches(6.1), CW - Inches(0.4), Inches(0.55), note, 11, True, NAVY)


def why_best(s, lang, img):
    if lang == "zh":
        title(s, "为什么当前方法最好？", "在 VALID∧full100 可行集上同时优化 sim 与 wall")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**相对 PIBT/M0/PP**：它们无法在多数图上 full100 或 VALID 不稳；「更快的 sim」来自提前失败，没有比较意义。",
            "**相对 ECBS/Hier**：三者都能 20/20 full100，但 ECBS avg_sim=7364、wall≈543s；Hier sim≈4018、wall≈1032s。Ours sim≈1708、wall≈28s——在同一正确性水平上显著更优。",
            "**机制解释**：Meeting 只在慢变量上花费窗 PP；会间用契约+局部修补，避免每步完备搜索；FACN 在开阔图抬 K 减少闲置；协议栈消除窄廊死锁。",
            "**Pareto 意义**：Ours 位于「高完成率」子集的左下角（低 sim、低 wall），而非牺牲 VALID 换速度。",
            "**增量证据**：相对旧 FULL 协议栈 avg_sim −9.0%（0 劣图）；FACN v2 再 −5.82%（GATE PASS）。",
        ], 12, 6)
    else:
        title(s, "Why Is the Proposed Method Best?", "Optimize sim and wall inside the VALID∧full100 feasible set")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**vs PIBT/M0/PP**: they miss full100 or VALID; smaller sim from early failure is not comparable.",
            "**vs ECBS/Hier**: all reach 20/20 full100, but ECBS sim=7364/wall≈543s and Hier sim≈4018/wall≈1032s; Ours≈1708/~28s.",
            "**Mechanism**: spend windowed PP only on slow vars; contracts+local repair avoid per-step complete search; FACN raises K on open maps; protocols kill corridor deadlocks.",
            "**Pareto**: Ours is bottom-left inside the high-completion subset — not speed by invalidity.",
            "**Increments**: vs legacy FULL −9.0% sim (0 worse); FACN v2 another −5.82% (GATE PASS).",
        ], 12, 6)


def case_ablate(s, lang, img):
    if lang == "zh":
        title(s, "组件消融与 SH20 案例（合并）", "完成率证据 + 并发油门证据")
        headers = ["组件关闭", "典型后果", "解释"]
        rows = [
            ["no_pgcp", "SH10=46, SH12=45, SH11=54", "廊门套件最关键"],
            ["no_stagenet", "SH10=60, SH17=32", "难图姿态头必要"],
            ["no_pgfa", "SH17=20; SH03 sim 翻倍", "互让死锁/空转"],
            ["noFACN", "SH20:660→813 等", "开阔图吞吐损失"],
        ]
        table(s, ML, CT, CW, Inches(3.4), headers, rows, fs=11)
        bullets(s, ML, Inches(4.7), CW, Inches(2.0), [
            "**SH20**：n≈30，TBC≈7 → 闲置池巨大；FACN 抬到 K≈10，sim 813→660。说明最大杠杆常是运营并发，而非微观边权。",
        ], 12, 4)
    else:
        title(s, "Component Ablations and SH20 Case", "Completion evidence + throttle evidence")
        headers = ["Disable", "Typical effect", "Reading"]
        rows = [
            ["no_pgcp", "SH10=46, SH12=45, SH11=54", "Corridor suite critical"],
            ["no_stagenet", "SH10=60, SH17=32", "Pose head necessary"],
            ["no_pgfa", "SH17=20; SH03 sim×2", "Mutual-yield / waste"],
            ["noFACN", "SH20:660→813 etc.", "Open-map throughput loss"],
        ]
        table(s, ML, CT, CW, Inches(3.4), headers, rows, fs=11)
        bullets(s, ML, Inches(4.7), CW, Inches(2.0), [
            "**SH20**: n≈30, TBC≈7 → large idle pool; FACN raises K≈10, sim 813→660. Lever is operational concurrency.",
        ], 12, 4)


def limit_concl(s, lang, img):
    if lang == "zh":
        title(s, "局限与结论", "诚实边界 + 可带走的主张")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**局限**：超难图仍可能压力大；FACN 跨图 held-out 泛化有差距；协议栈有维护成本；不宣称相对 ECBS 的完备最优。",
            "**结论 1**：Lifelong MAPD 适合双时间尺度——Meeting 协商慢变量，tick 修补快冲突。",
            "**结论 2**：有效 AI = StageNet（姿态）+ FACN（油门）+ 几何门控协议；可用消融逐项证明。",
            "**结论 3**：在 VALID∧full100 前提下，本文方法同时取得更低 sim 与更低 wall，位于实用 Pareto 前沿。",
            "**结论 4**：会议触发/输入/输出均可审计——可解释性内生于模块边界，而非事后解释。",
        ], 12, 6)
    else:
        title(s, "Limitations and Conclusions", "Honest bounds + takeaways")
        bullets(s, ML, CT, CW, Inches(5.9), [
            "**Limits**: hardest maps remain stressful; FACN held-out gap; protocol maintenance cost; no optimality claim vs ECBS.",
            "**C1**: Lifelong MAPD needs dual timescales — meetings for slow vars, ticks for fast repair.",
            "**C2**: Effective AI = StageNet + FACN + geometry-gated protocols, each ablatable.",
            "**C3**: Inside VALID∧full100, we improve both sim and wall — a practical Pareto point.",
            "**C4**: Meeting triggers/I/O are auditable by construction.",
        ], 12, 6)


def thanks(s, lang, img):
    card(s, Inches(0.8), Inches(2.2), Inches(11.7), Inches(3.2), LIGHT)
    tx(s, Inches(1), Inches(2.9), Inches(11.3), Inches(0.7),
       "谢谢 · Q&A" if lang == "zh" else "Thank You · Q&A", 32, True, NAVY, PP_ALIGN.CENTER)
    tx(s, Inches(1), Inches(3.9), Inches(11.3), Inches(0.5),
       "Meeting-RH × StageNet × FACN × Geometry-gated Protocols", 14, False, BLUE, PP_ALIGN.CENTER)


ORDER = [
    ("cover", cover),
    ("outline", outline),
    ("problem", problem),
    ("rules", rules),
    ("challenges", challenges),
    ("related1", related1),
    ("related2", related2),
    ("method", method),
    ("overview_fig", overview_fig),
    ("design", design),
    ("meet_fig", meet_fig),
    ("meet_io", meet_io),
    ("meet_steps", meet_steps),
    ("proto_fig", proto_fig),
    ("proto_stack", proto_stack),
    ("tools_gate", tools_gate),
    ("tools_corridor", tools_corridor),
    ("tools_e4_pulse", tools_e4_pulse),
    ("tools_dlo_region", tools_dlo_region),
    ("sn_fig", sn_fig),
    ("sn_ablate", sn_ablate),
    ("facn_fig", facn_fig),
    ("facn_ablate", facn_ablate),
    ("exp_setup", exp_setup),
    ("results", results),
    ("why_best", why_best),
    ("case_ablate", case_ablate),
    ("limit_concl", limit_concl),
    ("thanks", thanks),
]


def build(lang: str):
    img = IMG / lang
    prs = Presentation()
    prs.slide_width = SW
    prs.slide_height = SH
    brand = "Lifelong MAPD · Meeting-RH Hybrid Stack"
    n = len(ORDER)
    for i, (_, fn) in enumerate(ORDER, 1):
        s = news(prs)
        fn(s, lang, img)
        foot(s, i, n, brand)
    out = OUT / (f"Lifelong_MAPD_Hybrid_{'ZH' if lang == 'zh' else 'EN'}.pptx")
    prs.save(str(out))
    # also refresh Conf names for user convenience
    alt = OUT / (f"MioVerse_Conf_{'ZH' if lang == 'zh' else 'EN'}.pptx")
    prs.save(str(alt))
    return out, n


def bounds(path: Path):
    prs = Presentation(str(path))
    sw, sh = int(prs.slide_width), int(prs.slide_height)
    bad = []
    for si, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            L, T, W, H = int(shape.left), int(shape.top), int(shape.width), int(shape.height)
            if L < -2e4 or T < -2e4 or L + W > sw + 2e4 or T + H > sh + 2e4:
                bad.append((si, shape.name))
    return bad


def main():
    assert len(ORDER) <= 35, len(ORDER)
    for lang in ("zh", "en"):
        out, n = build(lang)
        bad = bounds(out)
        print(f"{out.name}: {n} slides", "OVERFLOW" if bad else "OK", bad[:5])
        if bad:
            sys.exit(1)
    print(f"Total pages each: {len(ORDER)}")


if __name__ == "__main__":
    main()
