# -*- coding: utf-8 -*-
"""Generate MioVerse AGV project introduction PPT (30+ slides, 16:9)."""
from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import nsmap
from pptx.oxml import parse_xml
from pptx.util import Inches, Pt, Emu

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "MioVerse_AGV_AI_Intro.pptx"
IMG = ROOT / "generateimage"
PPT_IMG = IMG / "mioverse-ppt"
MRSC_IMG = IMG / "mrsc-decentralized-arch"

# 16:9
SW, SH = Inches(13.333), Inches(7.5)
MARGIN_L, MARGIN_R = Inches(0.55), Inches(0.55)
MARGIN_T, MARGIN_B = Inches(0.45), Inches(0.35)
CONTENT_TOP = Inches(1.15)
CONTENT_BOTTOM = SH - MARGIN_B
CONTENT_W = SW - MARGIN_L - MARGIN_R
SAFE_RIGHT = SW - MARGIN_R
SAFE_BOTTOM = SH - MARGIN_B

# Colors (no purple)
NAVY = RGBColor(0x0B, 0x3D, 0x5C)
TEAL = RGBColor(0x0D, 0x94, 0x88)
CORAL = RGBColor(0xE1, 0x1D, 0x48)
SLATE = RGBColor(0x1E, 0x29, 0x3B)
MUTED = RGBColor(0x64, 0x74, 0x8B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF1, 0xF5, 0xF9)
CARD = RGBColor(0xE2, 0xEE, 0xF2)
ACCENT_BG = RGBColor(0xCC, 0xF0, 0xEB)
ORANGE = RGBColor(0xEA, 0x58, 0x0C)
GREEN = RGBColor(0x05, 0x96, 0x69)


def _set_run(run, size=16, bold=False, color=SLATE, font="Microsoft YaHei"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    # East Asian font
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
    if ea is None:
        from lxml import etree
        ea = etree.SubElement(rPr, "{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
    ea.set("typeface", font)


def add_bg(slide, color=WHITE):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, SH)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    # send to back
    spTree = slide.shapes._spTree
    sp = shape._element
    spTree.remove(sp)
    spTree.insert(2, sp)
    return shape


def add_top_bar(slide):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()
    return bar


def add_footer(slide, page: int, total: int):
    # left brand
    box = slide.shapes.add_textbox(MARGIN_L, SH - Inches(0.38), Inches(6), Inches(0.28))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "MioVerse · AGV Lifelong MAPD · AI Hybrid Stack"
    _set_run(run, 9, False, MUTED)
    # page number
    box2 = slide.shapes.add_textbox(SW - MARGIN_R - Inches(1.2), SH - Inches(0.38), Inches(1.2), Inches(0.28))
    tf2 = box2.text_frame
    p2 = tf2.paragraphs[0]
    p2.alignment = PP_ALIGN.RIGHT
    run2 = p2.add_run()
    run2.text = f"{page} / {total}"
    _set_run(run2, 9, False, MUTED)


def add_title(slide, text: str, subtitle: str | None = None):
    box = slide.shapes.add_textbox(MARGIN_L, Inches(0.28), CONTENT_W, Inches(0.55))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    _set_run(run, 26, True, NAVY)
    if subtitle:
        box2 = slide.shapes.add_textbox(MARGIN_L, Inches(0.78), CONTENT_W, Inches(0.32))
        tf2 = box2.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        run2 = p2.add_run()
        run2.text = subtitle
        _set_run(run2, 12, False, MUTED)


def add_accent_line(slide, top=Inches(1.05)):
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, MARGIN_L, top, Inches(1.4), Inches(0.045)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = TEAL
    line.line.fill.background()


def add_text_box(slide, left, top, width, height, text, size=14, bold=False, color=SLATE, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_after = Pt(2)
    run = p.add_run()
    run.text = text
    _set_run(run, size, bold, color)
    return box


def add_bullets(slide, left, top, width, height, items, size=14, color=SLATE, spacing=6):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(spacing)
        p.level = 0
        run = p.add_run()
        # support bold prefix with **x**
        if item.startswith("**") and "**" in item[2:]:
            end = item.index("**", 2)
            bold_part = item[2:end]
            rest = item[end + 2 :]
            run.text = "• " + bold_part
            _set_run(run, size, True, color)
            if rest:
                run2 = p.add_run()
                run2.text = rest
                _set_run(run2, size, False, color)
        else:
            run.text = "• " + item
            _set_run(run, size, False, color)
    return box


def add_card(slide, left, top, width, height, fill=CARD):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.adjustments[0] = 0.08
    return shape


def add_kpi_card(slide, left, top, width, height, value, label, accent=TEAL):
    add_card(slide, left, top, width, height, LIGHT)
    # accent strip
    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Inches(0.08), height)
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent
    strip.line.fill.background()
    add_text_box(slide, left + Inches(0.2), top + Inches(0.18), width - Inches(0.3), Inches(0.45),
                 value, size=22, bold=True, color=NAVY)
    add_text_box(slide, left + Inches(0.2), top + Inches(0.65), width - Inches(0.3), Inches(0.4),
                 label, size=11, bold=False, color=MUTED)


def add_table(slide, left, top, width, height, headers, rows, font_size=11):
    n_rows = 1 + len(rows)
    n_cols = len(headers)
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = table_shape.table
    # column widths equal
    col_w = int(width / n_cols)
    for c in range(n_cols):
        table.columns[c].width = col_w

    def fill_cell(cell, text, header=False):
        cell.text = ""
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = str(text)
        _set_run(run, font_size, header, WHITE if header else SLATE)
        # vertical center approx
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        # background
        fill = cell.fill
        fill.solid()
        if header:
            fill.fore_color.rgb = NAVY
        else:
            fill.fore_color.rgb = WHITE

    for j, h in enumerate(headers):
        fill_cell(table.cell(0, j), h, header=True)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            fill_cell(table.cell(i + 1, j), val, header=False)
            # zebra
            if i % 2 == 1:
                table.cell(i + 1, j).fill.solid()
                table.cell(i + 1, j).fill.fore_color.rgb = LIGHT
    return table_shape


def add_image(slide, path: Path, left, top, width, height=None):
    if not path.exists():
        add_card(slide, left, top, width, height or Inches(3.2), LIGHT)
        add_text_box(slide, left + Inches(0.2), top + Inches(1.2), width - Inches(0.4), Inches(0.5),
                     f"[missing] {path.name}", size=12, color=MUTED, align=PP_ALIGN.CENTER)
        return None
    if height is None:
        return slide.shapes.add_picture(str(path), left, top, width=width)
    return slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def fit_picture(slide, path: Path, left, top, max_w, max_h):
    """Add picture scaled to fit within box, preserving aspect ratio."""
    if not path.exists():
        add_card(slide, left, top, max_w, max_h, LIGHT)
        add_text_box(slide, left, top + max_h / 2 - Inches(0.2), max_w, Inches(0.4),
                     f"[missing] {path.name}", size=11, color=MUTED, align=PP_ALIGN.CENTER)
        return None
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    aspect = iw / ih
    box_aspect = max_w / max_h
    if aspect >= box_aspect:
        w = max_w
        h = max_w / aspect
    else:
        h = max_h
        w = max_h * aspect
    # center in box
    x = left + (max_w - w) / 2
    y = top + (max_h - h) / 2
    return slide.shapes.add_picture(str(path), x, y, width=w, height=h)


def new_slide(prs):
    blank = prs.slide_layouts[6]  # blank
    slide = prs.slides.add_slide(blank)
    add_bg(slide, WHITE)
    add_top_bar(slide)
    return slide


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------

def build_all():
    prs = Presentation()
    prs.slide_width = SW
    prs.slide_height = SH
    slides_meta = []  # list of (builder_name,) for page count

    builders = [
        slide_cover,
        slide_agenda,
        slide_competition,
        slide_problem,
        slide_constraints,
        slide_scoring_auto,
        slide_scoring_ai40,
        slide_challenge,
        slide_evolution,
        slide_evolution_img,
        slide_baseline_table,
        slide_why_ai,
        slide_rrsc_overview,
        slide_architecture_img,
        slide_meeting_pipeline,
        slide_meeting_img,
        slide_ai_roles,
        slide_stagenet,
        slide_asa,
        slide_facn_concept,
        slide_facn_img,
        slide_facn_afa,
        slide_facn_results,
        slide_facn_sh20,
        slide_pgcp_pgfa,
        slide_rulenet_cgpn_tern,
        slide_training,
        slide_vs_full,
        slide_sh01_40,
        slide_ablation_overview,
        slide_ablation_table,
        slide_ablation_insight,
        slide_vs_hier,
        slide_wall_clock,
        slide_mcp_llm,
        slide_explainability,
        slide_limitations,
        slide_roadmap,
        slide_takeaways,
        slide_thanks,
    ]

    total = len(builders)
    for i, fn in enumerate(builders, 1):
        slide = new_slide(prs)
        fn(slide)
        # cover/thanks: lighter footer or still page num
        add_footer(slide, i, total)
        slides_meta.append(fn.__name__)

    prs.save(str(OUT))
    return OUT, total, slides_meta


def slide_cover(slide):
    # full hero-ish
    add_card(slide, Inches(0.8), Inches(1.6), Inches(11.7), Inches(4.4), LIGHT)
    add_text_box(slide, Inches(1.1), Inches(1.9), Inches(11), Inches(0.4),
                 "2025 Siemens Xcelerator · MioVerse Track", size=14, color=TEAL, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1.1), Inches(2.45), Inches(11), Inches(1.0),
                 "仓库多 AGV 终身路径规划与任务调度", size=32, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1.1), Inches(3.5), Inches(11), Inches(0.45),
                 "MRSC 会议式混合栈 × StageNet × FACN × 可解释协议", size=18, color=SLATE, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1.1), Inches(4.2), Inches(11), Inches(0.4),
                 "突出 AI 深度参与 · 消融证据 · 决赛 AI/LLM 40% 叙事", size=14, color=MUTED, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1.1), Inches(5.2), Inches(11), Inches(0.35),
                 "项目介绍 PPT  ·  约 10–15 分钟讲解", size=12, color=MUTED, align=PP_ALIGN.CENTER)


def slide_agenda(slide):
    add_title(slide, "目录 Agenda", "约 40 页 · 建议讲解 10–15 分钟")
    add_accent_line(slide)
    items = [
        ("01", "竞赛背景与问题定义"),
        ("02", "评分规则：为何必须讲好 AI"),
        ("03", "系统演进：从 A* 到 MRSC"),
        ("04", "AI 架构：StageNet / FACN / 协议层"),
        ("05", "实验结果与 FACN 案例"),
        ("06", "消融实验：组件贡献证据"),
        ("07", "LLM/MCP 集成与可解释性"),
        ("08", "局限、展望与总结"),
    ]
    for i, (num, title) in enumerate(items):
        row, col = divmod(i, 2)
        left = MARGIN_L + col * Inches(6.2)
        top = CONTENT_TOP + Inches(0.15) + row * Inches(1.15)
        add_card(slide, left, top, Inches(5.9), Inches(0.95), LIGHT)
        add_text_box(slide, left + Inches(0.25), top + Inches(0.22), Inches(1.0), Inches(0.5),
                     num, size=22, bold=True, color=TEAL)
        add_text_box(slide, left + Inches(1.3), top + Inches(0.3), Inches(4.3), Inches(0.45),
                     title, size=16, bold=True, color=NAVY)


def slide_competition(slide):
    add_title(slide, "竞赛背景", "2025 Siemens Xcelerator Open Competition — MioVerse / JCIIOT")
    add_accent_line(slide)
    add_kpi_card(slide, MARGIN_L, CONTENT_TOP, Inches(3.9), Inches(1.35), "MioVerse", "赛道 · 仓储智能物流")
    add_kpi_card(slide, MARGIN_L + Inches(4.15), CONTENT_TOP, Inches(3.9), Inches(1.35), "MAPD", "终身多机取卸货调度")
    add_kpi_card(slide, MARGIN_L + Inches(8.3), CONTENT_TOP, Inches(3.9), Inches(1.35), "AI 权重", "决赛专家分 AI/LLM = 40%")

    add_bullets(slide, MARGIN_L, Inches(2.85), CONTENT_W, Inches(3.8), [
        "**目标**：在约束下尽快完成包裹从取货站到分拣口的运输，零碰撞。",
        "**初赛**：自动化评测（送达分 / 急件奖惩 / 碰撞罚分）。",
        "**决赛**：专家主观（AI 深度、算法创新、实用性、代码质量）。",
        "**本项目策略**：自动评测冲完成率与 sim，同时用可解释 AI 模块对齐专家分。",
        "**主系统**：greed_coop_mcts · RRSC/MRSC + FACN（非单纯启发式堆砌）。",
    ], size=15, spacing=10)


def slide_problem(slide):
    add_title(slide, "问题定义：仓库 Lifelong MAPD", "标准赛题设定（可扩展到 SH 压力图更大车队）")
    add_accent_line(slide)
    fit_picture(slide, PPT_IMG / "warehouse-mapd-scene.jpg",
                MARGIN_L, CONTENT_TOP, Inches(6.4), Inches(5.5))
    headers = ["要素", "规格"]
    rows = [
        ["地图", "约 20×20 m · 1 m 格"],
        ["AGV", "12 台（SH 可更大）"],
        ["任务", "~100 包 lifelong"],
        ["取货站", "6 个命名站点"],
        ["分拣口", "16 · 各 4 卸货邻格"],
        ["速度", "1 m/s · 转/装/卸各 1s"],
    ]
    add_table(slide, MARGIN_L + Inches(6.7), CONTENT_TOP, Inches(5.4), Inches(4.6),
              headers, rows, font_size=13)


def slide_constraints(slide):
    add_title(slide, "关键约束", "约束决定算法设计空间")
    add_accent_line(slide)
    cards = [
        ("单车单货", "每台 AGV 同时只运一个包裹"),
        ("FIFO 取货", "各取货站任务必须按序领取"),
        ("站点互斥", "每秒每站最多一车取货"),
        ("冲突定义", "同时同格 / 对向换位均算碰撞"),
        ("运动学", "正交移动；原地转 ±90°/180°"),
        ("急件时限", "超时扣分；按时额外加分"),
    ]
    for i, (t, d) in enumerate(cards):
        r, c = divmod(i, 3)
        left = MARGIN_L + c * Inches(4.15)
        top = CONTENT_TOP + r * Inches(2.5)
        add_card(slide, left, top, Inches(3.95), Inches(2.2), LIGHT)
        add_text_box(slide, left + Inches(0.25), top + Inches(0.4), Inches(3.4), Inches(0.5),
                     t, size=18, bold=True, color=TEAL)
        add_text_box(slide, left + Inches(0.25), top + Inches(1.05), Inches(3.4), Inches(0.8),
                     d, size=14, color=SLATE)


def slide_scoring_auto(slide):
    add_title(slide, "初赛评分：自动化评测", "满分 120（含 20 加分项）")
    add_accent_line(slide)
    headers = ["维度", "规则", "影响"]
    rows = [
        ["送达", "5 分钟内正确卸货：每包 +1", "完成率优先"],
        ["急件", "按时 +10 / 超时 −5", "需要优先级意识"],
        ["碰撞", "每次 −10；车消失任务丢失", "必须 VALID"],
        ["同分", "末包完成时间更早获胜", "压低 makespan/sim"],
    ]
    add_table(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(3.6), headers, rows, font_size=14)
    add_card(slide, MARGIN_L, Inches(5.2), CONTENT_W, Inches(1.4), ACCENT_BG)
    add_text_box(slide, MARGIN_L + Inches(0.3), Inches(5.45), CONTENT_W - Inches(0.5), Inches(0.9),
                 "工程目标：VALID≈100% + full100 尽量高 + sim 尽量低 + wall 可控。",
                 size=16, bold=True, color=NAVY)


def slide_scoring_ai40(slide):
    add_title(slide, "决赛评分：AI/LLM 占专家分 40%", "总分 = 自动评测 60% + 专家评测 40%")
    add_accent_line(slide)
    headers = ["评价维度", "权重", "评委关注点"]
    rows = [
        ["AI / LLM 技术应用", "40%", "深度、场景合理、有效、可解释"],
        ["算法创新", "30%", "相对传统突破、原创性"],
        ["实用性与可扩展性", "20%", "部署成本、计算开销、扩展"],
        ["代码质量", "10%", "规范、可维护"],
    ]
    add_table(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(3.5), headers, rows, font_size=14)
    add_bullets(slide, MARGIN_L, Inches(5.0), CONTENT_W, Inches(1.8), [
        "**因此本 PPT 重点**：不是只晒 sim，而是讲清 AI 模块职责、训练与消融证据。",
        "**叙事口令**：StageNet=姿态 · FACN=油门 · PGCP=喉道红绿灯 · ASA/AFA=几何门控。",
    ], size=14, spacing=8)


def slide_challenge(slide):
    add_title(slide, "核心难点", "为什么纯搜索 / 纯反应式都不够")
    add_accent_line(slide)
    items = [
        ("吞吐 vs 冲突", "多车同时作业提升吞吐，但窄廊/同 dest 极易死锁"),
        ("终身任务流", "不是一次静态 MAPF；任务持续到达、需持续再协调"),
        ("异构场景", "开阔大车队（SH20）与高密迷宫（SH11/17）需要不同并发策略"),
        ("可解释落地", "决赛要求 AI 决策可解释；黑盒端到端难过专家评审"),
        ("无 map-id", "禁止按图硬编码；策略需几何/状态泛化"),
        ("墙钟约束", "ECBS 完备但慢；需要会议窗口 + 局部贪婪的混合"),
    ]
    for i, (t, d) in enumerate(items):
        r, c = divmod(i, 2)
        left = MARGIN_L + c * Inches(6.2)
        top = CONTENT_TOP + r * Inches(1.7)
        add_card(slide, left, top, Inches(5.95), Inches(1.5), LIGHT)
        add_text_box(slide, left + Inches(0.25), top + Inches(0.25), Inches(5.4), Inches(0.4),
                     t, size=16, bold=True, color=NAVY)
        add_text_box(slide, left + Inches(0.25), top + Inches(0.75), Inches(5.4), Inches(0.55),
                     d, size=13, color=SLATE)


def slide_evolution(slide):
    add_title(slide, "系统演进路线", "由浅到深，保留可对照基线")
    add_accent_line(slide)
    steps = [
        ("1", "Baseline", "时空 A* + 贪心指派"),
        ("2", "ECBS", "窗口化冲突树搜索"),
        ("3", "Hier", "M0 ↔ ECBS 快慢切换"),
        ("4", "RL-RH-PP", "Transformer 优先级"),
        ("5", "Horizon", "最大努力冲突视界"),
        ("6", "MRSC+FACN", "会议混合栈 + AI 油门"),
    ]
    for i, (n, t, d) in enumerate(steps):
        left = MARGIN_L + i * Inches(2.1)
        top = CONTENT_TOP + Inches(0.3)
        add_card(slide, left, top, Inches(1.95), Inches(3.6), ACCENT_BG if i == 5 else LIGHT)
        add_text_box(slide, left + Inches(0.1), top + Inches(0.35), Inches(1.75), Inches(0.4),
                     n, size=20, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
        add_text_box(slide, left + Inches(0.1), top + Inches(1.2), Inches(1.75), Inches(0.7),
                     t, size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
        add_text_box(slide, left + Inches(0.1), top + Inches(2.2), Inches(1.75), Inches(1.1),
                     d, size=11, color=SLATE, align=PP_ALIGN.CENTER)
    add_text_box(slide, MARGIN_L, Inches(5.6), CONTENT_W, Inches(0.8),
                 "当前主系统：greed_coop_mcts · policy=rrsc/MRSC · FACN v2 可开关。",
                 size=14, bold=True, color=NAVY)


def slide_evolution_img(slide):
    add_title(slide, "演进可视化", "From Baseline to AI Hybrid Stack")
    add_accent_line(slide)
    fit_picture(slide, PPT_IMG / "system-evolution.jpg",
                MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(5.5))


def slide_baseline_table(slide):
    add_title(slide, "基线对照（SH01–20 × 官方 100）", "来源：ml_research compare_100")
    add_accent_line(slide)
    headers = ["Method", "full100", "VALID", "avg_done", "avg_sim", "avg_wall"]
    rows = [
        ["PIBT", "8/20", "20/20", "63.9", "62811", "40.5s"],
        ["M0 A*", "4/20", "20/20", "44.6", "942†", "1478s"],
        ["PP", "1/20", "7/20", "32.4", "848†", "1716s"],
        ["ECBS", "20/20", "20/20", "100", "7364", "543s"],
        ["Hier", "20/20", "20/20", "100", "4018–4616", "704–1032s"],
    ]
    add_table(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(4.2), headers, rows, font_size=13)
    add_text_box(slide, MARGIN_L, Inches(5.7), CONTENT_W, Inches(0.9),
                 "† 未完成图的 sim 不能直接解读为「更快」。ECBS/Hier 完成率高但墙钟昂贵 → 混合 AI 栈动机。",
                 size=13, color=MUTED)


def slide_why_ai(slide):
    add_title(slide, "为什么引入 AI？", "协议解决正确性，网络解决自适应")
    add_accent_line(slide)
    cols = [
        ("传统搜索", [
            "ECBS 完备但慢",
            "参数对场景敏感",
            "并发上限常静态",
            "难覆盖异构图族",
        ]),
        ("纯学习", [
            "端到端难保证 VALID",
            "可解释性弱",
            "决赛专家风险高",
            "难融入硬约束",
        ]),
        ("本项目混合", [
            "几何协议保底",
            "小网络学关键决策",
            "ASA/AFA 门控",
            "消融可证明贡献",
        ]),
    ]
    for i, (title, bullets) in enumerate(cols):
        left = MARGIN_L + i * Inches(4.15)
        fill = ACCENT_BG if i == 2 else LIGHT
        add_card(slide, left, CONTENT_TOP, Inches(3.95), Inches(5.3), fill)
        add_text_box(slide, left + Inches(0.2), CONTENT_TOP + Inches(0.3), Inches(3.5), Inches(0.5),
                     title, size=18, bold=True, color=NAVY)
        add_bullets(slide, left + Inches(0.2), CONTENT_TOP + Inches(1.1), Inches(3.5), Inches(3.8),
                    bullets, size=14, spacing=12)


def slide_rrsc_overview(slide):
    add_title(slide, "主策略 RRSC / MRSC", "Road-Rights → Stage 契约 → 优先级栈贪婪")
    add_accent_line(slide)
    add_bullets(slide, MARGIN_L, CONTENT_TOP, Inches(6.2), Inches(5.5), [
        "**RRSC**：全局路权/引导 → 阶段契约 → 局部 PIBT 式推进。",
        "**MRSC**：周期 Meeting（优先级 + StageNet + 窗 PP）+ 会间 follow/gap。",
        "**无窄巷硬编码、无 map-id**：开图与密图同一逻辑。",
        "**默认协议栈**：ASA + FG-CEBI + Multi-Band TBC + E4 + DLO + Pad-egress + Contested-dest。",
        "**AI 挂点**：StageNet / FACN /（可选）RuleNet·CGPN·TERN。",
        "**入口**：`python -m greed_coop_mcts --policy rrsc`。",
    ], size=15, spacing=10)
    fit_picture(slide, MRSC_IMG / "02-three-layers.jpg",
                MARGIN_L + Inches(6.4), CONTENT_TOP, Inches(5.6), Inches(5.3))


def slide_architecture_img(slide):
    add_title(slide, "AI 管线总览图", "TBC → FACN → Meeting → PGCP → Local")
    add_accent_line(slide)
    fit_picture(slide, PPT_IMG / "mrsc-ai-pipeline.jpg",
                MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(5.5))


def slide_meeting_pipeline(slide):
    add_title(slide, "Meeting 决策管线", "每 MEET_H≈12 tick 或 stall 触发")
    add_accent_line(slide)
    steps = [
        ("① 感知", "几何 + busy/stall\n剩余任务/冲突热"),
        ("② ASA", "决定 StageNet\n或 teacher"),
        ("③ Stage", "FINAL/APPROACH\nCLEAR + 优先级"),
        ("④ PGCP", "同 dest 门控\npad-egress / E4"),
        ("⑤ 规划", "窗 PP + DLO\nlegalize"),
        ("⑥ 会间", "跟契约 + PGFA\naltruism / gap"),
    ]
    for i, (t, d) in enumerate(steps):
        left = MARGIN_L + (i % 3) * Inches(4.15)
        top = CONTENT_TOP + (i // 3) * Inches(2.55)
        add_card(slide, left, top, Inches(3.95), Inches(2.3), LIGHT)
        add_text_box(slide, left + Inches(0.25), top + Inches(0.35), Inches(3.4), Inches(0.45),
                     t, size=18, bold=True, color=TEAL)
        add_text_box(slide, left + Inches(0.25), top + Inches(1.0), Inches(3.4), Inches(1.0),
                     d, size=14, color=SLATE)


def slide_meeting_img(slide):
    add_title(slide, "Meeting Pipeline 示意图", "已有架构素材复用")
    add_accent_line(slide)
    fit_picture(slide, MRSC_IMG / "09-meeting-pipeline.jpg",
                MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(5.5))


def slide_ai_roles(slide):
    add_title(slide, "AI 模块职责一览", "一个问题一个网络 / 协议")
    add_accent_line(slide)
    headers = ["模块", "类型", "学什么 / 管什么", "地位"]
    rows = [
        ["StageNet", "MLP", "阶段姿态 + 标量优先级", "难图关键"],
        ["FACN", "MLP", "并发油门 Δ → K", "主创新"],
        ["ASA / AFA", "门控", "何时信任网络", "安全阀"],
        ["PGCP / PGFA", "协议", "廊门 / 焦点利他", "完成率关键"],
        ["RuleNet", "Transformer", "全局方向场", "默认 OFF"],
        ["CGPN / TERN", "小网", "同 dest 微序 / 边残差", "可选"],
    ]
    add_table(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(5.2), headers, rows, font_size=13)


def slide_stagenet(slide):
    add_title(slide, "StageNet：姿态选择网络", "输出 FINAL / APPROACH / CLEAR + 优先级")
    add_accent_line(slide)
    fit_picture(slide, PPT_IMG / "stagenet-facn-nets.jpg",
                MARGIN_L, CONTENT_TOP, Inches(7.0), Inches(5.3))
    add_bullets(slide, MARGIN_L + Inches(7.2), CONTENT_TOP, Inches(4.8), Inches(5.3), [
        "**输入**：agent 特征。",
        "**输出**：候选阶段打分 + 优先级。",
        "**训练**：IL（stage CE + listwise pri）。",
        "**消融**：SH10/SH17 关闭后完成率崩。",
        "**与 FACN 正交**：姿态 ≠ 并发。",
    ], size=14, spacing=12)


def slide_asa(slide):
    add_title(slide, "ASA：自适应 StageNet 授权", "先几何门，再决定是否用网络")
    add_accent_line(slide)
    add_card(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(2.0), ACCENT_BG)
    add_text_box(slide, MARGIN_L + Inches(0.35), CONTENT_TOP + Inches(0.55), CONTENT_W - Inches(0.6), Inches(1.0),
                 "ASA 仅用 (n_agents, near_pad_deg2) 等几何量，决定 StageNet 或 teacher。",
                 size=16, bold=True, color=NAVY)
    add_bullets(slide, MARGIN_L, Inches(3.5), CONTENT_W, Inches(3.2), [
        "**小 fleets**：更常走 teacher，降低噪声。",
        "**中等规模（约 10–14）**：更多启用 StageNet。",
        "**极端迷宫 / 中开图**：特判，避免错误姿态。",
        "**设计哲学**：网络不是永远在线；可解释门控 = 决赛加分项。",
    ], size=15, spacing=10)


def slide_facn_concept(slide):
    add_title(slide, "FACN：车队并发油门", "Fleet Activation & Concurrency Network")
    add_accent_line(slide)
    add_bullets(slide, MARGIN_L, CONTENT_TOP, Inches(6.3), Inches(5.5), [
        "**问题**：TBC 给出粗粒度 K，开阔大车队会永久闲置。",
        "**输出**：Δ∈{−2..+4}，K=clip(K_TBC+Δ)。",
        "**作用点**：只改 claim / max_assigned 阀门，不改 BFS。",
        "**结构**：静态 10 维 + 动态 12 维 → 小 MLP 分类。",
        "**训练**：离线扫 K 得 oracle Δ* → 行为克隆。",
        "**SH20**：30 车、TBC≈7 → FACN 抬到 K≈10，sim 显著下降。",
    ], size=15, spacing=10)
    add_card(slide, MARGIN_L + Inches(6.6), CONTENT_TOP, Inches(5.5), Inches(5.3), LIGHT)
    add_text_box(slide, MARGIN_L + Inches(6.85), CONTENT_TOP + Inches(0.4), Inches(5.0), Inches(0.5),
                 "一句话", size=14, bold=True, color=TEAL)
    add_text_box(slide, MARGIN_L + Inches(6.85), CONTENT_TOP + Inches(1.2), Inches(5.0), Inches(3.5),
                 "StageNet = 姿态\nFACN = 油门\nPGCP dest-gate = 红绿灯\n\n宏观阀门 × 微观准入 × 姿态选择",
                 size=18, bold=True, color=NAVY)


def slide_facn_img(slide):
    add_title(slide, "FACN 概念图", "拥堵限流 · 开阔放活")
    add_accent_line(slide)
    fit_picture(slide, PPT_IMG / "facn-throttle-concept.jpg",
                MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(5.5))


def slide_facn_afa(slide):
    add_title(slide, "AFA：Adaptive FACN Authority", "与 ASA 同哲学的几何安全阀")
    add_accent_line(slide)
    headers = ["规则", "条件", "效果"]
    rows = [
        ["允许抬升", "开阔：near_pad 低且 free/N 高", "SH20 类拉高并发"],
        ["允许降低", "stall / 喉道占用高 / no_prog", "对齐 LoRR 减活思想"],
        ["否则", "几何不确定", "Δ=0，回退纯 TBC"],
        ["门禁图", "SH01/02/03/08", "Δ 受限，防回归"],
        ["粘滞", "TTL + MIN_GAP", "防止 thrash"],
    ]
    add_table(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(4.6), headers, rows, font_size=13)
    add_text_box(slide, MARGIN_L, Inches(6.0), CONTENT_W, Inches(0.6),
                 "SH39 曾因错误降并发进入死亡螺旋 → 修复后回退 TBC，保留 SH20 收益。",
                 size=13, color=CORAL)


def slide_facn_results(slide):
    add_title(slide, "FACN v2 实验结果", "SH01–40 · turn-aware motion")
    add_accent_line(slide)
    add_kpi_card(slide, MARGIN_L, CONTENT_TOP, Inches(3.9), Inches(1.4), "−5.82%", "avg sim vs noFACN")
    add_kpi_card(slide, MARGIN_L + Inches(4.15), CONTENT_TOP, Inches(3.9), Inches(1.4), "20 / 13 / 5", "better / same / worse")
    add_kpi_card(slide, MARGIN_L + Inches(8.3), CONTENT_TOP, Inches(3.9), Inches(1.4), "GATE PASS", "SH01/02/03/08 ≤15%")

    headers = ["Slot", "noFACN", "FACN", "Δsim", "K/Δ"]
    rows = [
        ["SH18", "1743", "1194", "−549", "7/+1"],
        ["SH24", "1036", "602", "−434", "10/+4"],
        ["SH25", "1626", "1269", "−357", "8/+2"],
        ["SH16", "1533", "1206", "−327", "7/+1"],
        ["SH20", "813", "660", "−153", "10/+3"],
    ]
    add_table(slide, MARGIN_L, Inches(2.9), CONTENT_W, Inches(3.7), headers, rows, font_size=13)


def slide_facn_sh20(slide):
    add_title(slide, "案例深挖：SH20", "开阔地板 + 大车队 = FACN 主战场")
    add_accent_line(slide)
    add_kpi_card(slide, MARGIN_L, CONTENT_TOP, Inches(3.9), Inches(1.5), "n≈30", "车队规模")
    add_kpi_card(slide, MARGIN_L + Inches(4.15), CONTENT_TOP, Inches(3.9), Inches(1.5), "TBC≈7", "静态并发上限")
    add_kpi_card(slide, MARGIN_L + Inches(8.3), CONTENT_TOP, Inches(3.9), Inches(1.5), "K≈10", "FACN 抬升后")

    add_bullets(slide, MARGIN_L, Inches(3.0), CONTENT_W, Inches(3.5), [
        "**闲置池**：约 23 台车在静态 cap 下可能长期不参与 → 吞吐浪费。",
        "**FACN 动作**：冷启动即根据几何抬升 Δ（如 +3/+4）。",
        "**结果**：813→660（v2）或更早快照 883→671；完成率保持 100。",
        "**启示**：AI 最有杠杆的往往不是「走哪」，而是「允许多少车同时忙」。",
    ], size=15, spacing=10)


def slide_pgcp_pgfa(slide):
    add_title(slide, "协议智能：PGCP / PGFA", "非神经网络，但是关键「智能决策」")
    add_accent_line(slide)
    add_card(slide, MARGIN_L, CONTENT_TOP, Inches(6.0), Inches(5.3), LIGHT)
    add_text_box(slide, MARGIN_L + Inches(0.3), CONTENT_TOP + Inches(0.3), Inches(5.4), Inches(0.45),
                 "PGCP · 廊道协议", size=18, bold=True, color=TEAL)
    add_bullets(slide, MARGIN_L + Inches(0.3), CONTENT_TOP + Inches(1.0), Inches(5.4), Inches(4.0), [
        "同 dest 门控，避免多车挤同一卸点。",
        "pad-egress / CEBI / E4 sticky 外撤。",
        "消融：SH07/10/11/12/17 关则掉完成率。",
        "**最关键协议层组件。**",
    ], size=14, spacing=10)

    add_card(slide, MARGIN_L + Inches(6.3), CONTENT_TOP, Inches(6.0), Inches(5.3), LIGHT)
    add_text_box(slide, MARGIN_L + Inches(6.6), CONTENT_TOP + Inches(0.3), Inches(5.4), Inches(0.45),
                 "PGFA · 焦点利他", size=18, bold=True, color=TEAL)
    add_bullets(slide, MARGIN_L + Inches(6.6), CONTENT_TOP + Inches(1.0), Inches(5.4), Inches(4.0), [
        "单一受益人 + 进度门 + 非对称让路。",
        "防止无条件利他导致死锁。",
        "SH11/17 关则掉完成；SH02/03 sim 暴涨。",
        "与 StageNet 互补：姿态 + 礼让。",
    ], size=14, spacing=10)


def slide_rulenet_cgpn_tern(slide):
    add_title(slide, "探索模块与负结果", "诚实展示：不是每个网络都有效")
    add_accent_line(slide)
    headers = ["模块", "设计意图", "现状结论"]
    rows = [
        ["GlobalRuleNet", "Transformer 全图方向场", "默认 OFF；消融 done/sim=full"],
        ["CGPN", "同 dest 2–3 车微优先级", "可选；部分图曾否决"],
        ["TERN", "边代价残差", "SoftGuidance 曾与 BASE 打平"],
        ["CAPRI（提案）", "会议注意力准入序", "未来工作，非默认主干"],
    ]
    add_table(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(3.8), headers, rows, font_size=13)
    add_card(slide, MARGIN_L, Inches(5.3), CONTENT_W, Inches(1.35), ACCENT_BG)
    add_text_box(slide, MARGIN_L + Inches(0.3), Inches(5.55), CONTENT_W - Inches(0.5), Inches(0.9),
                 "负结果也是 AI 研究叙事：RuleNet 证明「方向场≠本题杠杆」，从而把算力聚焦 StageNet+FACN。",
                 size=14, bold=True, color=NAVY)


def slide_training(slide):
    add_title(slide, "训练方法汇总", "全部小模型 · 以模仿学习为主")
    add_accent_line(slide)
    headers = ["模块", "方法", "标签 / 目标", "备注"]
    rows = [
        ["GlobalRuleNet", "IL", "全局方向场教师", "默认关闭"],
        ["StageNet", "IL", "stage CE + listwise pri", "难图必需"],
        ["FACN", "IL + Oracle 扫参", "Δ*=K*−K_TBC", "主创新"],
        ["CGPN", "短窗锦标赛蒸馏", "pairwise 相对序", "可选"],
        ["TERN", "回归", "解析边代价", "可选"],
        ["RL-RH-PP", "BC + PPO", "Transformer 优先级", "前代探索"],
    ]
    add_table(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(5.2), headers, rows, font_size=12)


def slide_vs_full(slide):
    add_title(slide, "协议栈 vs 旧 FULL（SH01–20）", "ASA + FG-CEBI + Multi-Band TBC + E4")
    add_accent_line(slide)
    add_kpi_card(slide, MARGIN_L, CONTENT_TOP, Inches(3.9), Inches(1.45), "−9.0%", "avg sim 1877→1708")
    add_kpi_card(slide, MARGIN_L + Inches(4.15), CONTENT_TOP, Inches(3.9), Inches(1.45), "0 / 20", "劣于旧系统的图数")
    add_kpi_card(slide, MARGIN_L + Inches(8.3), CONTENT_TOP, Inches(3.9), Inches(1.45), "14 / 20", "严格更优")

    headers = ["Slot", "增益 Δsim", "备注"]
    rows = [
        ["SH03", "−697", "大增益"],
        ["SH11", "−677", "Dense-TBC"],
        ["SH07", "−586", "走廊争用"],
        ["SH16", "−448", "严格优"],
        ["SH12", "−90", "E4 sticky"],
    ]
    add_table(slide, MARGIN_L, Inches(2.95), CONTENT_W, Inches(3.6), headers, rows, font_size=13)


def slide_sh01_40(slide):
    add_title(slide, "当前主结果：SH01–40（FACN v2）", "来源：greed_coop_mcts/results/SUMMARY.md")
    add_accent_line(slide)
    add_kpi_card(slide, MARGIN_L, CONTENT_TOP, Inches(3.0), Inches(1.35), "40/40", "VALID")
    add_kpi_card(slide, MARGIN_L + Inches(3.2), CONTENT_TOP, Inches(3.0), Inches(1.35), "39/40", "full100")
    add_kpi_card(slide, MARGIN_L + Inches(6.4), CONTENT_TOP, Inches(3.0), Inches(1.35), "~1481", "满图 avg sim")
    add_kpi_card(slide, MARGIN_L + Inches(9.6), CONTENT_TOP, Inches(3.0), Inches(1.35), "~22s", "满图 avg wall")

    headers = ["Slot", "done", "sim", "wall", "K/Δ"]
    rows = [
        ["SH20", "100", "660", "21.0", "10/+3"],
        ["SH24", "100", "602", "14.0", "10/+4"],
        ["SH13", "100", "697", "17.9", "8/+2"],
        ["SH31", "100", "3437", "71.2", "5/0"],
        ["SH17", "71", "83850", "1245", "6/0"],
    ]
    add_table(slide, MARGIN_L, Inches(2.8), CONTENT_W, Inches(3.7), headers, rows, font_size=13)


def slide_ablation_overview(slide):
    add_title(slide, "消融实验总览", "leave-one-component-out · policy=rrsc")
    add_accent_line(slide)
    fit_picture(slide, PPT_IMG / "ablation-concept.jpg",
                MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(5.5))


def slide_ablation_table(slide):
    add_title(slide, "消融：完成率掉队一览", "相对 full · SH01–20")
    add_accent_line(slide)
    headers = ["Slot", "掉队变体 (done)", "解读"]
    rows = [
        ["SH07", "no_pgcp=99", "走廊争用末局差一口气"],
        ["SH10", "no_stagenet=60 / no_pgcp=46", "窄廊同 dest；两者必需"],
        ["SH11", "no_pgcp=54 / no_pgfa=73", "高密：协议更关键"],
        ["SH12", "no_pgcp=45", "最依赖 PGCP"],
        ["SH17", "no_stagenet=32 / no_pgfa=20", "最难图；去 PGFA 最惨"],
    ]
    add_table(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(4.2), headers, rows, font_size=13)
    add_text_box(slide, MARGIN_L, Inches(5.7), CONTENT_W, Inches(0.8),
                 "RuleNet：20/20 图 done/sim 与 full 完全一致 → 对质量无贡献。",
                 size=14, bold=True, color=NAVY)


def slide_ablation_insight(slide):
    add_title(slide, "消融结论：组件重要性排序", "用证据回答「AI 有没有用」")
    add_accent_line(slide)
    ranks = [
        ("1", "PGCP", "完成率最关键协议", CORAL),
        ("2", "StageNet", "难图姿态不可或缺", ORANGE),
        ("3", "PGFA", "防死锁 + 压 sim", TEAL),
        ("4", "FACN", "开阔图吞吐杠杆", TEAL),
        ("5", "RuleNet", "当前无质量收益", MUTED),
    ]
    for i, (n, name, desc, color) in enumerate(ranks):
        top = CONTENT_TOP + i * Inches(0.95)
        add_card(slide, MARGIN_L, top, CONTENT_W, Inches(0.85), LIGHT)
        strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, MARGIN_L, top, Inches(0.12), Inches(0.85))
        strip.fill.solid()
        strip.fill.fore_color.rgb = color
        strip.line.fill.background()
        add_text_box(slide, MARGIN_L + Inches(0.35), top + Inches(0.2), Inches(0.7), Inches(0.45),
                     n, size=20, bold=True, color=color)
        add_text_box(slide, MARGIN_L + Inches(1.2), top + Inches(0.22), Inches(3.0), Inches(0.45),
                     name, size=18, bold=True, color=NAVY)
        add_text_box(slide, MARGIN_L + Inches(4.5), top + Inches(0.25), Inches(7.5), Inches(0.45),
                     desc, size=15, color=SLATE)


def slide_vs_hier(slide):
    add_title(slide, "早期 MRSC vs Hier", "展示演进中的对照（非最终栈）")
    add_accent_line(slide)
    headers = ["指标", "MRSC（早期）", "Hier"]
    rows = [
        ["full100", "16/20", "20/20"],
        ["avg done", "91.5", "100"],
        ["avg sim", "3390", "4616"],
        ["满图更快", "12 图", "—"],
        ["满图更慢", "4 图", "—"],
    ]
    add_table(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(4.0), headers, rows, font_size=14)
    add_text_box(slide, MARGIN_L, Inches(5.5), CONTENT_W, Inches(1.0),
                 "随后用 PGCP/PGFA/TBC/E4/FACN 补齐难图完成率，并继续压 sim。",
                 size=14, color=MUTED)


def slide_wall_clock(slide):
    add_title(slide, "墙钟优势", "实用可部署性（决赛 20% 维度）")
    add_accent_line(slide)
    headers = ["系统", "完成能力", "典型墙钟", "点评"]
    rows = [
        ["ECBS", "20/20 full", "~543s avg", "慢但稳"],
        ["Hier", "20/20 full", "704–1032s", "更慢"],
        ["MRSC+FACN", "39/40 full*", "~22s 满图均", "快两个数量级"],
    ]
    add_table(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(3.2), headers, rows, font_size=14)
    add_bullets(slide, MARGIN_L, Inches(4.7), CONTENT_W, Inches(2.0), [
        "*SH17 仍是压力图；其余满图 wall 多在十几到几十秒。",
        "混合栈 = 会议窗口搜索 + 局部贪婪 + 小神经网络，适合在线部署叙事。",
    ], size=14, spacing=8)


def slide_mcp_llm(slide):
    add_title(slide, "LLM / MCP 集成", "覆盖决赛「AI/LLM 技术应用」叙事")
    add_accent_line(slide)
    add_card(slide, MARGIN_L, CONTENT_TOP, Inches(6.0), Inches(5.3), LIGHT)
    add_text_box(slide, MARGIN_L + Inches(0.3), CONTENT_TOP + Inches(0.3), Inches(5.4), Inches(0.45),
                 "MCP 工具层", size=18, bold=True, color=TEAL)
    add_bullets(slide, MARGIN_L + Inches(0.3), CONTENT_TOP + Inches(1.0), Inches(5.4), Inches(4.0), [
        "FastMCP：`calculate(agv_position, agv_task)`",
        "LLM 可调用完整规划仿真管线",
        "写出轨迹 CSV，便于人机协作",
        "位置：`mcp/` 目录",
    ], size=14, spacing=12)

    add_card(slide, MARGIN_L + Inches(6.3), CONTENT_TOP, Inches(6.0), Inches(5.3), ACCENT_BG)
    add_text_box(slide, MARGIN_L + Inches(6.6), CONTENT_TOP + Inches(0.3), Inches(5.4), Inches(0.45),
                 "与核心 AI 的关系", size=18, bold=True, color=TEAL)
    add_bullets(slide, MARGIN_L + Inches(6.6), CONTENT_TOP + Inches(1.0), Inches(5.4), Inches(4.0), [
        "大模型不直接做逐步规划",
        "性能来自 StageNet/FACN/协议",
        "LLM 负责调用、解释、编排",
        "「可调用 + 可解释」对齐 40% 权重",
    ], size=14, spacing=12)


def slide_explainability(slide):
    add_title(slide, "可解释性设计", "专家评审可读的决策语言")
    add_accent_line(slide)
    items = [
        ("几何门控", "ASA / AFA 用可观察几何量决定是否信任网络"),
        ("角色解耦", "姿态 / 油门 / 廊门 / 微序 各管一事"),
        ("无 map-id", "禁止按图号特判，强调泛化与诚实性"),
        ("消融表格", "每个组件都能用 done/sim 说清贡献"),
        ("会议日志", "Meeting 周期决策可追溯（阶段、K、门控）"),
        ("负结果公开", "RuleNet OFF、CGPN 否决也写进文档"),
    ]
    for i, (t, d) in enumerate(items):
        r, c = divmod(i, 2)
        left = MARGIN_L + c * Inches(6.2)
        top = CONTENT_TOP + r * Inches(1.7)
        add_card(slide, left, top, Inches(5.95), Inches(1.5), LIGHT)
        add_text_box(slide, left + Inches(0.25), top + Inches(0.25), Inches(5.4), Inches(0.4),
                     t, size=16, bold=True, color=NAVY)
        add_text_box(slide, left + Inches(0.25), top + Inches(0.75), Inches(5.4), Inches(0.55),
                     d, size=13, color=SLATE)


def slide_limitations(slide):
    add_title(slide, "局限与诚实边界", "增强可信度")
    add_accent_line(slide)
    add_bullets(slide, MARGIN_L, CONTENT_TOP, CONTENT_W, Inches(5.5), [
        "**SH17 类超难图**：仍可能无法 full100 / sim 爆炸。",
        "**FACN held-out**：训 01–20 测 21–40 时 BEST 仅 4/20 → 泛化仍有差距。",
        "**RuleNet**：方向场路线在当前运动模型下未体现质量收益。",
        "**协议敏感性**：PGCP/PGFA 关断会伤完成率 → 需要维护成本。",
        "**非完备**：相对 ECBS，混合栈不承诺最优，只承诺实用高质量。",
    ], size=16, spacing=14)


def slide_roadmap(slide):
    add_title(slide, "展望 Roadmap", "下一步可讲的创新点")
    add_accent_line(slide)
    cards = [
        ("CAPRI", "会议时 contested pad 注意力排序准入（谁先）"),
        ("FACN v1.5", "区域容量 / GRAND-lite 分布再平衡"),
        ("Held-out 强化", "合成课程图 + 更好特征，提升跨图泛化"),
        ("LLM 解释层", "把 Meeting 决策自动生成自然语言报告"),
    ]
    for i, (t, d) in enumerate(cards):
        r, c = divmod(i, 2)
        left = MARGIN_L + c * Inches(6.2)
        top = CONTENT_TOP + r * Inches(2.5)
        add_card(slide, left, top, Inches(5.95), Inches(2.2), LIGHT)
        add_text_box(slide, left + Inches(0.3), top + Inches(0.4), Inches(5.3), Inches(0.5),
                     t, size=20, bold=True, color=TEAL)
        add_text_box(slide, left + Inches(0.3), top + Inches(1.1), Inches(5.3), Inches(0.8),
                     d, size=14, color=SLATE)


def slide_takeaways(slide):
    add_title(slide, "关键 Takeaways", "10 分钟演讲收束")
    add_accent_line(slide)
    points = [
        "从 A*/ECBS/Hier 演进到 MRSC 会议式混合栈。",
        "有效 AI = StageNet（姿态）+ FACN（油门）+ 可解释协议。",
        "SH01–20 vs FULL：avg sim −9%，0 劣图。",
        "FACN v2：SH01–40 avg sim −5.8%，GATE PASS。",
        "消融证明：PGCP/StageNet 保完成率；RuleNet 无质量收益。",
        "MCP 让 LLM 可调用求解器，对齐决赛 AI/LLM 40%。",
    ]
    for i, t in enumerate(points):
        top = CONTENT_TOP + i * Inches(0.8)
        add_card(slide, MARGIN_L, top, CONTENT_W, Inches(0.7), LIGHT if i % 2 == 0 else ACCENT_BG)
        add_text_box(slide, MARGIN_L + Inches(0.3), top + Inches(0.15), CONTENT_W - Inches(0.5), Inches(0.45),
                     f"{i+1}.  {t}", size=15, bold=True, color=NAVY)


def slide_thanks(slide):
    add_card(slide, Inches(0.8), Inches(1.8), Inches(11.7), Inches(4.0), LIGHT)
    add_text_box(slide, Inches(1.1), Inches(2.3), Inches(11), Inches(0.8),
                 "谢谢 · Q & A", size=36, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1.1), Inches(3.4), Inches(11), Inches(0.5),
                 "MioVerse AGV · MRSC × StageNet × FACN", size=18, color=TEAL, align=PP_ALIGN.CENTER)
    add_text_box(slide, Inches(1.1), Inches(4.2), Inches(11), Inches(0.8),
                 "代码：greed_coop_mcts/  ·  结果：results/  ·  配图：generateimage/",
                 size=13, color=MUTED, align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------------------
# Bounds checker
# ---------------------------------------------------------------------------

def check_bounds(pptx_path: Path, tol_emu: int = 20000):
    """Return list of overflow issues."""
    prs = Presentation(str(pptx_path))
    sw, sh = int(prs.slide_width), int(prs.slide_height)
    issues = []
    for si, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            left = int(shape.left)
            top = int(shape.top)
            width = int(shape.width)
            height = int(shape.height)
            right = left + width
            bottom = top + height
            name = getattr(shape, "name", "?")
            if left < -tol_emu:
                issues.append((si, name, f"left={left}"))
            if top < -tol_emu:
                issues.append((si, name, f"top={top}"))
            if right > sw + tol_emu:
                issues.append((si, name, f"right={right}>{sw}"))
            if bottom > sh + tol_emu:
                issues.append((si, name, f"bottom={bottom}>{sh}"))
    return issues, len(prs.slides)


if __name__ == "__main__":
    out, total, meta = build_all()
    print(f"Saved: {out}")
    print(f"Slides: {total}")
    issues, n = check_bounds(out)
    if issues:
        print(f"OVERFLOW issues: {len(issues)}")
        for it in issues[:50]:
            print(" ", it)
        sys.exit(1)
    else:
        print("Bounds check: OK (no overflow)")
        sys.exit(0)
