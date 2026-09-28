"""Tkinter GUI for MioVerse AGV warehouse map editing."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from pathlib import Path
from typing import List, Optional, Tuple

from .model import (
    DEFAULT_DROPOFF_NAMES,
    DEFAULT_PICKUP_NAMES,
    GRID_MAX,
    GRID_MIN,
    GRID_SIZE,
    PITCH_ARROWS,
    PITCHES,
    MapDocument,
    TaskSpec,
    default_custom_maps_dir,
    default_exports_dir,
    load_slot_document,
    next_task_id,
    official_l100_for_slot,
    slot_id,
    slot_index,
)

# Visual language aligned with curriculum_shape map_overview.py
COLORS = {
    "empty": "#F5F5F0",
    "grid": "#DDDDDD",
    "obstacle": "#404048",
    "pickup": "#4C8BF5",
    "pickup_edge": "#1a4a9a",
    "dropoff": "#3CB371",
    "dropoff_edge": "#1a5c38",
    "agv": "#E8A838",
    "agv_edge": "#8B5A00",
    "hover": "#CFE8FF",
    "label": "#222222",
    "bg": "#ECEAE4",
    "panel": "#F8F7F4",
    "urgent": "#C0392B",
}

MODES = (
    ("obstacle", "障碍 Obst"),
    ("pickup", "取货 Pickup"),
    ("dropoff", "卸货 Drop"),
    ("agv", "AGV 出生点"),
    ("erase", "擦除 Erase"),
)


class TaskEditDialog(simpledialog.Dialog):
    """Create / edit one task: pickup → dropoff + urgency."""

    def __init__(
        self,
        parent,
        title: str,
        *,
        pickup_names: List[str],
        dropoff_names: List[str],
        initial: Optional[TaskSpec] = None,
        existing: Optional[List[TaskSpec]] = None,
    ) -> None:
        self.pickup_names = pickup_names or list(DEFAULT_PICKUP_NAMES)
        self.dropoff_names = dropoff_names or list(DEFAULT_DROPOFF_NAMES)
        self.initial = initial
        self.existing = existing or []
        self.result_task: Optional[TaskSpec] = None
        super().__init__(parent, title=title)

    def body(self, master):
        tk.Label(master, text="任务 ID").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.tid_var = tk.StringVar(
            value=(self.initial.task_id if self.initial else "")
        )
        tk.Entry(master, textvariable=self.tid_var, width=28).grid(
            row=0, column=1, sticky=tk.EW, pady=2
        )

        tk.Label(master, text="取货站台 (start_point)").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        pu0 = (
            self.initial.start_point
            if self.initial
            else (self.pickup_names[0] if self.pickup_names else "")
        )
        self.pu_var = tk.StringVar(value=pu0)
        self.pu_combo = ttk.Combobox(
            master, textvariable=self.pu_var, values=self.pickup_names, width=26
        )
        self.pu_combo.grid(row=1, column=1, sticky=tk.EW, pady=2)

        tk.Label(master, text="卸货目的地 (end_point)").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        do0 = (
            self.initial.end_point
            if self.initial
            else (self.dropoff_names[0] if self.dropoff_names else "")
        )
        self.do_var = tk.StringVar(value=do0)
        self.do_combo = ttk.Combobox(
            master, textvariable=self.do_var, values=self.dropoff_names, width=26
        )
        self.do_combo.grid(row=2, column=1, sticky=tk.EW, pady=2)

        self.urgent_var = tk.BooleanVar(
            value=bool(self.initial and self.initial.is_urgent())
        )
        tk.Checkbutton(
            master,
            text="紧急 / Urgent（priority）",
            variable=self.urgent_var,
            command=self._on_urgent_toggle,
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=4)

        tk.Label(master, text="剩余时限 remaining_time").grid(
            row=4, column=0, sticky=tk.W, pady=2
        )
        rt = ""
        if self.initial and self.initial.remaining_time is not None:
            rt = str(self.initial.remaining_time)
        self.rt_var = tk.StringVar(value=rt)
        self.rt_entry = tk.Entry(master, textvariable=self.rt_var, width=28)
        self.rt_entry.grid(row=4, column=1, sticky=tk.EW, pady=2)

        tk.Label(
            master,
            text="说明：站台在左侧地图上放置；此处只配置「取货→卸货 + 是否紧急」。",
            wraplength=360,
            fg="#555",
            justify=tk.LEFT,
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))

        if not self.tid_var.get() and self.pu_var.get():
            self.tid_var.set(next_task_id(self.pu_var.get(), self.existing))
        self._on_urgent_toggle()
        return self.pu_combo

    def _on_urgent_toggle(self) -> None:
        if self.urgent_var.get():
            if not self.rt_var.get().strip():
                self.rt_var.set("120")
            self.rt_entry.configure(state=tk.NORMAL)
        else:
            self.rt_entry.configure(state=tk.NORMAL)

    def validate(self) -> bool:
        pu = self.pu_var.get().strip()
        do = self.do_var.get().strip()
        tid = self.tid_var.get().strip()
        if not pu or not do:
            messagebox.showwarning("任务", "请填写取货站台与卸货目的地。", parent=self)
            return False
        if not tid:
            tid = next_task_id(pu, self.existing)
            self.tid_var.set(tid)
        # unique id (allow keeping own id when editing)
        for t in self.existing:
            if t.task_id != tid:
                continue
            if self.initial is not None and t is self.initial:
                continue
            messagebox.showwarning("任务", f"任务 ID「{tid}」已存在。", parent=self)
            return False
        if self.urgent_var.get():
            rt_s = self.rt_var.get().strip()
            if not rt_s:
                messagebox.showwarning(
                    "任务", "紧急任务需要填写 remaining_time。", parent=self
                )
                return False
            try:
                int(rt_s)
            except ValueError:
                messagebox.showwarning("任务", "remaining_time 须为整数。", parent=self)
                return False
        return True

    def apply(self) -> None:
        urgent = self.urgent_var.get()
        rt: Optional[int] = None
        rt_s = self.rt_var.get().strip()
        if urgent and rt_s:
            rt = int(rt_s)
        elif rt_s and rt_s.lower() != "none":
            try:
                rt = int(rt_s)
            except ValueError:
                rt = None
        self.result_task = TaskSpec(
            task_id=self.tid_var.get().strip(),
            start_point=self.pu_var.get().strip(),
            end_point=self.do_var.get().strip(),
            priority="Urgent" if urgent else "Normal",
            remaining_time=rt if urgent else None,
        )


class MapEditorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MioVerse AGV Map Editor")
        self.root.configure(bg=COLORS["bg"])
        self.root.minsize(1180, 740)

        self.exports_dir = default_exports_dir()
        self.custom_maps_dir = default_custom_maps_dir()
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.custom_maps_dir.mkdir(parents=True, exist_ok=True)

        self.doc, kind, label = load_slot_document(
            1,
            exports_dir=self.exports_dir,
            custom_maps_dir=self.custom_maps_dir,
        )
        self.mode = tk.StringVar(value="obstacle")
        self.agv_pitch = tk.IntVar(value=90)
        self.status = tk.StringVar(value=self._source_status(kind, label))
        self.slot_var = tk.StringVar(value=slot_id(1))
        self.cell_px = 28
        self._undo: List[dict] = []
        self._painting = False
        self._paint_value: Optional[bool] = None
        self._hover: Optional[Tuple[int, int]] = None
        self._task_filter = tk.StringVar(value="")

        self._build_ui()
        self._bind_keys()
        self.redraw()
        self._refresh_slot_list()
        self._refresh_task_list()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg=COLORS["bg"], padx=8, pady=6)
        top.pack(fill=tk.X)

        tk.Label(
            top,
            text="MioVerse Map Editor",
            font=("Segoe UI Semibold", 14),
            bg=COLORS["bg"],
            fg="#1a1a1a",
        ).pack(side=tk.LEFT)

        tk.Label(top, text="  场景槽位", bg=COLORS["bg"]).pack(side=tk.LEFT, padx=(16, 4))
        self.slot_combo = ttk.Combobox(
            top,
            textvariable=self.slot_var,
            values=[slot_id(i) for i in range(1, 21)],
            width=16,
            state="readonly",
        )
        self.slot_combo.pack(side=tk.LEFT)
        self.slot_combo.bind("<<ComboboxSelected>>", self._on_slot_selected)

        ttk.Button(top, text="加载槽位", command=self.load_current_slot).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="保存", command=self.save_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="导出…", command=self.export_as).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="打开…", command=self.open_file).pack(side=tk.LEFT, padx=2)

        body = tk.Frame(self.root, bg=COLORS["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left tools
        tools = tk.Frame(body, bg=COLORS["panel"], padx=10, pady=10, width=210)
        tools.pack(side=tk.LEFT, fill=tk.Y)
        tools.pack_propagate(False)

        tk.Label(
            tools, text="地图站台（位置）", font=("Segoe UI Semibold", 11), bg=COLORS["panel"]
        ).pack(anchor=tk.W)
        tk.Label(
            tools,
            text="在网格上放置障碍 / 取货点 / 卸货点 / AGV。\n右侧「任务」只配置取货→卸货路线与紧急。",
            bg=COLORS["panel"],
            fg="#555",
            justify=tk.LEFT,
            wraplength=190,
            font=("Segoe UI", 8),
        ).pack(anchor=tk.W, pady=(0, 6))

        for key, label in MODES:
            tk.Radiobutton(
                tools,
                text=label,
                variable=self.mode,
                value=key,
                bg=COLORS["panel"],
                activebackground=COLORS["panel"],
                anchor=tk.W,
                command=self._update_status_mode,
            ).pack(fill=tk.X, pady=1)

        tk.Label(tools, text="AGV 朝向", bg=COLORS["panel"]).pack(anchor=tk.W, pady=(12, 2))
        pitch_row = tk.Frame(tools, bg=COLORS["panel"])
        pitch_row.pack(anchor=tk.W)
        for p in PITCHES:
            tk.Radiobutton(
                pitch_row,
                text=f"{PITCH_ARROWS[p]}{p}°",
                variable=self.agv_pitch,
                value=p,
                bg=COLORS["panel"],
                activebackground=COLORS["panel"],
            ).pack(side=tk.LEFT)

        tk.Label(
            tools,
            text="图例",
            font=("Segoe UI Semibold", 11),
            bg=COLORS["panel"],
        ).pack(anchor=tk.W, pady=(16, 4))
        for color, name in (
            (COLORS["obstacle"], "障碍"),
            (COLORS["pickup"], "取货点 Pickup"),
            (COLORS["dropoff"], "卸货点 Dropoff"),
            (COLORS["agv"], "AGV（箭头=朝向）"),
        ):
            row = tk.Frame(tools, bg=COLORS["panel"])
            row.pack(anchor=tk.W, pady=2)
            sw = tk.Canvas(row, width=16, height=16, highlightthickness=0, bg=COLORS["panel"])
            sw.pack(side=tk.LEFT)
            sw.create_rectangle(1, 1, 15, 15, fill=color, outline="#333")
            tk.Label(row, text=f"  {name}", bg=COLORS["panel"]).pack(side=tk.LEFT)

        tk.Label(
            tools,
            text=f"网格 {GRID_MIN}..{GRID_MAX}（可玩区域）",
            bg=COLORS["panel"],
            fg="#555",
        ).pack(anchor=tk.W, pady=(14, 4))

        ttk.Separator(tools, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Button(tools, text="载入官方场景地图", command=self.load_official_for_slot).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(tools, text="载入默认站台布局", command=self.load_baseline).pack(fill=tk.X, pady=2)
        ttk.Button(tools, text="清空全部", command=self.clear_all).pack(fill=tk.X, pady=2)
        ttk.Button(tools, text="撤销 Ctrl+Z", command=self.undo).pack(fill=tk.X, pady=2)
        ttk.Button(tools, text="同步到 custom_maps/", command=self.sync_to_custom_maps).pack(
            fill=tk.X, pady=8
        )

        self.counts_var = tk.StringVar(value="")
        tk.Label(
            tools,
            textvariable=self.counts_var,
            bg=COLORS["panel"],
            justify=tk.LEFT,
            fg="#333",
        ).pack(anchor=tk.W, pady=(8, 0))

        # Canvas
        canvas_wrap = tk.Frame(body, bg=COLORS["bg"])
        canvas_wrap.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        pad = 36
        size = self.cell_px * GRID_SIZE + pad * 2
        self.pad = pad
        self.canvas = tk.Canvas(
            canvas_wrap,
            width=size,
            height=size,
            bg=COLORS["empty"],
            highlightthickness=1,
            highlightbackground="#bbb",
        )
        self.canvas.pack(anchor=tk.NW)
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Button-3>", self._on_right)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda _e: self._clear_hover())

        # Right: task assignments panel
        tasks_panel = tk.Frame(body, bg=COLORS["panel"], padx=10, pady=10, width=320)
        tasks_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        tasks_panel.pack_propagate(False)

        tk.Label(
            tasks_panel,
            text="任务分配（路线）",
            font=("Segoe UI Semibold", 11),
            bg=COLORS["panel"],
        ).pack(anchor=tk.W)
        tk.Label(
            tasks_panel,
            text="每条任务 = 取货站台 → 卸货目的地 + 是否紧急。\n与地图上的站台位置分开编辑。\n加载官方 L100 时默认带入该场景任务表。",
            bg=COLORS["panel"],
            fg="#555",
            justify=tk.LEFT,
            wraplength=290,
            font=("Segoe UI", 8),
        ).pack(anchor=tk.W, pady=(0, 6))

        filt_row = tk.Frame(tasks_panel, bg=COLORS["panel"])
        filt_row.pack(fill=tk.X, pady=2)
        tk.Label(filt_row, text="筛选", bg=COLORS["panel"]).pack(side=tk.LEFT)
        filt_entry = tk.Entry(filt_row, textvariable=self._task_filter, width=18)
        filt_entry.pack(side=tk.LEFT, padx=4)
        filt_entry.bind("<KeyRelease>", lambda _e: self._refresh_task_list())

        list_frame = tk.Frame(tasks_panel, bg=COLORS["panel"])
        list_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        scroll = ttk.Scrollbar(list_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.task_list = tk.Listbox(
            list_frame,
            height=22,
            width=38,
            font=("Consolas", 9),
            yscrollcommand=scroll.set,
            exportselection=False,
        )
        self.task_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.task_list.yview)
        self.task_list.bind("<Double-Button-1>", lambda _e: self.edit_selected_task())

        btn_row = tk.Frame(tasks_panel, bg=COLORS["panel"])
        btn_row.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row, text="新建", command=self.add_task).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_row, text="编辑", command=self.edit_selected_task).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(btn_row, text="删除", command=self.delete_selected_task).pack(
            side=tk.LEFT, padx=1
        )

        btn_row2 = tk.Frame(tasks_panel, bg=COLORS["panel"])
        btn_row2.pack(fill=tk.X, pady=2)
        ttk.Button(btn_row2, text="清空任务", command=self.clear_tasks).pack(
            side=tk.LEFT, padx=1
        )
        ttk.Button(btn_row2, text="重载官方任务", command=self.reload_official_tasks).pack(
            side=tk.LEFT, padx=1
        )

        self.task_summary = tk.StringVar(value="")
        tk.Label(
            tasks_panel,
            textvariable=self.task_summary,
            bg=COLORS["panel"],
            fg="#333",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(6, 0))

        # Status bar
        bar = tk.Frame(self.root, bg="#ddd", padx=8, pady=4)
        bar.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(bar, textvariable=self.status, bg="#ddd", anchor=tk.W).pack(fill=tk.X)

    def _bind_keys(self) -> None:
        self.root.bind("<Control-z>", lambda _e: self.undo())
        self.root.bind("<Control-Z>", lambda _e: self.undo())
        self.root.bind("<Control-s>", lambda _e: self.save_current())
        self.root.bind("<Control-S>", lambda _e: self.save_current())

        def _set_mode(m: str):
            def _handler(_e=None, mode=m):
                self.mode.set(mode)
                self._update_status_mode()

            return _handler

        for i, key in enumerate(("o", "p", "d", "a", "e")):
            handler = _set_mode(MODES[i][0])
            self.root.bind(key, handler)
            self.root.bind(key.upper(), handler)

    # -------------------------------------------------------------- helpers
    def _push_undo(self) -> None:
        self._undo.append(self.doc.snapshot())
        if len(self._undo) > 80:
            self._undo.pop(0)

    def _xy_from_event(self, event) -> Optional[Tuple[int, int]]:
        x = (event.x - self.pad) // self.cell_px + GRID_MIN
        y = GRID_MAX - (event.y - self.pad) // self.cell_px
        if GRID_MIN <= x <= GRID_MAX and GRID_MIN <= y <= GRID_MAX:
            return int(x), int(y)
        return None

    def _cell_rect(self, x: int, y: int) -> Tuple[int, int, int, int]:
        # y grows upward (origin lower-left), matching map_overview
        col = x - GRID_MIN
        row_from_top = GRID_MAX - y
        x0 = self.pad + col * self.cell_px
        y0 = self.pad + row_from_top * self.cell_px
        return x0, y0, x0 + self.cell_px, y0 + self.cell_px

    def _update_counts(self) -> None:
        n_urgent = sum(1 for t in self.doc.tasks if t.is_urgent())
        self.counts_var.set(
            f"障碍 {len(self.doc.obstacles)}\n"
            f"取货 {len(self.doc.pickups)}\n"
            f"卸货 {len(self.doc.dropoffs)}\n"
            f"AGV  {len(self.doc.agvs)}\n"
            f"任务 {len(self.doc.tasks)}（急 {n_urgent}）"
        )

    def _update_status_mode(self) -> None:
        m = self.mode.get()
        tips = {
            "obstacle": "障碍：拖拽绘制 / 再点切换",
            "pickup": "取货站台：点击放置（自动命名）— 任务在右侧面板配置",
            "dropoff": "卸货站台：点击放置（自动命名）— 任务在右侧面板配置",
            "agv": "AGV：点击放置；再点同格循环朝向",
            "erase": "擦除：清除格内任意内容",
        }
        self.status.set(tips.get(m, ""))

    def _source_status(self, kind: str, label: str) -> str:
        n = len(self.doc.tasks) if hasattr(self, "doc") else 0
        extra = f"（任务 {n}）" if n else ""
        if kind == "custom":
            return f"已加载自定义存档 {label}{extra} — 左键绘制，右键擦除，Ctrl+Z 撤销"
        if kind == "official":
            return f"默认场景：官方 L100 {label}{extra} — 左键绘制，右键擦除，Ctrl+Z 撤销"
        return f"空槽位 {label}{extra} — 左键绘制，右键擦除，Ctrl+Z 撤销"

    def _apply_loaded_slot(self, n: int, *, confirm_custom: bool = False) -> bool:
        """Load slot n (custom preferred, else official L100). Returns False if cancelled."""
        sid = slot_id(n)
        custom_path = self.exports_dir / f"{sid}.json"
        if confirm_custom and custom_path.exists():
            if not messagebox.askyesno(
                "切换槽位",
                f"加载已保存的 {sid}？\n（当前未保存修改将丢失，除非先点保存）",
            ):
                self.doc.map_id = sid
                self.redraw()
                return False
        doc, kind, label = load_slot_document(
            n,
            exports_dir=self.exports_dir,
            custom_maps_dir=self.custom_maps_dir,
        )
        self.doc = doc
        self.slot_var.set(sid)
        self.redraw()
        self.status.set(self._source_status(kind, label))
        return True

    def _refresh_slot_list(self) -> None:
        self.slot_combo["values"] = [slot_id(i) for i in range(1, 21)]

    def _visible_task_indices(self) -> List[int]:
        q = self._task_filter.get().strip().lower()
        out: List[int] = []
        for i, t in enumerate(self.doc.tasks):
            if not q:
                out.append(i)
                continue
            blob = f"{t.task_id} {t.start_point} {t.end_point} {t.priority}".lower()
            if q in blob:
                out.append(i)
        return out

    def _refresh_task_list(self) -> None:
        if not hasattr(self, "task_list"):
            return
        self.task_list.delete(0, tk.END)
        indices = self._visible_task_indices()
        for i in indices:
            t = self.doc.tasks[i]
            flag = "急" if t.is_urgent() else "  "
            rt = t.remaining_time if t.remaining_time is not None else "-"
            line = f"{flag} {t.task_id:<12} {t.start_point}->{t.end_point}  {rt}"
            self.task_list.insert(tk.END, line)
            if t.is_urgent():
                self.task_list.itemconfig(tk.END, foreground=COLORS["urgent"])
        n_u = sum(1 for t in self.doc.tasks if t.is_urgent())
        shown = len(indices)
        total = len(self.doc.tasks)
        self.task_summary.set(
            f"显示 {shown}/{total} 条 · 紧急 {n_u} · CSV: task_id,start_point,end_point,priority,remaining_time"
        )

    def _selected_task_index(self) -> Optional[int]:
        sel = self.task_list.curselection()
        if not sel:
            return None
        visible = self._visible_task_indices()
        li = int(sel[0])
        if 0 <= li < len(visible):
            return visible[li]
        return None

    def _station_name_lists(self) -> Tuple[List[str], List[str]]:
        pu = [p.name for p in self.doc.pickups] or list(DEFAULT_PICKUP_NAMES)
        do = [p.name for p in self.doc.dropoffs] or list(DEFAULT_DROPOFF_NAMES)
        return pu, do

    # -------------------------------------------------------------- draw
    def redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        gx0 = self.pad
        gy0 = self.pad
        gx1 = self.pad + GRID_SIZE * self.cell_px
        gy1 = self.pad + GRID_SIZE * self.cell_px
        c.create_rectangle(gx0, gy0, gx1, gy1, fill=COLORS["empty"], outline="#aaa")

        obs = set(self.doc.obstacles)
        pickup_at = {(p.x, p.y): p for p in self.doc.pickups}
        drop_at = {(p.x, p.y): p for p in self.doc.dropoffs}
        agv_at = {(p.x, p.y): p for p in self.doc.agvs}

        for y in range(GRID_MIN, GRID_MAX + 1):
            for x in range(GRID_MIN, GRID_MAX + 1):
                r = self._cell_rect(x, y)
                fill = COLORS["empty"]
                outline = COLORS["grid"]
                label = ""
                text_color = COLORS["label"]
                if (x, y) in obs:
                    fill = COLORS["obstacle"]
                    text_color = "#eee"
                if (x, y) in pickup_at:
                    fill = COLORS["pickup"]
                    outline = COLORS["pickup_edge"]
                    label = pickup_at[(x, y)].name[:3]
                    text_color = "white"
                if (x, y) in drop_at:
                    fill = COLORS["dropoff"]
                    outline = COLORS["dropoff_edge"]
                    label = drop_at[(x, y)].name[:3]
                    text_color = "white"
                if (x, y) in agv_at:
                    fill = COLORS["agv"]
                    outline = COLORS["agv_edge"]
                    ag = agv_at[(x, y)]
                    arrow = PITCH_ARROWS.get(ag.pitch or 90, "↑")
                    label = f"{arrow}{ag.name[:2]}"
                    text_color = "#222"
                if self._hover == (x, y) and fill == COLORS["empty"]:
                    fill = COLORS["hover"]
                c.create_rectangle(*r, fill=fill, outline=outline)
                if label:
                    c.create_text(
                        (r[0] + r[2]) // 2,
                        (r[1] + r[3]) // 2,
                        text=label,
                        fill=text_color,
                        font=("Segoe UI", 7),
                    )

        # axis labels
        for i in range(GRID_MIN, GRID_MAX + 1):
            # x along bottom
            rx = self._cell_rect(i, GRID_MIN)
            c.create_text(
                (rx[0] + rx[2]) // 2,
                gy1 + 12,
                text=str(i),
                fill="#666",
                font=("Segoe UI", 8),
            )
            # y along left
            ry = self._cell_rect(GRID_MIN, i)
            c.create_text(
                gx0 - 12,
                (ry[1] + ry[3]) // 2,
                text=str(i),
                fill="#666",
                font=("Segoe UI", 8),
            )
        title = self.doc.map_id
        if self.doc.notes:
            title = f"{self.doc.map_id}  ·  {self.doc.notes}"
        c.create_text(
            gx0 + 40,
            14,
            text=title,
            fill="#333",
            font=("Segoe UI Semibold", 10),
            anchor=tk.W,
        )
        self._update_counts()
        self._refresh_task_list()

    def _clear_hover(self) -> None:
        if self._hover is not None:
            self._hover = None
            self.redraw()

    # -------------------------------------------------------------- input
    def _on_motion(self, event) -> None:
        cell = self._xy_from_event(event)
        if cell != self._hover:
            self._hover = cell
            if cell:
                kind = self.doc.cell_kind(*cell) or "空"
                self.status.set(f"({cell[0]}, {cell[1]}) — {kind} | 模式={self.mode.get()}")
            self.redraw()

    def _on_left_down(self, event) -> None:
        cell = self._xy_from_event(event)
        if not cell:
            return
        self._push_undo()
        self._painting = True
        mode = self.mode.get()
        x, y = cell
        if mode == "obstacle":
            on = (x, y) not in self.doc.obstacles
            self._paint_value = on
            self.doc.set_obstacle(x, y, on=on)
        elif mode == "erase":
            self._paint_value = False
            self.doc.erase_at(x, y)
        elif mode == "pickup":
            self.doc.add_pickup(x, y)
            self._painting = False
        elif mode == "dropoff":
            self.doc.add_dropoff(x, y)
            self._painting = False
        elif mode == "agv":
            if self.doc.cell_kind(x, y) == "agv":
                self.doc.cycle_agv_pitch(x, y)
            else:
                self.doc.add_agv(x, y, pitch=self.agv_pitch.get())
            self._painting = False
        self.redraw()

    def _on_left_drag(self, event) -> None:
        if not self._painting:
            return
        cell = self._xy_from_event(event)
        if not cell:
            return
        x, y = cell
        mode = self.mode.get()
        if mode == "obstacle" and self._paint_value is not None:
            self.doc.set_obstacle(x, y, on=self._paint_value)
        elif mode == "erase":
            self.doc.erase_at(x, y)
        self.redraw()

    def _on_left_up(self, _event) -> None:
        self._painting = False
        self._paint_value = None

    def _on_right(self, event) -> None:
        cell = self._xy_from_event(event)
        if not cell:
            return
        self._push_undo()
        self.doc.erase_at(*cell)
        self.redraw()

    # -------------------------------------------------------------- tasks
    def add_task(self) -> None:
        pu, do = self._station_name_lists()
        dlg = TaskEditDialog(
            self.root,
            "新建任务",
            pickup_names=pu,
            dropoff_names=do,
            existing=self.doc.tasks,
        )
        if dlg.result_task is None:
            return
        self._push_undo()
        self.doc.tasks.append(dlg.result_task)
        self.redraw()
        self.status.set(f"已添加任务 {dlg.result_task.task_id}")

    def edit_selected_task(self) -> None:
        idx = self._selected_task_index()
        if idx is None:
            messagebox.showinfo("任务", "请先选择一条任务。")
            return
        old = self.doc.tasks[idx]
        pu, do = self._station_name_lists()
        # ensure current names appear in combo lists
        if old.start_point not in pu:
            pu = [old.start_point] + pu
        if old.end_point not in do:
            do = [old.end_point] + do
        dlg = TaskEditDialog(
            self.root,
            "编辑任务",
            pickup_names=pu,
            dropoff_names=do,
            initial=old,
            existing=self.doc.tasks,
        )
        if dlg.result_task is None:
            return
        self._push_undo()
        self.doc.tasks[idx] = dlg.result_task
        self.redraw()
        self.status.set(f"已更新任务 {dlg.result_task.task_id}")

    def delete_selected_task(self) -> None:
        idx = self._selected_task_index()
        if idx is None:
            messagebox.showinfo("任务", "请先选择一条任务。")
            return
        tid = self.doc.tasks[idx].task_id
        if not messagebox.askyesno("删除任务", f"删除任务 {tid}？"):
            return
        self._push_undo()
        del self.doc.tasks[idx]
        self.redraw()
        self.status.set(f"已删除 {tid}")

    def clear_tasks(self) -> None:
        if not self.doc.tasks:
            return
        if not messagebox.askyesno("清空任务", "清空当前全部任务分配？（不影响地图站台）"):
            return
        self._push_undo()
        self.doc.tasks.clear()
        self.redraw()
        self.status.set("已清空任务列表")

    def reload_official_tasks(self) -> None:
        n = slot_index(self.slot_var.get()) or 1
        off = official_l100_for_slot(n)
        if off is None:
            messagebox.showwarning("官方任务", f"未找到槽位 {n} 对应的 L100 官方场景。")
            return
        base_id, path = off
        tmp = MapDocument.load_from_ladder_scenario(path, map_id=slot_id(n))
        if not messagebox.askyesno(
            "重载官方任务",
            f"用官方 {base_id} 的 {len(tmp.tasks)} 条任务覆盖当前任务列表？\n（地图站台/障碍不变）",
        ):
            return
        self._push_undo()
        self.doc.tasks = list(tmp.tasks)
        self.redraw()
        self.status.set(f"已载入官方任务 {base_id}（{len(self.doc.tasks)} 条）")

    # -------------------------------------------------------------- actions
    def undo(self) -> None:
        if not self._undo:
            self.status.set("没有可撤销的操作")
            return
        snap = self._undo.pop()
        self.doc.restore(snap)
        self.redraw()
        self.status.set("已撤销")

    def clear_all(self) -> None:
        if not messagebox.askyesno("清空", "清空当前地图的全部内容（含任务）？"):
            return
        self._push_undo()
        self.doc.clear_all()
        self.redraw()

    def load_baseline(self) -> None:
        self._push_undo()
        self.doc.load_baseline_layout()
        self.redraw()
        self.status.set("已载入默认站台 + 12 AGV 布局（任务列表保留）")

    def load_official_for_slot(self) -> None:
        n = slot_index(self.slot_var.get()) or 1
        off = official_l100_for_slot(n)
        if off is None:
            messagebox.showwarning("官方地图", f"未找到槽位 {n} 对应的 L100 官方场景。")
            return
        base_id, path = off
        self._push_undo()
        self.doc = MapDocument.load_from_ladder_scenario(path, map_id=slot_id(n))
        self.redraw()
        self.status.set(self._source_status("official", base_id))

    def _on_slot_selected(self, _event=None) -> None:
        sid = self.slot_var.get()
        n = slot_index(sid)
        if n is None:
            self.doc.map_id = sid
            self.redraw()
            return
        custom_path = self.exports_dir / f"{sid}.json"
        self._apply_loaded_slot(n, confirm_custom=custom_path.exists())

    def load_current_slot(self) -> None:
        sid = self.slot_var.get()
        n = slot_index(sid)
        if n is None:
            messagebox.showinfo("加载", f"{sid} 不是标准槽位。")
            return
        self._apply_loaded_slot(n, confirm_custom=False)

    def save_current(self) -> None:
        sid = self.slot_var.get() or self.doc.map_id
        self.doc.map_id = sid
        path = self.exports_dir / f"{sid}.json"
        self.doc.save(path)
        # also mirror to custom_maps
        mirror = self.custom_maps_dir / f"{sid}.json"
        self.doc.save(mirror)
        self._refresh_slot_list()
        self.status.set(f"已保存 → {path}  及  {mirror}")
        messagebox.showinfo(
            "已保存",
            f"JSON: {path}\n"
            f"位置 CSV:  {path.with_name(sid + '_position.csv')}\n"
            f"任务 CSV:  {path.with_name(sid + '_task.csv')}\n"
            f"镜像: {mirror}",
        )

    def sync_to_custom_maps(self) -> None:
        sid = self.doc.map_id
        path = self.custom_maps_dir / f"{sid}.json"
        self.doc.save(path)
        self.status.set(f"已同步到 {path}")

    def export_as(self) -> None:
        path = filedialog.asksaveasfilename(
            initialdir=str(self.exports_dir),
            initialfile=f"{self.doc.map_id}.json",
            defaultextension=".json",
            filetypes=[("Map JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        p = Path(path)
        self.doc.map_id = p.stem
        self.slot_var.set(self.doc.map_id)
        self.doc.save(p)
        self.status.set(f"已导出 {p}")

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(self.exports_dir),
            filetypes=[("Map JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        self.doc = MapDocument.load(Path(path))
        self.slot_var.set(self.doc.map_id)
        self.redraw()
        self.status.set(f"已打开 {path}（任务 {len(self.doc.tasks)}）")


def main() -> None:
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.25)
    except tk.TclError:
        pass
    MapEditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
