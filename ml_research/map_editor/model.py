"""Map document model + JSON/CSV I/O for custom AGV warehouse scenes."""
from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Playable aisle matches curriculum_shape / main warehouse (coords 1..20).
GRID_MIN = 1
GRID_MAX = 20
GRID_SIZE = GRID_MAX  # editable cells per axis

# Name pools aligned with ml_research.benchmarks.scenarios.generator
DEFAULT_PICKUP_NAMES = ["Tiger", "Dragon", "Horse", "Rabbit", "Ox", "Monkey"]
DEFAULT_DROPOFF_NAMES = [
    "Beijing",
    "Shanghai",
    "Suzhou",
    "Hangzhou",
    "Nanjing",
    "Wuhan",
    "Changsha",
    "Guangzhou",
    "Chengdu",
    "Xiamen",
    "Kunming",
    "Urumqi",
    "Shenzhen",
    "Dalian",
    "Tianjin",
    "Chongqing",
]
DEFAULT_AGV_NAMES = [
    "Optimus",
    "Bumblebee",
    "Jazz",
    "Sideswipe",
    "Wheeljack",
    "Ratchet",
    "Ironhide",
    "Hound",
    "Smokescreen",
    "Megatron",
    "Bluestreak",
    "RedAlert",
    "Prime",
    "Prowl",
    "Mirage",
    "Sunstreaker",
    "HotRod",
    "Kup",
    "Arcee",
    "Chromia",
    "Elita",
    "Flamewar",
    "Moonracer",
    "Novastar",
    "Greenlight",
    "Strongarm",
    "Windblade",
    "Override",
    "Thunderblast",
    "Slipstream",
]

PITCHES = (0, 90, 180, 270)
PITCH_ARROWS = {0: "→", 90: "↑", 180: "←", 270: "↓"}

Cell = Tuple[int, int]


def default_exports_dir() -> Path:
    return Path(__file__).resolve().parent / "exports"


def default_custom_maps_dir() -> Path:
    """Alias of exports (legacy curriculum_shape path removed)."""
    return default_exports_dir()


def default_scenarios_dir() -> Path:
    """Stress scenario JSON/CSV under benchmarks/scenarios."""
    return Path(__file__).resolve().parent.parent / "benchmarks" / "scenarios"


@dataclass
class PlacedPoint:
    name: str
    x: int
    y: int
    pitch: Optional[int] = None  # AGV only

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "x": self.x, "y": self.y}
        if self.pitch is not None:
            d["pitch"] = int(self.pitch)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlacedPoint":
        pitch = d.get("pitch")
        return cls(
            name=str(d["name"]),
            x=int(d["x"]),
            y=int(d["y"]),
            pitch=int(pitch) if pitch is not None and str(pitch) != "" else None,
        )


TASK_CSV_FIELDS = ["task_id", "start_point", "end_point", "priority", "remaining_time"]


@dataclass
class TaskSpec:
    """One transport assignment: pickup station → dropoff station (+ urgency).

    Matches curriculum_shape / main-copy task CSV columns.
    """

    task_id: str
    start_point: str  # pickup station name
    end_point: str  # dropoff / unload destination name
    priority: str = "Normal"  # Normal | Urgent
    remaining_time: Optional[int] = None  # required-ish for Urgent; None for Normal

    def is_urgent(self) -> bool:
        return str(self.priority).strip().lower() in ("urgent", "emergency", "true", "1")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "start_point": self.start_point,
            "end_point": self.end_point,
            "priority": "Urgent" if self.is_urgent() else "Normal",
            "remaining_time": self.remaining_time
            if self.remaining_time is not None
            else None,
        }

    def to_csv_row(self) -> Dict[str, Any]:
        d = self.to_dict()
        d["remaining_time"] = (
            d["remaining_time"] if d["remaining_time"] is not None else "None"
        )
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskSpec":
        pri = str(d.get("priority") or "Normal").strip()
        if pri.lower() in ("urgent", "emergency", "true", "1"):
            pri = "Urgent"
        else:
            pri = "Normal"
        rt_raw = d.get("remaining_time")
        rt: Optional[int] = None
        if rt_raw not in (None, "", "None", "none"):
            try:
                rt = int(rt_raw)
            except (TypeError, ValueError):
                rt = None
        return cls(
            task_id=str(d.get("task_id") or "").strip(),
            start_point=str(d.get("start_point") or d.get("pickup") or "").strip(),
            end_point=str(
                d.get("end_point") or d.get("destination") or d.get("dropoff") or ""
            ).strip(),
            priority=pri,
            remaining_time=rt,
        )


def load_tasks_csv(path: Path) -> List[TaskSpec]:
    path = Path(path)
    if not path.exists():
        return []
    out: List[TaskSpec] = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if not row:
                continue
            t = TaskSpec.from_dict(row)
            if t.task_id and t.start_point and t.end_point:
                out.append(t)
    return out


def write_tasks_csv(path: Path, tasks: List[TaskSpec]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TASK_CSV_FIELDS)
        w.writeheader()
        for t in tasks:
            w.writerow(t.to_csv_row())
    return path


def next_task_id(start_point: str, existing: List[TaskSpec]) -> str:
    """Allocate Tiger-1, Tiger-2, … from existing list for the same pickup."""
    station = start_point.strip() or "Task"
    max_n = 0
    prefix = station + "-"
    for t in existing:
        tid = t.task_id
        if tid.startswith(prefix):
            try:
                max_n = max(max_n, int(tid[len(prefix) :]))
            except ValueError:
                pass
    return f"{station}-{max_n + 1}"


@dataclass
class MapDocument:
    map_id: str = "SH_custom_01"
    pickups: List[PlacedPoint] = field(default_factory=list)
    dropoffs: List[PlacedPoint] = field(default_factory=list)
    agvs: List[PlacedPoint] = field(default_factory=list)
    obstacles: List[Cell] = field(default_factory=list)
    tasks: List[TaskSpec] = field(default_factory=list)
    notes: str = ""
    # Playable cells: x in [grid_min, grid_width], y in [grid_min, grid_height]
    grid_min: int = GRID_MIN
    grid_width: int = GRID_SIZE
    grid_height: int = GRID_SIZE

    @property
    def grid_max(self) -> int:
        """Backward-compat single axis max (max of width/height)."""
        return max(int(self.grid_width), int(self.grid_height))

    def in_bounds(self, x: int, y: int) -> bool:
        return (
            self.grid_min <= int(x) <= int(self.grid_width)
            and self.grid_min <= int(y) <= int(self.grid_height)
        )

    def resize(
        self,
        width: int,
        height: int,
        *,
        grid_min: Optional[int] = None,
        clip: bool = True,
    ) -> None:
        """Change scene size. If clip=True, drop entities outside the new bounds."""
        w = max(2, int(width))
        h = max(2, int(height))
        gmin = int(self.grid_min if grid_min is None else grid_min)
        self.grid_min = gmin
        self.grid_width = w
        self.grid_height = h
        if not clip:
            return

        def _ok(p: PlacedPoint) -> bool:
            return self.in_bounds(p.x, p.y)

        self.pickups = [p for p in self.pickups if _ok(p)]
        self.dropoffs = [p for p in self.dropoffs if _ok(p)]
        self.agvs = [p for p in self.agvs if _ok(p)]
        self.obstacles = [c for c in self.obstacles if self.in_bounds(c[0], c[1])]
        # Drop tasks whose stations disappeared
        pu = {p.name for p in self.pickups}
        do = {p.name for p in self.dropoffs}
        self.tasks = [
            t for t in self.tasks if t.start_point in pu and t.end_point in do
        ]

    def snapshot(self) -> Dict[str, Any]:
        return deepcopy(
            {
                "map_id": self.map_id,
                "pickups": [p.to_dict() for p in self.pickups],
                "dropoffs": [p.to_dict() for p in self.dropoffs],
                "agvs": [p.to_dict() for p in self.agvs],
                "obstacles": [list(c) for c in self.obstacles],
                "tasks": [t.to_dict() for t in self.tasks],
                "notes": self.notes,
                "grid_min": self.grid_min,
                "grid_width": self.grid_width,
                "grid_height": self.grid_height,
            }
        )

    def restore(self, snap: Dict[str, Any]) -> None:
        self.map_id = str(snap.get("map_id", self.map_id))
        self.pickups = [PlacedPoint.from_dict(p) for p in snap.get("pickups", [])]
        self.dropoffs = [PlacedPoint.from_dict(p) for p in snap.get("dropoffs", [])]
        self.agvs = [PlacedPoint.from_dict(p) for p in snap.get("agvs", [])]
        self.obstacles = [(int(c[0]), int(c[1])) for c in snap.get("obstacles", [])]
        self.tasks = [TaskSpec.from_dict(t) for t in snap.get("tasks", [])]
        self.notes = str(snap.get("notes", ""))
        self.grid_min = int(snap.get("grid_min", self.grid_min))
        self.grid_width = int(snap.get("grid_width", snap.get("grid_max", self.grid_width)))
        self.grid_height = int(snap.get("grid_height", snap.get("grid_max", self.grid_height)))

    def clear_all(self) -> None:
        self.pickups.clear()
        self.dropoffs.clear()
        self.agvs.clear()
        self.obstacles.clear()
        self.tasks.clear()

    def cell_kind(self, x: int, y: int) -> Optional[str]:
        for p in self.agvs:
            if p.x == x and p.y == y:
                return "agv"
        for p in self.pickups:
            if p.x == x and p.y == y:
                return "pickup"
        for p in self.dropoffs:
            if p.x == x and p.y == y:
                return "dropoff"
        if (x, y) in self.obstacles:
            return "obstacle"
        return None

    def erase_at(self, x: int, y: int) -> bool:
        changed = False
        before = len(self.obstacles)
        self.obstacles = [c for c in self.obstacles if c != (x, y)]
        if len(self.obstacles) != before:
            changed = True
        for attr in ("pickups", "dropoffs", "agvs"):
            lst: List[PlacedPoint] = getattr(self, attr)
            n = len(lst)
            setattr(self, attr, [p for p in lst if not (p.x == x and p.y == y)])
            if len(getattr(self, attr)) != n:
                changed = True
        return changed

    def _next_name(self, used: set, pool: List[str], prefix: str) -> str:
        for n in pool:
            if n not in used:
                return n
        i = 1
        while f"{prefix}{i}" in used:
            i += 1
        return f"{prefix}{i}"

    def add_pickup(self, x: int, y: int, name: Optional[str] = None) -> PlacedPoint:
        self.erase_at(x, y)
        used = {p.name for p in self.pickups}
        nm = name or self._next_name(used, DEFAULT_PICKUP_NAMES, "PU")
        pt = PlacedPoint(nm, x, y)
        self.pickups.append(pt)
        return pt

    def add_dropoff(self, x: int, y: int, name: Optional[str] = None) -> PlacedPoint:
        self.erase_at(x, y)
        used = {p.name for p in self.dropoffs}
        nm = name or self._next_name(used, DEFAULT_DROPOFF_NAMES, "DO")
        pt = PlacedPoint(nm, x, y)
        self.dropoffs.append(pt)
        return pt

    def add_agv(
        self, x: int, y: int, pitch: int = 90, name: Optional[str] = None
    ) -> PlacedPoint:
        self.erase_at(x, y)
        used = {p.name for p in self.agvs}
        nm = name or self._next_name(used, DEFAULT_AGV_NAMES, "AGV")
        pt = PlacedPoint(nm, x, y, pitch=int(pitch) % 360)
        self.agvs.append(pt)
        return pt

    def set_obstacle(self, x: int, y: int, on: bool = True) -> bool:
        cell = (x, y)
        if on:
            if self.cell_kind(x, y) in ("pickup", "dropoff", "agv"):
                self.erase_at(x, y)
            if cell not in self.obstacles:
                self.obstacles.append(cell)
                return True
            return False
        if cell in self.obstacles:
            self.obstacles = [c for c in self.obstacles if c != cell]
            return True
        return False

    def cycle_agv_pitch(self, x: int, y: int) -> Optional[int]:
        for p in self.agvs:
            if p.x == x and p.y == y and p.pitch is not None:
                idx = PITCHES.index(p.pitch) if p.pitch in PITCHES else 0
                p.pitch = PITCHES[(idx + 1) % len(PITCHES)]
                return p.pitch
        return None

    def load_baseline_layout(self) -> None:
        """Load default stations + AGVs scaled to current grid size (no obstacles).

        Does not clear the task list (stations vs assignments are separate).
        On the classic 20×20 grid this matches the competition layout.
        """
        self.pickups.clear()
        self.dropoffs.clear()
        self.agvs.clear()
        self.obstacles.clear()
        w, h = int(self.grid_width), int(self.grid_height)
        gmin = int(self.grid_min)

        def sx(x20: int) -> int:
            # map reference 1..20 → 1..w
            return max(gmin, min(w, int(round((x20 - 1) * (w - 1) / 19.0) + gmin)))

        def sy(y20: int) -> int:
            return max(gmin, min(h, int(round((y20 - 1) * (h - 1) / 19.0) + gmin)))

        defaults_pu = [
            ("Tiger", 1, 6),
            ("Dragon", 1, 10),
            ("Horse", 1, 14),
            ("Rabbit", 20, 6),
            ("Ox", 20, 10),
            ("Monkey", 20, 14),
        ]
        defaults_do = [
            ("Beijing", 6, 4),
            ("Shanghai", 6, 8),
            ("Suzhou", 6, 12),
            ("Hangzhou", 6, 16),
            ("Nanjing", 9, 4),
            ("Wuhan", 9, 8),
            ("Changsha", 9, 12),
            ("Guangzhou", 9, 16),
            ("Chengdu", 12, 4),
            ("Xiamen", 12, 8),
            ("Kunming", 12, 12),
            ("Urumqi", 12, 16),
            ("Shenzhen", 15, 4),
            ("Dalian", 15, 8),
            ("Tianjin", 15, 12),
            ("Chongqing", 15, 16),
        ]
        defaults_agv = [
            ("Optimus", 3, 1, 90),
            ("Bumblebee", 6, 1, 90),
            ("Jazz", 9, 1, 90),
            ("Sideswipe", 12, 1, 90),
            ("Wheeljack", 15, 1, 90),
            ("Ratchet", 18, 1, 90),
            ("Ironhide", 3, 20, 270),
            ("Hound", 6, 20, 270),
            ("Smokescreen", 9, 20, 270),
            ("Megatron", 12, 20, 270),
            ("Bluestreak", 15, 20, 270),
            ("RedAlert", 18, 20, 270),
        ]
        self.pickups = [PlacedPoint(n, sx(x), sy(y)) for n, x, y in defaults_pu]
        self.dropoffs = [PlacedPoint(n, sx(x), sy(y)) for n, x, y in defaults_do]
        self.agvs = [
            PlacedPoint(n, sx(x), sy(y), pitch=p) for n, x, y, p in defaults_agv
        ]

    def to_export_dict(self) -> Dict[str, Any]:
        return {
            "id": self.map_id,
            "map_id": self.map_id,
            "format_version": 3,
            "grid_min": int(self.grid_min),
            "grid_max": int(self.grid_max),
            "grid_width": int(self.grid_width),
            "grid_height": int(self.grid_height),
            "n_agvs": len(self.agvs),
            "n_pickups": len(self.pickups),
            "n_dropoffs": len(self.dropoffs),
            "n_tasks": len(self.tasks),
            "pickups": [p.to_dict() for p in self.pickups],
            "dropoffs": [p.to_dict() for p in self.dropoffs],
            "agvs": [p.to_dict() for p in self.agvs],
            "tasks": [t.to_dict() for t in self.tasks],
            "extra_obstacles": [list(c) for c in sorted(self.obstacles)],
            "notes": self.notes,
            "compatible_with": (
                "curriculum_shape ladder (position CSV + task CSV + extra_obstacles); "
                "grid_width/height for variable-size scenes"
            ),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_export_dict()
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # Companion position CSV (ladder-compatible columns)
        csv_path = path.with_name(path.stem + "_position.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["type", "name", "x", "y", "pitch"])
            for p in self.pickups:
                w.writerow(["start_point", p.name, p.x, p.y, ""])
            for p in self.dropoffs:
                w.writerow(["end_point", p.name, p.x, p.y, ""])
            for p in self.agvs:
                w.writerow(["agv", p.name, p.x, p.y, p.pitch if p.pitch is not None else 90])
        # Companion task CSV (same columns as official ladder scenarios)
        write_tasks_csv(path.with_name(path.stem + "_task.csv"), self.tasks)
        # Obstacles-only JSON sidecar (matches video export naming)
        obs_path = path.with_name(path.stem + "_obstacles.json")
        obs_path.write_text(
            json.dumps({"extra_obstacles": data["extra_obstacles"]}, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "MapDocument":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        doc = cls(map_id=str(data.get("id") or data.get("map_id") or path.stem))
        gmin = int(data.get("grid_min", GRID_MIN))
        gmax = int(data.get("grid_max", GRID_MAX))
        doc.grid_min = gmin
        doc.grid_width = int(data.get("grid_width", gmax))
        doc.grid_height = int(data.get("grid_height", gmax))
        doc.pickups = [PlacedPoint.from_dict(p) for p in data.get("pickups", [])]
        doc.dropoffs = [PlacedPoint.from_dict(p) for p in data.get("dropoffs", [])]
        doc.agvs = [PlacedPoint.from_dict(p) for p in data.get("agvs", [])]
        obs = data.get("extra_obstacles") or data.get("obstacles") or []
        doc.obstacles = [(int(c[0]), int(c[1])) for c in obs]
        doc.notes = str(data.get("notes", ""))

        raw_tasks = data.get("tasks")
        if raw_tasks:
            doc.tasks = [TaskSpec.from_dict(t) for t in raw_tasks]
        else:
            task_path: Optional[Path] = None
            raw_task = data.get("task_csv")
            if raw_task:
                cand = Path(str(raw_task))
                if cand.exists():
                    task_path = cand
            if task_path is None:
                companion = path.with_name(path.stem + "_task.csv")
                if companion.exists():
                    task_path = companion
            if task_path is not None:
                doc.tasks = load_tasks_csv(task_path)

        # Fallback: load companion position CSV if JSON lacks points
        if not doc.pickups and not doc.dropoffs and not doc.agvs:
            csv_path = path.with_name(path.stem + "_position.csv")
            if csv_path.exists():
                doc._load_position_csv(csv_path)
        return doc

    def _load_position_csv(self, csv_path: Path) -> None:
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row["type"]
                pitch = row.get("pitch") or None
                pt = PlacedPoint(
                    row["name"],
                    int(row["x"]),
                    int(row["y"]),
                    pitch=int(pitch) if pitch not in (None, "") else None,
                )
                if t == "start_point":
                    self.pickups.append(pt)
                elif t == "end_point":
                    self.dropoffs.append(pt)
                elif t == "agv":
                    if pt.pitch is None:
                        pt.pitch = 90
                    self.agvs.append(pt)

    @classmethod
    def load_from_ladder_scenario(
        cls, path: Path, map_id: Optional[str] = None
    ) -> "MapDocument":
        """Load official ladder JSON (+ position CSV) into an editor document."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        sid = str(data.get("id") or path.stem)
        doc = cls(map_id=map_id or sid)
        obs = data.get("extra_obstacles") or data.get("obstacles") or []
        doc.obstacles = [(int(c[0]), int(c[1])) for c in obs]
        shape = data.get("map_shape", "")
        intent = data.get("intent", "")
        base = data.get("base_scenario_id", "")
        doc.notes = f"official {sid}" + (f" | {shape}" if shape else "") + (
            f" | {intent}" if intent else ""
        )
        if base:
            doc.notes = f"from {base} @ L{int(data.get('ladder_n_tasks') or 100):03d}" + (
                f" | {shape}" if shape else ""
            )

        pos: Optional[Path] = None
        raw_pos = data.get("position_csv")
        if raw_pos:
            cand = Path(str(raw_pos))
            if cand.exists():
                pos = cand
        if pos is None:
            companion = path.with_name(path.stem + "_position.csv")
            if companion.exists():
                pos = companion
        if pos is not None:
            doc._load_position_csv(pos)

        # Official task list (L100 CSV referenced by scenario JSON)
        task_path: Optional[Path] = None
        raw_task = data.get("task_csv")
        if raw_task:
            cand = Path(str(raw_task))
            if cand.exists():
                task_path = cand
        if task_path is None:
            companion_task = path.with_name(path.stem + "_task.csv")
            if companion_task.exists():
                task_path = companion_task
        if task_path is not None:
            doc.tasks = load_tasks_csv(task_path)
        return doc


def slot_id(n: int) -> str:
    return f"SH_custom_{n:02d}"


def slot_index(sid: str) -> Optional[int]:
    """Parse SH_custom_NN → 1..20, else None."""
    if not sid.startswith("SH_custom_"):
        return None
    try:
        n = int(sid.split("_")[-1])
    except ValueError:
        return None
    return n if 1 <= n <= 20 else None


def list_official_l100(
    scenarios_dir: Optional[Path] = None,
) -> List[Tuple[str, Path]]:
    """Return [(base_scenario_id, L100.json path), ...] ordered SH01…SH20."""
    root = scenarios_dir or default_scenarios_dir()
    if not root.is_dir():
        return []

    ordered: List[Tuple[str, Path]] = []
    man_path = root / "ladder_manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8-sig"))
        bases = [str(b["id"]) for b in man.get("bases", [])]
        for bid in bases:
            p = root / f"{bid}_L100.json"
            if p.exists():
                ordered.append((bid, p))
        if len(ordered) >= 20:
            return ordered[:20]

    # Fallback: glob SH##_…_L100.json sorted by SH number
    found: Dict[int, Tuple[str, Path]] = {}
    for p in root.glob("SH*_L100.json"):
        stem = p.stem  # e.g. SH09_cross_agv15_L100
        if not stem.startswith("SH") or len(stem) < 4:
            continue
        try:
            num = int(stem[2:4])
        except ValueError:
            continue
        base = stem[: -len("_L100")] if stem.endswith("_L100") else stem
        found[num] = (base, p)
    return [found[i] for i in range(1, 21) if i in found]


def official_l100_for_slot(
    n: int, scenarios_dir: Optional[Path] = None
) -> Optional[Tuple[str, Path]]:
    """Map editor slot 1..20 → official L100 (base_id, path)."""
    maps = list_official_l100(scenarios_dir)
    if 1 <= n <= len(maps):
        return maps[n - 1]
    return None


def resolve_slot_source(
    n: int,
    exports_dir: Optional[Path] = None,
    custom_maps_dir: Optional[Path] = None,
    scenarios_dir: Optional[Path] = None,
) -> Tuple[str, Optional[Path], str]:
    """Prefer custom overlay, else official L100.

    Returns (kind, path, label) where kind is 'custom' | 'official' | 'empty'.
    """
    sid = slot_id(n)
    exports = exports_dir or default_exports_dir()
    customs = custom_maps_dir or default_custom_maps_dir()
    for root in (exports, customs):
        p = Path(root) / f"{sid}.json"
        if p.exists():
            return "custom", p, sid

    off = official_l100_for_slot(n, scenarios_dir)
    if off is not None:
        base_id, path = off
        return "official", path, base_id
    return "empty", None, sid


def load_slot_document(
    n: int,
    exports_dir: Optional[Path] = None,
    custom_maps_dir: Optional[Path] = None,
    scenarios_dir: Optional[Path] = None,
) -> Tuple[MapDocument, str, str]:
    """Load slot n: custom if saved, else official L100 scene.

    Returns (doc, kind, label).
    """
    sid = slot_id(n)
    kind, path, label = resolve_slot_source(
        n, exports_dir, custom_maps_dir, scenarios_dir
    )
    if kind == "custom" and path is not None:
        doc = MapDocument.load(path)
        doc.map_id = sid
        return doc, kind, label
    if kind == "official" and path is not None:
        doc = MapDocument.load_from_ladder_scenario(path, map_id=sid)
        return doc, kind, label
    doc = MapDocument(map_id=sid)
    doc.load_baseline_layout()
    return doc, "empty", sid


def list_slots(exports_dir: Optional[Path] = None) -> List[Tuple[str, Optional[Path]]]:
    """Return 20 slots with existing file path if present."""
    root = exports_dir or default_exports_dir()
    root.mkdir(parents=True, exist_ok=True)
    out: List[Tuple[str, Optional[Path]]] = []
    for i in range(1, 21):
        sid = slot_id(i)
        p = root / f"{sid}.json"
        out.append((sid, p if p.exists() else None))
    return out
